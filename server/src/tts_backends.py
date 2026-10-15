"""The model-specific half of the TTS worker.

`tts.py` owns the worker process, the request/response queues and the chunk
protocol; a backend owns only what one model family needs to turn
(text, voice, lang_code, optional reference audio) into an iterator of results
carrying `.audio`. That split is what lets a second model be measured on this
pipeline without touching the pipeline: `VOXTRANS_TTS_BACKEND` picks one, and
everything downstream -- suppression deadlines, chunk framing, the bench
suites -- sees the same protocol either way.

Both backends here produce 24 kHz float32 mono, which is what
`TTS_SAMPLE_RATE` and every duration calculation downstream assume; `load()`
rejects a model whose own sample rate disagrees rather than letting the
mismatch surface as audio played at the wrong speed.
"""

from pathlib import Path

import numpy as np

# Backend ids, as accepted by `VOXTRANS_TTS_BACKEND`.
QWEN3 = "qwen3"
KOKORO = "kokoro"

_VOICES_DIR = Path(__file__).parent / "voices"


class Qwen3TTSBackend:
    """Qwen3-TTS: autoregressive, streams sub-sentence chunks, and conditions
    on a reference clip -- which is what both the preset voices (a stored
    reference per speaker) and voice clone (the user's own mic audio) are."""

    name = QWEN3
    default_model = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    supports_voice_clone = True

    # The speakers this backend has a stored reference for. Each is bound to
    # the language its reference clip is spoken in.
    _CATALOG = (
        ("Ryan", "Ryan (EN)"),
        ("Aiden", "Aiden (EN)"),
        ("Vivian", "Vivian (ZH)"),
        ("Serena", "Serena (ZH)"),
    )

    @classmethod
    def catalog(cls) -> list[dict[str, str]]:
        return [{"value": v, "label": label} for v, label in cls._CATALOG]

    def __init__(self) -> None:
        self._model = None
        self._presets: dict[str, tuple[np.ndarray, str]] = {}
        self._sample_rate = 24000

    def load(self, model_name: str, sample_rate: int, log) -> None:
        from mlx_audio.tts.utils import load

        self._sample_rate = sample_rate
        self._model = load(model_name)
        self._presets = _load_preset_voices(log)

    def generate(self, text, voice, lang_code, ref_audio, ref_text, log):
        import mlx.core as mx

        # Cap max_tokens by text length to prevent runaway generation
        # ~12.5 codec tokens/sec, ~3 words/sec → ~4 tokens per character
        max_tok = min(2048, max(75, len(text) * 4))

        gen_kwargs = dict(
            lang_code=lang_code,
            stream=True,
            streaming_interval=0.5,
            max_tokens=max_tok,
            temperature=0.3,
            top_p=0.8,
        )

        if ref_audio is not None:
            # Voice clone mode: use user's mic audio as reference
            ref_resampled = _to_24k(ref_audio, self._sample_rate)
            gen_kwargs["ref_audio"] = mx.array(ref_resampled)
            if ref_text:
                gen_kwargs["ref_text"] = ref_text
            log.info("  voice clone: %d -> %d samples (16kHz -> %dHz)",
                     len(ref_audio), len(ref_resampled), self._sample_rate)
        else:
            # Preset mode: use pre-generated voice reference via ICL
            voice_key = voice.lower()
            if voice_key in self._presets:
                preset_audio, preset_text = self._presets[voice_key]
                gen_kwargs["ref_audio"] = mx.array(_to_24k(preset_audio, self._sample_rate))
                gen_kwargs["ref_text"] = preset_text
                log.info("  preset voice '%s' via ICL ref (%d samples)",
                         voice_key, preset_audio.size)
            else:
                # Fallback: no voice conditioning
                gen_kwargs["voice"] = voice
                log.warning("  preset voice '%s' not found, using unconditioned", voice_key)

        return self._model.generate(text, **gen_kwargs)


class KokoroBackend:
    """Kokoro-82M: non-autoregressive, ~20x smaller, and voice is a stored
    style vector rather than a reference clip -- so it cannot clone, and a
    clone request falls back to the mapped preset rather than failing."""

    name = KOKORO
    default_model = "prince-canuma/Kokoro-82M"
    supports_voice_clone = False

    # The app's voice setting is a speaker identity; the language comes from
    # the translation target, separately. Kokoro's voicepacks are per language,
    # so a name alone cannot select one -- "Ryan speaking Japanese" has to
    # resolve to a Japanese voicepack, not to Ryan's English one read through a
    # Japanese front-end. So each app voice is a slot, and the slot is filled
    # per language. An unmapped name is passed through unchanged, so a raw
    # voicepack name (e.g. "af_heart") also works from the same setting.
    _SLOTS = {"ryan": "m1", "aiden": "m2", "vivian": "f1", "serena": "f2"}

    # Kokoro's own voice inventory, by language. It is lopsided -- English has
    # twenty voices and French has one -- so a language fills the slots it can
    # and `_pick_voice` falls back within the same language for the rest.
    _VOICEPACKS = {
        "a": {"m1": "am_michael", "m2": "am_adam", "f1": "af_heart", "f2": "af_bella"},
        "z": {"m1": "zm_yunxi", "m2": "zm_yunjian", "f1": "zf_xiaoxiao", "f2": "zf_xiaoni"},
        "j": {"m1": "jm_kumo", "f1": "jf_alpha", "f2": "jf_nezumi"},
        "e": {"m1": "em_alex", "m2": "em_santa", "f1": "ef_dora"},
        "f": {"f1": "ff_siwis"},
        "i": {"m1": "im_nicola", "f1": "if_sara"},
        "h": {"m1": "hm_omega", "m2": "hm_psi", "f1": "hf_alpha", "f2": "hf_beta"},
        "p": {"m1": "pm_alex", "m2": "pm_santa", "f1": "pf_dora"},
    }

    # Kokoro's G2P is per language and keyed by a single character
    # (mlx_audio/tts/models/kokoro/pipeline.py: LANG_CODES). The app's language
    # list is wider than that set -- Korean, German and Russian have no Kokoro
    # G2P at all -- so an unsupported target must be a loud error here rather
    # than English phonemes read over foreign text.
    _LANGS = {
        "en": "a", "ja": "j", "zh": "z", "es": "e",
        "fr": "f", "hi": "h", "it": "i", "pt": "p",
    }

    # How a voicepack name reads: first character is the language, second is
    # the gender (`am_michael` = American male).
    _PACK_LANGS = {"a": "EN", "b": "EN-GB", "j": "JA", "z": "ZH",
                   "e": "ES", "f": "FR", "h": "HI", "i": "IT", "p": "PT"}

    def __init__(self) -> None:
        self._model = None

    @classmethod
    def catalog(cls) -> list[dict[str, str]]:
        """The app's slots first, then every voicepack actually on disk.

        The slots follow the target language and are the sane default; the
        raw voicepacks are listed because Kokoro's range lives in its
        inventory, and picking one is how you hear that range. Read from the
        downloaded snapshot rather than hard-coded, so the list cannot offer
        a voice this machine does not have.
        """
        entries = [{"value": name.capitalize(),
                    "label": f"{name.capitalize()} (자동 · 목표 언어)"}
                   for name in cls._SLOTS]
        for pack in cls._installed_voicepacks():
            lang = cls._PACK_LANGS.get(pack[0], pack[0].upper())
            gender = {"f": "여", "m": "남"}.get(pack[1:2], "")
            entries.append({"value": pack, "label": f"{pack} · {lang} {gender}".strip()})
        return entries

    @classmethod
    def _installed_voicepacks(cls) -> list[str]:
        from huggingface_hub import snapshot_download

        try:
            # `voices/*` would also demand the repo's `.pt` copies of every
            # voice, which the pipeline never fetches -- asking for those
            # makes an otherwise complete cache read as incomplete.
            local = Path(snapshot_download(
                repo_id=cls.default_model,
                allow_patterns=["voices/*.safetensors"],
                local_files_only=True,
            ))
        except Exception:
            # Nothing downloaded yet: the slots alone still work, since the
            # pipeline fetches the pack it needs on first use.
            return []
        return sorted(p.stem for p in (local / "voices").glob("*.safetensors"))

    def load(self, model_name: str, sample_rate: int, log) -> None:
        # Kokoro has no text front-end of its own: its pipeline reaches for
        # `misaki.en` and, unconditionally for English, `misaki.espeak` -- so
        # a missing G2P surfaces per request, mid-meeting, rather than here.
        # Failing at load is the difference between "the server did not start"
        # and "your voice silently stopped coming through".
        try:
            import misaki.en  # noqa: F401
            import misaki.espeak  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "the kokoro backend needs a G2P front-end: "
                "`pip install misaki num2words spacy espeakng-loader "
                "phonemizer-fork` (Japanese and Mandarin additionally need "
                "`misaki[ja]` / `misaki[zh]`)"
            ) from e

        from mlx_audio.tts.utils import load

        self._model = load(model_name)
        model_sr = getattr(self._model, "sample_rate", sample_rate)
        if model_sr != sample_rate:
            raise RuntimeError(
                f"kokoro model {model_name} outputs {model_sr} Hz, but the "
                f"pipeline assumes {sample_rate} Hz (config.TTS_SAMPLE_RATE)"
            )

    def generate(self, text, voice, lang_code, ref_audio, ref_text, log):
        code = self._LANGS.get((lang_code or "en").lower())
        if code is None:
            raise ValueError(
                f"kokoro has no voice for language '{lang_code}' "
                f"(supported: {', '.join(sorted(self._LANGS))})"
            )
        kokoro_voice = self._pick_voice(voice, code, log)
        if ref_audio is not None:
            log.warning("  voice clone requested, but kokoro cannot clone; "
                        "using preset voice '%s'", kokoro_voice)
        else:
            log.info("  kokoro voice '%s', lang '%s'", kokoro_voice, code)
        return self._model.generate(text, voice=kokoro_voice, lang_code=code)

    def _pick_voice(self, voice: str, code: str, log) -> str:
        """Resolve an app voice name to a voicepack of the target language."""
        slot = self._SLOTS.get((voice or "").lower())
        if slot is None:
            # Already a Kokoro voicepack name (or something only Kokoro can
            # judge) -- pass it through rather than second-guessing it.
            return voice or "af_heart"

        packs = self._VOICEPACKS[code]
        if slot in packs:
            return packs[slot]
        # This language does not have that exact slot. Keep the language --
        # a voice of the wrong language is the worse mismatch of the two --
        # and stay with the same gender if the language has one at all
        # (French, for instance, ships a single female voice).
        same_gender = sorted(s for s in packs if s[0] == slot[0])
        substitute = packs[same_gender[0] if same_gender else sorted(packs)[0]]
        log.warning("  no '%s' voice for language '%s'; using '%s'",
                    voice, code, substitute)
        return substitute


_BACKENDS = {b.name: b for b in (Qwen3TTSBackend, KokoroBackend)}

# What a client may offer in a backend picker, in the order it should show
# them: the incumbent first, because it is the default and the one with the
# voice clone.
AVAILABLE = [
    {"value": QWEN3, "label": "Qwen3-TTS (기본 · 보이스 클론)"},
    {"value": KOKORO, "label": "Kokoro-82M (빠름 · 클론 없음)"},
]


def default_model(backend_name: str) -> str:
    """The model repo a backend loads when `VOXTRANS_TTS_MODEL` is unset."""
    return _backend_class(backend_name).default_model


def catalog(backend_name: str) -> list[dict[str, str]]:
    """The voices a backend can speak with, as `{value, label}` for the app's
    voice picker. Static: no model is loaded to answer this."""
    return _backend_class(backend_name).catalog()


def make_backend(backend_name: str):
    """Instantiate a backend by id. Raises for an unknown id -- a typo in
    `VOXTRANS_TTS_BACKEND` must not silently fall back to the incumbent and
    then be reported as that backend's measurement."""
    return _backend_class(backend_name)()


def _backend_class(backend_name: str):
    try:
        return _BACKENDS[backend_name]
    except KeyError:
        raise ValueError(
            f"unknown TTS backend '{backend_name}' "
            f"(known: {', '.join(sorted(_BACKENDS))})"
        ) from None


def _load_preset_voices(log) -> dict[str, tuple[np.ndarray, str]]:
    """Load pre-generated voice reference files for preset speakers."""
    voices = {}
    if not _VOICES_DIR.exists():
        log.warning("Voices directory not found: %s", _VOICES_DIR)
        return voices
    for npy_file in _VOICES_DIR.glob("*.npy"):
        name = npy_file.stem.lower()
        txt_file = npy_file.with_suffix(".txt")
        if txt_file.exists():
            audio = np.load(npy_file)
            ref_text = txt_file.read_text().strip()
            voices[name] = (audio, ref_text)
            log.info("Loaded preset voice '%s': %d samples, ref='%s'",
                     name, audio.size, ref_text[:50])
    return voices


def _to_24k(audio_16k: np.ndarray, sample_rate: int) -> np.ndarray:
    """Reference clips are stored and captured at the mic's 16 kHz; the model
    conditions at its own output rate."""
    from scipy.signal import resample as scipy_resample

    num_samples = int(len(audio_16k) * sample_rate / 16000)
    return scipy_resample(audio_16k, num_samples).astype(np.float32)
