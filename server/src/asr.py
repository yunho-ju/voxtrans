import logging
import tempfile
import threading

import numpy as np
import soundfile as sf

from .config import ASR_MODEL, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Map lowercase language names to capitalized names expected by the Qwen3-ASR
# model (mlx_audio.stt backend).
_LANGUAGE_MAP = {
    "korean": "Korean",
    "english": "English",
    "japanese": "Japanese",
    "chinese": "Chinese",
    "spanish": "Spanish",
    "french": "French",
    "german": "German",
    "italian": "Italian",
    "portuguese": "Portuguese",
    "russian": "Russian",
}

# Same set of languages, mapped to the ISO 639-1 codes the whisper backend
# (`mlx_whisper`) expects instead (research.md D-ASR design decision 4;
# tasks.md T037c-2).
_WHISPER_LANGUAGE_MAP = {
    "korean": "ko",
    "english": "en",
    "japanese": "ja",
    "chinese": "zh",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
}

# `mlx_audio.stt.load()` can construct a whisper-family model but cannot
# actually transcribe with it: `mlx-community/whisper-large-v3-turbo`'s
# snapshot ships only config.json + weights.safetensors + README.md, and
# mlx_audio's whisper wrapper requires a HuggingFace `WhisperProcessor`
# (preprocessor_config.json + tokenizer files) that repo does not have --
# `generate()` raises `ValueError: Processor not found` (probed, not
# assumed; see evidence/asr-backend-probe.md). The standalone `mlx_whisper`
# package bundles its own tokenizer and does not need repo-provided
# processor files, so whisper-family model ids route there instead.


def _is_whisper_model(model_id: str) -> bool:
    """Pure: does `model_id` belong to the whisper family (routes to the
    `mlx_whisper` backend) rather than the incumbent Qwen3-ASR family
    (routes to `mlx_audio.stt`)? A substring check on the id is sufficient
    for the one candidate this feature compares against; no model is loaded
    to answer this."""
    return "whisper" in model_id.lower()


def _resolve_language_kwarg(model_id: str, language: str) -> str | None:
    """Pure: map a caller-supplied language name (`"auto"`, `"korean"`,
    `"English"`, ...) to the value `model_id`'s backend actually expects --
    a capitalized language name for Qwen3-ASR's `mlx_audio` path
    (`_LANGUAGE_MAP`), an ISO 639-1 code for whisper's `mlx_whisper` path
    (`_WHISPER_LANGUAGE_MAP`). Returns `None` for `"auto"` (or an empty
    value) on either backend, meaning "let the model auto-detect" -- the
    caller omits the `language=` kwarg entirely in that case."""
    if not language or language.lower() == "auto":
        return None
    key = language.lower()
    if _is_whisper_model(model_id):
        return _WHISPER_LANGUAGE_MAP.get(key, key)
    return _LANGUAGE_MAP.get(key, language.capitalize())


class ASREngine:
    def __init__(self) -> None:
        self._model = None
        # The model id `load()` actually resolved and loaded, bound to this
        # instance (Critic round 2 major): `transcribe()`'s backend/language
        # branching reads *this*, never the module-level `ASR_MODEL` global,
        # so a caller that reloads `src.config`/`src.asr` after `load()` has
        # already run (e.g. `bench.eval_asr._load_engine_for_model`, which
        # loads inside a `VOXTRANS_ASR_MODEL` override window and restores
        # the ambient default once the window closes) cannot silently change
        # which backend an already-loaded engine routes to. Production
        # behavior is unchanged: the server calls `load()` exactly once, and
        # `ASR_MODEL` never changes within a running process.
        self._model_id: str | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import time
        logger.info("Loading ASR model: %s", ASR_MODEL)
        t0 = time.time()
        self._model_id = ASR_MODEL
        if _is_whisper_model(self._model_id):
            # Same cache the module-level `mlx_whisper.transcribe()` call in
            # `transcribe()` below consults (`ModelHolder.get_model`,
            # keyed by model path + dtype) -- pre-warming it here means the
            # first real `transcribe()` call does not also pay the load
            # cost. `mx.float16` matches `mlx_whisper.transcribe`'s own
            # default dtype (`fp16=True`), so the cache key matches.
            import mlx.core as mx
            from mlx_whisper.transcribe import ModelHolder
            self._model = ModelHolder.get_model(self._model_id, mx.float16)
        else:
            from mlx_audio.stt import load
            self._model = load(self._model_id)
        logger.info("ASR model loaded in %.1fs", time.time() - t0)

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> str:
        """Transcribe float32 mono 16kHz audio array. Returns text."""
        if self._model is None or self._model_id is None:
            raise RuntimeError("ASR model not loaded")
        if audio.size == 0:
            return ""

        model_id = self._model_id
        duration_s = audio.size / SAMPLE_RATE
        logger.info("ASR: transcribing %.1fs audio, language='%s'", duration_s, language)

        lang_kwarg = _resolve_language_kwarg(model_id, language)
        if lang_kwarg is not None:
            logger.info("ASR: using explicit language='%s'", lang_kwarg)
        else:
            # "auto" mode: model defaults to English output.
            # For reliable non-English transcription, set explicit language.
            logger.info("ASR: auto mode (model defaults to English)")
        kwargs = {"language": lang_kwarg} if lang_kwarg is not None else {}

        with self._lock:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio, SAMPLE_RATE)
                if _is_whisper_model(model_id):
                    import mlx_whisper
                    result = mlx_whisper.transcribe(f.name, path_or_hf_repo=model_id, **kwargs)
                    text = result["text"].strip()
                else:
                    result = self._model.generate(f.name, **kwargs)
                    text = result.text.strip() if hasattr(result, "text") else str(result).strip()

        logger.info("ASR result: language='%s' → text='%s'", language, text[:100])
        return text


# Singleton
asr_engine = ASREngine()
