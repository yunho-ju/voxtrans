"""The TTS backend seam: selection, and the two translations a non-incumbent
backend has to perform -- the app's voice names and its ISO language codes
into whatever that model actually accepts.

No model is loaded here. `generate()` is exercised against a stub model, so
what is under test is the mapping, not mlx-audio.
"""

import logging

import numpy as np
import pytest

from src import tts_backends
from src.tts_backends import (
    KokoroBackend,
    Qwen3TTSBackend,
    catalog,
    default_model,
    make_backend,
)

log = logging.getLogger(__name__)


class _StubModel:
    """Records the kwargs a backend hands the model."""

    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return iter(())


def test_make_backend_returns_the_named_backend():
    assert isinstance(make_backend(tts_backends.QWEN3), Qwen3TTSBackend)
    assert isinstance(make_backend(tts_backends.KOKORO), KokoroBackend)


def test_unknown_backend_raises_rather_than_falling_back():
    # A typo in VOXTRANS_TTS_BACKEND must not silently run the incumbent and
    # then have that run reported as the other backend's measurement.
    with pytest.raises(ValueError, match="unknown TTS backend"):
        make_backend("kokoros")


def test_each_backend_declares_its_own_default_model():
    assert default_model(tts_backends.QWEN3) != default_model(tts_backends.KOKORO)
    assert "Qwen3-TTS" in default_model(tts_backends.QWEN3)
    assert "Kokoro" in default_model(tts_backends.KOKORO)


def test_kokoro_maps_app_voice_names_onto_its_own_voicepacks():
    backend = KokoroBackend()
    backend._model = _StubModel()

    backend.generate("hello", "Ryan", "en", None, None, log)

    _, kwargs = backend._model.calls[0]
    assert kwargs["voice"] == "am_michael"
    assert kwargs["lang_code"] == "a"


def test_kokoro_picks_a_voicepack_of_the_target_language():
    # The voice setting is a speaker identity and the language comes from the
    # translation target, so "Ryan speaking Japanese" must not resolve to
    # Ryan's English voicepack read through a Japanese front-end.
    backend = KokoroBackend()
    backend._model = _StubModel()

    backend.generate("こんにちは", "Ryan", "ja", None, None, log)

    _, kwargs = backend._model.calls[0]
    assert kwargs["voice"] == "jm_kumo"
    assert kwargs["lang_code"] == "j"


def test_kokoro_keeps_the_language_when_a_voice_slot_is_missing(caplog):
    # Japanese has one male voice, so Aiden has no slot of its own there.
    backend = KokoroBackend()
    backend._model = _StubModel()

    with caplog.at_level(logging.WARNING):
        backend.generate("こんにちは", "Aiden", "ja", None, None, log)

    assert backend._model.calls[0][1]["voice"] == "jm_kumo"
    assert "no 'Aiden' voice for language 'j'" in caplog.text


def test_kokoro_substitutes_across_gender_only_as_a_last_resort(caplog):
    # French ships exactly one voice, and it is female.
    backend = KokoroBackend()
    backend._model = _StubModel()

    with caplog.at_level(logging.WARNING):
        backend.generate("bonjour", "Ryan", "fr", None, None, log)

    assert backend._model.calls[0][1]["voice"] == "ff_siwis"


def test_every_supported_language_has_at_least_one_voicepack():
    # A language accepted by `generate` but absent from the voicepack table
    # would raise KeyError deep in the request path.
    for iso, code in KokoroBackend._LANGS.items():
        assert KokoroBackend._VOICEPACKS.get(code), iso


def test_kokoro_passes_through_an_unmapped_voicepack_name():
    # So a raw Kokoro voice can be selected from the same setting without a
    # code change while comparing voices.
    backend = KokoroBackend()
    backend._model = _StubModel()

    backend.generate("hello", "af_heart", "en", None, None, log)

    assert backend._model.calls[0][1]["voice"] == "af_heart"


def test_kokoro_rejects_a_language_it_has_no_g2p_for():
    # Korean is in the app's language list and not in Kokoro's. Failing loudly
    # beats reading Korean text through English phonemes.
    backend = KokoroBackend()
    backend._model = _StubModel()

    with pytest.raises(ValueError, match="no voice for language 'ko'"):
        backend.generate("안녕하세요", "Ryan", "ko", None, None, log)


def test_kokoro_falls_back_to_a_preset_when_asked_to_clone(caplog):
    backend = KokoroBackend()
    backend._model = _StubModel()

    with caplog.at_level(logging.WARNING):
        backend.generate("hello", "Ryan", "en", np.zeros(1600, dtype=np.float32), "ref", log)

    assert "cannot clone" in caplog.text
    assert backend._model.calls[0][1]["voice"] == "am_michael"


def test_qwen3_conditions_on_the_reference_clip_when_cloning():
    backend = Qwen3TTSBackend()
    backend._model = _StubModel()
    backend._sample_rate = 24000

    backend.generate("hello", "Ryan", "en", np.zeros(1600, dtype=np.float32), "ref text", log)

    _, kwargs = backend._model.calls[0]
    assert kwargs["ref_text"] == "ref text"
    assert kwargs["stream"] is True
    # 16 kHz mic audio resampled to the model's output rate.
    assert kwargs["ref_audio"].shape[0] == 2400


# The voice picker's list comes from the server because the backend does --
# a client-side list would be right for one backend and wrong for the other.


def test_qwen3_catalog_is_the_speakers_it_has_references_for():
    values = [v["value"] for v in catalog("qwen3")]
    assert values == ["Ryan", "Aiden", "Vivian", "Serena"]


def test_kokoro_catalog_offers_the_slots_first():
    entries = catalog("kokoro")
    assert [e["value"] for e in entries[:4]] == ["Ryan", "Aiden", "Vivian", "Serena"]
    # The slots are not fixed voices under this backend, and the label has to
    # say so or "Ryan (EN)" would be a lie in a Japanese session.
    assert "EN" not in entries[0]["label"]


def test_kokoro_catalog_lists_voicepacks_that_are_actually_on_disk():
    entries = catalog("kokoro")
    packs = [e for e in entries[4:]]
    if not packs:
        pytest.skip("Kokoro snapshot not downloaded on this machine")
    assert all("_" in e["value"] for e in packs)
    assert any(e["value"].startswith("am_") for e in packs)


def test_catalog_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown TTS backend"):
        catalog("kokoros")
