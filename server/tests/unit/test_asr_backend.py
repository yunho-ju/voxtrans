"""Tests for `src.asr`'s backend/language selection pure functions (tasks.md
T037c-2, `specs/001-realtime-latency/briefs/us1-asr-T037b-T037e.md` C2).

`_is_whisper_model` and `_resolve_language_kwarg` are pure: no model is ever
loaded here (the probe in `evidence/asr-backend-probe.md` established that
`mlx_audio.stt` cannot load `mlx-community/whisper-large-v3-turbo`'s
processor-less snapshot, so `ASREngine` branches to the standalone
`mlx_whisper` package for whisper-family model ids -- these tests exercise
only that branching decision and the language-argument mapping each backend
expects, never `ASREngine.load()`/`.transcribe()` themselves).
"""

from __future__ import annotations

import pytest

import src.asr as asr

_INCUMBENT_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


# --- _is_whisper_model: backend selection by model id -----------------------


def test_is_whisper_model_false_for_the_incumbent_qwen3_asr_id():
    assert asr._is_whisper_model(_INCUMBENT_MODEL) is False


def test_is_whisper_model_true_for_the_whisper_candidate_id():
    assert asr._is_whisper_model(_WHISPER_MODEL) is True


def test_is_whisper_model_is_case_insensitive():
    assert asr._is_whisper_model("MLX-COMMUNITY/WHISPER-LARGE-V3-TURBO") is True


# --- _resolve_language_kwarg: per-backend language argument mapping --------


@pytest.mark.parametrize(
    "language, expected",
    [
        ("korean", "Korean"),
        ("Korean", "Korean"),
        ("english", "English"),
        ("auto", None),
    ],
)
def test_resolve_language_kwarg_for_the_incumbent_model_uses_capitalized_names(language, expected):
    assert asr._resolve_language_kwarg(_INCUMBENT_MODEL, language) == expected


@pytest.mark.parametrize(
    "language, expected",
    [
        ("korean", "ko"),
        ("Korean", "ko"),
        ("english", "en"),
        ("auto", None),
    ],
)
def test_resolve_language_kwarg_for_a_whisper_model_uses_iso_codes(language, expected):
    assert asr._resolve_language_kwarg(_WHISPER_MODEL, language) == expected


def test_resolve_language_kwarg_falls_back_to_the_language_map_default_for_the_incumbent():
    """A language absent from `_LANGUAGE_MAP` still falls back to a
    capitalized guess, exactly like the pre-branch behavior."""
    assert asr._resolve_language_kwarg(_INCUMBENT_MODEL, "dutch") == "Dutch"


def test_resolve_language_kwarg_falls_back_to_the_raw_code_for_an_unmapped_whisper_language():
    """A language absent from the whisper ISO map falls back to the raw
    lowercased input -- still a plausible ISO-shaped value for whisper's
    `language=` kwarg, rather than a capitalized name whisper wouldn't
    recognize."""
    assert asr._resolve_language_kwarg(_WHISPER_MODEL, "dutch") == "dutch"
