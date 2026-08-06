"""Regression test for Critic round 2's major finding on `src.asr.ASREngine`
(tasks.md T037c-2, `us1-asr-T037b-T037e.md`).

Reproduces `bench.eval_asr._load_engine_for_model`'s exact shape without
loading any real model: `ASREngine.load()` runs inside a
`VOXTRANS_ASR_MODEL` override window (here, resolving to the whisper
candidate), then the window closes and `src.config`/`src.asr` are reloaded
back to the ambient default (here, the incumbent) -- exactly what
`bench.eval_asr._restore_asr_modules` does -- *before* `.transcribe()` is
ever called on the already-loaded engine.

Before the fix, `ASREngine.transcribe` re-read the module-level `ASR_MODEL`
global at call time rather than the model id the engine was actually
`load()`ed with. Because `importlib.reload` re-executes into the *same*
module `__dict__` (not a fresh one), even a method captured on an
already-constructed instance shares that dict via its `__globals__` and
therefore sees the *restored* value, not the one active when `load()` ran --
a whisper-loaded engine would silently route back to the incumbent's
`mlx_audio` backend (and crash there, since `self._model` is a
`mlx_whisper.Whisper` object with no `.generate()`).

The fix binds the resolved model id to the instance at `load()` time
(`self._model_id`) and has `transcribe()`'s backend/language branching read
that instance attribute instead of the module global.
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

import src.asr as src_asr
import src.config as src_config

_ENV_VAR = "VOXTRANS_ASR_MODEL"
_WHISPER_MODEL_ID = "mlx-community/whisper-large-v3-turbo"


def _reload_with_env(value: str | None) -> None:
    if value is None:
        os.environ.pop(_ENV_VAR, None)
    else:
        os.environ[_ENV_VAR] = value
    importlib.reload(src_config)
    importlib.reload(src_asr)


@pytest.fixture
def restore_asr_env():
    original = os.environ.get(_ENV_VAR)
    yield
    _reload_with_env(original)


def test_an_engine_loaded_under_a_whisper_override_still_transcribes_via_whisper_after_the_window_closes(
    monkeypatch, restore_asr_env
):
    import mlx_whisper
    from mlx_whisper.transcribe import ModelHolder

    monkeypatch.setattr(ModelHolder, "get_model", classmethod(lambda cls, path, dtype: "FAKE_WHISPER_MODEL"))
    calls: list[dict] = []

    def _fake_transcribe(audio_path, *, path_or_hf_repo, **kwargs):
        calls.append({"path_or_hf_repo": path_or_hf_repo, **kwargs})
        return {"text": "fake whisper transcript"}

    monkeypatch.setattr(mlx_whisper, "transcribe", _fake_transcribe)

    original = os.environ.get(_ENV_VAR)

    # 1. Open the override window (mirrors `_load_engine_for_model`): load
    #    an engine while ASR_MODEL resolves to the whisper candidate.
    _reload_with_env(_WHISPER_MODEL_ID)
    engine = src_asr.ASREngine()
    engine.load()

    # 2. Close the window: restore the ambient default and reload both
    #    modules back, the same way `_restore_asr_modules` does.
    _reload_with_env(original)

    # 3. transcribe() happens after the window closed -- must still use
    #    whisper's backend and language mapping, not the now-restored
    #    incumbent's.
    audio = np.zeros(1600, dtype=np.float32)
    text = engine.transcribe(audio, language="korean")

    assert text == "fake whisper transcript"
    assert len(calls) == 1, "must route to mlx_whisper.transcribe, not the incumbent's mlx_audio path"
    assert calls[0]["path_or_hf_repo"] == _WHISPER_MODEL_ID
    assert calls[0]["language"] == "ko", "must use whisper's ISO-code language mapping, not Qwen3-ASR's"
