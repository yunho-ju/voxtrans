"""Tests for `bench.suites.e2e` (T020 round-2 Critic major 1: a non-"chunk"
`recv_chunk` status was recorded as if it were a real `tts_first_chunk`
measurement).

Hermetic: `src.asr.ASREngine`/`src.translation.TranslationEngine`/
`src.tts.TTSEngine` are monkeypatched to stubs before `e2e.run()` is called
-- `e2e.py` does `from src.X import Y` *inside* `run()`, so patching the
class attribute on the real module is what a call made after the patch
actually sees. No model is loaded and no TTS worker subprocess is spawned;
only the real fixture WAVs are read (cheap, no models involved).
"""

from __future__ import annotations

import numpy as np
import pytest

import bench.suites.e2e as e2e
import src.asr
import src.tts
import src.translation

_BASE_CONFIG = {"vad_silence_ms": "1000"}


class _StubASREngine:
    def load(self) -> None:
        pass

    def transcribe(self, audio, language: str = "auto") -> str:
        return "안녕하세요"


class _StubTranslationEngine:
    def load(self) -> None:
        pass

    def translate(self, text: str, target_language: str) -> str:
        return "hello"


class _ScriptedTTSEngine:
    """Replays a scripted sequence of `recv_chunk()` results, one per call."""

    def __init__(self, script: list[tuple]) -> None:
        self._script = list(script)
        self.shutdown_called = False

    def load(self) -> None:
        pass

    def start_synthesis(self, text: str, voice: str = "Ryan", lang_code: str = "en") -> int:
        return 1

    def recv_chunk(self, timeout: float = 30.0):
        if not self._script:
            raise AssertionError("recv_chunk called more times than this test scripted")
        return self._script.pop(0)

    def shutdown(self) -> None:
        self.shutdown_called = True


_A_CHUNK = ("chunk", np.zeros(10, dtype=np.float32), 24000)
_DONE = ("done", None, 0)


def _install_stubs(monkeypatch: pytest.MonkeyPatch, tts_script: list[tuple]) -> dict:
    monkeypatch.setattr(src.asr, "ASREngine", _StubASREngine)
    monkeypatch.setattr(src.translation, "TranslationEngine", _StubTranslationEngine)
    holder: dict = {}

    def _make_tts_engine():
        engine = _ScriptedTTSEngine(tts_script)
        holder["tts"] = engine
        return engine

    monkeypatch.setattr(src.tts, "TTSEngine", _make_tts_engine)
    return holder


def test_run_succeeds_with_a_clean_warmup_and_one_clean_iteration(monkeypatch):
    _install_stubs(monkeypatch, [_A_CHUNK, _DONE, _A_CHUNK, _DONE])

    stages, samples = e2e.run(iterations=1, config=_BASE_CONFIG)

    assert samples[0]["warmed_up"] is True
    assert {s.stage for s in stages} >= {"vad_hold", "asr", "translation", "tts_first_chunk", "e2e_total"}


def test_run_raises_when_warmup_tts_errors_immediately(monkeypatch):
    holder = _install_stubs(monkeypatch, [("error", None, 0)])

    with pytest.raises(RuntimeError, match="warm-up"):
        e2e.run(iterations=1, config=_BASE_CONFIG)

    assert holder["tts"].shutdown_called is True, "shutdown must still run on a warm-up failure"


def test_run_raises_when_first_iteration_tts_status_is_not_chunk(monkeypatch):
    """The exact bug this round's major 1 named: a non-"chunk" status must
    not be recorded as a `tts_first_chunk` measurement."""
    holder = _install_stubs(monkeypatch, [_A_CHUNK, _DONE, ("error", None, 0)])

    with pytest.raises(RuntimeError, match="tts_first_chunk"):
        e2e.run(iterations=1, config=_BASE_CONFIG)

    assert holder["tts"].shutdown_called is True


def test_run_raises_when_draining_does_not_end_in_done(monkeypatch):
    holder = _install_stubs(monkeypatch, [_A_CHUNK, _DONE, _A_CHUNK, ("timeout", None, 0)])

    with pytest.raises(RuntimeError, match="did not complete cleanly"):
        e2e.run(iterations=1, config=_BASE_CONFIG)

    assert holder["tts"].shutdown_called is True
