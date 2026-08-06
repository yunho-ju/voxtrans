"""Tests for `bench.suites.asr` (tasks.md T037c-4, brief C5).

Mirrors `test_translation_suite.py`'s hermetic pattern: `src.asr.ASREngine`
is monkeypatched to a stub before `asr.run()` is called -- `asr.py` does
`from src.asr import ASREngine` *inside* `run()`, so a call made after the
patch sees the patched attribute. No model is ever loaded and no real
fixture audio is transcribed (the stub ignores its input).
"""

from __future__ import annotations

import pytest

import bench.suites.asr as asr
import src.asr


class _StubASREngine:
    def __init__(self, *, transcripts: list[str] | None = None, raises: Exception | None = None) -> None:
        self._transcripts = list(transcripts) if transcripts is not None else ["안녕하세요"]
        self.raises = raises
        self.load_called = False
        self.calls: list[tuple[str, ...]] = []

    def load(self) -> None:
        self.load_called = True

    def transcribe(self, audio, language: str = "auto") -> str:
        self.calls.append((language,))
        if self.raises is not None:
            raise self.raises
        if len(self._transcripts) > 1:
            return self._transcripts.pop(0)
        return self._transcripts[0]


def _install_stub(monkeypatch: pytest.MonkeyPatch, **kwargs) -> _StubASREngine:
    stub = _StubASREngine(**kwargs)
    monkeypatch.setattr(src.asr, "ASREngine", lambda: stub)
    return stub


def test_run_transcribes_a_warmup_pass_before_the_timed_iterations(monkeypatch):
    stub = _install_stub(monkeypatch)

    stages, samples = asr.run(iterations=3, config={"asr_model": "fake-model"})

    # The warm-up call must run before any of the 3 timed iterations, so the
    # stub must have recorded 1 (warm-up) + 3 (timed) = 4 calls total.
    assert len(stub.calls) == 4
    assert stub.load_called is True
    assert len(samples) == 3
    assert all(sample["warmed_up"] is True for sample in samples)
    assert {s.stage for s in stages} == {"asr"}


def test_run_passes_the_warmup_language_to_the_warmup_call_and_the_fixture_language_to_every_timed_call(monkeypatch):
    """The language actually passed to `transcribe()` is the real input to
    `ASREngine`'s backend/language branching (`src/asr.py`'s
    `_resolve_language_kwarg`) -- `asr.py:35-36`'s `_WARMUP_LANGUAGE`
    ("english") and `_FIXTURE_LANGUAGE` ("korean") constants must actually
    reach the engine, not just exist."""
    stub = _install_stub(monkeypatch, transcripts=["warmup", "iter0", "iter1"])

    asr.run(iterations=2, config={"asr_model": "fake-model"})

    assert stub.calls == [("english",), ("korean",), ("korean",)]


def test_run_raises_when_warmup_transcription_is_empty_and_records_no_sample(monkeypatch):
    stub = _install_stub(monkeypatch, transcripts=[""])

    with pytest.raises(RuntimeError, match="warm-up"):
        asr.run(iterations=1, config={"asr_model": "fake-model"})

    assert len(stub.calls) == 1, "the timed loop must never run once warm-up has failed"


def test_run_raises_when_warmup_transcription_raises(monkeypatch):
    _install_stub(monkeypatch, raises=RuntimeError("synthetic warm-up failure"))

    with pytest.raises(RuntimeError, match="synthetic warm-up failure"):
        asr.run(iterations=1, config={"asr_model": "fake-model"})


def test_run_reports_the_mean_duration_across_iterations(monkeypatch):
    _install_stub(monkeypatch, transcripts=["warmup", "iter0", "iter1"])

    stages, samples = asr.run(iterations=2, config={"asr_model": "fake-model"})

    asr_stage = next(s for s in stages if s.stage == "asr")
    assert asr_stage.duration_ms >= 0
    assert [s["iteration"] for s in samples] == [0, 1]
    assert samples[0]["text"] == "iter0"
    assert samples[1]["text"] == "iter1"
