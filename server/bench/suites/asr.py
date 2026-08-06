"""`asr` suite (contracts/benchmark-cli.md, tasks.md T037c-4).

Measures ASR-only latency on the fixed 5s Korean fixture
(`utterance_ko_5s.wav`), using `src.asr.ASREngine` directly -- the same
production path `e2e.py` already exercises (brief
`us1-asr-T037b-T037e.md` design decision 3) -- rather than calling
`mlx_audio.stt`/`mlx_whisper` in this module. Bench numbers this way are
production-path numbers: switching `ASR_MODEL` (via `VOXTRANS_ASR_MODEL` or
`--asr-model`) changes both what this suite measures and what the running
server actually loads, with no separate code path to keep in sync.

Runs one untimed warm-up transcription (`warmup_500ms.wav`) before the timed
loop, mirroring `translation.py`'s `_warm_up` (Critic T020 round-2 major 2):
without it, iteration 0's cold first call would land inside the reported
mean. Raises if the warm-up yields an empty transcript (translation.py's
same "raise, don't silently record warmed_up=True for nothing" pattern), so
`"warmed_up": True` on a sample can only be true because a real warm-up
transcription already succeeded.
"""

from __future__ import annotations

import time
from pathlib import Path

import soundfile as sf

from .. import harness

_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
_FIXTURE = _FIXTURES_DIR / "utterance_ko_5s.wav"
_WARMUP_FIXTURE = _FIXTURES_DIR / "warmup_500ms.wav"
# The fixture is Korean speech; the warm-up clip is English (research.md D3)
# -- same language choices `e2e.py`'s own warm-up/timed calls use.
_FIXTURE_LANGUAGE = "korean"
_WARMUP_LANGUAGE = "english"


def _warm_up(asr_engine, warmup_audio) -> None:
    """One untimed transcription so the timed loop measures steady state,
    not a cold first call. Raises if it produces an empty transcript
    (Critic-pattern mirror of `translation.py`'s zero-response raise): an
    empty result here would otherwise let this function return silently
    having warmed up nothing, while `run()`'s `"warmed_up": True` is still
    written."""
    text = asr_engine.transcribe(warmup_audio, language=_WARMUP_LANGUAGE)
    if not text:
        raise RuntimeError("asr: warm-up transcription produced an empty response")


def run(iterations: int, config: dict) -> tuple[list, list[dict]]:
    from src.asr import ASREngine

    audio, _sr = sf.read(str(_FIXTURE), dtype="float32")
    warmup_audio, _sr2 = sf.read(str(_WARMUP_FIXTURE), dtype="float32")

    asr_engine = ASREngine()
    asr_engine.load()

    _warm_up(asr_engine, warmup_audio)

    samples: list[dict] = []
    durations_ms: list[float] = []

    for i in range(iterations):
        start = time.perf_counter()
        text = asr_engine.transcribe(audio, language=_FIXTURE_LANGUAGE)
        duration_ms = (time.perf_counter() - start) * 1000

        durations_ms.append(duration_ms)
        samples.append(
            {
                "iteration": i,
                "asr_ms": duration_ms,
                "text": text[:200],
                "warmed_up": True,
            }
        )

    mean_duration = sum(durations_ms) / len(durations_ms)
    stages = [harness.stage_timing("asr", mean_duration)]
    return stages, samples
