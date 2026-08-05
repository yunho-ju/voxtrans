"""`e2e` suite (contracts/benchmark-cli.md, tasks.md T017).

Composes `vad_hold` + `asr` + `translation` + `tts_first_chunk` for the fixed
5s fixture (`utterance_ko_5s.wav`). `vad_hold` is *derived* from the
effective VAD silence-hold configuration, not timed — data-model.md §1: it
is a fixed window count times 32ms, so timing it against a fixture would
measure the fixture's trailing silence rather than the system. `e2e_total`
is the documented **sum** of the four stages
(contracts/benchmark-cli.md "Composition rule for e2e") — not an independent
wall-clock stopwatch over a real WebSocket path (that is what the `subtitle`
suite is for).

Runs one untimed warm-up pass (`warmup_500ms.wav`, T005) through all three
engines before the timed loop starts (Critic finding, T020 round 1, major
4): `T058` (readiness gating on warm-up) exists specifically because first
inference is measurably slower than steady state, so a suite that measures
only a cold first call would overstate the improvement any later
optimization appears to buy. Each sample records `"warmed_up": True` so a
reader of the evidence JSON does not have to take that on faith.
"""

from __future__ import annotations

import time
from pathlib import Path

import soundfile as sf

from .. import harness

_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
_FIXTURE = _FIXTURES_DIR / "utterance_ko_5s.wav"
_WARMUP_FIXTURE = _FIXTURES_DIR / "warmup_500ms.wav"
# The fixture is Korean speech; translate to English so the suite exercises
# a real cross-language pass rather than round-tripping the same language
# (which the model would treat as a same-language paraphrase, not a
# translation, and which is not the direction any real listen-mode session
# uses for this fixture).
_TARGET_LANGUAGE = "english"


def _warm_up(asr_engine, translation_engine, tts_engine) -> None:
    """One untimed inference per engine so the timed loop measures steady
    state, not a cold first call (Critic T020 round-1 major 4).

    Raises if the TTS leg does not actually complete (Critic T020 round-2
    major 1: `"warmed_up": True` must be *derived* from warm-up actually
    succeeding, not hardcoded regardless of outcome -- raising here, before
    `run()`'s loop or its `samples.append` is ever reached, is what makes
    that true: `"warmed_up": True` can only be written once this function
    has already returned without raising).
    """
    warmup_audio, _sr = sf.read(str(_WARMUP_FIXTURE), dtype="float32")
    asr_engine.transcribe(warmup_audio, language="english")
    translation_engine.translate("Hello there", _TARGET_LANGUAGE)
    tts_engine.start_synthesis("Hello there", voice="Ryan", lang_code="en")
    status, _audio, _sr2 = tts_engine.recv_chunk(timeout=30.0)
    if status != "chunk":
        raise RuntimeError(f"e2e: warm-up TTS synthesis did not produce a chunk (status={status!r})")
    while status == "chunk":
        status, _audio, _sr2 = tts_engine.recv_chunk(timeout=30.0)
    if status != "done":
        raise RuntimeError(f"e2e: warm-up TTS synthesis did not complete cleanly (status={status!r})")


def run(iterations: int, config: dict) -> tuple[list, list[dict]]:
    from src.asr import ASREngine
    from src.translation import TranslationEngine
    from src.tts import TTSEngine

    audio, _sr = sf.read(str(_FIXTURE), dtype="float32")

    asr_engine = ASREngine()
    asr_engine.load()
    translation_engine = TranslationEngine()
    translation_engine.load()
    tts_engine = TTSEngine()
    tts_engine.load()

    vad_hold_ms = float(config["vad_silence_ms"])

    samples: list[dict] = []
    asr_durations_ms: list[float] = []
    translation_durations_ms: list[float] = []
    tts_first_chunk_durations_ms: list[float] = []

    try:
        _warm_up(asr_engine, translation_engine, tts_engine)

        for i in range(iterations):
            t0 = time.perf_counter()
            text = asr_engine.transcribe(audio, language="korean")
            asr_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            translated = translation_engine.translate(text, _TARGET_LANGUAGE)
            translation_ms = (time.perf_counter() - t0) * 1000

            speak_text = translated or "hello"
            t0 = time.perf_counter()
            tts_engine.start_synthesis(speak_text, voice="Ryan", lang_code="en")
            status, _audio, _sr2 = tts_engine.recv_chunk(timeout=30.0)
            tts_first_chunk_ms = (time.perf_counter() - t0) * 1000
            # A non-"chunk" status here (Critic T020 round-2 major 1) means
            # synthesis errored or timed out immediately -- the elapsed time
            # above is a failure latency, not a `tts_first_chunk` measurement,
            # and recording it as one would let e.g. a 40ms worker crash read
            # as a budget PASS. Treated as a suite crash (run.py's `_run_one`
            # catches it and reports exit 2), the same as a genuine exception.
            if status != "chunk":
                raise RuntimeError(
                    f"e2e: tts_first_chunk did not receive a chunk on iteration {i} (status={status!r})"
                )
            # Drain remaining chunks so the next iteration starts on a clean
            # queue; anything other than a clean "done" at the end means
            # synthesis did not actually complete successfully.
            while status == "chunk":
                status, _audio, _sr2 = tts_engine.recv_chunk(timeout=30.0)
            if status != "done":
                raise RuntimeError(
                    f"e2e: tts synthesis did not complete cleanly on iteration {i} (status={status!r})"
                )

            asr_durations_ms.append(asr_ms)
            translation_durations_ms.append(translation_ms)
            tts_first_chunk_durations_ms.append(tts_first_chunk_ms)
            samples.append(
                {
                    "iteration": i,
                    "asr_ms": asr_ms,
                    "translation_ms": translation_ms,
                    "tts_first_chunk_ms": tts_first_chunk_ms,
                    "asr_text": text[:200],
                    # `(translated or "")` mirrors the same defensive
                    # assumption `speak_text` already makes above: `translate()`
                    # never actually returns `None` today, but this keeps the
                    # two lines from silently disagreeing about that (Critic
                    # T020 round-1 minor 3).
                    "translated_text": (translated or "")[:200],
                    "warmed_up": True,
                }
            )
    finally:
        tts_engine.shutdown()

    mean_asr = sum(asr_durations_ms) / len(asr_durations_ms)
    mean_translation = sum(translation_durations_ms) / len(translation_durations_ms)
    mean_tts_first_chunk = sum(tts_first_chunk_durations_ms) / len(tts_first_chunk_durations_ms)
    e2e_total = vad_hold_ms + mean_asr + mean_translation + mean_tts_first_chunk

    stages = [
        harness.stage_timing("vad_hold", vad_hold_ms),
        harness.stage_timing("asr", mean_asr),
        harness.stage_timing("translation", mean_translation),
        harness.stage_timing("tts_first_chunk", mean_tts_first_chunk),
        harness.stage_timing("e2e_total", e2e_total),
    ]
    return stages, samples
