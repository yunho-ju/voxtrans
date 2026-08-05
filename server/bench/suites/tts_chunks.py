"""`tts_chunks` suite (contracts/benchmark-cli.md, tasks.md T016).

20 consecutive syntheses of the same fixed phrase: mean per-generated-chunk
time, and `drift_ratio(5, 5)` between the first and last five requests'
mean per-chunk time (SC-003 — the constitution's binding prior 2, the 4.15x
`mx.clear_cache()`-per-chunk penalty).

Runs one untimed warm-up synthesis before the measured loop (Critic finding,
T020 round 1, major 5): without it, request 0 (the very first synthesis
after model load) is measurably slower than steady state and lands inside
the "first 5" drift window, inflating `first_mean` and biasing
`drift_ratio` *down* — exactly backwards from SC-003's purpose, which is to
catch a series that gets *slower*, not to hide it behind an inflated
baseline.

Reports its primary metric as stage `tts_chunk_mean`, deliberately distinct
from `tts_first_chunk` (the `e2e`/budgets.py stage that measures
submission-to-first-chunk latency for one request) — each sample's
`first_chunk_ms` records that same submission-to-first-chunk latency
per-request for reference, but it is not what `tts_chunk_mean` averages
(Critic T020 round-1 minor 4). `tts_chunk_mean` has no constitutional budget
of its own -- SC-003's pass condition is a *relative* improvement over
baseline plus the drift-ratio noise band, not an absolute millisecond
ceiling, so `bench.budgets` intentionally has no entry for it and
`StageTiming.budget_ms` is `None` here.
"""

from __future__ import annotations

import time

from .. import harness

_PHRASE = "The weather today is quite pleasant, don't you think? I might go for a walk later."
_DRIFT_WINDOW = 5


def _synthesize_one(engine, iteration_label: str) -> tuple[list[float], float]:
    """Run one synthesis to completion. Returns (per-chunk durations,
    submission-to-first-chunk latency)."""
    engine.start_synthesis(_PHRASE, voice="Ryan", lang_code="en")
    chunk_durations_ms: list[float] = []
    submit_time = time.perf_counter()
    first_chunk_ms: float | None = None
    start = submit_time
    while True:
        status, _audio, _sr = engine.recv_chunk(timeout=30.0)
        now = time.perf_counter()
        if status == "chunk":
            chunk_durations_ms.append((now - start) * 1000)
            if first_chunk_ms is None:
                first_chunk_ms = (now - submit_time) * 1000
            start = now
        elif status == "done":
            break
        elif status == "error":
            raise RuntimeError(f"tts_chunks: synthesis error on {iteration_label}")
        elif status == "timeout":
            raise RuntimeError(f"tts_chunks: timed out waiting for a chunk on {iteration_label}")
    return chunk_durations_ms, (first_chunk_ms if first_chunk_ms is not None else 0.0)


def run(iterations: int, config: dict) -> tuple[list, list[dict]]:
    from src.tts import TTSEngine

    engine = TTSEngine()
    engine.load()

    per_request_mean_chunk_ms: list[float] = []
    samples: list[dict] = []

    try:
        _synthesize_one(engine, "warmup")  # untimed, discarded (Critic major 5)

        for i in range(iterations):
            chunk_durations_ms, first_chunk_ms = _synthesize_one(engine, f"iteration {i}")

            mean_chunk_ms = (
                sum(chunk_durations_ms) / len(chunk_durations_ms) if chunk_durations_ms else 0.0
            )
            per_request_mean_chunk_ms.append(mean_chunk_ms)
            samples.append(
                {
                    "iteration": i,
                    "chunk_count": len(chunk_durations_ms),
                    "mean_chunk_ms": mean_chunk_ms,
                    "first_chunk_ms": first_chunk_ms,
                }
            )
    finally:
        engine.shutdown()

    overall_mean = sum(per_request_mean_chunk_ms) / len(per_request_mean_chunk_ms)
    window = min(_DRIFT_WINDOW, len(per_request_mean_chunk_ms) // 2) or 1
    drift = harness.drift_ratio(per_request_mean_chunk_ms, first_n=window, last_n=window)
    samples.append({"drift_ratio": drift, "drift_window": window, "warmed_up": True})

    stages = [harness.stage_timing("tts_chunk_mean", overall_mean)]
    return stages, samples
