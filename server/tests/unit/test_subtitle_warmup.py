"""Tests for `bench.suites.subtitle._run_sessions` (T024 follow-up: a
warm-up session must prime ASR/translation on the same server before the
measured sessions run, and its time must not count toward the reported
latency).

Hermetic: `session_runner` is a stub callable standing in for one full
network session (`_run_one_session`), so this proves the warm-up/measured
separation without a real server, socket, or `_PartialReader` (T020's
existing network-level tests already cover those separately).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import bench.suites.subtitle as subtitle


def test_run_sessions_calls_one_untimed_warmup_before_the_timed_iterations():
    calls: list[str] = []

    def _fake_session_runner(label: str):
        calls.append(label)
        return 100.0 + len(calls), "translation"

    subtitle._run_sessions(_fake_session_runner, iterations=3)

    assert len(calls) == 4, "expected 1 warm-up call + 3 timed calls"


def test_run_sessions_passes_a_distinct_label_to_every_call():
    """Critic round-2 minor 6: `SubtitleServerError` messages lost their
    session context (warm-up vs. which iteration) when the per-iteration
    body was extracted into `_run_one_session` -- restoring it here means
    `_run_sessions` must hand each call a label that identifies it."""
    labels: list[str] = []

    def _fake_session_runner(label: str):
        labels.append(label)
        return 10.0, "translation"

    subtitle._run_sessions(_fake_session_runner, iterations=3)

    assert labels == ["warmup", "iteration 0", "iteration 1", "iteration 2"]


def test_run_sessions_excludes_the_warmup_latency_from_the_measured_samples():
    def _fake_session_runner(label: str, _state=[0]):
        _state[0] += 1
        # The warm-up call (1st) returns an outlier latency that must never
        # reach the measured series; timed calls return small, distinct
        # values instead.
        latency = 9999.0 if _state[0] == 1 else 10.0 * _state[0]
        return latency, "translation"

    latencies_ms, samples = subtitle._run_sessions(_fake_session_runner, iterations=3)

    assert 9999.0 not in latencies_ms
    assert len(latencies_ms) == 3
    assert len(samples) == 3


def test_run_sessions_marks_every_timed_sample_warmed_up_and_numbers_iterations():
    def _fake_session_runner(label: str):
        return 50.0, "translation_partial"

    _, samples = subtitle._run_sessions(_fake_session_runner, iterations=2)

    assert [s["iteration"] for s in samples] == [0, 1]
    assert all(sample["warmed_up"] is True for sample in samples)
    assert all(sample["message_type"] == "translation_partial" for sample in samples)


def test_run_sessions_raises_when_the_warmup_session_fails_before_any_sample_is_recorded():
    def _failing_session_runner(label: str):
        raise RuntimeError("synthetic warm-up failure")

    with pytest.raises(RuntimeError, match="synthetic warm-up failure"):
        subtitle._run_sessions(_failing_session_runner, iterations=3)


def test_run_one_session_error_message_includes_the_given_label(monkeypatch):
    """`_run_one_session` itself must fold `label` into its
    `SubtitleServerError` messages, not just accept and ignore the
    parameter -- proven directly against the no-partial-arrived path, which
    needs no real server: a fake WS whose first `recv()` answers the
    session's initial "connected" status message (so `_run_one_session`
    gets past that unconditional first read) but never delivers a partial
    afterward forces `_run_one_session` down the "no partial arrived"
    raise. The module's real 30s timeout is shortened here so this stays a
    fast unit test."""
    monkeypatch.setattr(subtitle, "_FIRST_PARTIAL_TIMEOUT_S", 0.05)

    class _FakeWS:
        def __init__(self):
            self._answered_initial_recv = False

        def recv(self, timeout: float) -> str:
            if not self._answered_initial_recv:
                self._answered_initial_recv = True
                return '{"type": "status", "status": "connected"}'
            time.sleep(min(timeout, 0.02))
            raise TimeoutError("no message within timeout")

        def send(self, data) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    class _FakeWsClient:
        def connect(self, uri: str, open_timeout: float):
            return _FakeWS()

    with pytest.raises(subtitle.SubtitleServerError, match=r"\(iteration 7\)"):
        subtitle._run_one_session(
            _FakeWsClient(),
            "ws://fake",
            0x01,
            audio=np.zeros(4, dtype="float32"),
            payload=np.zeros(4, dtype="float32"),
            chunk_samples=4,
            chunk_duration_s=0.01,
            payload_duration_s=0.01,
            label="iteration 7",
        )
