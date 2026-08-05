"""Tests for `bench.suites.memory` (T020 round-2 Critic minor 2: a failed
memory-worker subprocess lost its stderr, leaving only a generic "exit code
1" with no diagnostic content).

Hermetic: `subprocess.run` is monkeypatched to fake worker outcomes; no
model is ever loaded and no real subprocess is spawned.
"""

from __future__ import annotations

import subprocess

import pytest

import bench.suites.memory as memory


def _fake_run_success(payload_json: str):
    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=f"loading...\n{payload_json}\n", stderr="")

    return _run


def _fake_run_failure(returncode: int, stderr: str):
    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

    return _run


def test_run_reports_worker_stderr_on_failure(monkeypatch):
    monkeypatch.setattr(memory.harness, "read_page_outs", lambda: 0)
    monkeypatch.setattr(
        memory.subprocess,
        "run",
        _fake_run_failure(1, "Traceback (most recent call last):\nRuntimeError: model snapshot corrupt"),
    )

    with pytest.raises(RuntimeError, match="model snapshot corrupt"):
        memory.run(iterations=1, config={})


def test_run_reports_worker_exit_code_on_failure(monkeypatch):
    monkeypatch.setattr(memory.harness, "read_page_outs", lambda: 0)
    monkeypatch.setattr(memory.subprocess, "run", _fake_run_failure(137, "killed"))

    with pytest.raises(RuntimeError, match="137"):
        memory.run(iterations=1, config={})


def test_run_parses_the_final_stdout_line_on_success(monkeypatch):
    monkeypatch.setattr(memory.harness, "read_page_outs", lambda: 0)
    payload = '{"main_phys_footprint_bytes": 1000, "tts_phys_footprint_bytes": 2000}'
    monkeypatch.setattr(memory.subprocess, "run", _fake_run_success(payload))

    stages, samples = memory.run(iterations=1, config={})

    assert samples[0]["resident_bytes"] == 3000
