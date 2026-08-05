"""Tests for bench/run.py (T020 round-1 Critic major 2, major 3, major 7).

Hermetic, in-process probes: every check function and the suite loader are
replaced with sentinels (never real model loads, never the real evidence
directory), the same style the Critic's own reproduction used to expose
major 2/3 in seconds. `monkeypatch.setattr(run, "_check_x", ...)` -- not
patching `builtins.__import__` or actually uninstalling `mlx` -- because the
bug under test is *ordering* and *isolation* in `run.py`'s own dispatch
logic, not any one check's internals (those are exercised live by T019's
`--suite e2e --label smoke` run instead).
"""

from __future__ import annotations

import argparse

import pytest

import bench.harness as harness
import bench.run as run


def _args(suite: str = "e2e", label: str = "test", iterations: int = 1, force: bool = False) -> argparse.Namespace:
    return run.build_parser().parse_args(
        ["--suite", suite, "--label", label, "--iterations", str(iterations)]
        + (["--force"] if force else [])
    )


def _hardware(cpu: str = harness.REFERENCE_CPU) -> harness.Hardware:
    return harness.Hardware(
        cpu=cpu, memory_bytes=1, os_version="26.5", python_version="3.13.0",
        mlx_version="0.32.0", page_outs=0,
    )


class _StubSuite:
    """A `bench.suites.*`-shaped stub: `run()` either returns canned
    stages/samples or raises, and records how many times it was called."""

    def __init__(self, *, raises: Exception | None = None):
        self.raises = raises
        self.call_count = 0

    def run(self, iterations: int, config: dict):
        self.call_count += 1
        if self.raises is not None:
            raise self.raises
        return [harness.stage_timing("asr", 100.0)], [{"iteration": 0}]


def _pass_all_precondition_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "_check_imports", lambda: None)
    monkeypatch.setattr(run, "_check_models", lambda model_override: None)
    monkeypatch.setattr(run, "_check_fixture", lambda suite: None)
    monkeypatch.setattr(run, "_check_subtitle_server_port_free", lambda suite: None)


# --- major 2: capture_hardware() must never run before an earlier check ---
# --- that would have failed anyway -----------------------------------------


def test_run_preconditions_does_not_capture_hardware_when_an_earlier_check_fails(monkeypatch):
    """If `_check_imports` (or any pre-hardware check) fails, `capture_hardware`
    must never even be called -- a venv that cannot import mlx must fail with
    the import check's exit code, not by raising inside `capture_hardware`'s
    own package-version lookup and crashing with Python's generic exit 1
    (Critic major 2)."""
    monkeypatch.setattr(run, "_check_imports", lambda: run.PreconditionFailure("venv_imports", "boom", 2))

    def _must_not_be_called():
        raise AssertionError("capture_hardware() must not run after an earlier check already failed")

    monkeypatch.setattr(run.harness, "capture_hardware", _must_not_be_called)

    failure, hw = run.run_preconditions("e2e", None)
    assert failure is not None
    assert failure.check == "venv_imports"
    assert failure.exit_code == 2
    assert hw is None


def test_run_preconditions_captures_hardware_only_after_earlier_checks_pass(monkeypatch):
    _pass_all_precondition_checks(monkeypatch)
    monkeypatch.setattr(run.harness, "capture_hardware", lambda: _hardware("Apple M2 Max"))

    failure, hw = run.run_preconditions("e2e", None)
    assert failure is not None
    assert failure.check == "reference_cpu"
    assert failure.exit_code == 3
    assert hw is not None
    assert hw.cpu == "Apple M2 Max"


def test_run_preconditions_returns_no_failure_and_the_captured_hardware_when_everything_passes(monkeypatch):
    _pass_all_precondition_checks(monkeypatch)
    monkeypatch.setattr(run.harness, "capture_hardware", lambda: _hardware())

    failure, hw = run.run_preconditions("e2e", None)
    assert failure is None
    assert hw is not None
    assert hw.is_reference is True


def test_run_one_never_loads_the_suite_when_a_precondition_fails(monkeypatch, tmp_path):
    """Critic T020 round-2 minor 3: a precondition failure must stop before
    the suite is ever loaded/run, not merely report the right exit code
    while still having run it."""
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    monkeypatch.setattr(run, "_check_imports", lambda: run.PreconditionFailure("venv_imports", "boom", 2))
    stub = _StubSuite()
    monkeypatch.setattr(run, "_load_suite", lambda suite: stub)

    exit_code = run._run_one("e2e", _args())

    assert exit_code == 2
    assert stub.call_count == 0, "the suite must never run once a precondition has already failed"


# --- major 3: a suite crash must not propagate, and must be exit 2, not 1 --


def test_suite_crash_returns_exit_2_not_1(monkeypatch, tmp_path):
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    _pass_all_precondition_checks(monkeypatch)
    monkeypatch.setattr(run.harness, "capture_hardware", lambda: _hardware())
    crashing = _StubSuite(raises=RuntimeError("synthetic suite crash"))
    monkeypatch.setattr(run, "_load_suite", lambda suite: crashing)

    exit_code = run._run_one("e2e", _args())

    assert exit_code == 2, "a suite crash must be reported as a harness failure (2), never budget-Red (1)"
    assert crashing.call_count == 1
    assert not (tmp_path / "e2e-test.json").exists(), "a crashed suite has no stages to write a record from"


def test_all_suites_still_run_after_one_crashes(monkeypatch, tmp_path):
    """`--suite all` must not abort the remaining suites just because one of
    them crashed (Critic major 3)."""
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    _pass_all_precondition_checks(monkeypatch)
    monkeypatch.setattr(run.harness, "capture_hardware", lambda: _hardware())

    stubs = {suite: _StubSuite(raises=RuntimeError("boom") if suite == "translation" else None) for suite in run.SUITES}
    monkeypatch.setattr(run, "_load_suite", lambda suite: stubs[suite])

    exit_code = run.main(["--suite", "all", "--label", "test"])

    for suite, stub in stubs.items():
        assert stub.call_count == 1, f"{suite} should have run exactly once regardless of another suite crashing"
    assert exit_code == 2, "the crashed suite's exit code (2) must win over the healthy suites' exit 0"


# --- overwrite guard: exit 2 before any precondition runs -------------------


def test_overwrite_guard_returns_exit_2_without_running_preconditions(monkeypatch, tmp_path):
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    (tmp_path / "e2e-test.json").write_text("{}")

    def _must_not_be_called(*a, **k):
        raise AssertionError("preconditions must not run once the overwrite guard has already failed")

    monkeypatch.setattr(run, "run_preconditions", _must_not_be_called)

    exit_code = run._run_one("e2e", _args(force=False))
    assert exit_code == 2


def test_force_flag_bypasses_the_overwrite_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    (tmp_path / "e2e-test.json").write_text("{}")
    _pass_all_precondition_checks(monkeypatch)
    monkeypatch.setattr(run.harness, "capture_hardware", lambda: _hardware())
    stub = _StubSuite()
    monkeypatch.setattr(run, "_load_suite", lambda suite: stub)

    exit_code = run._run_one("e2e", _args(force=True))

    assert exit_code == 0
    assert stub.call_count == 1


# --- exit code priority: 2 > 3 > 1 > 0 --------------------------------------


@pytest.mark.parametrize(
    "codes, expected",
    [
        ([0, 0, 0, 0, 0], 0),
        ([0, 1, 0, 0, 0], 1),
        ([0, 1, 3, 0, 0], 3),
        ([0, 1, 3, 2, 0], 2),
        ([3, 2, 1, 0, 0], 2),
    ],
)
def test_main_exit_code_priority_is_2_then_3_then_1(monkeypatch, codes, expected):
    codes_iter = iter(codes)
    monkeypatch.setattr(run, "_run_one", lambda suite, args: next(codes_iter))

    exit_code = run.main(["--suite", "all", "--label", "test"])

    assert exit_code == expected
