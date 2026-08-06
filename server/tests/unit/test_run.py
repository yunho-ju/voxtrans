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
import os

import pytest

import bench.fetch_models as fetch_models
import bench.harness as harness
import bench.run as run
import src.config as src_config

_ENV_VAR = "VOXTRANS_TRANSLATION_MODEL"
_ASR_ENV_VAR = "VOXTRANS_ASR_MODEL"


def _args(
    suite: str = "e2e",
    label: str = "test",
    iterations: int = 1,
    force: bool = False,
    model: str | None = None,
    asr_model: str | None = None,
) -> argparse.Namespace:
    return run.build_parser().parse_args(
        ["--suite", suite, "--label", label, "--iterations", str(iterations)]
        + (["--force"] if force else [])
        + (["--model", model] if model else [])
        + (["--asr-model", asr_model] if asr_model else [])
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


class _EnvCapturingStubSuite:
    """Records the *live* `VOXTRANS_TRANSLATION_MODEL` env value at the
    moment `run()` is invoked -- standing in for `memory`'s worker
    subprocess and `subtitle`'s own `src.main` subprocess, both of which
    inherit whatever this process's environment says at spawn time
    (Critic round-2 major 3)."""

    def __init__(self, *, raises: Exception | None = None):
        self.raises = raises
        self.captured_env: str | None = "not-yet-called"

    def run(self, iterations: int, config: dict):
        self.captured_env = os.environ.get(_ENV_VAR)
        if self.raises is not None:
            raise self.raises
        return [harness.stage_timing("asr", 100.0)], [{"iteration": 0}]


class _AsrEnvCapturingStubSuite:
    """Same shape as `_EnvCapturingStubSuite`, capturing `VOXTRANS_ASR_MODEL`
    instead -- T037c-4/C4's mirror of the translation env-propagation
    mechanism (`_asr_model_env_override`)."""

    def __init__(self, *, raises: Exception | None = None):
        self.raises = raises
        self.captured_env: str | None = "not-yet-called"

    def run(self, iterations: int, config: dict):
        self.captured_env = os.environ.get(_ASR_ENV_VAR)
        if self.raises is not None:
            raise self.raises
        return [harness.stage_timing("asr", 100.0)], [{"iteration": 0}]


def _pass_all_precondition_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "_check_imports", lambda: None)
    monkeypatch.setattr(run, "_check_models", lambda model_override, asr_model_override=None: None)
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
        ([0, 0, 0, 0, 0, 0], 0),
        ([0, 1, 0, 0, 0, 0], 1),
        ([0, 1, 3, 0, 0, 0], 3),
        ([0, 1, 3, 2, 0, 0], 2),
        ([3, 2, 1, 0, 0, 0], 2),
    ],
)
def test_main_exit_code_priority_is_2_then_3_then_1(monkeypatch, codes, expected):
    assert len(codes) == len(run.SUITES), "one exit code per suite `--suite all` runs"
    codes_iter = iter(codes)
    monkeypatch.setattr(run, "_run_one", lambda suite, args: next(codes_iter))

    exit_code = run.main(["--suite", "all", "--label", "test"])

    assert exit_code == expected


# --- major 3: `--model` overrides VOXTRANS_TRANSLATION_MODEL for every -----
# --- model-loading path this run touches, not just the resolved config ----
# --- dict (contracts/benchmark-cli.md:21) ----------------------------------


def _clear_env_var():
    os.environ.pop(_ENV_VAR, None)


def test_run_one_sets_the_env_var_for_the_duration_of_the_suite_run(monkeypatch, tmp_path):
    """`memory`'s worker and `subtitle`'s own `src.main` subprocess each
    inherit this process's environment at spawn time -- setting the
    override here, before the suite even loads, is what a subprocess
    spawned *inside* `suite_mod.run()` will see."""
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    _pass_all_precondition_checks(monkeypatch)
    monkeypatch.setattr(run.harness, "capture_hardware", lambda: _hardware())
    _clear_env_var()
    stub = _EnvCapturingStubSuite()
    monkeypatch.setattr(run, "_load_suite", lambda suite: stub)

    exit_code = run._run_one("memory", _args(suite="memory", model="acme/candidate-model"))

    assert exit_code == 0
    assert stub.captured_env == "acme/candidate-model", (
        "the suite's run() must see the override in the live environment, "
        "not merely in the resolved config dict"
    )
    assert os.environ.get(_ENV_VAR) is None, "the env var must be restored once the run finishes"


def test_run_one_restores_a_previously_set_env_var_after_finishing(monkeypatch, tmp_path):
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    _pass_all_precondition_checks(monkeypatch)
    monkeypatch.setattr(run.harness, "capture_hardware", lambda: _hardware())
    os.environ[_ENV_VAR] = "prior-ambient-value"
    stub = _EnvCapturingStubSuite()
    monkeypatch.setattr(run, "_load_suite", lambda suite: stub)

    try:
        exit_code = run._run_one("memory", _args(suite="memory", model="acme/candidate-model"))
        assert exit_code == 0
        assert stub.captured_env == "acme/candidate-model"
        assert os.environ.get(_ENV_VAR) == "prior-ambient-value"
    finally:
        _clear_env_var()


def test_run_one_does_not_touch_the_env_var_when_no_model_flag_is_given(monkeypatch, tmp_path):
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    _pass_all_precondition_checks(monkeypatch)
    monkeypatch.setattr(run.harness, "capture_hardware", lambda: _hardware())
    _clear_env_var()
    stub = _EnvCapturingStubSuite()
    monkeypatch.setattr(run, "_load_suite", lambda suite: stub)

    exit_code = run._run_one("e2e", _args(suite="e2e", model=None))

    assert exit_code == 0
    assert stub.captured_env is None, "no --model given must mean no env mutation at all"


def test_run_one_restores_the_env_var_even_when_the_suite_crashes(monkeypatch, tmp_path):
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    _pass_all_precondition_checks(monkeypatch)
    monkeypatch.setattr(run.harness, "capture_hardware", lambda: _hardware())
    _clear_env_var()
    crashing = _EnvCapturingStubSuite(raises=RuntimeError("synthetic crash"))
    monkeypatch.setattr(run, "_load_suite", lambda suite: crashing)

    exit_code = run._run_one("memory", _args(suite="memory", model="acme/candidate-model"))

    assert exit_code == 2
    assert crashing.captured_env == "acme/candidate-model"
    assert os.environ.get(_ENV_VAR) is None, "a crash must not leak the override into later runs"


# --- minor 1: `_check_models` must validate the *resolved* translation ------
# --- model (ambient env included), not only an explicit `--model` ----------


def test_check_models_validates_the_ambient_resolved_translation_model_without_a_model_flag(monkeypatch):
    """A `VOXTRANS_TRANSLATION_MODEL` set in the ambient environment (not
    via `--model`) must be precondition-checked too -- otherwise an invalid
    ambient override is discovered only mid-run, as a suite crash or worse,
    a silent download inside a timed region."""
    monkeypatch.setattr(src_config, "TRANSLATION_MODEL", "ambiently-overridden/fake-model")
    checked_repo_ids: list[str] = []

    def _fake_verify_model(repo_id):
        checked_repo_ids.append(repo_id)
        return fetch_models.VerifyResult(repo_id, repo_id != "ambiently-overridden/fake-model")

    monkeypatch.setattr(run.fetch_models, "verify_model", _fake_verify_model)
    monkeypatch.setattr(run.fetch_models, "verify_silero_vad", lambda: fetch_models.VerifyResult("silero-vad", True))

    failure = run._check_models(None)

    assert "ambiently-overridden/fake-model" in checked_repo_ids
    assert failure is not None
    assert failure.check == "model_snapshot"
    assert failure.exit_code == 2


def test_check_models_prefers_the_explicit_model_flag_over_the_ambient_value(monkeypatch):
    monkeypatch.setattr(src_config, "TRANSLATION_MODEL", "ambient/should-not-be-used-when-flag-given")
    checked_repo_ids: list[str] = []

    def _fake_verify_model(repo_id):
        checked_repo_ids.append(repo_id)
        return fetch_models.VerifyResult(repo_id, True)

    monkeypatch.setattr(run.fetch_models, "verify_model", _fake_verify_model)
    monkeypatch.setattr(run.fetch_models, "verify_silero_vad", lambda: fetch_models.VerifyResult("silero-vad", True))

    failure = run._check_models("acme/explicit-flag-model")

    assert failure is None
    assert "acme/explicit-flag-model" in checked_repo_ids


# --- T037c-3/C3: `_check_models` validates the *resolved* ASR model too ----
# --- (ambient VOXTRANS_ASR_MODEL or `--asr-model`), mirroring the ----------
# --- translation-model checks above -----------------------------------------


def test_check_models_validates_the_ambient_resolved_asr_model_without_an_asr_model_flag(monkeypatch):
    monkeypatch.setattr(src_config, "ASR_MODEL", "ambiently-overridden/fake-asr-model")
    checked_repo_ids: list[str] = []

    def _fake_verify_model(repo_id):
        checked_repo_ids.append(repo_id)
        return fetch_models.VerifyResult(repo_id, repo_id != "ambiently-overridden/fake-asr-model")

    monkeypatch.setattr(run.fetch_models, "verify_model", _fake_verify_model)
    monkeypatch.setattr(run.fetch_models, "verify_silero_vad", lambda: fetch_models.VerifyResult("silero-vad", True))

    failure = run._check_models(None)

    assert "ambiently-overridden/fake-asr-model" in checked_repo_ids
    assert failure is not None
    assert failure.check == "model_snapshot"
    assert failure.exit_code == 2
    assert "ambiently-overridden/fake-asr-model" in failure.reason


def test_check_models_prefers_the_explicit_asr_model_flag_over_the_ambient_value(monkeypatch):
    monkeypatch.setattr(src_config, "ASR_MODEL", "ambient/should-not-be-used-when-flag-given")
    checked_repo_ids: list[str] = []

    def _fake_verify_model(repo_id):
        checked_repo_ids.append(repo_id)
        return fetch_models.VerifyResult(repo_id, repo_id != "acme/explicit-asr-flag-model")

    monkeypatch.setattr(run.fetch_models, "verify_model", _fake_verify_model)
    monkeypatch.setattr(run.fetch_models, "verify_silero_vad", lambda: fetch_models.VerifyResult("silero-vad", True))

    failure = run._check_models(None, "acme/explicit-asr-flag-model")

    assert "acme/explicit-asr-flag-model" in checked_repo_ids
    assert failure is not None
    assert "acme/explicit-asr-flag-model" in failure.reason


def test_check_models_reports_exit_2_and_names_a_missing_asr_model_in_stderr(monkeypatch, tmp_path, capsys):
    """Brief acceptance condition 7 / C3: a nonexistent `--asr-model` id
    exits 2 with that id named in stderr."""
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    monkeypatch.setattr(run, "_check_imports", lambda: None)
    monkeypatch.setattr(run, "_check_fixture", lambda suite: None)
    monkeypatch.setattr(run, "_check_subtitle_server_port_free", lambda suite: None)

    def _fake_verify_model(repo_id):
        return fetch_models.VerifyResult(repo_id, repo_id != "acme/does-not-exist")

    monkeypatch.setattr(run.fetch_models, "verify_model", _fake_verify_model)
    monkeypatch.setattr(run.fetch_models, "verify_silero_vad", lambda: fetch_models.VerifyResult("silero-vad", True))

    exit_code = run._run_one("asr", _args(suite="asr", asr_model="acme/does-not-exist"))

    assert exit_code == 2
    assert "acme/does-not-exist" in capsys.readouterr().err


# --- T037c-4/C4: `--asr-model` propagates through the env, mirroring -------
# --- `--model`'s VOXTRANS_TRANSLATION_MODEL propagation exactly ------------


def _clear_asr_env_var():
    os.environ.pop(_ASR_ENV_VAR, None)


def test_run_one_sets_the_asr_model_env_var_for_the_duration_of_the_suite_run(monkeypatch, tmp_path):
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    _pass_all_precondition_checks(monkeypatch)
    monkeypatch.setattr(run.harness, "capture_hardware", lambda: _hardware())
    _clear_asr_env_var()
    stub = _AsrEnvCapturingStubSuite()
    monkeypatch.setattr(run, "_load_suite", lambda suite: stub)

    exit_code = run._run_one("asr", _args(suite="asr", asr_model="acme/candidate-asr-model"))

    assert exit_code == 0
    assert stub.captured_env == "acme/candidate-asr-model", (
        "the suite's run() must see the override in the live environment"
    )
    assert os.environ.get(_ASR_ENV_VAR) is None, "the env var must be restored once the run finishes"


def test_run_one_restores_a_previously_set_asr_model_env_var_after_finishing(monkeypatch, tmp_path):
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    _pass_all_precondition_checks(monkeypatch)
    monkeypatch.setattr(run.harness, "capture_hardware", lambda: _hardware())
    os.environ[_ASR_ENV_VAR] = "prior-ambient-asr-value"
    stub = _AsrEnvCapturingStubSuite()
    monkeypatch.setattr(run, "_load_suite", lambda suite: stub)

    try:
        exit_code = run._run_one("asr", _args(suite="asr", asr_model="acme/candidate-asr-model"))
        assert exit_code == 0
        assert stub.captured_env == "acme/candidate-asr-model"
        assert os.environ.get(_ASR_ENV_VAR) == "prior-ambient-asr-value"
    finally:
        _clear_asr_env_var()


def test_run_one_does_not_touch_the_asr_model_env_var_when_no_asr_model_flag_is_given(monkeypatch, tmp_path):
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    _pass_all_precondition_checks(monkeypatch)
    monkeypatch.setattr(run.harness, "capture_hardware", lambda: _hardware())
    _clear_asr_env_var()
    stub = _AsrEnvCapturingStubSuite()
    monkeypatch.setattr(run, "_load_suite", lambda suite: stub)

    exit_code = run._run_one("asr", _args(suite="asr", asr_model=None))

    assert exit_code == 0
    assert stub.captured_env is None, "no --asr-model given must mean no env mutation at all"


def test_run_one_restores_the_asr_model_env_var_even_when_the_suite_crashes(monkeypatch, tmp_path):
    monkeypatch.setattr(run.harness, "EVIDENCE_DIR", tmp_path)
    _pass_all_precondition_checks(monkeypatch)
    monkeypatch.setattr(run.harness, "capture_hardware", lambda: _hardware())
    _clear_asr_env_var()
    crashing = _AsrEnvCapturingStubSuite(raises=RuntimeError("synthetic crash"))
    monkeypatch.setattr(run, "_load_suite", lambda suite: crashing)

    exit_code = run._run_one("asr", _args(suite="asr", asr_model="acme/candidate-asr-model"))

    assert exit_code == 2
    assert crashing.captured_env == "acme/candidate-asr-model"
    assert os.environ.get(_ASR_ENV_VAR) is None, "a crash must not leak the override into later runs"


# --- T037c-4/C4: `_resolve_config`'s `asr_model` key reflects the override -


def test_resolve_config_asr_model_key_reflects_the_explicit_override():
    config = run._resolve_config(None, "acme/candidate-asr-model")
    assert config["asr_model"] == "acme/candidate-asr-model"


def test_resolve_config_asr_model_key_falls_back_to_config_default_when_no_override():
    config = run._resolve_config(None, None)
    assert config["asr_model"] == src_config.ASR_MODEL
