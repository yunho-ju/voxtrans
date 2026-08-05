"""Tests for bench/harness.py (T011 Red; passes once T014 lands).

Hermetic: every JSON write/overwrite-guard test uses `tmp_path`, never the
real `specs/001-realtime-latency/evidence/` directory (standing Builder-brief
constraint 1). Hardware capture is exercised by monkeypatching
`subprocess.run` rather than asserting this machine's real sysctl/vm_stat
values — two differently-faked sysctl outputs producing two different
`Hardware.cpu` values is the proof that capture is live, not hardcoded
(constraint 2: mutation-test your own tests).
"""

import json
import subprocess

import pytest

import bench.harness as harness


# --- StageTiming ---------------------------------------------------------


def test_stage_timing_within_budget_true_when_at_budget():
    stage = harness.StageTiming(stage="asr", duration_ms=400.0, budget_ms=400.0)
    assert stage.within_budget is True


def test_stage_timing_within_budget_false_when_over_budget():
    stage = harness.StageTiming(stage="asr", duration_ms=400.1, budget_ms=400.0)
    assert stage.within_budget is False


def test_stage_timing_within_budget_true_when_no_budget_defined():
    stage = harness.StageTiming(stage="tts_chunk_mean", duration_ms=99999.0, budget_ms=None)
    assert stage.within_budget is True


def test_stage_timing_helper_resolves_budget_from_bench_budgets():
    stage = harness.stage_timing("translation", 650.0)
    assert stage.budget_ms == 700.0


def test_stage_timing_helper_leaves_budget_none_for_unbudgeted_stage():
    stage = harness.stage_timing("translation_first_token", 120.0)
    assert stage.budget_ms is None


# --- Hardware: sysctl/vm_stat capture, never hardcoded --------------------


def _fake_run(brand_string: str, mem_bytes: str, page_outs: str):
    def _run(cmd, **kwargs):
        if cmd[:2] == ["sysctl", "-n"]:
            value = brand_string if cmd[2] == "machdep.cpu.brand_string" else mem_bytes
            return subprocess.CompletedProcess(cmd, 0, stdout=value + "\n", stderr="")
        if cmd == ["vm_stat"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"Pageouts: {page_outs}.\n", stderr="")
        raise AssertionError(f"unexpected command in capture_hardware(): {cmd}")

    return _run


def test_capture_hardware_reads_cpu_from_sysctl(monkeypatch):
    monkeypatch.setattr(harness.subprocess, "run", _fake_run("Apple M9 Ultra", "12345", "7"))
    hw = harness.capture_hardware()
    assert hw.cpu == "Apple M9 Ultra"


def test_capture_hardware_reads_a_different_cpu_when_sysctl_output_changes(monkeypatch):
    """Proof this is not hardcoded: a second, unrelated fake CPU string must
    come back out unchanged."""
    monkeypatch.setattr(harness.subprocess, "run", _fake_run("Totally Different Chip", "999", "0"))
    hw = harness.capture_hardware()
    assert hw.cpu == "Totally Different Chip"


def test_capture_hardware_reads_memory_and_pageouts(monkeypatch):
    monkeypatch.setattr(harness.subprocess, "run", _fake_run("Apple M1 Pro", "34359738368", "40094"))
    hw = harness.capture_hardware()
    assert hw.memory_bytes == 34359738368
    assert hw.page_outs == 40094


def test_hardware_is_reference_true_for_reference_cpu(monkeypatch):
    monkeypatch.setattr(harness.subprocess, "run", _fake_run(harness.REFERENCE_CPU, "1", "0"))
    hw = harness.capture_hardware()
    assert hw.is_reference is True


def test_hardware_is_reference_false_for_non_reference_cpu(monkeypatch):
    monkeypatch.setattr(harness.subprocess, "run", _fake_run("Apple M2 Max", "1", "0"))
    hw = harness.capture_hardware()
    assert hw.is_reference is False


# --- BenchmarkRecord: hardware_mismatch, passed, JSON round-trip ----------


def _hardware(cpu: str) -> harness.Hardware:
    return harness.Hardware(
        cpu=cpu, memory_bytes=34359738368, os_version="26.5",
        python_version="3.13.0", mlx_version="0.32.0", page_outs=40094,
    )


def _record(hardware: harness.Hardware, stages: list) -> harness.BenchmarkRecord:
    return harness.BenchmarkRecord(
        suite="e2e", label="test", command="python -m bench.run --suite e2e --label test",
        timestamp="2026-08-05T00:00:00", hardware=hardware,
        config={"translation_model": "x"}, stages=stages, samples=[{"iteration": 0}],
    )


def test_hardware_mismatch_true_for_non_reference_cpu():
    record = _record(_hardware("Apple M2 Max"), [harness.stage_timing("asr", 100.0)])
    assert record.hardware_mismatch is True


def test_hardware_mismatch_false_for_reference_cpu():
    record = _record(_hardware(harness.REFERENCE_CPU), [harness.stage_timing("asr", 100.0)])
    assert record.hardware_mismatch is False


def test_record_passed_true_when_every_stage_within_budget():
    record = _record(
        _hardware(harness.REFERENCE_CPU),
        [harness.stage_timing("asr", 100.0), harness.stage_timing("translation", 100.0)],
    )
    assert record.passed is True


def test_record_passed_false_when_any_stage_over_budget():
    record = _record(
        _hardware(harness.REFERENCE_CPU),
        [harness.stage_timing("asr", 100.0), harness.stage_timing("translation", 9999.0)],
    )
    assert record.passed is False


def test_benchmark_record_round_trips_through_json():
    original = _record(
        _hardware(harness.REFERENCE_CPU),
        [harness.stage_timing("asr", 350.0), harness.stage_timing("translation", 9999.0)],
    )
    restored = harness.BenchmarkRecord.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.to_dict() == original.to_dict()


# --- Evidence writer: overwrite guard, failing record still written -------


def test_write_record_creates_file_at_suite_label_json(tmp_path):
    record = _record(_hardware(harness.REFERENCE_CPU), [harness.stage_timing("asr", 100.0)])
    path = harness.write_record(record, evidence_dir=tmp_path)
    assert path == tmp_path / "e2e-test.json"
    assert path.exists()


def test_write_record_refuses_overwrite_without_force(tmp_path):
    record = _record(_hardware(harness.REFERENCE_CPU), [harness.stage_timing("asr", 100.0)])
    harness.write_record(record, evidence_dir=tmp_path)
    with pytest.raises(harness.EvidenceWriteRefused):
        harness.write_record(record, evidence_dir=tmp_path)


def test_write_record_overwrites_when_force_given(tmp_path):
    record = _record(_hardware(harness.REFERENCE_CPU), [harness.stage_timing("asr", 100.0)])
    path = harness.write_record(record, evidence_dir=tmp_path)
    changed = _record(_hardware(harness.REFERENCE_CPU), [harness.stage_timing("asr", 111.0)])
    harness.write_record(changed, evidence_dir=tmp_path, force=True)
    on_disk = harness.BenchmarkRecord.from_dict(json.loads(path.read_text()))
    assert on_disk.stages[0].duration_ms == 111.0


def test_failing_record_is_still_written_to_disk(tmp_path):
    record = _record(_hardware(harness.REFERENCE_CPU), [harness.stage_timing("translation", 9999.0)])
    assert record.passed is False
    path = harness.write_record(record, evidence_dir=tmp_path)
    on_disk = harness.BenchmarkRecord.from_dict(json.loads(path.read_text()))
    assert on_disk.passed is False


# --- drift_ratio: numeric noise band --------------------------------------


def test_drift_ratio_flat_series_is_one():
    assert harness.drift_ratio([10.0, 10.0, 10.0], first_n=3, last_n=3) == pytest.approx(1.0)


def test_drift_ratio_above_one_when_last_window_is_slower():
    ratio = harness.drift_ratio([10.0, 10.0, 20.0, 20.0], first_n=2, last_n=2)
    assert ratio == pytest.approx(2.0)


def test_drift_ratio_rejects_empty_window():
    with pytest.raises(ValueError):
        harness.drift_ratio([], first_n=1, last_n=1)


def test_drift_noise_band_is_defined_numerically_as_1_15():
    assert harness.DRIFT_NOISE_BAND == 1.15


# --- read_phys_footprint: T020 round-1 Critic blocker --------------------
#
# `ps -o rss=` does not see MLX/Metal allocations (Critic's own reproduction:
# a 2147MB MLX-resident array measured as 35MB via `ps`, but 2.0G via
# `vmmap --summary`'s "Physical footprint" line). `footprint(1)`'s JSON
# output carries the same number as `phys_footprint`; these tests fake that
# JSON rather than asserting this machine's real footprint, for the same
# "never hardcoded, never ambient" reason the sysctl tests above do.


def _fake_footprint_run(phys_footprint_bytes: int):
    def _run(cmd, **kwargs):
        assert cmd[0] == "footprint"
        json_path = cmd[cmd.index("-j") + 1]
        pid = int(cmd[cmd.index("-p") + 1])
        with open(json_path, "w") as f:
            json.dump(
                {"processes": [{"pid": pid, "auxiliary": {"phys_footprint": phys_footprint_bytes}}]},
                f,
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _run


def test_read_phys_footprint_returns_the_auxiliary_value(monkeypatch):
    monkeypatch.setattr(harness.subprocess, "run", _fake_footprint_run(2_147_483_648))
    assert harness.read_phys_footprint(12345) == 2_147_483_648


def test_read_phys_footprint_reads_a_different_value_when_footprint_output_changes(monkeypatch):
    """Proof this is not hardcoded: a second, different footprint value must
    come back out unchanged (mirrors the sysctl mutation-proof tests above)."""
    monkeypatch.setattr(harness.subprocess, "run", _fake_footprint_run(999))
    assert harness.read_phys_footprint(1) == 999


def test_read_phys_footprint_raises_when_no_process_reported(monkeypatch):
    def _empty_run(cmd, **kwargs):
        json_path = cmd[cmd.index("-j") + 1]
        with open(json_path, "w") as f:
            json.dump({"processes": []}, f)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(harness.subprocess, "run", _empty_run)
    with pytest.raises(RuntimeError):
        harness.read_phys_footprint(99999)
