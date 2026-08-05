"""Benchmark harness plumbing: `StageTiming`, `Hardware`, `BenchmarkRecord`,
the evidence writer, and `drift_ratio` (data-model.md §1-3, tasks.md T014).

`bench/run.py` is the only intended caller of `write_record`; suites under
`bench/suites/` produce `StageTiming`/sample lists and hand them to `run.py`,
which assembles the `BenchmarkRecord`.

Hardware is always captured fresh from `sysctl`/`vm_stat`/`platform` —
never hardcoded — so every record is self-describing about the machine it
ran on (constitution Principle II, data-model.md §3).
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib.metadata import version as _pkg_version
from pathlib import Path

from . import budgets

# The constitution's budget table is stated for exactly this machine
# (Performance Standards and Hardware Budget § "Reference hardware").
REFERENCE_CPU = "Apple M1 Pro"

# Noise band for `drift_ratio`: at or below this is flat, above it is
# degradation. Defined numerically here and nowhere else — "beyond
# measurement noise" is not a testable phrase on its own
# (contracts/benchmark-cli.md "Drift and noise", analyze finding H1).
DRIFT_NOISE_BAND = 1.15

EVIDENCE_DIR = (
    Path(__file__).resolve().parents[2] / "specs" / "001-realtime-latency" / "evidence"
)


# --- StageTiming -----------------------------------------------------------


@dataclass(frozen=True)
class StageTiming:
    """One timed (or derived) pipeline stage (data-model.md §1)."""

    stage: str
    duration_ms: float
    budget_ms: float | None

    @property
    def within_budget(self) -> bool:
        return self.budget_ms is None or self.duration_ms <= self.budget_ms

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "duration_ms": self.duration_ms,
            "budget_ms": self.budget_ms,
            "within_budget": self.within_budget,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StageTiming":
        return cls(stage=data["stage"], duration_ms=data["duration_ms"], budget_ms=data["budget_ms"])


def stage_timing(stage: str, duration_ms: float) -> StageTiming:
    """Build a `StageTiming`, resolving `budget_ms` from `bench.budgets` when
    the stage has one and leaving it `None` otherwise — e.g.
    `translation_first_token` has no constitutional budget of its own."""
    budget = budgets.BUDGETS.get(stage)
    return StageTiming(stage=stage, duration_ms=duration_ms, budget_ms=(budget.target_ms if budget else None))


# --- Hardware ---------------------------------------------------------------


def _sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout.strip()


def read_page_outs() -> int:
    """Current cumulative `vm_stat` "Pageouts" counter."""
    output = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"Pageouts:\s+(\d+)", output)
    if not match:
        raise RuntimeError(f"could not parse Pageouts from vm_stat output: {output!r}")
    return int(match.group(1))


def read_phys_footprint(pid: int) -> int:
    """`pid`'s physical footprint in bytes -- the resident-memory number the
    constitution's 8 GB ceiling is stated against, read via macOS's
    `footprint(1)` JSON output.

    Deliberately **not** `ps -o rss=`: `ps`'s RSS column does not see
    MLX/Metal allocations at all (Critic blocker, T020 round 1 -- a 2147 MB
    MLX-resident array measured as 35 MB via `ps`, while `vmmap --summary`'s
    "Physical footprint" line and `footprint(1)`'s `phys_footprint` both
    correctly showed ~2.0 GB). `footprint(1)` is used over parsing
    `vmmap --summary`'s text because its `-j` flag emits structured JSON with
    the same figure, which is far less fragile to parse.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        json_path = Path(handle.name)
    try:
        subprocess.run(
            ["footprint", "-p", str(pid), "-j", str(json_path)],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(json_path.read_text())
    finally:
        json_path.unlink(missing_ok=True)

    processes = payload.get("processes", [])
    if not processes:
        raise RuntimeError(f"footprint(1) reported no process for pid {pid}")
    return int(processes[0]["auxiliary"]["phys_footprint"])


@dataclass(frozen=True)
class Hardware:
    """Captured at run time, never hardcoded (data-model.md §3)."""

    cpu: str
    memory_bytes: int
    os_version: str
    python_version: str
    mlx_version: str
    page_outs: int

    @property
    def is_reference(self) -> bool:
        return self.cpu == REFERENCE_CPU

    def to_dict(self) -> dict:
        return {
            "cpu": self.cpu,
            "memory_bytes": self.memory_bytes,
            "os_version": self.os_version,
            "python_version": self.python_version,
            "mlx_version": self.mlx_version,
            "page_outs": self.page_outs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Hardware":
        return cls(**data)


def capture_hardware() -> Hardware:
    """Capture the current machine's hardware state via `sysctl`/`vm_stat`/
    `platform`/`importlib.metadata` — every field is read live (data-model.md
    §3); nothing here is a constant."""
    return Hardware(
        cpu=_sysctl("machdep.cpu.brand_string"),
        memory_bytes=int(_sysctl("hw.memsize")),
        os_version=platform.mac_ver()[0],
        python_version=platform.python_version(),
        mlx_version=_pkg_version("mlx"),
        page_outs=read_page_outs(),
    )


# --- BenchmarkRecord ---------------------------------------------------------


@dataclass
class BenchmarkRecord:
    """One complete run of one suite (data-model.md §2). `hardware_mismatch`
    and `passed` are derived, not caller-supplied, so there is exactly one
    place that can compute them wrong."""

    suite: str
    label: str
    command: str
    timestamp: str
    hardware: Hardware
    config: dict
    stages: list[StageTiming]
    samples: list[dict]
    hardware_mismatch: bool = field(init=False)
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        self.hardware_mismatch = not self.hardware.is_reference
        self.passed = all(stage.within_budget for stage in self.stages)

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "label": self.label,
            "command": self.command,
            "timestamp": self.timestamp,
            "hardware": self.hardware.to_dict(),
            "hardware_mismatch": self.hardware_mismatch,
            "config": self.config,
            "stages": [stage.to_dict() for stage in self.stages],
            "samples": self.samples,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkRecord":
        return cls(
            suite=data["suite"],
            label=data["label"],
            command=data["command"],
            timestamp=data["timestamp"],
            hardware=Hardware.from_dict(data["hardware"]),
            config=data["config"],
            stages=[StageTiming.from_dict(s) for s in data["stages"]],
            samples=data["samples"],
        )


# --- Evidence writer: overwrite guard --------------------------------------


class EvidenceWriteRefused(Exception):
    """Raised when writing a record would silently overwrite an existing
    evidence file and `force=True` was not given (contracts/benchmark-cli.md:
    "a baseline cannot be silently replaced by a post-optimization run")."""


def write_record(
    record: BenchmarkRecord, *, force: bool = False, evidence_dir: Path | None = None
) -> Path:
    """Write `record` to `<evidence_dir>/<suite>-<label>.json`, refusing to
    overwrite an existing file unless `force=True`. Writes unconditionally on
    `record.passed is False` too — a failing record is the Red artifact and
    must survive on disk (data-model.md §2 invariant, constitution Principle
    III). `evidence_dir` defaults to the real feature evidence directory;
    tests always pass `tmp_path` explicitly instead."""
    directory = evidence_dir if evidence_dir is not None else EVIDENCE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.suite}-{record.label}.json"
    if path.exists() and not force:
        raise EvidenceWriteRefused(
            f"{path} already exists; pass force=True (CLI: --force) to overwrite it"
        )
    path.write_text(json.dumps(record.to_dict(), indent=2) + "\n")
    return path


# --- drift ratio -------------------------------------------------------------


def drift_ratio(durations_ms: list[float], first_n: int, last_n: int) -> float:
    """Mean of the last `last_n` samples ÷ mean of the first `first_n`
    samples (contracts/benchmark-cli.md "Drift and noise"). A ratio at or
    below `DRIFT_NOISE_BAND` (1.15) is flat; above it is degradation."""
    if not durations_ms or first_n <= 0 or last_n <= 0:
        raise ValueError("drift_ratio requires a non-empty series and positive window sizes")
    first_window = durations_ms[:first_n]
    last_window = durations_ms[-last_n:]
    first_mean = sum(first_window) / len(first_window)
    last_mean = sum(last_window) / len(last_window)
    return last_mean / first_mean
