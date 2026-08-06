"""`bench` CLI (contracts/benchmark-cli.md, tasks.md T015).

Invocation (always from `server/`, through the feature virtualenv):

    python -m bench.run --suite <suite> [--label <label>] [--iterations N]
                         [--model <id>] [--force]

Exit codes: `0` every stage was within budget; `1` at least one budget
assertion failed (the Red state); `2` preconditions unmet (missing venv
import, missing model weights, missing fixture, or -- `subtitle` only -- the
suite's own server port already in use); `3` hardware mismatch (this machine
is not the reference CPU the constitution's budgets are stated for).

Five preconditions run before any clock starts
(contracts/benchmark-cli.md "Preconditions checked before any clock starts"):

    1. venv imports mlx, mlx_lm, mlx_audio.
    2. Every model id in the resolved config -- including the *ambient*
       `VOXTRANS_TRANSLATION_MODEL` override when no `--model` flag is
       given, and the Silero VAD cache, contract precondition 3a --
       resolves to a complete local snapshot (`bench.fetch_models`, never
       touching the network here).
    3. The suite's required fixture WAV exists at 16 kHz with the expected
       duration (only for suites that actually touch a fixture WAV).
    4. `sysctl` reports the reference CPU.
    5. For the `subtitle` suite only: this run's server port is not already
       occupied by a stale process (the suite starts its own server and must
       never reuse one it did not start).

Checks 1, 2, 3, and 5 run **before** `Hardware` is ever captured; check 4
runs last, against the `Hardware` captured right after they pass. This
ordering is deliberate (Critic finding, T020 round 1, major 2): capturing
hardware calls `importlib.metadata.version("mlx")`, which raises if `mlx`
is not installed at all -- if that ran first, a broken venv would crash
with an uncaught exception (Python's generic exit code 1) instead of
failing check 1 with exit code 2, indistinguishable from a legitimate
budget-Red run. Running the venv-import check first means it always catches
that case before `capture_hardware` is ever reached.

Failure of 1-3 or 5 exits `2`; failure of 4 exits `3` (contract "Exit
codes"). An additional, non-numbered overwrite guard (harness.py's
`EvidenceWriteRefused`, checked here before a suite runs so no model
inference is wasted) also exits `2`: it is an operational safety check, not
one of the five timing preconditions above, and is documented separately so
the two are never conflated.

A suite that raises once its preconditions have passed (a crash, not a
budget miss) is caught and reported as exit `2` -- a harness failure, not
`1` -- and does not stop the remaining suites when `--suite all` is running
several in sequence (Critic finding, T020 round 1, major 3).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import socket
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import fetch_models, harness

SUITES: tuple[str, ...] = ("e2e", "asr", "translation", "tts_chunks", "subtitle", "memory")

DEFAULT_ITERATIONS: dict[str, int] = {
    "e2e": 1,
    "asr": 3,
    "translation": 3,
    "tts_chunks": 20,
    "subtitle": 1,
    "memory": 1,
}

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
_E2E_FIXTURE = _FIXTURES_DIR / "utterance_ko_5s.wav"
_WARMUP_FIXTURE = _FIXTURES_DIR / "warmup_500ms.wav"

# Only suites that actually read a fixture WAV need one checked (Critic
# T020 round-1 minor 1): `translation`, `tts_chunks`, and `memory` never
# touch either fixture, so gating them on presence was either a no-op or,
# if a fixture were ever missing/corrupt, a false failure for a suite that
# does not need it. `e2e` also reads `_WARMUP_FIXTURE` for its warm-up pass
# (T020 round-1 major 4) -- listed here too (Critic T020 round-2 minor 1)
# so a missing warm-up fixture fails this precondition (exit 2, before any
# clock starts) instead of surfacing only once `e2e.run()` is already
# mid-flight. `asr` (T037c-4) reads both fixtures too, the same shape as
# `e2e`'s ASR leg. Duration ranges mirror `tests/fixtures/build_fixtures.sh`'s
# own `verify_wav` calls for each file.
_SUITE_FIXTURES: dict[str, tuple[tuple[Path, float, float], ...]] = {
    "e2e": ((_E2E_FIXTURE, 4.9, 5.1), (_WARMUP_FIXTURE, 0.4, 0.6)),
    "asr": ((_E2E_FIXTURE, 4.9, 5.1), (_WARMUP_FIXTURE, 0.4, 0.6)),
    "subtitle": ((_E2E_FIXTURE, 4.9, 5.1),),
}


@dataclass(frozen=True)
class PreconditionFailure:
    check: str
    reason: str
    exit_code: int


# --- the five preconditions --------------------------------------------------


def _check_imports() -> PreconditionFailure | None:
    try:
        import mlx  # noqa: F401
        import mlx_audio  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        return PreconditionFailure("venv_imports", str(exc), 2)
    return None


def _check_models(
    model_override: str | None, asr_model_override: str | None = None
) -> PreconditionFailure | None:
    # Validate the *resolved* translation and ASR models, not only an
    # explicit `--model`/`--asr-model` (Critic round-2 minor 1; T037c-3/C3
    # mirrors the same reasoning for `--asr-model`): `VOXTRANS_TRANSLATION_MODEL`/
    # `VOXTRANS_ASR_MODEL` can also be set ambiently (no flag at all), and an
    # invalid ambient override must fail this precondition -- exit 2, before
    # any clock starts -- rather than surface only mid-run as a suite crash
    # or, worse, a silent download inside a timed region.
    from src import config as src_config

    resolved_translation_model = model_override or src_config.TRANSLATION_MODEL
    resolved_asr_model = asr_model_override or src_config.ASR_MODEL
    repo_ids = set(fetch_models.MODEL_REPOS)
    repo_ids.add(resolved_translation_model)
    repo_ids.add(resolved_asr_model)
    for repo_id in sorted(repo_ids):
        result = fetch_models.verify_model(repo_id)
        if not result.ok:
            return PreconditionFailure("model_snapshot", f"{repo_id}: {result.reason}", 2)
    vad_result = fetch_models.verify_silero_vad()
    if not vad_result.ok:
        return PreconditionFailure("model_snapshot", f"silero-vad: {vad_result.reason}", 2)
    return None


def _check_fixture(suite: str) -> PreconditionFailure | None:
    fixtures = _SUITE_FIXTURES.get(suite, ())
    if not fixtures:
        return None  # this suite reads no fixture WAV

    from src.config import SAMPLE_RATE

    import soundfile as sf

    for fixture, min_duration_s, max_duration_s in fixtures:
        if not fixture.exists():
            return PreconditionFailure("fixture", f"missing fixture: {fixture}", 2)
        info = sf.info(str(fixture))
        if info.samplerate != SAMPLE_RATE:
            return PreconditionFailure(
                "fixture", f"{fixture.name} samplerate {info.samplerate} != {SAMPLE_RATE}", 2
            )
        if not (min_duration_s <= info.duration <= max_duration_s):
            return PreconditionFailure(
                "fixture",
                f"{fixture.name} duration {info.duration:.3f}s not in "
                f"[{min_duration_s}, {max_duration_s}]",
                2,
            )
    return None


def _check_reference_cpu(hw: harness.Hardware) -> PreconditionFailure | None:
    if not hw.is_reference:
        return PreconditionFailure(
            "reference_cpu", f"cpu={hw.cpu!r} != {harness.REFERENCE_CPU!r}", 3
        )
    return None


def _check_subtitle_server_port_free(suite: str) -> PreconditionFailure | None:
    if suite != "subtitle":
        return None
    from src.config import PORT

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", PORT)) == 0:
            return PreconditionFailure(
                "subtitle_server",
                f"port {PORT} is already in use by another process; the subtitle "
                "suite owns its own server lifecycle and must not reuse a stale one",
                2,
            )
    return None


def run_preconditions(
    suite: str, model_override: str | None, asr_model_override: str | None = None
) -> tuple[PreconditionFailure | None, harness.Hardware | None]:
    """Run the five checks, returning `(failure, hardware)`.

    Checks 1 (imports), 2 (models), 3 (fixture), and 5 (subtitle port) run
    first and need no `Hardware` at all. `Hardware` is captured only after
    all four pass, and check 4 (reference CPU) runs against exactly that
    capture. This order is load-bearing (Critic finding, T020 round 1,
    major 2): `capture_hardware()` raises if `mlx` is not installed, so
    capturing it before check 1 would let a broken venv crash with an
    uncaught exception instead of failing check 1 with exit code 2.

    `hardware` is `None` only when a pre-hardware check already failed --
    no clock has started, and hardware was never captured, when that
    happens.
    """
    for check in (
        _check_imports,
        lambda: _check_models(model_override, asr_model_override),
        lambda: _check_fixture(suite),
        lambda: _check_subtitle_server_port_free(suite),
    ):
        failure = check()
        if failure is not None:
            return failure, None

    hw = harness.capture_hardware()
    return _check_reference_cpu(hw), hw


# --- suite dispatch -----------------------------------------------------------

_TRANSLATION_MODEL_ENV_VAR = "VOXTRANS_TRANSLATION_MODEL"


@contextlib.contextmanager
def _translation_model_env_override(model: str | None):
    """Temporarily set `VOXTRANS_TRANSLATION_MODEL` to `model` for the
    duration of the `with` block, restoring the previous value (or its
    absence) on every exit path -- success, suite crash, or precondition
    short-circuit alike. A no-op when `model` is falsy.

    `--model` "overrides VOXTRANS_TRANSLATION_MODEL for this run only"
    (contracts/benchmark-cli.md:21) -- Critic round-2 major 3: that promise
    is broken by only ever changing the resolved `config` dict, because
    that dict is not the only place a model id gets resolved from. Every
    subprocess this run spawns (`memory`'s worker via
    `bench.suites._memory_worker`, `subtitle`'s own `src.main` server)
    inherits this process's environment at spawn time by default
    (`subprocess.run`/`Popen` with no `env=` kwarg), and any in-process
    module that reads `src.config.TRANSLATION_MODEL` at its own first
    import (e.g. `e2e`'s `TranslationEngine()`) resolves it from whatever
    this process's environment says at that moment. Setting the override
    here -- wrapping everything from `run_preconditions` onward, in
    `_run_one` -- is what makes every one of those sites see it, since any
    of them may be the first thing to import `src.config` in this process.
    """
    if not model:
        yield
        return
    previous = os.environ.get(_TRANSLATION_MODEL_ENV_VAR)
    os.environ[_TRANSLATION_MODEL_ENV_VAR] = model
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_TRANSLATION_MODEL_ENV_VAR, None)
        else:
            os.environ[_TRANSLATION_MODEL_ENV_VAR] = previous


_ASR_MODEL_ENV_VAR = "VOXTRANS_ASR_MODEL"


@contextlib.contextmanager
def _asr_model_env_override(model: str | None):
    """Temporarily set `VOXTRANS_ASR_MODEL` to `model` for the duration of
    the `with` block, restoring the previous value (or its absence) on
    every exit path -- exact mirror of `_translation_model_env_override`
    above (T037c-4/C4): `--asr-model` must reach every model-loading path
    this run touches -- `memory`'s worker subprocess, and any in-process
    module that reads `src.config.ASR_MODEL` at its own first import (e.g.
    `asr.py`'s `ASREngine()`) -- not just the resolved `config` dict."""
    if not model:
        yield
        return
    previous = os.environ.get(_ASR_MODEL_ENV_VAR)
    os.environ[_ASR_MODEL_ENV_VAR] = model
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_ASR_MODEL_ENV_VAR, None)
        else:
            os.environ[_ASR_MODEL_ENV_VAR] = previous


def _resolve_config(model_override: str | None, asr_model_override: str | None = None) -> dict:
    from src import config as src_config

    return {
        "translation_model": model_override or src_config.TRANSLATION_MODEL,
        "asr_model": asr_model_override or src_config.ASR_MODEL,
        "tts_model": src_config.TTS_MODEL,
        "vad_silence_ms": str(src_config.VAD_SILENCE_DURATION_MS),
        "mic_vad_silence_ms": str(src_config.MIC_VAD_SILENCE_DURATION_MS),
    }


def _load_suite(suite: str):
    if suite == "translation":
        from .suites import translation as suite_mod
    elif suite == "tts_chunks":
        from .suites import tts_chunks as suite_mod
    elif suite == "e2e":
        from .suites import e2e as suite_mod
    elif suite == "asr":
        from .suites import asr as suite_mod
    elif suite == "memory":
        from .suites import memory as suite_mod
    elif suite == "subtitle":
        from .suites import subtitle as suite_mod
    else:
        raise ValueError(f"unknown suite: {suite}")
    return suite_mod


# --- CLI ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bench.run", description=__doc__)
    parser.add_argument("--suite", required=True, choices=(*SUITES, "all"))
    parser.add_argument("--label", default="adhoc")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def _format_command(suite: str, args: argparse.Namespace, iterations: int) -> str:
    parts = ["python", "-m", "bench.run", "--suite", suite, "--label", args.label, "--iterations", str(iterations)]
    if args.model:
        parts += ["--model", args.model]
    if args.asr_model:
        parts += ["--asr-model", args.asr_model]
    if args.force:
        parts.append("--force")
    return " ".join(parts)


def _unit_for_stage(stage_name: str) -> str:
    """`StageTiming.duration_ms` is overloaded by the `memory` suite to
    carry megabytes, not milliseconds (`resident_memory_mb`) -- printing
    that value with a hardcoded "ms" suffix would misreport what the number
    actually is (self-caught during this round's live memory-suite
    verification, T020 round 1). Every other stage genuinely is a duration."""
    return "MB" if stage_name.endswith("_mb") else "ms"


def _format_table(record: harness.BenchmarkRecord) -> str:
    hw = record.hardware
    gb = hw.memory_bytes / (1024**3)
    lines = [
        f"suite={record.suite} label={record.label}  hw={hw.cpu}/{gb:.0f}GB  "
        f"os={hw.os_version}  mlx={hw.mlx_version}",
        f"  {'stage':<24}{'measured':>12}{'budget':>12}   verdict",
    ]
    for stage in record.stages:
        unit = _unit_for_stage(stage.stage)
        budget_str = f"{stage.budget_ms:.1f}{unit}" if stage.budget_ms is not None else "n/a"
        verdict = "PASS" if stage.within_budget else "FAIL"
        lines.append(f"  {stage.stage:<24}{stage.duration_ms:>10.1f}{unit}{budget_str:>12}   {verdict}")
    lines.append("  " + "-" * 56)
    return "\n".join(lines)


def _run_one(suite: str, args: argparse.Namespace) -> int:
    evidence_path = harness.EVIDENCE_DIR / f"{suite}-{args.label}.json"
    if evidence_path.exists() and not args.force:
        print(
            f"refusing to overwrite existing {evidence_path} without --force",
            file=sys.stderr,
        )
        return 2

    # Everything below runs inside both env-override windows (Critic round-2
    # major 3; T037c-4/C4 for `--asr-model`): `--model`/`--asr-model` must
    # each reach every model-loading path this run touches --
    # preconditions' own snapshot checks, the resolved `config` dict, and
    # any subprocess a suite spawns -- not just the `config` dict alone. See
    # `_translation_model_env_override`'s/`_asr_model_env_override`'s
    # docstrings.
    with _translation_model_env_override(args.model), _asr_model_env_override(args.asr_model):
        failure, hw = run_preconditions(suite, args.model, args.asr_model)
        if failure is not None:
            print(f"precondition failed [{failure.check}]: {failure.reason}", file=sys.stderr)
            return failure.exit_code

        config = _resolve_config(args.model, args.asr_model)
        iterations = args.iterations or DEFAULT_ITERATIONS[suite]
        suite_mod = _load_suite(suite)

        # A suite that raises after its preconditions passed is a harness
        # failure (a crash), not a budget miss -- it must exit 2, never the
        # generic exit 1 an uncaught exception would otherwise produce, and it
        # must not stop the remaining suites when `--suite all` is running
        # several in sequence (Critic finding, T020 round 1, major 3).
        try:
            stages, samples = suite_mod.run(iterations=iterations, config=config)
        except Exception as exc:
            print(f"suite {suite!r} crashed: {exc!r}", file=sys.stderr)
            return 2

        record = harness.BenchmarkRecord(
            suite=suite,
            label=args.label,
            command=_format_command(suite, args, iterations),
            timestamp=datetime.now().isoformat(),
            hardware=hw,
            config=config,
            stages=stages,
            samples=samples,
        )
        harness.write_record(record, force=args.force)
        print(_format_table(record))
        return 0 if record.passed else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suites = SUITES if args.suite == "all" else (args.suite,)

    exit_codes = [_run_one(suite, args) for suite in suites]
    for priority in (2, 3, 1):
        if priority in exit_codes:
            return priority
    return 0


if __name__ == "__main__":
    sys.exit(main())
