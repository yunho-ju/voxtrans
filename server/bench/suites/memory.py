"""`memory` suite (contracts/benchmark-cli.md, tasks.md T017).

Loads every model and reports **measured** resident memory (SC-005,
FR-005), plus the `vm_stat` page-out delta incurred by loading them.

The actual model-load + measurement happens in a dedicated child process
(`python -m bench.suites._memory_worker`), never in whichever process called
`run()` here. This is what keeps the measurement independent of suite
execution order: `--suite all` runs every suite in one long-lived process,
and a suite that ran before `memory`'s turn and never released its own model
(e.g. `translation`) would otherwise leave those weights resident too,
double-counting memory a clean startup never holds (Critic T020 round-1
major 1). A fresh subprocess starts from nothing every time.

Resident size is read as `phys_footprint` (`harness.read_phys_footprint`),
not `mx.get_active_memory()` and not `ps -o rss=`: the TTS model runs in its
own `multiprocessing` worker, so MLX's in-process accounting in the memory
worker process never sees it, and OS-level RSS (`ps`) does not see
MLX/Metal allocations at all -- both were tried and rejected; see
`harness.read_phys_footprint`'s docstring for the measured numbers that
ruled each out (Critic T020 round-1 blocker).

**Known gap, recorded rather than silently patched** (Critic T020 round-1
minor 2): `bench.budgets` has no `resident_memory_mb` entry, so this stage's
`within_budget` is always `True` and `--suite memory` can never exit `1` no
matter how much memory is resident. Adding a budget number here would be
this module unilaterally making the T024 reconciliation's call; the
constitution's own ceiling (8 GB) is stated in prose, not yet in
`bench/budgets.py`. Left for T024 to decide, not fixed here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .. import harness

_SERVER_DIR = Path(__file__).resolve().parents[2]
_WORKER_TIMEOUT_S = 300.0


def run(iterations: int, config: dict) -> tuple[list, list[dict]]:
    page_outs_before = harness.read_page_outs()

    result = subprocess.run(
        [sys.executable, "-m", "bench.suites._memory_worker"],
        cwd=str(_SERVER_DIR),
        capture_output=True,
        text=True,
        timeout=_WORKER_TIMEOUT_S,
        check=False,
    )
    page_outs_after = harness.read_page_outs()

    # `check=False` + an explicit branch, rather than `check=True`, because
    # `subprocess.CalledProcessError`'s default message is just the command
    # and the exit code -- the worker's actual stderr (why model loading
    # failed, a real traceback) would otherwise be silently dropped just
    # when it matters most for diagnosing the failure (Critic T020 round-2
    # minor 2).
    if result.returncode != 0:
        raise RuntimeError(
            f"memory worker exited {result.returncode}: {result.stderr.strip()[-2000:]}"
        )

    # The worker's own stdout may carry model-loading noise before its final
    # JSON line (mlx_audio/transformers print progress bars); only the last
    # line is the payload.
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    main_footprint = int(payload["main_phys_footprint_bytes"])
    tts_footprint = int(payload["tts_phys_footprint_bytes"])

    resident_bytes = main_footprint + tts_footprint
    resident_mb = resident_bytes / (1024 * 1024)
    page_out_delta = page_outs_after - page_outs_before

    stages = [harness.stage_timing("resident_memory_mb", resident_mb)]
    samples = [
        {
            "main_process_phys_footprint_bytes": main_footprint,
            "tts_process_phys_footprint_bytes": tts_footprint,
            "resident_bytes": resident_bytes,
            "resident_mb": resident_mb,
            "page_outs_before": page_outs_before,
            "page_outs_after": page_outs_after,
            "page_out_delta": page_out_delta,
        }
    ]
    return stages, samples
