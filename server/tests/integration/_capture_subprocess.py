"""Standalone capture worker for `test_output_reaches_device.py`.

Runs `conftest.record_from_device` in its own OS process. Never imported by
pytest directly (leading underscore, does not match pytest's default
`test_*.py` collection pattern) -- it is only ever invoked via
`subprocess.run`/`Popen`.

**Why a subprocess and not the in-process background thread the first
version of this module used**: Critic round 1 found (and this file's
sibling module's docstring records) that opening a PortAudio `InputStream`
*in the same process* as this suite's `ctypes`-loaded, `AVAudioEngine`-based
`audio_output_start` -- on the same device -- intermittently reads all-zero
samples regardless of which version of `AudioCapture.swift` is loaded, while
a wholly separate process capturing the same device at the same moment
reads real content. That in-process contention both (a) forced an
unrealistic "capture must open before the engine starts" ordering to get
any signal at all, and (b) meant an observed "all zero" capture could not
be trusted as specific to the code under test. Moving the capture into its
own process removes the contention and the ordering constraint it forced.

Writes the captured samples to the given `.npy` path (parent process reads
them back) and reports outcome on stdout, one line:

    OK <sample_rate> <requested_duration_s> <actual_duration_s>
    ABSENT <message>
    ERROR <message>

argv: <device_name> <duration_s> <sample_rate> <output_npy_path>
"""

from __future__ import annotations

import sys
from pathlib import Path

# Mirrors every test module in this package: `tests/` (this file's
# grandparent) is what needs to be on `sys.path` for `from integration
# import conftest` to resolve, exactly as pytest's own "prepend" import
# mode already arranges when tests run normally -- this script does not run
# under pytest, so it has to arrange that for itself.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from integration import conftest  # noqa: E402


def main(argv: list[str]) -> int:
    device_name, duration_s, sample_rate, output_path = argv[:4]
    try:
        result = conftest.record_from_device(
            device_name, float(duration_s), sample_rate=int(sample_rate)
        )
    except conftest.DeviceAbsentError as exc:
        print(f"ABSENT {exc}")
        return 2
    except conftest.DeviceCaptureError as exc:
        print(f"ERROR {exc}")
        return 1

    np.save(output_path, result.samples)
    print(f"OK {result.sample_rate} {result.requested_duration_s} {result.actual_duration_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
