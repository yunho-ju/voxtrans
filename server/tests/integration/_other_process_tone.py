"""Standalone tone-player worker for `test_tap_self_exclusion.py` (T030).

Plays a synthesized tone into a named PortAudio output device from its
**own OS process** -- never imported by pytest directly (leading
underscore, does not match pytest's `test_*.py` collection pattern), only
ever invoked via `subprocess.run`/`Popen`.

**Why this must be a separate process, not a background thread**: the whole
point of the test this supports is that `sck_start_capture`'s global system
audio tap excludes *this pytest process's own* Core Audio process object
(`kAudioHardwarePropertyTranslatePIDToProcessObject`,
`AudioCapture.swift:23`, research.md D5) -- a tone played on a background
thread of the pytest process itself would carry the *same* PID and would be
excluded too, which would prove nothing about whether the tap still
captures audio from processes other than this one. A tone from a distinct
PID is the only way to exercise the "other process" half of the exclusion
proof.

Prints one line on completion:

    OK
    ABSENT <message>
    ERROR <message>

argv: <device_name> <frequency_hz> <duration_s> <sample_rate>
"""

from __future__ import annotations

import sys
from pathlib import Path

# Mirrors _capture_subprocess.py: `tests/` (this file's grandparent) must be
# on `sys.path` for `from integration import conftest` to resolve, exactly
# as pytest's own "prepend" import mode arranges when tests run normally --
# this script does not run under pytest, so it arranges that for itself.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

from integration import conftest  # noqa: E402


def main(argv: list[str]) -> int:
    device_name, frequency_hz, duration_s, sample_rate = argv[:4]
    frequency_hz = float(frequency_hz)
    duration_s = float(duration_s)
    sample_rate = int(sample_rate)

    output_index = conftest._resolve_portaudio_output_index(device_name)
    if output_index is None:
        print(f"ABSENT {device_name!r} has no PortAudio-visible output side")
        return 2

    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    tone = (0.8 * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)
    stereo = np.column_stack([tone, tone])

    try:
        sd.play(stereo, samplerate=sample_rate, device=output_index, blocking=True)
    except Exception as exc:  # noqa: BLE001 - reported to the parent process, not raised here
        print(f"ERROR {exc}")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
