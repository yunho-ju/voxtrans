"""T025 (US2 round 3): `audio_output_enqueue`'s scheduled buffers must report
when they have actually been *heard*, not merely handed to the renderer
(`data-model.md` §3 `PlaybackLifecycle.scheduled_buffers`,
`contracts/playback-protocol.md` "When the client sends it").

**Why this cannot be a Python/PortAudio test alone**: identical reasoning to
`test_output_reaches_device.py`'s module docstring -- this exercises
`AudioCapture.swift` directly via a `ctypes`-loaded dylib (same
`swiftc -emit-library` recipe), not a Python-level audio abstraction, because
the defect under test is specific to which `AVAudioPlayerNodeCompletionCallbackType`
`audio_output_enqueue` schedules with. Before T025, `scheduleBuffer` was called
with `completionHandler: nil` -- no completion signal exists at all, so the new
`audio_output_outstanding_buffers` export this module depends on is simply
absent from the pre-fix dylib (a symbol-lookup failure at fixture setup, not a
timing assertion failure -- this *is* the Red for this module, see class
docstring "the Red this module produces").

**Case (a) -- completion is real-time-paced, not instantaneous**: enqueue a
tone of known duration and poll `audio_output_outstanding_buffers()`. If the
completion signal fired the moment the buffer was scheduled (the
`.dataConsumed` bug contracts/playback-protocol.md explicitly warns
`audio_output_enqueue` must not reproduce -- that callback type fires when a
buffer reaches the renderer, not when it has been heard), the outstanding
count would read 0 almost immediately. Real playback pacing means it must stay
above 0 for close to the tone's actual duration.

**Case (b) -- exact accounting, no excess**: multiple buffers enqueued back to
back must each decrement the count exactly once -- sampled throughout
playback, the count must never go negative (a double-fired completion would
show as exactly that) and must settle at precisely 0, not get stuck positive
(an under-fired completion) nor overshoot.

Real playback to `BlackHole 2ch` (`audio_device_lock`-serialized, skipped if
absent) drives both cases -- no capture subprocess is needed here (unlike
`test_output_reaches_device.py`): both cases are about *when* the completion
signal arrives relative to wall-clock time, not about the content that
reached the device's input side, which T014 already covers.

**Known limitation, found by mutation testing (US2-round3 evidence, corrected
per Critic round-3 minor 2)**: reverting `audio_output_enqueue` to schedule
with `.dataConsumed` instead of `.dataPlayedBack` was *not* caught by either
case above -- both still passed, with completion arriving at roughly the
tone's real duration either way. This is **not** a peculiarity of the machine
this suite runs on: `.dataConsumed` and `.dataPlayedBack` do genuinely differ
(the former fires once a buffer is handed to the player node's internal
queue, the latter once the device has actually reproduced it as sound), but
that gap is bounded by roughly one buffer's worth of *device output latency*
-- the time already-consumed audio sits in the hardware/driver's own output
queue before actually being heard, typically single-digit to a few tens of
milliseconds. This module's own timing tolerance (`elapsed_s >
total_duration_s * 0.5`, chosen to cleanly separate "instantaneous" from
"real-time-paced") is far coarser than that gap for the buffer durations
tested (0.3s-2s) -- a completion firing 10-30ms early is invisible against a
tolerance measured in hundreds of milliseconds. This is a **structural
detection gap by this test's own design**, not something specific to this
particular machine or AVAudioEngine version. `.dataPlayedBack` is still the
correct choice: it is what `contracts/playback-protocol.md` explicitly
requires, independent of whether this harness's timing granularity can force
a failure for the alternative. Recorded here rather than tightened into a
tolerance band that would need to be tighter than device output latency
itself, which would trade a real (if structurally hard to detect at this
granularity) defect for a flakier test.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from integration import conftest
from integration.test_capture_fixture import BLACKHOLE

_SWIFT_SRC = (
    Path(__file__).resolve().parents[3] / "client" / "src-tauri" / "swift" / "AudioCapture.swift"
)

# `audio_output_enqueue` always builds buffers as float32 mono at this fixed
# rate (AudioCapture.swift), independent of whatever nominal rate the
# selected output device itself runs at.
TTS_SAMPLE_RATE = 24_000

TONE_FREQUENCY_HZ = 440.0
TONE_AMPLITUDE = 0.5

# Time given to `audio_output_start` to finish settling before anything is
# enqueued -- a small, non-load-bearing buffer, mirroring
# `test_output_reaches_device.py`'s `OUTPUT_ENGINE_SETTLE_S`.
OUTPUT_ENGINE_SETTLE_S = 0.5

# How often to poll `audio_output_outstanding_buffers()` while waiting for it
# to reach 0 -- fine enough to catch a near-instant (`.dataConsumed`-style)
# completion (case a) without adding meaningful overhead of its own.
POLL_INTERVAL_S = 0.05


@pytest.fixture(autouse=True)
def _exclusive_device_access(audio_device_lock):
    """Serialize every test in this module against other device users --
    audio devices are shared, stateful hardware (same rule every other
    hardware-backed module in this suite applies)."""
    with audio_device_lock():
        yield


@pytest.fixture(scope="module")
def audio_capture_dylib(tmp_path_factory) -> Path:
    """Compile `AudioCapture.swift` as a standalone dylib -- identical build
    recipe to `test_output_reaches_device.py`'s fixture of the same name
    (see that module for why each flag is required)."""
    if shutil.which("swiftc") is None:
        pytest.fail(
            "swiftc not found -- required to build AudioCapture.swift for this test; "
            "the same tool client/src-tauri/build.rs already requires to build the app"
        )
    if not _SWIFT_SRC.exists():
        pytest.fail(f"AudioCapture.swift not found at {_SWIFT_SRC}")

    out_dir = tmp_path_factory.mktemp("playback_completion_dylib")
    lib_path = out_dir / "libAudioCaptureTest.dylib"
    result = subprocess.run(
        [
            "swiftc",
            "-emit-library",
            "-module-name",
            "AudioCapture",
            "-enable-testing",
            "-o",
            str(lib_path),
            str(_SWIFT_SRC),
            "-framework",
            "CoreAudio",
            "-framework",
            "AudioToolbox",
            "-framework",
            "AVFoundation",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            "swiftc failed to build AudioCapture.swift as a dylib "
            f"(rc={result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return lib_path


@pytest.fixture
def audio_lib(audio_capture_dylib: Path):
    """Binds the completion-tracking symbols this module needs.
    `audio_output_outstanding_buffers` (T025) does not exist on the pre-fix
    dylib -- looking it up here is exactly what produces this module's Red
    (an `AttributeError` at fixture setup, not a timing assertion failure;
    see module docstring)."""
    lib = ctypes.CDLL(str(audio_capture_dylib))
    lib.audio_output_start.argtypes = [ctypes.c_char_p]
    lib.audio_output_start.restype = ctypes.c_bool
    lib.audio_output_enqueue.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    lib.audio_output_enqueue.restype = None
    lib.audio_output_stop.argtypes = []
    lib.audio_output_stop.restype = None
    lib.audio_output_outstanding_buffers.argtypes = []
    lib.audio_output_outstanding_buffers.restype = ctypes.c_long
    return lib


@pytest.fixture
def blackhole_uid() -> str:
    uid = conftest.find_core_audio_device_uid(BLACKHOLE)
    if uid is None:
        pytest.skip(f"{BLACKHOLE!r} not present on this machine")
    return uid


def _synthesize_tone_24k(duration_s: float) -> np.ndarray:
    t = np.arange(int(TTS_SAMPLE_RATE * duration_s)) / TTS_SAMPLE_RATE
    return (TONE_AMPLITUDE * np.sin(2 * np.pi * TONE_FREQUENCY_HZ * t)).astype(np.float32)


def _poll_until_zero(audio_lib, timeout_s: float) -> tuple[float, list[int]]:
    """Polls `audio_output_outstanding_buffers()` every `POLL_INTERVAL_S`
    until it reads 0 or `timeout_s` elapses. Returns the elapsed wall time
    at the moment it first read 0, and every sample observed along the way
    (for case (b)'s "never negative" check) -- bounded, never hangs."""
    samples: list[int] = []
    start = time.monotonic()
    deadline = start + timeout_s
    while time.monotonic() < deadline:
        count = audio_lib.audio_output_outstanding_buffers()
        samples.append(count)
        if count == 0:
            return time.monotonic() - start, samples
        time.sleep(POLL_INTERVAL_S)
    pytest.fail(
        f"audio_output_outstanding_buffers() did not reach 0 within {timeout_s:.1f}s "
        f"(last observed: {samples[-1] if samples else 'never polled'})"
    )


def test_completion_is_paced_by_real_playback_time_not_instantaneous(audio_lib, blackhole_uid):
    """Case (a): enqueuing `TONE_DURATION_S` of audio must keep the
    outstanding count above 0 for close to that long -- an instantaneous
    drop to 0 is the `.dataConsumed`-style defect contracts/playback-protocol.md
    warns against (a buffer reaching the renderer is not the same as a
    buffer having been heard)."""
    tone_duration_s = 2.0
    tone = _synthesize_tone_24k(tone_duration_s)

    started = audio_lib.audio_output_start(blackhole_uid.encode("utf-8"))
    try:
        assert started, "audio_output_start returned false for BlackHole's own UID"
        time.sleep(OUTPUT_ENGINE_SETTLE_S)

        assert audio_lib.audio_output_outstanding_buffers() == 0, (
            "expected no outstanding buffers before anything is enqueued"
        )

        tone_ptr = tone.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        audio_lib.audio_output_enqueue(tone_ptr, len(tone))

        # Immediately after enqueue, the buffer must still be outstanding --
        # an instant-complete signal would already read 0 here.
        immediate_count = audio_lib.audio_output_outstanding_buffers()
        assert immediate_count > 0, (
            f"expected the just-enqueued buffer to still be outstanding immediately after "
            f"enqueue, got count={immediate_count} -- looks like completion fired before "
            f"playback could possibly have happened"
        )

        elapsed_s, _ = _poll_until_zero(audio_lib, timeout_s=tone_duration_s + 5.0)
        # Generous lower bound (not tight to the exact tone duration): engine
        # start/settle jitter on shared CI-like hardware is real, but an
        # instantaneous completion (the defect) would read close to 0s, not
        # close to tone_duration_s -- this margin cleanly separates the two.
        assert elapsed_s > tone_duration_s * 0.5, (
            f"expected completion to take roughly {tone_duration_s:.1f}s (real playback time), "
            f"got {elapsed_s:.3f}s -- looks like it fired without actually waiting for playback"
        )
    finally:
        audio_lib.audio_output_stop()


def test_completion_count_matches_enqueue_count_with_no_excess(audio_lib, blackhole_uid):
    """Case (b): several buffers enqueued back to back must each complete
    exactly once -- sampled throughout playback, the outstanding count must
    never go negative (a double-fired completion) and must settle at
    exactly 0 (not get stuck positive, an under-fired completion)."""
    buffer_duration_s = 0.3
    buffer_count = 5
    tone = _synthesize_tone_24k(buffer_duration_s)

    started = audio_lib.audio_output_start(blackhole_uid.encode("utf-8"))
    try:
        assert started, "audio_output_start returned false for BlackHole's own UID"
        time.sleep(OUTPUT_ENGINE_SETTLE_S)

        tone_ptr = tone.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        for _ in range(buffer_count):
            audio_lib.audio_output_enqueue(tone_ptr, len(tone))

        total_duration_s = buffer_duration_s * buffer_count
        elapsed_s, samples = _poll_until_zero(audio_lib, timeout_s=total_duration_s + 5.0)

        assert all(count >= 0 for count in samples), (
            f"expected the outstanding count to never go negative (a negative reading means a "
            f"completion fired more times than a buffer was enqueued), got samples={samples}"
        )
        assert elapsed_s > total_duration_s * 0.5, (
            f"expected all {buffer_count} buffers (~{total_duration_s:.1f}s total) to take "
            f"roughly that long to complete, got {elapsed_s:.3f}s"
        )
    finally:
        audio_lib.audio_output_stop()
