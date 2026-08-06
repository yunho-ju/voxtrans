"""T030 (US2): the system audio tap must not capture this app's own output
(FR-006, `research.md` D5, `AudioCapture.swift:23`).

**Why this is verify-first, not implement-first**: `research.md` D5 marks
`kAudioHardwarePropertyTranslatePIDToProcessObject` and
`CATapDescription(stereoGlobalTapButExcludeProcesses:)` as *documented but
unverified on this machine*, and the initializer takes process **object**
IDs, not PIDs -- passing `getpid()` directly would silently exclude nothing.
Before writing the fix, a standalone `swiftc`-compiled probe (kept out of
this suite; its transcript is in `evidence/US2.md`) confirmed, on this
machine (macOS 26.5):

1. Translating this process's own PID via
   `kAudioHardwarePropertyTranslatePIDToProcessObject` returns a valid,
   non-`kAudioObjectUnknown` `AudioObjectID` (`127` on the probe run), while
   a PID guaranteed not to correspond to any running process returns
   `kAudioObjectUnknown` (`0`) exactly as documented -- no silent PID
   passthrough.
2. Passing that process object into
   `CATapDescription(stereoGlobalTapButExcludeProcesses:)` genuinely
   excludes this process's own audio: a tap excluding self read **zero**
   samples ever delivered while this process played a tone to BlackHole (no
   other audio source was active), against the *same* tap with an empty
   exclude list reading the tone clearly (rms ~0.48).
3. Excluding a **different**, real, currently CoreAudio-connected process
   (not self) left self's own tone fully audible in the tap -- ruling out
   "any non-empty exclude list breaks tap delivery for everyone" as an
   alternative explanation for (2)'s zero-samples result.
4. Tap and aggregate-device creation succeeded (`status=0`) in every phase
   above -- TCC permission is not blocking this on this machine.

This cleared the API for implementation. `AudioCapture.swift`'s
`sck_start_capture` now translates its own PID at capture-start time and
passes the result to `stereoGlobalTapButExcludeProcesses:`.

**Why an empty capture in this suite's own Test 1 is a valid pass, not an
error**: point 2 above is not incidental -- a global tap with its *sole*
audio source excluded has nothing left to mix, so the Core Audio Tap
genuinely delivers no callback invocations at all in that condition (not
merely quiet ones). Treating "zero samples" as a capture failure (the
convention `_capture_subprocess.py`/`test_input_selection.py` use for
device-backed captures, which always have *some* ambient signal or open
device to prove liveness) would make this specific test permanently
unsatisfiable even against a correct fix. Aliveness of the tap mechanism
itself is instead proven by a **separate** test
(`test_tap_still_captures_a_different_processes_audio_while_excluding_self`,
constraint 5: "a control that would pass with nothing meaningful emitted is
not a control") using a genuinely different OS process's audio, mirroring
probe finding (3) above.

**Why the "different process" control is a real OS process, not a
background thread**: `sck_start_capture`'s exclusion is keyed on *this
process's* PID. A tone played on a background thread of the same pytest
process would carry the same PID and would be excluded too -- proving
nothing about whether the tap still captures audio from processes other
than this one. `_other_process_tone.py` (this package) is invoked via
`subprocess.run`, guaranteeing a distinct PID for the whole test.

**Why this test does not build a second, unexcluded tap for comparison**
(the literal "(a) excluded tap ~= silence, (b) tap without exclusion =
tone detected" framing in the task): the production `sck_start_capture`
entry point always excludes self once the fix lands, with no parameter to
disable that, and adding a test-only unexcluded-tap variant to
`AudioCapture.swift` would be test scaffolding in production code for this
test alone. Proving the *same* production tap still captures a genuinely
different process's audio (this module's Test 2) is the stronger and more
directly relevant check for *this* feature (targeted exclusion, not
blanket muting), and is exactly what probe finding (3) already confirmed.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample

from integration import conftest
from integration.test_capture_fixture import (
    BLACKHOLE,
    MATCH_SCORE_MIN,
    SILENT_RMS_MAX,
    _matched_filter_score,
    _rms,
)

_SWIFT_SRC = (
    Path(__file__).resolve().parents[3] / "client" / "src-tauri" / "swift" / "AudioCapture.swift"
)
_OTHER_PROCESS_TONE_SCRIPT = Path(__file__).resolve().parent / "_other_process_tone.py"

# sck_start_capture's fixed conversion target (AudioCapture.swift
# `targetSampleRate: Double = 16000`, :83), independent of whatever rate
# either tone below is synthesized/played at.
TAP_CAPTURE_SAMPLE_RATE = 16_000

# audio_output_enqueue always builds buffers as float32 mono at this fixed
# rate (AudioCapture.swift :335-340), matching test_output_reaches_device.py.
TTS_SAMPLE_RATE = 24_000

# Sample rate the "other process" helper plays at -- an ordinary PortAudio
# rate, matching test_capture_fixture.py's own SAMPLE_RATE convention.
OTHER_PROCESS_SAMPLE_RATE = 48_000

SELF_TONE_FREQUENCY_HZ = 1000.0
OTHER_TONE_FREQUENCY_HZ = 3000.0  # well separated from the self tone
TONE_DURATION_S = 3.0

# Settle before any tone starts, after sck_start_capture returns -- mirrors
# the probe's own finding that the aggregate device's IO cycle needs a
# moment before it reliably delivers callbacks.
TAP_SETTLE_S = 1.0

# Settle before enqueuing, after audio_output_start returns -- matches
# test_output_reaches_device.py's OUTPUT_ENGINE_SETTLE_S.
OUTPUT_SETTLE_S = 0.5

# Extra time given after a tone's nominal duration ends, to catch device
# buffering/scheduling tail before stopping capture.
TAIL_MARGIN_S = 1.0

_AUDIO_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_ssize_t,
    ctypes.c_ssize_t,
    ctypes.c_ssize_t,
)
# T035 (US4): `sck_start_capture` gained a second, error-reporting callback
# (research.md D6, "surfacing a failed rebuild") -- this module doesn't
# change any default device, so it never fires, but the real production
# signature now requires it on every call.
_ERROR_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p)


@pytest.fixture(autouse=True)
def _exclusive_device_access(audio_device_lock):
    """Serialize every test in this module against other device users --
    same rule every other hardware-backed module in this suite applies."""
    with audio_device_lock():
        yield


@pytest.fixture(scope="module")
def audio_capture_dylib(tmp_path_factory) -> Path:
    """Compile `AudioCapture.swift` as a standalone dylib -- identical
    build recipe to `test_input_selection.py`/`test_output_reaches_device.py`
    (see either for why each flag is required)."""
    if shutil.which("swiftc") is None:
        pytest.fail(
            "swiftc not found -- required to build AudioCapture.swift for this test; "
            "the same tool client/src-tauri/build.rs already requires to build the app"
        )
    if not _SWIFT_SRC.exists():
        pytest.fail(f"AudioCapture.swift not found at {_SWIFT_SRC}")

    out_dir = tmp_path_factory.mktemp("tap_self_exclusion_dylib")
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
    lib = ctypes.CDLL(str(audio_capture_dylib))
    lib.sck_start_capture.restype = ctypes.c_bool
    lib.sck_start_capture.argtypes = [_AUDIO_CALLBACK, _ERROR_CALLBACK]
    lib.sck_stop_capture.restype = None
    lib.sck_stop_capture.argtypes = []
    lib.audio_output_start.restype = ctypes.c_bool
    lib.audio_output_start.argtypes = [ctypes.c_char_p]
    lib.audio_output_enqueue.restype = None
    lib.audio_output_enqueue.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    lib.audio_output_stop.restype = None
    lib.audio_output_stop.argtypes = []
    # T032 Critic minor 4: proves the self tone was genuinely scheduled for
    # playback (see test_own_tts_output_is_excluded_from_the_system_audio_tap).
    lib.audio_output_outstanding_buffers.restype = ctypes.c_long
    lib.audio_output_outstanding_buffers.argtypes = []
    return lib


@pytest.fixture
def blackhole_uid() -> str:
    uid = conftest.find_core_audio_device_uid(BLACKHOLE)
    if uid is None:
        pytest.skip(f"{BLACKHOLE!r} not present on this machine")
    return uid


def _synthesize_tone(frequency_hz: float, duration_s: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    return (0.8 * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)


def _tone_at_tap_capture_rate(tone: np.ndarray, tone_sample_rate: int) -> np.ndarray:
    new_len = int(len(tone) * TAP_CAPTURE_SAMPLE_RATE / tone_sample_rate)
    return resample(tone.astype(np.float64), new_len)


def _run_tap_capture(lib, action) -> tuple[np.ndarray, list[str]]:
    """Start the real `sck_start_capture(callback)` FFI entry point, run
    `action()` while it accumulates whatever the Core Audio Tap delivers
    (mono float32 at `TAP_CAPTURE_SAMPLE_RATE`), then stop and return
    everything captured, plus any error-callback reports. An empty capture
    array is a legitimate outcome (see module docstring), not collapsed
    into a failure here.

    T037 Critic minor 1: `_on_error` used to call `pytest.fail` directly
    from inside the ctypes callback -- a Python exception raised from a C
    callback frame is swallowed by the C call stack it is running under
    (`sck_start_capture`'s own internals), not propagated to the test.
    Collecting into a plain list and letting each *caller* assert on it
    from the main thread (mirroring `test_device_change.py`'s `_on_error`
    pattern) is the fix; it also lets
    `test_own_tts_output_is_excluded_from_the_system_audio_tap` -- whose
    entire point is a capture window with *no* audio at all, which the new
    build-watchdog (T037 major 1) cannot distinguish from a broken capture
    path -- deliberately not assert error-free, while the sibling test
    with real audio playing throughout still does.
    """
    chunks: list[np.ndarray] = []
    errors: list[str] = []
    lock = threading.Lock()

    def _on_audio(data_ptr, num_samples, _sample_rate, _channels) -> None:
        if num_samples <= 0 or not data_ptr:
            return
        buf = np.ctypeslib.as_array(data_ptr, shape=(num_samples,)).copy()
        with lock:
            chunks.append(buf)

    c_callback = _AUDIO_CALLBACK(_on_audio)

    def _on_error(message: bytes | None) -> None:
        with lock:
            errors.append(message.decode("utf-8") if message else "")

    c_error_callback = _ERROR_CALLBACK(_on_error)

    ok = lib.sck_start_capture(c_callback, c_error_callback)
    assert ok, "sck_start_capture returned false"
    time.sleep(TAP_SETTLE_S)
    try:
        action()
    finally:
        lib.sck_stop_capture()

    with lock:
        captured = np.concatenate(chunks).astype(np.float64) if chunks else np.zeros(0, dtype=np.float64)
        return captured, list(errors)


def test_own_tts_output_is_excluded_from_the_system_audio_tap(audio_lib, blackhole_uid):
    """FR-006 (a): with nothing else producing system audio, the app's own
    TTS output played to BlackHole must not appear in `sck_start_capture`'s
    tap once the app's own process is excluded.

    Red on the pre-fix `stereoGlobalTapButExcludeProcesses([])`: the app's
    own tone is the *only* audio source in this test, so if it isn't
    excluded it is exactly what the tap delivers -- captured non-empty,
    matching the tone.

    T032 Critic minor 4: the `captured.size == 0` early return below is a
    valid pass for genuine self-exclusion (see module docstring), but on its
    own it cannot distinguish that from a *different* regression -- the self
    tone silently never being played at all (e.g. a broken
    `audio_output_enqueue`), which would also leave the tap empty for a
    reason that has nothing to do with exclusion working. Asserting
    `audio_output_outstanding_buffers() > 0` immediately after enqueuing
    proves the tone was genuinely scheduled for real playback before this
    test's early-return path can absorb that other failure mode too.
    """
    tone = _synthesize_tone(SELF_TONE_FREQUENCY_HZ, TONE_DURATION_S, TTS_SAMPLE_RATE)

    def _play_self_tone() -> None:
        started = audio_lib.audio_output_start(blackhole_uid.encode("utf-8"))
        assert started, "audio_output_start returned false for BlackHole's own UID"
        try:
            time.sleep(OUTPUT_SETTLE_S)
            tone_ptr = tone.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            audio_lib.audio_output_enqueue(tone_ptr, len(tone))
            assert audio_lib.audio_output_outstanding_buffers() > 0, (
                "expected the self tone to be genuinely scheduled for playback right after "
                "enqueuing -- an empty tap capture below would otherwise be indistinguishable "
                "from the tone silently never having played at all"
            )
            time.sleep(TONE_DURATION_S + TAIL_MARGIN_S)
        finally:
            audio_lib.audio_output_stop()

    captured, errors = _run_tap_capture(audio_lib, _play_self_tone)

    if captured.size == 0:
        # Nothing left to mix once self is excluded and nothing else was
        # playing -- see module docstring. Aliveness of the tap mechanism
        # itself is proven by the separate test below, not by this one.
        # Deliberately does NOT assert `not errors` here (T037 Critic minor
        # 1): this scenario is exactly what the build-watchdog (T037 major
        # 1) cannot distinguish from a broken capture path -- a real
        # rebuild-failure report would look identical to this test's own
        # designed "nothing to capture" outcome from inside Swift. The
        # sibling test below, with real audio playing throughout, is where
        # "no spurious error" is actually checkable.
        return

    rms = _rms(captured)
    assert rms <= SILENT_RMS_MAX, (
        f"expected near-silence with the app's own process excluded from the tap, "
        f"got rms={rms:.6g}"
    )
    reference = _tone_at_tap_capture_rate(tone, TTS_SAMPLE_RATE)
    score = abs(_matched_filter_score(reference, captured))
    assert score < MATCH_SCORE_MIN, (
        f"expected the app's own tone NOT to appear in the tap once self-excluded "
        f"(score < {MATCH_SCORE_MIN}), got {score:.4f}"
    )


def test_tap_still_captures_a_different_processes_audio_while_excluding_self(audio_lib, blackhole_uid):
    """FR-006 (b), the constraint-5 control this exclusion needs: proves
    `sck_start_capture`'s tap is not simply deaf to everything once
    self-exclusion is applied -- it still captures a genuinely different
    process's audio. See module docstring for why this uses a real,
    separate OS process (`_other_process_tone.py`) rather than a second,
    unexcluded tap or a background thread of this same process.

    Must pass against *both* the pre-fix and post-fix code (nothing about
    this test depends on self-exclusion existing at all) -- it is the
    stable half of this pair, confirming the tap mechanism and this test's
    own harness work, so a Red result on the other test in this module can
    be attributed to the missing exclusion specifically.
    """
    other_tone = _synthesize_tone(OTHER_TONE_FREQUENCY_HZ, TONE_DURATION_S, OTHER_PROCESS_SAMPLE_RATE)

    def _play_other_process_tone() -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(_OTHER_PROCESS_TONE_SCRIPT),
                BLACKHOLE,
                str(OTHER_TONE_FREQUENCY_HZ),
                str(TONE_DURATION_S),
                str(OTHER_PROCESS_SAMPLE_RATE),
            ],
            capture_output=True,
            text=True,
            timeout=TONE_DURATION_S + 15.0,
        )
        stdout = proc.stdout.strip()
        if stdout.startswith("ABSENT "):
            pytest.skip(stdout[len("ABSENT ") :])
        assert proc.returncode == 0 and stdout == "OK", (
            f"other-process tone player failed (rc={proc.returncode}): "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        time.sleep(TAIL_MARGIN_S)

    captured, errors = _run_tap_capture(audio_lib, _play_other_process_tone)

    assert not errors, f"expected no rebuild-failure reports -- real audio plays throughout this test, got: {errors}"
    assert captured.size > 0, (
        "expected the tap to still capture a different process's audio while "
        "self-excluded -- an empty capture here means the tap is dead, not "
        "correctly excluding only itself"
    )
    rms = _rms(captured)
    assert rms > SILENT_RMS_MAX, (
        f"expected live audio from the other process (rms > {SILENT_RMS_MAX}), got {rms:.6g}"
    )
    reference = _tone_at_tap_capture_rate(other_tone, OTHER_PROCESS_SAMPLE_RATE)
    score = abs(_matched_filter_score(reference, captured))
    assert score > MATCH_SCORE_MIN, (
        f"expected the other process's tone to still be captured by the tap "
        f"(score > {MATCH_SCORE_MIN}), got {score:.4f}"
    )
