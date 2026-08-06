"""T033 (US4): capture survives a default-output-device change while it is
running (spec.md SC-004, FR-008, research.md D6).

**The bug this closes** (`specs/002-audio-routing/briefs/app-batch-round1.md`
"이 티켓이 닫는 실증 버그"): on 2026-08-06, the system default output switched to
AirPods mid-session and the existing system-audio tap went silent -- capture
never recovered without restarting the app. `AudioCapture.swift` had never
registered a `kAudioHardwarePropertyDefaultOutputDevice` listener or an
`AVAudioEngineConfigurationChange` observer (grep-confirmed 0 hits before this
task, recorded in `evidence/T033-T034-red.md`). T035 added both listeners;
this module verifies the result.

**CORRECTION (T037 Critic round 1, recorded in full in
`evidence/T033-T034-red.md` "REVISED root cause")**: an earlier version of
this module concluded the reproducing stimulus (a default-output change to a
non-48kHz-rate device) was **unconditionally unfixable** by any client-side
rebuild, and shipped that scenario as `xfail(strict=True)`. That conclusion
was **falsified** by the independent Critic, who reproduced the same
fresh-start silence and then fixed it with a ~10-line change: the
aggregate device `sck_start_capture` creates defaults to a fixed nominal
rate independent of the system's actual current output rate; when they
differ, `engine.inputNode`'s format mismatches the aggregate's real IO
format and zero callbacks are ever delivered -- even though the tap and
aggregate device both report `IsAlive`/`IsRunning` = 1. Aligning the
aggregate's own `kAudioDevicePropertyNominalSampleRate` to the system
default output's current rate immediately after creating it (T035, this
round) fixes it. This module's mismatched-rate tests now assert the fixed
behaviour directly (no `xfail`) and are parametrized across several
standard rates, not just the two devices the original investigation
happened to try.

**Why this drives `sck_start_capture`, not `mic_start_capture` or
`audio_output_start`**: the reported bug is "listen mode" -- capturing the
*system's* audio (a meeting app's own output) via the global Core Audio
process tap `sck_start_capture` builds, exactly the capture path research.md
D6 targets. US3's mic-selection defect and US1's output-ordering defect are
different code paths, already closed by their own stories.

**Two independent defects, both covered here**: (1) the rate-mismatch
silence above -- a *fresh* `sck_start_capture` call is silent from the
first callback if the system default output is already at a non-48kHz
rate, no device change required at all (Critic's actual repro: this is
what most directly explains a report from a user whose default output was
already `44.1kHz`, not a mid-session AirPods switch); and (2) the same
condition arriving *mid-session*, via a real default-output change while
capture is already running, which is what T033 originally set out to test.
Both are covered below, parametrized across four standard rates
(32000/44100/88200/96000 Hz).

**How this reaches the real code path**: `AudioCapture.swift` is compiled
here as a standalone dylib (`swiftc -emit-library`, mirroring every other
hardware-backed module in this package -- see `test_tap_self_exclusion.py`'s
module docstring) and `sck_start_capture` / `sck_stop_capture` are called
directly via `ctypes`, in-process. This is safe specifically for `sck_*`
(and not `mic_start_capture`/`audio_output_start`, which the sibling US1
module's docstring documents as contention-prone against a same-process
PortAudio stream on the *same* device): `sck_start_capture` self-excludes
this process's own audio from the tap (T030/FR-006), and the tone this test
listens for is always played from a **different OS process**
(`_other_process_tone.py`), so there is no in-process PortAudio-vs-AVAudioEngine
contention on the same device to reproduce that earlier false-failure
signature.

**Never BlackHole as the mismatch candidate (Critic major 2)**: every tone
this module plays is routed through `BlackHole 2ch` at its native 48000 Hz.
The device whose *own* nominal rate is pushed to a mismatched value is
always a *different* output-capable device, and only to one of a small
whitelist of standard rates (`_CANDIDATE_MISMATCH_RATES`, all <= 96000 Hz)
-- this machine's shared virtual loopback device is never driven to an
extreme value (the earlier version of this fixture could, in principle,
have picked `384000`/`768000` Hz off `BlackHole`'s own available-rates
list; that architecture is gone, replaced by a fixed, reviewed whitelist).

**Simulating the device change**: entirely programmatic
(`AudioObjectSetPropertyData` on `kAudioHardwarePropertyDefaultOutputDevice`
and, for the mismatch tests, on the candidate's own
`kAudioDevicePropertyNominalSampleRate`) -- not a real unplug/replug, which
cannot run unattended or in CI (research.md, task brief). Every fixture in
this module restores whatever it changed in `finally` regardless of test
outcome (standing constraint: this is the user's real machine), and
independently re-checks the restoration via readback afterward as a second
guard.

**CORRECTION (T037 Critic round 2)**: the watchdog described immediately
above -- armed unconditionally on *every* successful build, including the
very first one -- was itself a false-positive generator: Critic round 2
reproduced it killing a perfectly healthy, silent "listen mode started
before anyone has spoken" session at 2.3s (`test_rebuild_that_produces_no_audio_is_reported_as_an_error`,
which *asserted that exact outcome as correct*, is removed this round --
its assumption was wrong, not merely its implementation). The confirmed
round-2 design, implemented here:

- **No watchdog on the initial build at all.** Fresh-start silence is
  normal; there is nothing to arm against.
- **The rebuild-path watchdog only arms if audio was genuinely flowing in
  the few seconds right before the rebuild** (`sckWasRecentlyFlowing`,
  fed by a ring buffer of recent tap-callback timestamps) -- verified by
  `test_watchdog_warns_without_stopping_capture_when_audio_was_flowing_before_rebuild`
  (armed) and `test_watchdog_does_not_warn_when_audio_was_already_silent_before_rebuild`
  (not armed, the direct counterpart to the false positive Critic found).
- **Armed and tripped, it now warns rather than errors**: the message is
  `WARN:`-prefixed and capture is left running exactly as-is -- Rust's
  `wait_for_stop_or_rebuild_error` (`client/src-tauri/src/audio_capture.rs`)
  forwards a `WARN:` message without ending the session, unlike every other
  message. `test_watchdog_warns_...` also proves capture keeps working
  after the warning (resumes real audio afterward), not merely that it
  wasn't torn down.
- **A separate finding (Critic round 2 major)**: the *previous* version of
  this now-removed test never actually exercised the rebuild mechanism --
  it passed purely because round 1's unconditional watchdog fired
  regardless of whether any listener ran at all. `test_rebuild_is_triggered_by_a_real_device_change`
  replaces that intent directly, asserting the rebuild's own real side
  effects (its NSLog transcript, captured via `capfd` since `AudioCapture.swift`'s
  `NSLog` calls write to this same process's stderr file descriptor --
  `capsys` cannot see them, only `capfd` can), and is mutation-tested by
  removing the listener registration call.

**Nominal-rate poller (Critic round 2 major)**: a default-output change
that only alters the *already-default* device's own rate -- measured to be
exactly the shape of an AirPods A2DP<->HFP profile switch -- fires neither
`kAudioHardwarePropertyDefaultOutputDevice` (the device identity never
changes) nor, on this machine, `AVAudioEngineConfigurationChange`. Neither
listener has anything to react to. `AudioCapture.swift` now polls: every
2s while capturing, compare the aggregate device's own nominal rate
against the system default output's current rate, read fresh each tick,
and rebuild on a mismatch through the same debounced path the listeners
use. `test_nominal_rate_poll_detects_a_rate_only_change_and_realigns`
verifies this deterministically with a single real device (no second
device, no listener trigger needed at all): change the current default's
own rate, wait past the poll interval, and confirm -- via a direct device
query, since the aggregate device this capture session creates is visible
to this same process even though it is marked "private" (verified
separately) -- that it has been realigned. **Limitation, disclosed**: this
machine cannot reproduce a genuine HFP-range rate (16/24kHz) to test
against; the poller applies whatever rate it reads with no hardcoded
whitelist, so it is expected to generalize, but that generalization is
unverified here.
"""

from __future__ import annotations

import ctypes
import ctypes.util
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
from integration.test_capture_fixture import BLACKHOLE, MATCH_SCORE_MIN, _matched_filter_score, _rms

_SWIFT_SRC = (
    Path(__file__).resolve().parents[3] / "client" / "src-tauri" / "swift" / "AudioCapture.swift"
)
_OTHER_PROCESS_TONE_SCRIPT = Path(__file__).resolve().parent / "_other_process_tone.py"

# sck_start_capture's fixed conversion target (AudioCapture.swift
# `targetSampleRate: Double = 16000`), independent of whatever rate the tone
# below is synthesized/played at.
TAP_CAPTURE_SAMPLE_RATE = 16_000

OTHER_PROCESS_SAMPLE_RATE = 48_000
TONE_FREQUENCY_HZ = 1500.0

# Total length of the continuous tone played from the other process for the
# long-running same-rate regression test. Must comfortably outlast settle +
# pre-change + post-change + tail below, plus the other process's own
# Python-interpreter-plus-PortAudio-stream-open startup lag (~0.5-1s,
# mirroring test_tap_self_exclusion.py's own margin).
TONE_DURATION_S = 15.0

# Settle before anything is asked to start listening, mirroring
# test_tap_self_exclusion.py's TAP_SETTLE_S.
TAP_SETTLE_S = 1.0

# How long to let the tone play before flipping -- long enough that the
# "before" window has solid tone content once the other process's own
# startup lag is excluded from scoring.
PRE_CHANGE_S = 3.0

# How long to keep the tone playing, and this test watching, after the flip
# -- comfortably longer than MAX_ACCEPTABLE_GAP_S below, so a realistic
# (much shorter) rebuild gap still leaves several seconds of clean "after"
# content for the matched filter to score confidently.
POST_CHANGE_S = 8.0

# Extra time after the tone's nominal end to catch trailing buffered audio.
TAIL_MARGIN_S = 1.0

# A brief interruption during rebuild is accepted (T035/T036); this is a
# permanent-silence ceiling, not a tuned performance budget -- generous
# relative to the sub-second teardown+recreate (`AudioHardwareDestroyProcessTap`
# / `AudioHardwareDestroyAggregateDevice` / `AudioHardwareCreateProcessTap` /
# `AudioHardwareCreateAggregateDevice` / `engine.start()`) T035 performs.
MAX_ACCEPTABLE_GAP_S = 6.0

# Shorter timing for the parametrized mismatch tests below (four rates x
# two scenarios would otherwise take several minutes at TONE_DURATION_S).
MISMATCH_TAP_SETTLE_S = 1.0
MISMATCH_PRE_CHANGE_S = 2.0
MISMATCH_TONE_DURATION_S = 6.0
MISMATCH_TAIL_MARGIN_S = 1.0

# Standard sample rates to verify the aggregate-rate-alignment fix against
# (Critic blocker: "88.2/96k, tap rate != agg rate 일반 케이스까지"), capped at
# 96000 (Critic major 2: never push a shared device to an extreme rate).
_CANDIDATE_MISMATCH_RATES = (32000.0, 44100.0, 88200.0, 96000.0)

# How long to poll for the watchdog's error report before concluding it
# never fired -- generous relative to AudioCapture.swift's own
# `sckAudioWatchdogTimeout` (2s), never asserted against that exact
# constant (constraint 4: no configuration-readback duplication).
WATCHDOG_POLL_CEILING_S = 6.0
WATCHDOG_POLL_INTERVAL_S = 0.1

_AUDIO_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_ssize_t,
    ctypes.c_ssize_t,
    ctypes.c_ssize_t,
)
_ERROR_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p)

_CORE_AUDIO = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
_CORE_FOUNDATION = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
_kAudioHardwarePropertyDefaultOutputDevice = conftest._fourcc("dOut")
_kAudioDevicePropertyStreams = conftest._fourcc("stm#")
_kAudioDevicePropertyNominalSampleRate = conftest._fourcc("nsrt")
_kAudioDevicePropertyAvailableNominalSampleRates = conftest._fourcc("nsr#")


class _AudioValueRange(ctypes.Structure):
    _fields_ = [("mMinimum", ctypes.c_double), ("mMaximum", ctypes.c_double)]


@pytest.fixture(autouse=True)
def _exclusive_device_access(audio_device_lock):
    """Serialize every test in this module against other device users --
    same rule every other hardware-backed module in this suite applies.

    T037 Critic round 2: a 1s settle after acquiring the lock, found
    necessary in this round -- Core Audio's own notification delivery for
    a device-property change can lag measurably behind the property change
    itself confirming via readback (which every fixture in this file
    already waits for before releasing the lock). Without this settle, a
    notification queued by the *previous* test's own device flip has been
    observed arriving just as the *next* test registers its own listener,
    landing as a spurious extra rebuild inside that next test's own timing
    -- reproduced directly running this module's full suite back-to-back
    versus each test in isolation (every test here passes in isolation;
    only specific orderings in the full run showed this). Not a defect in
    the production rebuild/watchdog/poller logic itself.
    """
    with audio_device_lock():
        time.sleep(1.0)
        yield


@pytest.fixture(scope="module")
def audio_capture_dylib(tmp_path_factory) -> Path:
    """Compile `AudioCapture.swift` as a standalone dylib -- identical build
    recipe to every sibling hardware-backed module in this package."""
    if shutil.which("swiftc") is None:
        pytest.fail(
            "swiftc not found -- required to build AudioCapture.swift for this test; "
            "the same tool client/src-tauri/build.rs already requires to build the app"
        )
    if not _SWIFT_SRC.exists():
        pytest.fail(f"AudioCapture.swift not found at {_SWIFT_SRC}")

    out_dir = tmp_path_factory.mktemp("device_change_dylib")
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
    return lib


def _device_has_output_stream(core_audio, device_id: int) -> bool:
    address = conftest._AudioObjectPropertyAddress(
        _kAudioDevicePropertyStreams,
        conftest._SCOPE_FOURCC["output"],
        conftest._kAudioObjectPropertyElementMain,
    )
    size = ctypes.c_uint32(0)
    status = core_audio.AudioObjectGetPropertyDataSize(
        ctypes.c_uint32(device_id), ctypes.byref(address), ctypes.c_uint32(0), None, ctypes.byref(size)
    )
    return status == 0 and size.value > 0


def _device_id_for_name(name: str) -> int | None:
    for device_id in conftest._core_audio_device_ids(_CORE_AUDIO):
        if conftest._core_audio_device_name(_CORE_AUDIO, _CORE_FOUNDATION, device_id) == name:
            return device_id
    return None


def _default_output_device_id() -> int | None:
    address = conftest._AudioObjectPropertyAddress(
        _kAudioHardwarePropertyDefaultOutputDevice,
        conftest._kAudioObjectPropertyScopeGlobal,
        conftest._kAudioObjectPropertyElementMain,
    )
    device_id = ctypes.c_uint32(0)
    size = ctypes.c_uint32(ctypes.sizeof(device_id))
    status = _CORE_AUDIO.AudioObjectGetPropertyData(
        ctypes.c_uint32(conftest._kAudioObjectSystemObject),
        ctypes.byref(address),
        ctypes.c_uint32(0),
        None,
        ctypes.byref(size),
        ctypes.byref(device_id),
    )
    return device_id.value if status == 0 else None


def _set_default_output_device_id(device_id: int) -> int:
    address = conftest._AudioObjectPropertyAddress(
        _kAudioHardwarePropertyDefaultOutputDevice,
        conftest._kAudioObjectPropertyScopeGlobal,
        conftest._kAudioObjectPropertyElementMain,
    )
    value = ctypes.c_uint32(device_id)
    return _CORE_AUDIO.AudioObjectSetPropertyData(
        ctypes.c_uint32(conftest._kAudioObjectSystemObject),
        ctypes.byref(address),
        ctypes.c_uint32(0),
        None,
        ctypes.c_uint32(ctypes.sizeof(value)),
        ctypes.byref(value),
    )


def _read_nominal_sample_rate(device_id: int) -> float | None:
    value = ctypes.c_double(0)
    status = conftest._core_audio_get_property(
        _CORE_AUDIO,
        device_id,
        _kAudioDevicePropertyNominalSampleRate,
        conftest._kAudioObjectPropertyScopeGlobal,
        conftest._kAudioObjectPropertyElementMain,
        value,
        ctypes.sizeof(value),
    )
    return value.value if status == 0 else None


def _set_nominal_sample_rate(device_id: int, rate: float) -> int:
    address = conftest._AudioObjectPropertyAddress(
        _kAudioDevicePropertyNominalSampleRate,
        conftest._kAudioObjectPropertyScopeGlobal,
        conftest._kAudioObjectPropertyElementMain,
    )
    value = ctypes.c_double(rate)
    return _CORE_AUDIO.AudioObjectSetPropertyData(
        ctypes.c_uint32(device_id),
        ctypes.byref(address),
        ctypes.c_uint32(0),
        None,
        ctypes.c_uint32(ctypes.sizeof(value)),
        ctypes.byref(value),
    )


def _available_nominal_sample_rates(device_id: int) -> list[float]:
    address = conftest._AudioObjectPropertyAddress(
        _kAudioDevicePropertyAvailableNominalSampleRates,
        conftest._kAudioObjectPropertyScopeGlobal,
        conftest._kAudioObjectPropertyElementMain,
    )
    size = ctypes.c_uint32(0)
    status = _CORE_AUDIO.AudioObjectGetPropertyDataSize(
        ctypes.c_uint32(device_id), ctypes.byref(address), ctypes.c_uint32(0), None, ctypes.byref(size)
    )
    if status != 0 or size.value == 0:
        return []
    count = size.value // ctypes.sizeof(_AudioValueRange)
    ranges = (_AudioValueRange * count)()
    status = _CORE_AUDIO.AudioObjectGetPropertyData(
        ctypes.c_uint32(device_id),
        ctypes.byref(address),
        ctypes.c_uint32(0),
        None,
        ctypes.byref(size),
        ctypes.byref(ranges),
    )
    if status != 0:
        return []
    rates = {r.mMinimum for r in ranges} | {r.mMaximum for r in ranges}
    return sorted(rates)


@pytest.fixture
def output_device_pair():
    """Resolves the current system default output device and one other
    output-capable, **same-nominal-rate** device to flip to and back.
    Skips (never fails) if fewer than two same-rate output-capable devices
    are visible -- ambient hardware state (constraint 1), not something
    under test. Restoration is the *test's* job (it needs the flip to
    happen at a specific moment mid-capture), but this fixture
    independently re-checks it afterward via readback as a second guard,
    since this is the user's real machine.
    """
    original_id = _default_output_device_id()
    if original_id is None:
        pytest.skip("could not resolve the system default output device")
    original_rate = _read_nominal_sample_rate(original_id)
    if original_rate is None:
        pytest.skip("could not read the system default output's current nominal sample rate")

    alternate_id = None
    for device_id in conftest._core_audio_device_ids(_CORE_AUDIO):
        if device_id == original_id or not _device_has_output_stream(_CORE_AUDIO, device_id):
            continue
        if _read_nominal_sample_rate(device_id) == original_rate:
            alternate_id = device_id
            break
    if alternate_id is None:
        pytest.skip(
            f"no other output-capable device currently at the same nominal rate as the "
            f"system default ({original_rate} Hz) on this machine"
        )

    yield original_id, alternate_id

    restored = _default_output_device_id()
    assert restored == original_id, (
        f"expected the system default output device to be restored to {original_id} after "
        f"the test, found {restored} -- machine audio state may be left altered"
    )


class _MismatchTarget:
    def __init__(self, original_id: int, original_rate: float, candidate_id: int, candidate_rate: float, candidate_original_rate: float):
        self.original_id = original_id
        self.original_rate = original_rate
        self.candidate_id = candidate_id
        self.candidate_rate = candidate_rate
        self.candidate_original_rate = candidate_original_rate


@pytest.fixture
def mismatch_target(request):
    """Resolves the system's current default output device and a
    *different*, non-`BlackHole` output-capable device that offers
    `request.param` Hz among its available nominal rates (Critic major 2:
    `BlackHole 2ch` -- this module's own shared tone-source device -- is
    never a mismatch candidate, and only a fixed, reviewed whitelist of
    standard rates is ever used). Skips (never fails) if no such device
    exists, or if the target rate already matches the current default's
    rate (nothing to test) -- ambient hardware state, not a defect.
    Applying the change and restoring it is each *test*'s job (the two
    tests using this fixture need the change to happen at different
    points); this fixture independently re-checks both the default device
    and the candidate's own rate are restored afterward, as a second
    guard.
    """
    target_rate = request.param
    original_id = _default_output_device_id()
    if original_id is None:
        pytest.skip("could not resolve the system default output device")
    original_rate = _read_nominal_sample_rate(original_id)
    if original_rate is None:
        pytest.skip("could not read the system default output's current nominal sample rate")

    blackhole_id = _device_id_for_name(BLACKHOLE)
    candidate_id = None
    for device_id in conftest._core_audio_device_ids(_CORE_AUDIO):
        if device_id == original_id or device_id == blackhole_id:
            continue
        if not _device_has_output_stream(_CORE_AUDIO, device_id):
            continue
        if target_rate in _available_nominal_sample_rates(device_id):
            candidate_id = device_id
            break
    if candidate_id is None:
        pytest.skip(
            f"no output-capable device (other than {BLACKHOLE!r}) offering {target_rate} Hz "
            f"on this machine"
        )
    if target_rate == original_rate:
        pytest.skip(
            f"target rate {target_rate} Hz already matches the system default's current "
            f"rate -- nothing to test"
        )

    candidate_original_rate = _read_nominal_sample_rate(candidate_id)
    if candidate_original_rate is None:
        pytest.skip("could not read the candidate device's own current nominal sample rate")

    yield _MismatchTarget(original_id, original_rate, candidate_id, target_rate, candidate_original_rate)

    restored_default = _default_output_device_id()
    restored_rate = _read_nominal_sample_rate(candidate_id)
    assert restored_default == original_id, (
        f"expected the system default output device to be restored to {original_id} after "
        f"the test, found {restored_default} -- machine audio state may be left altered"
    )
    assert restored_rate == candidate_original_rate, (
        f"expected the candidate device's nominal rate to be restored to "
        f"{candidate_original_rate} after the test, found {restored_rate} -- machine audio "
        f"state may be left altered"
    )


@pytest.fixture
def own_rate_mismatch(request):
    """Resolves the system's *current* default output device and one
    candidate rate from `_CANDIDATE_MISMATCH_RATES` that (a) the device
    itself already lists as an available nominal rate and (b) differs from
    its current rate -- no second device involved at all. This changes the
    current default's *own* rate rather than switching which device is
    default, mirroring the most literal reading of the real-world report
    the fresh-start test targets: a single output device natively running
    at (say) 44.1kHz, no device swap in sight. Deliberately independent of
    `mismatch_target` (which needs a second, non-`BlackHole` device and can
    legitimately skip when only one is present, e.g. an external USB audio
    interface that has gone to sleep) -- the fresh-start defect this
    fixture targets does not require a second device to reproduce or to
    verify the fix for.
    """
    target_rate = request.param
    original_id = _default_output_device_id()
    if original_id is None:
        pytest.skip("could not resolve the system default output device")
    # T037 Critic round 2 minor: symmetric with `mismatch_target` -- never
    # push `BlackHole 2ch` (this suite's own shared tone-source device) to
    # a mismatched rate, even via "the current default happens to already
    # be BlackHole" rather than an explicit candidate pick.
    if original_id == _device_id_for_name(BLACKHOLE):
        pytest.skip(f"the system default output is currently {BLACKHOLE!r} -- never used as a mismatch target")
    original_rate = _read_nominal_sample_rate(original_id)
    if original_rate is None:
        pytest.skip("could not read the system default output's current nominal sample rate")
    if target_rate == original_rate:
        pytest.skip(f"target rate {target_rate} Hz already matches the current default's rate")
    if target_rate not in _available_nominal_sample_rates(original_id):
        pytest.skip(
            f"the system default output device does not list {target_rate} Hz among its "
            f"available nominal rates"
        )

    yield original_id, original_rate, target_rate

    restored_rate = _read_nominal_sample_rate(original_id)
    assert restored_rate == original_rate, (
        f"expected the system default output's nominal rate to be restored to "
        f"{original_rate} Hz after the test, found {restored_rate} -- machine audio state "
        f"may be left altered"
    )


def _synthesize_tone(frequency_hz: float, duration_s: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    return (0.8 * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)


def _tone_at_tap_capture_rate(tone: np.ndarray, tone_sample_rate: int) -> np.ndarray:
    new_len = int(len(tone) * TAP_CAPTURE_SAMPLE_RATE / tone_sample_rate)
    return resample(tone.astype(np.float64), new_len)


def _play_tone_subprocess(duration_s: float) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            str(_OTHER_PROCESS_TONE_SCRIPT),
            BLACKHOLE,
            str(TONE_FREQUENCY_HZ),
            str(duration_s),
            str(OTHER_PROCESS_SAMPLE_RATE),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_capture_keeps_producing_audio_across_a_default_output_device_change(
    audio_lib, output_device_pair
):
    """SC-004 / FR-008: a real default-output-device change, mid-capture,
    must not permanently silence `sck_start_capture`'s tap, and the change
    must genuinely exercise T035's listener/rebuild path without corrupting
    it (constraint 5: a test that only checked the "before" window, or that
    never actually drove the real code, would prove nothing). Uses a
    same-rate stimulus -- see `test_default_output_change_to_a_mismatched_rate_recovers_after_rebuild`
    below for the rate-mismatch stimulus that was this story's original,
    load-bearing repro.
    """
    original_id, alternate_id = output_device_pair
    tone = _synthesize_tone(TONE_FREQUENCY_HZ, TONE_DURATION_S, OTHER_PROCESS_SAMPLE_RATE)
    reference = _tone_at_tap_capture_rate(tone, OTHER_PROCESS_SAMPLE_RATE)

    events: list[tuple[float, np.ndarray]] = []
    errors: list[str] = []
    lock = threading.Lock()

    def _on_audio(data_ptr, num_samples, _sample_rate, _channels) -> None:
        if num_samples <= 0 or not data_ptr:
            return
        buf = np.ctypeslib.as_array(data_ptr, shape=(num_samples,)).copy()
        ts = time.monotonic()
        with lock:
            events.append((ts, buf))

    def _on_error(message: bytes | None) -> None:
        with lock:
            errors.append(message.decode("utf-8") if message else "")

    c_audio_cb = _AUDIO_CALLBACK(_on_audio)
    c_error_cb = _ERROR_CALLBACK(_on_error)

    ok = audio_lib.sck_start_capture(c_audio_cb, c_error_cb)
    assert ok, "sck_start_capture returned false"

    proc: subprocess.Popen | None = None
    change_ts: float | None = None
    try:
        time.sleep(TAP_SETTLE_S)

        proc = _play_tone_subprocess(TONE_DURATION_S)

        time.sleep(PRE_CHANGE_S)

        change_status = _set_default_output_device_id(alternate_id)
        change_ts = time.monotonic()
        assert change_status == 0, f"failed to set default output device: status={change_status}"

        time.sleep(POST_CHANGE_S)

        stdout, stderr = proc.communicate(timeout=TONE_DURATION_S + 15.0)
        proc = None
        time.sleep(TAIL_MARGIN_S)
    finally:
        restore_status = _set_default_output_device_id(original_id)
        audio_lib.sck_stop_capture()
        if proc is not None:
            proc.kill()

    assert restore_status == 0, (
        f"failed to restore the original default output device: status={restore_status}"
    )
    assert change_ts is not None, "the device flip never ran"

    stdout_stripped = stdout.strip()
    if stdout_stripped.startswith("ABSENT "):
        pytest.skip(stdout_stripped[len("ABSENT ") :])
    assert proc is None and stdout_stripped == "OK", (
        f"other-process tone player failed: stdout={stdout!r} stderr={stderr!r}"
    )

    assert not errors, (
        f"expected no rebuild-failure reports during a device flip expected to succeed, "
        f"got: {errors}"
    )

    with lock:
        all_events = list(events)

    before = [(ts, buf) for ts, buf in all_events if ts < change_ts]
    after = [(ts, buf) for ts, buf in all_events if ts > change_ts]

    assert before, "expected some audio captured before the device change"
    assert after, (
        "expected capture to keep producing audio after the device change (SC-004) -- got "
        "nothing: capture went silent and never recovered"
    )

    before_samples = np.concatenate([b for _, b in before]).astype(np.float64)
    after_samples = np.concatenate([b for _, b in after]).astype(np.float64)

    before_rms = _rms(before_samples)
    assert before_rms > 0.0, f"expected non-silent audio before the change, got rms={before_rms:.6g}"
    before_score = abs(_matched_filter_score(reference, before_samples))
    assert before_score > MATCH_SCORE_MIN, (
        f"expected the pre-change capture to match the tone (score > {MATCH_SCORE_MIN}), "
        f"got {before_score:.4f} -- if this fails, the harness itself is broken, not the "
        f"device-change recovery this test targets"
    )

    after_rms = _rms(after_samples)
    assert after_rms > 0.0, (
        f"expected non-silent audio after the change, got rms={after_rms:.6g} -- capture "
        f"produced samples but they are pure silence"
    )
    after_score = abs(_matched_filter_score(reference, after_samples))
    assert after_score > MATCH_SCORE_MIN, (
        f"expected the post-change capture to still match the tone (score > "
        f"{MATCH_SCORE_MIN}), got {after_score:.4f} -- capture recovered *something* but not "
        f"the actual tone still playing"
    )

    gap_s = after[0][0] - before[-1][0]
    print(f"\nMEASURED_AUDIO_GAP_S={gap_s:.3f}")
    assert gap_s < MAX_ACCEPTABLE_GAP_S, (
        f"audio gap spanning the device-change rebuild was {gap_s:.3f}s, exceeding the "
        f"{MAX_ACCEPTABLE_GAP_S}s ceiling for 'brief gap, not permanent silence' (SC-004)"
    )


@pytest.mark.parametrize("mismatch_target", _CANDIDATE_MISMATCH_RATES, indirect=True)
def test_default_output_change_to_a_mismatched_rate_recovers_after_rebuild(
    audio_lib, mismatch_target
):
    """T037 Critic blocker fix, verified: a default-output change *mid-session*
    to a device whose own nominal rate differs from the previous default's
    -- this story's original, load-bearing repro -- must recover, not stay
    silent. Both "before" and "after" windows must independently match the
    tone (constraint 5: matching only "before" would also pass against the
    unfixed graph-construction-order code, since that's what it already did
    before this round's blocker fix; both must be checked to prove recovery,
    not mere survival of the *first* graph).
    """
    mc = mismatch_target
    tone = _synthesize_tone(TONE_FREQUENCY_HZ, MISMATCH_TONE_DURATION_S, OTHER_PROCESS_SAMPLE_RATE)
    reference = _tone_at_tap_capture_rate(tone, OTHER_PROCESS_SAMPLE_RATE)

    events: list[tuple[float, np.ndarray]] = []
    errors: list[str] = []
    lock = threading.Lock()

    def _on_audio(data_ptr, num_samples, _sample_rate, _channels) -> None:
        if num_samples <= 0 or not data_ptr:
            return
        buf = np.ctypeslib.as_array(data_ptr, shape=(num_samples,)).copy()
        ts = time.monotonic()
        with lock:
            events.append((ts, buf))

    def _on_error(message: bytes | None) -> None:
        with lock:
            errors.append(message.decode("utf-8") if message else "")

    c_audio_cb = _AUDIO_CALLBACK(_on_audio)
    c_error_cb = _ERROR_CALLBACK(_on_error)

    ok = audio_lib.sck_start_capture(c_audio_cb, c_error_cb)
    assert ok, "sck_start_capture returned false"

    proc: subprocess.Popen | None = None
    change_ts: float | None = None
    try:
        time.sleep(MISMATCH_TAP_SETTLE_S)

        proc = _play_tone_subprocess(MISMATCH_TONE_DURATION_S)

        time.sleep(MISMATCH_PRE_CHANGE_S)

        rate_status = _set_nominal_sample_rate(mc.candidate_id, mc.candidate_rate)
        assert rate_status == 0, f"failed to set candidate device's nominal rate: status={rate_status}"
        time.sleep(0.3)
        default_status = _set_default_output_device_id(mc.candidate_id)
        change_ts = time.monotonic()
        assert default_status == 0, f"failed to set default output device: status={default_status}"

        stdout, stderr = proc.communicate(timeout=MISMATCH_TONE_DURATION_S + 15.0)
        proc = None
        time.sleep(MISMATCH_TAIL_MARGIN_S)
    finally:
        restore_default_status = _set_default_output_device_id(mc.original_id)
        time.sleep(0.2)
        restore_rate_status = _set_nominal_sample_rate(mc.candidate_id, mc.candidate_original_rate)
        # A nominal-rate change is not instantaneous at the HAL level --
        # the fixture's own teardown reads this back immediately after this
        # function returns, so this settle window must happen *before*
        # returning, not just before the next `AudioObjectSetPropertyData`
        # call (mirrors every other rate-restore in this suite, e.g.
        # test_output_reaches_device.py's `mismatched_blackhole_rate`).
        time.sleep(0.2)
        audio_lib.sck_stop_capture()
        if proc is not None:
            proc.kill()

    assert restore_default_status == 0, f"failed to restore default output: status={restore_default_status}"
    assert restore_rate_status == 0, f"failed to restore candidate's rate: status={restore_rate_status}"
    assert change_ts is not None, "the device flip never ran"

    stdout_stripped = stdout.strip()
    if stdout_stripped.startswith("ABSENT "):
        pytest.skip(stdout_stripped[len("ABSENT ") :])
    assert proc is None and stdout_stripped == "OK", (
        f"other-process tone player failed: stdout={stdout!r} stderr={stderr!r}"
    )
    assert not errors, f"expected no rebuild-failure reports during an expected-to-succeed change, got: {errors}"

    with lock:
        all_events = list(events)

    before = [(ts, buf) for ts, buf in all_events if ts < change_ts]
    after = [(ts, buf) for ts, buf in all_events if ts > change_ts]

    assert before, "expected some audio captured before the device change"
    assert after, (
        f"expected capture to recover and keep producing audio after switching default output "
        f"to {mc.candidate_rate} Hz -- got nothing: capture went silent and never recovered"
    )

    before_samples = np.concatenate([b for _, b in before]).astype(np.float64)
    after_samples = np.concatenate([b for _, b in after]).astype(np.float64)

    before_score = abs(_matched_filter_score(reference, before_samples))
    assert before_score > MATCH_SCORE_MIN, (
        f"expected the pre-change capture to match the tone (score > {MATCH_SCORE_MIN}), "
        f"got {before_score:.4f}"
    )
    after_score = abs(_matched_filter_score(reference, after_samples))
    assert after_score > MATCH_SCORE_MIN, (
        f"expected the post-change capture (default output now at {mc.candidate_rate} Hz) to "
        f"match the tone (score > {MATCH_SCORE_MIN}), got {after_score:.4f}"
    )

    gap_s = after[0][0] - before[-1][0]
    print(f"\nMEASURED_AUDIO_GAP_S(rate={mc.candidate_rate})={gap_s:.3f}")
    assert gap_s < MAX_ACCEPTABLE_GAP_S, (
        f"audio gap spanning the device-change rebuild was {gap_s:.3f}s, exceeding the "
        f"{MAX_ACCEPTABLE_GAP_S}s ceiling"
    )


@pytest.mark.parametrize("own_rate_mismatch", _CANDIDATE_MISMATCH_RATES, indirect=True)
def test_fresh_start_at_a_mismatched_default_output_rate_produces_audio(
    audio_lib, own_rate_mismatch
):
    """T037 Critic blocker: covers the *fresh-start* half explicitly called
    out as untested -- a user whose system default output is already
    running at a non-48kHz rate (no device change involved at all) got a
    permanently silent capture from the very first `sck_start_capture`
    call. This is plausibly the more common real-world case (e.g. any
    default-output device natively running at 44.1kHz) than a mid-session
    switch, and needs only the *current* default device (no second device
    required -- see `own_rate_mismatch`'s docstring for why this is
    deliberately independent of `mismatch_target`).
    """
    original_id, original_rate, target_rate = own_rate_mismatch
    tone = _synthesize_tone(TONE_FREQUENCY_HZ, MISMATCH_TONE_DURATION_S, OTHER_PROCESS_SAMPLE_RATE)
    reference = _tone_at_tap_capture_rate(tone, OTHER_PROCESS_SAMPLE_RATE)

    rate_status = _set_nominal_sample_rate(original_id, target_rate)
    assert rate_status == 0, f"failed to set the default output's nominal rate: status={rate_status}"
    time.sleep(0.3)

    events: list[tuple[float, np.ndarray]] = []
    errors: list[str] = []
    lock = threading.Lock()

    def _on_audio(data_ptr, num_samples, _sample_rate, _channels) -> None:
        if num_samples <= 0 or not data_ptr:
            return
        buf = np.ctypeslib.as_array(data_ptr, shape=(num_samples,)).copy()
        with lock:
            events.append((time.monotonic(), buf))

    def _on_error(message: bytes | None) -> None:
        with lock:
            errors.append(message.decode("utf-8") if message else "")

    c_audio_cb = _AUDIO_CALLBACK(_on_audio)
    c_error_cb = _ERROR_CALLBACK(_on_error)

    proc: subprocess.Popen | None = None
    try:
        ok = audio_lib.sck_start_capture(c_audio_cb, c_error_cb)
        assert ok, "sck_start_capture returned false (default output already at a mismatched rate)"
        try:
            time.sleep(MISMATCH_TAP_SETTLE_S)
            proc = _play_tone_subprocess(MISMATCH_TONE_DURATION_S)
            stdout, stderr = proc.communicate(timeout=MISMATCH_TONE_DURATION_S + 15.0)
            proc = None
            time.sleep(MISMATCH_TAIL_MARGIN_S)
        finally:
            audio_lib.sck_stop_capture()
    finally:
        restore_rate_status = _set_nominal_sample_rate(original_id, original_rate)
        # See the analogous comment in
        # test_default_output_change_to_a_mismatched_rate_recovers_after_rebuild --
        # the fixture's teardown reads this back immediately after this
        # function returns, so the settle window belongs here.
        time.sleep(0.2)
        if proc is not None:
            proc.kill()

    assert restore_rate_status == 0, f"failed to restore the default output's rate: status={restore_rate_status}"

    stdout_stripped = stdout.strip()
    if stdout_stripped.startswith("ABSENT "):
        pytest.skip(stdout_stripped[len("ABSENT ") :])
    assert stdout_stripped == "OK", f"other-process tone player failed: stdout={stdout!r} stderr={stderr!r}"
    assert not errors, f"expected no error reports for a healthy fresh start, got: {errors}"

    with lock:
        all_events = list(events)

    assert all_events, (
        f"expected sck_start_capture to produce audio from a default output already at "
        f"{target_rate} Hz (fresh start, no device change) -- got nothing"
    )
    samples = np.concatenate([b for _, b in all_events]).astype(np.float64)
    rms = _rms(samples)
    assert rms > 0.0, f"expected non-silent audio, got rms={rms:.6g}"
    score = abs(_matched_filter_score(reference, samples))
    assert score > MATCH_SCORE_MIN, (
        f"expected the fresh-start capture to match the tone (score > {MATCH_SCORE_MIN}), "
        f"got {score:.4f}"
    )


def test_fresh_start_with_nothing_playing_produces_no_error(audio_lib):
    """T037 Critic round 2 blocker: a fresh `sck_start_capture` call with
    nothing playing at all -- the normal shape of listen mode started
    before anyone has spoken in a meeting -- must never be reported as an
    error, ever. Round 1's watchdog armed unconditionally on every build,
    including this one, and the Critic measured it killing exactly this
    healthy, silent session at 2.3s. Waits comfortably past every
    timeout this file's watchdog/poller machinery could plausibly fire
    within (the 4s rebuild-watchdog window and the 2s rate-poll interval),
    with no rebuild ever triggered at all -- the initial build is the only
    thing under test here.
    """
    errors: list[str] = []
    lock = threading.Lock()

    def _on_audio(_data_ptr, _num_samples, _sample_rate, _channels) -> None:
        pass  # deliberately nothing played -- see docstring

    def _on_error(message: bytes | None) -> None:
        with lock:
            errors.append(message.decode("utf-8") if message else "")

    c_audio_cb = _AUDIO_CALLBACK(_on_audio)
    c_error_cb = _ERROR_CALLBACK(_on_error)

    ok = audio_lib.sck_start_capture(c_audio_cb, c_error_cb)
    assert ok, "sck_start_capture returned false"
    try:
        time.sleep(WATCHDOG_POLL_CEILING_S)
    finally:
        audio_lib.sck_stop_capture()

    with lock:
        collected_errors = list(errors)
    assert not collected_errors, (
        f"expected a silent fresh start to produce no error or warning report at all "
        f"within {WATCHDOG_POLL_CEILING_S}s, got: {collected_errors}"
    )


def test_watchdog_does_not_warn_when_audio_was_already_silent_before_rebuild(
    audio_lib, output_device_pair
):
    """T037 Critic round 2 blocker, `sckWasRecentlyFlowing`'s "not armed"
    branch, the direct counterpart of the false positive the Critic found:
    a rebuild triggered while the capture path was *already* silent
    (nothing playing, not a lull in something that was) must not warn
    either -- the scenario closest to the real report (quiet room, listen
    mode already running when a device change happens). Uses the same
    already-proven-harmless same-rate trigger as
    `test_capture_keeps_producing_audio_across_a_default_output_device_change`,
    with nothing ever played, comfortably past
    `sckPreRebuildFlowingWindow` (3s) before the flip and past
    `sckRebuildAudioWatchdogTimeout` (4s) after it.
    """
    original_id, alternate_id = output_device_pair

    errors: list[str] = []
    lock = threading.Lock()

    def _on_audio(_data_ptr, _num_samples, _sample_rate, _channels) -> None:
        pass  # deliberately nothing played -- see docstring

    def _on_error(message: bytes | None) -> None:
        with lock:
            errors.append(message.decode("utf-8") if message else "")

    c_audio_cb = _AUDIO_CALLBACK(_on_audio)
    c_error_cb = _ERROR_CALLBACK(_on_error)

    ok = audio_lib.sck_start_capture(c_audio_cb, c_error_cb)
    assert ok, "sck_start_capture returned false"

    try:
        time.sleep(TAP_SETTLE_S)
        time.sleep(4.0)  # comfortably past the 3s "recently flowing" window

        change_status = _set_default_output_device_id(alternate_id)
        assert change_status == 0, f"failed to set default output device: status={change_status}"

        time.sleep(WATCHDOG_POLL_CEILING_S)  # comfortably past the 4s rebuild-watchdog timeout
    finally:
        restore_status = _set_default_output_device_id(original_id)
        audio_lib.sck_stop_capture()

    assert restore_status == 0, f"failed to restore the original default output device: status={restore_status}"

    with lock:
        collected_errors = list(errors)
    assert not collected_errors, (
        f"expected no warning for a rebuild that interrupted nothing (capture was already "
        f"silent beforehand), got: {collected_errors}"
    )


def test_watchdog_warns_without_stopping_capture_when_audio_was_flowing_before_rebuild(
    audio_lib, output_device_pair
):
    """T037 Critic round 2 blocker, `sckWasRecentlyFlowing`'s "armed"
    branch: if audio genuinely *was* flowing right before a rebuild and
    then stops arriving, that is worth a warning -- but capture must stay
    up (round 1's watchdog treated this same shape of event as a hard
    error and stopped capture outright, which is what turned a benign
    moment into a killed session). Constructed deterministically: the tone
    is killed mid-flight, right as an already-proven-harmless same-rate
    flip triggers a rebuild -- the rebuild always succeeds, but nothing is
    left to capture afterward, so "was flowing, now isn't" is real, not
    manufactured. After the warning, a *second* tone is played into the
    same still-running session to prove it is still genuinely capturing,
    not merely left un-torn-down.
    """
    original_id, alternate_id = output_device_pair
    tone = _synthesize_tone(TONE_FREQUENCY_HZ, TONE_DURATION_S, OTHER_PROCESS_SAMPLE_RATE)
    reference = _tone_at_tap_capture_rate(tone, OTHER_PROCESS_SAMPLE_RATE)

    events: list[tuple[float, np.ndarray]] = []
    warnings: list[str] = []
    lock = threading.Lock()

    def _on_audio(data_ptr, num_samples, _sample_rate, _channels) -> None:
        if num_samples <= 0 or not data_ptr:
            return
        buf = np.ctypeslib.as_array(data_ptr, shape=(num_samples,)).copy()
        with lock:
            events.append((time.monotonic(), buf))

    def _on_error(message: bytes | None) -> None:
        with lock:
            warnings.append(message.decode("utf-8") if message else "")

    c_audio_cb = _AUDIO_CALLBACK(_on_audio)
    c_error_cb = _ERROR_CALLBACK(_on_error)

    ok = audio_lib.sck_start_capture(c_audio_cb, c_error_cb)
    assert ok, "sck_start_capture returned false"

    proc: subprocess.Popen | None = None
    try:
        time.sleep(TAP_SETTLE_S)

        proc = _play_tone_subprocess(TONE_DURATION_S)
        time.sleep(3.0)  # solidly inside the 3s "recently flowing" window when killed
        proc.kill()
        proc.wait(timeout=5.0)
        proc = None

        change_status = _set_default_output_device_id(alternate_id)
        assert change_status == 0, f"failed to set default output device: status={change_status}"

        deadline = time.monotonic() + WATCHDOG_POLL_CEILING_S
        while time.monotonic() < deadline:
            with lock:
                if warnings:
                    break
            time.sleep(WATCHDOG_POLL_INTERVAL_S)

        with lock:
            warned = bool(warnings)
        assert warned, (
            f"expected a WARN report within {WATCHDOG_POLL_CEILING_S}s after a rebuild "
            f"interrupted audio that was genuinely flowing, got none"
        )

        # Prove the session is still genuinely alive, not merely left
        # un-torn-down: a second tone, played now, must still be captured.
        with lock:
            events.clear()
        proc = _play_tone_subprocess(TONE_DURATION_S)
        stdout, stderr = proc.communicate(timeout=TONE_DURATION_S + 15.0)
        proc = None
        time.sleep(1.0)
    finally:
        restore_status = _set_default_output_device_id(original_id)
        audio_lib.sck_stop_capture()
        if proc is not None:
            proc.kill()

    assert restore_status == 0, f"failed to restore the original default output device: status={restore_status}"

    with lock:
        collected_warnings = list(warnings)
        collected_events = list(events)

    assert collected_warnings[0].startswith("WARN:"), (
        f"expected the report to be WARN-prefixed (non-fatal), got: {collected_warnings[0]!r}"
    )

    stdout_stripped = stdout.strip()
    if stdout_stripped.startswith("ABSENT "):
        pytest.skip(stdout_stripped[len("ABSENT ") :])
    assert proc is None and stdout_stripped == "OK", (
        f"other-process tone player failed: stdout={stdout!r} stderr={stderr!r}"
    )
    assert collected_events, (
        "expected the same session to capture a second tone after the warning -- an empty "
        "capture here means the session was actually dead, not merely warned"
    )
    samples = np.concatenate([b for _, b in collected_events]).astype(np.float64)
    score = abs(_matched_filter_score(reference, samples))
    assert score > MATCH_SCORE_MIN, (
        f"expected the post-warning capture to match the second tone (score > "
        f"{MATCH_SCORE_MIN}), got {score:.4f} -- capture produced *something* but not the "
        f"actual tone played into it"
    )


def test_rebuild_is_triggered_by_a_real_device_change(audio_lib, output_device_pair, capfd):
    """T037 Critic round 2 major: the previous version of this test
    (`test_rebuild_that_produces_no_audio_is_reported_as_an_error`, removed
    this round) never actually exercised the rebuild path -- it passed
    purely because round 1's unconditional start-path watchdog fired
    regardless of whether any listener ever ran. This asserts the
    rebuild's own real side effect directly: `AudioCapture.swift` logs
    "device change detected ... rebuilding capture graph" and "rebuilt
    successfully" via `NSLog`, which reaches this same process's stderr
    file descriptor since the dylib is loaded in-process -- captured here
    via pytest's `capfd` (OS file-descriptor level; `capsys` cannot see
    `NSLog`'s C-level writes, only `capfd` can).
    """
    original_id, alternate_id = output_device_pair
    proc: subprocess.Popen | None = None

    def _on_audio(_data_ptr, _num_samples, _sample_rate, _channels) -> None:
        pass

    def _on_error(message: bytes | None) -> None:
        pass

    c_audio_cb = _AUDIO_CALLBACK(_on_audio)
    c_error_cb = _ERROR_CALLBACK(_on_error)

    ok = audio_lib.sck_start_capture(c_audio_cb, c_error_cb)
    assert ok, "sck_start_capture returned false"

    try:
        time.sleep(TAP_SETTLE_S)
        proc = _play_tone_subprocess(8.0)
        time.sleep(2.0)

        capfd.readouterr()  # drain start-of-session NSLog noise

        change_status = _set_default_output_device_id(alternate_id)
        assert change_status == 0, f"failed to set default output device: status={change_status}"

        time.sleep(2.0)
        captured = capfd.readouterr()

        stdout, stderr = proc.communicate(timeout=15.0)
        proc = None
    finally:
        restore_status = _set_default_output_device_id(original_id)
        audio_lib.sck_stop_capture()
        if proc is not None:
            proc.kill()

    assert restore_status == 0, f"failed to restore the original default output device: status={restore_status}"

    combined = captured.out + captured.err
    assert "rebuilding capture graph" in combined, (
        f"expected the real default-output-device change to trigger a rebuild (NSLog "
        f"'rebuilding capture graph'), found no such line in captured output:\n{combined}"
    )
    assert "rebuilt successfully" in combined, (
        f"expected the triggered rebuild to complete successfully, found no 'rebuilt "
        f"successfully' line in captured output:\n{combined}"
    )


def _find_voxtrans_tap_aggregate_id() -> int | None:
    for device_id in conftest._core_audio_device_ids(_CORE_AUDIO):
        if conftest._core_audio_device_name(_CORE_AUDIO, _CORE_FOUNDATION, device_id) == "VoxTrans Tap":
            return device_id
    return None


def test_nominal_rate_poll_detects_a_rate_only_change_and_realigns(audio_lib):
    """T037 Critic round 2 major: a default-output change that only alters
    the already-default device's own rate (measured shape of an AirPods
    A2DP<->HFP profile switch) fires neither
    `kAudioHardwarePropertyDefaultOutputDevice` (identity unchanged) nor,
    on this machine, `AVAudioEngineConfigurationChange` -- the
    nominal-rate poller is the only coverage. Verified deterministically
    with a single real device, no second device or listener trigger
    needed: change the current default's own rate, wait past the poll
    interval, and confirm via a direct device query that the aggregate
    device this capture session creates ("VoxTrans Tap") has been
    realigned -- confirmed separately that a "private" aggregate device is
    still visible to its own creating process.
    """
    original_id = _default_output_device_id()
    if original_id is None:
        pytest.skip("could not resolve the system default output device")
    if original_id == _device_id_for_name(BLACKHOLE):
        pytest.skip(f"the system default output is currently {BLACKHOLE!r} -- never mismatched")
    original_rate = _read_nominal_sample_rate(original_id)
    if original_rate is None:
        pytest.skip("could not read the system default output's current nominal sample rate")

    target_rate = 44100.0 if original_rate != 44100.0 else 32000.0
    if target_rate not in _available_nominal_sample_rates(original_id):
        pytest.skip(f"the system default output does not offer {target_rate} Hz")

    def _on_audio(_data_ptr, _num_samples, _sample_rate, _channels) -> None:
        pass

    def _on_error(message: bytes | None) -> None:
        pass

    c_audio_cb = _AUDIO_CALLBACK(_on_audio)
    c_error_cb = _ERROR_CALLBACK(_on_error)

    ok = audio_lib.sck_start_capture(c_audio_cb, c_error_cb)
    assert ok, "sck_start_capture returned false"

    try:
        time.sleep(TAP_SETTLE_S)
        agg_id_before = _find_voxtrans_tap_aggregate_id()
        assert agg_id_before is not None, "expected to find the 'VoxTrans Tap' aggregate device"
        rate_before = _read_nominal_sample_rate(agg_id_before)
        assert rate_before == original_rate, (
            f"expected the freshly-built aggregate device to already be aligned to the "
            f"system default's rate ({original_rate} Hz), got {rate_before}"
        )

        rate_status = _set_nominal_sample_rate(original_id, target_rate)
        assert rate_status == 0, f"failed to set the default output's nominal rate: status={rate_status}"

        # Poll interval is 2s in production; comfortable margin for one
        # full cycle plus the rebuild's own debounce and settle.
        time.sleep(6.0)
        agg_id_after = _find_voxtrans_tap_aggregate_id()
        # Read the realigned rate *before* restoring anything or stopping
        # capture -- `sck_stop_capture()` destroys the aggregate device
        # entirely, so `agg_id_after` is no longer a live device once the
        # `finally` block below has run.
        assert agg_id_after is not None, "expected the aggregate device to still exist after the poll-triggered rebuild"
        realigned_rate = _read_nominal_sample_rate(agg_id_after)
    finally:
        restore_status = _set_nominal_sample_rate(original_id, original_rate)
        time.sleep(0.3)
        audio_lib.sck_stop_capture()

    assert restore_status == 0, f"failed to restore the default output's rate: status={restore_status}"
    assert realigned_rate == target_rate, (
        f"expected the aggregate device's nominal rate to be realigned to {target_rate} Hz "
        f"after the nominal-rate poller detected the mismatch, got {realigned_rate} -- the "
        f"poller did not react to a rate-only change"
    )
