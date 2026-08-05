"""T014 (US1): synthesized (TTS) audio played to a selected output device
actually reaches that device's signal path, verified by content
(spec.md SC-001, FR-005, contracts/audio-devices.md "Selection ordering").

**Why this cannot be a Python/PortAudio test alone**: exactly the same
reasoning as `test_input_selection.py`'s module docstring (US3). Asserting
that `audio_output_start` *returns true* proves nothing about whether audio
actually reaches the device -- the reported defect is a correctly-selected
device that still receives no audio, because `AudioCapture.swift`'s
`audio_output_start` accessed `engine.mainMixerNode` (which lazily forms the
mixer->output connection at the *then-current* device's format) before
setting `kAudioOutputUnitProperty_CurrentDevice`
(`contracts/audio-devices.md` "Selection ordering", research.md D7). A test
built only on PortAudio playback would never touch this code path at all.
This module compiles `AudioCapture.swift` on the fly as a standalone dylib
(`swiftc -emit-library`, mirroring `test_input_selection.py` exactly) and
drives `audio_output_start` / `audio_output_enqueue` / `audio_output_stop`
directly via `ctypes`.

**Capture is a subprocess, not an in-process thread (Critic round 1
finding)**: an earlier version of this module opened the capture side (via
`sounddevice`/PortAudio) on a background *thread of this same process*.
Critic round 1 reproduced, against the **fixed** code, the identical
`rms=0` silence signature this module treats as a failure, under that
in-process arrangement (matched rate, capture opened after the engine, 1 in
10 trials) — proving the earlier harness's "all-zero" reading was not
specific to the code under test, and pooled evidence (pre-fix 2/21, post-fix
0/48) did not reach significance (Fisher's exact p ≈ 0.09). The root cause:
two separate frameworks (this process's `ctypes`-loaded `AVAudioEngine`
output, and an in-process PortAudio `InputStream`) contending for the same
physical device from the *same process* intermittently starves one side,
independent of `AudioCapture.swift`. `_capture_subprocess.py` (this
package) moves the capture side into its own OS process, which removes that
contention entirely and — just as importantly — allows the **realistic**
ordering (output selected and started first, then something starts
listening), rather than the contention-driven "capture must open first"
workaround the in-process version needed. See `evidence/US1.md` for the
full measurement.

**Case 1 (SC-001)**: play a known tone into `BlackHole 2ch` (system default
sample rate) and capture its input side: non-silence, an audible level
(`test_capture_fixture._assert_recorded_level_is_plausible`, anchored to
BlackHole's measured output volume scalar — not just gain-invariant content
matching), matched-filter content match, and a check that the content
persists for (approximately) the full expected duration rather than a brief
truncated burst (`_assert_matches_for_expected_duration`). This case can
pass against the *current* (pre-fix) code — recorded as such below, per the
brief's explicit allowance; the defect this story fixes is intermittent,
not deterministic on every call (see case 2).

**Case 2 (the ordering defect's condition)**: the same, but with
`BlackHole 2ch`'s own nominal sample rate first changed (queried and
restored via CoreAudio directly, `try`/`finally`) to a rate that differs
substantially from the system default output's *current* rate -- the
condition `research.md` D7 and `contracts/audio-devices.md` describe as
triggering the wrong-format graph. The underlying mechanism
(`engine.start()` racing a device reconfiguration against
`engine.mainMixerNode`'s auto-connect) is not deterministic on every
invocation on this OS/AVAudioEngine version, so this case repeats the
play-and-capture attempt `RATE_MISMATCH_TRIALS` times and requires **every**
one to succeed. `RATE_MISMATCH_TRIALS` is derived from a measured pre-fix
per-trial failure rate `p` (see `evidence/US1.md` "Trial count, derived
rather than assumed"): the smallest `n` such that `(1 - p) ** n <= 0.05` --
*if* `p` equals its point estimate (0.20, from 20 trials). That estimate's
95% confidence interval is wide enough (see the same section) that this
guarantee does not hold at the interval's lower bound; `n = 14` is a
runtime-cost tradeoff, not a proven 5% ceiling.

**Negative control**: `test_no_audio_enqueued_reads_near_silence...` -- an
`audio_output_start` session with nothing ever enqueued must read near
silence, but (constraint 5, "a control that would pass with nothing
meaningful emitted is not a control") that alone would also pass if the
capture path were silently dead. The *same* session then enqueues the tone
and must recover it -- proving the capture path that just read near-silence
was genuinely alive, not stuck, mirroring `test_input_selection.py`'s
`MIC_OPEN_RMS_MIN` reasoning (there: a real device's non-zero noise floor
proves it is open; here: the same session later hearing real content proves
the same thing, since `BlackHole 2ch` is a virtual loopback with no noise
floor to fall back on -- see `test_capture_fixture.py`'s `SILENT_RMS_MAX`
docstring).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import shutil
import subprocess
import sys
import tempfile
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
    _assert_recorded_level_is_plausible,
    _matched_filter_score,
    _rms,
)

_SWIFT_SRC = (
    Path(__file__).resolve().parents[3] / "client" / "src-tauri" / "swift" / "AudioCapture.swift"
)
_CAPTURE_SUBPROCESS = Path(__file__).resolve().parent / "_capture_subprocess.py"

# `audio_output_enqueue` always builds buffers as float32 mono at this fixed
# rate (AudioCapture.swift :345-368), independent of whatever nominal rate
# the selected output device itself runs at.
TTS_SAMPLE_RATE = 24_000

TONE_FREQUENCY_HZ = 1000.0
TONE_AMPLITUDE = 0.8
TONE_DURATION_S = 3.0

# "of the expected duration within tolerance": how much shorter than the
# full synthesized tone the scored capture window is allowed to be. Content
# is required to match across the *entire* window (see
# `_assert_matches_for_expected_duration`), so a window this close to the
# full synthesized length is itself the duration check -- a truncated or
# intermittently-dropped delivery would fail to match across the later
# portion of it.
DURATION_TOLERANCE_S = 0.5
CAPTURE_DURATION_S = TONE_DURATION_S - DURATION_TOLERANCE_S

CAPTURE_SAMPLE_RATE = 48_000

# Time given to `audio_output_start` to finish settling before anything is
# asked to start listening -- a small, non-load-bearing buffer (unlike the
# in-process version's settle windows, this one no longer has to win a race
# against device contention; see module docstring).
OUTPUT_ENGINE_SETTLE_S = 0.5

# `_capture_subprocess.py` needs its own Python interpreter startup (import
# numpy/sounddevice/conftest) plus PortAudio's own stream-open latency
# before it is actually listening. This settle window is used both as "how
# long to wait after launching it before enqueuing" and as the lead-in
# trimmed from the front of its capture before scoring -- so its accuracy
# controls where the *end* of the scored window falls relative to when the
# tone actually stops playing, not just the front.
#
# The real subprocess startup lag `s` (wall-clock time from process launch
# to its first actually-captured frame) is safe up to
# `s <= TONE_DURATION_S - CAPTURE_DURATION_S == DURATION_TOLERANCE_S`
# (0.5s): past that, the scored window's tail extends beyond when the tone
# stops, and the trailing segment captures real silence instead of tone
# content (a false-fail risk, not a false-pass one). Measured directly:
# total subprocess wall time for a 1-3s requested capture consistently
# exceeded the requested duration by ~0.6s -- i.e. `s` is *not* comfortably
# under the 0.5s safety margin above, it is measured ~0.1s over it. Kept as
# a known, quantified risk rather than adjusted this round (see
# evidence/US1.md "Remaining risks") -- three full green runs were already
# recorded against this exact timing and changing it now would need
# re-verifying all of them.
SUBPROCESS_STARTUP_S = 1.5

# Total duration requested from the capture subprocess for a play-and-score
# attempt: the startup lead-in (trimmed before scoring) plus the window
# actually scored.
CAPTURE_REQUEST_S = SUBPROCESS_STARTUP_S + CAPTURE_DURATION_S

# How many consecutive segments a scored capture is split into for
# `_assert_matches_for_expected_duration`'s per-segment check.
_DURATION_SEGMENTS = 3

# See module docstring "Case 2" and evidence/US1.md "Trial count, derived
# rather than assumed": derived from a measured pre-fix per-trial failure
# rate p at the rate `_pick_mismatched_rate` selects (point estimate 0.20,
# 4/20 trials) via n = ceil(ln(0.05) / ln(1 - p)) = 14. This gives the
# pre-fix code at most a 5% chance of passing every trial by luck *only if*
# p truly equals 0.20 -- the 95% CI on that estimate is wide (see
# evidence/US1.md), and n=14 does not hold that guarantee at the interval's
# lower bound. Kept at 14 anyway as a runtime-cost tradeoff (see
# evidence/US1.md "Remaining risks"), not because 14 is provably sufficient.
RATE_MISMATCH_TRIALS = 14

# A rate change is only a meaningful trigger for the ordering defect if it
# is *substantially* different from the current default, not merely
# numerically different (contracts/audio-devices.md's own example,
# 44100<->48000, measured indistinguishable from a matched rate on this
# machine -- CoreAudio's own automatic sample-rate conversion papers over a
# nearby mismatch). This is a ratio floor, not an absolute Hz floor, so it
# scales with whatever the machine's actual default happens to be.
_MISMATCH_RATIO_MIN = 1.5

# `_pick_mismatched_rate` targets the available rate whose ratio to the
# current default is *closest to this value*, among those clearing
# `_MISMATCH_RATIO_MIN`. Empirically chosen, not derived (see
# evidence/US1.md "Re-measurement with the subprocess harness"): measured
# pre-fix per-trial failure rates at three candidate ratios on this machine
# were non-monotonic in the ratio itself -- 1.84x -> 0/20, 8x -> 4/20, 16x
# (the largest available) -> 2/20 -- so "the largest available mismatch"
# and "the smallest sufficient mismatch" (the Minor finding's original
# suggestion, tried and found to produce zero detectable failures in 20
# trials) are both worse choices than the empirically best-observed ratio.
_TARGET_MISMATCH_RATIO = 8.0


@pytest.fixture(autouse=True)
def _exclusive_device_access(audio_device_lock):
    """Serialize every test in this module against other device users (same
    rule every other hardware-backed module in this suite applies -- audio
    devices are shared, stateful hardware)."""
    with audio_device_lock():
        yield


@pytest.fixture(scope="module")
def audio_capture_dylib(tmp_path_factory) -> Path:
    """Compile `AudioCapture.swift` as a standalone dylib so this test can
    call `audio_output_start` / `audio_output_enqueue` / `audio_output_stop`
    directly via `ctypes`, exercising the real code path (see module
    docstring). Identical build recipe to `test_input_selection.py`'s
    fixture of the same name -- see that module for why each flag is
    required."""
    if shutil.which("swiftc") is None:
        pytest.fail(
            "swiftc not found -- required to build AudioCapture.swift for this test; "
            "the same tool client/src-tauri/build.rs already requires to build the app"
        )
    if not _SWIFT_SRC.exists():
        pytest.fail(f"AudioCapture.swift not found at {_SWIFT_SRC}")

    out_dir = tmp_path_factory.mktemp("audio_output_dylib")
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
    lib.audio_output_start.argtypes = [ctypes.c_char_p]
    lib.audio_output_start.restype = ctypes.c_bool
    lib.audio_output_enqueue.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    lib.audio_output_enqueue.restype = None
    lib.audio_output_stop.argtypes = []
    lib.audio_output_stop.restype = None
    return lib


@pytest.fixture
def blackhole_uid() -> str:
    uid = conftest.find_core_audio_device_uid(BLACKHOLE)
    if uid is None:
        pytest.skip(f"{BLACKHOLE!r} not present on this machine")
    return uid


def _synthesize_tone_24k() -> np.ndarray:
    t = np.arange(int(TTS_SAMPLE_RATE * TONE_DURATION_S)) / TTS_SAMPLE_RATE
    return (TONE_AMPLITUDE * np.sin(2 * np.pi * TONE_FREQUENCY_HZ * t)).astype(np.float32)


def _tone_reference_at_capture_rate(tone_24k: np.ndarray) -> np.ndarray:
    new_len = int(len(tone_24k) * CAPTURE_SAMPLE_RATE / TTS_SAMPLE_RATE)
    return resample(tone_24k.astype(np.float64), new_len)


def _capture_via_subprocess(duration_s: float, sample_rate: int, timeout_s: float) -> conftest.CaptureResult:
    """Runs `_capture_subprocess.py` as a separate OS process and returns
    what it captured. See module docstring for why this replaced an
    in-process background thread."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        npy_path = Path(tmp_dir) / "capture.npy"
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(_CAPTURE_SUBPROCESS),
                    BLACKHOLE,
                    str(duration_s),
                    str(sample_rate),
                    str(npy_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise conftest.DeviceCaptureError(
                f"capture subprocess did not finish within {timeout_s:.1f}s: {exc}"
            ) from exc

        stdout = proc.stdout.strip()
        if stdout.startswith("ABSENT "):
            raise conftest.DeviceAbsentError(stdout[len("ABSENT ") :])
        if proc.returncode != 0 or not stdout.startswith("OK "):
            raise conftest.DeviceCaptureError(
                f"capture subprocess failed (rc={proc.returncode}): "
                f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )
        _, sr_str, requested_str, actual_str = stdout.split()
        samples = np.load(npy_path)
        return conftest.CaptureResult(samples, int(sr_str), float(requested_str), float(actual_str))


class _SubprocessCapture:
    """Runs `_capture_via_subprocess` on a background thread of *this*
    process and surfaces its result (or exception) back via
    `.join_and_get()`. The thread itself does no CoreAudio/PortAudio work --
    it only waits on the child process -- so this does not reintroduce the
    in-process device contention `_capture_subprocess.py` exists to avoid;
    it exists only so a capture can run concurrently with this thread
    driving `audio_output_enqueue`.
    """

    def __init__(self, duration_s: float, sample_rate: int = CAPTURE_SAMPLE_RATE, timeout_s: float | None = None):
        self._holder: dict[str, object] = {}
        timeout_s = timeout_s if timeout_s is not None else duration_s + 15.0
        self._thread = threading.Thread(
            target=self._run, args=(duration_s, sample_rate, timeout_s), daemon=True
        )

    def _run(self, duration_s: float, sample_rate: int, timeout_s: float) -> None:
        try:
            self._holder["result"] = _capture_via_subprocess(duration_s, sample_rate, timeout_s)
        except conftest.DeviceCaptureError as exc:
            self._holder["error"] = exc

    def start(self) -> None:
        self._thread.start()

    def join_and_get(self, timeout_s: float) -> conftest.CaptureResult:
        self._thread.join(timeout=timeout_s)
        assert not self._thread.is_alive(), "capture subprocess did not finish in time"
        if "error" in self._holder:
            # Only a genuinely-absent device converts to a skip; every other
            # capture failure is a real failure. Routed through the same
            # type gate `record_input_device` itself uses, rather than a
            # bare `raise`, so an absent BlackHole skips here too.
            conftest._skip_if_absent_else_raise(self._holder["error"])  # type: ignore[arg-type]
        return self._holder["result"]  # type: ignore[return-value]


def _play_tone_and_capture(audio_lib, blackhole_uid: str, tone_24k: np.ndarray) -> conftest.CaptureResult:
    """Drive the real `audio_output_start` -> `audio_output_enqueue` ->
    `audio_output_stop` sequence, **engine first** (the realistic order --
    see module docstring for why the in-process version of this module
    could not use it), starting the capture subprocess only once the engine
    has had `OUTPUT_ENGINE_SETTLE_S` to settle."""
    started = audio_lib.audio_output_start(blackhole_uid.encode("utf-8"))
    try:
        assert started, "audio_output_start returned false for BlackHole's own UID"
        time.sleep(OUTPUT_ENGINE_SETTLE_S)

        capture = _SubprocessCapture(CAPTURE_REQUEST_S)
        capture.start()
        time.sleep(SUBPROCESS_STARTUP_S)

        tone_ptr = tone_24k.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        audio_lib.audio_output_enqueue(tone_ptr, len(tone_24k))
        return capture.join_and_get(timeout_s=CAPTURE_REQUEST_S + 15.0)
    finally:
        audio_lib.audio_output_stop()


def _assert_matches_for_expected_duration(
    result: conftest.CaptureResult, tone_24k: np.ndarray, context: str = ""
) -> None:
    """Non-silence, an audible level, overall matched-filter content, and --
    "of the expected duration within tolerance" -- that the match holds
    across every consecutive segment of the scored window, not merely a
    brief burst at the start. A capture that only matched the *first*
    segment (a truncated or dropped-mid-stream delivery) would fail the
    later segments' checks even though the recording as a whole is
    non-empty and non-silent.

    `context` is prefixed to every assertion message -- e.g. a trial index,
    for the repeated-attempts case, so a failure records *which* attempt
    failed (data this module's own detection-power measurement depends on).
    """
    mono = result.samples.mean(axis=1).astype(np.float64)
    lead_samples = int(round(SUBPROCESS_STARTUP_S * result.sample_rate))
    scored = mono[lead_samples:]
    assert len(scored) > 0, f"{context}capture window shorter than the startup lead-in"

    rms = _rms(scored)
    assert rms > SILENT_RMS_MAX, f"{context}expected non-silence, got rms={rms:.6g}"

    # Level check (Major finding): matched-filter content alone is
    # gain-invariant and cannot distinguish "audible through a working
    # path" from "the same content 1000x too quiet through a broken one" --
    # see test_capture_fixture.py's own module docstring for why this is a
    # *separate* check from content matching. Anchored to BlackHole's own
    # measured output volume scalar (not an absolute constant). NOTE this
    # does *not* "skip only itself": `_assert_recorded_level_is_plausible`
    # calls `pytest.skip()` internally if the volume scalar cannot be read,
    # which propagates out of this whole function -- on a machine where
    # that scalar is unreadable, the content-matching assertion below never
    # runs either for whichever call site is active at the time (see
    # evidence/US1.md "Remaining risks"). Placed before the content check
    # (rather than after) because level and non-silence are the more
    # fundamental claims; this ordering was not revisited this round.
    canonical_tone_rms = TONE_AMPLITUDE / np.sqrt(2)  # RMS of a full-length sine at this amplitude
    _assert_recorded_level_is_plausible(scored, canonical_tone_rms)

    reference = _tone_reference_at_capture_rate(tone_24k)
    overall_score = abs(_matched_filter_score(reference, scored))
    assert overall_score > MATCH_SCORE_MIN, (
        f"{context}expected the capture to match the tone played to BlackHole's output side "
        f"(score > {MATCH_SCORE_MIN}), got {overall_score:.4f}"
    )

    segment_len = len(scored) // _DURATION_SEGMENTS
    assert segment_len > 0, f"{context}scored window too short to split into segments"
    for i in range(_DURATION_SEGMENTS):
        segment = scored[i * segment_len : (i + 1) * segment_len]
        segment_score = abs(_matched_filter_score(reference, segment))
        assert segment_score > MATCH_SCORE_MIN, (
            f"{context}expected segment {i + 1}/{_DURATION_SEGMENTS} of the capture (of the "
            f"expected ~{CAPTURE_DURATION_S:.1f}s duration) to still match the tone (score > "
            f"{MATCH_SCORE_MIN}), got {segment_score:.4f} -- looks like delivery was truncated "
            f"or dropped partway rather than sustained for the expected duration"
        )


def test_tone_played_to_selected_output_reaches_its_input_side(audio_lib, blackhole_uid):
    """Case 1 (SC-001): BlackHole 2ch selected as output, at whatever its
    current (system-default-matching, on this machine) nominal rate is. May
    already pass against the pre-fix code -- see module docstring; the
    defect this story fixes is not that this case fails, but that case 2
    (below) intermittently does.
    """
    tone = _synthesize_tone_24k()
    result = _play_tone_and_capture(audio_lib, blackhole_uid, tone)
    _assert_matches_for_expected_duration(result, tone)


# ---------------------------------------------------------------------------
# Case 2: an output device whose nominal rate differs from the system
# default's -- the condition contracts/audio-devices.md and research.md D7
# describe as exposing the graph-construction-order defect.
# ---------------------------------------------------------------------------

_CORE_AUDIO = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
_CORE_FOUNDATION = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

_kAudioDevicePropertyNominalSampleRate = conftest._fourcc("nsrt")
_kAudioDevicePropertyAvailableNominalSampleRates = conftest._fourcc("nsr#")
_kAudioHardwarePropertyDefaultOutputDevice = conftest._fourcc("dOut")


class _AudioValueRange(ctypes.Structure):
    _fields_ = [("mMinimum", ctypes.c_double), ("mMaximum", ctypes.c_double)]


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


def _pick_mismatched_rate(available_rates: list[float], exclude_rate: float) -> float | None:
    """The available rate whose ratio to `exclude_rate` clears
    `_MISMATCH_RATIO_MIN` and is closest to `_TARGET_MISMATCH_RATIO` --
    see that constant's docstring for why neither "the largest available
    mismatch" (an earlier version's choice: an unnecessarily extreme
    mismatch that also maximizes the blast radius if a test process is
    ever killed before its `finally` restores the original rate -- exactly
    what happened once during this round's own investigation, see
    evidence/US1.md) nor "the smallest sufficient mismatch" (tried, and
    measured to produce zero detectable pre-fix failures in 20 trials) is
    used instead."""
    candidates = [
        rate
        for rate in available_rates
        if rate > 0 and max(rate, exclude_rate) / min(rate, exclude_rate) >= _MISMATCH_RATIO_MIN
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda rate: abs(max(rate, exclude_rate) / min(rate, exclude_rate) - _TARGET_MISMATCH_RATIO),
    )


@pytest.fixture
def mismatched_blackhole_rate(blackhole_uid):
    """Changes `BlackHole 2ch`'s own nominal sample rate to one that differs
    substantially from the system default output's *current* rate, queried
    fresh (never assumed) -- confirms the change actually took effect via a
    CoreAudio readback, then restores the original rate in `finally`
    regardless of test outcome. Skips (never fails) if BlackHole is absent,
    its available rates cannot be read, or no sufficiently-different rate
    is available on this machine -- all ambient hardware state, not a
    defect under test.
    """
    blackhole_id = _device_id_for_name(BLACKHOLE)
    if blackhole_id is None:
        pytest.skip(f"{BLACKHOLE!r} not present on this machine")

    default_output_id = _default_output_device_id()
    if default_output_id is None:
        pytest.skip("could not resolve the system default output device")
    default_rate = _read_nominal_sample_rate(default_output_id)
    if default_rate is None:
        pytest.skip("could not read the system default output's nominal sample rate")

    available_rates = _available_nominal_sample_rates(blackhole_id)
    mismatched_rate = _pick_mismatched_rate(available_rates, default_rate)
    if mismatched_rate is None:
        pytest.skip(
            f"no available {BLACKHOLE!r} rate at least {_MISMATCH_RATIO_MIN}x away from "
            f"the system default's current rate ({default_rate} Hz) on this machine"
        )

    original_rate = _read_nominal_sample_rate(blackhole_id)
    if original_rate is None:
        pytest.skip(f"could not read {BLACKHOLE!r}'s own current nominal sample rate")

    status = _set_nominal_sample_rate(blackhole_id, mismatched_rate)
    assert status == 0, f"failed to set {BLACKHOLE!r}'s nominal sample rate: status={status}"
    time.sleep(0.2)
    confirmed_rate = _read_nominal_sample_rate(blackhole_id)
    try:
        assert confirmed_rate == mismatched_rate, (
            f"expected {BLACKHOLE!r}'s nominal rate to read back as {mismatched_rate}, "
            f"got {confirmed_rate} -- the change did not actually take effect"
        )
        yield mismatched_rate
    finally:
        restore_status = _set_nominal_sample_rate(blackhole_id, original_rate)
        time.sleep(0.2)
        restored_rate = _read_nominal_sample_rate(blackhole_id)
        assert restore_status == 0 and restored_rate == original_rate, (
            f"failed to restore {BLACKHOLE!r}'s nominal sample rate to {original_rate} "
            f"(status={restore_status}, read back {restored_rate}) -- machine audio state "
            f"may be left altered"
        )


def test_tone_reaches_output_selected_at_a_rate_that_differs_from_the_system_default(
    audio_lib, blackhole_uid, mismatched_blackhole_rate
):
    """Case 2: the condition contracts/audio-devices.md's "Selection
    ordering" and research.md D7 describe as exposing the defect --
    BlackHole 2ch selected as output while its own nominal rate differs
    substantially from the system default output's. See module docstring
    for why every one of `RATE_MISMATCH_TRIALS` repeated attempts is
    required to succeed, not just one.
    """
    tone = _synthesize_tone_24k()
    for trial in range(RATE_MISMATCH_TRIALS):
        result = _play_tone_and_capture(audio_lib, blackhole_uid, tone)
        _assert_matches_for_expected_duration(
            result, tone, context=f"trial {trial + 1}/{RATE_MISMATCH_TRIALS}: "
        )


def test_no_audio_enqueued_reads_near_silence_and_the_same_session_proves_capture_is_alive(
    audio_lib, blackhole_uid
):
    """Negative control (constraint 5): an `audio_output_start` session with
    nothing ever enqueued must read near silence -- but that alone is not a
    valid control, since a capture path that was silently dead would also
    "pass" it. The *same* session then enqueues the tone and must recover
    it, proving the near-silent reading above came from a genuinely live,
    listening capture path and not a broken one that reads zero regardless
    of what plays (see module docstring). Both captures run via the same
    subprocess-based, engine-first sequence `_play_tone_and_capture` uses.
    """
    started = audio_lib.audio_output_start(blackhole_uid.encode("utf-8"))
    try:
        assert started, "audio_output_start returned false for BlackHole's own UID"
        time.sleep(OUTPUT_ENGINE_SETTLE_S)

        silence_capture = _SubprocessCapture(CAPTURE_REQUEST_S)
        silence_capture.start()
        # Deliberately nothing enqueued during this window.
        silence_result = silence_capture.join_and_get(timeout_s=CAPTURE_REQUEST_S + 15.0)

        tone = _synthesize_tone_24k()
        live_capture = _SubprocessCapture(CAPTURE_REQUEST_S)
        live_capture.start()
        time.sleep(SUBPROCESS_STARTUP_S)
        tone_ptr = tone.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        audio_lib.audio_output_enqueue(tone_ptr, len(tone))
        live_result = live_capture.join_and_get(timeout_s=CAPTURE_REQUEST_S + 15.0)
    finally:
        audio_lib.audio_output_stop()

    # Trimmed by SUBPROCESS_STARTUP_S -- this test's own actual timing (the
    # subprocess's startup lead-in), not any other test's window (Minor
    # finding: an earlier version of this trim used the two-phase in-process
    # constant, which did not describe this test's own single-phase flow).
    silence_mono = silence_result.samples.mean(axis=1).astype(np.float64)
    lead_samples = int(round(SUBPROCESS_STARTUP_S * silence_result.sample_rate))
    silence_trimmed = silence_mono[lead_samples:]
    silence_rms = _rms(silence_trimmed)
    assert silence_rms < SILENT_RMS_MAX, (
        f"expected near-silence with nothing enqueued, got rms={silence_rms:.6g}"
    )

    reference = _tone_reference_at_capture_rate(tone)
    silence_score = abs(_matched_filter_score(reference, silence_trimmed))
    assert silence_score < MATCH_SCORE_MIN, (
        f"expected the silent capture not to match the tone (score < {MATCH_SCORE_MIN}), "
        f"got {silence_score:.4f}"
    )

    # Proves the capture path above was genuinely alive, not dead (see
    # module docstring) -- the same reasoning as
    # `_assert_matches_for_expected_duration`, applied inline since
    # `live_result` is scored against the same `tone` already resampled
    # above.
    live_mono = live_result.samples.mean(axis=1).astype(np.float64)
    live_lead_samples = int(round(SUBPROCESS_STARTUP_S * live_result.sample_rate))
    live_scored = live_mono[live_lead_samples:]
    live_rms = _rms(live_scored)
    assert live_rms > SILENT_RMS_MAX, (
        f"expected the same session to read live audio once the tone was enqueued "
        f"(rms > {SILENT_RMS_MAX}), got {live_rms:.6g} -- looks like a dead capture path, "
        f"not a genuinely silent one"
    )
    live_score = abs(_matched_filter_score(reference, live_scored))
    assert live_score > MATCH_SCORE_MIN, (
        f"expected the same session, now with the tone enqueued, to match it "
        f"(score > {MATCH_SCORE_MIN}), got {live_score:.4f} -- this is what proves the "
        f"near-silent reading above came from a live capture path, not a dead one"
    )
