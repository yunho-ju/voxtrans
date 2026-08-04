"""T006 (US3): microphone capture reads the device chosen in the app,
independent of the system default input (research.md D0/D1, SC-005, FR-009).

**Why this cannot be a Python/PortAudio test alone**: `test_capture_fixture.py`
proves that *something* can read a device's content correctly, using
PortAudio. It proves nothing about VoxTrans's own code, because the reported
bug is entirely inside VoxTrans's Swift `mic_start_capture` -- it opens
`engine.inputNode`, which resolves to the system default input, no matter
what a caller might want to select. A test that only proves PortAudio can
select a device does not exercise the defective code path at all.

**How this test reaches the real code path**: `AudioCapture.swift` is
compiled here, on the fly, as a standalone dylib (`swiftc -emit-library`,
*not* `build.rs`'s `-static` archive linked into the Tauri binary), and
loaded directly via `ctypes`. This is the "Swift harness" tasks.md's T002
deferred at Phase 1 ("revisit if a defect appears that is only observable
inside Swift") -- input-device selection is exactly that class of defect:
the graph is built correctly and still reads the wrong hardware, which is
invisible from any layer that does not call the real `mic_start_capture`.

**The two-device content distinction** (the task's stated trap): `BlackHole
2ch` is a loopback this test can write into via `sd.play()`; `MacBook Pro
마이크` is a real microphone that hears the room, structurally isolated from
BlackHole's loopback (there is no physical speaker feeding it). A synthesized
tone is played into BlackHole while capture is independently pointed, by
UID, at each device in turn:

- selecting BlackHole's UID: the captured audio must matched-filter-match
  the tone (`test_selecting_blackhole_reads_the_tone_fed_to_it`).
- selecting the microphone's UID, tone still playing into BlackHole: the
  captured audio must *not* match the tone
  (`test_selecting_the_microphone_does_not_read_blackholes_tone`) -- the
  negative control constraint 5 requires: if `mic_start_capture` ignored the
  UID and always opened the system default (as it does today), *this*
  assertion is the one that fails, since the default on this machine is the
  microphone, and the microphone genuinely does hear nothing resembling the
  tone. See the module's mutation notes in
  `specs/002-audio-routing/evidence/US3.md` for why the *positive* test
  above is what actually catches that specific mutant on this machine, and
  why both tests are kept regardless.

Neither device's identity is assumed to be the system default (constraint
1): both are located by name via CoreAudio directly
(`conftest.find_core_audio_device_uid`), never by index, and the test skips
--- never fails --- if either is not present on the machine running it.

**FR-009** (fabricated UID): `test_fabricated_uid_fails_without_capturing_anything`
asserts `mic_start_capture` returns `false` for a UID that cannot resolve to
any device, and that nothing is ever captured as a result -- the direct,
content-level proof that failure does not silently fall back to the default
(SC-005's "verified by signal content ... never configuration readback"
applies here too: the earlier bug's whole failure mode *was* a
config-looks-fine-but-reads-the-wrong-hardware defect, so "the call
returned false" is a behavioural fact, not configuration, and is paired
with "and therefore nothing arrived" to rule out a fallback that happens to
also return false after already having started capturing).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import sounddevice as sd
from scipy.signal import resample

from integration import conftest
from integration.test_capture_fixture import (
    BLACKHOLE,
    LEAD_IN_S,
    MATCH_SCORE_MIN,
    SAMPLE_RATE,
    TONE_DURATION_S,
    _matched_filter_score,
    _synthesize_tone,
)

MIC_DEVICE_NAME = "MacBook Pro 마이크"

# mic_start_capture's fixed internal output rate (AudioCapture.swift:
# `targetSampleRate: Double = 16000`), independent of whatever rate the
# reference tone is synthesized/played at.
MIC_CAPTURE_SAMPLE_RATE = 16_000

# How long to let mic_start_capture run before stopping it and scoring what
# arrived. Shorter than TONE_DURATION_S so the tone is still playing for the
# whole capture window; long enough to accumulate a matched-filter-worthy
# amount of audio at 16kHz.
CAPTURE_SECONDS = 2.5

# Floor for "the selected device is genuinely open and producing live
# audio", used by the negative control below. A real analog microphone
# always has some noise floor (thermal/electrical, or the room itself) --
# it does not read literal 0.0 RMS the way an idle virtual loopback does
# (test_capture_fixture.py's SILENT_RMS_MAX documents exactly 0.0 measured
# across every idle-BlackHole trial, "no hardware noise floor -- it is a
# virtual loopback"). Measured directly on the reference machine's built-in
# microphone: 0.00136-0.00554 RMS across six 2.5 s trials in a quiet room
# (input volume scalar 0.90). This floor sits >130x below even the quietest
# of those, while remaining >>0 -- enough margin that a quieter room should
# not make this flaky, while a genuinely dead capture (silently installed no
# tap, e.g. because `mic_start_capture` returned `true` without ever actually
# opening the device) reads exactly 0.0 and is caught.
#
# The discriminating gap is between "exactly 0.0" and ">=1.4e-3", i.e. two
# orders of magnitude on each side of this floor. Do NOT raise it toward
# 1e-3 on the strength of the upper end of the range -- the lower end is
# what bounds flakiness.
MIC_OPEN_RMS_MIN = 1e-5

_SWIFT_SRC = (
    Path(__file__).resolve().parents[3] / "client" / "src-tauri" / "swift" / "AudioCapture.swift"
)

_AUDIO_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_ssize_t,
    ctypes.c_ssize_t,
    ctypes.c_ssize_t,
)


@pytest.fixture(autouse=True)
def _exclusive_device_access(audio_device_lock):
    """Serialize every test in this module against other device users, the
    same rule `test_capture_fixture.py` applies to its own hardware-backed
    tests -- audio devices are shared, stateful hardware."""
    with audio_device_lock():
        yield


@pytest.fixture(scope="module")
def audio_capture_dylib(tmp_path_factory) -> Path:
    """Compile `AudioCapture.swift` as a standalone dylib so this test can
    call `mic_start_capture` / `audio_list_input_devices` directly via
    `ctypes`, exercising the real code path rather than a re-implementation
    of it (see module docstring). Session-scoped per test module: compiling
    is not the thing under test, so it happens once.
    """
    if shutil.which("swiftc") is None:
        pytest.fail(
            "swiftc not found -- required to build AudioCapture.swift for this test; "
            "the same tool client/src-tauri/build.rs already requires to build the app"
        )
    if not _SWIFT_SRC.exists():
        pytest.fail(f"AudioCapture.swift not found at {_SWIFT_SRC}")

    out_dir = tmp_path_factory.mktemp("audio_capture_dylib")
    lib_path = out_dir / "libAudioCaptureTest.dylib"
    result = subprocess.run(
        [
            "swiftc",
            "-emit-library",
            "-module-name",
            "AudioCapture",
            # `@_cdecl` functions in this file use Swift's default
            # (internal) access level, matching what build.rs's `-static`
            # archive already links fine intra-process. A plain dylib build
            # demotes non-public symbols to local linkage (confirmed:
            # `nm` shows lowercase `t`), which `dlsym`/ctypes cannot resolve
            # at all -- `-enable-testing` is the standard, minimally-
            # invasive way to expose `internal` symbols externally without
            # editing access modifiers in AudioCapture.swift itself
            # (which would cascade into making `AudioCallback` and friends
            # public too, a production-code change this test should not
            # require).
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
    return ctypes.CDLL(str(audio_capture_dylib))


@pytest.fixture
def blackhole_uid() -> str:
    uid = conftest.find_core_audio_device_uid(BLACKHOLE)
    if uid is None:
        pytest.skip(f"{BLACKHOLE!r} not present on this machine")
    return uid


@pytest.fixture
def mic_uid() -> str:
    uid = conftest.find_core_audio_device_uid(MIC_DEVICE_NAME)
    if uid is None:
        pytest.skip(f"{MIC_DEVICE_NAME!r} not present on this machine")
    return uid


class _MicCaptureFailure(RuntimeError):
    """`mic_start_capture` itself returned `false`."""


def _capture_mic(lib, device_uid: str, duration_s: float) -> np.ndarray:
    """Start the real `mic_start_capture(callback, device_uid)` FFI entry
    point, accumulate `duration_s` seconds of the mono float32 samples it
    delivers via callback, then stop. Raises `_MicCaptureFailure` if the
    call itself reports failure -- callers use that to assert FR-009's "no
    silent fallback": a failed start must arrive with **no** samples ever
    collected, not just a false return value.
    """
    chunks: list[np.ndarray] = []
    lock = threading.Lock()

    def _on_audio(data_ptr, num_samples, _sample_rate, _channels) -> None:
        if num_samples <= 0 or not data_ptr:
            return
        buf = np.ctypeslib.as_array(data_ptr, shape=(num_samples,)).copy()
        with lock:
            chunks.append(buf)

    c_callback = _AUDIO_CALLBACK(_on_audio)

    lib.mic_start_capture.restype = ctypes.c_bool
    lib.mic_start_capture.argtypes = [_AUDIO_CALLBACK, ctypes.c_char_p]
    lib.mic_stop_capture.restype = None
    lib.mic_stop_capture.argtypes = []

    ok = lib.mic_start_capture(c_callback, device_uid.encode("utf-8"))
    if not ok:
        # mic_start_capture must not have installed a tap in this case --
        # nothing to stop, and (checked by the caller) nothing captured.
        raise _MicCaptureFailure(
            f"mic_start_capture returned false for device_uid={device_uid!r}"
        )

    try:
        time.sleep(duration_s)
    finally:
        lib.mic_stop_capture()

    with lock:
        if not chunks:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(chunks).astype(np.float64)


def _play_tone_into_blackhole(tone: np.ndarray, sample_rate: int):
    """Play `tone` into BlackHole's output side on a background thread.
    Mirrors the playback half of
    `test_capture_fixture._play_reference_and_capture` -- reused via the
    same underlying `conftest._resolve_portaudio_output_index` helper that
    function uses -- but not that function itself, since its capture half
    is PortAudio-based and the capture side under test here must be the
    real Swift `mic_start_capture` path (see module docstring for why).
    Skips if BlackHole has no PortAudio-visible output side (same
    ambient-state rule the fixture itself follows).
    """
    output_index = conftest._resolve_portaudio_output_index(BLACKHOLE)
    if output_index is None:
        pytest.skip(f"{BLACKHOLE!r} has no PortAudio-visible output side")

    stereo = np.column_stack([tone, tone]).astype(np.float32)
    play_exception: list[BaseException] = []

    def _play() -> None:
        try:
            sd.play(stereo, samplerate=sample_rate, device=output_index, blocking=True)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the test thread below
            play_exception.append(exc)

    thread = threading.Thread(target=_play)
    thread.start()
    return thread, play_exception


def _tone_at_mic_capture_rate(tone: np.ndarray) -> np.ndarray:
    """Resample the reference tone (synthesized at `SAMPLE_RATE`) to
    `MIC_CAPTURE_SAMPLE_RATE`, mirroring the `resample(...)` pattern
    `test_capture_fixture.test_speech_played_into_loopback_is_recovered`
    already uses when reference and capture rates differ -- required here
    because `mic_start_capture` always converts to a fixed 16kHz, unlike
    the PortAudio fixture's native-rate capture."""
    new_len = int(len(tone) * MIC_CAPTURE_SAMPLE_RATE / SAMPLE_RATE)
    return resample(tone.astype(np.float64), new_len)


def test_selecting_blackhole_reads_the_tone_fed_to_it(audio_lib, blackhole_uid):
    """SC-005, positive case: selecting BlackHole's UID makes captured
    audio carry the tone fed to BlackHole's loopback, verified by matched-
    filter content -- never by reading back which UID was configured."""
    tone = _synthesize_tone()
    play_thread, play_exception = _play_tone_into_blackhole(tone, SAMPLE_RATE)
    try:
        time.sleep(LEAD_IN_S)
        captured = _capture_mic(audio_lib, blackhole_uid, CAPTURE_SECONDS)
    finally:
        play_thread.join(timeout=TONE_DURATION_S + 2.0)

    assert not play_thread.is_alive(), "reference playback thread did not finish (sd.play stuck)"
    assert not play_exception, f"reference playback failed: {play_exception}"
    assert captured.size > 0, "expected non-empty capture when selecting BlackHole's UID"

    reference = _tone_at_mic_capture_rate(tone)
    score = abs(_matched_filter_score(reference, captured))
    assert score > MATCH_SCORE_MIN, (
        f"expected audio captured while BlackHole was selected to match the tone fed to "
        f"it (score > {MATCH_SCORE_MIN}), got {score:.4f}"
    )


def test_selecting_the_microphone_does_not_read_blackholes_tone(audio_lib, mic_uid):
    """SC-005, negative control (constraint 5: a test that would pass with
    nothing meaningful emitted is not a test): with the *same* tone playing
    into BlackHole, selecting the microphone's UID instead must **not**
    read that tone. The microphone hears the room, structurally isolated
    from BlackHole's loopback -- if `mic_start_capture` ignored the
    selected UID entirely, this assertion is the one built to catch that on
    a machine whose system default happens to already be the microphone
    (see `specs/002-audio-routing/evidence/US3.md`'s mutation notes for the
    two-device case this alone does not cover, and why the positive test
    above covers it).

    Critic round 1 finding 2: an earlier version of this test treated an
    *empty* capture as an automatic pass ("no match is possible"). That
    made `mic_start_capture` returning `true` for the microphone's UID
    while silently installing no tap -- zero samples, ever -- score as a
    perfect negative result, which is exactly the RC-6/RC-7 failure class
    (research.md) this feature exists to catch: a device that reports
    success while producing nothing. The capture is now required to be
    both non-empty *and* to exceed `MIC_OPEN_RMS_MIN` -- proof the selected
    device is genuinely open and live -- strictly before its *content* is
    allowed to differ from the tone.
    """
    tone = _synthesize_tone()
    play_thread, play_exception = _play_tone_into_blackhole(tone, SAMPLE_RATE)
    try:
        time.sleep(LEAD_IN_S)
        captured = _capture_mic(audio_lib, mic_uid, CAPTURE_SECONDS)
    finally:
        play_thread.join(timeout=TONE_DURATION_S + 2.0)

    assert not play_thread.is_alive(), "reference playback thread did not finish (sd.play stuck)"
    assert not play_exception, f"reference playback failed: {play_exception}"

    assert captured.size > 0, (
        "expected the microphone to be genuinely open and capturing audio -- an empty "
        "capture from a device that reported a successful start is silent failure "
        "(research.md RC-6/RC-7), not a valid negative result"
    )
    captured_rms = float(np.sqrt(np.mean(captured**2)))
    assert captured_rms > MIC_OPEN_RMS_MIN, (
        f"expected the microphone capture to contain live audio (rms > {MIC_OPEN_RMS_MIN}), "
        f"got rms={captured_rms:.8f} -- looks like a silently-dead capture (a tap that was "
        f"never actually installed), not a genuinely open microphone reading nothing that "
        f"happens to match the tone"
    )

    reference = _tone_at_mic_capture_rate(tone)
    score = abs(_matched_filter_score(reference, captured))
    assert score < MATCH_SCORE_MIN, (
        f"expected audio captured while the microphone was selected NOT to match the tone "
        f"fed to BlackHole (score < {MATCH_SCORE_MIN}), got {score:.4f} -- capture is reading "
        f"the wrong device"
    )


def test_fabricated_uid_fails_without_capturing_anything(audio_lib, blackhole_uid, mic_uid):
    """FR-009: a UID that resolves to no real device must fail the start
    and must never silently fall back to the default -- checked here as a
    content-level fact (zero samples ever arrive), not a configuration
    readback. `blackhole_uid`/`mic_uid` are only used to prove real devices
    *are* present on this machine (so a `false` return cannot be explained
    by "there are no input devices at all"); the fabricated UID itself
    deliberately does not collide with either.
    """
    fake_uid = "com.voxtrans.test.fabricated-uid-does-not-exist-12345"
    assert fake_uid not in (blackhole_uid, mic_uid)

    with pytest.raises(_MicCaptureFailure):
        _capture_mic(audio_lib, fake_uid, duration_s=0.5)


def _list_input_devices(lib) -> dict[str, str]:
    """Call the real `audio_list_input_devices` FFI entry point and parse
    its `name|uid` lines. Shared by the two enumeration tests below."""
    lib.audio_list_input_devices.restype = ctypes.c_void_p
    ptr = lib.audio_list_input_devices()
    assert ptr, "audio_list_input_devices returned a null pointer"

    raw = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.free(ctypes.c_void_p(ptr))

    devices: dict[str, str] = {}
    for line in raw.splitlines():
        parts = line.split("|", 1)
        assert len(parts) == 2, f"malformed device line (expected name|uid): {line!r}"
        name, uid = parts
        assert name, f"empty device name in line: {line!r}"
        assert uid, f"empty device uid in line: {line!r}"
        devices[name] = uid
    return devices


def test_input_device_enumeration_lists_real_devices_by_uid(audio_lib, blackhole_uid, mic_uid):
    """T008: `audio_list_input_devices` must enumerate real devices whose
    UIDs match what CoreAudio itself reports -- compared against
    `conftest.find_core_audio_device_uid`'s independent resolution, not
    against `audio_list_input_devices`'s own prior output."""
    devices = _list_input_devices(audio_lib)

    assert devices.get(BLACKHOLE) == blackhole_uid
    assert devices.get(MIC_DEVICE_NAME) == mic_uid


def test_input_device_enumeration_matches_independent_coreaudio_ground_truth(audio_lib):
    """`contracts/audio-devices.md`'s verification table: the reported input
    name set 'must match' an independent source -- match, not merely
    contain or overlap. `audio_list_input_devices` filters devices by
    `kAudioDevicePropertyStreams` scoped to input
    (`AudioCapture.swift:487-491`); removing or breaking that filter would
    enumerate every device, input and output alike (e.g. a speaker would
    appear as a selectable microphone and fail on start). Set equality
    against `conftest.list_core_audio_input_device_names` -- resolved
    independently via CoreAudio, not through anything
    `audio_list_input_devices` itself computes -- is what actually catches
    that: a prior version of this suite only checked well-formedness and
    specific known devices' UIDs, which a "no filter at all" bug would
    still satisfy (both `BlackHole 2ch` and the microphone would still be
    present, just alongside everything else)."""
    reported_names = set(_list_input_devices(audio_lib).keys())
    ground_truth_names = conftest.list_core_audio_input_device_names()

    assert reported_names == ground_truth_names
