"""T017a (US1): the output device disappearing mid-playback must not hang
the playback thread, and the next `audio_output_start` after that must
still succeed (spec.md Edge Cases -> "Output device disappears mid-playback";
specs/002-audio-routing/briefs/US1-round2-T017-T017b.md).

**Why an aggregate device, not just unplugging BlackHole**: this suite runs
unattended and repeatedly: it needs a device it can destroy *itself*,
on-demand, mid-test, and be certain no other test or human is depending on
it -- unplugging the one real loopback device this machine has (BlackHole
2ch) would also break every other hardware-backed test in this suite for as
long as it stayed unplugged. A non-private aggregate device built over
BlackHole gets its own distinct UID and its own lifecycle, entirely owned by
this module, while still driving real audio hardware underneath.

**No CFDictionary/CFArray building precedent existed before this file**
(brief: "aggregate device 생성/파괴의 테스트 선례 없음"). `AudioCapture.swift`
:37-57 builds an aggregate device from Swift using an `NSDictionary`
literal, which is not available from Python; `_create_output_aggregate_over`
below builds the same `CFDictionaryRef` shape directly via `ctypes` calls
into CoreFoundation/CoreAudio, reusing `conftest`'s already-established
low-level primitives (`_fourcc`, `_AudioObjectPropertyAddress`,
`_kCFStringEncodingUTF8`, device enumeration/name/UID) rather than adding a
new compiled Swift source just for a two-function, test-only need.

**What this module intentionally does not attempt**: driving the real
server/client WebSocket pipeline end-to-end to prove "the session accepts
the next utterance" (tasks.md T017a's literal wording) needs a translation
model and a running Tauri app -- neither is available to a server-only
pytest module, and no WS-driving test precedent exists in this suite (see
brief "테스트 인프라"). This module instead proves the playback-layer
equivalent directly: after the device that was mid-playback is destroyed,
`audio_output_stop()` returns promptly and a **fresh** `audio_output_start`
against the (still-present) real BlackHole UID succeeds and delivers audio,
end to end, verified by signal content -- reusing
`test_output_reaches_device.py`'s exact play-and-capture helper. The
session-level WS claim is deferred to US2's T021 integration test, which
has the model and pipeline this would need.

Server `tts_playing`: the brief's "조사로 확정된 사실" section establishes
that this flag cannot leak from the code path exercised here -- its only
release site is `_translate_and_speak`'s `finally`
(`server/ws_handler.py:381-385`), which has no coupling to client playback
completion (no such coupling exists yet; that is what US2 T021/T022 add).
Nothing in this module could make that flag observable one way or the
other, so this module does not attempt to.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest

from integration import conftest
from integration.test_output_reaches_device import (
    OUTPUT_ENGINE_SETTLE_S,
    _assert_matches_for_expected_duration,
    _play_tone_and_capture,
    _synthesize_tone_24k,
)
from integration.test_capture_fixture import BLACKHOLE

_SWIFT_SRC = (
    Path(__file__).resolve().parents[3] / "client" / "src-tauri" / "swift" / "AudioCapture.swift"
)

# How long `audio_output_stop()` is given to return once the device it was
# playing to has been destroyed out from under it, before this test treats
# the playback thread as hung (brief T017a case 1: "유계 시간(예: 5s)").
STOP_RESPONSIVENESS_TIMEOUT_S = 5.0


@pytest.fixture(autouse=True)
def _exclusive_device_access(audio_device_lock):
    """Serialize against every other test in this suite that touches audio
    devices -- same rule `test_output_reaches_device.py` applies, and
    doubly important here since this module also creates/destroys a real
    CoreAudio aggregate device."""
    with audio_device_lock():
        yield


@pytest.fixture(scope="module")
def audio_capture_dylib(tmp_path_factory) -> Path:
    """Identical build recipe to `test_output_reaches_device.py`'s fixture
    of the same name -- see that module for why each flag is required."""
    if shutil.which("swiftc") is None:
        pytest.fail(
            "swiftc not found -- required to build AudioCapture.swift for this test; "
            "the same tool client/src-tauri/build.rs already requires to build the app"
        )
    if not _SWIFT_SRC.exists():
        pytest.fail(f"AudioCapture.swift not found at {_SWIFT_SRC}")

    out_dir = tmp_path_factory.mktemp("audio_output_lost_dylib")
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


# ---------------------------------------------------------------------------
# Aggregate device creation/destruction: a CFDictionaryRef built directly via
# ctypes (see module docstring). Only what `AudioHardwareCreateAggregateDevice`
# needs for a single-subdevice, non-private aggregate: name, uid, private,
# subdevices.
# ---------------------------------------------------------------------------

_CORE_AUDIO = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
_CORE_FOUNDATION = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

_CORE_FOUNDATION.CFStringCreateWithCString.restype = ctypes.c_void_p
_CORE_FOUNDATION.CFStringCreateWithCString.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_uint32,
]
_CORE_FOUNDATION.CFNumberCreate.restype = ctypes.c_void_p
_CORE_FOUNDATION.CFNumberCreate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
_CORE_FOUNDATION.CFArrayCreate.restype = ctypes.c_void_p
_CORE_FOUNDATION.CFArrayCreate.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_long,
    ctypes.c_void_p,
]
_CORE_FOUNDATION.CFDictionaryCreate.restype = ctypes.c_void_p
_CORE_FOUNDATION.CFDictionaryCreate.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_long,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_CORE_AUDIO.AudioHardwareCreateAggregateDevice.restype = ctypes.c_int32
_CORE_AUDIO.AudioHardwareCreateAggregateDevice.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint32),
]
_CORE_AUDIO.AudioHardwareDestroyAggregateDevice.restype = ctypes.c_int32
_CORE_AUDIO.AudioHardwareDestroyAggregateDevice.argtypes = [ctypes.c_uint32]

# The exported "const struct" symbols CFDictionaryCreate/CFArrayCreate need
# a *pointer to*, not a value of -- `.in_dll` binds a ctypes object to the
# symbol's address, and `ctypes.byref` on that object yields exactly that
# address, without needing this module to declare the real (unused, from
# Python's side) struct layout. Same technique CPython's own `ctypes`
# documentation shows for opaque exported constants.
_kCFTypeDictionaryKeyCallBacks = ctypes.c_byte.in_dll(
    _CORE_FOUNDATION, "kCFTypeDictionaryKeyCallBacks"
)
_kCFTypeDictionaryValueCallBacks = ctypes.c_byte.in_dll(
    _CORE_FOUNDATION, "kCFTypeDictionaryValueCallBacks"
)
_kCFTypeArrayCallBacks = ctypes.c_byte.in_dll(_CORE_FOUNDATION, "kCFTypeArrayCallBacks")

_kCFNumberSInt32Type = 3


def _cfstr(value: str) -> int:
    ptr = _CORE_FOUNDATION.CFStringCreateWithCString(
        None, value.encode("utf-8"), conftest._kCFStringEncodingUTF8
    )
    assert ptr, f"CFStringCreateWithCString failed for {value!r}"
    return ptr


def _cfnum(value: int) -> int:
    c_value = ctypes.c_int32(value)
    ptr = _CORE_FOUNDATION.CFNumberCreate(
        None, _kCFNumberSInt32Type, ctypes.byref(c_value)
    )
    assert ptr, f"CFNumberCreate failed for {value!r}"
    return ptr


def _cfarray(items: list[int]) -> int:
    arr = (ctypes.c_void_p * len(items))(*items)
    ptr = _CORE_FOUNDATION.CFArrayCreate(
        None, arr, len(items), ctypes.byref(_kCFTypeArrayCallBacks)
    )
    assert ptr, "CFArrayCreate failed"
    return ptr


def _cfdict(pairs: dict[str, int]) -> int:
    keys = (ctypes.c_void_p * len(pairs))(*[_cfstr(k) for k in pairs])
    values = (ctypes.c_void_p * len(pairs))(*pairs.values())
    ptr = _CORE_FOUNDATION.CFDictionaryCreate(
        None,
        keys,
        values,
        len(pairs),
        ctypes.byref(_kCFTypeDictionaryKeyCallBacks),
        ctypes.byref(_kCFTypeDictionaryValueCallBacks),
    )
    assert ptr, "CFDictionaryCreate failed"
    return ptr


def _create_output_aggregate_over(base_device_uid: str, aggregate_uid: str) -> int:
    """Creates a non-private aggregate device with `base_device_uid` as its
    only subdevice. Returns the aggregate's `AudioObjectID`. Raises (not
    skips) `RuntimeError` on failure -- callers convert to a skip, since
    aggregate creation failing is ambient machine/permission state, not the
    thing under test."""
    subdevice = _cfdict({"uid": _cfstr(base_device_uid)})
    subdevice_list = _cfarray([subdevice])
    description = _cfdict(
        {
            "name": _cfstr("VoxTrans Test Aggregate"),
            "uid": _cfstr(aggregate_uid),
            "private": _cfnum(0),
            "subdevices": subdevice_list,
        }
    )
    device_id = ctypes.c_uint32(0)
    status = _CORE_AUDIO.AudioHardwareCreateAggregateDevice(description, ctypes.byref(device_id))
    if status != 0:
        raise RuntimeError(f"AudioHardwareCreateAggregateDevice failed: status={status}")
    # `AudioHardwareCreateAggregateDevice` returning noErr does not mean the
    # device is immediately visible to a subsequent device-list query --
    # measured directly: back-to-back create/destroy cycles across two
    # tests in the same session intermittently made `findAudioDeviceByUID`
    # (AudioCapture.swift) report "device not found" for a UID that had
    # just been created successfully. A short settle closes that race.
    time.sleep(0.3)
    return device_id.value


def _destroy_aggregate(device_id: int) -> None:
    status = _CORE_AUDIO.AudioHardwareDestroyAggregateDevice(device_id)
    if status != 0:
        raise RuntimeError(f"AudioHardwareDestroyAggregateDevice failed: status={status}")
    # Mirrors the settle in `_create_output_aggregate_over`: measured
    # flakiness (below) came from creating a *new* aggregate over the same
    # BlackHole subdevice immediately after destroying the previous one,
    # not only from using a just-created one immediately.
    time.sleep(0.3)


@pytest.fixture
def output_aggregate(blackhole_uid: str):
    """Creates a non-private aggregate device over BlackHole 2ch and yields
    `(aggregate_uid, aggregate_device_id)`. Tracks whether the aggregate has
    already been destroyed (tests in this module destroy it themselves, as
    the thing under test) so the teardown never double-destroys or leaks it
    -- an aggregate device left behind would contaminate every other audio
    test in this suite (brief: "aggregate가 남으면 이후 모든 오디오 테스트가
    오염된다")."""
    aggregate_uid = f"com.voxtrans.test.output-aggregate.{uuid.uuid4()}"
    try:
        device_id = _create_output_aggregate_over(blackhole_uid, aggregate_uid)
    except RuntimeError as exc:
        pytest.skip(f"could not create a test aggregate device over BlackHole: {exc}")

    state = {"destroyed": False}

    def destroy_now() -> None:
        if not state["destroyed"]:
            _destroy_aggregate(device_id)
            state["destroyed"] = True

    try:
        yield aggregate_uid, device_id, destroy_now
    finally:
        destroy_now()


def test_stop_is_responsive_after_the_output_device_is_destroyed_mid_playback(
    audio_lib, output_aggregate
):
    """T017a case 1: `audio_output_start` on a device that is then destroyed
    while audio is actively being enqueued to it -- `audio_output_stop()`
    must still return within `STOP_RESPONSIVENESS_TIMEOUT_S`, not hang
    forever on a playback graph whose device object no longer exists."""
    aggregate_uid, _device_id, destroy_now = output_aggregate

    started = audio_lib.audio_output_start(aggregate_uid.encode("utf-8"))
    assert started, "audio_output_start returned false for the freshly created aggregate's own UID"

    time.sleep(OUTPUT_ENGINE_SETTLE_S)

    tone = _synthesize_tone_24k()
    tone_ptr = tone.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    stop_feeding = threading.Event()

    def _feed() -> None:
        while not stop_feeding.is_set():
            audio_lib.audio_output_enqueue(tone_ptr, len(tone))
            time.sleep(0.05)

    feeder = threading.Thread(target=_feed, daemon=True)
    feeder.start()
    try:
        # Let a handful of real enqueue calls land against the live
        # aggregate before pulling it out from under the graph.
        time.sleep(0.3)

        destroy_now()

        # Give the playback graph a moment to actually notice the device is
        # gone (Core Audio device-loss notifications are not synchronous
        # with the destroy call).
        time.sleep(0.3)

        stop_returned = threading.Event()

        def _call_stop() -> None:
            audio_lib.audio_output_stop()
            stop_returned.set()

        stopper = threading.Thread(target=_call_stop, daemon=True)
        stopper.start()
        stopper.join(timeout=STOP_RESPONSIVENESS_TIMEOUT_S)

        assert stop_returned.is_set(), (
            f"audio_output_stop() did not return within {STOP_RESPONSIVENESS_TIMEOUT_S:.1f}s "
            "of the output device being destroyed mid-playback -- the playback graph appears "
            "to be hung"
        )
    finally:
        stop_feeding.set()
        feeder.join(timeout=STOP_RESPONSIVENESS_TIMEOUT_S)
        assert not feeder.is_alive(), "enqueue-feeding thread did not stop in time"


def test_a_fresh_start_after_the_output_device_is_lost_reaches_the_device_again(
    audio_lib, output_aggregate, blackhole_uid
):
    """T017a case 2 ('the session accepts the next utterance', translated to
    the playback layer -- see module docstring for why): once the device
    that was mid-playback has been destroyed and cleaned up, a **fresh**
    `audio_output_start` against BlackHole 2ch's own (still-present, real)
    UID must succeed and actually deliver audio again, reusing
    `test_output_reaches_device.py`'s exact play-and-capture-and-verify
    helper -- the same signal-content proof as T014, now run *after* a
    device loss rather than from a clean start."""
    aggregate_uid, _device_id, destroy_now = output_aggregate

    started = audio_lib.audio_output_start(aggregate_uid.encode("utf-8"))
    assert started, "audio_output_start returned false for the freshly created aggregate's own UID"
    time.sleep(OUTPUT_ENGINE_SETTLE_S)
    destroy_now()
    audio_lib.audio_output_stop()

    tone = _synthesize_tone_24k()
    try:
        result = _play_tone_and_capture(audio_lib, blackhole_uid, tone)
    except AssertionError as exc:
        # Critic round 2 minor finding 4: `_play_tone_and_capture`'s own
        # assertion message ("audio_output_start returned false for
        # BlackHole's own UID") reads as a genuine BlackHole failure, but
        # this call immediately follows destroying the test aggregate
        # device -- if CoreAudio has not finished settling after that
        # destroy (see `_destroy_aggregate`'s own settle sleep and its
        # measured-flake docstring above), the *symptom* here would be
        # indistinguishable from an actually-broken BlackHole. Re-raising
        # with that context prevents a future reader from chasing the
        # wrong cause.
        raise AssertionError(
            f"post-device-loss recovery: {exc} -- this call immediately follows destroying "
            "the test aggregate device (output_aggregate fixture); if this is a spurious "
            "'audio_output_start returned false', suspect insufficient CoreAudio settle time "
            "after the aggregate was destroyed, not a genuine BlackHole failure"
        ) from exc
    _assert_matches_for_expected_duration(result, tone, context="post-device-loss recovery: ")
