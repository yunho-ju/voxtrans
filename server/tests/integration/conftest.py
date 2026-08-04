"""Audio-capture fixture for hardware-backed integration tests (T003).

**Critic round 2 design change**: capture is `sounddevice` (PortAudio), not
`ffmpeg -f avfoundation`. The ffmpeg path lost ~20% of every recording in
~39ms splices (Critic round 1 finding 3), which collapsed the matched
filter's entire dynamic range and produced a content check that could not
tell a true match from an unrelated recording (Critic round 2 blocker).
PortAudio does not have this defect: it accumulates real callback frames
until it has exactly as many as requested rather than stopping on a
wall-clock timer, so there is nothing to splice. Measured directly against
real hardware, 25/25 captures returned exactly the requested frame count
(ratio 1.000 every time; see evidence/T003-T004.md "Critic round 2 fixes").
`sounddevice` was already a declared dependency (`requirements-dev.txt`)
and already used to *play* reference signals in test_capture_fixture.py;
this uses it to *record* as well.

Devices are still resolved by *name*, not index, for the same reason as
before (research.md D8): index stability is not guaranteed across device
hot-plug. `_list_portaudio_input_devices` calls `sd.query_devices()` fresh
on every call rather than caching a table in Python, mirroring Critic
round 1 finding 6 for the previous ffmpeg-based enumeration — though see
that function's docstring for why this fixture deliberately does *not*
force PortAudio's own device table to rescan (`sd._terminate()`/
`sd._initialize()`), which was tried and found to break concurrent
playback in the same process.

Every test that needs a specific device must be able to run on a machine
that does not have it (e.g. no BlackHole installed) without failing the
suite, so the fixture skips rather than fails when the name is not found —
but only when the name is genuinely absent from an otherwise-plausible
listing. `DeviceAbsentError` and `EnumerationError` (a broken or empty
listing) are distinct exception types precisely so that "this one device
isn't here" and "the enumeration mechanism itself is broken" can never be
confused with each other by accident — see `_skip_if_absent_else_raise` and
Critic round 1 findings 4 and 5 in `specs/002-audio-routing/evidence/T003-T004.md`.

Also exposes:

- `audio_device_lock`: a plain interprocess lock (a `flock` on a fixed path
  in the system temp directory) that any test recording from or playing to
  a real device should hold for its full duration. Audio devices are
  shared, stateful hardware — two tests driving them at once will
  contaminate each other's recordings — and this lock is how
  test_capture_fixture.py guarantees that cannot happen, including across
  separate pytest processes.
- `read_core_audio_volume_scalar` / `blackhole_expected_gain`: read a named
  device's `kAudioDevicePropertyVolumeScalar` directly via CoreAudio
  (`ctypes`, no `pyobjc`) and convert it to BlackHole's documented gain
  model. This exists so a level check can be anchored to *measured hardware
  state* rather than an absolute constant tied to one volume-slider
  position (Critic round 1 finding 2) or to nothing at all, which cannot
  tell a working path from one delivering audio 60dB too quiet (Critic
  round 2 finding 2).
"""

import contextlib
import ctypes
import ctypes.util
import fcntl
import tempfile
import time
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pytest
import sounddevice as sd

# How long past the requested recording duration a capture is given to
# accumulate the requested number of frames before it is judged hung and
# aborted, rather than waited on forever. Generous relative to the observed
# ~0.2-0.3s PortAudio stream startup latency.
_CAPTURE_TIMEOUT_MARGIN_S = 5.0

_DEVICE_LOCK_PATH = Path(tempfile.gettempdir()) / "voxtrans-audio-device.lock"


class DeviceCaptureError(RuntimeError):
    """Base: something went wrong trying to capture from a real device.

    Always a real failure. The one exception this is not raised as is
    `DeviceAbsentError` below, which is the only case `record_input_device`
    treats as a skip.
    """


class DeviceAbsentError(DeviceCaptureError):
    """The named device was not found among a plausible (non-empty) device
    enumeration. This is ambient machine state, not a defect — the only
    `DeviceCaptureError` subtype `record_input_device` converts to a skip.
    """


class EnumerationError(DeviceCaptureError):
    """PortAudio's device enumeration itself returned something
    implausible — most concretely, zero input devices, which is not a
    legitimate state for any Mac (there is always at least a built-in
    microphone). This is a broken enumeration, never treated as "device
    absent": a test that skips on this would go green while genuinely
    unable to observe anything (Critic round 1 finding 4).
    """


def _list_portaudio_input_devices() -> dict[str, int]:
    """Return {device name: PortAudio index} for every currently visible
    input-capable device, resolved fresh (re-queried, never cached in
    Python) on every call.

    Deliberately does **not** call `sd._terminate()`/`sd._initialize()` to
    force PortAudio's own device table to rescan: that tears down every
    active PortAudio stream process-wide, including a reference signal
    being played by `sd.play()` in another thread of the *same* process —
    exactly the play-while-capture pattern this test suite depends on.
    Confirmed directly: with the terminate/reinitialize call in place, a
    concurrent `sd.play()` failed silently (`PaMacCore (AUHAL) Error ...
    err='-50'`) and the resulting capture was all zeros. `sd.query_devices()`
    alone still reflects PortAudio's live device table; what it does not
    guarantee is that PortAudio's *own* internal table is rescanned on a
    genuine hot-plug without a fresh process — a known limitation, not
    something this fixture's own code can fix without reintroducing the
    bug above (see evidence/T003-T004.md "Critic round 2 fixes").
    """
    devices: dict[str, int] = {}
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0 and info["name"]:
            devices[info["name"]] = index
    return devices


def _resolve_portaudio_output_index(device_name: str) -> int | None:
    """Return the PortAudio index of `device_name`'s *output* side, or
    `None` if it has none or is not present. See `_list_portaudio_input_devices`
    for why this does not force a PortAudio-level device rescan."""
    for index, info in enumerate(sd.query_devices()):
        if info["name"] == device_name and info["max_output_channels"] > 0:
            return index
    return None


class CaptureResult(NamedTuple):
    """What `record_from_device` returns.

    `actual_duration_s` is exposed explicitly because a caller building a
    duration assertion (e.g. a future SC-001 test) should read the real
    number rather than assume the request was honored — even though, on
    this PortAudio path, it always has been so far (see module docstring).
    """

    samples: np.ndarray
    sample_rate: int
    requested_duration_s: float
    actual_duration_s: float


def record_from_device(
    device_name: str,
    duration_s: float,
    sample_rate: int = 48_000,
    channels: int = 2,
) -> CaptureResult:
    """Record `duration_s` seconds from the named PortAudio input device.

    Raises `EnumerationError` if the device listing itself is empty (never
    a legitimate "device absent" signal — every Mac has a built-in
    microphone). Raises `DeviceAbsentError` if the listing is non-empty but
    does not contain `device_name`. Raises the `DeviceCaptureError` base
    directly if the stream cannot be opened, or if it does not finish
    accumulating the requested frames within a bounded timeout.

    There is deliberately no "partial success" / duration-floor path:
    either this returns a `CaptureResult` with `actual_duration_s` exactly
    equal to `requested_duration_s` (by construction — the loop below only
    exits normally once `frame_count` frames have arrived), or it raises.
    This is the direct fix for Critic round 1 finding 3 / round 2 finding
    5: the previous `ffmpeg -t`-based capture silently returned ~80% of
    the requested duration because it stopped on a wall-clock deadline
    regardless of how much audio had actually arrived; this accumulates
    real callback data until it has enough, so there is nothing to be
    silently short of. Confirmed against real hardware: 25/25 captures
    returned `actual_duration_s / requested_duration_s == 1.000` exactly
    (see evidence/T003-T004.md "Critic round 2 fixes").
    """
    devices = _list_portaudio_input_devices()
    if not devices:
        raise EnumerationError(
            "PortAudio reported zero input devices, which is not plausible "
            "on any Mac — treating this as a broken enumeration, not a "
            "legitimately absent device"
        )
    index = devices.get(device_name)
    if index is None:
        raise DeviceAbsentError(
            f"PortAudio input device {device_name!r} not present "
            f"(currently visible: {sorted(devices)})"
        )

    frame_count = int(round(duration_s * sample_rate))
    chunks: list[np.ndarray] = []

    def _callback(indata: np.ndarray, _frames: int, _time_info, _status) -> None:
        chunks.append(indata.copy())

    try:
        stream = sd.InputStream(
            device=index,
            channels=channels,
            samplerate=sample_rate,
            dtype="float32",
            callback=_callback,
        )
    except sd.PortAudioError as exc:
        raise DeviceCaptureError(
            f"failed to open an input stream on {device_name!r}: {exc}"
        ) from exc

    timeout_s = duration_s + _CAPTURE_TIMEOUT_MARGIN_S
    deadline = time.monotonic() + timeout_s
    try:
        with stream:
            while sum(len(chunk) for chunk in chunks) < frame_count:
                if time.monotonic() > deadline:
                    raise DeviceCaptureError(
                        f"recording from {device_name!r} did not accumulate "
                        f"{duration_s:.3f}s of audio within {timeout_s:.3f}s"
                    )
                time.sleep(0.005)
    except sd.PortAudioError as exc:
        raise DeviceCaptureError(
            f"PortAudio error while recording from {device_name!r}: {exc}"
        ) from exc

    samples = np.concatenate(chunks, axis=0)[:frame_count]
    actual_duration_s = len(samples) / sample_rate
    return CaptureResult(samples, sample_rate, duration_s, actual_duration_s)


def _skip_if_absent_else_raise(exc: DeviceCaptureError) -> None:
    """The single place that decides skip vs. fail for `record_input_device`.

    Only `DeviceAbsentError` skips. Every other `DeviceCaptureError`
    (including `EnumerationError`) propagates as a real failure. Deciding
    this by exception *type* rather than by substring-matching `str(exc)`
    is what fixes Critic round 1 finding 5.
    """
    if isinstance(exc, DeviceAbsentError):
        pytest.skip(str(exc))
    raise exc


@pytest.fixture
def record_input_device() -> Callable[..., CaptureResult]:
    """Factory fixture: `record_input_device(name, duration_s) -> CaptureResult`.

    Skips the test if the named input device is not present on this
    machine, rather than failing it — device presence is ambient machine
    state (constraint 1 in the task brief), not something a test can assert
    on. Any other capture problem (a broken enumeration, a stream that
    won't open, a timeout) is a real failure, never a skip.
    """

    def _record(device_name: str, duration_s: float = 1.0) -> CaptureResult:
        try:
            return record_from_device(device_name, duration_s)
        except DeviceCaptureError as exc:
            _skip_if_absent_else_raise(exc)
            raise  # unreachable: _skip_if_absent_else_raise always raises

    return _record


@pytest.fixture
def audio_device_lock() -> Callable[[], contextlib.AbstractContextManager]:
    """Returns a context-manager factory that holds an exclusive interprocess
    lock on a fixed path while inside the `with` block. Callers wrap the
    entire span of device use (not just one recording call) so that no two
    tests — in this process or another — can drive audio devices at once.
    """

    @contextlib.contextmanager
    def _lock():
        _DEVICE_LOCK_PATH.touch(exist_ok=True)
        with open(_DEVICE_LOCK_PATH, "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    return _lock


# ---------------------------------------------------------------------------
# CoreAudio volume-scalar reading (Critic round 1 finding 2 / round 2 finding 2)
# ---------------------------------------------------------------------------


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), byteorder="big")


class _AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


_kAudioObjectSystemObject = 1
_kAudioHardwarePropertyDevices = _fourcc("dev#")
_kAudioObjectPropertyScopeGlobal = _fourcc("glob")
_kAudioObjectPropertyElementMain = 0
_kAudioObjectPropertyName = _fourcc("lnam")
_kAudioDevicePropertyVolumeScalar = _fourcc("volm")
_kAudioDevicePropertyStreams = _fourcc("stm#")
_SCOPE_FOURCC = {"input": _fourcc("inpt"), "output": _fourcc("outp")}
_kCFStringEncodingUTF8 = 0x08000100


def _core_audio_get_property(
    core_audio, object_id: int, selector: int, scope: int, element: int, buf, size: int
) -> int:
    address = _AudioObjectPropertyAddress(selector, scope, element)
    out_size = ctypes.c_uint32(size)
    return core_audio.AudioObjectGetPropertyData(
        ctypes.c_uint32(object_id),
        ctypes.byref(address),
        ctypes.c_uint32(0),
        None,
        ctypes.byref(out_size),
        ctypes.byref(buf),
    )


def _core_audio_device_ids(core_audio) -> list[int]:
    address = _AudioObjectPropertyAddress(
        _kAudioHardwarePropertyDevices, _kAudioObjectPropertyScopeGlobal, _kAudioObjectPropertyElementMain
    )
    size = ctypes.c_uint32(0)
    status = core_audio.AudioObjectGetPropertyDataSize(
        ctypes.c_uint32(_kAudioObjectSystemObject), ctypes.byref(address), ctypes.c_uint32(0), None, ctypes.byref(size)
    )
    if status != 0 or size.value == 0:
        return []
    count = size.value // ctypes.sizeof(ctypes.c_uint32)
    ids = (ctypes.c_uint32 * count)()
    status = _core_audio_get_property(
        core_audio,
        _kAudioObjectSystemObject,
        _kAudioHardwarePropertyDevices,
        _kAudioObjectPropertyScopeGlobal,
        _kAudioObjectPropertyElementMain,
        ids,
        size.value,
    )
    return list(ids) if status == 0 else []


def _core_audio_device_name(core_audio, core_foundation, device_id: int) -> str | None:
    cf_string_ref = ctypes.c_void_p()
    status = _core_audio_get_property(
        core_audio,
        device_id,
        _kAudioObjectPropertyName,
        _kAudioObjectPropertyScopeGlobal,
        _kAudioObjectPropertyElementMain,
        cf_string_ref,
        ctypes.sizeof(cf_string_ref),
    )
    if status != 0 or not cf_string_ref.value:
        return None
    core_foundation.CFStringGetCStringPtr.restype = ctypes.c_char_p
    core_foundation.CFStringGetCStringPtr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    name_ptr = core_foundation.CFStringGetCStringPtr(cf_string_ref, _kCFStringEncodingUTF8)
    if name_ptr:
        return name_ptr.decode("utf-8")
    buf = ctypes.create_string_buffer(256)
    ok = core_foundation.CFStringGetCString(cf_string_ref, buf, 256, _kCFStringEncodingUTF8)
    return buf.value.decode("utf-8") if ok else None


# kAudioDevicePropertyVolumeScalar is documented by CoreAudio as ranging
# over [0.0, 1.0] (0 = fully attenuated, 1 = unity). A value outside that
# range cannot be a genuine reading of this property -- it means the
# ctypes marshalling produced garbage (wrong struct layout, misread
# memory, an uninitialized buffer) even though the call itself reported
# success (`status == 0`). Critic round 3 finding 3: a `status == 0` float
# was being trusted unconditionally; this bound rejects the class of
# corrupted-but-superficially-successful read it is possible to construct.
# It does NOT and cannot protect against a *plausible* wrong value (e.g. a
# test double that reports a genuine-looking but incorrect scalar) --
# that is what `_assert_recorded_level_is_plausible`'s own hermetic test
# in test_capture_fixture.py exists for, since no range check on the
# reader can substitute for testing the assertion logic that consumes it.
_VOLUME_SCALAR_VALID_RANGE = (0.0, 1.0)


def read_core_audio_volume_scalar(device_name: str, scope: str) -> float | None:
    """Read `kAudioDevicePropertyVolumeScalar` for `device_name` directly via
    CoreAudio (`ctypes`, no `pyobjc` dependency). `scope` is `"input"` or
    `"output"`.

    Returns `None` if the device cannot be found, the property is not
    implemented for that scope (some devices/scopes have no software volume
    control -- confirmed present on this machine for 3 of 4 visible devices,
    e.g. a physical microphone's output scope), or the property claims
    success but returns a value outside CoreAudio's documented [0.0, 1.0]
    range for it (see `_VOLUME_SCALAR_VALID_RANGE`). Callers should treat
    `None` as "cannot verify level" and scope whatever they skip as
    narrowly as possible — never let it silently swallow an unrelated
    assertion (Critic round 3 finding 2).
    """
    if scope not in _SCOPE_FOURCC:
        raise ValueError(f"scope must be 'input' or 'output', got {scope!r}")
    core_audio = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    core_foundation = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    for device_id in _core_audio_device_ids(core_audio):
        if _core_audio_device_name(core_audio, core_foundation, device_id) != device_name:
            continue
        value = ctypes.c_float(0)
        status = _core_audio_get_property(
            core_audio, device_id, _kAudioDevicePropertyVolumeScalar, _SCOPE_FOURCC[scope], 0, value, ctypes.sizeof(value)
        )
        if status != 0:
            return None
        low, high = _VOLUME_SCALAR_VALID_RANGE
        if not (low <= value.value <= high):
            return None
        return value.value
    return None


# ---------------------------------------------------------------------------
# Device UID lookup (added for T006, US3 input-device selection)
#
# `find_core_audio_device_uid` resolves a device's UID directly via CoreAudio,
# the same way `read_core_audio_volume_scalar` above resolves a device's
# volume: independently of whatever UID enumeration the application code
# under test provides. This matters specifically here (constraint 3 in the
# US3 task brief -- never gate a test with the mechanism it verifies): T008
# adds a new `audio_list_input_devices` Swift function, and a test that used
# that same function to discover which UID to select would not be able to
# tell "the device selection is broken" apart from "the enumeration is
# broken". Resolving the UID here, through a separate CoreAudio property
# read, keeps those two failures distinguishable.
# ---------------------------------------------------------------------------

_kAudioDevicePropertyDeviceUID = _fourcc("uid ")


def _core_audio_device_uid(core_audio, core_foundation, device_id: int) -> str | None:
    """Read `kAudioDevicePropertyDeviceUID` for `device_id` directly via
    CoreAudio, mirroring `_core_audio_device_name`'s CFString handling but
    for the UID property."""
    cf_string_ref = ctypes.c_void_p()
    status = _core_audio_get_property(
        core_audio,
        device_id,
        _kAudioDevicePropertyDeviceUID,
        _kAudioObjectPropertyScopeGlobal,
        _kAudioObjectPropertyElementMain,
        cf_string_ref,
        ctypes.sizeof(cf_string_ref),
    )
    if status != 0 or not cf_string_ref.value:
        return None
    core_foundation.CFStringGetCStringPtr.restype = ctypes.c_char_p
    core_foundation.CFStringGetCStringPtr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    uid_ptr = core_foundation.CFStringGetCStringPtr(cf_string_ref, _kCFStringEncodingUTF8)
    if uid_ptr:
        return uid_ptr.decode("utf-8")
    buf = ctypes.create_string_buffer(256)
    ok = core_foundation.CFStringGetCString(cf_string_ref, buf, 256, _kCFStringEncodingUTF8)
    return buf.value.decode("utf-8") if ok else None


def find_core_audio_device_uid(device_name: str) -> str | None:
    """Resolve `device_name`'s CoreAudio device UID directly (`ctypes`, no
    `pyobjc`), independently of any UID enumeration the application under
    test provides. Returns `None` if no device with that name is currently
    visible -- callers should treat that as ambient machine state (ie. skip,
    never fail), the same rule `record_input_device` already follows.
    """
    core_audio = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    core_foundation = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    for device_id in _core_audio_device_ids(core_audio):
        if _core_audio_device_name(core_audio, core_foundation, device_id) != device_name:
            continue
        return _core_audio_device_uid(core_audio, core_foundation, device_id)
    return None


# ---------------------------------------------------------------------------
# Independent input-device-set ground truth (Critic round 1 finding 4)
#
# `AudioCapture.swift`'s `audio_list_input_devices` filters devices by
# `kAudioDevicePropertyStreams` scoped to *input*. A test that only checks
# "the returned list is non-empty and well-formed" is satisfied just as well
# by a filter bug that enumerates every device (input and output alike) --
# `contracts/audio-devices.md`'s verification table requires the reported
# input name set to *match* an independent source, not merely overlap it.
# `list_core_audio_input_device_names` resolves that independent source: the
# same underlying CoreAudio property, read directly here rather than through
# anything `AudioCapture.swift` computes, so a bug in the Swift-side filter
# cannot also move this side of the comparison.
# ---------------------------------------------------------------------------


def _device_has_input_stream(core_audio, device_id: int) -> bool:
    address = _AudioObjectPropertyAddress(
        _kAudioDevicePropertyStreams, _SCOPE_FOURCC["input"], _kAudioObjectPropertyElementMain
    )
    size = ctypes.c_uint32(0)
    status = core_audio.AudioObjectGetPropertyDataSize(
        ctypes.c_uint32(device_id), ctypes.byref(address), ctypes.c_uint32(0), None, ctypes.byref(size)
    )
    return status == 0 and size.value > 0


def list_core_audio_input_device_names() -> set[str]:
    """Every currently-visible device name whose *input* scope has at least
    one stream, resolved directly via CoreAudio -- independent of
    `audio_list_input_devices` (the Swift function this is compared
    against)."""
    core_audio = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    core_foundation = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    names: set[str] = set()
    for device_id in _core_audio_device_ids(core_audio):
        if not _device_has_input_stream(core_audio, device_id):
            continue
        name = _core_audio_device_name(core_audio, core_foundation, device_id)
        if name:
            names.add(name)
    return names


def blackhole_expected_gain(volume_scalar: float) -> float:
    """BlackHole's documented volume-scalar-to-gain model: linear dB over
    `[-64, 0]` as the scalar ranges over `[0, 1]`.

    Empirically confirmed to match measured loopback attenuation to 4
    significant figures (Critic round 1 evidence: scalar 0.1875 ->
    predicted gain 0.0025119, measured 0.0025115-0.0025119 across three
    playback amplitudes).
    """
    db = -64.0 * (1.0 - volume_scalar)
    return 10.0 ** (db / 20.0)
