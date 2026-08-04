"""Hermetic tests for conftest.py's internal contracts (Critic rounds 1-2 on T003/T004).

Everything in this module is hardware-independent: `sounddevice` calls are
monkeypatched to return fabricated device tables and streams, so these
tests run (and must pass) on any machine, with or without BlackHole
installed, and always run — never skipped, unlike test_capture_fixture.py's
hardware-backed proofs.

Each test here exists because a specific Critic finding showed the
real-hardware discrimination proof in test_capture_fixture.py was not
enough on its own: several properties of `conftest.py`'s *internals* (a
frozen device table, an empty enumeration, a skip decided by substring, a
capture that could silently return less than requested) would not show up
as a real-hardware test failure at all, or would show up as a *skip*
rather than a failure.
"""

import time

import numpy as np
import pytest
import sounddevice as sd

from integration import conftest


def _fake_device(name: str, *, input_channels: int = 0, output_channels: int = 0) -> dict:
    return {
        "name": name,
        "max_input_channels": input_channels,
        "max_output_channels": output_channels,
        "default_samplerate": 48_000.0,
    }


_DEVICE_LIST_A = [
    _fake_device("MacBook Pro 마이크", input_channels=1),
    _fake_device("BlackHole 2ch", input_channels=2, output_channels=2),
]

# Same devices, BlackHole moved to a different index -- simulates a real
# device-list reorder after a hot-plug (research.md D8's stated reason for
# resolving by name, not index).
_DEVICE_LIST_B = [
    _fake_device("'YUNHO의 iPhone' 마이크", input_channels=1),
    _fake_device("MacBook Pro 마이크", input_channels=1),
    _fake_device("BlackHole 2ch", input_channels=2, output_channels=2),
]


class _FakeInputStream:
    """Stand-in for `sd.InputStream`: on `__enter__`, synchronously invokes
    the real callback with `total_frames_to_deliver` frames of zeros (all
    at once), then behaves like an open stream until `__exit__`. Passing 0
    delivers nothing, ever -- for exercising the timeout path."""

    def __init__(self, total_frames_to_deliver: int):
        self.total_frames_to_deliver = total_frames_to_deliver

    def __call__(self, *, device, channels, samplerate, dtype, callback, **kwargs):
        self._callback = callback
        self._channels = channels
        return self

    def __enter__(self):
        if self.total_frames_to_deliver > 0:
            chunk = np.zeros((self.total_frames_to_deliver, self._channels), dtype=np.float32)
            self._callback(chunk, self.total_frames_to_deliver, None, None)
        return self

    def __exit__(self, *exc_info):
        return False


class _FailingInputStream:
    """Stand-in for `sd.InputStream` that raises on construction, like a
    device that is present in the listing but cannot actually be opened
    (e.g. already in use)."""

    def __call__(self, **kwargs):
        raise sd.PortAudioError("device busy")


# ---------------------------------------------------------------------------
# Enumeration freshness (Critic round 1 finding 6, carried over to PortAudio)
# ---------------------------------------------------------------------------


def test_enumeration_reflects_the_most_recent_device_list(monkeypatch):
    """A hardcoded {"BlackHole 2ch": 0, ...} table would return the same
    index forever. This proves resolution tracks a change in what
    `sd.query_devices()` reports between two calls -- which a frozen table
    cannot do."""
    monkeypatch.setattr(conftest.sd, "query_devices", lambda: _DEVICE_LIST_A)
    first = conftest._list_portaudio_input_devices()
    assert first["BlackHole 2ch"] == 1

    monkeypatch.setattr(conftest.sd, "query_devices", lambda: _DEVICE_LIST_B)
    second = conftest._list_portaudio_input_devices()
    assert second["BlackHole 2ch"] == 2


# ---------------------------------------------------------------------------
# Empty enumeration is a failure, not a skip (Critic round 1 finding 4)
# ---------------------------------------------------------------------------


def test_empty_enumeration_raises_enumeration_error_never_device_absent(monkeypatch):
    monkeypatch.setattr(conftest.sd, "query_devices", lambda: [])

    with pytest.raises(conftest.EnumerationError) as excinfo:
        conftest.record_from_device("BlackHole 2ch", 0.1)

    assert not isinstance(excinfo.value, conftest.DeviceAbsentError)


# ---------------------------------------------------------------------------
# Skip vs. fail decided by exception type (Critic round 1 finding 5)
# ---------------------------------------------------------------------------


def test_device_absent_error_is_skipped_regardless_of_its_message():
    with pytest.raises(pytest.skip.Exception):
        conftest._skip_if_absent_else_raise(conftest.DeviceAbsentError("anything at all"))


def test_a_message_containing_not_present_does_not_skip_unless_it_is_actually_device_absent_error():
    """A genuine capture failure whose supplied text happens to contain the
    phrase "not present" must still be a real failure, because it is not a
    `DeviceAbsentError`.

    Deliberately does not use `pytest.raises(DeviceCaptureError)`: a
    substring-matching mutant makes `_skip_if_absent_else_raise` call
    `pytest.skip()` instead, which raises `Skipped` -- a type
    `pytest.raises(DeviceCaptureError)` does not match, so it re-raises
    `Skipped` uncaught and the *test itself* silently reports as SKIPPED
    rather than FAILED (confirmed in Critic round 1 by deliberately
    reintroducing the substring-matching mutant with the `pytest.raises`
    form of this test). Catching broadly and calling `pytest.fail()`
    explicitly ensures any wrong outcome -- including a stray skip -- is a
    loud failure instead.
    """
    misleading = conftest.DeviceCaptureError(
        "recording from 'BlackHole 2ch' failed: device reports input hardware not present"
    )

    try:
        conftest._skip_if_absent_else_raise(misleading)
    except conftest.DeviceCaptureError:
        return
    except BaseException as exc:  # noqa: BLE001 - deliberately broad, see docstring
        pytest.fail(f"expected a DeviceCaptureError to propagate, got {type(exc).__name__}: {exc}")
    pytest.fail("expected _skip_if_absent_else_raise to raise, it returned normally")


def test_stream_open_failure_propagates_end_to_end_not_skipped(monkeypatch):
    """Same defect, exercised through the real `record_from_device` path: a
    device present in the listing that cannot actually be opened is a real
    failure, never a skip."""
    monkeypatch.setattr(conftest.sd, "query_devices", lambda: _DEVICE_LIST_A)
    monkeypatch.setattr(conftest.sd, "InputStream", _FailingInputStream())

    with pytest.raises(conftest.DeviceCaptureError) as excinfo:
        conftest.record_from_device("BlackHole 2ch", 0.1)

    assert not isinstance(excinfo.value, conftest.DeviceAbsentError)


# ---------------------------------------------------------------------------
# Exact-duration invariant (Critic round 1 finding 3 / round 2 finding 5):
# capture either returns exactly what was requested, or raises. There is no
# partial-success path to bound with a floor.
# ---------------------------------------------------------------------------


def test_successful_capture_delivers_exactly_the_requested_duration(monkeypatch):
    monkeypatch.setattr(conftest.sd, "query_devices", lambda: _DEVICE_LIST_A)
    sample_rate, duration_s, channels = 48_000, 0.2, 2
    frame_count = int(round(duration_s * sample_rate))
    monkeypatch.setattr(conftest.sd, "InputStream", _FakeInputStream(frame_count))

    result = conftest.record_from_device(
        "BlackHole 2ch", duration_s, sample_rate=sample_rate, channels=channels
    )

    assert result.actual_duration_s == pytest.approx(duration_s)
    assert result.requested_duration_s == duration_s
    assert result.samples.shape == (frame_count, channels)


def test_capture_that_never_receives_enough_frames_times_out_rather_than_returning_partial_data(monkeypatch):
    """Before this design (the ffmpeg path), a capture that fell short of
    the request could return successfully with less audio than asked for.
    Under PortAudio, a capture that never accumulates enough frames must
    raise -- it must never return a `CaptureResult` with
    `actual_duration_s < requested_duration_s`."""
    monkeypatch.setattr(conftest.sd, "query_devices", lambda: _DEVICE_LIST_A)
    monkeypatch.setattr(conftest.sd, "InputStream", _FakeInputStream(total_frames_to_deliver=0))
    # Keep the hermetic test fast: the timeout margin is a module constant,
    # not a parameter, so shrink it for the duration of this test rather
    # than actually waiting out the real ~5s default.
    monkeypatch.setattr(conftest, "_CAPTURE_TIMEOUT_MARGIN_S", 0.05)

    start = time.monotonic()
    with pytest.raises(conftest.DeviceCaptureError, match="did not accumulate"):
        conftest.record_from_device("BlackHole 2ch", 0.05)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"timeout should be bounded by the shrunk margin, took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Returned array is writeable (Critic round 1 finding 9)
# ---------------------------------------------------------------------------


def test_returned_samples_array_is_writeable(monkeypatch):
    monkeypatch.setattr(conftest.sd, "query_devices", lambda: _DEVICE_LIST_A)
    sample_rate, duration_s, channels = 48_000, 0.05, 2
    frame_count = int(round(duration_s * sample_rate))
    monkeypatch.setattr(conftest.sd, "InputStream", _FakeInputStream(frame_count))

    result = conftest.record_from_device(
        "BlackHole 2ch", duration_s, sample_rate=sample_rate, channels=channels
    )

    assert result.samples.flags.writeable
    result.samples[0, 0] = 1.0  # would raise ValueError on a read-only view


# ---------------------------------------------------------------------------
# blackhole_expected_gain (Critic round 1 finding 2 / round 2 finding 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("volume_scalar", "expected_gain"),
    [
        (1.0, 1.0),  # unity: 0 dB
        (0.0, pytest.approx(10 ** (-64.0 / 20.0))),  # fully attenuated: -64 dB
        (0.1875, pytest.approx(0.0025119, rel=1e-4)),  # measured on this machine, Critic round 1
    ],
)
def test_blackhole_expected_gain_matches_the_documented_linear_db_model(volume_scalar, expected_gain):
    assert conftest.blackhole_expected_gain(volume_scalar) == expected_gain


# ---------------------------------------------------------------------------
# read_core_audio_volume_scalar sanity-bounds its own output (Critic round
# 3 finding 3): a ctypes call can report success while returning garbage.
# Fully hermetic -- `_core_audio_device_ids`/`_core_audio_device_name` are
# also monkeypatched, so this needs no real device and no real CoreAudio
# property access, only the (harmless, always-available) library load.
# ---------------------------------------------------------------------------


def _patch_fake_single_device(monkeypatch, name: str = "Fake Device") -> None:
    monkeypatch.setattr(conftest, "_core_audio_device_ids", lambda core_audio: [1])
    monkeypatch.setattr(
        conftest, "_core_audio_device_name", lambda core_audio, core_foundation, device_id: name
    )


def test_read_core_audio_volume_scalar_rejects_an_out_of_range_value(monkeypatch):
    """A `status == 0` ctypes read is not proof the value is real --
    CoreAudio documents this property as ranging over [0.0, 1.0], so a
    superficially-successful read outside that range is treated as
    unreadable rather than trusted."""
    _patch_fake_single_device(monkeypatch)

    def fake_get_property(core_audio, object_id, selector, scope, element, buf, size):
        assert selector == conftest._kAudioDevicePropertyVolumeScalar
        buf.value = 4.2  # out of the documented [0.0, 1.0] range
        return 0  # reports success

    monkeypatch.setattr(conftest, "_core_audio_get_property", fake_get_property)

    assert conftest.read_core_audio_volume_scalar("Fake Device", "output") is None


def test_read_core_audio_volume_scalar_accepts_an_in_range_value(monkeypatch):
    _patch_fake_single_device(monkeypatch)

    def fake_get_property(core_audio, object_id, selector, scope, element, buf, size):
        assert selector == conftest._kAudioDevicePropertyVolumeScalar
        buf.value = 0.42
        return 0

    monkeypatch.setattr(conftest, "_core_audio_get_property", fake_get_property)

    assert conftest.read_core_audio_volume_scalar("Fake Device", "output") == pytest.approx(0.42, abs=1e-6)


def test_read_core_audio_volume_scalar_returns_none_when_the_property_read_fails(monkeypatch):
    _patch_fake_single_device(monkeypatch)

    def fake_get_property(core_audio, object_id, selector, scope, element, buf, size):
        return -1  # a real OSStatus error code, not zero

    monkeypatch.setattr(conftest, "_core_audio_get_property", fake_get_property)

    assert conftest.read_core_audio_volume_scalar("Fake Device", "output") is None
