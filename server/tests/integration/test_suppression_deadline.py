"""T022 (Red) — the suppression deadline bounds FR-004, and release is
idempotent per FR-012 (contracts/playback-protocol.md, data-model.md §4
SuppressionWindow).

See test_no_self_transcription.py's module docstring for the shared design
choices (deterministic engine stubs at the call boundary, fake websocket
injected directly rather than FastAPI TestClient, why a real ClientSession
cannot be constructed here). The stub classes and helpers are intentionally
duplicated rather than imported from that module, matching the existing
`tests/unit/test_protocol_playback_complete.py` precedent of each test file
carrying its own small doubles.
"""

import asyncio
import json

import numpy as np
import pytest

from src.asr import asr_engine
from src.audio_buffer import AudioBuffer
from src.config import TTS_SAMPLE_RATE
from src.translation import translation_engine
from src.tts import tts_engine
from src.ws_handler import _handle_control, _process_speak_segment


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []

    async def send_text(self, data: str) -> None:
        self.sent_text.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)


class _FakeVad:
    def __init__(self) -> None:
        self.process_chunk_calls = 0
        self.reset_calls = 0
        self.flush_calls = 0

    def process_chunk(self, chunk):
        self.process_chunk_calls += 1
        return []

    def reset(self) -> None:
        self.reset_calls += 1

    def flush(self):
        self.flush_calls += 1
        return None


class _SpeakSessionStub:
    def __init__(self, session_id: str = "stub-session") -> None:
        self.session_id = session_id
        self.language = "auto"
        self.target_language = ""
        self.mode = "both"
        self.audio_buffer = AudioBuffer()
        self.vad = _FakeVad()
        self.speak_target_language = "English"
        self.tts_voice = "Ryan"
        self.voice_clone = False
        self.ref_audio: np.ndarray | None = None
        self.ref_text = ""
        self.mic_buffer = AudioBuffer()
        self.mic_vad = _FakeVad()
        self.mic_mode = "auto"
        self.mic_active = False
        self.tts_playing = False
        self.suppression = None
        self.transmission_active = False
        self._segment_counter = 0

    def next_segment_id(self) -> int:
        self._segment_counter += 1
        return self._segment_counter


def _install_deterministic_engines(
    monkeypatch: pytest.MonkeyPatch, synthesized_samples: int, chunk_samples: int = 4800,
) -> None:
    monkeypatch.setattr(asr_engine, "transcribe", lambda audio, language: "stub transcript")
    monkeypatch.setattr(translation_engine, "translate", lambda text, target: "stub translation")

    chunks: list[np.ndarray] = []
    remaining = synthesized_samples
    while remaining > 0:
        n = min(chunk_samples, remaining)
        chunks.append(np.zeros(n, dtype=np.float32))
        remaining -= n
    state = {"i": 0}

    def _start_synthesis(text, voice, lang_code, ref_audio, ref_text) -> int:
        state["i"] = 0
        return 1

    def _recv_chunk(timeout: float):
        i = state["i"]
        if i < len(chunks):
            state["i"] += 1
            return ("chunk", chunks[i], TTS_SAMPLE_RATE)
        return ("done", None, 0)

    monkeypatch.setattr(tts_engine, "start_synthesis", _start_synthesis)
    monkeypatch.setattr(tts_engine, "recv_chunk", _recv_chunk)


async def _wait_until(predicate, timeout: float, interval: float = 0.01) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(interval)

    await asyncio.wait_for(_poll(), timeout=timeout)


async def _start_segment(websocket, session, ws_lock, background_tasks) -> None:
    await _process_speak_segment(
        websocket, session, np.zeros(8000, dtype=np.float32), ws_lock, background_tasks,
    )


# --- FR-004 -------------------------------------------------------------

@pytest.mark.asyncio
async def test_deadline_releases_suppression_when_playback_complete_never_arrives(monkeypatch):
    """FR-004: a client that never sends playback_complete costs at most the
    deadline (synthesized_samples/TTS_SAMPLE_RATE + 1.0s), never forever.

    Red today, but *not* because suppression currently hangs forever --
    the opposite: today's code (ws_handler.py:381-385) clears tts_playing
    unconditionally the instant transmission ends, which for this
    near-instant stub is almost immediately, long before any deadline
    concept would fire. `session.suppression` does not exist at all on the
    current ClientSession, so `session.suppression is not None` never
    becomes true and the first _wait_until above times out -- this is red
    on 'the deadline mechanism exists', not on 'suppression is unbounded'.
    """
    synthesized_samples = 2_400  # 0.1s -> deadline = start + 0.1 + 1.0 = 1.1s
    _install_deterministic_engines(monkeypatch, synthesized_samples)

    websocket = _FakeWebSocket()
    session = _SpeakSessionStub()
    ws_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()

    await _start_segment(websocket, session, ws_lock, background_tasks)
    await _wait_until(lambda: session.suppression is not None, timeout=2.0)
    assert session.tts_playing is True

    # Client silently drops playback_complete. Suppression must still clear
    # on its own, bounded by the deadline.
    await _wait_until(lambda: session.tts_playing is False, timeout=3.0)

    # The session recovers: a later segment is processed normally, not
    # wedged by leftover suppression state.
    sent_before = len(websocket.sent_text)
    await _start_segment(websocket, session, ws_lock, background_tasks)
    assert len(websocket.sent_text) > sent_before


# --- FR-012 ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_playback_complete_delivered_twice_is_not_an_error(monkeypatch):
    """FR-012: a duplicate playback_complete for an already-released segment
    is ignored, not an error -- no error frame, no exception, no double
    mutation."""
    _install_deterministic_engines(monkeypatch, synthesized_samples=2_400)

    websocket = _FakeWebSocket()
    session = _SpeakSessionStub()
    ws_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()

    await _start_segment(websocket, session, ws_lock, background_tasks)
    await _wait_until(lambda: session.suppression is not None, timeout=2.0)
    segment_id = session.suppression.segment_id

    payload = json.dumps({"type": "playback_complete", "segment_id": segment_id})
    await _handle_control(websocket, session, payload, ws_lock, background_tasks)
    assert session.tts_playing is False
    sent_before = list(websocket.sent_text)

    await _handle_control(websocket, session, payload, ws_lock, background_tasks)  # duplicate
    assert session.tts_playing is False
    assert websocket.sent_text == sent_before, "duplicate playback_complete must not send an error frame"


@pytest.mark.asyncio
async def test_playback_complete_with_boolean_segment_id_does_not_release(monkeypatch):
    """FR-012 edge case: `bool` is a subclass of `int` in Python (`True ==
    1`), so a `{"segment_id": true}` frame must not be treated as matching
    segment 1's suppression window -- `_handle_control`'s `isinstance(x,
    int)` guard must exclude `bool` explicitly, not rely on `isinstance`
    alone."""
    _install_deterministic_engines(monkeypatch, synthesized_samples=2_400)

    websocket = _FakeWebSocket()
    session = _SpeakSessionStub()
    ws_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()

    await _start_segment(websocket, session, ws_lock, background_tasks)
    await _wait_until(lambda: session.suppression is not None, timeout=2.0)
    assert session.suppression.segment_id == 1

    payload = json.dumps({"type": "playback_complete", "segment_id": True})
    await _handle_control(websocket, session, payload, ws_lock, background_tasks)

    assert session.tts_playing is True, "a boolean segment_id must not release segment 1"
    assert session.suppression.released is False


@pytest.mark.asyncio
async def test_late_playback_complete_after_deadline_does_not_disturb_next_segment(monkeypatch):
    """FR-012: a playback_complete that arrives after its segment's deadline
    already released it is ignored -- critically, it must not release a
    *different*, currently-suppressing segment that happens to be active by
    the time the stale message arrives (segment_id must gate the release,
    not just 'is something suppressed right now')."""
    _install_deterministic_engines(monkeypatch, synthesized_samples=2_400)  # deadline ~1.1s

    websocket = _FakeWebSocket()
    session = _SpeakSessionStub()
    ws_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()

    await _start_segment(websocket, session, ws_lock, background_tasks)
    await _wait_until(lambda: session.suppression is not None, timeout=2.0)
    stale_segment_id = session.suppression.segment_id

    await _wait_until(lambda: session.tts_playing is False, timeout=3.0)  # deadline fires

    await _start_segment(websocket, session, ws_lock, background_tasks)
    await _wait_until(
        lambda: session.suppression is not None and session.suppression.segment_id != stale_segment_id,
        timeout=2.0,
    )
    assert session.tts_playing is True

    stale_payload = json.dumps({"type": "playback_complete", "segment_id": stale_segment_id})
    await _handle_control(websocket, session, stale_payload, ws_lock, background_tasks)

    assert session.tts_playing is True, "a stale playback_complete must not release a newer segment"
    assert session.suppression.segment_id != stale_segment_id
    assert session.suppression.released is False
