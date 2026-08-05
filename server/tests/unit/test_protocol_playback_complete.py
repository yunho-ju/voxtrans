"""T020 (Red) — `MessageType.PLAYBACK_COMPLETE` and its reception in
`_handle_control` (contracts/playback-protocol.md).

Scope for this round only: the message type exists, parses per the contract,
and `_handle_control` receives it harmlessly (validation + logging, no error
frame, no session-state mutation). Suppression release wiring is a later
round (T028) and is deliberately not exercised here.

Pure protocol/dispatch test: no audio devices, no models. A stub session
object stands in for `ClientSession` — constructing the real one runs
`torch.hub.load` for the Silero VAD via `VoiceActivityDetector.__init__`,
which is forbidden while the dev server holds the real model cache during
manual testing.
"""

import asyncio

import pytest

from src.protocol import MessageType, parse_control_message
from src.ws_handler import _handle_control


class _FakeWebSocket:
    """Records every text frame sent, without opening a real socket."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, data: str) -> None:
        self.sent.append(data)


class _StubSession:
    """Minimal session double exposing only what a correctly-scoped
    playback_complete branch may touch this round: nothing. Any attribute
    access beyond session_id would be a signal that release logic (T028)
    leaked into this round.
    """

    def __init__(self) -> None:
        self.session_id = "stub-session"
        self.mic_active = False
        self.tts_playing = False
        # `_release_suppression` (ws_handler.py) reads `session.suppression`
        # directly (Critic T032 minor 1 -- a `getattr(..., None)` fallback
        # used to stand in for this so this stub didn't need the attribute
        # at all, which meant a real-session field rename could silently
        # degrade release into a no-op with nothing here to notice). `None`
        # is exactly what a real `ClientSession` starts with
        # (`session.py`), so `test_handle_control_playback_complete_...`
        # below now exercises the real early-return gate rather than an
        # attribute-missing shortcut.
        self.suppression = None


async def _run_control(data: str) -> tuple[_FakeWebSocket, _StubSession]:
    websocket = _FakeWebSocket()
    session = _StubSession()
    await _handle_control(websocket, session, data, asyncio.Lock(), set())
    return websocket, session


# --- wire shape (contract) --------------------------------------------------

def test_parse_control_message_matches_playback_complete_contract_shape() -> None:
    # contracts/playback-protocol.md: {"type": "playback_complete", "segment_id": 7}
    parsed = parse_control_message('{"type": "playback_complete", "segment_id": 7}')

    assert parsed is not None
    assert parsed["type"] == MessageType.PLAYBACK_COMPLETE.value
    assert parsed["segment_id"] == 7


# --- _handle_control dispatch -----------------------------------------------

@pytest.mark.asyncio
async def test_handle_control_playback_complete_for_unknown_segment_sends_no_error() -> None:
    # FR-012 (this round's half): an unrecognized segment_id must not produce
    # a protocol error, unlike a genuinely unrecognized message type.
    websocket, _ = await _run_control('{"type": "playback_complete", "segment_id": 999}')

    assert websocket.sent == []


@pytest.mark.asyncio
async def test_handle_control_playback_complete_missing_segment_id_does_not_raise() -> None:
    # Malformed input stays harmless: no exception, no error frame.
    websocket, _ = await _run_control('{"type": "playback_complete"}')

    assert websocket.sent == []


@pytest.mark.asyncio
async def test_handle_control_playback_complete_does_not_mutate_session_state() -> None:
    # Release wiring (suppression clearing) is T028, a later round; this
    # round must be inert on session state.
    websocket, session = await _run_control('{"type": "playback_complete", "segment_id": 1}')

    assert session.mic_active is False
    assert session.tts_playing is False


@pytest.mark.asyncio
async def test_handle_control_unrecognized_type_still_sends_error() -> None:
    # Control case: proves the dispatch else-branch is still reachable and
    # still errors, so the no-error assertions above are actually exercising
    # the playback_complete branch rather than a catch-all that swallows
    # every message type.
    websocket, _ = await _run_control('{"type": "not_a_real_type"}')

    assert len(websocket.sent) == 1
    assert "Unknown message type" in websocket.sent[0]
