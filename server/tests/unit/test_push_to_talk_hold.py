"""Push-to-talk commits on release, never mid-hold.

The hold *is* the utterance: while the key is held nothing is segmented,
transcribed, or spoken, and the whole hold is committed as one segment when
the key comes up. This is the contract every other PTT translator implements.

Why it matters here, and why these are regression tests rather than unit
trivia: `_handle_speak_audio` returns early (discarding, not queueing) while
`tts_playing` is set. With mid-hold VAD splitting, a pause in a long sentence
ended a segment, TTS for that first half began *while the user was still
talking*, and every mic frame after that was dropped -- the tail of the
sentence vanished with no error anywhere. Buffering the hold removes the
overlap that made the drop reachable.

Pure buffering/dispatch tests: no audio devices, no models. A stub session
stands in for `ClientSession` -- constructing the real one runs
`torch.hub.load` for the Silero VAD, which must not happen here.
"""

import asyncio

import numpy as np
import pytest

from src.audio_buffer import AudioBuffer
from src.config import PTT_MAX_HOLD_S, PTT_MIN_HOLD_S, SAMPLE_RATE
from src.ws_handler import _commit_hold, _flush_mic_and_process, _handle_speak_audio


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, data: str) -> None:
        self.sent.append(data)


class _ExplodingVad:
    """Any use of the mic VAD during a push-to-talk hold is the bug this
    change removes, so the double fails loudly instead of quietly splitting.
    """

    def process_chunk(self, chunk):  # pragma: no cover - must never run
        raise AssertionError("mic VAD ran during a push-to-talk hold")

    def flush(self):  # pragma: no cover - must never run
        raise AssertionError("mic VAD flushed during a push-to-talk hold")

    def reset(self) -> None:
        pass


class _StubSession:
    def __init__(self, mic_mode: str = "push_to_talk") -> None:
        self.session_id = "stub-session"
        self.mic_mode = mic_mode
        self.mic_active = True
        self.tts_playing = False
        self.transmission_active = False
        self.mic_buffer = AudioBuffer()
        self.mic_vad = _ExplodingVad()
        self.mic_hold_chunks: list[np.ndarray] = []


def _audio_bytes(seconds: float) -> bytes:
    """Float32 PCM at the pipeline's sample rate, non-silent."""
    samples = np.full(int(SAMPLE_RATE * seconds), 0.2, dtype=np.float32)
    return samples.tobytes()


def _feed(session, data: bytes, processed: list) -> None:
    async def _fake_process(websocket, sess, segment, ws_lock, background_tasks):
        processed.append(segment)

    import src.ws_handler as ws_handler

    original = ws_handler._process_speak_segment
    ws_handler._process_speak_segment = _fake_process
    try:
        asyncio.run(
            _handle_speak_audio(
                _FakeWebSocket(), session, data, asyncio.Lock(), set()
            )
        )
    finally:
        ws_handler._process_speak_segment = original


def _release(session, processed: list) -> None:
    async def _fake_process(websocket, sess, segment, ws_lock, background_tasks):
        processed.append(segment)

    import src.ws_handler as ws_handler

    original = ws_handler._process_speak_segment
    ws_handler._process_speak_segment = _fake_process
    try:
        asyncio.run(
            _flush_mic_and_process(
                _FakeWebSocket(), session, asyncio.Lock(), set()
            )
        )
    finally:
        ws_handler._process_speak_segment = original


def test_hold_emits_nothing_while_the_key_is_down():
    session = _StubSession()
    processed: list = []

    for _ in range(4):
        _feed(session, _audio_bytes(1.0), processed)

    assert processed == [], "push-to-talk must not transcribe mid-hold"
    assert session.mic_hold_chunks, "hold audio must be buffered, not dropped"


def test_release_commits_the_whole_hold_as_one_segment():
    session = _StubSession()
    processed: list = []

    for _ in range(3):
        _feed(session, _audio_bytes(1.0), processed)
    _release(session, processed)

    assert len(processed) == 1, "the hold is one utterance, not several"
    held_s = processed[0].shape[0] / SAMPLE_RATE
    assert held_s == pytest.approx(3.0, abs=0.1), (
        "the committed segment must carry the entire hold -- a short one means "
        "audio was split or discarded mid-hold"
    )
    assert session.mic_hold_chunks == []


def test_a_pause_mid_hold_does_not_split_the_utterance():
    """The exact reported symptom: speak, pause, keep speaking, release."""
    session = _StubSession()
    processed: list = []

    _feed(session, _audio_bytes(2.0), processed)
    _feed(session, np.zeros(SAMPLE_RATE, dtype=np.float32).tobytes(), processed)
    _feed(session, _audio_bytes(2.0), processed)
    assert processed == [], "a mid-hold pause must not commit the first half"

    _release(session, processed)
    assert len(processed) == 1
    held_s = processed[0].shape[0] / SAMPLE_RATE
    assert held_s == pytest.approx(5.0, abs=0.1), (
        "everything after the pause was lost -- the tail-drop regression"
    )


def test_audio_is_ignored_while_the_key_is_up():
    session = _StubSession()
    session.mic_active = False
    processed: list = []

    _feed(session, _audio_bytes(1.0), processed)

    assert processed == []
    assert session.mic_hold_chunks == []


def test_a_key_bounce_never_reaches_asr():
    session = _StubSession()
    session.mic_hold_chunks = [
        np.full(int(SAMPLE_RATE * (PTT_MIN_HOLD_S / 2)), 0.2, dtype=np.float32)
    ]

    assert _commit_hold(session) is None
    assert session.mic_hold_chunks == []


def test_a_stuck_key_commits_at_the_cap_instead_of_growing_forever():
    """Fed one second at a time, as a real client streams it -- `AudioBuffer`
    has its own single-call size guard that a 60-second blob would trip first,
    which would test that guard instead of this cap.
    """
    session = _StubSession()
    processed: list = []

    for _ in range(int(PTT_MAX_HOLD_S) + 1):
        _feed(session, _audio_bytes(1.0), processed)

    assert len(processed) == 1, "the cap must commit rather than buffer forever"
    committed_s = processed[0].shape[0] / SAMPLE_RATE
    assert committed_s == pytest.approx(PTT_MAX_HOLD_S, abs=1.0)
    # The key is still down, so a fresh hold starts accumulating immediately --
    # the cap commits the backlog, it does not end the hold.
    leftover_s = sum(c.shape[0] for c in session.mic_hold_chunks) / SAMPLE_RATE
    assert leftover_s < 2.0


def test_tts_playback_discards_the_hold_rather_than_queueing_behind_itself():
    """Unchanged feedback-loop guard: audio arriving while our own TTS plays
    is still dropped. Buffering it would feed the app its own voice.
    """
    session = _StubSession()
    session.tts_playing = True
    processed: list = []

    _feed(session, _audio_bytes(1.0), processed)

    assert processed == []
    assert session.mic_hold_chunks == []


def test_auto_mode_still_goes_through_the_vad():
    session = _StubSession(mic_mode="auto")
    processed: list = []

    with pytest.raises(AssertionError, match="mic VAD ran"):
        _feed(session, _audio_bytes(1.0), processed)
