"""T021 (Red) — mic suppression must outlast transmission, not just match it
(contracts/playback-protocol.md, SC-002, SC-003, and FR-007's listen-path
guard which the same file's scope covers per the brief's T029 item).

Scope: the suppression *state machine* in ws_handler.py, not the ASR/
translation/TTS models. `_install_deterministic_engines` monkeypatches the
three engine singletons at the exact boundary ws_handler calls them
(`asr_engine.transcribe`, `translation_engine.translate`,
`tts_engine.start_synthesis`/`recv_chunk`) so the mechanism under test --
`session.tts_playing`, `session.suppression`, sample accumulation, chunk
streaming -- all runs for real. No audio devices, no real models (both
forbidden this round).

WS driving choice: a fake websocket object passed directly to the
`_handle_*`/`_process_speak_segment`/`_translate_and_speak` functions,
rather than FastAPI's TestClient. Rationale (per brief section "WS 구동"):
these tests need to control wall-clock timing precisely (simulate playback
continuing for seconds after transmission ends, probe the suppression gate
mid-window, measure release latency) -- a real TestClient websocket adds an
uncontrolled network/ASGI-transport layer between the test and the code
under test for no benefit here, since nothing in this round touches
`handle_websocket`'s accept/receive loop itself (T029's guard and T028's
release live in the per-message handlers, which take a plain object
duck-typed to `send_text`/`send_bytes`). The same pattern is already
established in `tests/unit/test_protocol_playback_complete.py`'s
`_FakeWebSocket`.

A real `ClientSession` cannot be constructed here: its `__init__` builds two
`VoiceActivityDetector`s, whose `__init__` calls `create_vad_model()`, which
raises unless `load_vad_model()` (a real `torch.hub.load`) has already run
process-wide -- forbidden this round. `_SpeakSessionStub` below is a session
double exposing exactly the attributes the code paths under test touch, with
fake `vad`/`mic_vad` objects that record calls instead of running Silero.
"""

import asyncio
import json
import time

import numpy as np
import pytest

from src.asr import asr_engine
from src.audio_buffer import AudioBuffer
from src.config import CHUNK_BYTES, TTS_SAMPLE_RATE
from src.translation import translation_engine
from src.tts import tts_engine
from src.ws_handler import (
    _flush_mic_and_process,
    _handle_control,
    _handle_listen_audio,
    _handle_speak_audio,
    _process_speak_segment,
    _release_suppression,
)


class _FakeWebSocket:
    """Records every frame sent, without opening a real socket."""

    def __init__(self) -> None:
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []

    async def send_text(self, data: str) -> None:
        self.sent_text.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)


class _FakeVad:
    """Stands in for VoiceActivityDetector without loading Silero.

    Records call counts so a test can assert a suppression gate prevented
    this from ever being reached, rather than asserting on VAD output
    (which would require the real model to produce a meaningful segment).

    `emit=True` makes `process_chunk` return a speech segment on every call
    -- for tests that need to prove a gate *upstream* of the VAD (not the
    VAD's own silence detection) is what's blocking a second segment from
    starting: with `emit=True`, any mic chunk that reaches `process_chunk`
    at all would produce a new segment if nothing stopped it first.
    """

    def __init__(self, emit: bool = False) -> None:
        self.process_chunk_calls = 0
        self.reset_calls = 0
        self.flush_calls = 0
        self._emit = emit

    def process_chunk(self, chunk: np.ndarray) -> list[np.ndarray]:
        self.process_chunk_calls += 1
        if self._emit:
            return [np.zeros(8000, dtype=np.float32)]
        return []

    def reset(self) -> None:
        self.reset_calls += 1

    def flush(self) -> np.ndarray | None:
        self.flush_calls += 1
        return None


class _SpeakSessionStub:
    """Session double exposing exactly what the speak/listen suppression
    paths touch. See module docstring for why this isn't a real
    ClientSession."""

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
    """Stub ASR/translation/TTS at the engine-call boundary only (brief
    design guideline 1): everything downstream -- tts_playing bookkeeping,
    total_samples accumulation, chunk streaming to the websocket -- still
    runs for real. `synthesized_samples` controls exactly how much "playback"
    the resulting suppression deadline will be computed from.
    """
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


def _install_engines_with_realistic_timing(
    monkeypatch: pytest.MonkeyPatch,
    synthesized_samples: int,
    translate_delay_s: float,
    per_chunk_delay_s: float,
    chunk_samples: int = 4800,
) -> None:
    """Like `_install_deterministic_engines`, but `translate` and each
    `recv_chunk` call block for a realistic amount of time via `time.sleep`
    -- run inside `loop.run_in_executor`'s thread pool, the same place a
    real model's inference would block, never on the event loop itself.
    Still a stub (deterministic output, no real ASR/translation/TTS model
    loaded -- forbidden this round): the only thing added is latency, to
    make "transmission takes longer than the suppression deadline's initial
    margin" a reproducible condition rather than one requiring real models.
    """
    monkeypatch.setattr(asr_engine, "transcribe", lambda audio, language: "stub transcript")

    def _translate(text, target):
        time.sleep(translate_delay_s)
        return "stub translation"

    monkeypatch.setattr(translation_engine, "translate", _translate)

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
            time.sleep(per_chunk_delay_s)
            return ("chunk", chunks[i], TTS_SAMPLE_RATE)
        return ("done", None, 0)

    monkeypatch.setattr(tts_engine, "start_synthesis", _start_synthesis)
    monkeypatch.setattr(tts_engine, "recv_chunk", _recv_chunk)


async def _wait_until(predicate, timeout: float, interval: float = 0.01) -> None:
    """Bounded poll. Raises asyncio.TimeoutError (a real test failure, never
    a silent pass) if `predicate` doesn't become true in time."""

    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(interval)

    await asyncio.wait_for(_poll(), timeout=timeout)


# --- SC-002 / SC-003 ---------------------------------------------------------

@pytest.mark.asyncio
async def test_mic_stays_suppressed_through_playback_and_releases_promptly(monkeypatch):
    """SC-002 (playback outlasting transmission by >=3s produces zero
    self-transcribed segments) and SC-003 (release within 500ms of
    playback_complete), in one real-time scenario per the brief's
    instruction to keep at least one case at the literal SC figures.

    Red today: session.tts_playing is cleared unconditionally when
    _translate_and_speak's transmission loop ends (ws_handler.py:381-385),
    which happens almost immediately for this stub (no real-time synthesis
    delay). So by the time this test's ~3.2s playback-simulation window
    even starts, the mic gate has already reopened -- the mid-window probe
    below would already see tts_playing=False and reach the VAD.

    SC-003 caveat: the real SC-003 claim is "release follows the *client's*
    last played sample within 500ms" -- but playback happens client-side,
    which this round (server-only, no client code touched per brief scope)
    has no way to observe. `release_elapsed` below only measures how long
    `_handle_control`'s `playback_complete` branch itself takes to run in
    this process, i.e. that the *server's* reaction to receiving the message
    is prompt -- a necessary but not sufficient condition for the full
    client-to-server SC-003 latency, which is unverifiable without the
    round-3 client wiring (T025/T026, out of scope here).
    """
    synthesized_samples = 96_000  # 4s @ 24kHz -> deadline = now + 4 + 1 = 5s,
    # comfortably past this test's ~3.2s + 0.5s window so release here is
    # driven by playback_complete, not the deadline (that's T022's job).
    _install_deterministic_engines(monkeypatch, synthesized_samples)

    websocket = _FakeWebSocket()
    session = _SpeakSessionStub()
    ws_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()

    await _process_speak_segment(
        websocket, session, np.zeros(8000, dtype=np.float32), ws_lock, background_tasks,
    )
    await _wait_until(lambda: session.suppression is not None, timeout=2.0)
    segment_id = session.suppression.segment_id
    assert session.tts_playing is True

    # ~3.2s of simulated ongoing client-side playback: repeatedly probe the
    # mic gate. SC-002: it must never reach the VAD.
    mic_chunk = bytes(CHUNK_BYTES)
    probe_deadline = time.monotonic() + 3.2
    while time.monotonic() < probe_deadline:
        await _handle_speak_audio(websocket, session, mic_chunk, ws_lock, background_tasks)
        await asyncio.sleep(0.2)

    assert session.tts_playing is True, "suppression must still hold >3s after transmission ended"
    assert session.mic_vad.process_chunk_calls == 0, "mic audio must never reach the VAD while suppressed"

    release_start = time.monotonic()
    await _handle_control(
        websocket, session,
        json.dumps({"type": "playback_complete", "segment_id": segment_id}),
        ws_lock, background_tasks,
    )
    release_elapsed = time.monotonic() - release_start
    assert release_elapsed < 0.5, "SC-003: release must follow playback_complete within 500ms"
    assert session.tts_playing is False

    # Gate is open again: mic audio now reaches the VAD.
    await _handle_speak_audio(websocket, session, mic_chunk, ws_lock, background_tasks)
    assert session.mic_vad.process_chunk_calls == 1


@pytest.mark.asyncio
async def test_push_to_talk_flush_still_discards_audio_while_suppressed(monkeypatch):
    """Existing behaviour at ws_handler.py:216 (_flush_mic_and_process's own
    tts_playing gate) must survive the T028 refactor unchanged: audio
    buffered during suppression is discarded on push-to-talk release, never
    processed retroactively once suppression lifts. Not expected to be red
    on its own -- this function isn't touched by T028 -- but the brief's
    'confirm existing behaviour' bullet is verified here explicitly."""
    _install_deterministic_engines(monkeypatch, synthesized_samples=2_400)  # 0.1s, keep this test fast

    websocket = _FakeWebSocket()
    session = _SpeakSessionStub()
    session.mic_mode = "push_to_talk"
    ws_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()

    await _process_speak_segment(
        websocket, session, np.zeros(8000, dtype=np.float32), ws_lock, background_tasks,
    )
    await _wait_until(lambda: session.suppression is not None, timeout=2.0)
    assert session.tts_playing is True

    resets_before = session.mic_vad.reset_calls
    await _flush_mic_and_process(websocket, session, ws_lock, background_tasks)

    assert session.mic_vad.reset_calls == resets_before + 1
    assert session.mic_vad.flush_calls == 0, "flush() must never run while suppressed"


# --- Deadline-vs-transmission race (Critic round blocker) -------------------
# A fixed deadline armed once, at the moment suppression starts, and never
# updated until the very end of `_translate_and_speak` cannot survive
# translation inference plus chunk streaming together taking longer than the
# deadline's margin -- the two prior tests above never exercise this because
# `_install_deterministic_engines` has zero latency, so transmission always
# finishes near-instantly regardless of how the deadline is computed.

@pytest.mark.asyncio
async def test_tts_playing_never_drops_before_transmission_ends_with_realistic_engine_timing(monkeypatch):
    """Critic-reproduced blocker: translation taking 0.6s plus 40 chunks x
    20ms of streaming (~1.4s total transmission -- conservative relative to
    spec.md:43's "8s translation transmits in ~1s") must never let
    `session.tts_playing` go False before the `tts_end` frame is actually
    sent, regardless of the suppression deadline's fixed initial margin
    (1.0s, `SUPPRESSION_DEADLINE_MARGIN_S`). Before the fix, the deadline was
    armed once at `_start_suppression` time and not updated until
    `_translate_and_speak`'s `finally`, so a transmission exceeding 1.0s end
    to end let the deadline watcher release suppression mid-transmission --
    which also let a second segment begin concurrently with the first still
    streaming to the same websocket (the invariant `_process_speak_segment`'s
    `tts_playing` gate exists to prevent). Red before the fix (deadline now
    armed only once TTS synthesis starts, and extended on every chunk
    received rather than only at the very end) -- see US2.md "Round 2"
    "Mutation testing" / red capture for the stashed-fix Red run.
    """
    _install_engines_with_realistic_timing(
        monkeypatch, synthesized_samples=192_000, translate_delay_s=0.6, per_chunk_delay_s=0.02,
    )

    websocket = _FakeWebSocket()
    session = _SpeakSessionStub()
    ws_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()

    await _process_speak_segment(
        websocket, session, np.zeros(8000, dtype=np.float32), ws_lock, background_tasks,
    )
    assert session.tts_playing is True

    def _tts_end_sent() -> bool:
        return any('"tts_end"' in frame for frame in websocket.sent_text)

    dropped_before_tts_end = False
    poll_deadline = time.monotonic() + 5.0
    while time.monotonic() < poll_deadline:
        if _tts_end_sent():
            break
        if session.tts_playing is False:
            dropped_before_tts_end = True
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("transmission never finished within the 5s test budget")

    assert not dropped_before_tts_end, (
        "session.tts_playing went False (suppression released) before the "
        "tts_end frame was sent -- suppression must outlast the entire "
        "transmission, not just the deadline's initial margin"
    )


# --- Transmission interlock (Critic round 3, major 2) ------------------------
# The "no concurrent transmissions" invariant _process_speak_segment's gate
# exists to enforce (two prior Critic-round blockers were both instances of
# this invariant breaking because suppression was released too early) must
# not depend solely on suppression staying up for the transmission's full
# duration. `session.transmission_active` is a second, independent flag that
# `_translate_and_speak` holds `True` for its own entire lifetime regardless
# of when/why suppression itself releases.

@pytest.mark.asyncio
async def test_second_segment_blocked_even_if_suppression_released_early(monkeypatch):
    """If suppression is ever released while a transmission is still
    genuinely in flight -- for any reason, including a bug this test
    simulates directly by calling `_release_suppression` itself rather than
    waiting for a real mistiming -- a second segment must still not be able
    to start while the first transmission task is still running.

    Red before the fix: `_process_speak_segment`'s gate checked only
    `session.tts_playing`, which this test's direct `_release_suppression`
    call clears -- letting the emitting mic VAD's segment straight through
    to a second call into `_process_speak_segment` while the first
    transmission was still streaming.

    Signal design: rather than polling for a *second* `start_synthesis` call
    (timing-dependent -- the second segment's own translate/start_synthesis
    steps take an unknown number of event-loop turns to reach), this checks
    `session.suppression`'s identity and `session.tts_playing` immediately
    after `_handle_speak_audio` returns. Both are deterministic at that exact
    point: if the gate did *not* block a second segment, `_process_speak_segment`
    would have already run its synchronous prefix -- `_start_suppression`
    (replacing `session.suppression` with a brand new window) and
    `session.tts_playing = True` -- before creating the new
    `_translate_and_speak` task (which is what's actually timing-dependent,
    and isn't what this test needs to observe).
    """
    _install_deterministic_engines(monkeypatch, synthesized_samples=192_000, chunk_samples=4800)
    # Slow down chunk delivery so the first transmission is still genuinely
    # in flight throughout the test (zero-delay chunks could finish on their
    # own before the test gets a chance to force an early release).
    original_recv_chunk = tts_engine.recv_chunk

    def _slow_recv_chunk(timeout: float):
        time.sleep(0.05)
        return original_recv_chunk(timeout)

    monkeypatch.setattr(tts_engine, "recv_chunk", _slow_recv_chunk)

    websocket = _FakeWebSocket()
    session = _SpeakSessionStub()
    session.mic_vad = _FakeVad(emit=True)  # any chunk that reaches process_chunk yields a segment
    ws_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()

    await _process_speak_segment(
        websocket, session, np.zeros(8000, dtype=np.float32), ws_lock, background_tasks,
    )
    await _wait_until(lambda: session.suppression is not None, timeout=2.0)
    first_segment_id = session.suppression.segment_id

    # Force an early release directly -- simulating suppression clearing for
    # any reason (bug, race, deadline miscalculation) while transmission is
    # still genuinely in flight, without waiting for a real mistiming.
    _release_suppression(session, first_segment_id)
    assert session.tts_playing is False

    # A mic frame arrives right after: with tts_playing now False, only the
    # interlock (transmission_active) can still block a second segment.
    mic_chunk = bytes(CHUNK_BYTES)
    await _handle_speak_audio(websocket, session, mic_chunk, ws_lock, background_tasks)

    assert session.suppression.segment_id == first_segment_id, (
        "a second segment's SuppressionWindow replaced the first's while "
        "the first transmission was still running"
    )
    assert session.tts_playing is False, (
        "a second segment set tts_playing back to True while the first "
        "transmission was still running"
    )

    # Cleanup wait so the (still-running) first transmission's background
    # task doesn't outlive the test.
    await _wait_until(lambda: session.transmission_active is False, timeout=5.0)


# --- finally quadrant: mid-stream exception after chunks (Critic round 4) ---
# `recv_chunk` can raise mid-stream after real audio has already been sent --
# the actual failure mode is tts.py:183-196's `_check_alive()` (TTS worker
# process died and the restart budget is exhausted). Releasing suppression
# immediately in that case reopens the mic while audio the client already
# received is still playing -- the same defect this round exists to fix,
# recurring one level deeper in the `finally` block's own release logic.

@pytest.mark.asyncio
async def test_mid_stream_exception_after_chunks_leaves_release_to_the_watcher(monkeypatch):
    """Red before the fix: `finally`'s `not (reached_transmission_end and
    watcher_armed)` released immediately on *any* path that didn't reach a
    clean "done"/tts_end -- including this one, even though 2 real chunks
    (0.4s of audio) had already been sent to the client and the watcher was
    already armed with an accurate deadline (that audio's duration + the
    1.0s margin) by the time the exception hit. Critic reproduced with 4.0s
    of audio actually sent before the exception; this test uses a smaller
    magnitude (0.4s) for speed -- the property under test ("watcher already
    armed and accurate" vs "transmission ended normally") doesn't depend on
    the specific duration.
    """
    monkeypatch.setattr(asr_engine, "transcribe", lambda audio, language: "stub transcript")
    monkeypatch.setattr(translation_engine, "translate", lambda text, target: "stub translation")

    chunk = np.zeros(4800, dtype=np.float32)  # 0.2s @ 24kHz
    state = {"i": 0}

    def _start_synthesis(text, voice, lang_code, ref_audio, ref_text) -> int:
        state["i"] = 0
        return 1

    def _recv_chunk(timeout: float):
        state["i"] += 1
        if state["i"] <= 2:
            return ("chunk", chunk, TTS_SAMPLE_RATE)
        # tts.py's recv_chunk calls _check_alive() as its first line, which
        # raises exactly this once the worker has died and the restart
        # budget is exhausted (tts.py:183-196).
        raise RuntimeError("TTS worker not running, max restarts exceeded")

    monkeypatch.setattr(tts_engine, "start_synthesis", _start_synthesis)
    monkeypatch.setattr(tts_engine, "recv_chunk", _recv_chunk)

    websocket = _FakeWebSocket()
    session = _SpeakSessionStub()
    ws_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()

    await _process_speak_segment(
        websocket, session, np.zeros(8000, dtype=np.float32), ws_lock, background_tasks,
    )
    assert len(background_tasks) == 1, "exactly one _translate_and_speak task should be running"
    transmit_task = next(iter(background_tasks))
    # Deterministic: _translate_and_speak catches the RuntimeError internally
    # (its own `except Exception` clause), so awaiting the task waits for the
    # *entire* function -- including `finally`'s release decision -- to run
    # to completion, without racing on wall-clock polling.
    await asyncio.wait_for(transmit_task, timeout=2.0)

    assert len(websocket.sent_bytes) == 2, "both chunks before the exception should have been sent"
    assert session.tts_playing is True, (
        "suppression must not release immediately on a mid-stream exception "
        "-- 2 chunks of already-sent audio are still playing client-side"
    )

    # The watcher's deadline (0.4s of audio + 1.0s margin) fires next.
    await _wait_until(lambda: session.tts_playing is False, timeout=3.0)


# --- FR-007 (T029 listen-path guard) -----------------------------------------
# Not among the brief's enumerated Red-case bullets for T021, but explicitly
# in this round's scope ("설계 지침 4: T029") and in the contract's own
# verification table ("Listen path is guarded"). Flagged here per the
# reporting instruction rather than silently added.

@pytest.mark.asyncio
async def test_listen_audio_produces_no_segment_while_tts_playing(monkeypatch):
    """FR-007: synthesized audio reaching the system tap while speak-mode
    TTS is playing must not produce a transcription. Red today:
    _handle_listen_audio has no tts_playing guard at all (ws_handler.py:148)."""
    websocket = _FakeWebSocket()
    session = _SpeakSessionStub()
    session.tts_playing = True
    ws_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()

    listen_chunk = bytes(CHUNK_BYTES)
    await _handle_listen_audio(websocket, session, listen_chunk, ws_lock, background_tasks)

    assert session.vad.process_chunk_calls == 0
    assert websocket.sent_text == []


@pytest.mark.asyncio
async def test_listen_audio_produces_no_segment_when_transmission_active_but_tts_playing_is_false(monkeypatch):
    """FR-007 (Critic round 4, minor 1): the exact state the transmission
    interlock (round 3, major 2) exists to cover -- suppression released
    early (`tts_playing=False`) while a transmission is still genuinely
    running (`transmission_active=True`) -- must also keep the listen/both-
    mode tap from transcribing the still-streaming synthesized audio. Red
    before the fix: `_handle_listen_audio`'s guard checked only
    `tts_playing`, so this exact combination let synthesized audio reach the
    VAD."""
    websocket = _FakeWebSocket()
    session = _SpeakSessionStub()
    session.tts_playing = False
    session.transmission_active = True
    ws_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()

    listen_chunk = bytes(CHUNK_BYTES)
    await _handle_listen_audio(websocket, session, listen_chunk, ws_lock, background_tasks)

    assert session.vad.process_chunk_calls == 0
    assert websocket.sent_text == []
