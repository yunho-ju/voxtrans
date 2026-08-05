import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .asr import asr_engine
from .config import AUDIO_PREFIX_MIC, AUDIO_PREFIX_SYSTEM, AUDIO_PREFIX_TTS, MAX_SESSIONS, TTS_SAMPLE_RATE
from .protocol import (
    ConfigureMessage,
    make_error,
    make_speak_transcription,
    make_speak_translation,
    make_status,
    make_transcription,
    make_translation,
    make_tts_end,
    parse_control_message,
)
from .session import ClientSession, SuppressionWindow
from .translation import translation_engine
from .tts import tts_engine

logger = logging.getLogger(__name__)

# Max pending background tasks per session (rate limiting)
MAX_PENDING_TASKS = 10

# Suppression deadline margin (data-model.md §4 D4): a starting value, not a
# defended one -- SC-003 measures the residual between actual playback
# completion and release, and the margin moves on that evidence.
SUPPRESSION_DEADLINE_MARGIN_S = 1.0

# Single semaphore for ALL MLX model inference (ASR + translation share Metal GPU;
# concurrent access causes MTLCommandBuffer assertion failures)
_mlx_semaphore = asyncio.Semaphore(1)

# Bounded thread pool for model inference (prevent unlimited thread creation)
_executor = ThreadPoolExecutor(max_workers=1)

# Active sessions
_sessions: dict[str, ClientSession] = {}


def _active_meeting_file() -> str | None:
    config_path = os.path.expanduser("~/meeting_current.json")
    try:
        with open(config_path) as f:
            return json.load(f).get("filename")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _append_to_meeting(speaker: str, text: str) -> None:
    path = _active_meeting_file()
    if not path:
        return
    try:
        ts = datetime.now().strftime("%H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {speaker}: {text}\n")
    except OSError:
        pass


def _append_translation(translation: str) -> None:
    path = _active_meeting_file()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"  → {translation}\n")
    except OSError:
        pass


def _release_suppression(session: ClientSession, segment_id: int) -> None:
    """Idempotent mic-suppression release (data-model.md §4, FR-012).

    The single function every exit path shares: `playback_complete`
    reception, the deadline timer, and session close/error all call this
    exact function rather than mutating `tts_playing` themselves. Releasing
    twice, or releasing a `segment_id` that isn't the session's current
    suppressing segment (stale/unknown), is a no-op -- never an error.

    Reads `session.suppression` directly -- every real `ClientSession`
    initializes it to `None` (`session.py`), so there is nothing to
    tolerate here. A `getattr(session, "suppression", None)` fallback
    previously stood in this spot to accommodate a test double that didn't
    model the attribute at all, which made this production code path
    defensive against a stub rather than the real session (Critic T032
    minor 1) -- a future rename of `suppression` on the real session would
    have silently degraded release into a permanent no-op (unbounded
    suppression) with no test catching it.
    `tests/unit/test_protocol_playback_complete.py`'s `_StubSession` now
    sets `self.suppression = None` itself instead.
    """
    window = session.suppression
    if window is None or window.segment_id != segment_id or window.released:
        return
    window.released = True
    session.tts_playing = False
    session.mic_buffer.clear()
    session.mic_vad.reset()


async def _watch_suppression_deadline(session: ClientSession, segment_id: int) -> None:
    """Background task started by `_arm_suppression_watcher`, once the first
    real TTS chunk has actually been received (see that function's docstring
    for why arming is deferred that far). Releases via `_release_suppression`
    once the window's deadline passes, unless `playback_complete` released it
    first (FR-004). Re-reads `window.deadline` after every wake-up so a
    deadline extended mid-flight -- `_translate_and_speak`'s chunk loop
    extends it on every chunk received, which is later than this task's
    start -- is honoured rather than firing on a stale earlier estimate.
    Each sleep is bounded by whatever the deadline currently is, so this can
    never wait unboundedly.
    """
    while True:
        window = session.suppression
        if window is None or window.segment_id != segment_id or window.released:
            return
        remaining = window.deadline - time.monotonic()
        if remaining <= 0:
            _release_suppression(session, segment_id)
            return
        await asyncio.sleep(remaining)


def _start_suppression(session: ClientSession, segment_id: int) -> None:
    """Creates a fresh SuppressionWindow for `segment_id` at the moment
    `session.tts_playing` is set True. Deliberately does **not** start the
    deadline watcher yet -- see `_arm_suppression_watcher`, called later by
    `_translate_and_speak`'s chunk loop once the first real TTS chunk
    arrives, for why. `window.deadline` is set here to a placeholder (unused
    until armed) so the field is never observed uninitialized.
    """
    window = SuppressionWindow(
        segment_id=segment_id,
        synthesized_samples=0,
        deadline=time.monotonic() + SUPPRESSION_DEADLINE_MARGIN_S,
    )
    session.suppression = window


def _arm_suppression_watcher(
    session: ClientSession, segment_id: int, background_tasks: set[asyncio.Task],
) -> None:
    """Starts the deadline watcher for `segment_id`'s SuppressionWindow.
    Called exactly once, by `_translate_and_speak`'s chunk-receiving loop,
    the moment the *first real TTS chunk* is received -- not at
    `_start_suppression` time, and (Critic round 3) not merely once
    `tts_engine.start_synthesis` returns either.

    Why not at `start_synthesis` return: `start_synthesis` only enqueues a
    synthesis request and returns immediately (`tts.py:198-214`) -- it does
    not mean audio exists yet. Arming there still left a fixed
    `SUPPRESSION_DEADLINE_MARGIN_S` (1.0s) window between "queued" and
    "first chunk actually produced," which the real TTS pipeline's own
    documented baseline (~600ms) leaves only ~400ms of headroom against
    ordinary variance -- Critic-reproduced with a 1.2s first-chunk delay:
    the watcher fired, released suppression, and let a second segment start
    synthesizing (`_response_q` cross-talk risk, tts.py:232-237) before any
    audio from the first had even been sent.

    Calling this only once the first chunk has actually arrived removes
    that fixed window entirely: by the time this runs, the caller has
    already updated `window.deadline` from real `synthesized_samples` (see
    the chunk loop), so the watcher never observes a placeholder,
    margin-only deadline -- only ever a sample-derived one. Every chunk
    after the first keeps extending it the same way, so (barring a
    mid-stream stall longer than the margin -- see "Remaining risks") the
    deadline stays ahead of "now" for the rest of the stream.

    Translation inference and `start_synthesis` themselves, and the gap up
    to the first chunk, are **not** covered by the watcher at all (it
    doesn't exist yet) -- covered instead by `_translate_and_speak`'s own
    `finally` (path c) and session teardown (path d); see that function's
    docstring "Why this doesn't fully close the pre-synthesis gap".
    """
    window = session.suppression
    if window is None or window.segment_id != segment_id:
        return
    task = asyncio.create_task(_watch_suppression_deadline(session, segment_id))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)


async def handle_websocket(websocket: WebSocket) -> None:
    if len(_sessions) >= MAX_SESSIONS:
        await websocket.close(code=1013, reason="Max sessions reached")
        return

    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    session = ClientSession(session_id)
    _sessions[session_id] = session
    ws_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()

    logger.info("Client connected: %s (total: %d)", session_id, len(_sessions))
    await websocket.send_text(make_status("connected", session_id=session_id))

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                await _handle_audio(websocket, session, message["bytes"], ws_lock, background_tasks)
            elif "text" in message and message["text"]:
                await _handle_control(websocket, session, message["text"], ws_lock, background_tasks)

    except (WebSocketDisconnect, RuntimeError):
        logger.info("Client disconnected: %s", session_id)
    except Exception:
        logger.exception("Error in session %s", session_id)
        try:
            await websocket.send_text(make_error("Internal server error"))
        except Exception:
            pass
    finally:
        for task in list(background_tasks):
            task.cancel()
        for task in list(background_tasks):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        # Session teardown is itself a suppression exit path (FR-012): if the
        # client vanished mid-playback, release explicitly rather than
        # leaving a released=False window on a session about to be discarded.
        if session.suppression is not None:
            _release_suppression(session, session.suppression.segment_id)
        session.close()
        _sessions.pop(session_id, None)
        logger.info("Session cleaned up: %s (remaining: %d)", session_id, len(_sessions))


async def _handle_audio(
    websocket: WebSocket,
    session: ClientSession,
    data: bytes,
    ws_lock: asyncio.Lock,
    background_tasks: set[asyncio.Task],
) -> None:
    # Parse prefix byte: first byte determines audio type
    if len(data) == 0:
        return

    prefix = data[0]
    # Backward compat: if first byte doesn't match known prefix,
    # treat entire message as system audio (0x01)
    if prefix in (AUDIO_PREFIX_SYSTEM, AUDIO_PREFIX_MIC, AUDIO_PREFIX_TTS):
        audio_data = data[1:]
    else:
        prefix = AUDIO_PREFIX_SYSTEM
        audio_data = data

    if prefix == AUDIO_PREFIX_SYSTEM and session.mode in ("listen", "both"):
        await _handle_listen_audio(websocket, session, audio_data, ws_lock, background_tasks)
    elif prefix == AUDIO_PREFIX_MIC and session.mode in ("speak", "both"):
        await _handle_speak_audio(websocket, session, audio_data, ws_lock, background_tasks)


async def _handle_listen_audio(
    websocket: WebSocket,
    session: ClientSession,
    data: bytes,
    ws_lock: asyncio.Lock,
    background_tasks: set[asyncio.Task],
) -> None:
    """Handle system audio for listen mode (existing logic)."""
    # Suppress the listen path while TTS is playing: synthesized audio
    # reaching the system tap must not produce a self-transcription (FR-007),
    # the same gate _handle_speak_audio already had. Also checks
    # `transmission_active` (Critic round 4, minor 1), not just
    # `tts_playing`: the interlock's whole point is to cover the case where
    # suppression released early while a transmission is still genuinely
    # running -- tts_playing=False, transmission_active=True -- and without
    # this the listen/both-mode tap would transcribe the still-streaming
    # synthesized audio in exactly that window.
    if session.tts_playing or session.transmission_active:
        return

    chunks = session.audio_buffer.add(data)
    for chunk in chunks:
        speech_segments = session.vad.process_chunk(chunk)
        for speech_segment in speech_segments:
            text = await _transcribe(speech_segment, session.language)
            if text:
                segment_id = session.next_segment_id()
                logger.info("[listen] seg=%d | ASR(lang=%s) → '%s' | translate_to='%s'",
                            segment_id, session.language, text[:80], session.target_language)
                async with ws_lock:
                    await websocket.send_text(
                        make_transcription(text, is_final=True, segment_id=segment_id)
                    )
                _append_to_meeting(f"Speaker-{session.session_id[:4]}", text)
                if session.target_language:
                    # Rate limit: skip translation if too many pending
                    pending = sum(1 for t in background_tasks if not t.done())
                    if pending >= MAX_PENDING_TASKS:
                        logger.warning("Session %s: dropping translation (pending=%d)",
                                       session.session_id, pending)
                    else:
                        task = asyncio.create_task(
                            _translate_and_send(websocket, ws_lock, text, session.target_language, segment_id)
                        )
                        background_tasks.add(task)
                        task.add_done_callback(background_tasks.discard)


async def _handle_speak_audio(
    websocket: WebSocket,
    session: ClientSession,
    data: bytes,
    ws_lock: asyncio.Lock,
    background_tasks: set[asyncio.Task],
) -> None:
    """Handle mic audio for speak mode: ASR → translate → TTS."""
    # Suppress mic input while TTS is playing (prevent feedback loop)
    if session.tts_playing:
        return

    # Push-to-talk: discard audio when mic is not active
    if session.mic_mode == "push_to_talk" and not session.mic_active:
        return

    chunks = session.mic_buffer.add(data)
    for chunk in chunks:
        speech_segments = session.mic_vad.process_chunk(chunk)
        for speech_segment in speech_segments:
            await _process_speak_segment(
                websocket, session, speech_segment, ws_lock, background_tasks
            )


async def _flush_mic_and_process(
    websocket: WebSocket,
    session: ClientSession,
    ws_lock: asyncio.Lock,
    background_tasks: set[asyncio.Task],
) -> None:
    """Flush the mic VAD buffer (push-to-talk release) and process any speech."""
    if session.tts_playing:
        session.mic_vad.reset()
        session.mic_buffer.clear()
        return
    segment = session.mic_vad.flush()
    if segment is not None:
        await _process_speak_segment(
            websocket, session, segment, ws_lock, background_tasks
        )
    session.mic_buffer.clear()


async def _process_speak_segment(
    websocket: WebSocket,
    session: ClientSession,
    speech_segment: np.ndarray,
    ws_lock: asyncio.Lock,
    background_tasks: set[asyncio.Task],
) -> None:
    """Transcribe a speak-mode speech segment and kick off translate+TTS."""
    # `transmission_active` (Critic round 3, major 2) closes the gate even if
    # `tts_playing`/suppression was released early for any reason while a
    # transmission is still genuinely running -- see _translate_and_speak's
    # docstring.
    if session.tts_playing or session.transmission_active:
        return
    text = await _transcribe(speech_segment, session.language)
    if not text:
        return
    if session.tts_playing or session.transmission_active:  # re-check after transcribe (may have changed during await)
        return
    logger.info("[speak] ASR(lang=%s) → '%s' | speak_target='%s' voice='%s' clone=%s",
                session.language, text[:80], session.speak_target_language,
                session.tts_voice, session.voice_clone)

    # Save first speech as voice reference (trim to max 5s for stable ICL)
    if session.ref_audio is None:
        max_ref_samples = 5 * 16000  # 5 seconds at 16kHz
        ref = speech_segment[:max_ref_samples].copy()
        session.ref_audio = ref
        session.ref_text = text
        logger.info("Session %s: saved voice reference (%d samples, text='%s')",
                    session.session_id, ref.size, text[:50])

    segment_id = session.next_segment_id()
    async with ws_lock:
        await websocket.send_text(
            make_speak_transcription(text, segment_id)
        )

    # Lock TTS immediately to prevent concurrent tasks (race condition fix)
    session.tts_playing = True
    session.mic_buffer.clear()
    session.mic_vad.reset()
    _start_suppression(session, segment_id)

    task = asyncio.create_task(
        _translate_and_speak(
            websocket, ws_lock, text,
            session.speak_target_language,
            session.tts_voice,
            segment_id,
            session,
            background_tasks,
        )
    )
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)


async def _transcribe(audio: np.ndarray, language: str) -> str:
    async with _mlx_semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, asr_engine.transcribe, audio, language)


async def _translate_and_send(
    websocket: WebSocket,
    ws_lock: asyncio.Lock,
    text: str,
    target_language: str,
    segment_id: int,
) -> None:
    try:
        async with _mlx_semaphore:
            loop = asyncio.get_event_loop()
            translated = await loop.run_in_executor(
                _executor, translation_engine.translate, text, target_language,
            )
        if translated:
            async with ws_lock:
                await websocket.send_text(make_translation(translated, segment_id))
            _append_translation(translated)
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Translation failed for segment %d", segment_id)


async def _translate_and_speak(
    websocket: WebSocket,
    ws_lock: asyncio.Lock,
    text: str,
    target_language: str,
    voice: str,
    segment_id: int,
    session: ClientSession,
    background_tasks: set[asyncio.Task],
) -> None:
    """Translate text, then stream TTS audio chunks as they are generated.

    Note: session.tts_playing is already True and a SuppressionWindow already
    created (both by `_start_suppression`, called by the caller before this
    task was created) -- but the deadline *watcher* is armed later, by
    `_arm_suppression_watcher`, once the first real chunk arrives in step 4
    below. See that function's docstring for why not any earlier.

    `session.transmission_active` (Critic round 3, major 2): set `True` for
    this function's entire lifetime via the outer try/finally below,
    independent of `tts_playing`/suppression's own release timing.
    `_process_speak_segment` and `_handle_control`'s `speak_text` branch gate
    on `tts_playing or transmission_active`, not `tts_playing` alone -- so
    even if suppression is ever released while this function is still
    genuinely running (a bug, a race, a future deadline miscalculation), a
    second segment still cannot start and collide with this one on the same
    websocket (`_response_q` cross-talk risk, tts.py:232-237). This is a
    second, independent line of defense: suppression's own deadline/release
    correctness (documented below) is still the thing keeping the *mic*
    closed; `transmission_active` is what keeps a *second transmission* from
    starting, and does not depend on the first being correct.

    T028: mic suppression must outlast *transmission* (this function ending)
    to actually outlast *playback* (client-side), which routinely takes
    several seconds longer since TTS synthesis runs far faster than real
    time. So this function no longer releases suppression itself on the
    success path -- each chunk received in step 4 extends the window's
    deadline to reflect real synthesized duration so far, and lets
    `playback_complete` (`_handle_control`) or the deadline watcher
    (`_watch_suppression_deadline`) release it later, whichever comes first
    (FR-002, FR-004).

Every path that never arms the watcher -- empty translation, `start_synthesis`
    failing, cancellation or an exception before any chunk ever arrives, or
    reaching "done" with zero chunks ever produced -- released immediately in
    `finally`, via the same `_release_suppression` function (FR-012): nothing
    was queued for the client to play, so there is nothing to wait on, and no
    watcher exists to do it later. Once the watcher *is* armed (the first
    chunk has arrived), release is left entirely to it from then on --
    including if this function then exits abnormally mid-stream (e.g.
    `recv_chunk` raising after several chunks were already sent, tts.py:
    183-196): the watcher's deadline already accounts for exactly the audio
    actually sent, and releasing here instead would reopen the mic while
    already-sent audio is still playing client-side (Critic round 4).

    Why this doesn't fully close the pre-synthesis gap: translation
    inference (step 1), `start_synthesis` (step 3), and the wait for the
    first chunk (start of step 4) all run before the deadline watcher exists
    at all (`_arm_suppression_watcher` hasn't been called yet). If any of
    these blocks indefinitely without ever returning/arriving, no deadline
    bounds suppression until this task is itself cancelled (session
    teardown, path (d)) -- but that is not a regression: the exact same
    unbounded-block characteristic existed for these calls before this
    round's `SuppressionWindow` existed at all (the original code's
    unconditional `finally` release never ran either, if the call never
    returned). FR-004's guarantee ("no `playback_complete` -> release at
    `synthesized_duration + 1.0s`") is about a client that stops responding
    *after* synthesis has genuinely started producing audio, which is
    exactly the case the watcher is armed for.
    """
    session.transmission_active = True
    total_samples = 0
    watcher_armed = False
    try:
        loop = asyncio.get_event_loop()

        # 1. Translate
        async with _mlx_semaphore:
            translated = await loop.run_in_executor(
                _executor, translation_engine.translate, text, target_language,
            )
        if not translated:
            return

        async with ws_lock:
            await websocket.send_text(make_speak_translation(translated, segment_id))

        # 2. Language code: model expects ISO code ("en", "ko", etc.)
        lang_code = _language_to_tts_code(target_language)

        # 3. Start streaming TTS synthesis. Note: a successful return here
        # only means the request was *enqueued* (tts.py:198-214) -- not that
        # audio exists yet. Deliberately not arming the deadline watcher at
        # this point (see _arm_suppression_watcher's docstring); it's armed
        # in step 4 below, on the first real chunk.
        # Only use voice clone if ref_audio is actually available
        use_clone = session.voice_clone and session.ref_audio is not None
        ref_audio = session.ref_audio if use_clone else None
        ref_text = session.ref_text if use_clone else None
        req_id = await loop.run_in_executor(
            _executor, tts_engine.start_synthesis,
            translated, voice, lang_code, ref_audio, ref_text,
        )
        if req_id < 0:
            return

        # 4. Stream each chunk to client as soon as it arrives
        while True:
            status, audio_np, sr = await loop.run_in_executor(
                None, tts_engine.recv_chunk, 10.0,
            )
            if status == "chunk":
                total_samples += audio_np.size
                # Extend the deadline on every chunk (before arming, so the
                # watcher never observes a placeholder value -- see
                # _arm_suppression_watcher's docstring): since synthesis runs
                # far faster than real time, this keeps the deadline ahead
                # of "now" for as long as chunks keep arriving.
                window = session.suppression
                if window is not None and window.segment_id == segment_id:
                    window.synthesized_samples = total_samples
                    window.deadline = (
                        time.monotonic() + total_samples / TTS_SAMPLE_RATE + SUPPRESSION_DEADLINE_MARGIN_S
                    )
                if not watcher_armed:
                    # First real audio produced -- only now do we know the
                    # server is actually generating playable audio, and only
                    # now does the deadline watcher start (Critic round 3;
                    # see _arm_suppression_watcher's docstring for why
                    # arming any earlier -- even right after start_synthesis
                    # succeeded -- was still too early).
                    _arm_suppression_watcher(session, segment_id, background_tasks)
                    watcher_armed = True
                payload = bytes([AUDIO_PREFIX_TTS]) + audio_np.tobytes()
                async with ws_lock:
                    await websocket.send_bytes(payload)
            elif status == "done":
                break
            elif status == "error":
                logger.error("TTS error for segment %d", segment_id)
                break
            # timeout: keep waiting

        # 5. Signal TTS end
        async with ws_lock:
            await websocket.send_text(make_tts_end(segment_id))

    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Translate+speak failed for segment %d", segment_id)
    finally:
        # Release immediately unless the watcher is armed -- gated purely on
        # `watcher_armed`, not on how (or whether) the try block above
        # completed (Critic round 4: an earlier version of this branch also
        # required "reached the end of transmission normally," which got one
        # of the four resulting scenarios wrong -- see below).
        #
        # Whenever the watcher *is* armed, its `window.deadline` is already
        # correct for exactly the audio actually sent so far (kept current
        # by every chunk processed in the loop above), independent of
        # whether the function is exiting normally or abnormally -- so the
        # right thing is always to leave it alone and let the watcher
        # release on schedule, never to release here:
        #   - Normal full transmission (>=1 chunk sent, loop reached "done"
        #     or "error" and fell through to `tts_end`): let the watcher
        #     release it later, as designed.
        #   - Streaming real chunks (>=1 already sent, watcher already
        #     armed) when this function exits *abnormally* mid-stream --
        #     most concretely, `recv_chunk` raising (tts.py:183-196's
        #     `_check_alive()` on worker death + `_MAX_RESTARTS` exceeded).
        #     Audio already sent to the client is still playing; releasing
        #     immediately here would reopen the mic while that's true (the
        #     exact defect this round exists to fix, recurring one level
        #     deeper). Leave the already-armed, already-accurate watcher to
        #     release on schedule here too.
        # Whenever the watcher is *not* armed, nothing was ever queued for
        # the client to play and no watcher exists to guard anything, so
        # release immediately -- covers both "done" arriving with zero
        # chunks ever produced, and never reaching a first chunk at all
        # (empty translation, `start_synthesis` failing, cancellation, or an
        # exception before/during the first `recv_chunk` call).
        #
        # Cancellation and session teardown are still independently safe
        # regardless of this branch: handle_websocket's own `finally`
        # (:228-232) releases explicitly if a window is still active when
        # the session closes.
        if not watcher_armed:
            _release_suppression(session, segment_id)
        session.transmission_active = False


def _language_to_tts_code(language: str) -> str:
    """Map language name to TTS language code."""
    mapping = {
        "english": "en",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
    }
    return mapping.get(language.lower(), "en")


async def _handle_control(
    websocket: WebSocket,
    session: ClientSession,
    data: str,
    ws_lock: asyncio.Lock,
    background_tasks: set[asyncio.Task],
) -> None:
    msg = parse_control_message(data)
    if msg is None:
        async with ws_lock:
            await websocket.send_text(make_error("Invalid JSON"))
        return

    msg_type = msg.get("type")
    if msg_type == "configure":
        config = ConfigureMessage.from_dict(msg)
        session.configure(
            config.language,
            config.target_language,
            mode=config.mode,
            tts_voice=config.tts_voice,
            speak_target_language=config.speak_target_language,
            voice_clone=config.voice_clone,
            mic_mode=config.mic_mode,
        )
        async with ws_lock:
            await websocket.send_text(
                make_status(
                    "configured",
                    language=config.language,
                    target_language=config.target_language,
                    mode=session.mode,
                    tts_voice=session.tts_voice,
                    speak_target_language=session.speak_target_language,
                    voice_clone=session.voice_clone,
                    mic_mode=session.mic_mode,
                )
            )
    elif msg_type == "mic_active":
        active = msg.get("active", False)
        was_active = session.mic_active
        session.mic_active = active
        logger.debug("Session %s: mic_active=%s", session.session_id, active)
        # On release (active → inactive), flush VAD buffer
        if was_active and not active:
            await _flush_mic_and_process(websocket, session, ws_lock, background_tasks)
    elif msg_type == "speak_text":
        text = msg.get("text", "").strip()
        if not text:
            async with ws_lock:
                await websocket.send_text(make_error("Empty text"))
            return
        # `transmission_active` (Critic round 3, major 2): same interlock as
        # _process_speak_segment's gate -- see _translate_and_speak's
        # docstring.
        if session.tts_playing or session.transmission_active:
            async with ws_lock:
                await websocket.send_text(make_error("TTS busy"))
            return
        logger.info("[type] speak_text: '%s' | speak_target='%s' voice='%s' clone=%s ref=%s",
                    text[:80], session.speak_target_language,
                    session.tts_voice, session.voice_clone,
                    "yes" if session.ref_audio is not None else "no")
        session.tts_playing = True
        session.mic_buffer.clear()
        session.mic_vad.reset()
        segment_id = session.next_segment_id()
        _start_suppression(session, segment_id)
        async with ws_lock:
            await websocket.send_text(make_speak_transcription(text, segment_id))
        task = asyncio.create_task(
            _translate_and_speak(
                websocket, ws_lock, text,
                session.speak_target_language,
                session.tts_voice,
                segment_id,
                session,
                background_tasks,
            )
        )
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
    elif msg_type == "playback_complete":
        segment_id = msg.get("segment_id")
        logger.debug(
            "Session %s: playback_complete segment_id=%s", session.session_id, segment_id
        )
        # Release wiring (T028): a playback_complete for an unknown, stale,
        # or already-released segment_id is a no-op, not an error (FR-012)
        # -- _release_suppression enforces that. `bool` is excluded even
        # though it's a subclass of `int` in Python: {"segment_id": true}
        # must not be treated as segment_id 1.
        if isinstance(segment_id, int) and not isinstance(segment_id, bool):
            _release_suppression(session, segment_id)
    else:
        async with ws_lock:
            await websocket.send_text(make_error(f"Unknown message type: {msg_type}"))
