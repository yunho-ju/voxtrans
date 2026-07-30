import asyncio
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .asr import asr_engine
from .config import AUDIO_PREFIX_MIC, AUDIO_PREFIX_SYSTEM, AUDIO_PREFIX_TTS, MAX_SESSIONS

# Max pending background tasks per session (rate limiting)
MAX_PENDING_TASKS = 10
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
from .session import ClientSession
from .translation import translation_engine
from .tts import tts_engine

logger = logging.getLogger(__name__)

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
    if session.tts_playing:
        return
    text = await _transcribe(speech_segment, session.language)
    if not text:
        return
    if session.tts_playing:  # re-check after transcribe (may have changed during await)
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

    task = asyncio.create_task(
        _translate_and_speak(
            websocket, ws_lock, text,
            session.speak_target_language,
            session.tts_voice,
            segment_id,
            session,
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
) -> None:
    """Translate text, then stream TTS audio chunks as they are generated.

    Note: session.tts_playing is already True (set by _process_speak_segment).
    This function MUST reset it in all exit paths (guaranteed by finally block).
    """
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

        # 3. Start streaming TTS synthesis
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
        total_samples = 0
        while True:
            status, audio_np, sr = await loop.run_in_executor(
                None, tts_engine.recv_chunk, 10.0,
            )
            if status == "chunk":
                total_samples += audio_np.size
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

        # Minimal delay for buffer flush only (BlackHole has no speaker→mic feedback)
        await asyncio.sleep(0.3)

    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Translate+speak failed for segment %d", segment_id)
    finally:
        # Always re-enable mic regardless of how we exit
        session.mic_buffer.clear()
        session.mic_vad.reset()
        session.tts_playing = False


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
        if session.tts_playing:
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
        async with ws_lock:
            await websocket.send_text(make_speak_transcription(text, segment_id))
        task = asyncio.create_task(
            _translate_and_speak(
                websocket, ws_lock, text,
                session.speak_target_language,
                session.tts_voice,
                segment_id,
                session,
            )
        )
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
    else:
        async with ws_lock:
            await websocket.send_text(make_error(f"Unknown message type: {msg_type}"))
