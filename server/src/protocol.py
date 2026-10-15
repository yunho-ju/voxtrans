from dataclasses import dataclass
from enum import Enum
from typing import Any
import json


class MessageType(str, Enum):
    CONFIGURE = "configure"
    MIC_ACTIVE = "mic_active"
    TRANSCRIPTION_PARTIAL = "transcription_partial"
    TRANSCRIPTION_FINAL = "transcription_final"
    TRANSLATION = "translation"
    SPEAK_TRANSCRIPTION = "speak_transcription"
    SPEAK_TRANSLATION = "speak_translation"
    SPEAK_TEXT = "speak_text"
    TTS_END = "tts_end"
    PLAYBACK_COMPLETE = "playback_complete"
    ERROR = "error"
    STATUS = "status"


def _parse_pace(raw) -> tuple[float, float, float] | None:
    """Read a subtitle pace out of a configure message.

    Anything that is not three numbers is treated as "not supplied" -- the
    session then keeps its current pace instead of adopting a malformed one.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        return None


@dataclass
class ConfigureMessage:
    language: str = "auto"
    target_language: str = ""
    mode: str = "listen"
    tts_voice: str = ""
    # Which TTS backend to run. "" means "leave whatever is loaded alone" --
    # a client that does not know about backends must not reset one.
    tts_backend: str = ""
    speak_target_language: str = ""
    voice_clone: bool = True
    mic_mode: str = "auto"  # "auto" (VAD) or "push_to_talk"
    translation_prompt: str = ""  # "" = server default template
    # (intermediate_speech_s, intermediate_silence_ms, max_speech_s), or None
    # to leave the session's pace alone. Values are clamped server-side.
    subtitle_pace: tuple[float, float, float] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ConfigureMessage":
        return cls(
            language=data.get("language", "auto"),
            target_language=data.get("target_language", ""),
            mode=data.get("mode", "listen"),
            tts_voice=data.get("tts_voice", ""),
            tts_backend=data.get("tts_backend", ""),
            speak_target_language=data.get("speak_target_language", ""),
            voice_clone=data.get("voice_clone", True),
            mic_mode=data.get("mic_mode", "auto"),
            translation_prompt=data.get("translation_prompt", ""),
            subtitle_pace=_parse_pace(data.get("subtitle_pace")),
        )


def make_transcription(text: str, is_final: bool, segment_id: int = 0) -> str:
    msg_type = MessageType.TRANSCRIPTION_FINAL if is_final else MessageType.TRANSCRIPTION_PARTIAL
    return json.dumps({
        "type": msg_type.value,
        "text": text,
        "is_final": is_final,
        "segment_id": segment_id,
    })


def make_translation(text: str, segment_id: int) -> str:
    return json.dumps({
        "type": MessageType.TRANSLATION.value,
        "text": text,
        "segment_id": segment_id,
    })


def make_speak_transcription(text: str, segment_id: int) -> str:
    return json.dumps({
        "type": MessageType.SPEAK_TRANSCRIPTION.value,
        "text": text,
        "segment_id": segment_id,
    })


def make_speak_translation(text: str, segment_id: int) -> str:
    return json.dumps({
        "type": MessageType.SPEAK_TRANSLATION.value,
        "text": text,
        "segment_id": segment_id,
    })


def make_tts_end(segment_id: int) -> str:
    return json.dumps({
        "type": MessageType.TTS_END.value,
        "segment_id": segment_id,
    })


def make_error(message: str) -> str:
    return json.dumps({
        "type": MessageType.ERROR.value,
        "message": message,
    })


def make_status(status: str, **kwargs: Any) -> str:
    return json.dumps({
        "type": MessageType.STATUS.value,
        "status": status,
        **kwargs,
    })


def parse_control_message(data: str) -> dict | None:
    try:
        msg = json.loads(data)
        if isinstance(msg, dict) and "type" in msg:
            return msg
    except json.JSONDecodeError:
        pass
    return None
