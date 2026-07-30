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
    ERROR = "error"
    STATUS = "status"


@dataclass
class ConfigureMessage:
    language: str = "auto"
    target_language: str = ""
    mode: str = "listen"
    tts_voice: str = ""
    speak_target_language: str = ""
    voice_clone: bool = True
    mic_mode: str = "auto"  # "auto" (VAD) or "push_to_talk"

    @classmethod
    def from_dict(cls, data: dict) -> "ConfigureMessage":
        return cls(
            language=data.get("language", "auto"),
            target_language=data.get("target_language", ""),
            mode=data.get("mode", "listen"),
            tts_voice=data.get("tts_voice", ""),
            speak_target_language=data.get("speak_target_language", ""),
            voice_clone=data.get("voice_clone", True),
            mic_mode=data.get("mic_mode", "auto"),
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
