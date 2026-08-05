import logging
from dataclasses import dataclass

import numpy as np

from .audio_buffer import AudioBuffer
from .config import (
    MIC_VAD_SILENCE_WINDOWS,
    TTS_DEFAULT_VOICE,
    VAD_INTERMEDIATE_SILENCE_WINDOWS,
    VAD_INTERMEDIATE_SPEECH_WINDOWS,
)
from .vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


@dataclass
class SuppressionWindow:
    """Tracks one segment's microphone-suppression window
    (data-model.md §4). Replaces the unconditional `tts_playing = False`
    release that used to run in `_translate_and_speak`'s `finally` at
    transmission end: release now happens whichever comes first, of
    `playback_complete` arriving for `segment_id` or `deadline` passing.
    """

    segment_id: int
    synthesized_samples: int
    deadline: float  # time.monotonic() timestamp
    released: bool = False


class ClientSession:
    """Per-client session holding VAD state and audio buffer."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.language = "auto"
        self.target_language = ""
        self.audio_buffer = AudioBuffer()
        self.vad = VoiceActivityDetector(
            intermediate_speech_windows=VAD_INTERMEDIATE_SPEECH_WINDOWS,
            intermediate_silence_windows=VAD_INTERMEDIATE_SILENCE_WINDOWS,
        )
        self._segment_counter = 0

        # Speak mode (mic → TTS pipeline)
        self.mode = "listen"
        self.tts_voice = TTS_DEFAULT_VOICE
        self.speak_target_language = "English"
        self.mic_buffer = AudioBuffer()
        self.mic_vad = VoiceActivityDetector(silence_windows=MIC_VAD_SILENCE_WINDOWS)
        self.tts_playing = False  # suppress mic while TTS is active
        self.suppression: SuppressionWindow | None = None  # active release window, if any
        self.transmission_active = False  # True for _translate_and_speak's
        # whole lifetime, independent of tts_playing/suppression release
        # timing -- see ws_handler._translate_and_speak (Critic round 3):
        # the "no concurrent transmissions" gate must hold even if
        # suppression is ever released early, for any reason.
        self.ref_audio: np.ndarray | None = None  # voice cloning reference
        self.ref_text: str = ""
        self.voice_clone = True  # use voice cloning (ref_audio) or preset voice
        self.mic_mode = "auto"  # "auto" (VAD) or "push_to_talk"
        self.mic_active = False  # for push-to-talk: True while key held

    def next_segment_id(self) -> int:
        self._segment_counter += 1
        return self._segment_counter

    def configure(
        self,
        language: str,
        target_language: str = "",
        mode: str = "",
        tts_voice: str = "",
        speak_target_language: str = "",
        voice_clone: bool = True,
        mic_mode: str = "",
    ) -> None:
        self.language = language
        self.target_language = target_language
        if mode:
            self.mode = mode
        if tts_voice:
            self.tts_voice = tts_voice
        if speak_target_language:
            self.speak_target_language = speak_target_language
        self.voice_clone = voice_clone
        if mic_mode:
            self.mic_mode = mic_mode
        logger.info(
            "Session %s: language=%s, target=%s, mode=%s, voice=%s, speak_target=%s, clone=%s, mic=%s",
            self.session_id, language, target_language,
            self.mode, self.tts_voice, self.speak_target_language,
            self.voice_clone, self.mic_mode,
        )

    def reset(self) -> None:
        self.audio_buffer.clear()
        self.vad.reset()
        self.mic_buffer.clear()
        self.mic_vad.reset()

    def close(self) -> None:
        """Full cleanup: release buffers, VAD models, and references."""
        self.reset()
        self.ref_audio = None
        self.ref_text = ""
        self.vad._model = None
        self.mic_vad._model = None
        logger.info("Session %s: closed and resources released", self.session_id)
