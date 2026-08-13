import logging
from dataclasses import dataclass

import numpy as np

from .audio_buffer import AudioBuffer
from .config import (
    MIC_VAD_SILENCE_WINDOWS,
    SUBTITLE_PACE_DEFAULT,
    SUBTITLE_PACE_MAX_S_RANGE,
    SUBTITLE_PACE_PRESETS,
    SUBTITLE_PACE_SILENCE_MS_RANGE,
    SUBTITLE_PACE_SPEECH_S_RANGE,
    TTS_DEFAULT_VOICE,
    VAD_WINDOW_DURATION_MS,
)
from .vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


def _clamp(value: float, bounds: tuple[int, int], fallback: int) -> int:
    """Coerce a client-supplied number into `bounds`, or fall back."""
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return fallback
    low, high = bounds
    return max(low, min(high, number))


def clamp_pace(
    speech_s: float, silence_ms: float, max_s: float
) -> tuple[int, int, int]:
    """Sanitise a hand-entered subtitle pace.

    These reach the audio path straight from a settings field, so every value
    is clamped rather than trusted, and a `max_s` below `speech_s` is raised to
    meet it -- otherwise the forced split fires before the intermediate one can
    ever apply, and the setting silently does nothing.
    """
    defaults = SUBTITLE_PACE_PRESETS[SUBTITLE_PACE_DEFAULT]
    speech = _clamp(speech_s, SUBTITLE_PACE_SPEECH_S_RANGE, defaults[0])
    silence = _clamp(silence_ms, SUBTITLE_PACE_SILENCE_MS_RANGE, defaults[1])
    maximum = _clamp(max_s, SUBTITLE_PACE_MAX_S_RANGE, defaults[2])
    return speech, silence, max(maximum, speech)


def pace_windows(speech_s: float, silence_ms: float, max_s: float) -> tuple[int, int, int]:
    """Convert a subtitle pace, in human units, into VAD window counts."""
    speech, silence, maximum = clamp_pace(speech_s, silence_ms, max_s)
    return (
        speech * 1000 // VAD_WINDOW_DURATION_MS,
        silence // VAD_WINDOW_DURATION_MS,
        maximum * 1000 // VAD_WINDOW_DURATION_MS,
    )


@dataclass
class SuppressionWindow:
    """Tracks one segment's microphone-suppression window
    (data-model.md §4). Replaces the unconditional `tts_playing = False`
    release that used to run in `_translate_and_speak`'s `finally` at
    transmission end: release now happens whichever comes first, of
    `playback_complete` arriving for `segment_id` or `deadline` passing.
    """

    segment_id: int
    # Kept for data-model.md §4 schema fidelity; the deadline itself is
    # computed from the chunk loop's running total in _translate_and_speak,
    # so nothing reads this field back today.
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
        self.subtitle_pace = SUBTITLE_PACE_PRESETS[SUBTITLE_PACE_DEFAULT]
        speech_w, silence_w, max_w = pace_windows(*self.subtitle_pace)
        self.vad = VoiceActivityDetector(
            intermediate_speech_windows=speech_w,
            intermediate_silence_windows=silence_w,
            max_speech_windows=max_w,
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
        self.translation_prompt = ""  # "" = TRANSLATION_PROMPT_DEFAULT
        # Push-to-talk hold buffer: raw chunks accumulated while the key is
        # held, committed as one utterance on release. Never fed to mic_vad --
        # the hold defines the segment boundary, not silence.
        self.mic_hold_chunks: list[np.ndarray] = []

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
        translation_prompt: str = "",
        subtitle_pace: tuple[float, float, float] | None = None,
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
        # Empty means "use the server default" -- assigned unconditionally so
        # clearing the field in settings restores the default mid-session.
        self.translation_prompt = translation_prompt
        if subtitle_pace is not None:
            cleaned = clamp_pace(*subtitle_pace)
            if cleaned != self.subtitle_pace:
                self.subtitle_pace = cleaned
                # Retuned in place: rebuilding would reload the Silero model
                # and drop whatever is mid-utterance.
                self.vad.set_pace(*pace_windows(*cleaned))
        logger.info(
            "Session %s: language=%s, target=%s, mode=%s, voice=%s, speak_target=%s, clone=%s, mic=%s, pace=%s",
            self.session_id, language, target_language,
            self.mode, self.tts_voice, self.speak_target_language,
            self.voice_clone, self.mic_mode, self.subtitle_pace,
        )

    def reset(self) -> None:
        self.audio_buffer.clear()
        self.vad.reset()
        self.mic_buffer.clear()
        self.mic_vad.reset()
        self.mic_hold_chunks = []

    def close(self) -> None:
        """Full cleanup: release buffers, VAD models, and references."""
        self.reset()
        self.ref_audio = None
        self.ref_text = ""
        self.vad._model = None
        self.mic_vad._model = None
        logger.info("Session %s: closed and resources released", self.session_id)
