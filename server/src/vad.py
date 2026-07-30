import logging
from enum import Enum

import numpy as np
import torch

from .config import (
    SAMPLE_RATE,
    VAD_INTERMEDIATE_SILENCE_WINDOWS,
    VAD_INTERMEDIATE_SPEECH_WINDOWS,
    VAD_MAX_SPEECH_WINDOWS,
    VAD_MIN_SPEECH_WINDOWS,
    VAD_SILENCE_WINDOWS,
    VAD_THRESHOLD,
    VAD_WINDOW_SAMPLES,
)

logger = logging.getLogger(__name__)

# Whether torch.hub.load has been called at least once (caches the download)
_vad_loaded = False


def load_vad_model() -> None:
    """Pre-download/cache the Silero VAD model so per-session loads are fast."""
    global _vad_loaded
    logger.info("Loading Silero VAD model...")
    torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    _vad_loaded = True
    logger.info("Silero VAD loaded and cached")


def create_vad_model():
    """Create a fresh, independent VAD model instance."""
    if not _vad_loaded:
        raise RuntimeError("VAD not loaded. Call load_vad_model() first.")
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    model.eval()
    return model


class VadState(Enum):
    SILENCE = "silence"
    SPEECH = "speech"


class VoiceActivityDetector:
    """Silero VAD wrapper with state machine for speech segmentation.

    Accepts arbitrary-length audio chunks, internally windows them into
    512-sample frames as required by Silero VAD.
    """

    def __init__(
        self,
        silence_windows: int = VAD_SILENCE_WINDOWS,
        intermediate_speech_windows: int = 0,
        intermediate_silence_windows: int = 0,
    ) -> None:
        self._model = create_vad_model()
        self._silence_windows_threshold = silence_windows
        self._intermediate_speech_windows = intermediate_speech_windows
        self._intermediate_silence_windows = intermediate_silence_windows
        self._remainder = np.array([], dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._state = VadState.SILENCE
        self._speech_chunks: list[np.ndarray] = []
        self._speech_window_count = 0
        self._silence_counter = 0
        self._remainder = np.array([], dtype=np.float32)
        if self._model is not None:
            self._model.reset_states()

    def process_chunk(self, chunk: np.ndarray) -> list[np.ndarray]:
        """Process an audio chunk of any size.

        Returns a list of complete speech segments (may be empty, one, or more
        if a forced split occurs within a single chunk).
        """
        # Prepend any leftover samples from previous call
        if self._remainder.size > 0:
            audio = np.concatenate([self._remainder, chunk])
        else:
            audio = chunk

        results = []
        offset = 0
        while offset + VAD_WINDOW_SAMPLES <= len(audio):
            window = audio[offset : offset + VAD_WINDOW_SAMPLES].copy()
            offset += VAD_WINDOW_SAMPLES
            segment = self._process_window(window)
            if segment is not None:
                results.append(segment)

        # Save leftover for next call
        self._remainder = audio[offset:].copy() if offset < len(audio) else np.array([], dtype=np.float32)
        return results

    @torch.inference_mode()
    def _process_window(self, window: np.ndarray) -> np.ndarray | None:
        """Process a single 512-sample VAD window."""
        tensor = torch.from_numpy(window)
        prob = self._model(tensor, SAMPLE_RATE).item()

        if self._state == VadState.SILENCE:
            if prob >= VAD_THRESHOLD:
                self._state = VadState.SPEECH
                self._speech_chunks = [window]
                self._speech_window_count = 1
                self._silence_counter = 0
                logger.debug("VAD: SILENCE -> SPEECH (prob=%.3f)", prob)
            return None

        # state == SPEECH
        self._speech_chunks.append(window)
        self._speech_window_count += 1

        if prob < VAD_THRESHOLD:
            self._silence_counter += 1
        else:
            self._silence_counter = 0

        # Force split at max duration
        if self._speech_window_count >= VAD_MAX_SPEECH_WINDOWS:
            logger.debug("VAD: forced split at %d windows", self._speech_window_count)
            return self._flush()

        # Intermediate split: after long continuous speech, split at short pauses
        if (
            self._intermediate_speech_windows > 0
            and self._speech_window_count >= self._intermediate_speech_windows
            and self._silence_counter >= self._intermediate_silence_windows
        ):
            logger.debug(
                "VAD: intermediate split at %d windows (silence=%d)",
                self._speech_window_count, self._silence_counter,
            )
            return self._flush()

        # Silence timeout -> end of speech
        if self._silence_counter >= self._silence_windows_threshold:
            if self._speech_window_count >= VAD_MIN_SPEECH_WINDOWS:
                logger.debug(
                    "VAD: SPEECH -> SILENCE (windows=%d, silence=%d)",
                    self._speech_window_count,
                    self._silence_counter,
                )
                return self._flush()
            else:
                logger.debug("VAD: discarding short speech (%d windows)", self._speech_window_count)
                self._reset_to_silence()
                return None

        return None

    def flush(self) -> np.ndarray | None:
        """Force-flush any accumulated speech, regardless of silence count."""
        if self._state == VadState.SPEECH and self._speech_window_count >= VAD_MIN_SPEECH_WINDOWS:
            logger.debug("VAD: forced flush (%d windows)", self._speech_window_count)
            return self._flush()
        self._reset_to_silence()
        return None

    def _flush(self) -> np.ndarray:
        audio = np.concatenate(self._speech_chunks)
        self._reset_to_silence()
        return audio

    def _reset_to_silence(self) -> None:
        self._state = VadState.SILENCE
        self._speech_chunks = []
        self._speech_window_count = 0
        self._silence_counter = 0
        if self._model is not None:
            self._model.reset_states()
