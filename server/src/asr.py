import logging
import tempfile
import threading

import numpy as np
import soundfile as sf

from .config import ASR_MODEL, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Map lowercase language names to capitalized names expected by ASR model
_LANGUAGE_MAP = {
    "korean": "Korean",
    "english": "English",
    "japanese": "Japanese",
    "chinese": "Chinese",
    "spanish": "Spanish",
    "french": "French",
    "german": "German",
    "italian": "Italian",
    "portuguese": "Portuguese",
    "russian": "Russian",
}


class ASREngine:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import time
        from mlx_audio.stt import load
        logger.info("Loading ASR model: %s", ASR_MODEL)
        t0 = time.time()
        self._model = load(ASR_MODEL)
        logger.info("ASR model loaded in %.1fs", time.time() - t0)

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> str:
        """Transcribe float32 mono 16kHz audio array. Returns text."""
        if self._model is None:
            raise RuntimeError("ASR model not loaded")
        if audio.size == 0:
            return ""

        duration_s = audio.size / SAMPLE_RATE
        logger.info("ASR: transcribing %.1fs audio, language='%s'", duration_s, language)

        with self._lock:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio, SAMPLE_RATE)
                kwargs = {}
                if language and language != "auto":
                    # Map to capitalized name expected by Qwen3-ASR model
                    lang_name = _LANGUAGE_MAP.get(language.lower(), language.capitalize())
                    kwargs["language"] = lang_name
                    logger.info("ASR: using explicit language='%s'", lang_name)
                else:
                    # "auto" mode: model defaults to English output.
                    # For reliable non-English transcription, set explicit language.
                    logger.info("ASR: auto mode (model defaults to English)")
                result = self._model.generate(f.name, **kwargs)

        text = result.text.strip() if hasattr(result, "text") else str(result).strip()
        logger.info("ASR result: language='%s' → text='%s'", language, text[:100])
        return text


# Singleton
asr_engine = ASREngine()
