import logging
import threading

from .config import (
    TRANSLATION_MODEL,
    TRANSLATION_PROMPT_DEFAULT,
    TRANSLATION_PROMPT_MAX_LENGTH,
)

logger = logging.getLogger(__name__)


class TranslationEngine:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._generate = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import time
        from mlx_lm import load, generate
        logger.info("Loading translation model: %s", TRANSLATION_MODEL)
        t0 = time.time()
        self._model, self._tokenizer = load(TRANSLATION_MODEL)
        self._generate = generate
        logger.info("Translation model loaded in %.1fs", time.time() - t0)

    VALID_LANGUAGES = {
        "korean", "english", "japanese", "chinese", "spanish",
        "french", "german", "italian", "portuguese", "russian",
    }

    MAX_INPUT_LENGTH = 2000

    @staticmethod
    def render_prompt(template: str, lang_display: str, text: str) -> str:
        """Fill a prompt template, falling back to the default if it is unusable.

        A user-authored template reaches here from the app's settings, so every
        failure mode is a typo, not a bug: a missing `{text}` placeholder would
        translate an empty string, and an unknown placeholder or a stray brace
        raises from `str.format`. Both fall back to the default template -- a
        broken prompt must not silently produce empty or garbage translations.
        """
        candidate = (template or "").strip()
        if not candidate or "{text}" not in candidate:
            if candidate:
                logger.warning("Translation prompt lacks {text}; using default")
            candidate = TRANSLATION_PROMPT_DEFAULT
        if len(candidate) > TRANSLATION_PROMPT_MAX_LENGTH:
            logger.warning(
                "Translation prompt is %d chars (max %d); using default",
                len(candidate), TRANSLATION_PROMPT_MAX_LENGTH,
            )
            candidate = TRANSLATION_PROMPT_DEFAULT
        try:
            return candidate.format(lang=lang_display, text=text)
        except (KeyError, IndexError, ValueError):
            logger.warning("Translation prompt has a bad placeholder; using default")
            return TRANSLATION_PROMPT_DEFAULT.format(lang=lang_display, text=text)

    def translate(self, text: str, target_language: str, prompt_template: str = "") -> str:
        """Translate text to target_language. Returns translated text.

        `prompt_template` comes from the session's settings; empty means the
        built-in default (`TRANSLATION_PROMPT_DEFAULT`).
        """
        if self._model is None:
            raise RuntimeError("Translation model not loaded")
        if not text.strip():
            return ""

        # Truncate overly long input to prevent model context overflow
        if len(text) > self.MAX_INPUT_LENGTH:
            logger.warning("Translation input truncated from %d to %d chars", len(text), self.MAX_INPUT_LENGTH)
            text = text[:self.MAX_INPUT_LENGTH]

        # Sanitize language name
        lang = target_language.strip().lower()
        if lang not in self.VALID_LANGUAGES:
            lang = "english"
        lang_display = lang.capitalize()

        prompt = self.render_prompt(prompt_template, lang_display, text)
        logger.info("Translation: target='%s' input='%s'", lang_display, text[:80])
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        with self._lock:
            result = self._generate(
                self._model,
                self._tokenizer,
                prompt=chat_prompt,
                max_tokens=512,
            )
        # Strip any residual think tags
        translated = result.strip()
        if "</think>" in translated:
            translated = translated.split("</think>")[-1].strip()
        logger.info("Translation result: '%s'", translated[:80])
        return translated


# Singleton
translation_engine = TranslationEngine()
