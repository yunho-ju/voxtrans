"""The translation prompt is user-editable from settings, so every way a
hand-typed template can be wrong must degrade to the default rather than
produce empty or malformed translations.

`render_prompt` is a pure static method: no model load, no MLX, no devices.
"""

import pytest

from src.config import TRANSLATION_PROMPT_DEFAULT, TRANSLATION_PROMPT_MAX_LENGTH
from src.protocol import ConfigureMessage
from src.translation import TranslationEngine


render = TranslationEngine.render_prompt


def test_a_custom_template_is_used_verbatim():
    out = render("Say it in {lang}, casually:\n{text}", "Korean", "hello")
    assert out == "Say it in Korean, casually:\nhello"


def test_empty_template_falls_back_to_the_default():
    assert render("", "Korean", "hello") == TRANSLATION_PROMPT_DEFAULT.format(
        lang="Korean", text="hello"
    )


def test_whitespace_only_template_falls_back_to_the_default():
    assert render("   \n  ", "Korean", "hello").endswith("hello")


def test_a_template_without_text_falls_back_rather_than_translating_nothing():
    """The dangerous typo: without {text} the model receives an instruction
    and no transcript, and would answer something plausible and wrong.
    """
    out = render("Translate everything to {lang}.", "Korean", "hello")
    assert "hello" in out
    assert out == TRANSLATION_PROMPT_DEFAULT.format(lang="Korean", text="hello")


def test_an_unknown_placeholder_falls_back_instead_of_raising():
    out = render("To {lang} for {audience}: {text}", "Korean", "hello")
    assert out == TRANSLATION_PROMPT_DEFAULT.format(lang="Korean", text="hello")


def test_a_stray_brace_falls_back_instead_of_raising():
    out = render("Translate this { to {lang}: {text}", "Korean", "hello")
    assert out == TRANSLATION_PROMPT_DEFAULT.format(lang="Korean", text="hello")


def test_an_oversized_template_falls_back():
    out = render("{text} " + "x" * TRANSLATION_PROMPT_MAX_LENGTH, "Korean", "hello")
    assert out == TRANSLATION_PROMPT_DEFAULT.format(lang="Korean", text="hello")


def test_braces_in_the_transcript_are_not_reinterpreted():
    """The transcript is a value, not part of the template: ASR output
    containing braces must not be treated as a placeholder.
    """
    out = render("{text}", "Korean", "say {lang} out loud")
    assert out == "say {lang} out loud"


def test_lang_alone_is_a_valid_template():
    assert render("{text} -> {lang}", "Japanese", "hi") == "hi -> Japanese"


def test_configure_message_defaults_to_the_server_template():
    assert ConfigureMessage.from_dict({"type": "configure"}).translation_prompt == ""


def test_configure_message_carries_a_custom_template():
    msg = ConfigureMessage.from_dict(
        {"type": "configure", "translation_prompt": "{text} in {lang}"}
    )
    assert msg.translation_prompt == "{text} in {lang}"
