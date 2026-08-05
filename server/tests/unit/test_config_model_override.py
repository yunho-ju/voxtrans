"""Tests for the `VOXTRANS_TRANSLATION_MODEL` env override in
`server/src/config.py` (research.md D8, tasks.md T026, FR-003).

`config.py` module-level constants are resolved once, at import time, so
each case here mutates the real `os.environ` directly and reloads the
module to observe the effect -- then restores the *exact* prior
environment state and reloads again in a `finally`, regardless of whether
the assertion passed, so no ambient state leaks into any other test in the
suite (Gauntlet standing constraint 1).
"""

from __future__ import annotations

import importlib
import os

import src.config as config

_ENV_VAR = "VOXTRANS_TRANSLATION_MODEL"


def _reload_with_env(value: str | None) -> None:
    if value is None:
        os.environ.pop(_ENV_VAR, None)
    else:
        os.environ[_ENV_VAR] = value
    importlib.reload(config)


def test_translation_model_resolves_from_env_var_when_set():
    original = os.environ.get(_ENV_VAR)
    try:
        _reload_with_env("acme/other-translation-model")
        assert config.TRANSLATION_MODEL == "acme/other-translation-model"
    finally:
        _reload_with_env(original)


def test_translation_model_falls_back_to_constant_when_unset():
    original = os.environ.get(_ENV_VAR)
    try:
        _reload_with_env(None)
        assert config.TRANSLATION_MODEL == config._TRANSLATION_MODEL_DEFAULT
    finally:
        _reload_with_env(original)


def test_translation_model_treats_empty_env_value_as_unset():
    """FR-003: an empty value (e.g. `VOXTRANS_TRANSLATION_MODEL=` in a shell
    profile) must not be treated as an explicit override -- it falls back to
    the constant exactly like the unset case above."""
    original = os.environ.get(_ENV_VAR)
    try:
        _reload_with_env("")
        assert config.TRANSLATION_MODEL == config._TRANSLATION_MODEL_DEFAULT
    finally:
        _reload_with_env(original)


def test_translation_model_treats_whitespace_only_env_value_as_unset():
    """Critic round-2 minor 1: a whitespace-only value (e.g. a shell profile
    line like `VOXTRANS_TRANSLATION_MODEL=" "`) is non-empty and therefore
    truthy to a bare `or` check -- it must still resolve to "unset", exactly
    like the empty-string case above, not be adopted as a literal
    whitespace model id."""
    original = os.environ.get(_ENV_VAR)
    try:
        _reload_with_env("   ")
        assert config.TRANSLATION_MODEL == config._TRANSLATION_MODEL_DEFAULT
    finally:
        _reload_with_env(original)


def test_translation_model_strips_surrounding_whitespace_from_a_real_override():
    original = os.environ.get(_ENV_VAR)
    try:
        _reload_with_env("  acme/other-translation-model  ")
        assert config.TRANSLATION_MODEL == "acme/other-translation-model"
    finally:
        _reload_with_env(original)
