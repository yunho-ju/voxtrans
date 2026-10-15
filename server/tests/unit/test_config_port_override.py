"""Tests for the `VOXTRANS_PORT` override in `server/src/config.py`.

No default port is exclusively ours: anything else already listening there is handed
the app's `/health` probe instead of the server, which the app can only read as
"still loading". Moving the server is the way out, so the port has to be an
override rather than a constant -- and a typo in that override must not quietly
put the server back on the port the user was trying to leave.

Same mechanics as `test_config_model_override.py`: `config.py` resolves its
constants once at import time, so each case mutates the real `os.environ`,
reloads, and restores the exact prior state in a `finally`.
"""

from __future__ import annotations

import importlib
import os

import pytest

import src.config as config

_ENV_VAR = "VOXTRANS_PORT"


def _reload_with_env(value: str | None) -> None:
    if value is None:
        os.environ.pop(_ENV_VAR, None)
    else:
        os.environ[_ENV_VAR] = value
    importlib.reload(config)


def test_port_resolves_from_env_var_when_set():
    original = os.environ.get(_ENV_VAR)
    try:
        _reload_with_env("8770")
        assert config.PORT == 8770
    finally:
        _reload_with_env(original)


def test_port_falls_back_to_the_default_when_unset():
    original = os.environ.get(_ENV_VAR)
    try:
        _reload_with_env(None)
        assert config.PORT == config._PORT_DEFAULT
    finally:
        _reload_with_env(original)


@pytest.mark.parametrize("value", ["", "   "])
def test_port_treats_empty_and_whitespace_only_values_as_unset(value):
    """`VOXTRANS_PORT=` in a shell profile is not an override."""
    original = os.environ.get(_ENV_VAR)
    try:
        _reload_with_env(value)
        assert config.PORT == config._PORT_DEFAULT
    finally:
        _reload_with_env(original)


def test_port_strips_surrounding_whitespace_from_a_real_override():
    original = os.environ.get(_ENV_VAR)
    try:
        _reload_with_env("  8770  ")
        assert config.PORT == 8770
    finally:
        _reload_with_env(original)


@pytest.mark.parametrize("value", ["eight-thousand", "87.65", "0", "70000", "-1"])
def test_port_rejects_a_value_that_is_not_a_usable_port(value):
    """Refusing loudly beats falling back: a silent fallback lands the server
    on the default port, the one the override existed to escape, and the app would
    then spend its whole poll window reporting that as "still loading"."""
    original = os.environ.get(_ENV_VAR)
    try:
        with pytest.raises(ValueError, match="VOXTRANS_PORT"):
            _reload_with_env(value)
    finally:
        _reload_with_env(original)
