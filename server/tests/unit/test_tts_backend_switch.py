"""Switching the TTS backend at runtime, without loading a model.

The worker process and the model load are stubbed out: what is under test is
the decision to reload (or not) and what state survives it.
"""

import pytest

from src import tts as tts_module
from src.protocol import ConfigureMessage
from src.tts import TTSEngine


@pytest.fixture
def engine(monkeypatch):
    """A TTSEngine whose worker lifecycle records instead of spawning."""
    eng = TTSEngine()
    loads: list[tuple[str, str]] = []
    class RunningWorker:
        def is_alive(self):
            return True

    eng._process = RunningWorker()

    def fake_load(self):
        loads.append((self._backend_name, self._model_name))
        self._process = RunningWorker()

    def fake_shutdown(self):
        self._process = None

    monkeypatch.setattr(
        TTSEngine, "_load_locked", fake_load,
    )
    monkeypatch.setattr(TTSEngine, "_shutdown_locked", fake_shutdown)
    eng.loads = loads
    return eng


def test_switching_reloads_the_worker_on_the_new_backend(engine):
    engine.switch_backend("kokoro")

    assert engine.backend == "kokoro"
    assert engine.loads == [("kokoro", "prince-canuma/Kokoro-82M")]


def test_switching_to_the_running_backend_does_nothing(engine):
    # The reload costs a model load and silences TTS while it runs, so a
    # configure message that merely repeats the current backend must not
    # trigger one -- the app sends configure on every settings change.
    engine.switch_backend(engine.backend)

    assert engine.loads == []


def test_an_unknown_backend_leaves_the_running_one_alone(engine):
    before = engine.backend

    with pytest.raises(ValueError, match="unknown TTS backend"):
        engine.switch_backend("kokoros")

    assert engine.backend == before
    assert engine.loads == []


def test_switching_resets_the_crash_allowance(engine):
    # The previous backend's crashes say nothing about the new one's; carrying
    # the count over would let a fresh backend inherit an exhausted budget and
    # refuse to start.
    engine._restart_count = TTSEngine._MAX_RESTARTS

    engine.switch_backend("kokoro")

    assert engine._restart_count == 0


def test_switching_back_uses_the_configured_model_for_that_backend(engine, monkeypatch):
    # `VOXTRANS_TTS_MODEL` overrides the repo for the *configured* backend
    # only -- carrying that override onto the other backend would try to load
    # a Qwen3 repo as Kokoro.
    monkeypatch.setattr(tts_module, "TTS_BACKEND", "qwen3")
    monkeypatch.setattr(tts_module, "TTS_MODEL", "someone/Qwen3-TTS-fork")

    engine.switch_backend("kokoro")
    engine.switch_backend("qwen3")

    assert engine.loads[0] == ("kokoro", "prince-canuma/Kokoro-82M")
    assert engine.loads[1] == ("qwen3", "someone/Qwen3-TTS-fork")


def test_configure_carries_the_backend_and_defaults_to_leaving_it_alone():
    assert ConfigureMessage.from_dict({"tts_backend": "kokoro"}).tts_backend == "kokoro"
    # A client that knows nothing about backends must not reset the server's.
    assert ConfigureMessage.from_dict({}).tts_backend == ""
