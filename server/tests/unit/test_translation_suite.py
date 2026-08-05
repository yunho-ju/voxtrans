"""Tests for `bench.suites.translation` (T020 round-2 Critic major 2: the
suite measured no warm-up generation, so iteration 0's cold first call sat
inside the reported mean).

Hermetic: `mlx_lm.load`/`mlx_lm.stream_generate` are monkeypatched on the
real `mlx_lm` module -- `translation.py` does `from mlx_lm import load,
stream_generate` *inside* `run()`, so a call made after the patch sees the
patched attributes. No model is ever loaded.
"""

from __future__ import annotations

import mlx_lm
import pytest

import bench.suites.translation as translation


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "PROMPT"


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


def _fake_load(model_id: str):
    return "MODEL", _FakeTokenizer()


def test_run_generates_an_untimed_warmup_pass_before_the_timed_iterations(monkeypatch):
    calls: list[int] = []

    def _fake_stream_generate(model, tokenizer, prompt, max_tokens):
        calls.append(max_tokens)
        yield _FakeResponse("hi")

    monkeypatch.setattr(mlx_lm, "load", _fake_load)
    monkeypatch.setattr(mlx_lm, "stream_generate", _fake_stream_generate)

    stages, samples = translation.run(iterations=2, config={"translation_model": "fake-model"})

    # The suite's real generations always ask for max_tokens=512; the
    # warm-up in `_warm_up` deliberately asks for a smaller 32, so the first
    # recorded call being 32 is proof the warm-up ran, and ran *before* the
    # two measured iterations that follow it.
    assert calls[0] == 32, "the first stream_generate call must be the untimed warm-up"
    assert calls[1:] == [512, 512], "the two measured iterations must follow the warm-up, unchanged"
    assert len(samples) == 2
    assert all(sample["warmed_up"] is True for sample in samples)
    assert {s.stage for s in stages} == {"translation", "translation_first_token"}


def test_run_raises_when_warmup_generation_fails_and_records_no_sample(monkeypatch):
    def _failing_stream_generate(model, tokenizer, prompt, max_tokens):
        raise RuntimeError("synthetic warm-up failure")
        yield  # pragma: no cover -- unreachable; keeps this a generator function

    monkeypatch.setattr(mlx_lm, "load", _fake_load)
    monkeypatch.setattr(mlx_lm, "stream_generate", _failing_stream_generate)

    with pytest.raises(RuntimeError, match="synthetic warm-up failure"):
        translation.run(iterations=1, config={"translation_model": "fake-model"})


def test_run_raises_when_warmup_generation_yields_zero_responses(monkeypatch):
    """T020 round-3 Critic minor: a warm-up that silently produces nothing
    (an empty `stream_generate`) must not be treated as a successful
    warm-up -- it would leave the timed loop measuring a cold model while
    still recording `"warmed_up": True`."""

    def _empty_stream_generate(model, tokenizer, prompt, max_tokens):
        return iter(())

    monkeypatch.setattr(mlx_lm, "load", _fake_load)
    monkeypatch.setattr(mlx_lm, "stream_generate", _empty_stream_generate)

    with pytest.raises(RuntimeError, match="zero responses"):
        translation.run(iterations=1, config={"translation_model": "fake-model"})
