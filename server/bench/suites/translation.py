"""`translation` suite (contracts/benchmark-cli.md, tasks.md T016).

Measures translation-only latency and first-token latency for the
configured (or `--model`-overridden) translation model, on a fixed prompt.
Loads the model directly via `mlx_lm.load`/`mlx_lm.stream_generate` rather
than going through `src.translation.TranslationEngine`: this Phase-2 round
only needs a stream to time first-token latency, and
`TranslationEngine.translate_stream()` does not exist until Phase 5
(tasks.md T047) -- this suite has no dependency on that later change.

Runs one untimed warm-up generation before the timed loop (Critic finding,
T020 round 2, major 2): without it, iteration 0's cold first call landed
inside the 3-iteration mean, at roughly a 13% systematic bias on this
machine, and that bias scales with model size -- exactly the axis
T024/T034's model comparison measures, so an un-warmed suite would not just
be noisy, it would be biased in the one dimension the comparison cares
about. Each sample records `"warmed_up": True`, derived the same way
`e2e.py` derives it: if warm-up fails, this function raises before the
timed loop or any `samples.append` is ever reached, so the field can only
be `True` when it is actually true.
"""

from __future__ import annotations

import time

from .. import harness

_PROMPT_TEXT = "Good morning, how are you doing today? I was hoping we could reschedule our meeting."
_TARGET_LANGUAGE = "Korean"


def _build_prompt(tokenizer) -> str:
    prompt = (
        f"Translate to {_TARGET_LANGUAGE} in a natural, conversational tone. "
        "Do not include emojis or special symbols. Reply with ONLY the translation.\n\n"
        f"{_PROMPT_TEXT}"
    )
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _warm_up(stream_generate, model, tokenizer, chat_prompt: str) -> None:
    """One untimed generation so the timed loop measures steady state, not a
    cold first call (Critic T020 round-2 major 2).

    Raises if the warm-up yields **zero** responses (Critic T020 round-3
    minor): an empty `stream_generate` would otherwise let this function
    return silently having warmed up nothing, while `run()`'s
    `"warmed_up": True` is still written -- the same derive-from-raising
    pattern as everywhere else in this module, so that field can only be
    `True` once a real response was actually produced.

    This does *not* subsume the `first_token_ms is not None else total_ms`
    fallback in `run()`'s own timed loop below -- that fallback covers
    whichever *real, timed* iteration independently yields zero responses,
    which remains possible on its own (sampling variance) even once
    warm-up has already produced output with the same prompt and model.
    What this raise does rule out is the deterministic case: warm-up runs
    the exact same prompt/model/tokenizer combination the timed loop will
    reuse, before any timed iteration runs, so a structural
    always-zero-output failure surfaces here first, loudly -- not only
    discovered (and silently masked by that fallback) on iteration 0.
    """
    generated_anything = False
    for _ in stream_generate(model, tokenizer, chat_prompt, max_tokens=32):
        generated_anything = True
    if not generated_anything:
        raise RuntimeError("translation: warm-up generation produced zero responses")


def run(iterations: int, config: dict) -> tuple[list, list[dict]]:
    from mlx_lm import load, stream_generate

    model_id = config["translation_model"]
    model, tokenizer = load(model_id)
    chat_prompt = _build_prompt(tokenizer)

    _warm_up(stream_generate, model, tokenizer, chat_prompt)

    samples: list[dict] = []
    total_durations_ms: list[float] = []
    first_token_durations_ms: list[float] = []

    for i in range(iterations):
        start = time.perf_counter()
        first_token_ms: float | None = None
        text = ""
        for response in stream_generate(model, tokenizer, chat_prompt, max_tokens=512):
            text += response.text
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - start) * 1000
        total_ms = (time.perf_counter() - start) * 1000

        total_durations_ms.append(total_ms)
        first_token_durations_ms.append(first_token_ms if first_token_ms is not None else total_ms)
        samples.append(
            {
                "iteration": i,
                "total_ms": total_ms,
                "first_token_ms": first_token_ms,
                "output": text.strip()[:200],
                "warmed_up": True,
            }
        )

    mean_total = sum(total_durations_ms) / len(total_durations_ms)
    mean_first_token = sum(first_token_durations_ms) / len(first_token_durations_ms)

    stages = [
        harness.stage_timing("translation", mean_total),
        harness.stage_timing("translation_first_token", mean_first_token),
    ]
    return stages, samples
