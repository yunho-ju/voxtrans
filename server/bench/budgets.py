"""Single source of truth for every latency budget number in this codebase
(data-model.md §4, tasks.md T013).

A budget number appearing anywhere else — in a suite, in `harness.py`, in a
docstring doing its own arithmetic — is a duplicate and a Critic finding;
everything else must resolve a stage's target through this module.

**PROVISIONAL**: the numbers below are copied verbatim from
`.specify/memory/constitution.md` § "Performance Standards and Hardware
Budget" as it stood before the T023-T025 budget reconciliation. They are
known to be arithmetically inconsistent (analyze finding C1:
`vad_hold + asr + translation + tts_first_chunk` = 600+400+700+500 = 2200 ms,
which exceeds `e2e_total`'s 1500 ms) — `test_budget_table_is_self_consistent`
in `tests/unit/test_budgets.py` asserts the sum stays within `e2e_total` and
is *expected to fail* against these numbers. That failure is deliberate: it
is the mechanism that stops the contradiction from being silently encoded as
passing code. Resolving it is T024's blocking human decision, not this
module's.
"""

from __future__ import annotations

from dataclasses import dataclass

_CONSTITUTION_SOURCE = (
    ".specify/memory/constitution.md § Performance Standards and Hardware Budget"
)


@dataclass(frozen=True)
class Budget:
    """One stage's latency target, with a citation for where it comes from
    (data-model.md §4)."""

    stage: str
    target_ms: float
    source: str


BUDGETS: dict[str, Budget] = {
    budget.stage: budget
    for budget in (
        Budget("vad_hold", 600.0, _CONSTITUTION_SOURCE),
        Budget("asr", 400.0, _CONSTITUTION_SOURCE),
        Budget("translation", 700.0, _CONSTITUTION_SOURCE),
        Budget("tts_first_chunk", 500.0, _CONSTITUTION_SOURCE),
        Budget("e2e_total", 1500.0, _CONSTITUTION_SOURCE),
        Budget("subtitle_first_token", 500.0, _CONSTITUTION_SOURCE),
    )
}

# Stages that compose `e2e_total` by addition
# (contracts/benchmark-cli.md "Composition rule for e2e").
E2E_STAGES: tuple[str, ...] = ("vad_hold", "asr", "translation", "tts_first_chunk")


def get(stage: str) -> Budget:
    """Look up a stage's `Budget`. Raises `KeyError` for an unknown stage —
    there is no silent default, since a caller reaching for a nonexistent
    budget is a bug, not a legitimate "no budget" case (that case is
    `StageTiming.budget_ms is None`, produced by stages that are never in
    `BUDGETS` at all, e.g. `translation_first_token`)."""
    return BUDGETS[stage]


def target_ms(stage: str) -> float:
    """Shorthand for `get(stage).target_ms`."""
    return BUDGETS[stage].target_ms
