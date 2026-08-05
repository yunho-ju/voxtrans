"""Single source of truth for every latency budget number in this codebase
(data-model.md §4, tasks.md T013).

A budget number appearing anywhere else — in a suite, in `harness.py`, in a
docstring doing its own arithmetic — is a duplicate and a Critic finding;
everything else must resolve a stage's target through this module.

The numbers below are the T024-approved reconciliation (constitution v1.1.0,
user decision recorded in
`specs/001-realtime-latency/evidence/budget-reconciliation.md`, 2026-08-06):
Option A, stage targets summing exactly to the 2000 ms total, and the
subtitle budget anchored to speech-end *detection* (physical speech end +
VAD hold), i.e. `vad_hold + 500 ms` from the bench suite's physical-speech-end
anchor. `test_budget_table_is_self_consistent` now passes by construction —
it remains the guard that any future edit keeps the table arithmetic honest.
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
        Budget("vad_hold", 550.0, _CONSTITUTION_SOURCE),
        Budget("asr", 450.0, _CONSTITUTION_SOURCE),
        Budget("translation", 550.0, _CONSTITUTION_SOURCE),
        Budget("tts_first_chunk", 450.0, _CONSTITUTION_SOURCE),
        Budget("e2e_total", 2000.0, _CONSTITUTION_SOURCE),
        # Anchored to speech-end *detection*: the bench suite measures from
        # physical speech end, so its budget is the approved 500 ms plus the
        # 550 ms vad_hold target (constitution v1.1.0, C3).
        Budget("subtitle_first_token", 1050.0, _CONSTITUTION_SOURCE),
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
