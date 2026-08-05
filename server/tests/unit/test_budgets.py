"""Tests for bench/budgets.py (T010 Red; passes once T013 lands).

`bench.budgets` is the single definition of every latency budget number in
this codebase (data-model.md §4) — a number appearing anywhere else is a
duplicate and a Critic finding.

History: `test_budget_table_is_self_consistent` deliberately FAILED from
T013 until T024 — the pre-reconciliation constitution's targets summed to
2200 ms against a 1500 ms total (analyze C1), and the failing assertion was
the mechanism that kept that contradiction from being silently encoded. The
T024 decision (constitution v1.1.0, 2026-08-06, recorded in
evidence/budget-reconciliation.md) replaced the table with one that sums
exactly to its 2000 ms total, so the assertion now passes — and remains the
guard that any future budget edit keeps the arithmetic honest. It must not
be loosened or skipped.
"""

import pytest

import bench.budgets as budgets


def test_every_budget_has_a_source_citation():
    for stage, budget in budgets.BUDGETS.items():
        assert budget.stage == stage
        assert budget.source, f"budget {stage!r} has no source citation"


def test_budget_values_match_constitution_performance_standards():
    # Constitution v1.1.0 (T024-approved Option A). subtitle_first_token is
    # detection-anchored: 500 ms after detection = vad_hold target + 500 from
    # the bench suite's physical-speech-end anchor.
    assert budgets.target_ms("vad_hold") == 550.0
    assert budgets.target_ms("asr") == 450.0
    assert budgets.target_ms("translation") == 550.0
    assert budgets.target_ms("tts_first_chunk") == 450.0
    assert budgets.target_ms("e2e_total") == 2000.0
    assert budgets.target_ms("subtitle_first_token") == 550.0 + 500.0


def test_get_unknown_stage_raises_key_error():
    with pytest.raises(KeyError):
        budgets.get("not_a_real_stage")


def test_budget_table_is_self_consistent():
    """Per-stage targets that compose `e2e_total` must sum to no more than
    `e2e_total` itself — expected to FAIL against the constitution's current
    numbers (analyze finding C1). See module docstring: do not silence this."""
    stage_sum = sum(budgets.target_ms(stage) for stage in budgets.E2E_STAGES)
    assert stage_sum <= budgets.target_ms("e2e_total"), (
        f"per-stage targets sum to {stage_sum} ms, which exceeds e2e_total "
        f"({budgets.target_ms('e2e_total')} ms) — unresolved until T024"
    )
