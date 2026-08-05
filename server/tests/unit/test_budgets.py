"""Tests for bench/budgets.py (T010 Red; passes once T013 lands).

`bench.budgets` is the single definition of every latency budget number in
this codebase (data-model.md §4) — a number appearing anywhere else is a
duplicate and a Critic finding.

`test_budget_table_is_self_consistent` is expected to keep FAILING even
after T013 lands: the constitution's current per-stage targets sum to more
than `e2e_total` (analyze finding C1: 600+400+700+500 = 2200 > 1500). That
arithmetic contradiction is not this task's to resolve — T024 is a blocking
human decision that reconciles it. This assertion exists specifically so a
future edit cannot quietly encode the contradiction into passing code; it
must not be loosened, skipped, or have its numbers "fixed" here.
"""

import pytest

import bench.budgets as budgets


def test_every_budget_has_a_source_citation():
    for stage, budget in budgets.BUDGETS.items():
        assert budget.stage == stage
        assert budget.source, f"budget {stage!r} has no source citation"


def test_budget_values_match_constitution_performance_standards():
    assert budgets.target_ms("vad_hold") == 600.0
    assert budgets.target_ms("asr") == 400.0
    assert budgets.target_ms("translation") == 700.0
    assert budgets.target_ms("tts_first_chunk") == 500.0
    assert budgets.target_ms("e2e_total") == 1500.0
    assert budgets.target_ms("subtitle_first_token") == 500.0


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
