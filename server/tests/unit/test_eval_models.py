"""Tests for `bench.eval_models` (tasks.md T031).

Model invocation is injected via `translate_fn` so this module's table
construction is fully testable without loading any real model -- real
candidate model calls happen in T032 (primary agent), not here
("실모델 로드 금지... 전부 스텁/단위 수준", brief).
"""

from __future__ import annotations

import pytest

import bench.eval_models as eval_models
import bench.fetch_models as fetch_models
import bench.harness as harness


def _pairs() -> list[eval_models.EvalPair]:
    return [
        eval_models.EvalPair("ko-en-01", "korean", "english", "안녕하세요", "conversational turn"),
        eval_models.EvalPair("en-ko-01", "english", "korean", "Hello there", "conversational turn"),
    ]


def _candidates() -> tuple[eval_models.Candidate, ...]:
    return (
        eval_models.Candidate("A", "org/model-a"),
        eval_models.Candidate("B", "org/model-b"),
    )


def test_load_eval_pairs_reads_the_real_eval_set_file():
    pairs = eval_models.load_eval_pairs()
    assert len(pairs) == 20
    assert all(isinstance(p, eval_models.EvalPair) for p in pairs)
    assert {p.id for p in pairs if p.id.startswith("ko-en-")}.__len__() == 10


def test_run_comparison_calls_translate_fn_for_every_pair_and_candidate():
    calls: list[tuple[str, str]] = []

    def _stub_translate(candidate: eval_models.Candidate, pair: eval_models.EvalPair) -> str:
        calls.append((candidate.label, pair.id))
        return f"[{candidate.label}]->{pair.text}"

    rows = eval_models.run_comparison(_pairs(), _candidates(), _stub_translate)

    assert len(calls) == 4, "2 pairs x 2 candidates must each be translated exactly once"
    assert len(rows) == 2
    assert rows[0]["outputs"]["A"] == "[A]->안녕하세요"
    assert rows[0]["outputs"]["B"] == "[B]->안녕하세요"


def test_render_markdown_includes_every_candidate_column_and_every_pair_row():
    def _stub_translate(candidate: eval_models.Candidate, pair: eval_models.EvalPair) -> str:
        return f"out-{candidate.label}-{pair.id}"

    rows = eval_models.run_comparison(_pairs(), _candidates(), _stub_translate)
    markdown = eval_models.render_markdown(rows, _candidates())

    for candidate in _candidates():
        assert candidate.label in markdown
    for pair in _pairs():
        assert pair.id in markdown
        assert pair.text in markdown
        assert f"out-A-{pair.id}" in markdown
        assert f"out-B-{pair.id}" in markdown


def test_render_markdown_leaves_reviewer_decision_blank():
    """FR-011 / Principle IV / data-model.md §6 invariant: `reviewer_decision`
    is written only after the user states it -- this tool must never fill
    it in on its own."""

    def _stub_translate(candidate: eval_models.Candidate, pair: eval_models.EvalPair) -> str:
        return "irrelevant"

    rows = eval_models.run_comparison(_pairs(), _candidates(), _stub_translate)
    markdown = eval_models.render_markdown(rows, _candidates())

    assert "reviewer_decision:" in markdown
    decision_line = next(line for line in markdown.splitlines() if line.startswith("reviewer_decision:"))
    assert decision_line.strip() == "reviewer_decision:", (
        f"reviewer_decision must be blank, found: {decision_line!r}"
    )
    assert "accept" not in markdown.lower()
    assert "reject" not in markdown.lower()


# --- major 1: a recorded reviewer_decision is inviolable --------------------


def test_write_comparison_writes_when_no_existing_file(tmp_path):
    path = tmp_path / "model-comparison.md"

    result = eval_models.write_comparison("fresh content\n", path)

    assert result == path
    assert path.read_text() == "fresh content\n"


def test_write_comparison_refuses_overwrite_without_force_when_no_decision_recorded(tmp_path):
    path = tmp_path / "model-comparison.md"
    path.write_text("old content\nreviewer_decision:\ndecided_at:\n")

    with pytest.raises(harness.EvidenceWriteRefused):
        eval_models.write_comparison("new content\n", path, force=False)

    assert path.read_text() == "old content\nreviewer_decision:\ndecided_at:\n"


def test_write_comparison_overwrites_when_no_decision_recorded_and_force_true(tmp_path):
    path = tmp_path / "model-comparison.md"
    path.write_text("old content\nreviewer_decision:\ndecided_at:\n")

    eval_models.write_comparison("new content\nreviewer_decision:\ndecided_at:\n", path, force=True)

    assert path.read_text() == "new content\nreviewer_decision:\ndecided_at:\n"


@pytest.mark.parametrize("decision", ["accept", "reject"])
def test_write_comparison_refuses_even_with_force_once_a_decision_is_recorded(tmp_path, decision):
    """FR-011 / Principle IV / data-model.md §6 invariant: `T034`'s recorded
    decision must survive a re-run -- `--force` overwrites a stale,
    undecided comparison, but it must never be able to erase a decision
    that already exists (Critic round-2 major 1)."""
    path = tmp_path / "model-comparison.md"
    path.write_text(f"old content\nreviewer_decision: {decision}\ndecided_at: 2026-08-06\n")

    with pytest.raises(harness.EvidenceWriteRefused):
        eval_models.write_comparison("new content\nreviewer_decision:\ndecided_at:\n", path, force=True)

    assert path.read_text() == f"old content\nreviewer_decision: {decision}\ndecided_at: 2026-08-06\n"


# --- major 2: `_build_prompt` must translate in the pair's own direction ----


class _FakeTokenizer:
    """Mirrors `test_translation_suite.py`'s `_FakeTokenizer`: returns the
    raw prompt content so the test can inspect what `_build_prompt` actually
    asked for, instead of an opaque templated string."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[0]["content"]


def test_build_prompt_targets_english_for_a_ko_to_en_pair():
    pair = eval_models.EvalPair("ko-en-01", "korean", "english", "안녕하세요", "conversational turn")

    prompt = eval_models._build_prompt(_FakeTokenizer(), pair)

    assert "Translate to English" in prompt
    assert "안녕하세요" in prompt


def test_build_prompt_targets_korean_for_an_en_to_ko_pair():
    pair = eval_models.EvalPair("en-ko-01", "english", "korean", "Hello there", "conversational turn")

    prompt = eval_models._build_prompt(_FakeTokenizer(), pair)

    assert "Translate to Korean" in prompt
    assert "Hello there" in prompt


def test_build_prompt_includes_the_self_correction_sentence_verbatim():
    pair = eval_models.EvalPair(
        "en-ko-05", "english", "korean", "Tuesday no wait Wednesday for coffee", "mid-sentence self-correction"
    )

    prompt = eval_models._build_prompt(_FakeTokenizer(), pair)

    assert "Tuesday no wait Wednesday for coffee" in prompt


def test_build_prompt_includes_the_technical_term_sentence_verbatim():
    pair = eval_models.EvalPair(
        "ko-en-08", "korean", "english", "웹소켓 연결이 끊기면 재접속을 시도해요", "technical term"
    )

    prompt = eval_models._build_prompt(_FakeTokenizer(), pair)

    assert "웹소켓 연결이 끊기면 재접속을 시도해요" in prompt


# --- minor 2: markdown cells must not be corruptible by model output -------


def test_render_markdown_escapes_pipes_and_newlines_in_a_candidate_output():
    def _stub_translate(candidate: eval_models.Candidate, pair: eval_models.EvalPair) -> str:
        return "line one\nline two | not a real column"

    rows = eval_models.run_comparison(_pairs(), _candidates(), _stub_translate)
    markdown = eval_models.render_markdown(rows, _candidates())

    table_lines = [line for line in markdown.splitlines() if line.startswith("| ko-en-01")]
    assert len(table_lines) == 1, "the row must stay on a single markdown table line"
    row_line = table_lines[0]
    assert "line one<br>line two \\| not a real column" in row_line
    assert "\n" not in row_line.replace("<br>", "")


# --- minor 5: DEFAULT_CANDIDATES stays in sync with config/fetch_models ----


def test_default_candidates_incumbent_matches_config_default():
    import src.config as config

    incumbent = next(c for c in eval_models.DEFAULT_CANDIDATES if "incumbent" in c.label)
    assert incumbent.model_id == config._TRANSLATION_MODEL_DEFAULT


def test_default_candidates_are_all_known_model_repos():
    candidate_ids = {c.model_id for c in eval_models.DEFAULT_CANDIDATES}
    unknown = candidate_ids - set(fetch_models.MODEL_REPOS)
    assert not unknown, f"candidate ids not in fetch_models.MODEL_REPOS: {unknown}"
