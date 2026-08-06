"""Tests for `bench.eval_asr` (tasks.md T037c-5,
`specs/001-realtime-latency/briefs/us1-asr-T037b-T037e.md` C6/C7).

Model/audio invocation is injected (`transcribe_fn` for `run_comparison`,
`list_voices_fn` for the voice-presence check) so this module's table
construction and voice guard are fully testable without ever shelling out to
`say`/`afconvert` or loading any real ASR model -- the actual candidate
measurement run is T037d's job (the primary agent), not this round's.

The write-guard tests below exercise `eval_asr.write_comparison` directly:
it *is* `bench.eval_models.write_comparison` (T037c-5 brief: "write_comparison
재사용... import 또는 일반화" -- reused via import, not reimplemented), so
these tests lock that `eval_asr` keeps using the shared guard rather than
drifting to its own copy, mirroring `test_eval_models.py:90-133`'s four
cases in the ASR module's own namespace.
"""

from __future__ import annotations

import pytest

import bench.eval_asr as eval_asr
import bench.harness as harness


def _pairs() -> list[eval_asr.EvalPair]:
    return [
        eval_asr.EvalPair("ko-en-03", "korean", "english", "회의는 다음 달 3일이에요", "numbers/dates"),
        eval_asr.EvalPair("en-ko-03", "english", "korean", "Excuse me, sir", "honorifics, polite register"),
    ]


def _candidates() -> tuple[eval_asr.AsrCandidate, ...]:
    return (
        eval_asr.AsrCandidate("incumbent", "org/incumbent-model"),
        eval_asr.AsrCandidate("whisper-candidate", "org/whisper-model"),
    )


# --- load_quality_pairs: the hand-picked honorifics/numbers subset ---------


def test_load_quality_pairs_returns_exactly_the_nine_brief_designated_ids():
    pairs = eval_asr.load_quality_pairs()

    assert [p.id for p in pairs] == [
        "ko-en-03", "ko-en-04", "ko-en-06", "ko-en-07",
        "en-ko-02", "en-ko-03", "en-ko-06", "en-ko-07", "en-ko-09",
    ]
    assert all(p.note in ("numbers/dates", "honorifics", "honorifics, service register", "honorifics, polite register") for p in pairs)


# --- run_comparison: transcribe_fn is the sole injection point -------------


def test_run_comparison_calls_transcribe_fn_for_every_pair_and_candidate():
    calls: list[tuple[str, str]] = []

    def _stub_transcribe(candidate: eval_asr.AsrCandidate, pair: eval_asr.EvalPair, audio_path) -> str:
        calls.append((candidate.label, pair.id))
        return f"[{candidate.label}]->{pair.text}"

    pairs_with_audio = [(pair, f"/fake/{pair.id}.wav") for pair in _pairs()]
    rows = eval_asr.run_comparison(pairs_with_audio, _candidates(), _stub_transcribe)

    assert len(calls) == 4, "2 pairs x 2 candidates must each be transcribed exactly once"
    assert len(rows) == 2
    assert rows[0]["transcripts"]["incumbent"] == "[incumbent]->회의는 다음 달 3일이에요"
    assert rows[0]["transcripts"]["whisper-candidate"] == "[whisper-candidate]->회의는 다음 달 3일이에요"


# --- render_markdown: reference/incumbent/candidate columns + blank decision


def test_render_markdown_includes_reference_text_and_every_candidate_transcript():
    def _stub_transcribe(candidate, pair, audio_path):
        return f"out-{candidate.label}-{pair.id}"

    pairs_with_audio = [(pair, f"/fake/{pair.id}.wav") for pair in _pairs()]
    rows = eval_asr.run_comparison(pairs_with_audio, _candidates(), _stub_transcribe)
    markdown = eval_asr.render_markdown(rows, _candidates())

    for pair in _pairs():
        assert pair.id in markdown
        assert pair.text in markdown, "the reference (source) text must appear in the rendered table"
        assert f"out-incumbent-{pair.id}" in markdown
        assert f"out-whisper-candidate-{pair.id}" in markdown


def test_render_markdown_notes_the_synthetic_audio_limitation():
    rows = eval_asr.run_comparison(
        [(pair, f"/fake/{pair.id}.wav") for pair in _pairs()], _candidates(), lambda c, p, a: "x"
    )
    markdown = eval_asr.render_markdown(rows, _candidates())

    assert "synthetic" in markdown.lower()


def test_render_markdown_leaves_reviewer_decision_blank():
    rows = eval_asr.run_comparison(
        [(pair, f"/fake/{pair.id}.wav") for pair in _pairs()], _candidates(), lambda c, p, a: "irrelevant"
    )
    markdown = eval_asr.render_markdown(rows, _candidates())

    assert "reviewer_decision:" in markdown
    decision_line = next(line for line in markdown.splitlines() if line.startswith("reviewer_decision:"))
    assert decision_line.strip() == "reviewer_decision:"
    assert "decided_at:" in markdown
    assert "accept" not in markdown.lower()
    assert "reject" not in markdown.lower()


# --- C6: write-guard, mirroring test_eval_models.py:90-133 in the ASR ------
# --- module's own namespace (eval_asr.write_comparison *is* the shared -----
# --- eval_models.write_comparison, reused not reimplemented) ---------------


def test_write_comparison_writes_when_no_existing_file(tmp_path):
    path = tmp_path / "asr-comparison.md"

    result = eval_asr.write_comparison("fresh content\n", path)

    assert result == path
    assert path.read_text() == "fresh content\n"


def test_write_comparison_refuses_overwrite_without_force_when_no_decision_recorded(tmp_path):
    path = tmp_path / "asr-comparison.md"
    path.write_text("old content\nreviewer_decision:\ndecided_at:\n")

    with pytest.raises(harness.EvidenceWriteRefused):
        eval_asr.write_comparison("new content\n", path, force=False)

    assert path.read_text() == "old content\nreviewer_decision:\ndecided_at:\n"


def test_write_comparison_overwrites_when_no_decision_recorded_and_force_true(tmp_path):
    path = tmp_path / "asr-comparison.md"
    path.write_text("old content\nreviewer_decision:\ndecided_at:\n")

    eval_asr.write_comparison("new content\nreviewer_decision:\ndecided_at:\n", path, force=True)

    assert path.read_text() == "new content\nreviewer_decision:\ndecided_at:\n"


@pytest.mark.parametrize("decision", ["accept", "reject"])
def test_write_comparison_refuses_even_with_force_once_a_decision_is_recorded(tmp_path, decision):
    path = tmp_path / "asr-comparison.md"
    path.write_text(f"old content\nreviewer_decision: {decision}\ndecided_at: 2026-08-06\n")

    with pytest.raises(harness.EvidenceWriteRefused):
        eval_asr.write_comparison("new content\nreviewer_decision:\ndecided_at:\n", path, force=True)

    assert path.read_text() == f"old content\nreviewer_decision: {decision}\ndecided_at: 2026-08-06\n"


# --- C7: a missing macOS voice fails loudly and by name, no silent fallback


def test_synthesize_pair_raises_naming_the_voice_when_korean_voice_is_missing():
    pair = eval_asr.EvalPair("ko-en-03", "korean", "english", "text", "numbers/dates")

    with pytest.raises(RuntimeError, match="Yuna"):
        eval_asr.synthesize_pair(pair, out_dir="/unused", list_voices_fn=lambda: "Samantha    en_US\n")


def test_synthesize_pair_raises_naming_the_voice_when_english_voice_is_missing():
    pair = eval_asr.EvalPair("en-ko-03", "english", "korean", "text", "honorifics")

    with pytest.raises(RuntimeError, match="Samantha"):
        eval_asr.synthesize_pair(pair, out_dir="/unused", list_voices_fn=lambda: "Yuna    ko_KR\n")


def test_require_voice_does_not_raise_when_the_voice_is_present():
    eval_asr._require_voice("Yuna", lambda: "Yuna    ko_KR\nSamantha    en_US\n")


def test_require_voice_matches_the_voice_name_token_exactly_not_a_substring():
    """A voice list line like "YunaPremium ko_KR" must not satisfy a
    request for "Yuna" -- exact first-token match only, the same
    `awk '{print $1}'` parsing `build_fixtures.sh` uses."""
    with pytest.raises(RuntimeError, match="Yuna"):
        eval_asr._require_voice("Yuna", lambda: "YunaPremium    ko_KR\n")


# --- DEFAULT_CANDIDATES stays in sync with fetch_models.MODEL_REPOS --------


def test_default_candidates_incumbent_matches_config_default():
    import src.config as config

    incumbent = next(c for c in eval_asr.DEFAULT_CANDIDATES if "incumbent" in c.label)
    assert incumbent.model_id == config._ASR_MODEL_DEFAULT


def test_default_candidates_are_all_known_model_repos():
    import bench.fetch_models as fetch_models

    candidate_ids = {c.model_id for c in eval_asr.DEFAULT_CANDIDATES}
    unknown = candidate_ids - set(fetch_models.MODEL_REPOS)
    assert not unknown, f"candidate ids not in fetch_models.MODEL_REPOS: {unknown}"


# --- Critic round 1 minor 3: restoring the env var alone is not enough -----
# --- to undo a same-process candidate switch -- src.config/src.asr's own --
# --- module state (a bound-name copy of ASR_MODEL) must be reloaded back --
# --- too, or a later same-process ASREngine() silently keeps the last -----
# --- candidate's model even though the env var looks restored -------------


def test_restore_asr_modules_reloads_src_asr_and_src_config_back_to_the_prior_value(monkeypatch):
    import importlib

    import src.asr as src_asr
    import src.config as src_config

    original = src_config.ASR_MODEL
    try:
        # Simulate `_load_engine_for_model` having switched to a candidate
        # mid-process, the way the real function under test does.
        monkeypatch.setenv(eval_asr._ASR_MODEL_ENV_VAR, "acme/some-candidate")
        importlib.reload(src_config)
        importlib.reload(src_asr)
        assert src_asr.ASR_MODEL == "acme/some-candidate"  # sanity: the bug this guards against

        eval_asr._restore_asr_modules(eval_asr._ASR_MODEL_ENV_VAR, None)

        assert src_config.ASR_MODEL == original
        assert src_asr.ASR_MODEL == original, (
            "src.asr's own bound-name copy of ASR_MODEL must be reloaded back too, "
            "not just the env var"
        )
    finally:
        monkeypatch.delenv(eval_asr._ASR_MODEL_ENV_VAR, raising=False)
        importlib.reload(src_config)
        importlib.reload(src_asr)
