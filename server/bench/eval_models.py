"""Model quality comparison (FR-011 / Principle IV, data-model.md §6,
tasks.md T031).

Runs every `eval-set.jsonl` pair through each candidate translation model
and writes the side-by-side table to
`specs/001-realtime-latency/evidence/model-comparison.md`, with
`reviewer_decision` left blank -- the decision belongs to the user (T034),
never to this tool (data-model.md §6 invariant: `TRANSLATION_MODEL` MUST
NOT change until this file carries an `accept` decision).

Model invocation is injected (`translate_fn` / `TranslateFn`) precisely so
`run_comparison` and `render_markdown` are testable without loading any
real model. `main()` is the only function that touches `mlx_lm`, and it
does so with a local import (matching `bench/suites/translation.py`'s own
pattern) so importing this module never requires MLX to be installed.
Real candidate measurement is T032's job (the primary agent), not this
round's -- this round is tool + unit tests only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import harness
from src import config as _src_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_SET_PATH = _REPO_ROOT / "specs" / "001-realtime-latency" / "eval-set.jsonl"
_EVIDENCE_PATH = _REPO_ROOT / "specs" / "001-realtime-latency" / "evidence" / "model-comparison.md"

_TARGET_LANGUAGE_LABEL = {"korean": "Korean", "english": "English"}


@dataclass(frozen=True)
class EvalPair:
    """One line of `eval-set.jsonl` (data-model.md §5)."""

    id: str
    source_lang: str
    target_lang: str
    text: str
    note: str


@dataclass(frozen=True)
class Candidate:
    """One model under comparison (data-model.md §6 "candidates")."""

    label: str
    model_id: str


# The three models named in `research.md` D2 / tasks.md T032 -- the
# incumbent plus the two Qwen3.5 candidates. `main()`'s default; tests
# always pass their own small candidate tuples instead. The incumbent id
# is read from `config._TRANSLATION_MODEL_DEFAULT` (Critic round-2 minor
# 5), not re-typed as a literal, so the two can never silently drift apart;
# `test_default_candidates_are_all_known_model_repos` keeps every id here
# in sync with `fetch_models.MODEL_REPOS`.
DEFAULT_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("qwen3-8b (incumbent)", _src_config._TRANSLATION_MODEL_DEFAULT),
    Candidate("qwen35-4b", "mlx-community/Qwen3.5-4B-MLX-4bit"),
    Candidate("qwen35-2b", "mlx-community/Qwen3.5-2B-4bit"),
)

TranslateFn = Callable[[Candidate, EvalPair], str]


def load_eval_pairs(path: Path = _EVAL_SET_PATH) -> list[EvalPair]:
    """Parse `eval-set.jsonl` into `EvalPair`s, in file order."""
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            pairs.append(EvalPair(**data))
    return pairs


def run_comparison(
    eval_pairs: list[EvalPair],
    candidates: tuple[Candidate, ...],
    translate_fn: TranslateFn,
) -> list[dict]:
    """For every pair, call `translate_fn` once per candidate and collect
    the outputs side by side. `translate_fn` is the sole injection point
    for model inference (module docstring) -- this function never imports
    `mlx_lm` itself, so it is fully testable with a stub."""
    rows = []
    for pair in eval_pairs:
        outputs = {candidate.label: translate_fn(candidate, pair) for candidate in candidates}
        rows.append({"pair": pair, "outputs": outputs})
    return rows


def _escape_cell(value: str) -> str:
    """Escape a value for placement inside a single markdown table cell
    (Critic round-2 minor 2): an embedded `|` would otherwise be misread as
    a column separator, and an embedded newline would break the row onto
    multiple physical lines and desync every column after it. The smallest
    candidate model is the likeliest to ignore the "reply with ONLY the
    translation" instruction and emit exactly one of these."""
    return value.replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")


def render_markdown(rows: list[dict], candidates: tuple[Candidate, ...]) -> str:
    """Render the FR-011 side-by-side table. `reviewer_decision` and
    `decided_at` are emitted blank by construction -- this function never
    receives a decision to write, so it cannot fill one in even by
    accident (data-model.md §6 invariant)."""
    lines = [
        "# Model comparison (FR-011 / Principle IV)",
        "",
        "Candidates: " + ", ".join(f"{c.label} (`{c.model_id}`)" for c in candidates),
        "",
        f"| id | note | source text | {' | '.join(c.label for c in candidates)} |",
        "|" + "---|" * (3 + len(candidates)),
    ]
    for row in rows:
        pair: EvalPair = row["pair"]
        cells = " | ".join(_escape_cell(row["outputs"][c.label]) for c in candidates)
        lines.append(
            f"| {_escape_cell(pair.id)} | {_escape_cell(pair.note)} | {_escape_cell(pair.text)} | {cells} |"
        )
    lines += [
        "",
        "## Reviewer decision",
        "",
        "reviewer_decision:",
        "decided_at:",
    ]
    return "\n".join(lines) + "\n"


def _existing_reviewer_decision(path: Path) -> str | None:
    """The non-empty `reviewer_decision` value already recorded at `path`,
    or `None` if the file does not exist or carries no decision yet."""
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("reviewer_decision:"):
            value = line.split(":", 1)[1].strip()
            return value or None
    return None


def write_comparison(markdown: str, path: Path = _EVIDENCE_PATH, *, force: bool = False) -> Path:
    """Write `markdown` to `path`, refusing to ever erase a recorded
    `reviewer_decision` -- with or without `--force` (Critic round-2 major
    1; data-model.md §6 invariant: `TRANSLATION_MODEL` MUST NOT change
    until this file carries an `accept` decision, which is meaningless if a
    plain re-run can silently wipe that decision back out). Reuses
    `harness.EvidenceWriteRefused`, the same overwrite-guard vocabulary
    `harness.write_record` uses for `BenchmarkRecord`s.

    A file that exists but carries no decision yet is an ordinary stale
    comparison -- `force=True` overwrites it, exactly like any other
    evidence file; `force=False` refuses it the same way."""
    existing_decision = _existing_reviewer_decision(path)
    if existing_decision:
        raise harness.EvidenceWriteRefused(
            f"{path} already records reviewer_decision={existing_decision!r}; a recorded "
            "decision is inviolable and must never be overwritten, even with force=True (FR-011)"
        )
    if path.exists() and not force:
        raise harness.EvidenceWriteRefused(
            f"{path} already exists with no decision recorded; pass force=True (CLI: --force) to overwrite it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def _build_prompt(tokenizer, pair: EvalPair) -> str:
    target = _TARGET_LANGUAGE_LABEL[pair.target_lang]
    prompt = (
        f"Translate to {target} in a natural, conversational tone. "
        "Do not include emojis or special symbols. Reply with ONLY the translation.\n\n"
        f"{pair.text}"
    )
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def main(candidates: tuple[Candidate, ...] = DEFAULT_CANDIDATES, *, force: bool = False) -> Path:
    """Load every candidate model, translate every eval pair, and write
    `evidence/model-comparison.md` via `write_comparison` -- so a recorded
    reviewer decision is protected here too, not just when called directly
    from a test. Not exercised by this round's unit tests -- it is the one
    function that actually touches `mlx_lm`."""
    from mlx_lm import generate, load

    eval_pairs = load_eval_pairs()
    model_cache: dict[str, tuple] = {}

    def _translate(candidate: Candidate, pair: EvalPair) -> str:
        if candidate.model_id not in model_cache:
            model_cache[candidate.model_id] = load(candidate.model_id)
        model, tokenizer = model_cache[candidate.model_id]
        prompt = _build_prompt(tokenizer, pair)
        return generate(model, tokenizer, prompt, max_tokens=256).strip()

    rows = run_comparison(eval_pairs, candidates, _translate)
    markdown = render_markdown(rows, candidates)
    return write_comparison(markdown, _EVIDENCE_PATH, force=force)


if __name__ == "__main__":
    main()
