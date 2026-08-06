"""ASR model quality comparison (FR-011 / Principle IV, tasks.md T037c-5).

Companion to `bench.eval_models` (T031), for the ASR-model decision instead
of the translation-model one (brief `us1-asr-T037b-T037e.md`). Synthesizes
the honorifics/numbers subset of `eval-set.jsonl` via macOS `say` +
`afconvert` (design decision 6 -- deterministic, offline, the same pipeline
`tests/fixtures/build_fixtures.sh` uses), transcribes each clip through
every candidate ASR model, and writes the side-by-side table to
`specs/001-realtime-latency/evidence/asr-comparison.md`, with
`reviewer_decision` left blank -- the decision belongs to the user (T037e),
never to this tool.

`write_comparison` is `bench.eval_models.write_comparison` itself (imported,
not reimplemented) -- the overwrite guard ("a recorded reviewer_decision is
inviolable") is identical for both comparison files, so there is exactly one
implementation of it.

Model/audio invocation is injected (`transcribe_fn` for `run_comparison`,
`list_voices_fn` for the voice-presence check) so table construction and the
voice guard are testable without ever shelling out to `say`/`afconvert` or
loading a real ASR model. `main()` is the only function that touches either
-- real candidate measurement is T037d's job (the primary agent), not this
round's (T037c-5 is script + tests only).
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .eval_models import EvalPair, _escape_cell, load_eval_pairs, write_comparison
from src import config as _src_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_SET_PATH = _REPO_ROOT / "specs" / "001-realtime-latency" / "eval-set.jsonl"
_EVIDENCE_PATH = _REPO_ROOT / "specs" / "001-realtime-latency" / "evidence" / "asr-comparison.md"

# The honorifics/numbers subset of eval-set.jsonl (brief design decision 6):
# the two categories most sensitive to an ASR mis-recognition (a dropped
# honorific, a wrong digit) rather than a translation-quality concern. Order
# matches eval-set.jsonl's own id order.
_QUALITY_PAIR_IDS: tuple[str, ...] = (
    "ko-en-03", "ko-en-04", "ko-en-06", "ko-en-07",
    "en-ko-02", "en-ko-03", "en-ko-06", "en-ko-07", "en-ko-09",
)

# macOS voices `tests/fixtures/build_fixtures.sh` also requires -- same
# voice-per-language assignment (research.md D3).
_KOREAN_VOICE = "Yuna"
_ENGLISH_VOICE = "Samantha"
# `say` rates tuned so the two 5s fixtures land at natural pace
# (build_fixtures.sh); reused here for the same natural-pace reason, not
# because these particular short phrases need exactly ~5s.
_SAY_RATE = {"korean": 190, "english": 248}


@dataclass(frozen=True)
class AsrCandidate:
    """One ASR model under comparison. `stage_ms`/`resident_mb` are the
    aggregate numbers a separate `bench.run --suite asr`/`--suite memory`
    invocation already measured for this candidate (T037d); `None` renders
    as "n/a" -- this module never measures them itself."""

    label: str
    model_id: str
    stage_ms: float | None = None
    resident_mb: float | None = None


# T037e's own decision reads this default pair; the incumbent id is read
# from `config._ASR_MODEL_DEFAULT` (mirrors eval_models.DEFAULT_CANDIDATES,
# Critic round-2 minor 5 there), not re-typed as a literal, so the two can
# never silently drift apart. `test_default_candidates_are_all_known_model_repos`
# keeps both ids in sync with `fetch_models.MODEL_REPOS`.
DEFAULT_CANDIDATES: tuple[AsrCandidate, ...] = (
    AsrCandidate("incumbent (Qwen3-ASR-1.7B-8bit)", _src_config._ASR_MODEL_DEFAULT),
    AsrCandidate("whisper-large-v3-turbo", "mlx-community/whisper-large-v3-turbo"),
)

TranscribeFn = Callable[[AsrCandidate, EvalPair, "str | Path"], str]


def load_quality_pairs(path: Path = _EVAL_SET_PATH) -> list[EvalPair]:
    """The 9 honorifics/numbers `eval-set.jsonl` pairs named in
    `_QUALITY_PAIR_IDS`, in that fixed order. Raises if any expected id is
    absent from the eval set (a stale/edited eval-set.jsonl silently
    dropping one of these must not silently narrow this comparison)."""
    pairs_by_id = {pair.id: pair for pair in load_eval_pairs(path)}
    missing = [pair_id for pair_id in _QUALITY_PAIR_IDS if pair_id not in pairs_by_id]
    if missing:
        raise ValueError(f"eval-set.jsonl is missing expected quality-comparison id(s): {missing}")
    return [pairs_by_id[pair_id] for pair_id in _QUALITY_PAIR_IDS]


def _voice_for(pair: EvalPair) -> str:
    return _KOREAN_VOICE if pair.source_lang == "korean" else _ENGLISH_VOICE


def _real_list_voices() -> str:
    return subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout


def _require_voice(voice: str, list_voices_fn: Callable[[], str]) -> None:
    """Fail loudly and by name if `voice` is not installed -- no silent
    fallback to a different voice (brief C7, `build_fixtures.sh`'s own
    norm). First whitespace-delimited token per line, mirroring
    `build_fixtures.sh`'s `awk '{print $1}'` -- so a line like
    "YunaPremium ko_KR" does not satisfy a request for "Yuna"."""
    available = {line.split()[0] for line in list_voices_fn().splitlines() if line.strip()}
    if voice not in available:
        raise RuntimeError(
            f"macOS voice '{voice}' is not installed. Install it via System "
            "Settings > Accessibility > Spoken Content > Manage Voices, then re-run."
        )


def synthesize_pair(
    pair: EvalPair,
    out_dir: "str | Path",
    *,
    list_voices_fn: Callable[[], str] = _real_list_voices,
) -> Path:
    """Synthesize `pair.text` into a 16kHz mono float32 WAV under `out_dir`
    via `say` -> `afconvert` (`build_fixtures.sh`'s own pipeline,
    research.md D3): deterministic and offline. `out_dir` is caller-supplied
    (never defaults to `tests/fixtures/`) so this never pollutes the test
    fixture corpus. Raises, naming the required voice, before invoking `say`
    at all if that voice is not installed (brief C7)."""
    voice = _voice_for(pair)
    _require_voice(voice, list_voices_fn)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aiff_path = out_dir / f"{pair.id}.aiff"
    wav_path = out_dir / f"{pair.id}.wav"
    rate = _SAY_RATE[pair.source_lang]
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff_path), pair.text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "F32@16000", "-c", "1", str(aiff_path), str(wav_path)], check=True
    )
    return wav_path


def _rows_from_transcripts(
    pairs: list[EvalPair],
    candidates: tuple[AsrCandidate, ...],
    transcripts_by_candidate: dict[str, dict[str, str]],
) -> list[dict]:
    """Assemble one row per pair from a `{candidate_label: {pair_id: text}}`
    transcript map -- the single place row construction happens (Critic
    round 1 major 2). `run_comparison` below (candidate-inner, one
    `transcribe_fn` call per pair) and `main()` (candidate-outer, so each
    candidate's model loads exactly once) build that map in different loop
    orders for different reasons, but both hand it to this one function
    rather than each assembling rows itself -- so the rows the unit tests
    exercise via `run_comparison` are the exact same rows `main()` produces,
    not a parallel reimplementation that could silently drift from it."""
    return [
        {
            "pair": pair,
            "transcripts": {c.label: transcripts_by_candidate[c.label][pair.id] for c in candidates},
        }
        for pair in pairs
    ]


def run_comparison(
    pairs_with_audio: list[tuple[EvalPair, "str | Path"]],
    candidates: tuple[AsrCandidate, ...],
    transcribe_fn: TranscribeFn,
) -> list[dict]:
    """For every (pair, audio_path), call `transcribe_fn` once per candidate,
    then assemble rows via `_rows_from_transcripts` -- the same row assembly
    `main()` uses. `transcribe_fn` is the sole injection point for ASR
    inference -- this function never loads a model or shells out itself, so
    it is fully testable with a stub."""
    pairs: list[EvalPair] = []
    transcripts_by_candidate: dict[str, dict[str, str]] = {c.label: {} for c in candidates}
    for pair, audio_path in pairs_with_audio:
        pairs.append(pair)
        for candidate in candidates:
            transcripts_by_candidate[candidate.label][pair.id] = transcribe_fn(candidate, pair, audio_path)
    return _rows_from_transcripts(pairs, candidates, transcripts_by_candidate)


def _fmt_ms(value: float | None) -> str:
    return f"{value:.1f}ms" if value is not None else "n/a"


def _fmt_mb(value: float | None) -> str:
    return f"{value:.0f}MB" if value is not None else "n/a"


def render_markdown(rows: list[dict], candidates: tuple[AsrCandidate, ...]) -> str:
    """Render the side-by-side table: reference (source) text against every
    candidate's transcript, each candidate's aggregate ASR-stage latency and
    resident memory summarized in the "Candidates:" line (brief item 5).
    `reviewer_decision`/`decided_at` are emitted blank by construction."""
    lines = [
        "# ASR comparison (US1+, T037c/T037d, FR-011 / Principle IV)",
        "",
        "Candidates: "
        + ", ".join(
            f"{c.label} (`{c.model_id}`, asr={_fmt_ms(c.stage_ms)}, resident={_fmt_mb(c.resident_mb)})"
            for c in candidates
        ),
        "",
        "**Audio is synthetic** (macOS `say` + `afconvert`, "
        "`bench.eval_asr.synthesize_pair`, the same pipeline "
        "`tests/fixtures/build_fixtures.sh` uses): both candidates transcribe "
        "the exact same synthesized clips, so any TTS-artifact bias lands on "
        "both equally -- fair for a *relative* comparison, but not evidence "
        "of either candidate's accuracy on real human speech.",
        "",
        f"| id | note | source text | {' | '.join(c.label for c in candidates)} |",
        "|" + "---|" * (3 + len(candidates)),
    ]
    for row in rows:
        pair: EvalPair = row["pair"]
        cells = " | ".join(_escape_cell(row["transcripts"][c.label]) for c in candidates)
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


_ASR_MODEL_ENV_VAR = "VOXTRANS_ASR_MODEL"


def _restore_asr_modules(env_var: str, previous: str | None) -> None:
    """Restore `env_var` to `previous` (or unset it) and reload
    `src.config`/`src.asr` so their own module state matches (Critic round 1
    minor 3). Restoring only the env var is not enough: `src/asr.py`'s
    `from .config import ASR_MODEL` is a bound-name copy, so it stays
    resolved to whatever the *last* `importlib.reload(src_asr)` saw --
    without this second reload, a later same-process `src.asr.ASREngine()`
    construction (e.g. production code sharing this interpreter) would
    silently keep resolving the last-loaded candidate's model, even though
    the env var itself looks restored to normal."""
    import importlib
    import os

    import src.asr as src_asr
    import src.config as src_config

    if previous is None:
        os.environ.pop(env_var, None)
    else:
        os.environ[env_var] = previous
    importlib.reload(src_config)
    importlib.reload(src_asr)


def _load_engine_for_model(model_id: str):
    """Construct a loaded `ASREngine` resolving `ASR_MODEL` as `model_id`,
    within this one process. `src/asr.py`'s `from .config import ASR_MODEL`
    is a bound-name copy resolved at import time, so switching candidates
    within one process (rather than one `bench.run` invocation per
    candidate) needs a real reload of both modules -- the same
    `importlib.reload` pattern `test_config_model_override.py` uses to
    observe a different `VOXTRANS_ASR_MODEL` value, applied here to also
    reload `src.asr` (which captured the old value already). `finally`
    restores both the env var *and* that same module state via
    `_restore_asr_modules`, not just the env var."""
    import os

    import src.asr as src_asr
    import src.config as src_config
    import importlib

    previous = os.environ.get(_ASR_MODEL_ENV_VAR)
    os.environ[_ASR_MODEL_ENV_VAR] = model_id
    try:
        importlib.reload(src_config)
        importlib.reload(src_asr)
        engine = src_asr.ASREngine()
        engine.load()
        return engine
    finally:
        _restore_asr_modules(_ASR_MODEL_ENV_VAR, previous)


def main(
    candidates: tuple[AsrCandidate, ...] = DEFAULT_CANDIDATES,
    *,
    force: bool = False,
    audio_dir: "str | Path | None" = None,
) -> Path:
    """Synthesize every quality pair's audio, transcribe it through every
    candidate ASR model, and write `evidence/asr-comparison.md` via
    `write_comparison` -- so a recorded reviewer decision is protected here
    too. Not exercised by this round's unit tests (T037c-5 is script + tests
    only; T037d is the primary agent's real measurement run) -- it is the
    one function that actually touches `say`/`afconvert`/`ASREngine`.
    `candidates` should already carry real `stage_ms`/`resident_mb` numbers
    read from that run's `asr-<label>.json`/`memory-<label>.json` evidence;
    this function does not measure those itself."""
    import soundfile as sf

    pairs = load_quality_pairs()
    work_dir = Path(audio_dir) if audio_dir is not None else Path(tempfile.mkdtemp(prefix="asr-eval-audio-"))
    pairs_with_audio = [(pair, synthesize_pair(pair, work_dir)) for pair in pairs]

    transcripts_by_candidate: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        engine = _load_engine_for_model(candidate.model_id)
        transcripts_by_candidate[candidate.label] = {}
        for pair, audio_path in pairs_with_audio:
            audio, _sr = sf.read(str(audio_path), dtype="float32")
            transcripts_by_candidate[candidate.label][pair.id] = engine.transcribe(
                audio, language=pair.source_lang
            )

    rows = _rows_from_transcripts([pair for pair, _audio_path in pairs_with_audio], candidates, transcripts_by_candidate)
    markdown = render_markdown(rows, candidates)
    return write_comparison(markdown, _EVIDENCE_PATH, force=force)


if __name__ == "__main__":
    main()
