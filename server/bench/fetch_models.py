"""Idempotent model-weight bootstrap and local completeness verification
(research.md D2, tasks.md T007).

Two kinds of local asset are covered:

* The five Hugging Face model repos every suite in `contracts/benchmark-cli.md`
  depends on (`MODEL_REPOS`). `ensure_model()` verifies a repo's local
  snapshot is *complete* (not merely present — see `check_snapshot_completeness`)
  and fetches it via `snapshot_download` only when it is missing or partial.
* The Silero VAD, which `src/vad.py` loads via `torch.hub.load` from GitHub
  on first use. `ensure_silero_vad()` makes sure that clone/cache exists
  ahead of time.

  Caveat (Critic finding 3, round 1; corrected finding 2, round 2):
  `src/vad.py` has **two** ref-less `"snakers4/silero-vad"` call sites
  (no `:branch` pin on either), and both matter — `:29` in `load_vad_model`,
  which runs once at startup, **and** `:42` in `create_vad_model`, which
  runs on *every* `VoiceActivityDetector()` construction. Because the repo
  string is ref-less, `torch.hub.load`'s internal ref-resolution step makes
  a live HTTPS call to GitHub before it ever consults the cache this module
  populates, at both sites — measured 55-490 ms per `VoiceActivityDetector()`
  construction on a normally-connected machine, even with a fully warm
  cache (`:42`, the one that actually matters for a benchmark's timed
  region). What priming the cache actually guarantees is narrower: when that
  resolution call fails outright (no network at all), `torch.hub`'s cache
  fallback succeeds instead of raising, so construction still works
  *offline*. It does not, by itself, remove the network round-trip on a
  reachable machine, so it does not fully satisfy
  `contracts/benchmark-cli.md` precondition 3a on its own — see
  `evidence/T007.md` "Outstanding / risks" for what closing that gap
  requires in `src/vad.py` at **both** `:29` and `:42` (out of this module's
  scope).

Both `ensure_*` functions accept `verify_only=True`, which performs the same
local-only check but never touches the network — this is what a benchmark
harness's precondition check (exit code 2 on failure, per
`contracts/benchmark-cli.md`) should call.

Run directly (`python -m bench.fetch_models`) to fetch anything missing and
verify everything: exits 0 if every asset ends up complete, 2 otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError

logger = logging.getLogger(__name__)

# The five model repositories every feature depends on (research.md D2).
# Hand-maintained mirror of research.md D2 and of
# tests/conftest.py:FEATURE_MODEL_REPOS — keep all three in sync.
MODEL_REPOS: tuple[str, ...] = (
    "mlx-community/Qwen3-ASR-1.7B-8bit",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "mlx-community/Qwen3.5-4B-MLX-4bit",
    "mlx-community/Qwen3.5-2B-4bit",
)

# Silero VAD is not a Hugging Face repo; torch.hub clones it from GitHub.
# The directory names torch.hub actually consults under its hub cache root:
# its offline cache fallback (torch/hub.py) only ever looks for "..._main"
# or "..._master", and the online path only ever creates "..._master" for
# this repo (its default branch). A directory for any other ref (e.g. a
# pinned release tag) is not something torch.hub will find, so it must not
# be treated as satisfying the cache precondition (Critic finding 5, round 1).
_SILERO_HUB_DIR_CANDIDATES = ("snakers4_silero-vad_main", "snakers4_silero-vad_master")
SILERO_VAD_REPO = "snakers4/silero-vad"
SILERO_VAD_MODEL = "silero_vad"


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of a local-only completeness check for one asset."""

    name: str
    ok: bool
    reason: str = ""


# --- Hugging Face model repos ---------------------------------------------


class _CorruptIndexError(Exception):
    """model.safetensors.index.json exists but could not be parsed."""


def _required_shard_files(snapshot_dir: Path) -> list[str]:
    """Weight shard filenames a sharded-safetensors repo requires, read from
    its own `model.safetensors.index.json` `weight_map`.

    The index file each repo ships with *is* its manifest, so this needs no
    external file list and no network call. Returns `[]` for a repo with no
    such index (nothing extra to check beyond ordinary presence).

    Raises `_CorruptIndexError` if the index file is present but unreadable
    or not valid JSON — that is itself evidence of a partial download (a
    small file that ordinarily transfers atomically), and must not be
    silently treated as "no shards required."
    """
    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return []
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise _CorruptIndexError(f"unreadable model.safetensors.index.json: {exc}") from exc
    weight_map = index.get("weight_map", {})
    return sorted(set(weight_map.values()))


def check_snapshot_completeness(snapshot_dir: Path) -> VerifyResult:
    """Local-only completeness check for one already-resolved HF snapshot
    directory. Detects three distinct partial-download shapes:

    1. A transfer interrupted mid-file leaves a stray `<hash>.incomplete`
       blob in the repo's `blobs/` dir (it never got renamed to its final
       hash and symlinked in).
    2. A transfer interrupted *between* files leaves no partial blob at all —
       a file the snapshot's own index says it needs is simply absent, or its
       symlink is broken (points at a blob that was never written).
    3. The index file itself is truncated/corrupt (see `_CorruptIndexError`).

    Scope: this verifies the weight shards named in `model.safetensors.index.json`
    (in practice, on all five `MODEL_REPOS`, that is a single `model.safetensors`
    file each — none of the five is actually sharded) plus mid-transfer blob
    markers. Other files a snapshot carries (tokenizer/config files, the TTS
    repo's `speech_tokenizer/` subdirectory) are presence-checked only via the
    snapshot resolving at all, the same guarantee `has_local_snapshot` in
    `conftest.py` documents — a corrupt-but-present `tokenizer.json` would not
    be caught here.
    """
    name = snapshot_dir.name
    blobs_dir = snapshot_dir.parent.parent / "blobs"
    if blobs_dir.is_dir():
        incomplete = sorted(p.name for p in blobs_dir.glob("*.incomplete"))
        if incomplete:
            return VerifyResult(name, False, f"incomplete blob(s) present: {incomplete}")

    try:
        shard_names = _required_shard_files(snapshot_dir)
    except _CorruptIndexError as exc:
        return VerifyResult(name, False, str(exc))

    for shard_name in shard_names:
        shard_path = snapshot_dir / shard_name
        if not shard_path.exists():  # follows symlinks; False for missing or broken ones
            return VerifyResult(name, False, f"missing required shard file: {shard_name}")

    return VerifyResult(name, True)


def verify_model(repo_id: str, cache_dir: Path | str | None = None) -> VerifyResult:
    """Local-only: does repo_id resolve to a complete snapshot? Never
    downloads (`local_files_only=True`)."""
    try:
        snapshot_dir = Path(snapshot_download(repo_id, cache_dir=cache_dir, local_files_only=True))
    except (LocalEntryNotFoundError, HFValidationError, OSError) as exc:
        logger.debug("no local snapshot for %s: %s", repo_id, exc)
        return VerifyResult(repo_id, False, "no local snapshot")
    result = check_snapshot_completeness(snapshot_dir)
    return VerifyResult(repo_id, result.ok, result.reason)


def ensure_model(repo_id: str, cache_dir: Path | str | None = None, verify_only: bool = False) -> VerifyResult:
    """Idempotent: verify repo_id has a complete local snapshot, fetching it
    only if it is missing or partial and `verify_only` is False. A snapshot
    that is already complete never touches the network."""
    result = verify_model(repo_id, cache_dir)
    if result.ok or verify_only:
        return result
    try:
        snapshot_download(repo_id, cache_dir=cache_dir)
    except Exception as exc:  # network error, auth error, disk full, etc.
        return VerifyResult(repo_id, False, f"download failed: {exc}")
    return verify_model(repo_id, cache_dir)


# --- Silero VAD -------------------------------------------------------------


def _silero_repo_dir(hub_dir: Path) -> Path | None:
    for candidate in _SILERO_HUB_DIR_CANDIDATES:
        candidate_dir = hub_dir / candidate
        if candidate_dir.is_dir():
            return candidate_dir
    return None


def verify_silero_vad(torch_hub_dir: Path | str | None = None) -> VerifyResult:
    """Local-only: is the Silero VAD repo already cloned into the torch hub
    cache? Never touches the network."""
    hub_dir = Path(torch_hub_dir) if torch_hub_dir is not None else Path(torch.hub.get_dir())
    repo_dir = _silero_repo_dir(hub_dir) if hub_dir.is_dir() else None
    if repo_dir is None:
        return VerifyResult(SILERO_VAD_REPO, False, "not cached locally")
    if not any(repo_dir.iterdir()):
        return VerifyResult(SILERO_VAD_REPO, False, "cached repo dir is empty")
    return VerifyResult(SILERO_VAD_REPO, True)


def ensure_silero_vad(torch_hub_dir: Path | str | None = None, verify_only: bool = False) -> VerifyResult:
    """Idempotent: verify the Silero VAD repo is cached locally, fetching it
    only if absent/empty and `verify_only` is False.

    Fetching only ensures the cache exists (the one public API for that,
    `torch.hub.load`, also builds a throwaway model instance as an
    unavoidable side effect of running the repo's `hubconf.py` — that
    instantiation is not a benchmark warm-up and is not treated as one; it is
    discarded immediately). No forward pass is run here. Actual model warm-up
    that matters for a benchmark budget happens later, inside the timed
    region it belongs to.

    What this cache buys a later `VoiceActivityDetector()` construction: a
    working *offline* fallback, not the removal of every network call on a
    connected machine — `src/vad.py`'s ref-less repo string means torch.hub
    still makes a live ref-resolution HTTPS call first on every construction
    regardless of cache state. See the module docstring's "Caveat".

    `torch_hub_dir`, when given, is honored for the fetch too (not just the
    verify above): `torch.hub.load` has no destination-directory parameter of
    its own, so this scopes `torch.hub.set_dir()` around the call and always
    restores the previous value afterward, in production this parameter is
    `None` and no global state is ever touched (Critic finding 1, round 2 —
    `cache_dir` on `ensure_model` is symmetric between its verify and fetch
    calls; this makes `torch_hub_dir` symmetric too).
    """
    result = verify_silero_vad(torch_hub_dir)
    if result.ok or verify_only:
        return result

    previous_hub_dir = torch.hub.get_dir() if torch_hub_dir is not None else None
    if torch_hub_dir is not None:
        torch.hub.set_dir(str(torch_hub_dir))
    try:
        torch.hub.load(repo_or_dir=SILERO_VAD_REPO, model=SILERO_VAD_MODEL, trust_repo=True)
    except Exception as exc:
        return VerifyResult(SILERO_VAD_REPO, False, f"fetch failed: {exc}")
    finally:
        if torch_hub_dir is not None:
            torch.hub.set_dir(previous_hub_dir)
    return verify_silero_vad(torch_hub_dir)


# --- CLI ---------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    *,
    cache_dir: Path | str | None = None,
    torch_hub_dir: Path | str | None = None,
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check completeness only; never touch the network",
    )
    args = parser.parse_args(argv)

    results = [ensure_model(repo_id, cache_dir=cache_dir, verify_only=args.verify_only) for repo_id in MODEL_REPOS]
    results.append(ensure_silero_vad(torch_hub_dir=torch_hub_dir, verify_only=args.verify_only))

    ok = True
    for result in results:
        status = "OK" if result.ok else "MISSING/PARTIAL"
        suffix = f"  ({result.reason})" if result.reason else ""
        print(f"{status:16s} {result.name}{suffix}")
        ok = ok and result.ok

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
