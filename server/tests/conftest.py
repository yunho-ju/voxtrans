"""Shared pytest configuration for the server test suite (T004).

Registers the `bench` and `needs_models` markers, exposes a `fixtures_dir`
fixture, and skips `needs_models` tests when a required model snapshot is
not present locally. Also exposes VAD fixtures (`vad_model_cached`,
`vad_factory`, `vad`, `replay_wav_through_vad`, T052/T053/T057) used by
`tests/unit/test_vad_replay.py` and `tests/unit/test_vad_short_utterance.py`.

Also puts `server/` (this file's parent) on `sys.path` so tests can
`import src.*` / `import bench.*` regardless of whether pytest is invoked
from the repo root (`pytest server/tests`) or from `server/` (`pytest tests`).

A single top-level `conftest.py` is deliberate: neither `tests/` nor
`tests/unit/` has an `__init__.py`, so a second `conftest.py` nested under
`tests/unit/` would be imported by pytest under the same bare module name
`conftest` as this file, colliding with `test_conftest_contract.py` and
`test_fetch_models.py`'s existing `from conftest import ...` (confirmed:
adding one during T052/T053 broke both with `ImportError: cannot import
name 'FEATURE_MODEL_REPOS' from 'conftest'`).

`from src.vad import ...` for the VAD fixtures below is deliberately done
*inside* those fixture functions, not at module level: `test_conftest_contract.py`
copies this file's source verbatim (`_CONFTEST_PATH.read_text()`) into an
isolated `pytester` sandbox that has no `server/src` on disk, to test
`pytest_collection_modifyitems` in true isolation. A module-level `src.vad`
import broke that (`ModuleNotFoundError: No module named 'src'`) even though
none of that test's dummy files ever request the VAD fixtures -- pytest
still imports the whole conftest.py module up front. `from __future__ import
annotations` (below) lets the VAD fixtures still carry real type
annotations without evaluating `VoiceActivityDetector` at that same
module-import time.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pytest
import soundfile as sf
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError

if TYPE_CHECKING:
    from src.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)

_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

_TESTS_DIR = Path(__file__).resolve().parent

# The model repos every feature depends on (research.md D2), plus the T037c
# ASR candidate `--asr-model` can select in addition to the incumbent. This
# is what a bare `@pytest.mark.needs_models` (no repo ids) checks. Kept in
# sync with `bench.fetch_models.MODEL_REPOS` (see that module's own comment).
FEATURE_MODEL_REPOS = (
    "mlx-community/Qwen3-ASR-1.7B-8bit",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "mlx-community/Qwen3.5-4B-MLX-4bit",
    "mlx-community/Qwen3.5-2B-4bit",
    "mlx-community/whisper-large-v3-turbo",
)


def has_local_snapshot(repo_id: str) -> bool:
    """Return True iff repo_id resolves to a local snapshot folder in the HF cache.

    This is presence, not completeness: `snapshot_download(local_files_only=True)`
    resolves the cached ref and returns its folder without checking that every
    expected file is inside it, so a partially-downloaded repo also returns
    True here. A real completeness check belongs to the model-fetch bootstrap
    (see tasks.md T007), not this test helper.

    Never touches the network: `local_files_only=True` makes this a pure
    local cache lookup.
    """
    try:
        snapshot_download(repo_id, local_files_only=True)
        return True
    except (LocalEntryNotFoundError, HFValidationError, OSError) as exc:
        logger.debug("no local snapshot for %s: %s", repo_id, exc)
        return False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "bench: benchmark that asserts a latency/memory budget"
    )
    config.addinivalue_line(
        "markers",
        "needs_models(*repo_ids): skip unless every named model snapshot is "
        "present locally; no args means all five feature models",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    for item in items:
        marker = item.get_closest_marker("needs_models")
        if marker is None:
            continue
        repo_ids = marker.args or FEATURE_MODEL_REPOS
        missing = [repo_id for repo_id in repo_ids if not has_local_snapshot(repo_id)]
        if missing:
            item.add_marker(
                pytest.mark.skip(reason=f"missing local snapshot(s): {', '.join(missing)}")
            )


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to server/tests/fixtures, independent of invocation cwd."""
    return _TESTS_DIR / "fixtures"


def _local_silero_cache_dir(torch_module) -> Path | None:
    """Resolve the local Silero VAD checkout torch.hub already has cached
    (`~/.cache/torch/hub/snakers4_silero-vad_<ref>`, honoring `TORCH_HOME`/
    `XDG_CACHE_HOME` via `torch.hub.get_dir()`), or `None` if neither ref
    is present. Mirrors the two refs `torch.hub._parse_repo_info`'s own
    `except URLError` offline fallback already tries, just checked
    proactively instead of only after a failed network call."""
    hub_dir = Path(torch_module.hub.get_dir())
    for ref in ("main", "master"):
        candidate = hub_dir / f"snakers4_silero-vad_{ref}"
        if candidate.is_dir():
            return candidate
    return None


@pytest.fixture(scope="session")
def vad_model_cached():
    """Load Silero VAD once per test session (T052/T053), preferring the
    local torch.hub cache over the network (Critic round 1, T057 minor 3).

    Unpatched, `torch.hub.load("snakers4/silero-vad", ...)` always makes one
    network call first (`_parse_repo_info`'s GitHub ref lookup) even when
    the repo is already cached, and only falls back to the cache on a
    `URLError` (no route at all) -- a GitHub 5xx raises `HTTPError` instead,
    which is *not* caught, so a flaky GitHub still fails this fixture even
    though a perfectly good local checkout is sitting right there. This
    fixture does not touch `src/vad.py` (out of scope for this task): it
    monkeypatches `torch.hub.load` itself, for the rest of the test session,
    to redirect the one call shape `src.vad.create_vad_model`/
    `load_vad_model` actually make (`repo_or_dir="snakers4/silero-vad"`,
    all-keyword) straight to `source="local"` against the resolved cache
    directory when one is present -- skipping the network entirely, not
    merely tolerating its absence. Any other call shape (a different repo,
    or this same repo with no local cache) passes through unchanged, so the
    original network-with-cache-fallback behavior is preserved when there
    truly is nothing cached yet.

    Session-scoped and restored at teardown because `VoiceActivityDetector.
    __init__` calls `create_vad_model()` (a fresh `torch.hub.load`) on every
    construction, not just once -- every `vad`/`vad_factory`-built detector
    for the rest of the session needs the patched loader, not just this
    fixture's own one-time `load_vad_model()` warm-up call.
    """
    import torch

    original_hub_load = torch.hub.load

    def _local_cache_preferring_hub_load(*args, **kwargs):
        if not args and kwargs.get("repo_or_dir") == "snakers4/silero-vad":
            local_dir = _local_silero_cache_dir(torch)
            if local_dir is not None:
                patched_kwargs = dict(kwargs)
                patched_kwargs["repo_or_dir"] = str(local_dir)
                patched_kwargs["source"] = "local"
                return original_hub_load(**patched_kwargs)
        return original_hub_load(*args, **kwargs)

    torch.hub.load = _local_cache_preferring_hub_load
    try:
        from src.vad import load_vad_model

        load_vad_model()
        yield
    finally:
        torch.hub.load = original_hub_load


@pytest.fixture
def vad_factory(vad_model_cached: None) -> Callable[..., VoiceActivityDetector]:
    """Factory fixture: `vad_factory(**kwargs) -> VoiceActivityDetector`,
    forwarding `**kwargs` straight to the constructor so a test can build
    either of the two production shapes `session.py` actually uses --
    `session.py:55`'s mic one (`silence_windows` only) or `session.py:44-47`'s
    listen one (`silence_windows` plus `intermediate_speech_windows`/
    `intermediate_silence_windows`) -- rather than only ever observing the
    bare-default construction (Critic round 1, T057 major finding)."""
    from src.vad import VoiceActivityDetector

    def _make(**kwargs) -> VoiceActivityDetector:
        return VoiceActivityDetector(**kwargs)

    return _make


@pytest.fixture
def vad(vad_factory: Callable[..., VoiceActivityDetector]) -> VoiceActivityDetector:
    """A fresh `VoiceActivityDetector` using the production defaults from
    `src.config` with no override -- the same shape `session.py:55` builds
    `Session.mic_vad` with (`silence_windows` only, no intermediate split)."""
    return vad_factory()


def _replay_wav_through_vad(detector: VoiceActivityDetector, path: Path) -> list[np.ndarray]:
    data, samplerate = sf.read(path, dtype="float32", always_2d=False)
    assert samplerate == 16000, f"expected 16000 Hz, got {samplerate}"
    segments = list(detector.process_chunk(data))
    trailing = detector.flush()
    if trailing is not None:
        segments.append(trailing)
    return segments


@pytest.fixture
def replay_wav_through_vad() -> Callable[[VoiceActivityDetector, Path], list[np.ndarray]]:
    """Feed an entire fixture WAV through a `VoiceActivityDetector` in one
    call, then `flush()` to force out any speech still in progress at
    end-of-file, and return every complete segment produced, in order.

    Feeding the whole file as a single `process_chunk` call rather than in
    network-sized (100ms) pieces is equivalent for segmentation purposes:
    `VoiceActivityDetector` windows any input into VAD_WINDOW_SAMPLES frames
    internally and only carries a same-call remainder across window
    boundaries (`vad.py:90-107`), so chunk boundaries chosen by a caller
    never influence the state machine.
    """
    return _replay_wav_through_vad
