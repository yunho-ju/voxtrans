"""Shared pytest configuration for the server test suite (T004).

Registers the `bench` and `needs_models` markers, exposes a `fixtures_dir`
fixture, and skips `needs_models` tests when a required model snapshot is
not present locally.

Also puts `server/` (this file's parent) on `sys.path` so tests can
`import src.*` / `import bench.*` regardless of whether pytest is invoked
from the repo root (`pytest server/tests`) or from `server/` (`pytest tests`).
"""

import logging
import sys
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError

logger = logging.getLogger(__name__)

_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

_TESTS_DIR = Path(__file__).resolve().parent

# The five model repos every feature depends on (research.md D2). This is
# what a bare `@pytest.mark.needs_models` (no repo ids) checks.
FEATURE_MODEL_REPOS = (
    "mlx-community/Qwen3-ASR-1.7B-8bit",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "mlx-community/Qwen3.5-4B-MLX-4bit",
    "mlx-community/Qwen3.5-2B-4bit",
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
