"""Locks down the contract server/tests/conftest.py must satisfy (T004).

Covers: marker registration, the fixtures_dir fixture, sys.path resolution,
the local-snapshot presence check, and the needs_models skip mechanism.

Every test here is hermetic: cache state under test is built in `tmp_path`
and pointed at explicitly (either by patching
`huggingface_hub.constants.HF_HUB_CACHE` for in-process checks, or by
setting `HF_HOME` for the out-of-process `pytester` probes). No test reads
the real, ambient HF cache, so outcomes never depend on which of the five
feature models happen to be downloaded on the machine running the suite.
"""

from collections.abc import Iterable
from pathlib import Path

import pytest
from huggingface_hub import constants as hf_constants

from conftest import FEATURE_MODEL_REPOS, has_local_snapshot

pytest_plugins = ["pytester"]

_CONFTEST_PATH = Path(__file__).resolve().parent.parent / "conftest.py"


def _seed_fake_snapshot(hub_dir: Path, repo_id: str) -> None:
    """Create a minimal fake HF cache entry for repo_id under hub_dir.

    Mirrors the real cache layout (`models--<org>--<name>/refs/main` +
    `snapshots/<hash>/`) closely enough for
    `snapshot_download(local_files_only=True)` to resolve it — presence,
    not completeness, exactly as `has_local_snapshot`'s docstring says.
    """
    org, name = repo_id.split("/", 1)
    model_dir = hub_dir / f"models--{org}--{name}"
    (model_dir / "snapshots" / "deadbeef").mkdir(parents=True, exist_ok=True)
    (model_dir / "refs").mkdir(parents=True, exist_ok=True)
    (model_dir / "refs" / "main").write_text("deadbeef")


def _seed_fake_hf_home(hf_home: Path, repo_ids: Iterable[str] = ()) -> None:
    """Build a fake $HF_HOME with a hub/ cache seeded for each repo_id."""
    hub_dir = hf_home / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    for repo_id in repo_ids:
        _seed_fake_snapshot(hub_dir, repo_id)


def test_bench_and_needs_models_markers_are_registered(pytestconfig: pytest.Config) -> None:
    # Registered markers show up in the ini "markers" list; an unregistered
    # marker never would, even though pytest lets you use it without
    # --strict-markers. Entries may carry a "(args)" signature (pytest's own
    # builtins do, e.g. "parametrize(argnames, argvalues)"), so strip that
    # before comparing bare names.
    marker_names = {
        line.split(":", 1)[0].split("(", 1)[0].strip() for line in pytestconfig.getini("markers")
    }
    assert "bench" in marker_names
    assert "needs_models" in marker_names


def test_fixtures_dir_resolves_to_tests_fixtures(fixtures_dir: Path) -> None:
    assert isinstance(fixtures_dir, Path)
    assert fixtures_dir == Path(__file__).resolve().parent.parent / "fixtures"


def test_sys_path_insertion_makes_src_importable() -> None:
    # server/tests has no __init__.py, so `import src.config` only resolves
    # if conftest.py put server/ on sys.path — this is the load-bearing
    # assertion for the "import src.* / bench.* from either invocation
    # directory" design constraint.
    import src.config as config

    assert config.SAMPLE_RATE == 16000


def test_feature_model_repos_matches_hardcoded_research_md_d2_mirror() -> None:
    # This is a HAND-MAINTAINED MIRROR of specs/001-realtime-latency/research.md
    # D2, not a parse of that file — parsing a markdown table would be more
    # fragile than the drift risk it guards against. If D2's model list ever
    # changes, this literal must be updated in the same change, or this test
    # keeps passing while silently no longer describing D2.
    #
    # What it does catch: a typo introduced directly into FEATURE_MODEL_REPOS
    # without a matching D2 change, checked against a literal list rather
    # than the local machine's cache — a typo here would otherwise only
    # surface as every bare needs_models test silently skipping forever
    # (round 1 Finding 3 / round 3 Finding 2).
    assert FEATURE_MODEL_REPOS == (
        "mlx-community/Qwen3-ASR-1.7B-8bit",
        "mlx-community/Qwen3-8B-4bit",
        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "mlx-community/Qwen3.5-4B-MLX-4bit",
        "mlx-community/Qwen3.5-2B-4bit",
    )


def test_has_local_snapshot_true_for_a_present_synthetic_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub_dir = tmp_path / "hub"
    _seed_fake_snapshot(hub_dir, "acme/widget")
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hub_dir))

    assert has_local_snapshot("acme/widget") is True


def test_has_local_snapshot_false_for_an_absent_synthetic_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub_dir = tmp_path / "hub"
    hub_dir.mkdir()
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hub_dir))

    assert has_local_snapshot("acme/does-not-exist") is False


def test_needs_models_named_arg_skips_when_named_snapshot_absent(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hf_home = tmp_path / "hf-home"
    _seed_fake_hf_home(hf_home)  # nothing seeded: the named repo is absent
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pytester.makeconftest(_CONFTEST_PATH.read_text())
    pytester.makepyfile(
        test_probe="""
        import pytest

        @pytest.mark.needs_models("acme/widget-9000")
        def test_needs_absent_model():
            assert False, "should have been skipped, not executed"
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(skipped=1)


def test_needs_models_named_arg_runs_when_named_snapshot_present(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hf_home = tmp_path / "hf-home"
    _seed_fake_hf_home(hf_home, ["acme/widget-9000"])
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pytester.makeconftest(_CONFTEST_PATH.read_text())
    pytester.makepyfile(
        test_probe="""
        import pytest

        @pytest.mark.needs_models("acme/widget-9000")
        def test_needs_present_model():
            assert True
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)


def test_needs_models_named_arg_skips_when_only_some_named_snapshots_present(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # ANY-missing-skips, not ALL-missing-skips: two ids named, only the
    # *first* one seeded — a naive `repo_ids[:1]` check would only look at
    # that (present) first entry and wrongly conclude nothing is missing,
    # missing the second (absent) one entirely.
    hf_home = tmp_path / "hf-home"
    _seed_fake_hf_home(hf_home, ["acme/widget-9000"])
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pytester.makeconftest(_CONFTEST_PATH.read_text())
    pytester.makepyfile(
        test_probe="""
        import pytest

        @pytest.mark.needs_models("acme/widget-9000", "acme/widget-9001")
        def test_needs_partially_present_models():
            assert False, "should have been skipped, not executed"
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(skipped=1)


def test_needs_models_bare_marker_skips_when_a_default_snapshot_is_absent(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Bare @pytest.mark.needs_models (no repo ids) falls back to
    # FEATURE_MODEL_REPOS. An empty cache is missing all five, so the
    # fallback branch must skip rather than run.
    hf_home = tmp_path / "hf-home"
    _seed_fake_hf_home(hf_home)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pytester.makeconftest(_CONFTEST_PATH.read_text())
    pytester.makepyfile(
        test_probe="""
        import pytest

        @pytest.mark.needs_models
        def test_needs_all_default_models():
            assert False, "should have been skipped, not executed"
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(skipped=1)


def test_needs_models_bare_marker_runs_when_all_default_snapshots_present(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Same fallback branch, but the synthetic cache is seeded from
    # FEATURE_MODEL_REPOS itself (not the real machine's cache), so this is
    # true regardless of what happens to be downloaded locally.
    hf_home = tmp_path / "hf-home"
    _seed_fake_hf_home(hf_home, FEATURE_MODEL_REPOS)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pytester.makeconftest(_CONFTEST_PATH.read_text())
    pytester.makepyfile(
        test_probe="""
        import pytest

        @pytest.mark.needs_models
        def test_needs_all_default_models():
            assert True
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)


def test_needs_models_bare_marker_skips_when_only_some_default_snapshots_present(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # ANY-missing-skips, not ALL-missing-skips: 4 of 5 FEATURE_MODEL_REPOS
    # seeded, deliberately omitting FEATURE_MODEL_REPOS[1] (not the first
    # entry — a naive `repo_ids[:1]` check would only look at the first,
    # present, entry and wrongly conclude nothing is missing).
    missing_index = 1
    present_repo_ids = [
        repo_id for i, repo_id in enumerate(FEATURE_MODEL_REPOS) if i != missing_index
    ]
    assert len(present_repo_ids) == 4  # sanity: genuinely 4 of 5, not all 5
    hf_home = tmp_path / "hf-home"
    _seed_fake_hf_home(hf_home, present_repo_ids)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pytester.makeconftest(_CONFTEST_PATH.read_text())
    pytester.makepyfile(
        test_probe="""
        import pytest

        @pytest.mark.needs_models
        def test_needs_all_default_models():
            assert False, "should have been skipped, not executed"
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(skipped=1)
