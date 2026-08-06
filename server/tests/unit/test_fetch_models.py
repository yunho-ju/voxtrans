"""Tests for server/bench/fetch_models.py (T007).

Covers: the five repo ids match research.md D2, local-only completeness
detection for complete/partial/absent synthetic snapshots, verify-only mode
never touching the network, idempotent fetch, and the Silero VAD cache
presence check.

Hermetic: every cache state under test (HF hub layout and torch.hub layout)
is built from scratch in `tmp_path` and passed to the code under test via
explicit `cache_dir` / `torch_hub_dir` parameters. No test reads the real,
ambient `~/.cache/huggingface/hub` or `~/.cache/torch/hub`, so results do not
depend on how many of the five feature models or the Silero VAD happen to be
downloaded on the machine running the suite.
"""

import json
from pathlib import Path

import pytest

from conftest import FEATURE_MODEL_REPOS

import bench.fetch_models as fetch_models


@pytest.fixture(autouse=True)
def _block_real_network_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safety net applied to every test in this module: block the
    network-touching branches of `snapshot_download` (`local_files_only`
    falsy) and `torch.hub.load` by default.

    Several tests below call `fetch_models.main()` with real `MODEL_REPOS`
    ids (there is nothing else to inject — `main()` reads the module-level
    constant directly). A mutation that makes `--verify-only` stop being
    honored (Critic finding 1, round 1) would otherwise make those tests
    attempt genuine multi-GB downloads instead of failing fast — this
    happened once already while mutation-testing this fix. Individual tests
    that want to observe or simulate a fetch install their own
    `monkeypatch.setattr` *after* this fixture runs, which overrides it for
    that test only.
    """

    def blocked_snapshot_download(repo_id, *args, **kwargs):
        if kwargs.get("local_files_only"):
            return _real_snapshot_download(repo_id, *args, **kwargs)
        raise RuntimeError(f"blocked: real network fetch attempted for {repo_id!r} in a test")

    def blocked_torch_load(*args, **kwargs):
        raise RuntimeError("blocked: real torch.hub.load attempted in a test")

    monkeypatch.setattr(fetch_models, "snapshot_download", blocked_snapshot_download)
    monkeypatch.setattr(fetch_models.torch.hub, "load", blocked_torch_load)


_real_snapshot_download = fetch_models.snapshot_download


# --- synthetic HF hub cache builders -----------------------------------

def _seed_snapshot(
    hub_dir: Path,
    repo_id: str,
    *,
    shard_names: list[str] | None = None,
    omit_shards: list[str] | None = None,
    incomplete_blob: bool = False,
    empty_shard_symlink_target: bool = False,
) -> Path:
    """Build a synthetic HF hub cache entry for repo_id under hub_dir,
    mirroring the real on-disk layout (models--org--name/{blobs,refs,snapshots}).

    shard_names: files listed in model.safetensors.index.json's weight_map.
    omit_shards: shard file names to leave out of the snapshot dir entirely
        (simulates a transfer that stopped between files).
    incomplete_blob: if True, leave a stray "<hash>.incomplete" file in blobs/
        (simulates a transfer interrupted mid-file).
    empty_shard_symlink_target: if True, point one shard's symlink at a blob
        hash that was never written (a broken symlink).

    Returns the snapshot directory path.
    """
    shard_names = shard_names if shard_names is not None else ["model.safetensors"]
    omit_shards = omit_shards or []

    org, name = repo_id.split("/", 1)
    repo_dir = hub_dir / f"models--{org}--{name}"
    blobs_dir = repo_dir / "blobs"
    snapshot_dir = repo_dir / "snapshots" / "deadbeef"
    refs_dir = repo_dir / "refs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text("deadbeef")

    # A small config file, always present and complete.
    config_blob = blobs_dir / "configblob0000000000000000000000000000"
    config_blob.write_text("{}")
    (snapshot_dir / "config.json").symlink_to(config_blob)

    # The index.json manifest that names the required weight shards.
    weight_map = {f"tensor_{i}": shard for i, shard in enumerate(shard_names)}
    index_blob = blobs_dir / "indexblob00000000000000000000000000000"
    index_blob.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (snapshot_dir / "model.safetensors.index.json").symlink_to(index_blob)

    for i, shard in enumerate(shard_names):
        if shard in omit_shards:
            continue
        blob_hash = f"shardblob{i:032d}"
        blob_path = blobs_dir / blob_hash
        if not (empty_shard_symlink_target and i == 0):
            blob_path.write_bytes(b"0" * 1024)
        (snapshot_dir / shard).symlink_to(blobs_dir / blob_hash)

    if incomplete_blob:
        (blobs_dir / "strayblob0000000000000000000000.incomplete").write_bytes(b"partial")

    return snapshot_dir


# --- MODEL_REPOS identity ------------------------------------------------

def test_model_repos_matches_research_md_d2() -> None:
    # Hand-maintained mirror of research.md D2's table, in the same order.
    # A typo here would otherwise silently diverge from what T008 actually
    # downloads.
    assert fetch_models.MODEL_REPOS == (
        "mlx-community/Qwen3-ASR-1.7B-8bit",
        "mlx-community/Qwen3-8B-4bit",
        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "mlx-community/Qwen3.5-4B-MLX-4bit",
        "mlx-community/Qwen3.5-2B-4bit",
        "mlx-community/whisper-large-v3-turbo",
    )


def test_model_repos_matches_conftest_feature_model_repos() -> None:
    # Guards the two hand-maintained mirrors (this module's and conftest's)
    # against drifting apart from each other.
    assert fetch_models.MODEL_REPOS == FEATURE_MODEL_REPOS


# --- verify_model: completeness detection --------------------------------

def test_verify_model_true_for_complete_synthetic_snapshot(tmp_path: Path) -> None:
    hub_dir = tmp_path / "hub"
    _seed_snapshot(hub_dir, "acme/widget", shard_names=["model-00001.safetensors", "model-00002.safetensors"])

    result = fetch_models.verify_model("acme/widget", cache_dir=hub_dir)

    assert result.ok is True


def test_verify_model_false_when_snapshot_absent(tmp_path: Path) -> None:
    hub_dir = tmp_path / "hub"
    hub_dir.mkdir()

    result = fetch_models.verify_model("acme/does-not-exist", cache_dir=hub_dir)

    assert result.ok is False
    assert "no local snapshot" in result.reason


def test_verify_model_false_when_a_shard_file_is_missing(tmp_path: Path) -> None:
    # Transfer stopped between files: the shard's symlink was never created,
    # no partial blob is left behind either.
    hub_dir = tmp_path / "hub"
    _seed_snapshot(
        hub_dir,
        "acme/widget",
        shard_names=["model-00001.safetensors", "model-00002.safetensors"],
        omit_shards=["model-00002.safetensors"],
    )

    result = fetch_models.verify_model("acme/widget", cache_dir=hub_dir)

    assert result.ok is False
    assert "model-00002.safetensors" in result.reason


def test_verify_model_false_when_a_shard_symlink_is_broken(tmp_path: Path) -> None:
    # The symlink exists but its blob target was never written (corruption
    # after the fact, or a process killed between blob-write and rename).
    hub_dir = tmp_path / "hub"
    _seed_snapshot(
        hub_dir,
        "acme/widget",
        shard_names=["model.safetensors"],
        empty_shard_symlink_target=True,
    )

    result = fetch_models.verify_model("acme/widget", cache_dir=hub_dir)

    assert result.ok is False


def test_verify_model_false_when_incomplete_blob_marker_present(tmp_path: Path) -> None:
    # A transfer interrupted mid-file leaves "<hash>.incomplete" in blobs/.
    hub_dir = tmp_path / "hub"
    _seed_snapshot(hub_dir, "acme/widget", incomplete_blob=True)

    result = fetch_models.verify_model("acme/widget", cache_dir=hub_dir)

    assert result.ok is False
    assert "incomplete" in result.reason


def test_verify_model_false_when_index_json_is_corrupt(tmp_path: Path) -> None:
    # A truncated model.safetensors.index.json is itself a symptom of a
    # partial download: it is a small file that ordinarily transfers
    # atomically, so corrupt/unreadable content here means the write was
    # interrupted. The file's symlink and blob both still *exist* (this is
    # not the broken-symlink or missing-file case above), so this exercises
    # a third, distinct failure shape.
    hub_dir = tmp_path / "hub"
    snapshot_dir = _seed_snapshot(hub_dir, "acme/widget")
    index_blob = (snapshot_dir / "model.safetensors.index.json").resolve()
    index_blob.write_bytes(b"not valid json {")

    result = fetch_models.verify_model("acme/widget", cache_dir=hub_dir)

    assert result.ok is False


def test_exactly_five_of_five_present_and_complete_reports_all_ok(tmp_path: Path) -> None:
    # Standing constraint 1: identical code path regardless of how many of
    # the real five snapshots happen to be present on the host machine — here
    # exercised with all five present, synthetically.
    hub_dir = tmp_path / "hub"
    for repo_id in fetch_models.MODEL_REPOS:
        _seed_snapshot(hub_dir, repo_id)

    results = [fetch_models.verify_model(repo_id, cache_dir=hub_dir) for repo_id in fetch_models.MODEL_REPOS]

    assert all(r.ok for r in results)


def test_one_of_five_missing_is_reported_as_not_ok(tmp_path: Path) -> None:
    hub_dir = tmp_path / "hub"
    for repo_id in fetch_models.MODEL_REPOS[1:]:
        _seed_snapshot(hub_dir, repo_id)
    # MODEL_REPOS[0] deliberately not seeded.

    results = [fetch_models.verify_model(repo_id, cache_dir=hub_dir) for repo_id in fetch_models.MODEL_REPOS]

    assert results[0].ok is False
    assert all(r.ok for r in results[1:])


def test_zero_of_five_present_reports_none_ok(tmp_path: Path) -> None:
    hub_dir = tmp_path / "hub"
    hub_dir.mkdir()

    results = [fetch_models.verify_model(repo_id, cache_dir=hub_dir) for repo_id in fetch_models.MODEL_REPOS]

    assert not any(r.ok for r in results)


# --- ensure_model: verify-only vs fetch, idempotency ----------------------

def test_ensure_model_verify_only_reports_missing_without_touching_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub_dir = tmp_path / "hub"
    hub_dir.mkdir()
    network_calls = []

    real_snapshot_download = fetch_models.snapshot_download

    def spy(repo_id, *args, **kwargs):
        if not kwargs.get("local_files_only"):
            network_calls.append(repo_id)
            raise AssertionError("network fetch must not happen in verify-only mode")
        return real_snapshot_download(repo_id, *args, **kwargs)

    monkeypatch.setattr(fetch_models, "snapshot_download", spy)

    result = fetch_models.ensure_model("acme/widget", cache_dir=hub_dir, verify_only=True)

    assert result.ok is False
    assert network_calls == []


def test_ensure_model_noop_when_already_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub_dir = tmp_path / "hub"
    _seed_snapshot(hub_dir, "acme/widget")
    real_snapshot_download = fetch_models.snapshot_download

    def spy(repo_id, *args, **kwargs):
        if not kwargs.get("local_files_only"):
            raise AssertionError("must not re-fetch an already-complete snapshot")
        return real_snapshot_download(repo_id, *args, **kwargs)

    monkeypatch.setattr(fetch_models, "snapshot_download", spy)

    result = fetch_models.ensure_model("acme/widget", cache_dir=hub_dir, verify_only=False)

    assert result.ok is True


def test_ensure_model_fetches_when_missing_and_not_verify_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub_dir = tmp_path / "hub"
    hub_dir.mkdir()
    fetch_calls = []
    real_snapshot_download = fetch_models.snapshot_download

    def fake_snapshot_download(repo_id, *args, **kwargs):
        if kwargs.get("local_files_only"):
            # Delegate to the real (network-free) local resolver: it fails
            # before the fake fetch below has seeded anything, and succeeds
            # after, exactly like the real cache would.
            return real_snapshot_download(repo_id, *args, **kwargs)
        fetch_calls.append(repo_id)
        return str(_seed_snapshot(hub_dir, repo_id))

    monkeypatch.setattr(fetch_models, "snapshot_download", fake_snapshot_download)

    result = fetch_models.ensure_model("acme/widget", cache_dir=hub_dir, verify_only=False)

    assert fetch_calls == ["acme/widget"]
    assert result.ok is True


# --- Silero VAD cache presence check ---------------------------------------

def test_verify_silero_vad_false_when_hub_dir_absent(tmp_path: Path) -> None:
    hub_dir = tmp_path / "torchhub"  # never created

    result = fetch_models.verify_silero_vad(torch_hub_dir=hub_dir)

    assert result.ok is False


def test_verify_silero_vad_false_when_repo_dir_missing(tmp_path: Path) -> None:
    hub_dir = tmp_path / "torchhub"
    hub_dir.mkdir()

    result = fetch_models.verify_silero_vad(torch_hub_dir=hub_dir)

    assert result.ok is False


def test_verify_silero_vad_false_when_repo_dir_empty(tmp_path: Path) -> None:
    # An interrupted git clone can leave an empty directory behind.
    hub_dir = tmp_path / "torchhub"
    (hub_dir / "snakers4_silero-vad_master").mkdir(parents=True)

    result = fetch_models.verify_silero_vad(torch_hub_dir=hub_dir)

    assert result.ok is False


def test_verify_silero_vad_true_when_repo_dir_present_and_populated(tmp_path: Path) -> None:
    hub_dir = tmp_path / "torchhub"
    repo_dir = hub_dir / "snakers4_silero-vad_master"
    repo_dir.mkdir(parents=True)
    (repo_dir / "hubconf.py").write_text("# stub")

    result = fetch_models.verify_silero_vad(torch_hub_dir=hub_dir)

    assert result.ok is True


def test_verify_silero_vad_false_when_only_a_non_default_ref_dir_present(tmp_path: Path) -> None:
    # torch.hub's own offline fallback only ever looks for "..._main" or
    # "..._master" (and its online path, on this repo, only ever creates
    # "..._master"). A directory for some other ref (e.g. a pinned release
    # tag) existing is not what torch.hub will find, so it must not count as
    # cached — a wildcard match here would pass this check while the real
    # load still reaches the network. Critic finding 5, round 1.
    hub_dir = tmp_path / "torchhub"
    repo_dir = hub_dir / "snakers4_silero-vad_v5.1.2"
    repo_dir.mkdir(parents=True)
    (repo_dir / "hubconf.py").write_text("# stub")

    result = fetch_models.verify_silero_vad(torch_hub_dir=hub_dir)

    assert result.ok is False


def test_ensure_silero_vad_verify_only_reports_missing_without_fetching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub_dir = tmp_path / "torchhub"
    hub_dir.mkdir()
    load_calls = []

    def spy(*args, **kwargs):
        # Record the call before doing anything else, so the assertion on
        # load_calls fails on its own merits even if a broken implementation
        # were to swallow an exception raised from here (Critic finding 2,
        # round 1: an AssertionError raised from this spy alone is caught by
        # ensure_silero_vad's `except Exception`, which turns it into a
        # (falsely reassuring) ok=False result without the spy ever having
        # proven non-invocation).
        load_calls.append((args, kwargs))
        raise RuntimeError("torch.hub.load must not run in verify-only mode")

    monkeypatch.setattr(fetch_models.torch.hub, "load", spy)

    result = fetch_models.ensure_silero_vad(torch_hub_dir=hub_dir, verify_only=True)

    assert result.ok is False
    assert load_calls == []


def test_ensure_silero_vad_fetch_honors_torch_hub_dir_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # torch.hub.load has no destination-directory argument of its own; it
    # always writes into whatever torch.hub.get_dir() currently resolves to.
    # A correct ensure_silero_vad must redirect that global before calling
    # load and restore it after, or the fetch lands in the ambient cache
    # while verification keeps checking the injected torch_hub_dir —
    # permanently and silently mismatched (Critic finding 1, round 2).
    injected_dir = tmp_path / "injected"
    injected_dir.mkdir()

    def fake_load(*args, **kwargs):
        # Mirrors what the real torch.hub.load does: writes into whatever
        # the global hub dir currently is.
        repo_dir = Path(fetch_models.torch.hub.get_dir()) / "snakers4_silero-vad_master"
        repo_dir.mkdir(parents=True)
        (repo_dir / "hubconf.py").write_text("# stub")

    monkeypatch.setattr(fetch_models.torch.hub, "load", fake_load)

    result = fetch_models.ensure_silero_vad(torch_hub_dir=injected_dir, verify_only=False)

    assert result.ok is True
    assert any(injected_dir.iterdir())  # the fetch landed in the injected dir, not the ambient one


def test_ensure_silero_vad_fetch_restores_ambient_torch_hub_dir_afterward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mutating the global torch.hub dir to honor the parameter must not leak
    # into any other test: assert it is restored exactly, success or not.
    injected_dir = tmp_path / "injected"
    injected_dir.mkdir()
    ambient_before = fetch_models.torch.hub.get_dir()

    def fake_load(*args, **kwargs):
        repo_dir = Path(fetch_models.torch.hub.get_dir()) / "snakers4_silero-vad_master"
        repo_dir.mkdir(parents=True)
        (repo_dir / "hubconf.py").write_text("# stub")

    monkeypatch.setattr(fetch_models.torch.hub, "load", fake_load)

    fetch_models.ensure_silero_vad(torch_hub_dir=injected_dir, verify_only=False)

    assert fetch_models.torch.hub.get_dir() == ambient_before


# --- CLI: main() ------------------------------------------------------------

def _seed_complete_silero_cache(torch_hub_dir: Path) -> None:
    repo_dir = torch_hub_dir / "snakers4_silero-vad_master"
    repo_dir.mkdir(parents=True)
    (repo_dir / "hubconf.py").write_text("# stub")


def test_main_no_args_exits_0_when_everything_already_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    hub_dir = tmp_path / "hub"
    torch_hub_dir = tmp_path / "torchhub"
    for repo_id in fetch_models.MODEL_REPOS:
        _seed_snapshot(hub_dir, repo_id)
    _seed_complete_silero_cache(torch_hub_dir)

    real_snapshot_download = fetch_models.snapshot_download
    fetch_calls = []

    def spy_snapshot_download(repo_id, *args, **kwargs):
        if kwargs.get("local_files_only"):
            return real_snapshot_download(repo_id, *args, **kwargs)
        # Real MODEL_REPOS ids are used here; never delegate a fetch-branch
        # call to the real network (see the analogous comment in
        # test_main_verify_only_never_calls_snapshot_download_or_torch_hub_load).
        fetch_calls.append(repo_id)
        raise RuntimeError("must not re-fetch an already-complete snapshot")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("torch.hub.load must not run when Silero is already cached")

    monkeypatch.setattr(fetch_models, "snapshot_download", spy_snapshot_download)
    monkeypatch.setattr(fetch_models.torch.hub, "load", fail_if_called)

    exit_code = fetch_models.main([], cache_dir=hub_dir, torch_hub_dir=torch_hub_dir)

    assert exit_code == 0
    assert fetch_calls == []  # idempotent: already complete, so no fetch attempted
    out = capsys.readouterr().out
    assert out.count("OK") == len(fetch_models.MODEL_REPOS) + 1  # every model + Silero
    assert fetch_models.SILERO_VAD_REPO in out


def test_main_verify_only_exits_0_when_everything_already_complete(tmp_path: Path) -> None:
    hub_dir = tmp_path / "hub"
    torch_hub_dir = tmp_path / "torchhub"
    for repo_id in fetch_models.MODEL_REPOS:
        _seed_snapshot(hub_dir, repo_id)
    _seed_complete_silero_cache(torch_hub_dir)

    exit_code = fetch_models.main(["--verify-only"], cache_dir=hub_dir, torch_hub_dir=torch_hub_dir)

    assert exit_code == 0


def test_main_verify_only_exits_2_when_a_model_is_missing(tmp_path: Path) -> None:
    hub_dir = tmp_path / "hub"
    torch_hub_dir = tmp_path / "torchhub"
    for repo_id in fetch_models.MODEL_REPOS[1:]:
        _seed_snapshot(hub_dir, repo_id)
    # MODEL_REPOS[0] deliberately not seeded.
    _seed_complete_silero_cache(torch_hub_dir)

    exit_code = fetch_models.main(["--verify-only"], cache_dir=hub_dir, torch_hub_dir=torch_hub_dir)

    assert exit_code == 2


def test_main_verify_only_exits_2_when_only_silero_is_missing(tmp_path: Path) -> None:
    # All MODEL_REPOS complete, Silero cache absent: this is the case that
    # would silently pass if main() ever stopped checking the Silero
    # precondition (Critic finding 1, round 1, mutant 2).
    hub_dir = tmp_path / "hub"
    torch_hub_dir = tmp_path / "torchhub"
    for repo_id in fetch_models.MODEL_REPOS:
        _seed_snapshot(hub_dir, repo_id)
    torch_hub_dir.mkdir()  # Silero never seeded

    exit_code = fetch_models.main(["--verify-only"], cache_dir=hub_dir, torch_hub_dir=torch_hub_dir)

    assert exit_code == 2


def test_main_no_args_exits_2_when_a_model_is_missing_and_fetch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Default (fetch) mode, but the fetch itself is made to fail without
    # touching the real network, so the exit-code contract is exercised
    # end-to-end (constructor -> ensure_model -> main) without a real
    # download attempt.
    hub_dir = tmp_path / "hub"
    torch_hub_dir = tmp_path / "torchhub"
    for repo_id in fetch_models.MODEL_REPOS[1:]:
        _seed_snapshot(hub_dir, repo_id)
    _seed_complete_silero_cache(torch_hub_dir)
    real_snapshot_download = fetch_models.snapshot_download

    def fake_snapshot_download(repo_id, *args, **kwargs):
        if kwargs.get("local_files_only"):
            return real_snapshot_download(repo_id, *args, **kwargs)
        raise OSError("simulated network failure, no real connection attempted")

    monkeypatch.setattr(fetch_models, "snapshot_download", fake_snapshot_download)

    exit_code = fetch_models.main([], cache_dir=hub_dir, torch_hub_dir=torch_hub_dir)

    assert exit_code == 2


def test_main_verify_only_never_calls_snapshot_download_or_torch_hub_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing seeded at all: every MODEL_REPOS entry and Silero all missing.
    # If --verify-only stopped being honored anywhere in main()'s wiring,
    # this is the case that would reach the network for every one of those
    # assets (Critic finding 1, round 1, mutant 3).
    hub_dir = tmp_path / "hub"
    hub_dir.mkdir()
    torch_hub_dir = tmp_path / "torchhub"
    torch_hub_dir.mkdir()
    real_snapshot_download = fetch_models.snapshot_download
    fetch_calls = []
    load_calls = []

    def spy_snapshot_download(repo_id, *args, **kwargs):
        if kwargs.get("local_files_only"):
            return real_snapshot_download(repo_id, *args, **kwargs)
        # Real MODEL_REPOS ids are used here (main() reads the module-level
        # MODEL_REPOS directly, so there is nothing else to inject) — never
        # delegate a fetch-branch call to the real network, or a broken
        # implementation under test would trigger a genuine multi-GB
        # download instead of failing this assertion.
        fetch_calls.append(repo_id)
        raise RuntimeError("snapshot_download fetch must not run in verify-only mode")

    def spy_torch_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise RuntimeError("torch.hub.load must not run in verify-only mode")

    monkeypatch.setattr(fetch_models, "snapshot_download", spy_snapshot_download)
    monkeypatch.setattr(fetch_models.torch.hub, "load", spy_torch_load)

    exit_code = fetch_models.main(["--verify-only"], cache_dir=hub_dir, torch_hub_dir=torch_hub_dir)

    assert exit_code == 2
    assert fetch_calls == []
    assert load_calls == []
