from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from icgs.data.collection.generation.distributed_contracts import (
    ARCHIVE_DATASET_IDENTITY,
    ARCHIVE_EPISODE_SCHEMA_VERSION,
    ARCHIVE_FORMAT_ID,
)
from icgs.data.datasets.generation_view_index import (
    DEFAULT_ROLE_SPECS,
    build_provisional_episode_pointers,
    finalize_generation_views,
)
from test_generation_views import _episode, _write_archive
from scripts import generation_finalize_views as finalization_cli


def _snapshot(
    root: Path,
    records: list[dict],
    *,
    source_snapshot_created_at_s: float = 1_758_780_000.0,
    source_revision: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    entries = [_write_archive(root, record) for record in records]
    payload = {
        "manifest_version": 3,
        "dataset_identity": ARCHIVE_DATASET_IDENTITY,
        "archive_format_id": ARCHIVE_FORMAT_ID,
        "episode_schema_version": ARCHIVE_EPISODE_SCHEMA_VERSION,
        "view_status": "PROVISIONAL",
        "hf_prefix": "datasets/icgs/archive-v1",
        "source_snapshot_created_at_s": source_snapshot_created_at_s,
        "total_episodes": len(entries),
        "episodes": entries,
        "failure_attempts": [{
            "attempt_id": "att-crashed",
            "episode_id": None,
            "outcome": "simulator_crash",
        }],
    }
    payload["total_failure_attempts"] = len(payload["failure_attempts"])
    if source_revision is not None:
        payload["source_revision"] = source_revision
    path = root / "dataset_manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_per_episode_provisional_pointers_bind_archive_and_never_claim_final_role(tmp_path: Path):
    episode = _episode("ep-provisional", outcome="valid_failure", n_actions=3)
    row = _write_archive(tmp_path, episode)
    manifest = json.loads((tmp_path / row["archive_manifest"]).read_text(encoding="utf-8"))

    pointers = build_provisional_episode_pointers(manifest)

    assert {pointer["view"] for pointer in pointers} == {
        "D_geom", "D_temporal", "D_dyn", "D_task",
    }
    assert all(pointer["status"] == "PROVISIONAL" for pointer in pointers)
    assert all(pointer["view_status"] == "PROVISIONAL" for pointer in pointers)
    assert all(pointer["role"] == "all" and pointer["mix"] is False for pointer in pointers)
    assert all(pointer["episode_id"] == "ep-provisional" for pointer in pointers)
    assert all(pointer["archive_ref"] == "episodes/T01/ep-provisional" for pointer in pointers)
    assert all(pointer["archive_manifest_sha256"] == row["file_sha256"]["episode.manifest.json"] for pointer in pointers)
    assert {pointer["sample_count"] for pointer in pointers} == {2, 3, 4}


def test_final_views_are_role_specific_mixed_only_for_train_temporal_and_dyn(tmp_path: Path):
    records = [
        _episode("ep-train-nominal", n_actions=6),
        _episode("ep-train-valid-failure", outcome="valid_failure", n_actions=1),
        _episode("ep-train-perturbed", kind="perturbed", n_actions=7),
        _episode("ep-validation", subset="train_val", n_actions=2),
        _episode("ep-development", split="dev", n_actions=3),
        _episode("ep-test", split="test", n_actions=4),
    ]
    dataset_manifest = _snapshot(tmp_path / "source", records)
    output_dir = tmp_path / "finalized" / ("a" * 40)

    receipt = finalize_generation_views(
        dataset_manifest,
        source_revision="a" * 40,
        role_specs=DEFAULT_ROLE_SPECS,
        mixture_seed=20260920,
        output_dir=output_dir,
    )

    assert receipt.status == "FINAL"
    assert receipt.source_revision == "a" * 40
    assert receipt.source_manifest_sha256 == hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()
    assert len(receipt.view_ids) == 12
    assert set(receipt.counts) == set(receipt.view_ids)
    train_geom = json.loads((output_dir / "train" / "D_geom.json").read_text(encoding="utf-8"))
    train_dyn = json.loads((output_dir / "train" / "D_dyn.json").read_text(encoding="utf-8"))
    validation_geom = json.loads((output_dir / "validation" / "D_geom.json").read_text(encoding="utf-8"))
    evaluation_geom = json.loads((output_dir / "evaluation" / "D_geom.json").read_text(encoding="utf-8"))

    assert train_geom["status"] == "FINAL" and train_geom["view_status"] == "FINAL"
    assert {row["episode_id"] for row in train_geom["samples"]} == {
        "ep-train-nominal", "ep-train-valid-failure", "ep-train-perturbed",
    }
    assert {row["episode_id"] for row in validation_geom["samples"]} == {"ep-validation"}
    assert {row["episode_id"] for row in evaluation_geom["samples"]} == {
        "ep-development", "ep-test",
    }
    assert "ep-train-valid-failure" in {row["episode_id"] for row in train_dyn["samples"]}
    assert len(train_dyn["samples"]) == 10
    assert sum(row["episode_kind"] == "nominal" for row in train_dyn["samples"]) == 7
    assert sum(row["episode_kind"] == "perturbed" for row in train_dyn["samples"]) == 3
    assert train_dyn["mixture"]["applied"] is True
    assert train_dyn["mixture"]["seed"] == 20260920
    assert validation_geom["mixture"]["applied"] is False
    assert receipt.counts["train/D_dyn"] == 10
    assert not any("att-crashed" in json.dumps(row) for row in train_geom["samples"])


def test_finalization_is_byte_deterministic_and_uses_source_snapshot_time(tmp_path: Path):
    dataset_manifest = _snapshot(tmp_path / "source", [_episode("ep-stable", n_actions=2)])
    output_dir = tmp_path / "finalized" / ("b" * 40)
    kwargs = {
        "source_revision": "b" * 40,
        "role_specs": DEFAULT_ROLE_SPECS,
        "mixture_seed": 41,
        "output_dir": output_dir,
    }

    first = finalize_generation_views(dataset_manifest, **kwargs)
    first_bytes = {
        str(path.relative_to(output_dir)): path.read_bytes()
        for path in output_dir.rglob("*.json")
    }
    second = finalize_generation_views(dataset_manifest, **kwargs)
    second_bytes = {
        str(path.relative_to(output_dir)): path.read_bytes()
        for path in output_dir.rglob("*.json")
    }

    assert first_bytes == second_bytes
    assert first.view_manifest_sha256 == second.view_manifest_sha256
    view = json.loads((output_dir / "train" / "D_geom.json").read_text(encoding="utf-8"))
    assert view["finalization_timestamp_s"] == 1_758_780_000.0
    assert view["finalization_timestamp_source"] == "source_snapshot_created_at_s"


def test_changed_dataset_snapshot_changes_digest_and_counts(tmp_path: Path):
    first_manifest = _snapshot(tmp_path / "first", [_episode("ep-first", n_actions=1)])
    second_manifest = _snapshot(
        tmp_path / "second",
        [_episode("ep-first", n_actions=1), _episode("ep-added", n_actions=3)],
    )
    first = finalize_generation_views(
        first_manifest,
        source_revision="c" * 40,
        role_specs=DEFAULT_ROLE_SPECS,
        mixture_seed=7,
        output_dir=tmp_path / "first-final",
    )
    second = finalize_generation_views(
        second_manifest,
        source_revision="d" * 40,
        role_specs=DEFAULT_ROLE_SPECS,
        mixture_seed=7,
        output_dir=tmp_path / "second-final",
    )

    assert first.source_manifest_sha256 != second.source_manifest_sha256
    assert first.counts["train/D_geom"] == 2
    assert second.counts["train/D_geom"] == 6


def test_finalization_fails_closed_on_revision_conflict_or_missing_snapshot_binding(tmp_path: Path):
    pinned_manifest = _snapshot(
        tmp_path / "pinned",
        [_episode("ep-revision", n_actions=1)],
        source_revision="e" * 40,
    )
    with pytest.raises(ValueError, match="source revision"):
        finalize_generation_views(
            pinned_manifest,
            source_revision="f" * 40,
            role_specs=DEFAULT_ROLE_SPECS,
            mixture_seed=1,
            output_dir=tmp_path / "conflict",
        )

    missing_timestamp = _snapshot(
        tmp_path / "no-time",
        [_episode("ep-no-time", n_actions=1)],
        source_snapshot_created_at_s=None,
    )
    payload = json.loads(missing_timestamp.read_text(encoding="utf-8"))
    payload.pop("source_snapshot_created_at_s")
    missing_timestamp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="source snapshot timestamp"):
        finalize_generation_views(
            missing_timestamp,
            source_revision="7" * 40,
            role_specs=DEFAULT_ROLE_SPECS,
            mixture_seed=1,
            output_dir=tmp_path / "no-time-final",
        )


def test_finalization_rejects_noncanonical_role_specs(tmp_path: Path):
    dataset_manifest = _snapshot(tmp_path / "source", [_episode("ep-role-spec", n_actions=1)])
    role_specs = {
        **DEFAULT_ROLE_SPECS,
        "evaluation": {"splits": ["train"], "subsets": ["train_core"]},
    }
    with pytest.raises(ValueError, match="role specification"):
        finalize_generation_views(
            dataset_manifest,
            source_revision="8" * 40,
            role_specs=role_specs,
            mixture_seed=1,
            output_dir=tmp_path / "bad-roles",
        )


def test_cli_uploads_pinned_snapshot_only_to_explicit_disjoint_prefix(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    dataset_manifest = _snapshot(source_root, [_episode("ep-cli", n_actions=2)])
    source_prefix = "datasets/icgs/archive-v1"
    source_revision = "9" * 40
    downloaded = []
    commits = []

    def fake_download(*, repo_id: str, repo_type: str, filename: str, revision: str) -> str:
        assert repo_id == "org/dataset"
        assert repo_type == "dataset"
        assert revision == source_revision
        downloaded.append(filename)
        relative = filename.removeprefix(source_prefix + "/")
        return str(source_root / relative)

    class FakeApi:
        def repo_info(self, *, repo_id: str, repo_type: str):
            assert repo_id == "org/dataset" and repo_type == "dataset"
            return SimpleNamespace(sha="a" * 40)

        def file_exists(self, **_kwargs) -> bool:
            return False

        def create_commit(self, **kwargs):
            commits.append({
                "paths": [operation.path_in_repo for operation in kwargs["operations"]],
                "contents": {
                    operation.path_in_repo: Path(operation.path_or_fileobj).read_bytes()
                    for operation in kwargs["operations"]
                },
                "parent_commit": kwargs["parent_commit"],
            })

    monkeypatch.setattr(finalization_cli, "hf_hub_download", fake_download)
    monkeypatch.setattr(finalization_cli, "HfApi", FakeApi)

    exit_code = finalization_cli.main([
        "--repo-id", "org/dataset",
        "--source-prefix", source_prefix,
        "--source-revision", source_revision,
        "--output-prefix", "snapshots/final-views",
        "--mixture-seed", "41",
    ])

    assert exit_code == 0
    assert set(downloaded) == {
        f"{source_prefix}/dataset_manifest.json",
        f"{source_prefix}/episodes/ep-cli/episode.manifest.json",
    }
    assert len(commits) == 1
    committed = commits[0]
    assert committed["parent_commit"] == "a" * 40
    expected_prefix = f"snapshots/final-views/{source_revision}/seed-41/"
    assert all(path.startswith(expected_prefix) for path in committed["paths"])
    assert len(committed["paths"]) == 13
    train_view = json.loads(committed["contents"][expected_prefix + "train/D_geom.json"])
    assert train_view["source_prefix"] == source_prefix
    assert train_view["source_revision"] == source_revision


def test_cli_rejects_output_prefix_that_overlaps_source():
    with pytest.raises(ValueError, match="separate"):
        finalization_cli._ensure_disjoint_prefixes(
            "datasets/icgs/archive-v1",
            "datasets/icgs/archive-v1/final-views",
        )
