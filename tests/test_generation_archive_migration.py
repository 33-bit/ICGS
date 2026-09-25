from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

try:
    from scripts.generation_archive_migrate import (
        MigrationError,
        MigrationTargetWriter,
        V2ArtifactReader,
        migrate_episode,
    )
except (ImportError, ModuleNotFoundError) as error:
    MigrationError = MigrationTargetWriter = V2ArtifactReader = migrate_episode = None
    _migration_import_error = str(error)
else:
    _migration_import_error = None

try:
    from icgs.data.collection.generation.distributed_contracts import ArchiveProfileConfig
    from icgs.data.collection.generation.episode_archive import (
        EpisodeArchiveReader,
        EpisodeArchiveWriter,
    )
except ModuleNotFoundError as error:
    ArchiveProfileConfig = EpisodeArchiveReader = EpisodeArchiveWriter = None
    _archive_import_error = str(error)
else:
    _archive_import_error = None


def _require_migration_api() -> None:
    assert _migration_import_error is None, f"v2 migration API is required: {_migration_import_error}"
    assert _archive_import_error is None, f"canonical archive API is required: {_archive_import_error}"


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _inventory(root: Path, *, filename: str = "artifact_manifest.json") -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() != filename
    }


def _complete_v2_inventory(root: Path) -> dict[str, dict[str, object]]:
    inventory = _inventory(root)
    manifest = root / "artifact_manifest.json"
    inventory["artifact_manifest.json"] = {"sha256": _sha256(manifest), "bytes": manifest.stat().st_size}
    return inventory


def _episode_record(outcome: str = "success") -> dict:
    pose_0 = np.eye(4, dtype=np.float64)
    pose_1 = np.eye(4, dtype=np.float64)
    pose_1[0, 3] = 0.25
    return {
        "schema_version": "icgs_episode_v2",
        "provenance": {
            "episode_id": "episode-migrate-0001",
            "attempt_id": "attempt-migrate-0001",
            "program_id": "T01",
            "split": "train",
            "scene_seed": 17,
            "asset_instance_id": "asset-migrate-1",
            "execution_mode": "scripted_waypoint_v1",
            "calibration_id": "calibration-v1",
            "observation_origin": "measured",
            "raw_commands_id": "raw-v1",
            "materialized_commands_id": "materialized-v1",
            "episode_kind": "nominal",
            "outcome": outcome,
            "dataset_version": "icgs-primary-v3",
            "collection_seed": 20260920,
            "failure_type": None if outcome == "success" else "predicate_unsatisfied",
            "terminal_reason": "predicate_satisfied" if outcome == "success" else "predicate_failed",
            "terminal_t": 1,
            "valid_observation_until": 1,
            "terminated": True,
            "truncated": False,
        },
        "online_observations": [
            {
                "points": [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                "point_valid": [True, False],
                "T_w_e": pose_0.tolist(),
                "grip": 0,
            },
            {
                "points": [[6.0, 7.0, 8.0]],
                "point_valid": [True],
                "T_w_e": pose_1.tolist(),
                "grip": 1,
            },
        ],
        "transitions": [
            {
                "command": {"T_w_e": pose_1.tolist(), "grip": 1, "duration_s": 0.1},
                "achieved_duration_s": 0.08,
                "physics_substeps": 4,
                "before_boundary": 0,
                "after_boundary": 1,
            }
        ],
        "dt": [0.08],
        "robot_states": [
            {"joint_positions": [0.1, 0.2], "debug_state": {"calibrated": True}},
            {"joint_positions": [0.3, 0.4], "debug_state": {"calibrated": False}},
        ],
        "object_states": [{"block": [1.0, 2.0, 3.0]}, {"block": [2.0, 3.0, 4.0]}],
        "events": [{"event_id": "place", "target": "drawer"}],
        "rho": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32).tolist(),
        "rho_valid": [[True, False], [True, True]],
    }


def _write_episode_fixture(root: Path, outcome: str = "success", *, malformed: bool = False) -> dict:
    root.mkdir(parents=True)
    record = _episode_record(outcome)
    if malformed:
        record["transitions"] = []
        record["dt"] = []
    _json_write(root / "episode.json", record)

    layout = root / "layout"
    _json_write(layout / "episode.json", {
        "episode_id": "episode-migrate-0001",
        "program_id": "T01",
        "split": "train",
        "layout_version": 2,
        "outcome": outcome,
    })
    _json_write(layout / "result.json", {
        "success": outcome == "success",
        "outcome": outcome,
        "terminal_reason": "predicate_satisfied" if outcome == "success" else "predicate_failed",
    })
    point_frames = [
        np.asarray(record["online_observations"][index]["points"], dtype=np.float32)
        for index in range(2)
    ]
    point_offsets = np.asarray([0, 2, 3], dtype=np.int64)
    point_values = np.concatenate(point_frames, axis=0)
    point_path = layout / "observations" / "pointcloud" / "frames.npz"
    point_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(point_path, points=point_values, offsets=point_offsets)
    pose_values = np.asarray([item["T_w_e"] for item in record["online_observations"]], dtype=np.float64)
    grip_values = np.asarray([item["grip"] for item in record["online_observations"]], dtype=np.float32)
    robot_dir = layout / "robot"
    robot_dir.mkdir(parents=True, exist_ok=True)
    np.save(robot_dir / "ee_pose.npy", pose_values)
    np.save(robot_dir / "gripper.npy", grip_values)
    np.save(robot_dir / "joint_state.npy", np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64))
    action_path = layout / "actions" / "actions.npy"
    action_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(action_path, np.asarray([[0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float64))
    task_dir = layout / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    _json_write(task_dir / "events.json", [{"event_id": "place", "primitive": "reach"}])
    (task_dir / "collisions.jsonl").write_text('{"kind":"none"}\n', encoding="utf-8")
    np.save(task_dir / "rho.npy", np.asarray(record["rho"], dtype=np.float32))
    np.save(task_dir / "rho_valid.npy", np.asarray(record["rho_valid"], dtype=np.bool_))
    layout_files = _inventory(layout, filename="layout_manifest.json")
    _json_write(layout / "layout_manifest.json", {
        "layout_version": 2,
        "episode_id": "episode-migrate-0001",
        "success": outcome == "success",
        "boundaries": 2,
        "files": layout_files,
    })

    execution = {
        "source_run_id": "run-v2-fixture",
        "code_revision": "b" * 40,
        "preprocessing_identity": "measured_raw_v2_fixture",
        "predicate_distances_m": {"drawer": [0.01, 0.02]},
        "unique_debug_note": "kept from execution.json",
        "_robot_states": [{"arm": [1.0, 2.0]}, {"arm": [3.0, 4.0]}],
    }
    _json_write(root / "execution.json", execution)
    telemetry_path = root / "telemetry.npz"
    np.savez_compressed(
        telemetry_path,
        ee_pose=pose_values,
        gripper=grip_values,
        action=np.load(action_path, allow_pickle=False),
        achieved_dt=np.asarray([0.08], dtype=np.float64),
        unique_sensor_state=np.asarray([[1, 2], [3, 4]], dtype=np.int16),
    )
    _json_write(root / "views" / "D_geom.json", {
        "view": "D_geom",
        "episode_id": "episode-migrate-0001",
        "pointers": [{"boundary": 0, "role": "train"}],
    })
    _json_write(root / "artifact_manifest.json", {
        "attempt_id": "attempt-migrate-0001",
        "episode_id": "episode-migrate-0001",
        "program_id": "T01",
        "outcome": outcome,
        "files": _inventory(root),
    })
    return record


def _write_attempt_fixture(root: Path, outcome: str) -> dict:
    root.mkdir(parents=True)
    attempt = {
        "attempt_id": f"attempt-{outcome}-0001",
        "episode_id": None,
        "program_id": "T01",
        "outcome": outcome,
        "error_type": "simulator_exception" if outcome == "simulator_crash" else "observation_schema",
        "error": "fixture failure",
        "traceback": "fixture traceback line 1\nfixture traceback line 2",
        "scene_seed": 19,
        "split": "train",
        "episode_kind": "nominal",
        "valid_observation_until": 1,
    }
    _json_write(root / "attempt.json", attempt)
    prefix_path = root / "valid_prefix.npz"
    poses = np.stack([np.eye(4), np.eye(4)]).astype(np.float64)
    poses[1, 0, 3] = 0.5
    np.savez_compressed(
        prefix_path,
        actions=np.asarray([[0.5, 0.0, 0.0]], dtype=np.float64),
        points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        point_offsets=np.asarray([0, 2, 3], dtype=np.int64),
        T_w_e=poses,
        grip=np.asarray([0.0, 1.0], dtype=np.float32),
        point_valid=np.asarray([True, False, True], dtype=np.bool_),
    )
    _json_write(root / "execution.json", {"attempt_debug_note": "attempt metadata retained"})
    np.savez_compressed(
        root / "telemetry.npz",
        unique_attempt_sensor_state=np.asarray([[7, 8], [9, 10]], dtype=np.int16),
    )
    _json_write(root / "artifact_manifest.json", {
        "attempt_id": attempt["attempt_id"],
        "episode_id": None,
        "program_id": "T01",
        "outcome": outcome,
        "files": _inventory(root),
    })
    return attempt


def _identity(kind: str = "episode", record_id: str = "episode-migrate-0001") -> dict:
    plural = "episodes" if kind == "episode" else "attempts"
    return {
        "source_repo_id": "test/generation-v2",
        "source_revision": "a" * 40,
        "source_prefix": "legacy-v2",
        "source_artifact_path": f"legacy-v2/{plural}/T01/{record_id}",
        "source_kind": kind,
        "program_id": "T01",
        "record_id": record_id,
        "target_repo_id": "test/generation-v3",
        "target_prefix": "lossless-v1",
        "target_artifact_path": f"lossless-v1/{plural}/T01/{record_id}",
        "target_receipt_path": f"lossless-v1/migration-receipts/{kind}/T01/{record_id}.json",
    }


def _target(tmp_path: Path, *, name: str = "target"):
    _require_migration_api()
    output_dir = tmp_path / name / "archive"
    receipt_path = tmp_path / name / "archive.migration_receipt.json"
    failure_path = tmp_path / "receipts" / f"{name}.migration_failure.json"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    profile = ArchiveProfileConfig(chunk_boundaries=1, local_artifact_retention="keep")
    bound_writer = MigrationTargetWriter(
        EpisodeArchiveWriter(profile),
        output_dir=output_dir,
        receipt_path=receipt_path,
        failure_receipt_path=failure_path,
    )
    return bound_writer, profile, failure_path


def _run_migration(source_dir: Path, tmp_path: Path, *, kind: str = "episode", record_id: str = "episode-migrate-0001"):
    target, profile, failure_path = _target(tmp_path, name=f"{kind}-{record_id}")
    receipt = migrate_episode(
        V2ArtifactReader(source_dir),
        target,
        source_identity=_identity(kind, record_id),
        target_profile=profile,
    )
    return receipt, target, failure_path


class _FakeRepoFile:
    def __init__(self, path: str, size: int) -> None:
        self.path = path
        self.size = size


class _FakeCommitOperationAdd:
    def __init__(self, *, path_in_repo: str, path_or_fileobj: str) -> None:
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class _FakeHub:
    def __init__(self, source_repo: str, source_revision: str, source_files: dict[str, bytes], target_repo: str) -> None:
        self.source_repo = source_repo
        self.target_repo = target_repo
        self.source_revision = source_revision
        self.target_head = "f" * 40
        self.revisions = {
            source_repo: {source_revision: dict(source_files)},
            target_repo: {self.target_head: {}},
        }
        self.branches = {target_repo: {"main": self.target_head}}
        self.list_calls: list[tuple[str, str, str]] = []
        self.commit_calls: list[dict] = []
        self._commit_counter = 0

    def repo_info(self, *, repo_id: str, repo_type: str, revision: str | None = None):
        assert repo_type == "dataset"
        if repo_id == self.source_repo:
            sha = self.source_revision
        elif repo_id == self.target_repo:
            if revision is None or revision in self.branches[repo_id]:
                sha = self.branches[repo_id].get(revision or "main")
            else:
                sha = revision
        else:
            raise AssertionError(f"unexpected repo id: {repo_id}")
        if sha not in self.revisions[repo_id]:
            raise AssertionError(f"unexpected revision: {repo_id}@{sha}")
        return SimpleNamespace(sha=sha)

    def list_repo_tree(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        path_in_repo: str,
        recursive: bool,
        expand: bool,
    ):
        assert repo_type == "dataset" and recursive is True and expand is False
        self.list_calls.append((repo_id, revision, path_in_repo))
        files = self.revisions[repo_id][revision]
        return [
            _FakeRepoFile(path, len(value))
            for path, value in sorted(files.items())
            if path == path_in_repo or path.startswith(path_in_repo + "/")
        ]

    def file_exists(self, *, repo_id: str, repo_type: str, revision: str, filename: str) -> bool:
        assert repo_type == "dataset"
        return filename in self.revisions[repo_id][revision]

    def create_commit(self, *, repo_id: str, repo_type: str, revision: str, parent_commit: str, operations, **_kwargs):
        assert repo_type == "dataset"
        assert revision == "main"
        assert parent_commit == self.branches[repo_id]["main"]
        self._commit_counter += 1
        commit_oid = f"{self._commit_counter:040x}"
        snapshot = dict(self.revisions[repo_id][parent_commit])
        for operation in operations:
            snapshot[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
        self.revisions[repo_id][commit_oid] = snapshot
        self.branches[repo_id]["main"] = commit_oid
        self.commit_calls.append({"parent": parent_commit, "paths": sorted(snapshot)})
        return SimpleNamespace(oid=commit_oid)

    def hub_download(self, *, repo_id: str, repo_type: str, filename: str, revision: str, cache_dir: str) -> str:
        assert repo_type == "dataset"
        value = self.revisions[repo_id][revision][filename]
        destination = Path(cache_dir) / hashlib.sha256(f"{repo_id}@{revision}/{filename}".encode()).hexdigest()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(value)
        return str(destination)


def _install_fake_hub(monkeypatch, hub: _FakeHub) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: hub,
        hf_hub_download=hub.hub_download,
        CommitOperationAdd=_FakeCommitOperationAdd,
    ))


def _fake_hub_for_source(source: Path) -> tuple[_FakeHub, dict[str, str]]:
    identity = _identity()
    remote_files = {
        f"{identity['source_artifact_path']}/{path.relative_to(source).as_posix()}": path.read_bytes()
        for path in source.rglob("*") if path.is_file()
    }
    hub = _FakeHub(
        identity["source_repo_id"], identity["source_revision"], remote_files, identity["target_repo_id"]
    )
    return hub, identity


def _cli_args(tmp_path: Path, identity: dict[str, str]) -> list[str]:
    return [
        "--source-repo-id", identity["source_repo_id"],
        "--source-revision", identity["source_revision"],
        "--source-prefix", identity["source_prefix"],
        "--kind", identity["source_kind"],
        "--program-id", identity["program_id"],
        "--record-id", identity["record_id"],
        "--target-repo-id", identity["target_repo_id"],
        "--target-prefix", identity["target_prefix"],
        "--target-branch", "main",
        "--receipt-dir", str(tmp_path / "cli-receipts"),
        "--scratch-root", str(tmp_path / "cli-scratch"),
        "--max-source-bytes", "10000000",
        "--max-target-bytes", "10000000",
        "--max-scratch-bytes", "30000000",
        "--max-decoded-sidecar-bytes", "10000000",
        "--chunk-boundaries", "1",
    ]


def test_success_episode_migration_recovers_typed_layout_values_and_preserves_debug(tmp_path: Path) -> None:
    _require_migration_api()
    source = tmp_path / "v2-success"
    expected = _write_episode_fixture(source, "success")

    receipt, target, _failure_path = _run_migration(source, tmp_path)

    reader = EpisodeArchiveReader(target.output_dir / "episode.manifest.json")
    restored = reader.to_episode_record()
    assert _plain(restored) == _plain(expected)
    np.testing.assert_array_equal(reader.observation(0)["points"], np.asarray(expected["online_observations"][0]["points"], dtype=np.float32))
    assert reader.observation(0)["points"].dtype == np.float32
    assert reader.observation(0)["T_w_e"].dtype == np.float64
    assert reader.observation(0)["grip"] == 0.0
    assert restored["provenance"] == expected["provenance"]
    assert restored["events"] == expected["events"]
    assert restored["robot_states"] == expected["robot_states"]
    assert reader.debug_metadata["source_execution"]["unique_debug_note"] == "kept from execution.json"
    assert reader.debug_metadata["v2_migration"]["source_identity"] == _identity()
    assert reader.debug_metadata["v2_migration"]["target_identity"] == receipt.target_identity
    assert reader.manifest.payload["source_run_id"] == "run-v2-fixture"
    assert reader.manifest.payload["code_revision"] == "b" * 40
    assert reader.raw_arrays["telemetry_unique_sensor_state"].dtype == np.int16
    assert receipt.field_counts["numeric_dtype_ambiguities"] == sum(
        warning.startswith("dtype_ambiguous:") for warning in receipt.conversion_warnings
    )
    assert receipt.source_artifact_hashes["episode.json"]["sha256"] == _sha256(source / "episode.json")
    assert receipt.target_archive_hashes["artifact_manifest.json"]["bytes"] > 0
    assert receipt.timeline_counts["observations"] == 2
    assert receipt.timeline_counts["transitions"] == 1
    assert any("dtype" in warning for warning in receipt.conversion_warnings)
    assert (target.receipt_path).is_file()


def test_sensitive_episode_semantic_change_is_rejected_with_redacted_receipt(tmp_path: Path) -> None:
    _require_migration_api()
    source = tmp_path / "v2-sensitive-field"
    record = _write_episode_fixture(source, "success")
    record["provenance"]["intervention_params"] = {"api_key": "fixture-secret-should-not-leak"}
    _json_write(source / "episode.json", record)
    artifact_manifest = json.loads((source / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifact_manifest["files"] = _inventory(source)
    _json_write(source / "artifact_manifest.json", artifact_manifest)
    source_before = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*") if path.is_file()
    }
    target, profile, failure_path = _target(tmp_path, name="sensitive")

    with pytest.raises(MigrationError, match="redaction would alter v2 semantics"):
        migrate_episode(
            V2ArtifactReader(source), target,
            source_identity=_identity(), target_profile=profile,
        )

    failure_text = failure_path.read_text(encoding="utf-8")
    assert "fixture-secret-should-not-leak" not in failure_text
    assert "redaction would alter v2 semantics" in failure_text
    assert not target.output_dir.exists()
    assert {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*") if path.is_file()
    } == source_before


def test_compressed_sidecar_decoded_budget_is_checked_before_numpy_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _require_migration_api()
    source = tmp_path / "v2-compressed-sidecar"
    _write_episode_fixture(source, "success")
    compressed_sidecar = source / "extra" / "highly_compressible.npz"
    compressed_sidecar.parent.mkdir(parents=True)
    np.savez_compressed(compressed_sidecar, payload=np.zeros(600 * 1024, dtype=np.uint8))
    second_compressed_sidecar = source / "extra" / "also_compressible.npz"
    np.savez_compressed(second_compressed_sidecar, payload=np.zeros(600 * 1024, dtype=np.uint8))
    assert compressed_sidecar.stat().st_size < 1024 * 1024
    assert second_compressed_sidecar.stat().st_size < 1024 * 1024
    artifact_manifest = json.loads((source / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifact_manifest["files"] = _inventory(source)
    _json_write(source / "artifact_manifest.json", artifact_manifest)
    target, profile, failure_path = _target(tmp_path, name="decoded-sidecar-budget")

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("NumPy loading started before aggregate decoded-sidecar preflight")

    monkeypatch.setattr(np, "load", unexpected_load)
    with pytest.raises(MigrationError, match="decoded sidecar.*max-decoded-sidecar-bytes"):
        migrate_episode(
            V2ArtifactReader(source, max_decoded_sidecar_bytes=1024 * 1024),
            target,
            source_identity=_identity(),
            target_profile=profile,
        )

    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert "max-decoded-sidecar-bytes" in failure["error"]["message"]
    assert not target.output_dir.exists()


def test_npy_declared_shape_is_bounded_before_numpy_load(tmp_path: Path, monkeypatch) -> None:
    _require_migration_api()
    source = tmp_path / "v2-large-npy-header"
    _write_episode_fixture(source, "success")
    large_sidecar = source / "extra" / "large.npy"
    large_sidecar.parent.mkdir(parents=True)
    with large_sidecar.open("wb") as stream:
        np.lib.format.write_array_header_1_0(stream, {
            "descr": np.dtype(np.uint8).str,
            "fortran_order": False,
            "shape": (4 * 1024 * 1024,),
        })
    assert large_sidecar.stat().st_size < 1024
    artifact_manifest = json.loads((source / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifact_manifest["files"] = _inventory(source)
    _json_write(source / "artifact_manifest.json", artifact_manifest)
    target, profile, failure_path = _target(tmp_path, name="npy-decoded-sidecar-budget")

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("NumPy loading started before aggregate decoded-sidecar preflight")

    monkeypatch.setattr(np, "load", unexpected_load)
    with pytest.raises(MigrationError, match="decoded sidecar.*max-decoded-sidecar-bytes"):
        migrate_episode(
            V2ArtifactReader(source, max_decoded_sidecar_bytes=1024 * 1024),
            target,
            source_identity=_identity(),
            target_profile=profile,
        )

    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert "max-decoded-sidecar-bytes" in failure["error"]["message"]
    assert not target.output_dir.exists()


def test_valid_failure_episode_remains_an_episode_archive(tmp_path: Path) -> None:
    _require_migration_api()
    source = tmp_path / "v2-valid-failure"
    expected = _write_episode_fixture(source, "valid_failure")

    receipt, target, _failure_path = _run_migration(source, tmp_path, record_id="episode-migrate-0001")

    reader = EpisodeArchiveReader(target.output_dir / "episode.manifest.json")
    assert reader.manifest.payload["archive_kind"] == "episode"
    assert reader.manifest.payload["outcome"] == "valid_failure"
    assert reader.to_episode_record()["provenance"] == expected["provenance"]
    assert receipt.timeline_counts["transitions"] == 1


def test_crash_attempt_migrates_as_attempt_and_keeps_prefix(tmp_path: Path) -> None:
    _require_migration_api()
    source = tmp_path / "v2-crash"
    expected = _write_attempt_fixture(source, "simulator_crash")

    receipt, target, _failure_path = _run_migration(
        source, tmp_path, kind="attempt", record_id=expected["attempt_id"]
    )

    reader = EpisodeArchiveReader(target.output_dir / "attempt.manifest.json")
    assert reader.manifest.payload["archive_kind"] == "attempt"
    assert reader.manifest.payload["episode_id"] is None
    assert reader.to_episode_record() == expected
    assert reader.raw_arrays["points"].dtype == np.float32
    assert reader.raw_arrays["point_valid"].dtype == np.bool_
    assert reader.raw_arrays["telemetry_unique_attempt_sensor_state"].dtype == np.int16
    assert reader.debug_metadata["source_execution"]["attempt_debug_note"] == "attempt metadata retained"
    assert reader.manifest.payload["source_run_id"].startswith("v2-migration:")
    assert reader.manifest.payload["code_revision"] == "unknown-v2-generator-revision"
    assert reader.debug_metadata["v2_migration"]["original_identity_fields_unavailable"] == [
        "source_run_id", "code_revision",
    ]
    assert any("source_run_id_unavailable" in warning for warning in receipt.conversion_warnings)
    assert receipt.timeline_counts["observations"] == 2
    assert receipt.timeline_counts["transitions"] == 1


def test_invalid_observation_attempt_remains_outside_episode_archive(tmp_path: Path) -> None:
    _require_migration_api()
    source = tmp_path / "v2-invalid-observation"
    expected = _write_attempt_fixture(source, "invalid_observation")

    receipt, target, _failure_path = _run_migration(
        source, tmp_path, kind="attempt", record_id=expected["attempt_id"]
    )

    reader = EpisodeArchiveReader(target.output_dir / "attempt.manifest.json")
    assert reader.manifest.payload["outcome"] == "invalid_observation"
    assert reader.to_episode_record() == expected
    assert receipt.timeline_counts["observations"] == 2


def test_malformed_source_writes_failure_receipt_without_target_archive(tmp_path: Path) -> None:
    _require_migration_api()
    source = tmp_path / "v2-malformed"
    _write_episode_fixture(source, "success", malformed=True)
    target, profile, failure_path = _target(tmp_path, name="malformed")

    with pytest.raises(MigrationError, match="transition|episode"):
        migrate_episode(
            V2ArtifactReader(source), target,
            source_identity=_identity(), target_profile=profile,
        )

    assert failure_path.is_file()
    assert json.loads(failure_path.read_text(encoding="utf-8"))["status"] == "failed"
    assert not target.output_dir.exists()


def test_source_hash_mismatch_is_rejected_and_recorded(tmp_path: Path) -> None:
    _require_migration_api()
    source = tmp_path / "v2-hash-mismatch"
    _write_episode_fixture(source, "success")
    with (source / "episode.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    target, profile, failure_path = _target(tmp_path, name="hash-mismatch")

    with pytest.raises(MigrationError, match="hash|inventory|integrity"):
        migrate_episode(
            V2ArtifactReader(source), target,
            source_identity=_identity(), target_profile=profile,
        )

    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert "sha256" in failure["error"]["message"].casefold() or "hash" in failure["error"]["message"].casefold()
    assert failure["source_artifact_hashes"]["episode.json"]["sha256"] != failure["declared_source_artifact_hashes"]["episode.json"]["sha256"]
    assert not target.output_dir.exists()


def test_nested_artifact_manifest_name_is_included_in_source_hash_inventory(tmp_path: Path) -> None:
    _require_migration_api()
    source = tmp_path / "v2-nested-manifest-name"
    _write_episode_fixture(source, "success")
    nested = source / "debug" / "artifact_manifest.json"
    _json_write(nested, {"debug_artifact": "kept"})
    artifact_manifest = json.loads((source / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifact_manifest["files"] = _inventory(source)
    _json_write(source / "artifact_manifest.json", artifact_manifest)

    receipt, target, _failure_path = _run_migration(source, tmp_path)

    assert receipt.source_artifact_hashes["debug/artifact_manifest.json"] == {
        "sha256": _sha256(nested), "bytes": nested.stat().st_size,
    }
    reader = EpisodeArchiveReader(target.output_dir / "episode.manifest.json")
    assert reader.debug_metadata["source_sidecars"]["debug/artifact_manifest.json"] == {"debug_artifact": "kept"}


def test_unlisted_nested_artifact_manifest_name_is_rejected(tmp_path: Path) -> None:
    _require_migration_api()
    source = tmp_path / "v2-unlisted-nested-manifest"
    _write_episode_fixture(source, "success")
    _json_write(source / "extra" / "artifact_manifest.json", {"not_in_inventory": True})
    target, profile, failure_path = _target(tmp_path, name="unlisted-nested-manifest")

    with pytest.raises(MigrationError, match="inventory is incomplete"):
        migrate_episode(
            V2ArtifactReader(source), target,
            source_identity=_identity(), target_profile=profile,
        )

    assert json.loads(failure_path.read_text(encoding="utf-8"))["status"] == "failed"
    assert not target.output_dir.exists()


def test_identical_migration_rerun_reuses_verified_archive_without_rewriting(tmp_path: Path) -> None:
    _require_migration_api()
    source = tmp_path / "v2-idempotent"
    _write_episode_fixture(source, "success")
    first, target, _failure_path = _run_migration(source, tmp_path, record_id="episode-migrate-0001")
    manifest_path = target.output_dir / "episode.manifest.json"
    original_mtime = manifest_path.stat().st_mtime_ns

    second = migrate_episode(
        V2ArtifactReader(source), target,
        source_identity=_identity(), target_profile=target.archive_writer.profile,
    )

    assert second.as_dict() == first.as_dict()
    assert manifest_path.stat().st_mtime_ns == original_mtime


def test_hf_cli_publishes_only_selected_source_and_reuses_identical_target_offline(tmp_path: Path, monkeypatch) -> None:
    _require_migration_api()
    source = tmp_path / "v2-hf-source"
    _write_episode_fixture(source, "success")
    original_source_bytes = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    hub, identity = _fake_hub_for_source(source)
    original_remote_source = dict(hub.revisions[identity["source_repo_id"]][identity["source_revision"]])
    _install_fake_hub(monkeypatch, hub)

    from scripts.generation_archive_migrate import main

    args = _cli_args(tmp_path, identity)
    assert main(args) == 0
    committed_count = len(hub.commit_calls)
    assert committed_count == 1
    local_receipt_path = tmp_path / "cli-receipts" / "episode-T01-episode-migrate-0001.json"
    receipt = json.loads(local_receipt_path.read_text(encoding="utf-8"))
    assert receipt["source_artifact_hashes"] == _complete_v2_inventory(source)
    target_revision = hub.branches[identity["target_repo_id"]]["main"]
    target_snapshot = hub.revisions[identity["target_repo_id"]][target_revision]
    remote_target_inventory = {
        path.removeprefix(identity["target_artifact_path"] + "/"): {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        for path, payload in target_snapshot.items()
        if path.startswith(identity["target_artifact_path"] + "/")
    }
    assert receipt["target_archive_hashes"] == remote_target_inventory
    assert set(remote_target_inventory) == set(receipt["target_archive_hashes"])
    assert main(args) == 0

    source_tree_queries = [path for repo_id, _revision, path in hub.list_calls if repo_id == identity["source_repo_id"]]
    assert source_tree_queries == [identity["source_artifact_path"], identity["source_artifact_path"]]
    assert len(hub.commit_calls) == committed_count
    assert {path: path.read_bytes() for path in source.rglob("*") if path.is_file()} == original_source_bytes
    assert hub.revisions[identity["source_repo_id"]][identity["source_revision"]] == original_remote_source


def test_hf_cli_enforces_decoded_sidecar_byte_cap_before_loading(tmp_path: Path, monkeypatch) -> None:
    _require_migration_api()
    source = tmp_path / "v2-hf-compressed-sidecar"
    _write_episode_fixture(source, "success")
    compressed_sidecar = source / "extra" / "highly_compressible.npz"
    compressed_sidecar.parent.mkdir(parents=True)
    np.savez_compressed(compressed_sidecar, payload=np.zeros(2 * 1024 * 1024, dtype=np.uint8))
    artifact_manifest = json.loads((source / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifact_manifest["files"] = _inventory(source)
    _json_write(source / "artifact_manifest.json", artifact_manifest)
    hub, identity = _fake_hub_for_source(source)
    _install_fake_hub(monkeypatch, hub)
    from scripts.generation_archive_migrate import main
    args = _cli_args(tmp_path, identity)
    args[args.index("--max-decoded-sidecar-bytes") + 1] = str(1024 * 1024)

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("NumPy loading started before CLI decoded-sidecar preflight")

    monkeypatch.setattr(np, "load", unexpected_load)
    assert main(args) == 2
    failure_paths = list((tmp_path / "cli-receipts").glob("*.failure.json"))
    assert len(failure_paths) == 1
    failure = json.loads(failure_paths[0].read_text(encoding="utf-8"))
    assert "max-decoded-sidecar-bytes" in failure["error"]["message"]
    assert not hub.commit_calls


def test_hf_cli_redacts_migration_exception_from_stdout_and_failure_receipt(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _require_migration_api()
    source = tmp_path / "v2-hf-secret-error"
    _write_episode_fixture(source, "success")
    hub, identity = _fake_hub_for_source(source)

    def fail_commit(**_kwargs):
        raise RuntimeError("upload rejected api_key=fixture-api-secret token=fixture-token-secret")

    monkeypatch.setattr(hub, "create_commit", fail_commit)
    _install_fake_hub(monkeypatch, hub)
    from scripts.generation_archive_migrate import main

    assert main(_cli_args(tmp_path, identity)) == 2
    stdout = capsys.readouterr().out
    failure_paths = list((tmp_path / "cli-receipts").glob("*.failure.json"))
    assert len(failure_paths) == 1
    failure_text = failure_paths[0].read_text(encoding="utf-8")

    for secret in ("fixture-api-secret", "fixture-token-secret"):
        assert secret not in stdout
        assert secret not in failure_text
    assert "api_key=[REDACTED]" in stdout
    assert "token=[REDACTED]" in stdout
    assert "api_key=[REDACTED]" in failure_text
    assert "token=[REDACTED]" in failure_text


def test_hf_cli_redacts_failure_receipt_write_exception_from_stdout(tmp_path: Path, monkeypatch, capsys) -> None:
    _require_migration_api()
    source = tmp_path / "v2-hf-receipt-write-error"
    _write_episode_fixture(source, "success")
    hub, identity = _fake_hub_for_source(source)

    def fail_commit(**_kwargs):
        raise RuntimeError("api_key=fixture-upload-secret")

    monkeypatch.setattr(hub, "create_commit", fail_commit)
    _install_fake_hub(monkeypatch, hub)
    from scripts import generation_archive_migrate

    original_write = generation_archive_migrate._write_exclusive_json

    def fail_failure_receipt(path, payload):
        if Path(path).name.endswith(".failure.json"):
            raise OSError("secret=fixture-receipt-write-secret")
        return original_write(path, payload)

    monkeypatch.setattr(generation_archive_migrate, "_write_exclusive_json", fail_failure_receipt)

    assert generation_archive_migrate.main(_cli_args(tmp_path, identity)) == 2
    stdout = capsys.readouterr().out
    assert "fixture-upload-secret" not in stdout
    assert "fixture-receipt-write-secret" not in stdout
    assert "api_key=[REDACTED]" in stdout
    assert "secret=[REDACTED]" in stdout


def test_hf_cli_rejects_post_commit_corruption_before_writing_success_receipt(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _require_migration_api()
    source = tmp_path / "v2-hf-post-commit-corruption"
    _write_episode_fixture(source, "success")
    hub, identity = _fake_hub_for_source(source)
    original_create_commit = hub.create_commit

    def create_then_corrupt(**kwargs):
        commit = original_create_commit(**kwargs)
        snapshot = hub.revisions[identity["target_repo_id"]][commit.oid]
        remote_path = next(
            path for path in snapshot
            if path.startswith(identity["target_artifact_path"] + "/")
        )
        payload = snapshot[remote_path]
        snapshot[remote_path] = payload[:-1] + bytes([payload[-1] ^ 0x01])
        return commit

    monkeypatch.setattr(hub, "create_commit", create_then_corrupt)
    _install_fake_hub(monkeypatch, hub)
    from scripts.generation_archive_migrate import main

    assert main(_cli_args(tmp_path, identity)) == 2
    stdout = capsys.readouterr().out
    assert "verification hash mismatch" in stdout
    assert len(hub.commit_calls) == 1
    assert not (tmp_path / "cli-receipts" / "episode-T01-episode-migrate-0001.json").exists()
    failure_paths = list((tmp_path / "cli-receipts").glob("*.failure.json"))
    assert len(failure_paths) == 1
    assert json.loads(failure_paths[0].read_text(encoding="utf-8"))["status"] == "failed"


def test_hf_cli_rejects_nested_target_prefix_before_hub_access(tmp_path: Path, monkeypatch) -> None:
    _require_migration_api()
    source = tmp_path / "v2-hf-prefix-check"
    _write_episode_fixture(source, "success")
    hub, identity = _fake_hub_for_source(source)
    _install_fake_hub(monkeypatch, hub)
    identity["target_prefix"] = identity["source_prefix"] + "/new-archive"
    args = _cli_args(tmp_path, identity)

    from scripts.generation_archive_migrate import main
    assert main(args) == 2
    assert not hub.list_calls
    assert not hub.commit_calls
    assert list((tmp_path / "cli-receipts").glob("*.failure.json"))


@pytest.mark.parametrize("conflict", ["partial", "mutated"])
def test_hf_cli_fails_closed_on_partial_or_conflicting_target(tmp_path: Path, monkeypatch, conflict: str) -> None:
    _require_migration_api()
    source = tmp_path / "v2-hf-conflict"
    _write_episode_fixture(source, "success")
    original_source_bytes = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    hub, identity = _fake_hub_for_source(source)
    original_remote_source = dict(hub.revisions[identity["source_repo_id"]][identity["source_revision"]])
    _install_fake_hub(monkeypatch, hub)
    from scripts.generation_archive_migrate import main
    args = _cli_args(tmp_path, identity)
    assert main(args) == 0
    commits_before_conflict = len(hub.commit_calls)

    target_revision = hub.branches[identity["target_repo_id"]]["main"]
    remote_files = hub.revisions[identity["target_repo_id"]][target_revision]
    if conflict == "partial":
        remote_files.pop(identity["target_receipt_path"])
    else:
        target_chunk = next(path for path in remote_files if path.startswith(identity["target_artifact_path"] + "/data/"))
        remote_files[target_chunk] = remote_files[target_chunk][:-1] + bytes([remote_files[target_chunk][-1] ^ 0x01])

    assert main(args) == 2
    assert len(hub.commit_calls) == commits_before_conflict
    assert {path: path.read_bytes() for path in source.rglob("*") if path.is_file()} == original_source_bytes
    assert hub.revisions[identity["source_repo_id"]][identity["source_revision"]] == original_remote_source
    assert list((tmp_path / "cli-receipts").glob("*.failure.json"))
