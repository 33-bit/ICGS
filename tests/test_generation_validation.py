from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from icgs.data.collection.generation.batch import AttemptPlan
from icgs.data.collection.generation.distributed_contracts import ArchiveProfileConfig, GenerationJob
from icgs.data.collection.generation.distributed_contracts import (
    ValidationGateResult,
    ValidationPlan,
    ValidationReceipt,
)
from icgs.data.collection.generation.distributed_validation import (
    ingest_validated_result,
    load_validation_plan,
    validate_validation_receipt,
    validate_closed_result,
)
from icgs.data.collection.generation.episode_archive import EpisodeArchiveReader, EpisodeArchiveWriter
from icgs.data.collection.generation import distributed_validation as validation
from icgs.data.collection.generation.rlbench_attempt import RawAttempt, materialize_raw_attempt, write_closed_attempt_result


@dataclass
class Obs:
    wrist_point_cloud: np.ndarray
    gripper_pose: np.ndarray
    gripper_open: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    front_rgb: np.ndarray
    wrist_depth: np.ndarray
    wrist_mask: np.ndarray
    front_mask: np.ndarray


def _job(index: int = 1, split: str = "train") -> GenerationJob:
    plan = AttemptPlan(
        program_id="T01", split=split, episode_index=index,
        episode_id=f"episode-t01-{index:05d}", episode_kind="nominal",
        scene_seed=100 + index, collection_seed=20260920,
        randomization={"scene_signature": f"sig-{index}", "asset_instance_id": f"asset-{index}"},
        intervention=None,
    )
    return GenerationJob.create(
        job_id=f"job-{index}", run_id="run-1", attempt_id=f"att-{plan.episode_id}",
        episode_id=plan.episode_id, program_id="T01", plan=plan,
        code_revision="a" * 40, manifest_sha256="b" * 64,
        output_root="/content/run/staging",
    )


def _binding() -> dict:
    return {
        "program_id": "T01", "split": "train", "asset_family_id": "family-1",
        "source_lineage_id": "lineage-1", "asset_version": "v1",
        "randomization": {
            "translation_m": {"x": [-0.012, 0.012], "y": [-0.012, 0.012], "z": [0.0, 0.0]},
            "yaw_deg": [-30.0, 30.0], "scale": [0.8, 1.2],
            "camera_profile_id": "rlbench-wrist-depth-v1",
        },
    }


def _raw(success: bool) -> RawAttempt:
    observations = []
    for index in range(3):
        observations.append(Obs(
            wrist_point_cloud=np.asarray([[0.1 + index, 0.0, 0.8]], dtype=np.float32),
            gripper_pose=np.asarray([0.1 + index * 0.01, 0, 0.8, 0, 0, 0, 1], dtype=np.float64),
            gripper_open=1.0, joint_positions=np.zeros(7), joint_velocities=np.zeros(7),
            front_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            wrist_depth=np.zeros((2, 2), dtype=np.float32),
            wrist_mask=np.zeros((2, 2), dtype=np.uint8),
            front_mask=np.zeros((2, 2), dtype=np.uint8),
        ))
    actions = tuple(np.asarray([0.1, 0, 0.8, 0, 0, 0, 1, 1], dtype=np.float64) for _ in range(2))
    return RawAttempt(
        observations=tuple(observations), actions=actions,
        scene_states=({}, {}, {}), collision_events=(), sim_time_s=0.1,
        predicates_ok=success,
        terminal_reason="predicate_satisfied" if success else "predicate_failed",
    )


def _closed(tmp_path: Path, outcome: str, index: int = 1):
    job = _job(index)
    materialized = materialize_raw_attempt(_raw(outcome == "success"), job, _binding())
    result = write_closed_attempt_result(materialized, job, tmp_path / f"result-{index}")
    return job, result


def _archive_closed(
    tmp_path: Path,
    monkeypatch,
    outcome: str,
    index: int = 1,
    *,
    randomization_overrides: dict | None = None,
):
    profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention="keep")
    monkeypatch.setenv("ICGS_GENERATION_ARCHIVE_PROFILE", json.dumps(profile.as_dict()))
    job = _job(index)
    if randomization_overrides:
        plan = replace(
            job.plan,
            randomization={**job.plan.randomization, **randomization_overrides},
        )
        job = replace(job, plan=plan)
    if outcome == "success":
        raw = _raw(True)
    elif outcome == "valid_failure":
        raw = _raw(False)
    elif outcome == "invalid_observation":
        source = _raw(False)
        raw = RawAttempt(
            observations=source.observations[:1], actions=(), scene_states=(),
            collision_events=(), sim_time_s=0.0, predicates_ok=False,
            terminal_reason="observation incomplete",
        )
    else:
        source = _raw(False)
        raw = replace(source, simulator_crash=True)
    materialized = materialize_raw_attempt(raw, job, _binding())
    result = write_closed_attempt_result(materialized, job, tmp_path / f"archive-{outcome}-{index}")
    return job, result, profile


def _archive_file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def _write_canonical_json(path: Path, payload: dict) -> bytes:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    return encoded


def _repair_archive_debug_hashes(root: Path, manifest_name: str) -> dict[str, str]:
    manifest_path = root / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    debug_path = root / manifest["debug_path"]
    debug_bytes = debug_path.read_bytes()
    manifest["debug_bytes"] = len(debug_bytes)
    manifest["debug_sha256"] = hashlib.sha256(debug_bytes).hexdigest()
    manifest_bytes = _write_canonical_json(manifest_path, manifest)

    artifact_path = root / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    for relative, content in (
        (manifest["debug_path"], debug_bytes),
        (manifest_name, manifest_bytes),
    ):
        artifact["files"][relative].update({
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    _write_canonical_json(artifact_path, artifact)
    return _archive_file_hashes(root)


@pytest.mark.parametrize("outcome", ["success", "valid_failure"])
def test_episode_outcomes_require_complete_timeline_and_hashes(tmp_path: Path, outcome: str):
    job, result = _closed(tmp_path, outcome)
    validated = validate_closed_result(job, result)
    assert validated.outcome == outcome
    assert validated.episode_entry is not None
    assert validated.episode_entry["outcome"] == outcome
    assert validated.episode_entry["attempt_plan"] == job.plan.as_dict()


@pytest.mark.parametrize("outcome", ["success", "valid_failure"])
def test_archive_profile_validates_episode_and_emits_durable_hf_reference(
    tmp_path: Path, monkeypatch, outcome: str
):
    job, result, profile = _archive_closed(tmp_path, monkeypatch, outcome)

    validated = validate_closed_result(job, result, archive_profile=profile)

    assert validated.episode_entry is not None
    entry = validated.episode_entry
    assert "result_dir" not in entry
    assert entry["archive_ref"] == f"episodes/{job.program_id}/{job.episode_id}"
    assert entry["archive_manifest"] == (
        f"episodes/{job.program_id}/{job.episode_id}/episode.manifest.json"
    )
    assert entry["file_sha256"] == result.file_sha256
    assert entry["archive_format_id"] == profile.archive_format_id
    assert entry["episode_schema_version"] == profile.episode_schema_version
    assert entry["dataset_identity"] == profile.dataset_identity
    assert entry["job_id"] == job.job_id
    assert entry["run_id"] == job.run_id
    assert entry["manifest_sha256"] == job.manifest_sha256
    assert entry["source_run_id"] == job.run_id
    assert entry["code_revision"] == job.code_revision
    assert entry["preprocessing_identity"] == "rlbench_measured_v1"
    assert set(entry["file_sha256"]) == {
        str(path.relative_to(result.result_dir))
        for path in Path(result.result_dir).rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("outcome", ["simulator_crash", "invalid_observation"])
def test_archive_profile_validates_attempts_without_episode_training_entry(
    tmp_path: Path, monkeypatch, outcome: str
):
    job, result, profile = _archive_closed(tmp_path, monkeypatch, outcome)

    validated = validate_closed_result(job, result, archive_profile=profile)

    assert validated.episode_entry is None
    assert validated.attempt_entry is not None
    assert validated.attempt_entry["outcome"] == outcome
    assert "result_dir" not in validated.attempt_entry
    assert validated.attempt_entry["archive_ref"] == (
        f"attempts/{job.program_id}/{job.attempt_id}"
    )
    assert validated.attempt_entry["file_sha256"] == result.file_sha256


def test_archive_attempt_manifest_row_excludes_local_paths_and_unprojected_arrays(
    tmp_path: Path, monkeypatch
):
    job, result, profile = _archive_closed(tmp_path, monkeypatch, "simulator_crash")
    original = EpisodeArchiveReader(Path(result.result_dir) / "attempt.manifest.json")
    attempt = original.to_episode_record()
    attempt["result_dir"] = "/private/worker/staging/result"
    attempt["untrusted_numeric"] = np.asarray([1, 2, 3], dtype=np.int32)
    attempt["untrusted_nested"] = {"local_path": "/private/worker/cache"}
    target = tmp_path / "attempt-with-extra-metadata"
    EpisodeArchiveWriter(profile).write_attempt(
        attempt,
        prefix_arrays=original.raw_arrays,
        debug_metadata=dict(original.debug_metadata),
        output_dir=target,
    )
    hashes = _archive_file_hashes(target)

    validated = validate_closed_result(
        job,
        replace(result, result_dir=str(target), file_sha256=hashes),
        archive_profile=profile,
    )

    row = validated.attempt_entry
    assert row is not None
    assert row["episode_id"] is None
    assert row["episode_kind"] == job.plan.episode_kind
    assert "result_dir" not in row
    assert "untrusted_numeric" not in row
    assert "untrusted_nested" not in row
    assert "/private/worker/" not in json.dumps(row, allow_nan=False)


def test_archive_profile_rejects_legacy_dense_json_result(tmp_path: Path, monkeypatch):
    profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention="keep")
    job, result = _closed(tmp_path, "success")

    with pytest.raises(ValueError, match="legacy dense JSON|archive manifest"):
        validate_closed_result(job, result, archive_profile=profile)


def test_archive_profile_rejects_chunk_inventory_corruption(tmp_path: Path, monkeypatch):
    job, result, profile = _archive_closed(tmp_path, monkeypatch, "success")
    chunk = next(
        path for path in Path(result.result_dir).rglob("data/chunk-*.npz")
    )
    chunk.write_bytes(chunk.read_bytes() + b"corruption")
    hashes = dict(result.file_sha256)
    hashes[str(chunk.relative_to(result.result_dir))] = hashlib.sha256(chunk.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="chunk checksum|artifact manifest"):
        validate_closed_result(job, replace(result, file_sha256=hashes), archive_profile=profile)


def test_archive_profile_rejects_array_corruption_after_file_hashes_are_repaired(
    tmp_path: Path, monkeypatch
):
    job, result, profile = _archive_closed(tmp_path, monkeypatch, "success")
    root = Path(result.result_dir)
    manifest_path = root / "episode.manifest.json"
    artifact_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    chunk_inventory = manifest["chunk_inventory"][0]
    chunk_path = root / chunk_inventory["path"]
    with np.load(chunk_path, allow_pickle=False) as loaded:
        chunk_arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    mutable_name = next(
        name for name, value in chunk_arrays.items()
        if value.size and value.dtype.kind in "fiu"
    )
    changed = chunk_arrays[mutable_name]
    changed.flat[0] = changed.flat[0] + 1
    np.savez_compressed(chunk_path, **chunk_arrays)

    def canonical_json(path: Path, payload: dict) -> bytes:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        path.write_text(encoded, encoding="utf-8")
        return encoded.encode("utf-8")

    changed_chunk = chunk_path.read_bytes()
    chunk_digest = hashlib.sha256(changed_chunk).hexdigest()
    chunk_inventory.update({"bytes": len(changed_chunk), "sha256": chunk_digest})
    artifact["files"][chunk_inventory["path"]].update({
        "bytes": len(changed_chunk), "sha256": chunk_digest,
    })
    manifest_bytes = canonical_json(manifest_path, manifest)
    artifact["files"]["episode.manifest.json"].update({
        "bytes": len(manifest_bytes), "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    })
    canonical_json(artifact_path, artifact)
    repaired_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }

    with pytest.raises(ValueError, match="array piece checksum mismatch"):
        validate_closed_result(
            job,
            replace(result, file_sha256=repaired_hashes),
            archive_profile=profile,
        )


def test_archive_profile_ingestion_is_immutable_and_idempotent(tmp_path: Path, monkeypatch):
    job, result, profile = _archive_closed(tmp_path, monkeypatch, "valid_failure")
    validated = validate_closed_result(job, result, archive_profile=profile)
    original = {"manifest_version": 3, "episodes": [], "failure_attempts": []}

    updated = ingest_validated_result(original, validated)
    assert "result_dir" not in updated["episodes"][0]
    assert updated["episodes"][0]["archive_ref"].startswith("episodes/")
    assert ingest_validated_result(updated, validated) == updated

    remounted = tmp_path / "different-local-mount" / "archive-valid_failure-1"
    import shutil
    shutil.copytree(result.result_dir, remounted)
    remounted_result = replace(result, result_dir=str(remounted))
    remounted_validated = validate_closed_result(job, remounted_result, archive_profile=profile)
    assert remounted_validated.episode_entry == validated.episode_entry
    assert ingest_validated_result(updated, remounted_validated) == updated


def test_archive_profile_binds_plan_and_result_timeline_to_job(tmp_path: Path, monkeypatch):
    job, result, profile = _archive_closed(tmp_path, monkeypatch, "success")
    conflicting_plan = replace(job.plan, scene_seed=job.plan.scene_seed + 1)
    conflicting_job = replace(job, plan=conflicting_plan)

    with pytest.raises(ValueError, match="attempt plan|job identity|provenance"):
        validate_closed_result(conflicting_job, result, archive_profile=profile)
    conflicting_manifest_job = replace(job, manifest_sha256="c" * 64)
    with pytest.raises(ValueError, match="manifest_sha256"):
        validate_closed_result(conflicting_manifest_job, result, archive_profile=profile)
    with pytest.raises(ValueError, match="timeline"):
        validate_closed_result(
            job,
            replace(result, timeline={"actions": 1, "observations": 2, "durations": 1}),
            archive_profile=profile,
        )


def test_archive_profile_rejects_provenance_conflict_after_repairing_file_inventory(
    tmp_path: Path, monkeypatch
):
    job, result, profile = _archive_closed(tmp_path, monkeypatch, "success")
    original = EpisodeArchiveReader(Path(result.result_dir) / "episode.manifest.json")
    record = original.to_episode_record()
    record["provenance"]["scene_seed"] += 1
    target = tmp_path / "conflicting-provenance"
    EpisodeArchiveWriter(profile).write_episode(
        record,
        raw_arrays=original.raw_arrays,
        debug_metadata=dict(original.debug_metadata),
        output_dir=target,
    )
    hashes = {
        str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in target.rglob("*") if path.is_file()
    }

    with pytest.raises(ValueError, match="archive provenance mismatch for scene_seed"):
        validate_closed_result(
            job,
            replace(result, result_dir=str(target), file_sha256=hashes),
            archive_profile=profile,
        )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("asset_instance_id", "asset-other"),
        ("subset", "train_val"),
        ("train_subset", "train_val"),
    ],
)
def test_archive_profile_binds_asset_and_subset_provenance_to_plan(
    tmp_path: Path,
    monkeypatch,
    field: str,
    wrong_value: str,
):
    job, result, profile = _archive_closed(
        tmp_path,
        monkeypatch,
        "success",
        randomization_overrides={"train_subset": "train_core"},
    )
    original = EpisodeArchiveReader(Path(result.result_dir) / "episode.manifest.json")
    record = original.to_episode_record()
    record["provenance"][field] = wrong_value
    target = tmp_path / f"conflicting-{field}"
    EpisodeArchiveWriter(profile).write_episode(
        record,
        raw_arrays=original.raw_arrays,
        debug_metadata=dict(original.debug_metadata),
        output_dir=target,
    )
    hashes = _archive_file_hashes(target)

    with pytest.raises(ValueError, match=f"archive provenance mismatch for {field}"):
        validate_closed_result(
            job,
            replace(result, result_dir=str(target), file_sha256=hashes),
            archive_profile=profile,
        )


def test_archive_profile_binds_derived_subset_when_plan_omits_subset_fields(
    tmp_path: Path, monkeypatch
):
    job, result, profile = _archive_closed(tmp_path, monkeypatch, "success")
    assert "train_subset" not in job.plan.randomization
    assert "subset" not in job.plan.randomization
    original = EpisodeArchiveReader(Path(result.result_dir) / "episode.manifest.json")
    record = original.to_episode_record()
    correct_subset = record["provenance"]["subset"]
    assert record["provenance"]["train_subset"] == correct_subset
    wrong_subset = "train_val" if correct_subset == "train_core" else "train_core"
    record["provenance"]["subset"] = wrong_subset
    record["provenance"]["train_subset"] = wrong_subset
    target = tmp_path / "conflicting-derived-subset"
    EpisodeArchiveWriter(profile).write_episode(
        record,
        raw_arrays=original.raw_arrays,
        debug_metadata=dict(original.debug_metadata),
        output_dir=target,
    )
    hashes = _archive_file_hashes(target)

    with pytest.raises(ValueError, match="archive provenance mismatch for subset"):
        validate_closed_result(
            job,
            replace(result, result_dir=str(target), file_sha256=hashes),
            archive_profile=profile,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_run_id", "run-tampered", "disagrees with debug metadata"),
        ("code_revision", "c" * 40, "disagrees with debug metadata"),
        ("preprocessing_identity", "other-preprocessing-v1", "disagrees with debug metadata"),
        ("source_run_id", " ", "must be a nonblank string"),
        ("code_revision", "", "must be a nonblank string"),
        ("preprocessing_identity", "", "must be a nonblank string"),
    ],
)
def test_archive_profile_binds_identity_fields_to_debug_metadata_after_hash_repair(
    tmp_path: Path,
    monkeypatch,
    field: str,
    value: str,
    message: str,
):
    job, result, profile = _archive_closed(tmp_path, monkeypatch, "success")
    root = Path(result.result_dir)
    debug_path = root / "debug.json"
    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    debug[field] = value
    _write_canonical_json(debug_path, debug)
    hashes = _repair_archive_debug_hashes(root, "episode.manifest.json")

    with pytest.raises(ValueError, match=f"{field}.*{message}"):
        validate_closed_result(
            job,
            replace(result, file_sha256=hashes),
            archive_profile=profile,
        )


def test_archive_profile_rejects_inconsistent_partial_modality_boundaries(
    tmp_path: Path, monkeypatch
):
    job, result, profile = _archive_closed(tmp_path, monkeypatch, "success")
    original = EpisodeArchiveReader(Path(result.result_dir) / "episode.manifest.json")
    record = original.to_episode_record()
    raw_arrays = dict(original.raw_arrays)
    raw_arrays["front_rgb_frame_boundaries"] = np.asarray([0], dtype=np.int64)
    target = tmp_path / "mismatched-modality-index"
    EpisodeArchiveWriter(profile).write_episode(
        record,
        raw_arrays=raw_arrays,
        debug_metadata=dict(original.debug_metadata),
        output_dir=target,
    )
    hashes = {
        str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in target.rglob("*") if path.is_file()
    }

    with pytest.raises(ValueError, match="front_rgb_frame_boundaries"):
        validate_closed_result(
            job,
            replace(result, result_dir=str(target), file_sha256=hashes),
            archive_profile=profile,
        )


def test_archive_profile_accepts_valid_partial_modality_boundaries(
    tmp_path: Path, monkeypatch
):
    job, result, profile = _archive_closed(tmp_path, monkeypatch, "success")
    original = EpisodeArchiveReader(Path(result.result_dir) / "episode.manifest.json")
    record = original.to_episode_record()
    raw_arrays = dict(original.raw_arrays)
    raw_arrays["front_rgb_frames"] = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    raw_arrays["front_rgb_frame_boundaries"] = np.asarray([0], dtype=np.int64)
    target = tmp_path / "valid-partial-modality-index"
    EpisodeArchiveWriter(profile).write_episode(
        record,
        raw_arrays=raw_arrays,
        debug_metadata=dict(original.debug_metadata),
        output_dir=target,
    )
    hashes = {
        str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in target.rglob("*") if path.is_file()
    }

    validated = validate_closed_result(
        job,
        replace(result, result_dir=str(target), file_sha256=hashes),
        archive_profile=profile,
    )

    assert validated.episode_entry is not None


def test_optional_modality_validation_accepts_archive_array_alias_metadata():
    from types import SimpleNamespace

    payload = {
        "raw_arrays": {
            "front_rgb_frames": {"$archive_array": "front-rgb-alias"},
            "front_rgb_frame_boundaries": {"$archive_array": "boundary-index"},
        },
        "timeline": {"observations": 2},
        "array_specs": {},
        "array_aliases": [{
            "name": "front-rgb-alias",
            "target": "canonical-front-rgb",
            "shape": [1, 2, 2, 3],
        }],
    }

    class Reader:
        manifest = SimpleNamespace(payload=payload)

        def read_array(self, name):
            assert name == "boundary-index"
            return np.asarray([0], dtype=np.int64)

    validation._validate_optional_modality_declarations(Reader())


def test_hash_mismatch_is_rejected_without_mutating_files(tmp_path: Path):
    job, result = _closed(tmp_path, "success")
    episode = Path(result.result_dir) / "episode.json"
    before = episode.read_bytes()
    episode.write_bytes(before + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_closed_result(job, result)
    assert episode.read_bytes() == before + b"\n"


def test_ingest_is_immutable_and_idempotent_for_equal_episode(tmp_path: Path):
    job, result = _closed(tmp_path, "valid_failure")
    validated = validate_closed_result(job, result)
    original = {"manifest_version": 3, "episodes": [], "failure_attempts": []}
    original_bytes = json.dumps(original, sort_keys=True)
    updated = ingest_validated_result(original, validated)
    assert json.dumps(original, sort_keys=True) == original_bytes
    assert updated["episodes"][0]["outcome"] == "valid_failure"
    repeated = ingest_validated_result(updated, validated)
    assert repeated == updated


def test_episode_without_training_layout_is_rejected(tmp_path: Path):
    import hashlib
    import shutil
    from dataclasses import replace

    job, result = _closed(tmp_path, "success")
    shutil.rmtree(Path(result.result_dir) / "layout")
    hashes = {
        str(path.relative_to(result.result_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(result.result_dir).rglob("*") if path.is_file()
    }
    with pytest.raises(ValueError, match="training layout"):
        validate_closed_result(job, replace(result, file_sha256=hashes))


def test_ingest_rejects_conflicting_episode_id_without_mutating_manifest(tmp_path: Path):
    job, result = _closed(tmp_path, "success")
    validated = validate_closed_result(job, result)
    manifest = {
        "manifest_version": 3,
        "episodes": [{"episode_id": job.episode_id, "program_id": "T99", "outcome": "success"}],
        "failure_attempts": [],
    }
    before = json.dumps(manifest, sort_keys=True)
    with pytest.raises(ValueError, match="immutable episode conflict"):
        ingest_validated_result(manifest, validated)
    assert json.dumps(manifest, sort_keys=True) == before


def test_validation_failure_captures_identities_trace_source_and_actual_files(tmp_path: Path):
    job, result = _closed(tmp_path, "success")
    source_path = tmp_path / "queue" / "ready" / job.job_id
    try:
        raise ValueError("malformed closed episode")
    except ValueError as exc:
        assert hasattr(validation, "ValidationFailure"), "ValidationFailure is required"
        job = replace(job, output_root=str(tmp_path))
        failure = validation.ValidationFailure.capture(
            job,
            result,
            exc,
            source_path=source_path,
            allowed_result_root=tmp_path,
        )

    diagnostic = failure.as_dict()
    episode_path = Path(result.result_dir) / "episode.json"
    assert diagnostic["job_identity"]["job_id"] == job.job_id
    assert diagnostic["result_identity"]["attempt_id"] == result.attempt_id
    assert diagnostic["result_identity"]["outcome"] == "success"
    assert diagnostic["exception_type"] == "ValueError"
    assert diagnostic["exception_message"] == "malformed closed episode"
    assert "ValueError: malformed closed episode" in diagnostic["traceback"]
    assert diagnostic["source_path"] == str(source_path)
    assert diagnostic["actual_file_inventory"]["episode.json"]["sha256"] == hashlib.sha256(
        episode_path.read_bytes()
    ).hexdigest()


def test_validation_failure_inventory_does_not_follow_symlinks(tmp_path: Path):
    from dataclasses import replace

    job, result = _closed(tmp_path, "success")
    job = replace(job, output_root=str(tmp_path))
    outside_file = tmp_path / "outside-secret.txt"
    outside_file.write_text("outside", encoding="utf-8")
    link = Path(result.result_dir) / "outside-link"
    try:
        link.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    try:
        raise ValueError("inventory test")
    except ValueError as exc:
        assert hasattr(validation, "ValidationFailure"), "ValidationFailure is required"
        failure = validation.ValidationFailure.capture(
            job,
            result,
            exc,
            source_path=tmp_path / "ready" / job.job_id,
            allowed_result_root=tmp_path,
        )

    assert failure.actual_file_inventory["outside-link"]["kind"] == "symlink"
    assert "bytes" not in failure.actual_file_inventory["outside-link"]


def test_closed_result_rejects_dangling_symlink_artifact(tmp_path: Path):
    job, result = _closed(tmp_path, "success")
    link = Path(result.result_dir) / "dangling-artifact"
    try:
        link.symlink_to(tmp_path / "missing-outside-file")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")

    with pytest.raises(ValueError, match="symlink"):
        validate_closed_result(job, result)


def test_bounded_validation_plan_fixture_parses_with_required_gates():
    plan = load_validation_plan(Path("tests/fixtures/generation_validation_config.json"))

    assert plan.validation_mode is True
    assert plan.worker_count == 2
    assert plan.max_jobs <= 8
    assert plan.max_episodes <= 2
    assert plan.max_attempts <= 4
    assert plan.hf_subfolder.startswith("validation/validation-cpu-20260922/")
    assert plan.gate_names == (
        "success",
        "valid_failure",
        "invalid_observation",
        "infrastructure_failure",
        "malformed_result",
        "coordinator_restart",
        "publication",
    )


def test_validation_plan_rejects_unbounded_or_production_fault_injection(tmp_path: Path):
    payload = json.loads(Path("tests/fixtures/generation_validation_config.json").read_text())
    payload["max_jobs"] = 9
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="max_jobs"):
        load_validation_plan(path)

    payload = json.loads(Path("tests/fixtures/generation_validation_config.json").read_text())
    payload["validation_mode"] = False
    payload["fault_injection"] = {"malformed_result": True}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fault_injection"):
        load_validation_plan(path, validation_mode=False)


def test_validation_receipt_requires_evidence_for_pass_and_reason_for_not_run():
    plan = ValidationPlan.from_dict({
        "plan_id": "bounded",
        "validation_mode": True,
        "worker_count": 2,
        "max_jobs": 2,
        "max_episodes": 1,
        "max_attempts": 2,
        "hf_subfolder": "validation/validation-cpu-20260922/bounded",
        "gates": [
            {"name": name, "required": True}
            for name in (
                "success", "valid_failure", "invalid_observation",
                "infrastructure_failure", "malformed_result",
                "coordinator_restart", "publication",
            )
        ],
        "fault_injection": {},
    })
    receipt = ValidationReceipt(
        plan_id=plan.plan_id,
        status="NOT_RUN",
        gates={
            "success": ValidationGateResult(status="PASS", reason="", evidence={}),
            "valid_failure": ValidationGateResult(status="NOT_RUN", reason="not exercised", evidence={}),
            "invalid_observation": ValidationGateResult(status="NOT_RUN", reason="not exercised", evidence={}),
            "infrastructure_failure": ValidationGateResult(status="NOT_RUN", reason="not exercised", evidence={}),
            "malformed_result": ValidationGateResult(status="NOT_RUN", reason="not exercised", evidence={}),
            "coordinator_restart": ValidationGateResult(status="NOT_RUN", reason="not exercised", evidence={}),
            "publication": ValidationGateResult(status="NOT_RUN", reason="not exercised", evidence={}),
        },
    )
    with pytest.raises(ValueError, match="evidence"):
        validate_validation_receipt(receipt, plan)

    receipt = ValidationReceipt(
        plan_id=plan.plan_id,
        status="NOT_RUN",
        gates={name: ValidationGateResult(status="NOT_RUN", reason="blocked", evidence={}) for name in plan.gate_names},
    )
    assert validate_validation_receipt(receipt, plan).status == "NOT_RUN"
