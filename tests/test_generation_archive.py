from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from icgs.data.collection.generation.distributed_contracts import ArchiveProfileConfig
from icgs.data.schemas.episode_records import validate_episode

try:
    from icgs.data.collection.generation.episode_archive import (
        ArchiveManifest,
        EpisodeArchiveReader,
        EpisodeArchiveWriter,
        _ArchiveArrays,
        _array_digest,
        _canonical_json,
        validate_archive_manifest,
    )
except ModuleNotFoundError as error:
    ArchiveManifest = EpisodeArchiveReader = EpisodeArchiveWriter = None
    _ArchiveArrays = None
    validate_archive_manifest = None
    _archive_import_error = str(error)
else:
    _archive_import_error = None


@pytest.fixture(autouse=True)
def _archive_api_ready(request):
    if _archive_import_error is not None and request.node.name != "test_generation_archive_api_is_available":
        pytest.skip("archive implementation is not present yet")


def test_generation_archive_api_is_available():
    assert _archive_import_error is None, "generation archive API is required"


def _episode(outcome: str = "success") -> dict:
    points_a = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    points_b = np.asarray([[6.0, 7.0, 8.0]], dtype=np.float32)
    pose_a = np.eye(4, dtype=np.float64)
    pose_b = np.eye(4, dtype=np.float64)
    pose_b[0, 3] = 0.25
    return {
        "schema_version": "icgs_episode_v2",
        "provenance": {
            "episode_id": "episode-test-00001",
            "attempt_id": "att-episode-test-00001",
            "program_id": "T01",
            "split": "train",
            "scene_seed": 17,
            "asset_instance_id": "asset-test-1",
            "execution_mode": "scripted_waypoint_v1",
            "calibration_id": "calibration-test",
            "observation_origin": "measured",
            "raw_commands_id": "raw-test-v1",
            "materialized_commands_id": "materialized-test-v1",
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
            {"points": points_a, "point_valid": np.asarray([True, True]), "T_w_e": pose_a, "grip": 0},
            {"points": points_b, "point_valid": np.asarray([True]), "T_w_e": pose_b, "grip": 1},
        ],
        "transitions": [
            {
                "command": {"T_w_e": pose_b, "grip": 1, "duration_s": 0.1},
                "achieved_duration_s": 0.08,
                "physics_substeps": 4,
                "before_boundary": 0,
                "after_boundary": 1,
            }
        ],
        "dt": [0.08],
        "robot_states": [
            {"T_w_e": pose_a, "grip": 0, "joint_positions": np.asarray([0.1, 0.2])},
            {"T_w_e": pose_b, "grip": 1, "joint_positions": np.asarray([0.3, 0.4])},
        ],
        "object_states": [{"block": np.asarray([1.0, 2.0, 3.0])}, {"block": np.asarray([2.0, 3.0, 4.0])}],
        "snapshot": {"sim_time": None, "snapshot_fidelity": None},
        "rho": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "rho_valid": np.asarray([[True, False], [True, True]], dtype=bool),
        "events": [{"event_id": "place", "target": "drawer"}],
    }


def _profile(*, chunk_boundaries: int = 1) -> ArchiveProfileConfig:
    return ArchiveProfileConfig(chunk_boundaries=chunk_boundaries, local_artifact_retention="keep")


def _debug(**extra: object) -> dict:
    return {
        "source_run_id": "run-test",
        "code_revision": "a" * 40,
        "preprocessing_identity": "measured_raw_v1",
        **extra,
    }


def test_episode_archive_roundtrips_raw_fields_and_random_boundary_access(tmp_path: Path):
    record = _episode()
    measured_actions = np.asarray([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float64)
    measured_points = np.concatenate([item["points"] for item in record["online_observations"]], axis=0)
    writer = EpisodeArchiveWriter(_profile(chunk_boundaries=1))

    manifest = writer.write_episode(
        record,
        raw_arrays={"measured_actions": measured_actions, "measured_points": measured_points},
        debug_metadata=_debug(stderr=""),
        output_dir=tmp_path / "episode",
    )

    assert isinstance(manifest, ArchiveManifest)
    reader = EpisodeArchiveReader(tmp_path / "episode" / "episode.manifest.json")
    assert tuple(reader.iter_boundaries()) == (0, 1)
    assert reader.observation(1)["points"].dtype == np.float32
    np.testing.assert_array_equal(reader.observation(1)["points"], record["online_observations"][1]["points"])
    np.testing.assert_array_equal(reader.observation(1)["point_valid"], record["online_observations"][1]["point_valid"])
    transition = reader.transition(0)
    assert transition["achieved_duration_s"] == pytest.approx(0.08)
    assert transition["command"]["duration_s"] == pytest.approx(0.1)
    np.testing.assert_array_equal(transition["command"]["T_w_e"], record["transitions"][0]["command"]["T_w_e"])

    restored = reader.to_episode_record()
    validate_episode(restored)
    np.testing.assert_array_equal(restored["online_observations"][0]["points"], record["online_observations"][0]["points"])
    np.testing.assert_array_equal(restored["rho_valid"], record["rho_valid"])
    np.testing.assert_array_equal(reader.raw_arrays["measured_actions"], measured_actions)
    np.testing.assert_array_equal(reader.raw_arrays["measured_points"], measured_points)
    aliases = {item["name"]: item["target"] for item in manifest.as_dict()["array_aliases"]}
    assert aliases["raw_measured_points"] == "online_points"
    measured_alias = next(item for item in manifest.as_dict()["array_aliases"] if item["name"] == "raw_measured_points")
    assert measured_alias["semantic_role"] == "raw_arrays/measured_points"
    assert measured_alias["pieces"] == manifest.as_dict()["array_specs"]["online_points"]["pieces"]
    assert validate_archive_manifest(tmp_path / "episode" / "episode.manifest.json")["valid"] is True


def test_content_identical_task_labels_do_not_alias_online_points(tmp_path: Path):
    record = _episode()
    point_rows = np.asarray([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]], dtype=np.float32)
    for index, row in enumerate(point_rows[:2]):
        record["online_observations"][index]["points"] = row[None, :]
        record["online_observations"][index]["point_valid"] = np.asarray([True])
    record["online_observations"].append({
        "points": point_rows[2:3],
        "point_valid": np.asarray([True]),
        "T_w_e": np.eye(4, dtype=np.float64),
        "grip": 0,
    })
    record["transitions"].append({
        "command": {"T_w_e": np.eye(4, dtype=np.float64), "grip": 0, "duration_s": 0.08},
        "achieved_duration_s": 0.08,
        "physics_substeps": 4,
        "before_boundary": 1,
        "after_boundary": 2,
    })
    record["dt"] = [0.08, 0.08]
    record["robot_states"].append({"T_w_e": np.eye(4, dtype=np.float64), "grip": 0, "joint_positions": np.asarray([0.5, 0.6])})
    record["object_states"].append({"block": np.asarray([3.0, 4.0, 5.0])})
    record["rho"] = point_rows.copy()
    record["rho_valid"] = np.ones((3, 3), dtype=bool)

    manifest = EpisodeArchiveWriter(_profile(chunk_boundaries=1)).write_episode(
        record, raw_arrays={}, debug_metadata=_debug(), output_dir=tmp_path / "episode"
    )

    rho_name = manifest.as_dict()["record_metadata"]["rho"]["$archive_array"]
    aliases = {item["name"]: item for item in manifest.as_dict()["array_aliases"]}
    assert rho_name not in aliases
    assert manifest.as_dict()["array_specs"][rho_name]["semantic_role"] == "rho"


def test_distinct_measured_points_remain_a_separate_array(tmp_path: Path):
    record = _episode()
    measured_points = np.concatenate([item["points"] for item in record["online_observations"]], axis=0).copy()
    measured_points[0, 0] += 0.5
    target = tmp_path / "episode"

    manifest = EpisodeArchiveWriter(_profile()).write_episode(
        record,
        raw_arrays={"measured_points": measured_points},
        debug_metadata=_debug(),
        output_dir=target,
    ).as_dict()

    measured_name = manifest["raw_arrays"]["measured_points"]["$archive_array"]
    assert measured_name in manifest["array_specs"]
    assert all(item["name"] != measured_name for item in manifest["array_aliases"])
    reader = EpisodeArchiveReader(target / "episode.manifest.json")
    assert not np.array_equal(reader.read_array(measured_name), reader.read_array("online_points"))


@pytest.mark.parametrize("outcome", ["success", "valid_failure"])
def test_success_and_valid_failure_have_the_same_required_archive_arrays(tmp_path: Path, outcome: str):
    target = tmp_path / outcome
    manifest = EpisodeArchiveWriter(_profile()).write_episode(
        _episode(outcome), raw_arrays={}, debug_metadata=_debug(outcome=outcome), output_dir=target
    )

    assert manifest.as_dict()["outcome"] == outcome
    archive = manifest.as_dict()
    available_arrays = set(archive["array_specs"]) | {item["name"] for item in archive["array_aliases"]}
    assert {"online_points", "online_points_offsets", "online_validity_packed", "commands", "command_grips", "dt", "substeps"} <= available_arrays
    validity_spec = archive["array_specs"]["online_validity_packed"]
    assert validity_spec["dtype"] == np.dtype(np.uint8).str
    assert validity_spec["shape"][0] < sum(len(item["points"]) for item in _episode(outcome)["online_observations"])
    assert archive["local_write_limits"]["max_chunk_bytes"] == _profile().max_chunk_bytes
    assert archive["local_write_limits"]["spool_peak_bytes"] > 0
    assert archive["local_write_limits"]["local_peak_bytes_upper_bound"] >= archive["local_write_limits"]["spool_peak_bytes"]
    assert (target / "debug.json").is_file()
    assert (target / "artifact_manifest.json").is_file()


def test_attempt_archive_keeps_null_episode_identity_and_measured_prefix(tmp_path: Path):
    attempt = {
        "schema_version": "icgs_episode_v2",
        "attempt_id": "att-token=hf_abcdefghijklmnopqrstuvwxyz0123456789",
        "episode_id": None,
        "program_id": "T01",
        "episode_kind": "nominal",
        "outcome": "simulator_crash",
        "terminal_reason": "simulator_exception",
        "error_type": "RuntimeError",
        "error": "simulator stopped token=hf_abcdefghijklmnopqrstuvwxyz0123456789",
        "traceback": "Traceback: bounded evidence",
        "diagnostics": {"nested": ["Authorization: Bearer abcdef123456", "token=hf_abcdefghijklmnopqrstuvwxyz0123456789"]},
    }

    manifest = EpisodeArchiveWriter(_profile()).write_attempt(
        attempt,
        prefix_arrays={"actions": np.asarray([[1.0, 2.0]], dtype=np.float64)},
        debug_metadata=_debug(
            exit_code=1,
            timeout=False,
            stderr="simulator process failed token=hf_abcdefghijklmnopqrstuvwxyz0123456789",
        ),
        output_dir=tmp_path / "attempt",
    )

    assert manifest.as_dict()["archive_kind"] == "attempt"
    assert manifest.as_dict()["episode_id"] is None
    assert manifest.as_dict()["attempt_id"] == "att-token=[REDACTED]"
    assert manifest.as_dict()["timeline"]["transitions"] == 1
    reader = EpisodeArchiveReader(tmp_path / "attempt" / "attempt.manifest.json")
    assert reader.to_episode_record()["outcome"] == "simulator_crash"
    assert reader.to_episode_record()["error"] == "simulator stopped token=[REDACTED]"
    assert reader.to_episode_record()["diagnostics"]["nested"] == [
        "Authorization: Bearer [REDACTED]",
        "token=[REDACTED]",
    ]
    np.testing.assert_array_equal(reader.raw_arrays["actions"], [[1.0, 2.0]])
    assert reader.debug_metadata["traceback"] == "Traceback: bounded evidence"
    assert reader.debug_metadata["stderr"] == "simulator process failed token=[REDACTED]"
    assert reader.debug_metadata["timeout"] is False
    manifest_text = (tmp_path / "attempt" / "attempt.manifest.json").read_text(encoding="utf-8")
    assert "hf_abcdefghijklmnopqrstuvwxyz0123456789" not in manifest_text
    assert "abcdef123456" not in manifest_text
    artifact_text = (tmp_path / "attempt" / "artifact_manifest.json").read_text(encoding="utf-8")
    assert "hf_abcdefghijklmnopqrstuvwxyz0123456789" not in artifact_text


def test_attempt_closed_inventory_redacts_sensitive_keys_and_prefix_metadata(tmp_path: Path):
    synthetic_secrets = (
        "synthetic-password-credential",
        "synthetic-api-key-credential",
        "synthetic-client-secret-credential",
        "synthetic-prefix-api-key-credential",
        "synthetic-debug-authorization-credential",
    )
    attempt = {
        "schema_version": "icgs_episode_v2",
        "attempt_id": "att-sensitive-fields-00001",
        "episode_id": None,
        "program_id": "T01",
        "outcome": "simulator_crash",
        "password": synthetic_secrets[0],
        "api_key": synthetic_secrets[1],
        "nested": {"client_secret": synthetic_secrets[2]},
    }

    EpisodeArchiveWriter(_profile()).write_attempt(
        attempt,
        prefix_arrays={
            "actions": np.asarray([[1.0, 2.0]], dtype=np.float64),
            "api_key": synthetic_secrets[3],
        },
        debug_metadata=_debug(authorization=synthetic_secrets[4]),
        output_dir=tmp_path / "attempt",
    )

    archived_bytes = b"".join(path.read_bytes() for path in (tmp_path / "attempt").rglob("*") if path.is_file())
    assert all(secret.encode("utf-8") not in archived_bytes for secret in synthetic_secrets)


def test_archive_rejects_object_arrays_before_npz_serialization(tmp_path: Path):
    with pytest.raises(ValueError, match="object arrays are forbidden"):
        EpisodeArchiveWriter(_profile()).write_episode(
            _episode(),
            raw_arrays={"unsafe": np.asarray([{"pickle": True}], dtype=object)},
            debug_metadata=_debug(),
            output_dir=tmp_path / "episode",
        )
    assert not (tmp_path / "episode").exists()
    assert not list(tmp_path.glob(".*.archive-spool-*"))


def test_archive_fails_closed_when_a_chunk_exceeds_its_profile_cap(tmp_path: Path):
    profile = ArchiveProfileConfig(chunk_boundaries=1, max_chunk_bytes=1024)
    raw_image = np.zeros((1, 64, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="max_chunk_bytes"):
        EpisodeArchiveWriter(profile).write_episode(
            _episode(),
            raw_arrays={"front_rgb_frames": raw_image},
            debug_metadata=_debug(),
            output_dir=tmp_path / "episode",
        )

    assert not (tmp_path / "episode").exists()
    assert not list(tmp_path.glob(".*.archive-spool-*"))


def test_writer_counts_npy_headers_toward_uncompressed_chunk_cap(tmp_path: Path):
    scratch = tmp_path / "spool"
    scratch.mkdir()
    arrays = _ArchiveArrays(scratch, max_chunk_bytes=1024)

    with pytest.raises(ValueError, match="max_chunk_bytes"):
        arrays.add_array("near_cap", np.zeros((237,), dtype=np.float32), "test/near_cap")

    assert not list(scratch.rglob("*.npy"))


def test_same_role_arrays_with_different_logical_ranges_are_not_aliased(tmp_path: Path):
    scratch = tmp_path / "spool"
    scratch.mkdir()
    arrays = _ArchiveArrays(scratch, max_chunk_bytes=4096)
    values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    arrays.add(
        "first",
        values.dtype,
        values.shape,
        "test/same_role",
        [(0, values, {"array_start": 0, "array_stop": 4})],
    )
    arrays.add(
        "second",
        values.dtype,
        values.shape,
        "test/same_role",
        [(1, values, {"timeline_start": 0, "timeline_stop": 4})],
    )

    assert not arrays.aliases
    assert set(arrays.specs) == {"first", "second"}


def test_attempt_prefix_arrays_are_chunked_against_inferred_timeline_counts(tmp_path: Path):
    attempt = {
        "schema_version": "icgs_episode_v2",
        "attempt_id": "att-long-prefix-00001",
        "episode_id": None,
        "program_id": "T01",
        "outcome": "simulator_crash",
        "valid_observation_until": 5,
    }
    prefix_arrays = {
        "actions": np.arange(10, dtype=np.float64).reshape(5, 2),
        "T_w_e": np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 6, axis=0),
    }

    manifest = EpisodeArchiveWriter(_profile(chunk_boundaries=2)).write_attempt(
        attempt,
        prefix_arrays=prefix_arrays,
        debug_metadata=_debug(),
        output_dir=tmp_path / "attempt",
    ).as_dict()

    assert manifest["timeline"]["observations"] == 6
    assert manifest["timeline"]["transitions"] == 5
    for key, expected_stops in (("actions", [2, 4, 5]), ("T_w_e", [2, 4, 6])):
        logical_name = manifest["raw_arrays"][key]["$archive_array"]
        pieces = manifest["array_specs"][logical_name]["pieces"]
        assert len(pieces) == 3
        assert [piece["timeline_stop"] for piece in pieces] == expected_stops


def test_archive_manifest_rejects_schema_identity_mismatch(tmp_path: Path):
    manifest = EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=tmp_path / "episode"
    ).as_dict()
    manifest["episode_schema_version"] = "icgs_episode_v2"

    with pytest.raises(ValueError, match="episode_schema_version"):
        ArchiveManifest.from_dict(manifest)


def test_archive_manifest_rejects_path_traversal_and_missing_alias_target(tmp_path: Path):
    target = tmp_path / "episode"
    measured_points = np.concatenate([item["points"] for item in _episode()["online_observations"]], axis=0)
    manifest = EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={"measured_points": measured_points}, debug_metadata=_debug(), output_dir=target
    ).as_dict()

    traversing = json.loads(json.dumps(manifest))
    traversing["chunk_inventory"][0]["path"] = "../chunk.npz"
    with pytest.raises(ValueError, match="contained relative path"):
        ArchiveManifest.from_dict(traversing)

    missing_target = json.loads(json.dumps(manifest))
    missing_target["array_aliases"][0]["target"] = "missing_array"
    with pytest.raises(ValueError, match="alias target is missing"):
        ArchiveManifest.from_dict(missing_target)

    incompatible_role = json.loads(json.dumps(manifest))
    measured_alias = next(item for item in incompatible_role["array_aliases"] if item["name"] == "raw_measured_points")
    measured_alias["semantic_role"] = "rho"
    with pytest.raises(ValueError, match="incompatible semantic roles"):
        ArchiveManifest.from_dict(incompatible_role)

    changed_piece_map = json.loads(json.dumps(manifest))
    measured_alias = next(item for item in changed_piece_map["array_aliases"] if item["name"] == "raw_measured_points")
    measured_alias["pieces"][0]["point_start"] += 1
    with pytest.raises(ValueError, match="logical piece mapping"):
        ArchiveManifest.from_dict(changed_piece_map)


def _rewrite_manifest_only(root: Path, mutate, *, manifest_name: str = "episode.manifest.json") -> None:
    manifest_path = root / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_bytes = _canonical_json(manifest)
    manifest_path.write_bytes(manifest_bytes)
    artifact_path = root / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["files"][manifest_name] = {
        "bytes": len(manifest_bytes),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    artifact_path.write_bytes(_canonical_json(artifact))


def _remove_archive_array(root: Path, name: str) -> None:
    manifest_path = root / "episode.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest["array_specs"].pop(name)
    keys_by_path = {}
    for piece in spec["pieces"]:
        keys_by_path.setdefault(piece["path"], set()).add(piece["key"])
    for relative, removed_keys in keys_by_path.items():
        chunk_path = root / relative
        with np.load(chunk_path, allow_pickle=False) as loaded:
            arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files if key not in removed_keys}
        np.savez_compressed(chunk_path, **arrays)
        inventory = next(item for item in manifest["chunk_inventory"] if item["path"] == relative)
        inventory["bytes"] = chunk_path.stat().st_size
        inventory["sha256"] = hashlib.sha256(chunk_path.read_bytes()).hexdigest()
    manifest_bytes = _canonical_json(manifest)
    manifest_path.write_bytes(manifest_bytes)
    artifact_path = root / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    for relative, removed_keys in keys_by_path.items():
        chunk_entry = next(item for item in manifest["chunk_inventory"] if item["path"] == relative)
        artifact["files"][relative] = {"bytes": chunk_entry["bytes"], "sha256": chunk_entry["sha256"]}
    artifact["files"][manifest_path.name] = {
        "bytes": len(manifest_bytes),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    artifact_path.write_bytes(_canonical_json(artifact))


def test_archive_validation_rejects_piece_range_gap_after_hashes_are_repaired(tmp_path: Path):
    target = tmp_path / "episode"
    EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    )

    def introduce_gap(manifest):
        piece = manifest["array_specs"]["commands"]["pieces"][0]
        piece["transition_start"] = 1

    _rewrite_manifest_only(target, introduce_gap)

    with pytest.raises(ValueError, match="range|coverage|transition"):
        validate_archive_manifest(target / "episode.manifest.json")


def test_archive_validation_rejects_point_validity_count_mismatch_after_hashes_are_repaired(tmp_path: Path):
    target = tmp_path / "episode"
    EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    )

    def misalign_validity(manifest):
        manifest["array_specs"]["online_validity_packed"]["pieces"][0]["validity_point_count"] -= 1

    _rewrite_manifest_only(target, misalign_validity)

    with pytest.raises(ValueError, match="point ranges|validity point count|align"):
        validate_archive_manifest(target / "episode.manifest.json")


@pytest.mark.parametrize("name", ["commands", "online_poses"])
def test_archive_validation_rejects_missing_required_timeline_array(tmp_path: Path, name: str):
    target = tmp_path / "episode"
    EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    )
    _remove_archive_array(target, name)

    with pytest.raises(ValueError, match="missing required arrays"):
        validate_archive_manifest(target / "episode.manifest.json")


def test_archive_validation_rejects_dangling_array_reference_after_hashes_are_repaired(tmp_path: Path):
    target = tmp_path / "episode"
    EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    )

    def remove_label_target(manifest):
        manifest["record_metadata"]["rho"]["$archive_array"] = "missing_label_array"

    _rewrite_manifest_only(target, remove_label_target)

    with pytest.raises(ValueError, match="array reference|archive array"):
        validate_archive_manifest(target / "episode.manifest.json")


@pytest.mark.parametrize("field_path", ["label", "state"])
def test_archive_validation_rejects_cross_role_label_or_state_reference_after_hashes_are_repaired(
    tmp_path: Path, field_path: str
):
    target = tmp_path / "episode"
    EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    )

    def cross_wire_reference(manifest):
        duration_reference = manifest["record_metadata"]["dt"]["$archive_array"]
        if field_path == "label":
            manifest["record_metadata"]["rho"]["$archive_array"] = duration_reference
        else:
            manifest["record_metadata"]["object_states"][0]["block"]["$archive_array"] = duration_reference

    _rewrite_manifest_only(target, cross_wire_reference)

    with pytest.raises(ValueError, match="semantic role mismatch"):
        validate_archive_manifest(target / "episode.manifest.json")


def test_archive_validation_rejects_state_boundary_cardinality_mismatch(tmp_path: Path):
    target = tmp_path / "episode"
    EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    )

    _rewrite_manifest_only(target, lambda manifest: manifest["record_metadata"]["object_states"].pop())

    with pytest.raises(ValueError, match="object_states must align"):
        validate_archive_manifest(target / "episode.manifest.json")


def test_attempt_archive_roundtrips_invalid_observation_with_empty_prefix(tmp_path: Path):
    attempt = {
        "schema_version": "icgs_episode_v2",
        "attempt_id": "att-invalid-observation-00001",
        "episode_id": None,
        "program_id": "T01",
        "outcome": "invalid_observation",
        "valid_observation_until": None,
    }

    manifest = EpisodeArchiveWriter(_profile()).write_attempt(
        attempt,
        prefix_arrays={},
        debug_metadata=_debug(error="observation did not contain valid points"),
        output_dir=tmp_path / "attempt",
    ).as_dict()

    assert manifest["timeline"]["observations"] == 0
    assert manifest["timeline"]["transitions"] == 0
    assert EpisodeArchiveReader(tmp_path / "attempt" / "attempt.manifest.json").to_episode_record()["outcome"] == "invalid_observation"


def test_attempt_with_valid_observation_boundary_requires_measured_prefix_arrays(tmp_path: Path):
    attempt = {
        "schema_version": "icgs_episode_v2",
        "attempt_id": "att-missing-prefix-00001",
        "episode_id": None,
        "program_id": "T01",
        "outcome": "simulator_crash",
        "valid_observation_until": 0,
    }

    with pytest.raises(ValueError, match="requires archived measured prefix arrays"):
        EpisodeArchiveWriter(_profile()).write_attempt(
            attempt,
            prefix_arrays={},
            debug_metadata=_debug(),
            output_dir=tmp_path / "attempt",
        )


@pytest.mark.parametrize(
    "prefix_arrays",
    [
        {"points": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)},
        {"point_offsets": np.asarray([0, 1], dtype=np.int64)},
    ],
)
def test_attempt_ragged_point_values_and_offsets_must_be_archived_together(tmp_path: Path, prefix_arrays):
    attempt = {
        "schema_version": "icgs_episode_v2",
        "attempt_id": "att-incomplete-point-prefix-00001",
        "episode_id": None,
        "program_id": "T01",
        "outcome": "invalid_observation",
        "valid_observation_until": 0,
    }

    with pytest.raises(ValueError, match="points and point_offsets must be provided together"):
        EpisodeArchiveWriter(_profile()).write_attempt(
            attempt,
            prefix_arrays=prefix_arrays,
            debug_metadata=_debug(),
            output_dir=tmp_path / "attempt",
        )


def test_attempt_validation_rejects_prefix_count_mismatch_after_hashes_are_repaired(tmp_path: Path):
    target = tmp_path / "attempt"
    EpisodeArchiveWriter(_profile()).write_attempt(
        {
            "schema_version": "icgs_episode_v2",
            "attempt_id": "att-prefix-mismatch-00001",
            "episode_id": None,
            "program_id": "T01",
            "outcome": "simulator_crash",
        },
        prefix_arrays={"actions": np.asarray([[1.0, 2.0]], dtype=np.float64)},
        debug_metadata=_debug(),
        output_dir=target,
    )

    _rewrite_manifest_only(target, lambda payload: payload["timeline"].update(transitions=0), manifest_name="attempt.manifest.json")

    with pytest.raises(ValueError, match="does not align"):
        validate_archive_manifest(target / "attempt.manifest.json")


def _attempt_with_measured_prefix(target: Path) -> None:
    EpisodeArchiveWriter(_profile()).write_attempt(
        {
            "schema_version": "icgs_episode_v2",
            "attempt_id": "att-measured-prefix-00001",
            "episode_id": None,
            "program_id": "T01",
            "outcome": "simulator_crash",
            "valid_observation_until": 0,
        },
        prefix_arrays={"T_w_e": np.eye(4, dtype=np.float64)[None, :, :]},
        debug_metadata=_debug(),
        output_dir=target,
    )


def test_attempt_validation_rejects_rehashed_manifest_without_measured_prefix_reference(tmp_path: Path):
    target = tmp_path / "attempt"
    _attempt_with_measured_prefix(target)
    _rewrite_manifest_only(
        target,
        lambda payload: payload["raw_arrays"].pop("T_w_e"),
        manifest_name="attempt.manifest.json",
    )

    with pytest.raises(ValueError, match="requires a measured-prefix reference"):
        validate_archive_manifest(target / "attempt.manifest.json")


def test_attempt_validation_rejects_source_timeline_valid_until_conflict_after_rehash(tmp_path: Path):
    target = tmp_path / "attempt"
    _attempt_with_measured_prefix(target)
    _rewrite_manifest_only(
        target,
        lambda payload: payload["record_metadata"]["attempt"].update(valid_observation_until=1),
        manifest_name="attempt.manifest.json",
    )

    with pytest.raises(ValueError, match="valid_observation_until.*disagrees"):
        validate_archive_manifest(target / "attempt.manifest.json")


def _rewrite_archive_array(root: Path, name: str, mutate) -> None:
    manifest_path = root / "episode.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest["array_specs"][name]
    assert len(spec["pieces"]) == 1
    piece = spec["pieces"][0]
    chunk_path = root / piece["path"]
    with np.load(chunk_path, allow_pickle=False) as loaded:
        arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    arrays[piece["key"]] = mutate(arrays[piece["key"]])
    np.savez_compressed(chunk_path, **arrays)
    changed = arrays[piece["key"]]
    digest = _array_digest(changed.dtype, changed.shape, [changed])
    piece["sha256"] = digest
    piece["byte_count"] = int(changed.nbytes)
    spec["sha256"] = digest
    spec["byte_count"] = int(changed.nbytes)
    chunk_entry = next(item for item in manifest["chunk_inventory"] if item["path"] == piece["path"])
    chunk_entry["bytes"] = chunk_path.stat().st_size
    chunk_entry["sha256"] = __import__("hashlib").sha256(chunk_path.read_bytes()).hexdigest()
    manifest_bytes = _canonical_json(manifest)
    manifest_path.write_bytes(manifest_bytes)
    artifact_path = root / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["files"][piece["path"]] = {"bytes": chunk_entry["bytes"], "sha256": chunk_entry["sha256"]}
    artifact["files"][manifest_path.name] = {
        "bytes": len(manifest_bytes),
        "sha256": __import__("hashlib").sha256(manifest_bytes).hexdigest(),
    }
    artifact_path.write_bytes(_canonical_json(artifact))


def test_archive_validation_rejects_nonmonotonic_offsets_after_hashes_are_repaired(tmp_path: Path):
    target = tmp_path / "episode"
    EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    )

    _rewrite_archive_array(target, "online_points_offsets", lambda _value: np.asarray([0, 4, 3], dtype=np.int64))

    with pytest.raises(ValueError, match="monotonic"):
        validate_archive_manifest(target / "episode.manifest.json")


def test_archive_validation_does_not_materialize_all_online_points(tmp_path: Path, monkeypatch):
    target = tmp_path / "episode"
    EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    )
    original_read_array = EpisodeArchiveReader.read_array

    def guarded_read_array(reader, name: str):
        if name == "online_points":
            pytest.fail("archive validation must inspect online points chunk by chunk")
        return original_read_array(reader, name)

    monkeypatch.setattr(EpisodeArchiveReader, "read_array", guarded_read_array)

    assert validate_archive_manifest(target / "episode.manifest.json")["valid"] is True


def test_archive_validation_rejects_chunk_corruption(tmp_path: Path):
    target = tmp_path / "episode"
    manifest = EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    ).as_dict()
    chunk_path = target / manifest["chunk_inventory"][0]["path"]
    with chunk_path.open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(ValueError, match="artifact manifest checksum mismatch"):
        validate_archive_manifest(target / "episode.manifest.json")


def test_archive_validation_rejects_a_truncated_npz_chunk(tmp_path: Path):
    target = tmp_path / "episode"
    manifest = EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    ).as_dict()
    chunk_path = target / manifest["chunk_inventory"][0]["path"]
    chunk_path.write_bytes(b"PK\x03\x04partial")

    with pytest.raises(ValueError, match="artifact manifest checksum mismatch"):
        validate_archive_manifest(target / "episode.manifest.json")


def test_archive_validation_rejects_symlinked_chunk_paths(tmp_path: Path):
    target = tmp_path / "episode"
    manifest = EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    ).as_dict()
    chunk_path = target / manifest["chunk_inventory"][0]["path"]
    chunk_path.unlink()
    chunk_path.symlink_to(tmp_path / "outside.npz")

    with pytest.raises(ValueError, match="symlinks"):
        validate_archive_manifest(target / "episode.manifest.json")


def test_archive_reader_loads_only_requested_chunks_and_keeps_lru_bounded(tmp_path: Path):
    target = tmp_path / "episode"
    manifest = EpisodeArchiveWriter(_profile(chunk_boundaries=1)).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    ).as_dict()
    chunk_sizes = {}
    for spec in manifest["array_specs"].values():
        for piece in spec["pieces"]:
            chunk_sizes[piece["path"]] = chunk_sizes.get(piece["path"], 0) + piece["byte_count"]
    cache_cap = max(chunk_sizes.values())
    reader = EpisodeArchiveReader(target / "episode.manifest.json", cache_bytes=cache_cap)
    loaded = []
    original_load_chunk = reader._load_chunk

    def tracking_load_chunk(relative: str):
        loaded.append(relative)
        return original_load_chunk(relative)

    reader._load_chunk = tracking_load_chunk
    reader.observation(1)

    assert set(loaded) == set(chunk_sizes)
    assert reader._cache_size <= cache_cap
    assert len(reader._cache) <= 1


def test_archive_reader_cache_keys_bind_archive_and_resolved_preprocessing_identity(tmp_path: Path):
    target = tmp_path / "episode"
    EpisodeArchiveWriter(_profile(chunk_boundaries=1)).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    )
    manifest_path = target / "episode.manifest.json"
    preprocessing_hash = "b" * 64
    first = EpisodeArchiveReader(
        manifest_path,
        resolved_preprocessing_identity="icgs_native_2048_v1",
        resolved_preprocessing_sha256=preprocessing_hash,
    )
    same = EpisodeArchiveReader(
        manifest_path,
        resolved_preprocessing_identity="icgs_native_2048_v1",
        resolved_preprocessing_sha256=preprocessing_hash,
    )
    different = EpisodeArchiveReader(
        manifest_path,
        resolved_preprocessing_identity="icgs_native_2048_v2",
        resolved_preprocessing_sha256="c" * 64,
    )

    assert first.cache_identity == same.cache_identity
    assert first.cache_identity != different.cache_identity
    first.observation(0)
    assert first._cache
    assert all(key[0] == first.cache_identity for key in first._cache)


@pytest.mark.parametrize("consumer", ["reader", "validator"])
def test_npz_uncompressed_cap_is_checked_before_loading_members(tmp_path: Path, monkeypatch, consumer: str):
    target = tmp_path / "episode"
    manifest = EpisodeArchiveWriter(_profile()).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    ).as_dict()

    def lower_cap(payload):
        payload["archive_profile"]["max_chunk_bytes"] = 1024
        payload["local_write_limits"]["max_chunk_bytes"] = 1024

    _rewrite_manifest_only(target, lower_cap)
    manifest_path = target / "episode.manifest.json"
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("ZIP member was decompressed before cap check"))

    with pytest.raises(ValueError, match="max_chunk_bytes"):
        if consumer == "validator":
            validate_archive_manifest(manifest_path)
        else:
            EpisodeArchiveReader(manifest_path)._load_chunk(manifest["chunk_inventory"][0]["path"])


def test_task_labels_read_one_boundary_without_materializing_the_full_episode(tmp_path: Path, monkeypatch):
    target = tmp_path / "episode"
    EpisodeArchiveWriter(_profile(chunk_boundaries=1)).write_episode(
        _episode(), raw_arrays={}, debug_metadata=_debug(), output_dir=target
    )
    reader = EpisodeArchiveReader(target / "episode.manifest.json")
    monkeypatch.setattr(
        reader,
        "to_episode_record",
        lambda: pytest.fail("task_labels must not materialize the full episode"),
    )

    labels = reader.task_labels(1)

    np.testing.assert_array_equal(labels["rho"], _episode()["rho"][1])
    np.testing.assert_array_equal(labels["rho_valid"], [True, True])


def test_debug_json_contains_array_references_instead_of_dense_numeric_lists(tmp_path: Path):
    target = tmp_path / "episode"
    EpisodeArchiveWriter(_profile()).write_episode(
        _episode(),
        raw_arrays={},
        debug_metadata=_debug(
            predicate_distances=np.asarray([0.2, 0.3], dtype=np.float64),
            scalar_array=np.asarray(7.0, dtype=np.float32),
        ),
        output_dir=target,
    )

    payload = json.loads((target / "debug.json").read_text(encoding="utf-8"))
    assert payload["predicate_distances"]["$archive_array"]
    assert payload["scalar_array"]["$archive_array"]
    assert "0.2" not in json.dumps(payload)
