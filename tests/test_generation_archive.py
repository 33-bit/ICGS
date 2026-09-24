from __future__ import annotations

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
        _array_digest,
        _canonical_json,
        validate_archive_manifest,
    )
except ModuleNotFoundError as error:
    ArchiveManifest = EpisodeArchiveReader = EpisodeArchiveWriter = None
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
    assert validate_archive_manifest(tmp_path / "episode" / "episode.manifest.json")["valid"] is True


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
        "attempt_id": "att-crash-00001",
        "episode_id": None,
        "program_id": "T01",
        "episode_kind": "nominal",
        "outcome": "simulator_crash",
        "terminal_reason": "simulator_exception",
        "error_type": "RuntimeError",
        "error": "simulator stopped",
        "traceback": "Traceback: bounded evidence",
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
    assert manifest.as_dict()["timeline"]["transitions"] == 1
    reader = EpisodeArchiveReader(tmp_path / "attempt" / "attempt.manifest.json")
    assert reader.to_episode_record()["outcome"] == "simulator_crash"
    np.testing.assert_array_equal(reader.raw_arrays["actions"], [[1.0, 2.0]])
    assert reader.debug_metadata["traceback"] == "Traceback: bounded evidence"
    assert reader.debug_metadata["stderr"] == "simulator process failed token=[REDACTED]"
    assert reader.debug_metadata["timeout"] is False


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
