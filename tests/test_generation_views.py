from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from icgs.data.collection.generation.distributed_contracts import (
    ARCHIVE_DATASET_IDENTITY,
    ARCHIVE_EPISODE_SCHEMA_VERSION,
    ARCHIVE_FORMAT_ID,
    ArchiveProfileConfig,
)
from icgs.data.collection.generation.episode_archive import EpisodeArchiveReader, EpisodeArchiveWriter
from icgs.data.datasets.episodes import adapter_a0, adapter_a1
from icgs.data.datasets.generation_views import build_generation_view

try:
    from icgs.data.datasets.generation_archive import ArchiveDatasetIndex, SampleRef
except ModuleNotFoundError:
    ArchiveDatasetIndex = SampleRef = None


def _episode(
    episode_id: str,
    *,
    outcome: str = "success",
    split: str = "train",
    subset: str | None = None,
    kind: str = "nominal",
    n_actions: int = 5,
    event_t: int | None = 2,
    intervention: dict | None = None,
) -> dict:
    subset = ("train_core" if split == "train" else None) if subset is None else subset
    observations = []
    for boundary in range(n_actions + 1):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = 10.0 + boundary
        points = np.asarray(
            [[10.0 + boundary + point / 100.0, point / 50.0, 0.25] for point in range(32)],
            dtype=np.float32,
        )
        observations.append({
            "points": points,
            "point_valid": np.ones(len(points), dtype=np.bool_),
            "T_w_e": pose,
            "grip": boundary % 2,
        })
    transitions = []
    for index in range(n_actions):
        transitions.append({
            "command": {
                "T_w_e": observations[index + 1]["T_w_e"],
                "grip": observations[index + 1]["grip"],
                "duration_s": 0.1,
            },
            "achieved_duration_s": 0.1,
            "physics_substeps": 4,
            "before_boundary": index,
            "after_boundary": index + 1,
        })
    record = {
        "schema_version": "icgs_episode_v2",
        "provenance": {
            "episode_id": episode_id,
            "attempt_id": f"att-{episode_id}",
            "program_id": "T01",
            "split": split,
            "subset": subset,
            "scene_seed": 17,
            "asset_instance_id": f"asset-{episode_id}",
            "execution_mode": "scripted_waypoint_v1",
            "calibration_id": "calibration-fixture-v1",
            "observation_origin": "measured",
            "raw_commands_id": "raw-fixture-v1",
            "materialized_commands_id": "materialized-fixture-v1",
            "episode_kind": kind,
            "outcome": outcome,
            "dataset_version": "icgs-primary-v3",
            "collection_seed": 20260920,
            "failure_type": None if outcome == "success" else "predicate_unsatisfied",
            "terminal_reason": "predicate_satisfied" if outcome == "success" else "predicate_failed",
            "terminal_t": n_actions - 1,
            "valid_observation_until": n_actions,
            "terminated": True,
            "truncated": False,
            "external_intervention": bool(intervention),
            "episode_has_external_intervention": bool(intervention),
            "intervention_id": None if intervention is None else intervention.get("intervention_id"),
            "application_scope": None if intervention is None else intervention.get("application_scope"),
            "application_t": None if intervention is None else intervention.get("application_t"),
        },
        "online_observations": observations,
        "transitions": transitions,
        "dt": [0.1] * n_actions,
        "robot_states": [{"T_w_e": item["T_w_e"], "grip": item["grip"]} for item in observations],
        "object_states": [{"block": np.asarray([index, 0.0, 0.0])} for index in range(n_actions + 1)],
        "events": [] if event_t is None else [{"step_id": f"event-{event_t}", "start_t": event_t}],
        "rho": np.linspace(0.0, 1.0, (n_actions + 1) * 2, dtype=np.float32).reshape(n_actions + 1, 2),
        "rho_valid": np.ones((n_actions + 1, 2), dtype=np.bool_),
    }
    if intervention is not None:
        record["intervention"] = dict(intervention)
    return record


def _write_archive(root: Path, record: dict, *, raw_arrays: dict | None = None) -> dict:
    archive_rel = f"episodes/{record['provenance']['episode_id']}"
    archive_dir = root / archive_rel
    profile = ArchiveProfileConfig(chunk_boundaries=1, local_artifact_retention="keep")
    EpisodeArchiveWriter(profile).write_episode(
        record,
        raw_arrays=raw_arrays or {},
        debug_metadata={
            "source_run_id": "run-fixture",
            "code_revision": "a" * 40,
            "preprocessing_identity": "measured_raw_v1",
        },
        output_dir=archive_dir,
    )
    manifest_path = archive_dir / "episode.manifest.json"
    archive = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = record["provenance"]
    return {
        "archive_ref": archive_rel,
        "archive_manifest": f"{archive_rel}/episode.manifest.json",
        "archive_format_id": ARCHIVE_FORMAT_ID,
        "episode_schema_version": ARCHIVE_EPISODE_SCHEMA_VERSION,
        "dataset_identity": ARCHIVE_DATASET_IDENTITY,
        "manifest_sha256": "b" * 64,
        "file_sha256": {
            "episode.manifest.json": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "episode_id": provenance["episode_id"],
        "attempt_id": provenance["attempt_id"],
        "program_id": provenance["program_id"],
        "split": provenance["split"],
        "subset": provenance["subset"],
        "outcome": provenance["outcome"],
        "episode_kind": provenance["episode_kind"],
        "preprocessing_identity": archive["preprocessing_identity"],
    }


def _dataset_manifest(root: Path, entries: list[dict], *, failure_attempts: list[dict] | None = None) -> Path:
    path = root / "dataset_manifest.json"
    path.write_text(json.dumps({
        "manifest_version": 3,
        "dataset_identity": ARCHIVE_DATASET_IDENTITY,
        "archive_format_id": ARCHIVE_FORMAT_ID,
        "episode_schema_version": ARCHIVE_EPISODE_SCHEMA_VERSION,
        "view_status": "PROVISIONAL",
        "episodes": entries,
        "failure_attempts": failure_attempts or [],
    }, sort_keys=True), encoding="utf-8")
    return path


def _require_archive_dataset_api():
    assert ArchiveDatasetIndex is not None and SampleRef is not None, (
        "lazy archive-backed generation views require ArchiveDatasetIndex and SampleRef"
    )


def test_archive_dataset_index_api_is_available():
    _require_archive_dataset_api()


def test_index_rejects_archive_manifest_digest_mismatch(tmp_path: Path):
    _require_archive_dataset_api()
    entry = _write_archive(tmp_path, _episode("ep-tampered"))
    entry["file_sha256"]["episode.manifest.json"] = "0" * 64
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        ArchiveDatasetIndex.from_manifest(_dataset_manifest(tmp_path, [entry]))


def test_archive_index_rejects_unknown_role(tmp_path: Path):
    _require_archive_dataset_api()
    entry = _write_archive(tmp_path, _episode("ep-role"))
    index = ArchiveDatasetIndex.from_manifest(_dataset_manifest(tmp_path, [entry]))
    with pytest.raises(ValueError, match="role"):
        list(index.sample_refs("D_geom", "trian"))


def test_legacy_record_views_reject_unknown_role_and_split_subset_leakage():
    train_missing_subset = _episode("legacy-train-missing", subset=None)
    train_missing_subset["provenance"].pop("subset")
    train_arbitrary_subset = _episode("legacy-train-arbitrary", subset="unassigned")
    held_out_with_subset = _episode("legacy-eval-subset", split="dev", subset="train_core")
    held_out = _episode("legacy-eval", split="dev")

    with pytest.raises(ValueError, match="role"):
        build_generation_view([train_missing_subset], "D_geom", role="trian", mix=False)
    assert build_generation_view([train_missing_subset], "D_geom", role="train", mix=False) == []
    assert build_generation_view([train_arbitrary_subset], "D_geom", role="train", mix=False) == []
    assert build_generation_view([held_out_with_subset], "D_geom", role="evaluation", mix=False) == []
    assert len(build_generation_view([held_out], "D_geom", role="evaluation", mix=False)) == 6
    assert len(build_generation_view(
        [train_missing_subset, train_arbitrary_subset, held_out_with_subset, held_out],
        "D_geom",
        role="all",
        mix=False,
    )) == 24


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("split", "mystery"), ("subset", "unassigned"), ("subset", None)],
)
def test_index_rejects_unknown_split_or_subset(tmp_path: Path, field: str, invalid: str):
    _require_archive_dataset_api()
    entry = _write_archive(tmp_path, _episode(f"ep-invalid-{field}"))
    archive_path = tmp_path / entry["archive_manifest"]
    archive_payload = json.loads(archive_path.read_text(encoding="utf-8"))
    archive_payload[field] = invalid
    archive_payload["record_metadata"]["provenance"][field] = invalid
    archive_path.write_text(
        json.dumps(archive_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    entry[field] = invalid
    entry["file_sha256"]["episode.manifest.json"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=field):
        ArchiveDatasetIndex.from_manifest(_dataset_manifest(tmp_path, [entry]))


def test_index_and_refs_are_metadata_only_and_include_valid_failures(tmp_path: Path, monkeypatch):
    _require_archive_dataset_api()
    success = _episode("ep-success")
    valid_failure = _episode("ep-valid-failure", outcome="valid_failure")
    validation = _episode("ep-validation", subset="train_val")
    held_out = _episode("ep-held-out", split="dev")
    entries = [
        _write_archive(tmp_path, success),
        _write_archive(tmp_path, valid_failure),
        _write_archive(tmp_path, validation),
        _write_archive(tmp_path, held_out),
    ]
    manifest_path = _dataset_manifest(tmp_path, entries, failure_attempts=[{
        "episode_id": None,
        "outcome": "simulator_crash",
        "attempt_id": "attempt-crashed",
    }])

    def forbidden_chunk_read(*_args, **_kwargs):
        raise AssertionError("building an archive index or SampleRef must not inflate a point chunk")

    monkeypatch.setattr(EpisodeArchiveReader, "_load_chunk", forbidden_chunk_read)
    index = ArchiveDatasetIndex.from_manifest(manifest_path)
    refs = list(index.sample_refs("D_geom", "train"))

    expected_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert index.dataset_manifest_sha256 == expected_manifest_sha256
    assert refs[0].dataset_manifest_sha256 == expected_manifest_sha256
    with pytest.raises(ValueError, match="role"):
        list(index.sample_refs("D_geom", "trian"))
    assert len(refs) == 12
    assert {ref.episode_id for ref in refs} == {"ep-success", "ep-valid-failure"}
    assert all(isinstance(ref, SampleRef) for ref in refs)
    assert [ref.t for ref in refs if ref.episode_id == "ep-success"] == list(range(6))
    assert refs[0]["pointcloud"] == ("ep-success", "pointcloud", 0)
    assert refs[0]["rgb"] is None
    assert refs[0]["depth"] is None
    assert refs[0]["mask"] is None
    assert len(list(index.sample_refs("D_geom", "validation"))) == 6
    assert len(list(index.sample_refs("D_geom", "evaluation"))) == 6
    assert len(list(index.sample_refs("D_geom", "all"))) == 24


def test_temporal_and_dynamics_refs_keep_causal_cross_chunk_windows(tmp_path: Path):
    _require_archive_dataset_api()
    record = _episode("ep-causal", n_actions=5, event_t=4)
    entry = _write_archive(tmp_path, record)
    index = ArchiveDatasetIndex.from_manifest(_dataset_manifest(tmp_path, [entry]), sample_seed=41)

    temporal = list(build_generation_view(index, "D_temporal", role="train", mix=False))
    at_four = next(ref for ref in temporal if ref.episode_id == "ep-causal" and ref.t == 4)
    assert at_four["observations"] == (
        ("ep-causal", "observation", 1),
        ("ep-causal", "observation", 2),
        ("ep-causal", "observation", 3),
        ("ep-causal", "observation", 4),
    )
    assert at_four["actions"] == (
        ("ep-causal", "action", 1),
        ("ep-causal", "action", 2),
        ("ep-causal", "action", 3),
    )
    assert ("ep-causal", "action", 4) not in at_four["actions"]
    with pytest.raises(TypeError):
        at_four.fields["t"] = 99
    with pytest.raises(AttributeError):
        at_four["actions"].append(("ep-causal", "action", 4))

    reader = index.reader_for(at_four)
    resolved = [reader.observation(boundary)["points"][0, 0] for _, _, boundary in at_four["observations"]]
    assert resolved == [11.0, 12.0, 13.0, 14.0]
    random_access = [reader.observation(boundary)["points"][0, 0] for boundary in (4, 1, 5, 0)]
    assert random_access == [14.0, 11.0, 15.0, 10.0]
    assert reader.transition(3)["achieved_duration_s"] == pytest.approx(0.1)

    dyn = list(build_generation_view(index, "D_dyn", role="train", mix=False))
    dyn_at_four = next(ref for ref in dyn if ref.t == 4)
    assert dyn_at_four["history"] == (0, 1, 2, 3)
    assert dyn_at_four["observation_t1"] == 5
    assert dyn_at_four["dt"] == ("ep-causal", "dt", 4)


def test_dyn_intervention_flags_only_mark_matching_timestep(tmp_path: Path):
    _require_archive_dataset_api()
    intervention = {
        "intervention_id": "object_displacement_v1",
        "application_scope": "timestep",
        "application_t": 2,
        "external_intervention": True,
    }
    record = _episode("ep-intervention", kind="perturbed", intervention=intervention)
    entry = _write_archive(tmp_path, record)
    index = ArchiveDatasetIndex.from_manifest(_dataset_manifest(tmp_path, [entry]))

    refs = list(build_generation_view(index, "D_dyn", role="train", mix=False))
    assert [ref["transition_has_external_intervention"] for ref in refs] == [False, False, True, False, False]
    assert [ref["intervention_id"] for ref in refs] == [None, None, "object_displacement_v1", None, None]


def test_archive_transition_views_preserve_training_mixture_default(tmp_path: Path):
    _require_archive_dataset_api()
    nominal = _episode("ep-nominal", kind="nominal", event_t=None)
    perturbed = _episode("ep-perturbed", kind="perturbed", event_t=None)
    entries = [_write_archive(tmp_path, nominal), _write_archive(tmp_path, perturbed)]
    index = ArchiveDatasetIndex.from_manifest(_dataset_manifest(tmp_path, entries))

    all_refs = build_generation_view(index, "D_dyn", role="train", mix=False)
    mixed_refs = build_generation_view(index, "D_dyn", role="train")
    assert len(all_refs) == 10
    assert len(mixed_refs) == 7
    assert sum(ref["episode_kind"] == "perturbed" for ref in mixed_refs) == 2


def test_geom_optional_modalities_are_refs_only_when_archived(tmp_path: Path):
    _require_archive_dataset_api()
    record = _episode("ep-rgb")
    rgb = np.arange(6 * 2 * 2 * 3, dtype=np.uint8).reshape(6, 2, 2, 3)
    entry = _write_archive(tmp_path, record, raw_arrays={"rgb": rgb})
    index = ArchiveDatasetIndex.from_manifest(_dataset_manifest(tmp_path, [entry]))

    ref = list(index.sample_refs("D_geom", "train"))[-1]
    assert ref["rgb"] == ("ep-rgb", "rgb", 5)
    assert ref["depth"] is None
    assert index.read_optional_modality(ref, "rgb").shape == (2, 2, 3)
    np.testing.assert_array_equal(index.read_optional_modality(ref, "rgb"), rgb[5])


def test_producer_named_sparse_modalities_resolve_only_captured_boundaries(tmp_path: Path, monkeypatch):
    _require_archive_dataset_api()
    record = _episode("ep-sparse-modalities")
    rgb = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    rgb_boundaries = np.asarray([0, 2, 5], dtype=np.int64)
    depth = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)
    depth_boundaries = np.asarray([1, 4], dtype=np.int64)
    mask = np.arange(2 * 2 * 2, dtype=np.uint8).reshape(2, 2, 2)
    mask_boundaries = np.asarray([2, 5], dtype=np.int64)
    entry = _write_archive(tmp_path, record, raw_arrays={
        "front_rgb_frames": rgb,
        "front_rgb_frame_boundaries": rgb_boundaries,
        "wrist_depth_frames": depth,
        "wrist_depth_frame_boundaries": depth_boundaries,
        "wrist_mask_frames": mask,
        "wrist_mask_frame_boundaries": mask_boundaries,
    })
    manifest_path = _dataset_manifest(tmp_path, [entry])

    def forbidden_chunk_read(*_args, **_kwargs):
        raise AssertionError("optional-modality indexing must not load image or point chunks")

    with monkeypatch.context() as patch:
        patch.setattr(EpisodeArchiveReader, "_load_chunk", forbidden_chunk_read)
        index = ArchiveDatasetIndex.from_manifest(manifest_path)
        refs = list(index.sample_refs("D_geom", "train"))

    assert refs[2]["rgb"] == ("ep-sparse-modalities", "rgb", 2)
    assert refs[2]["depth"] == ("ep-sparse-modalities", "depth", 2)
    assert refs[2]["mask"] == ("ep-sparse-modalities", "mask", 2)
    np.testing.assert_array_equal(index.read_optional_modality(refs[2], "rgb"), rgb[1])
    np.testing.assert_array_equal(index.read_optional_modality(refs[5], "rgb"), rgb[2])
    assert index.read_optional_modality(refs[3], "rgb") is None
    np.testing.assert_array_equal(index.read_optional_modality(refs[1], "depth"), depth[0])
    assert index.read_optional_modality(refs[2], "depth") is None
    np.testing.assert_array_equal(index.read_optional_modality(refs[5], "mask"), mask[1])


def test_task_view_keeps_label_references_and_resolves_rows_across_chunks(tmp_path: Path):
    _require_archive_dataset_api()
    record = _episode("ep-task", n_actions=5, event_t=3)
    entry = _write_archive(tmp_path, record)
    index = ArchiveDatasetIndex.from_manifest(_dataset_manifest(tmp_path, [entry]))

    refs = list(build_generation_view(index, "D_task", role="train", mix=False))
    at_three = next(ref for ref in refs if ref.t == 3)
    assert at_three["events"] == (("ep-task", "event", "event-3"),)
    assert at_three["rho"] == ("ep-task", "rho", 3)
    assert at_three["nu"] is None
    labels = index.reader_for(at_three).task_labels(3)
    np.testing.assert_array_equal(labels["rho"], record["rho"][3])
    np.testing.assert_array_equal(labels["rho_valid"], np.ones(2, dtype=np.bool_))


def test_preprocessing_identity_hash_seed_and_resolved_points_are_stable(tmp_path: Path, monkeypatch):
    _require_archive_dataset_api()
    record = _episode("ep-preprocess", n_actions=1, event_t=None)
    entry = _write_archive(tmp_path, record)
    manifest_path = _dataset_manifest(tmp_path, [entry])

    from icgs.data.datasets import generation_archive

    monkeypatch.setattr(
        generation_archive,
        "_remove_statistical_outliers",
        lambda points, nb_neighbors=20, std_ratio=2.0: (np.asarray(points), np.arange(len(points))),
    )
    first_index = ArchiveDatasetIndex.from_manifest(manifest_path, sample_seed=123)
    second_index = ArchiveDatasetIndex.from_manifest(manifest_path, sample_seed=123)
    ref_a = next(first_index.sample_refs("D_geom", "train"))
    ref_b = next(second_index.sample_refs("D_geom", "train"))
    next_boundary = list(first_index.sample_refs("D_geom", "train"))[1]

    assert ref_a.preprocessing_identity == ref_b.preprocessing_identity
    assert ref_a.preprocessing_sha256 == ref_b.preprocessing_sha256
    assert ref_a.sample_seed == ref_b.sample_seed
    assert ref_a.preprocessing_identity == "icgs_ip_native_sor20_std2_mt19937_2048_local_v1"
    assert ref_a.preprocessing_sha256 == "4aff1a385d4b956555f6d21e7ea44011723f4123fa3418ae25de70fb4dd96d10"
    assert ref_a.sample_seed != next_boundary.sample_seed
    assert first_index.metadata_payload() == second_index.metadata_payload()
    assert first_index.metadata_payload()["sample_seed"] == 123
    assert first_index.metadata_payload()["preprocessing_sha256"] == ref_a.preprocessing_sha256
    differently_seeded_index = ArchiveDatasetIndex.from_manifest(manifest_path, sample_seed=124)
    with pytest.raises(ValueError, match="sample reference"):
        differently_seeded_index.reader_for(ref_a)
    # Open3D is absent locally; this test isolates the native filtering boundary
    # to verify deterministic sampling and frame conversion on a tiny fixture.
    points_a = first_index.preprocess_observation(ref_a)
    points_b = second_index.preprocess_observation(ref_b)
    assert points_a.shape == (2048, 3)
    np.testing.assert_array_equal(points_a, points_b)
    assert float(points_a[:, 0].max()) < 1.0


def test_archive_generation_refs_do_not_change_a0_a1_adapters():
    a0 = adapter_a0({"points": np.asarray([[1.0, 2.0, 3.0]]), "valid": np.asarray([True]), "archive_manifest": "unused"})
    assert set(a0) == {"points", "valid"}
    a1 = adapter_a1({
        "boundary": 1,
        "history_transitions": ("previous",),
        "target_transitions": ("next",),
        "archive_manifest": "unused",
    })
    assert a1["transitions"] == ("previous", "next")
    assert a1["history_transitions"] == ("previous",)
    assert a1["target_transitions"] == ("next",)
