from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from icgs.data.datasets.generation_view_index import DEFAULT_ROLE_SPECS, finalize_generation_views
from icgs.data.datasets.training_export import (
    TRAINING_EXPORT_SCHEMA,
    TRAINING_PREFIX,
    TrainingExportReader,
    export_training_views,
)
from test_generation_views import _dataset_manifest, _episode, _write_archive


REVISION = "a" * 40
VIEWS_REVISION = "b" * 40


def _source_and_views(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    entries = [
        _write_archive(source, _episode("ep-nominal", n_actions=7, event_t=None)),
        _write_archive(source, _episode("ep-failure", kind="perturbed", outcome="valid_failure", n_actions=3, event_t=None)),
        _write_archive(source, _episode("ep-validation", subset="train_val", n_actions=2, event_t=None)),
        _write_archive(source, _episode("ep-evaluation", split="dev", n_actions=2, event_t=1)),
    ]
    manifest = _dataset_manifest(source, entries)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.update({
        "view_status": "PROVISIONAL",
        "hf_prefix": "archive/source",
        "source_revision": REVISION,
        "source_snapshot_created_at_s": 1_758_780_000.0,
        "total_episodes": len(entries),
        "total_failure_attempts": 0,
    })
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    views = tmp_path / "views"
    finalize_generation_views(
        manifest,
        source_revision=REVISION,
        role_specs=DEFAULT_ROLE_SPECS,
        mixture_seed=20260920,
        output_dir=views,
    )
    return source, views


def _export(tmp_path):
    source, views = _source_and_views(tmp_path)
    output = tmp_path / "training"
    export_training_views(
        views, repo_id="33bit/icgs", source_revision=REVISION,
        views_revision=VIEWS_REVISION, source_prefix="archive/source",
        source_manifest=source / "dataset_manifest.json", output_dir=output,
    )
    return source, views, output


def test_export_is_compact_revision_bound_and_idempotent(tmp_path: Path):
    source, views = _source_and_views(tmp_path)
    output = tmp_path / "training"

    first = export_training_views(
        views,
        repo_id="33bit/icgs",
        source_revision=REVISION,
        views_revision=VIEWS_REVISION,
        source_prefix="archive/source",
        source_manifest=source / "dataset_manifest.json",
        output_dir=output,
    )
    second = export_training_views(
        views,
        repo_id="33bit/icgs",
        source_revision=REVISION,
        views_revision=VIEWS_REVISION,
        source_prefix="archive/source",
        source_manifest=source / "dataset_manifest.json",
        output_dir=output,
    )

    assert first.as_dict() | {"output_directory": ""} == second.as_dict() | {"output_directory": ""}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == TRAINING_EXPORT_SCHEMA
    assert manifest["target_prefix"] == TRAINING_PREFIX
    assert manifest["source"]["archive_revision"] == REVISION
    assert manifest["source"]["views_revision"] == VIEWS_REVISION
    assert not list(output.rglob("*.npz"))
    assert len(manifest["views"]) == 12
    assert all(item["sha256"] == hashlib.sha256(
        (output / item["path"]).read_bytes()
    ).hexdigest() for item in manifest["views"])
    assert source.exists()


def test_local_reader_preserves_final_views_and_a0_a1_adapters(tmp_path: Path):
    source, views = _source_and_views(tmp_path)
    output = tmp_path / "training"
    export_training_views(
        views,
        repo_id="33bit/icgs",
        source_revision=REVISION,
        views_revision=VIEWS_REVISION,
        source_prefix="archive/source",
        source_manifest=source / "dataset_manifest.json",
        output_dir=output,
    )

    reader = TrainingExportReader.from_local(output, source_root=source)
    geom_ref = next(reader.sample_refs("D_geom", "train"))
    geom = reader.read_sample(geom_ref)
    assert geom["episode_id"] == geom_ref.episode_id
    assert geom["points"].shape == (32, 3)
    assert geom["valid"].shape == (32,)
    assert reader.a0(geom_ref)["points"].shape == (32, 3)

    dyn_ref = next(reader.sample_refs("D_dyn", "train"))
    dyn = reader.read_sample(dyn_ref)
    assert dyn["boundary"] == dyn_ref.t
    assert len(dyn["transitions"]) == dyn_ref.t + 1
    a1 = reader.a1(dyn_ref, supervised_intervals=1)
    assert a1["boundary"] == dyn_ref.t
    assert len(a1["target_transitions"]) == 1


def test_hf_reader_downloads_only_pinned_files_and_checks_hashes(tmp_path: Path):
    source, views = _source_and_views(tmp_path)
    output = tmp_path / "training"
    export_training_views(
        views,
        repo_id="33bit/icgs",
        source_revision=REVISION,
        views_revision=VIEWS_REVISION,
        source_prefix="archive/source",
        source_manifest=source / "dataset_manifest.json",
        output_dir=output,
    )

    files: dict[str, bytes] = {
        "training/manifest.json": (output / "manifest.json").read_bytes(),
    }
    manifest = json.loads(files["training/manifest.json"])
    for item in manifest["views"]:
        files[f"training/{item['path']}"] = (output / item["path"]).read_bytes()
    for path in (source / "dataset_manifest.json", *source.rglob("episode.manifest.json"), *source.rglob("chunk-*.npz")):
        files[f"archive/source/{path.relative_to(source).as_posix()}"] = path.read_bytes()

    requests: list[str] = []

    def download(_repo_id: str, filename: str, _revision: str):
        requests.append(filename)
        return files[filename]

    reader = TrainingExportReader.from_hf(
        "33bit/icgs",
        VIEWS_REVISION,
        cache_dir=tmp_path / "cache",
        download_file=download,
    )
    ref = next(reader.sample_refs("D_geom", "train"))
    sample = reader.read_sample(ref)
    assert sample["points"].shape == (32, 3)
    assert "training/manifest.json" in requests
    assert any(item.startswith("archive/source/") for item in requests)

    files["training/manifest.json"] = files["training/manifest.json"].replace(b"icgs_training_export_v1", b"tampered_export_v1")
    with pytest.raises(ValueError, match="schema_version"):
        TrainingExportReader.from_hf(
            "33bit/icgs",
            VIEWS_REVISION,
            cache_dir=tmp_path / "cache-tampered",
            download_file=lambda _repo, filename, _revision: files[filename],
        )


def test_export_rejects_source_or_view_revision_mismatch(tmp_path: Path):
    source, views = _source_and_views(tmp_path)
    with pytest.raises(ValueError, match="source revision"):
        export_training_views(
            views,
            repo_id="33bit/icgs",
            source_revision="c" * 40,
            views_revision=VIEWS_REVISION,
            source_prefix="archive/source",
            source_manifest=source / "dataset_manifest.json",
            output_dir=tmp_path / "bad-source",
        )
    with pytest.raises(ValueError, match="target prefix"):
        export_training_views(
            views,
            repo_id="33bit/icgs",
            source_revision=REVISION,
            views_revision=VIEWS_REVISION,
            source_prefix="archive/source",
            source_manifest=source / "dataset_manifest.json",
            output_dir=tmp_path / "wrong-prefix",
            target_prefix="training-v2",
        )


def test_dynamics_keeps_history_from_reset_and_separates_targets(tmp_path):
    source, _, output = _export(tmp_path)
    reader = TrainingExportReader.from_local(output, source_root=source)
    ref = next(ref for ref in reader.sample_refs("D_dyn") if ref.episode_id == "ep-nominal" and ref.t == 6)
    sample = reader.read_sample(ref)
    assert [row["before_boundary"] for row in sample["history_transitions"]] == [0, 1, 2, 3, 4, 5]
    assert [row["before_boundary"] for row in sample["target_transitions"]] == [6]
    assert sample["provenance"]["outcome"] == "success"
    assert sample["observation_t"]["T_w_e"][0, 3] == 16.0
    assert sample["observation_t1"]["T_w_e"][0, 3] == 17.0
    with pytest.raises(ValueError, match="window"):
        reader.read_sample(ref, supervised_intervals=2)


def test_references_are_source_bound_and_cannot_be_forged(tmp_path):
    source, _, output = _export(tmp_path)
    reader = TrainingExportReader.from_local(output, source_root=source)
    ref = next(reader.sample_refs("D_geom"))
    for forged in (
        replace(ref, archive_manifest="other/episode.manifest.json"),
        replace(ref, sample_seed=ref.sample_seed + 1),
        replace(ref, fields={**ref.fields, "pointcloud": (ref.episode_id, "pointcloud", 999)}),
    ):
        with pytest.raises(ValueError, match="belong"):
            reader.read_sample(forged)
    failures = [reader.read_sample(item) for item in reader.sample_refs("D_geom") if item.episode_id == "ep-failure"]
    assert len(failures) == 4
    assert {item["provenance"]["outcome"] for item in failures} == {"valid_failure"}
    assert {ref.episode_id for ref in reader.sample_refs("D_geom", "validation")} == {"ep-validation"}
    assert {ref.episode_id for ref in reader.sample_refs("D_geom", "evaluation")} == {"ep-evaluation"}


def test_task_sample_resolves_events_and_preserves_label_masks(tmp_path):
    source, _, output = _export(tmp_path)
    reader = TrainingExportReader.from_local(output, source_root=source)
    ref = next(reader.sample_refs("D_task", "evaluation"))
    sample = reader.read_sample(ref)
    assert sample["events"] == [{"step_id": "event-1", "start_t": 1}]
    np.testing.assert_array_equal(sample["labels"]["rho_valid"], [True, True])


def test_export_rejects_partial_conflicting_and_symlinked_targets(tmp_path):
    source, views, output = _export(tmp_path)
    kwargs = dict(repo_id="33bit/icgs", source_revision=REVISION, views_revision=VIEWS_REVISION,
                  source_prefix="archive/source", source_manifest=source / "dataset_manifest.json")
    (output / "views/train/D_dyn.json").unlink()
    with pytest.raises(ValueError, match="partial"):
        export_training_views(views, output_dir=output, **kwargs)
    assert not (output / "views/train/D_dyn.json").exists()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export_training_views(views, output_dir=linked / "training", **kwargs)
    assert not list(outside.iterdir())


def test_duplicate_view_inventory_and_modified_view_bytes_are_rejected(tmp_path):
    source, _, output = _export(tmp_path)
    path = output / "manifest.json"
    original = path.read_bytes()
    manifest = json.loads(original)
    manifest["views"].append(manifest["views"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="twelve"):
        TrainingExportReader.from_local(output, source_root=source)
    path.write_bytes(original)
    view = output / "views/train/D_geom.json"
    view.write_bytes(view.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        TrainingExportReader.from_local(output, source_root=source)


def test_sample_seed_corruption_and_source_hash_mismatch_fail_before_writes(tmp_path):
    source, views = _source_and_views(tmp_path)
    path = views / "train/D_geom.json"
    data = json.loads(path.read_bytes())
    data["samples"][0]["sample_seed"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="seed"):
        export_training_views(views, repo_id="33bit/icgs", source_revision=REVISION,
                              views_revision=VIEWS_REVISION, source_prefix="archive/source",
                              source_manifest=source / "dataset_manifest.json", output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
