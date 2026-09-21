"""Validation for the owner-approved primary composition manifest.

This module is deliberately simulator-free.  It gates declarative protocol
metadata before any RLBench process is launched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from icgs.data.collection.programs import program_catalog


PRIMARY_SPLITS = {"train", "development", "test"}
EXECUTION_STATUSES = {"planned", "generation_authorized", "pilot_certified", "generation_certified"}
REQUIRED_FIELDS = {
    "program_id", "split", "family", "ordered_steps", "scene_id",
    "asset_family_id", "asset_version", "source_lineage_id", "workspace",
    "randomization", "seed_demos", "execution_modes", "controller",
    "predicates", "execution_status",
}


def load_approved_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"approved composition manifest not found: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"approved composition manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(data, dict):
        raise ValueError("approved composition manifest must be a JSON object")
    errors = validate_approved_manifest(data)
    if errors:
        raise ValueError("invalid approved composition manifest: " + "; ".join(errors))
    return data


def _is_range(value: Any, *, length: int = 2) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        return False
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(value == value and abs(value) != float("inf") for value in values) and values[0] <= values[-1]


def _check_ranges(row: Mapping[str, Any], errors: list[str]) -> None:
    workspace = row.get("workspace")
    if not isinstance(workspace, Mapping) or set(workspace) != {"x_m", "y_m", "z_m"}:
        errors.append(f"{row.get('program_id')}.workspace must contain x_m/y_m/z_m")
    elif not all(_is_range(workspace[key]) for key in ("x_m", "y_m", "z_m")):
        errors.append(f"{row.get('program_id')}.workspace has invalid ranges")

    randomization = row.get("randomization")
    required_randomization = {
        "translation_m", "yaw_deg", "scale", "camera_profile_id",
        "lighting_profile_id", "waypoint_delta_m",
    }
    if not isinstance(randomization, Mapping) or not required_randomization <= randomization.keys():
        errors.append(f"{row.get('program_id')}.randomization is incomplete")
        return
    translation = randomization["translation_m"]
    if not isinstance(translation, Mapping) or set(translation) != {"x", "y", "z"}:
        errors.append(f"{row.get('program_id')}.randomization.translation_m is incomplete")
    elif not all(_is_range(translation[key]) for key in ("x", "y", "z")):
        errors.append(f"{row.get('program_id')}.randomization.translation_m has invalid ranges")
    for key in ("yaw_deg", "scale", "waypoint_delta_m"):
        if not _is_range(randomization[key]):
            errors.append(f"{row.get('program_id')}.randomization.{key} has invalid range")


def validate_approved_manifest(data: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deterministic diagnostics; an empty tuple means valid."""
    errors: list[str] = []
    if not isinstance(data, Mapping):
        return ("manifest must be a mapping",)
    if data.get("manifest_version") != 1:
        errors.append("manifest_version must be 1")
    rows = data.get("catalog")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        return tuple(errors + ["catalog must be a sequence"])

    expected = {
        program_id: spec
        for program_id, spec in program_catalog().items()
        if spec.split in {"train", "development", "test"}
    }
    expected_ids = set(expected)
    seen: set[str] = set()
    split_counts = {split: 0 for split in PRIMARY_SPLITS}
    asset_splits: dict[str, str] = {}
    lineage_splits: dict[str, str] = {}

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"catalog[{index}] must be a mapping")
            continue
        program_id = row.get("program_id")
        prefix = f"catalog[{index}]/{program_id}"
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"{prefix} missing fields: {sorted(missing)}")
            continue
        if not isinstance(program_id, str) or program_id not in expected:
            errors.append(f"{prefix} unknown program_id")
            continue
        if program_id in seen:
            errors.append(f"{prefix} duplicate program_id")
        seen.add(program_id)
        split = row.get("split")
        canonical_split = expected[program_id].split
        if split != canonical_split:
            errors.append(f"{prefix} split {split!r} != catalog split {canonical_split!r}")
        if split in split_counts:
            split_counts[split] += 1
        if tuple(row.get("ordered_steps", ())) != tuple(expected[program_id].steps):
            errors.append(f"{prefix} ordered_steps do not match program catalog")
        if row.get("execution_status") not in EXECUTION_STATUSES:
            errors.append(f"{prefix} has invalid execution_status")
        if row.get("execution_status") == "generation_certified" and split != "train":
            errors.append(f"{prefix} generation_certified is only valid for train")
        seeds = row.get("seed_demos")
        if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence) or len(seeds) != 5:
            errors.append(f"{prefix}.seed_demos must contain exactly five IDs")
        elif len(set(seeds)) != len(seeds) or not all(isinstance(seed, str) and seed.strip() for seed in seeds):
            errors.append(f"{prefix}.seed_demos must contain unique non-empty IDs")
        modes = row.get("execution_modes")
        if isinstance(modes, (str, bytes)) or not isinstance(modes, Sequence) or not modes:
            errors.append(f"{prefix}.execution_modes must be non-empty")
        for field in ("family", "scene_id", "asset_family_id", "asset_version", "source_lineage_id"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                errors.append(f"{prefix}.{field} must be non-empty")
        asset = row.get("asset_family_id")
        if isinstance(asset, str):
            previous = asset_splits.setdefault(asset, split)
            if previous != split:
                errors.append(f"{prefix}.asset_family_id crosses splits")
        lineage = row.get("source_lineage_id")
        if isinstance(lineage, str):
            previous = lineage_splits.setdefault(lineage, split)
            if previous != split:
                errors.append(f"{prefix}.source_lineage_id crosses splits")
        _check_ranges(row, errors)

    if seen != expected_ids:
        errors.append(f"catalog IDs differ from canonical catalog: missing={sorted(expected_ids-seen)} extra={sorted(seen-expected_ids)}")
    if split_counts != {"train": 20, "development": 4, "test": 12}:
        errors.append(f"split cardinality mismatch: {split_counts}")
    return tuple(errors)


def validate_episode_provenance(
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate one episode's owner-approved program/seed/mode provenance."""
    errors = list(validate_approved_manifest(manifest))
    if errors:
        return tuple(errors)
    program_id = provenance.get("program_id")
    row = next((item for item in manifest["catalog"] if item["program_id"] == program_id), None)
    if row is None:
        return (f"episode program_id {program_id!r} is not in approved catalog",)
    episode_split = provenance.get("split")
    approved_split = row["split"]
    if episode_split == "dev":
        episode_split = "development"
    if episode_split != approved_split:
        errors.append(f"episode split {provenance.get('split')!r} != approved split {row['split']!r}")
    if provenance.get("source_lineage_id") != row["source_lineage_id"]:
        errors.append("episode source_lineage_id does not match approved binding")
    if provenance.get("asset_family_id") != row["asset_family_id"]:
        errors.append("episode asset_family_id does not match approved binding")
    if provenance.get("seed_id") not in row["seed_demos"]:
        errors.append("episode seed_id is not in approved seed_demos")
    if provenance.get("execution_mode") not in row["execution_modes"]:
        errors.append("episode execution_mode is not in approved execution_modes")
    if row["execution_status"] == "planned":
        errors.append("planned program cannot publish an episode")
    return tuple(errors)


__all__ = ["load_approved_manifest", "validate_approved_manifest", "validate_episode_provenance"]
