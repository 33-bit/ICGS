"""Executable task-binding manifests for bounded ICGS data collection.

The catalog in :mod:`programs` describes task intent.  This module supplies the
small, explicit bridge to a runnable simulator scene, assets, seeds, expert
commands, controller and predicate protocols.  It validates metadata only; it
does not import or launch RLBench.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from icgs.data.collection.programs import program_catalog


_REQUIRED_RANDOMIZATION = frozenset({
    "translation_bounds_m",
    "yaw_deg",
    "scale",
    "camera_profile_id",
    "lighting_profile_id",
})


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"binding {field} must be a non-empty string")
    return value.strip()


def _range_pair(value: Any, field: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"binding {field} must contain exactly two finite numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result) or result[0] > result[1]:
        raise ValueError(f"binding {field} must be an ordered finite range")
    return result


def _bounds(value: Any, field: str) -> tuple[tuple[float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"binding {field} must contain three [min,max] ranges")
    return tuple(_range_pair(item, f"{field}[{index}]") for index, item in enumerate(value))


def _positive_tolerance(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"binding {field} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"binding {field} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class ProgramBinding:
    """Complete metadata required before one program can enter collection."""

    program_id: str
    split: str
    scene_id: str
    asset_family_id: str
    asset_version: str
    source_lineage_root: str
    workspace_bounds_m: tuple[tuple[float, float], ...]
    randomization: Mapping[str, Any]
    seed_ids: tuple[str, ...]
    expert_id: str
    waypoint_protocol_id: str
    controller_protocol_id: str
    predicate_protocol_id: str
    predicate_tolerances: Mapping[str, float]
    calibration_id: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProgramBinding":
        if not isinstance(value, Mapping):
            raise ValueError("program binding must be a mapping")
        required = (
            "program_id", "split", "scene_id", "asset_family_id", "asset_version",
            "source_lineage_root", "workspace_bounds_m", "randomization", "seed_ids",
            "expert_id", "waypoint_protocol_id", "controller_protocol_id",
            "predicate_protocol_id", "predicate_tolerances", "calibration_id",
        )
        for field in required:
            if field not in value:
                raise ValueError(f"binding missing required field: {field}")

        program_id = _nonempty(value["program_id"], "program_id")
        catalog = program_catalog()
        if program_id not in catalog:
            raise ValueError(f"binding program_id is not in catalog: {program_id!r}")
        declared_split = _nonempty(value["split"], "split")
        expected_split = "dev" if catalog[program_id].split == "development" else catalog[program_id].split
        if declared_split != expected_split:
            raise ValueError(
                f"binding catalog split {expected_split} != declared split {declared_split}"
            )

        randomization = value["randomization"]
        if not isinstance(randomization, Mapping):
            raise ValueError("binding randomization must be a mapping")
        missing_randomization = _REQUIRED_RANDOMIZATION - set(randomization)
        if missing_randomization:
            raise ValueError(
                f"binding randomization missing required fields: {sorted(missing_randomization)}"
            )
        scale = _range_pair(randomization["scale"], "randomization.scale")
        yaw = _range_pair(randomization["yaw_deg"], "randomization.yaw_deg")
        translation = _bounds(randomization["translation_bounds_m"], "randomization.translation_bounds_m")
        normalized_randomization = {
            "translation_bounds_m": translation,
            "yaw_deg": yaw,
            "scale": scale,
            "camera_profile_id": _nonempty(randomization["camera_profile_id"], "randomization.camera_profile_id"),
            "lighting_profile_id": _nonempty(randomization["lighting_profile_id"], "randomization.lighting_profile_id"),
        }

        seeds = value["seed_ids"]
        if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence) or not seeds:
            raise ValueError("binding seed_ids must be a non-empty sequence")
        seed_ids = tuple(_nonempty(seed, "seed_ids") for seed in seeds)
        if len(set(seed_ids)) != len(seed_ids):
            raise ValueError("binding seed_ids must be unique")

        tolerances = value["predicate_tolerances"]
        if not isinstance(tolerances, Mapping) or not tolerances:
            raise ValueError("binding predicate_tolerances must be a non-empty mapping")
        predicate_tolerances = {
            _nonempty(key, "predicate_tolerances key"): _positive_tolerance(item, f"predicate_tolerances.{key}")
            for key, item in tolerances.items()
        }

        return cls(
            program_id=program_id,
            split=declared_split,
            scene_id=_nonempty(value["scene_id"], "scene_id"),
            asset_family_id=_nonempty(value["asset_family_id"], "asset_family_id"),
            asset_version=_nonempty(value["asset_version"], "asset_version"),
            source_lineage_root=_nonempty(value["source_lineage_root"], "source_lineage_root"),
            workspace_bounds_m=_bounds(value["workspace_bounds_m"], "workspace_bounds_m"),
            randomization=normalized_randomization,
            seed_ids=seed_ids,
            expert_id=_nonempty(value["expert_id"], "expert_id"),
            waypoint_protocol_id=_nonempty(value["waypoint_protocol_id"], "waypoint_protocol_id"),
            controller_protocol_id=_nonempty(value["controller_protocol_id"], "controller_protocol_id"),
            predicate_protocol_id=_nonempty(value["predicate_protocol_id"], "predicate_protocol_id"),
            predicate_tolerances=predicate_tolerances,
            calibration_id=_nonempty(value["calibration_id"], "calibration_id"),
        )


def load_binding_manifest(
    path: str | Path,
    *,
    required_program_ids: Sequence[str] | None = None,
) -> dict[str, ProgramBinding]:
    """Load and validate a concrete program-binding manifest before collection."""

    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"program binding manifest not found: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"program binding manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(data, Mapping) or data.get("manifest_version") != 1:
        raise ValueError("program binding manifest_version must be 1")
    rows = data.get("bindings")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("program binding manifest bindings must be a non-empty sequence")

    bindings: dict[str, ProgramBinding] = {}
    asset_splits: dict[str, str] = {}
    lineage_splits: dict[str, str] = {}
    for row in rows:
        binding = ProgramBinding.from_mapping(row)
        if binding.program_id in bindings:
            raise ValueError(f"duplicate program binding: {binding.program_id}")
        previous_asset_split = asset_splits.setdefault(binding.asset_family_id, binding.split)
        if previous_asset_split != binding.split:
            raise ValueError("asset family crosses splits")
        previous_lineage_split = lineage_splits.setdefault(binding.source_lineage_root, binding.split)
        if previous_lineage_split != binding.split:
            raise ValueError("source lineage crosses splits")
        bindings[binding.program_id] = binding

    if required_program_ids is not None:
        required = tuple(_nonempty(item, "required_program_ids") for item in required_program_ids)
        missing = sorted(set(required) - set(bindings))
        if missing:
            raise ValueError(f"binding manifest missing required programs: {missing}")
    return bindings


__all__ = ["ProgramBinding", "load_binding_manifest"]
