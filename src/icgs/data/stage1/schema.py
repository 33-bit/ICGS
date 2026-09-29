"""``icgs_stage1_episode_v1``: one executed episode, source-neutral.

Timeline convention (all arrays in one ``arrays.npz``):

* ``step_*`` arrays have ``S + 1`` rows. Row ``s`` is the measured simulator
  state at physics boundary ``s``; row 0 precedes the first recorded step.
* ``cmd_*`` arrays have ``S`` rows. Row ``s`` is the command that was *issued to
  the controller* while advancing boundary ``s`` to ``s + 1``. It is never
  derived from achieved state. ``*_valid`` masks mark rows where that command
  channel was written during the step.
* ``frame_*`` and ``cam_<name>_*`` arrays have ``F`` rows, one per rendered
  observation; ``frame_step[f]`` is the boundary at which it was rendered.

Perturbation and outcome are separate blocks. A perturbation says what was
changed before or during execution; the outcome says what the simulator
measured afterwards. Neither is inferred from the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA_VERSION = "icgs_stage1_episode_v1"

# What was changed. "benchmark_factor" = an upstream benchmark perturbation
# (for RoboHiMan: an enabled Colosseum variation factor, i.e. AP/CP levels).
PERTURBATION_FAMILIES = (
    "none",
    "benchmark_factor",
    "execution_pose_offset",
    "object_displacement",
    "gripper_timing",
)

# What happened. "failure" = executed to its end and the task success condition
# was false; "invalid_execution" = the controller could not execute a command
# (for example no collision-free path); "simulator_error" = crash, attempt only.
OUTCOME_STATUSES = (
    "success",
    "failure",
    "timeout",
    "invalid_execution",
    "simulator_error",
)

# Recoverability is a claim that needs executed evidence (a successful recovery
# or a proven absorbing state); without it the value stays "unknown".
RECOVERABILITY = ("unknown", "recoverable", "terminal")

# Model-facing online fields. Everything else in a record is offline supervision
# or provenance and must not enter a deployed network input.
ONLINE_FIELDS = ("points", "point_valid", "T_w_e", "grip")

_REQUIRED_BLOCKS = (
    "schema_version",
    "episode_id",
    "source",
    "perturbation",
    "outcome",
    "lineage",
    "environment",
    "controller",
    "code",
    "cameras",
    "predicates",
    "events",
    "arrays",
    "counts",
)
_SOURCE_FIELDS = ("benchmark", "benchmark_revision", "task", "task_family", "level", "variation_index")
_LEVELS = ("A", "AP", "C", "CP", "custom")


def _require(mapping: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return mapping


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def perturbation_families(perturbation: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the sorted set of perturbation families actually applied."""
    families = set()
    for factor in perturbation.get("benchmark_factors", ()):
        if factor.get("enabled"):
            families.add("benchmark_factor")
    for item in perturbation.get("execution", ()):
        families.add(item["family"])
    return tuple(sorted(families)) or ("none",)


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    manifest = _require(manifest, "manifest")
    missing = [name for name in _REQUIRED_BLOCKS if name not in manifest]
    if missing:
        raise ValueError(f"stage1 manifest missing blocks: {missing}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {manifest['schema_version']!r}")
    _nonempty(manifest["episode_id"], "episode_id")

    source = _require(manifest["source"], "source")
    for field in _SOURCE_FIELDS:
        if field not in source:
            raise ValueError(f"source.{field} is required")
    if source["level"] not in _LEVELS:
        raise ValueError(f"source.level must be one of {_LEVELS}")

    perturbation = _require(manifest["perturbation"], "perturbation")
    for key in ("benchmark_factors", "execution"):
        if not isinstance(perturbation.get(key), Sequence):
            raise ValueError(f"perturbation.{key} must be a list")
    for item in perturbation["execution"]:
        item = _require(item, "perturbation.execution[]")
        if item.get("family") not in PERTURBATION_FAMILIES or item.get("family") == "none":
            raise ValueError(f"unknown execution perturbation family {item.get('family')!r}")
    declared = tuple(perturbation.get("families", ()))
    if declared != perturbation_families(perturbation):
        raise ValueError("perturbation.families must equal the families actually applied")

    outcome = _require(manifest["outcome"], "outcome")
    if outcome.get("status") not in OUTCOME_STATUSES:
        raise ValueError(f"outcome.status must be one of {OUTCOME_STATUSES}")
    if outcome.get("recoverability") not in RECOVERABILITY:
        raise ValueError(f"outcome.recoverability must be one of {RECOVERABILITY}")
    if outcome["recoverability"] != "unknown" and not outcome.get("recoverability_evidence"):
        raise ValueError("a recoverability claim needs recoverability_evidence")
    if outcome["status"] == "success" and outcome.get("task_success_final") is not True:
        raise ValueError("success requires the measured final task success")
    if outcome["status"] == "simulator_error":
        raise ValueError("simulator_error is an attempt record, not a stage1 episode")

    lineage = _require(manifest["lineage"], "lineage")
    _nonempty(lineage.get("collection_run_id"), "lineage.collection_run_id")
    if "parent_episode_id" not in lineage:
        raise ValueError("lineage.parent_episode_id must be present (null for roots)")

    controller = _require(manifest["controller"], "controller")
    dt = controller.get("physics_dt")
    if not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("controller.physics_dt must be positive and finite")
    _nonempty(controller.get("native_command"), "controller.native_command")

    counts = _require(manifest["counts"], "counts")
    for name in ("steps", "frames"):
        value = counts.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"counts.{name} must be a nonnegative integer")
    if counts["steps"] < 1:
        raise ValueError("an episode needs at least one executed physics step")

    arrays = _require(manifest["arrays"], "arrays")
    _nonempty(arrays.get("sha256"), "arrays.sha256")
    predicates = _require(manifest["predicates"], "predicates")
    if not isinstance(predicates.get("names"), Sequence):
        raise ValueError("predicates.names must be a list")


_STEP_REQUIRED = (
    "step_sim_time",
    "step_joint_positions",
    "step_joint_velocities",
    "step_tip_pose",
    "step_gripper_joint_positions",
    "step_predicates",
    "step_task_success",
)
_CMD_REQUIRED = (
    "cmd_arm_joint_target",
    "cmd_arm_joint_target_valid",
    "cmd_gripper_joint_velocity",
    "cmd_gripper_joint_velocity_valid",
    "cmd_grasp_event",
    "cmd_phase",
)


def validate_arrays(manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    """Check timeline alignment between manifest counts and stored arrays."""
    steps = int(manifest["counts"]["steps"])
    frames = int(manifest["counts"]["frames"])
    for name in _STEP_REQUIRED:
        if name not in arrays:
            raise ValueError(f"missing array {name}")
        if arrays[name].shape[0] != steps + 1:
            raise ValueError(f"{name} must have S+1={steps + 1} rows, got {arrays[name].shape[0]}")
    for name in _CMD_REQUIRED:
        if name not in arrays:
            raise ValueError(f"missing array {name}")
        if arrays[name].shape[0] != steps:
            raise ValueError(f"{name} must have S={steps} rows, got {arrays[name].shape[0]}")
    times = arrays["step_sim_time"]
    if not np.all(np.diff(times) > 0):
        raise ValueError("step_sim_time must strictly increase")
    names = list(manifest["predicates"]["names"])
    if arrays["step_predicates"].shape[1:] != (len(names),):
        raise ValueError("step_predicates columns must match predicates.names")
    if "frame_step" in arrays or frames:
        frame_step = arrays.get("frame_step")
        if frame_step is None or frame_step.shape != (frames,):
            raise ValueError("frame_step must have F rows")
        if frames and (frame_step.min() < 0 or frame_step.max() > steps):
            raise ValueError("frame_step must index recorded boundaries")
        if frames and not np.all(np.diff(frame_step) >= 0):
            raise ValueError("frame_step must be nondecreasing")
    for key, value in arrays.items():
        if value.dtype == np.dtype(object):
            raise ValueError(f"{key} must not use object dtype")
        if np.issubdtype(value.dtype, np.floating) and not key.endswith("_nan_ok"):
            valid_key = f"{key}_valid"
            data = value if valid_key not in arrays else value[arrays[valid_key]]
            if not np.isfinite(data).all():
                raise ValueError(f"{key} contains non-finite values in valid rows")


__all__ = [
    "ONLINE_FIELDS",
    "OUTCOME_STATUSES",
    "PERTURBATION_FAMILIES",
    "RECOVERABILITY",
    "SCHEMA_VERSION",
    "perturbation_families",
    "validate_arrays",
    "validate_manifest",
]
