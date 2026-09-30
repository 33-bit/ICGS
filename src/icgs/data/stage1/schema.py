"""``icgs_stage1_episode_v1``: one executed episode, source-neutral.

Timeline convention (all arrays in one ``arrays.npz``):

* ``step_*`` arrays have ``S + 1`` rows. Row ``s`` is the measured simulator
  state at physics boundary ``s``; row 0 precedes the first recorded step.
* ``cmd_*`` arrays have ``S`` rows. Row ``s`` is the command that was *issued to
  the controller* while advancing boundary ``s`` to ``s + 1``: exactly one
  physics step of ``timing.physics_dt_s`` seconds. It is never derived from
  achieved state. ``*_valid`` masks mark rows where that channel was written.
  A transition is ``(step_*[s], cmd_*[s]) -> step_*[s + 1]``.
* ``prof_*`` arrays are host profiling only (e.g. ``prof_step_wall_s``, the
  wall-clock seconds the simulator took); they are not action durations.
* ``monitor_*`` arrays are per-episode task-monitor states with ``*_valid``
  masks (``icgs.data.stage1.labels``). ``monitor_event_id`` is a monitor
  diagnostic, not the ICGS alignment target, which is context-dependent and
  built in the ``D_task`` view.
* ``frame_*`` and ``cam_<name>_*`` arrays have ``F`` rows, one per rendered
  observation; ``frame_step[f]`` is the boundary at which it was rendered.

Perturbation and outcome are separate blocks. A perturbation says what was
changed before or during execution; the outcome says what the simulator
measured afterwards. Neither is inferred from the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import re

import numpy as np


SCHEMA_VERSION = "icgs_stage1_episode_v2"
EPISODE_ID_PATTERN = r"^ep-[a-z0-9_]+-[0-9a-f]{8}-[0-9]{4,}$"
PHYSICS_TRANSITION = "one physics step: (step[s], cmd[s]) -> step[s+1]"

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
    "timing",
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
    if not re.match(EPISODE_ID_PATTERN, str(manifest["episode_id"])):
        raise ValueError(f"episode_id {manifest['episode_id']!r} must match ep-<task>-<run8>-<index>")
    if not str(manifest["episode_id"]).startswith(f"ep-{_require(manifest['source'], 'source').get('task')}-"):
        raise ValueError("episode_id must name the source task")

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

    timing = _require(manifest["timing"], "timing")
    step = timing.get("physics_dt_s")
    if not isinstance(step, (int, float)) or not np.isfinite(step) or step <= 0 or abs(step - dt) > 1e-9:
        raise ValueError("timing.physics_dt_s must equal controller.physics_dt")
    if timing.get("transition") != PHYSICS_TRANSITION:
        raise ValueError("timing.transition must declare the physics-step transition semantics")
    stride = timing.get("frame_stride")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("timing.frame_stride must be a positive integer")
    if abs(float(timing.get("model_dt_s", -1)) - stride * dt) > 1e-9:
        raise ValueError("timing.model_dt_s must equal frame_stride * physics_dt")
    cameras = _require(manifest["cameras"], "cameras")
    if cameras.get("masks_recorded"):
        legend = _require(cameras.get("mask_legend"), "cameras.mask_legend")
        for handle, entry in legend.items():
            if not str(handle).lstrip("-").isdigit():
                raise ValueError("mask_legend keys must be simulator handles")
            for key in ("object", "instance", "role", "category"):
                if key not in entry:
                    raise ValueError(f"mask_legend[{handle}] missing {key}")
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
_MONITOR_STEP = ("monitor_event_id", "monitor_event_id_valid", "monitor_rho", "monitor_rho_valid",
                 "monitor_nu", "monitor_nu_valid", "monitor_epsilon", "monitor_epsilon_valid")
_FORBIDDEN_PREFIXES = ("label_",)
_FORBIDDEN_NAMES = ("cmd_wall_s",)
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
    for name in arrays:
        if name.startswith(_FORBIDDEN_PREFIXES) or name in _FORBIDDEN_NAMES:
            raise ValueError(f"array {name} uses a retired ambiguous name")
    for name in _MONITOR_STEP:
        if name not in arrays:
            raise ValueError(f"missing array {name}")
        if arrays[name].shape[0] != steps + 1:
            raise ValueError(f"{name} must have S+1 rows")
    for name in ("rho", "nu", "epsilon"):
        if arrays[f"monitor_{name}_valid"].shape != arrays[f"monitor_{name}"].shape:
            raise ValueError(f"monitor_{name}_valid must match monitor_{name}")
    times = arrays["step_sim_time"]
    if not np.all(np.diff(times) > 0):
        raise ValueError("step_sim_time must strictly increase")
    dt = float(manifest["timing"]["physics_dt_s"])
    spacing = float(np.spacing(np.float32(times[-1]))) + 1e-9  # the simulator clock is float32
    if not np.allclose(np.diff(times), dt, atol=spacing):
        raise ValueError("every command row must span exactly one physics step")
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
    for name, value in arrays.items():
        # frame_step maps every rendered frame to its physics boundary.
        if name.startswith(("cam_", "frame_")) and value.shape[:1] != (frames,):
            raise ValueError(f"{name} must have F={frames} rows (one per frame_step entry)")
    for key, value in arrays.items():
        if value.dtype == np.dtype(object):
            raise ValueError(f"{key} must not use object dtype")
        if np.issubdtype(value.dtype, np.floating) and not key.endswith("_nan_ok"):
            valid_key = f"{key}_valid"
            data = value if valid_key not in arrays else value[arrays[valid_key]]
            if not np.isfinite(data).all():
                raise ValueError(f"{key} contains non-finite values in valid rows")


__all__ = [
    "EPISODE_ID_PATTERN",
    "ONLINE_FIELDS",
    "PHYSICS_TRANSITION",
    "OUTCOME_STATUSES",
    "PERTURBATION_FAMILIES",
    "RECOVERABILITY",
    "SCHEMA_VERSION",
    "perturbation_families",
    "validate_arrays",
    "validate_manifest",
]
