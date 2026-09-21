"""Validation for phase-1 ``icgs_episode_v2`` records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from icgs.data.collection.v3.protocol import V3_PROTOCOL


_REQUIRED_PROVENANCE = (
    "episode_id",
    "program_id",
    "split",
    "scene_seed",
    "asset_instance_id",
    "execution_mode",
    "calibration_id",
    "observation_origin",
    "raw_commands_id",
    "materialized_commands_id",
    "episode_kind",
    "outcome",
)
_OPTIONAL_PROVENANCE = (
    "attempt_id",
    "program_semantics_version",
    "subset",
    "source_lineage_id",
    "asset_family_id",
    "dataset_version",
    "collection_seed",
    "episode_index",
    "camera_profile_id",
    "camera_intrinsics_id",
    "camera_extrinsics_id",
    "action_space_id",
    "orientation_convention",
    "gripper_unit",
    "lighting_profile_id",
    "controller_version",
    "simulator_version",
    "physics_engine_version",
    "predicate_protocol_id",
    "program_manifest_version",
    "execution_source",
    "intervention_id",
    "intervention_type",
    "intervention_params",
    "intervention_seed",
    "intervention_frame",
    "application_scope",
    "application_t",
    "external_intervention",
    "episode_has_external_intervention",
    "source_episode_id",
    "base_episode_id",
    "magnitude_bucket",
    "held_out",
    "failure_type",
    "terminal_reason",
    "terminal_t",
    "valid_observation_until",
    "terminated",
    "truncated",
    "initial_scene_intervention_id",
    "train_subset",
    "scene_signature",
    *V3_PROTOCOL.phase2_reserved_fields,
)
_OUTCOMES = frozenset(V3_PROTOCOL.outcome_classes)
_SPLITS = frozenset({"train", "dev", "test", "development"})
_KINDS = frozenset({"nominal", "perturbed"})
_ONLINE_FIELDS = frozenset({"points", "T_w_e", "grip", "point_valid"})


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def validate_episode_v2(record: Mapping[str, Any]) -> None:
    record = _require_mapping(record, "episode record")
    if record.get("schema_version") != V3_PROTOCOL.episode_schema_version:
        raise ValueError("episode schema_version must be icgs_episode_v2")
    provenance = _require_mapping(record.get("provenance"), "provenance")
    missing = [field for field in _REQUIRED_PROVENANCE if field not in provenance]
    if missing:
        raise ValueError(f"incomplete episode provenance: missing {missing}")
    unknown = set(provenance) - set(_REQUIRED_PROVENANCE) - set(_OPTIONAL_PROVENANCE)
    if unknown:
        raise ValueError(f"unknown episode provenance fields: {sorted(unknown)}")
    if provenance["split"] not in _SPLITS:
        raise ValueError("episode provenance split must be train, dev, or test")
    if provenance["episode_kind"] not in _KINDS:
        raise ValueError("episode_kind must be nominal or perturbed")
    if provenance["outcome"] not in _OUTCOMES:
        raise ValueError("episode outcome must be success, valid_failure, simulator_crash, or invalid_observation")
    if provenance["outcome"] in {"simulator_crash", "invalid_observation"}:
        raise ValueError("crash and invalid observation must be stored as attempt records, not episodes")
    if provenance["observation_origin"] != "measured":
        raise ValueError("executed episode observation_origin must be measured")

    observations = record.get("online_observations")
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise ValueError("online_observations must be a sequence")
    transitions = record.get("transitions")
    if isinstance(transitions, (str, bytes)) or not isinstance(transitions, Sequence) or not transitions:
        raise ValueError("episode transitions must be a nonempty sequence")
    n_actions = len(transitions)
    if len(observations) != n_actions + 1:
        raise ValueError("online observations must cover T+1 boundaries")

    robot_states = record.get("robot_states", record.get("robot_state"))
    object_states = record.get("object_states", record.get("object_state"))
    if robot_states is not None:
        if not isinstance(robot_states, Sequence) or len(robot_states) != n_actions + 1:
            raise ValueError("robot/object states must cover T+1 boundaries")
    if object_states is not None:
        if not isinstance(object_states, Sequence) or len(object_states) != n_actions + 1:
            raise ValueError("robot/object states must cover T+1 boundaries")

    dt_values = record.get("dt")
    if dt_values is None:
        dt_values = [item.get("achieved_duration_s") for item in transitions]
    if not isinstance(dt_values, Sequence) or len(dt_values) != n_actions:
        raise ValueError("dt must have T values")
    for index, (transition, dt) in enumerate(zip(transitions, dt_values)):
        if not isinstance(transition, Mapping):
            raise ValueError(f"transition {index} must be a mapping")
        achieved = transition.get("achieved_duration_s")
        if achieved is None or not np.isfinite(achieved) or achieved < 0:
            raise ValueError("episode transition timing must be nonnegative")
        if not np.isfinite(dt) or abs(float(dt) - float(achieved)) > 1e-9:
            raise ValueError("dt must match achieved_duration_s")
        command = transition.get("command")
        if not isinstance(command, Mapping):
            raise ValueError(f"transition {index} command must be a mapping")

    timestamps = record.get("timestamps")
    if timestamps is not None:
        if not isinstance(timestamps, Sequence) or len(timestamps) != n_actions + 1:
            raise ValueError("timestamps must cover T+1 boundaries")
        previous = None
        for value in timestamps:
            if previous is not None and float(value) < float(previous):
                raise ValueError("timestamps must be monotonic")
            previous = value

    for index, observation in enumerate(observations):
        observation = _require_mapping(observation, f"online observation {index}")
        extra = set(observation) - _ONLINE_FIELDS
        if extra:
            raise ValueError(f"privileged or unknown online field: {sorted(extra)[0]}")
        missing_obs = _ONLINE_FIELDS - set(observation)
        if missing_obs:
            raise ValueError(f"online observation {index} is incomplete")
        points = np.asarray(observation["points"])
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("observation points must be finite Nx3")

    label_shapes = {}
    for name in ("rho", "nu", "epsilon"):
        if name in record:
            value = np.asarray(record[name])
            if value.ndim != 2 or value.shape[0] != n_actions + 1:
                raise ValueError(f"{name} must align to T+1 boundaries")
            if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
                raise ValueError(f"{name} must contain probabilities in [0,1]")
            label_shapes[name] = value.shape
    for name in ("rho", "nu", "epsilon"):
        valid_name = f"{name}_valid"
        if valid_name in record:
            valid = np.asarray(record[valid_name])
            if valid.dtype != np.bool_ or valid.shape != label_shapes.get(name):
                raise ValueError(f"{valid_name} must be boolean and match {name}")
