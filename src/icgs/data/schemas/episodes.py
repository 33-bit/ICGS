"""Validation for materialized, causal ``icgs_episode_v1`` records.

This module validates the in-memory record boundary only; it does not implement
the plan's JSON-manifest/NPZ-shard persistence format.  Provenance remains
separate from online observations, so program identity cannot be a model input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np

from icgs.contracts.method import ExecutedTransition


_ONLINE_FIELDS = frozenset({"points", "T_w_e", "grip", "point_valid"})
_REQUIRED_ONLINE_FIELDS = _ONLINE_FIELDS
_EPISODE_FIELDS = frozenset({"schema_version", "provenance", "online_observations", "transitions"})
_PROVENANCE_FIELDS = frozenset({
    "episode_id",
    "program_id",
    "source_lineage_id",
    "asset_family_id",
    "split",
    "calibration_id",
    "observation_origin",
    "raw_commands_id",
    "materialized_commands_id",
})
_SPLITS = frozenset({"train", "dev", "test"})
_OBSERVATION_ATOL = 1e-6


def validate_online_fields(mapping: Mapping[str, Any]) -> None:
    """Reject any field that could carry privileged or unapproved online data.

    This function intentionally checks only the allowlist.  Value shape, dtype,
    and numerical validity are checked separately by :func:`validate_episode`.
    """

    if not isinstance(mapping, Mapping):
        raise ValueError("online fields must be a mapping")
    for key in mapping:
        if key not in _ONLINE_FIELDS:
            raise ValueError(f"privileged or unknown online field: {key}")


def _array(value: Any, *, name: str, shape: tuple[int | None, ...], boolean: bool = False) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.dtype == np.dtype(object):
        raise ValueError(f"{name} must not use object dtype")
    if value.ndim != len(shape) or any(expected is not None and actual != expected
                                       for actual, expected in zip(value.shape, shape)):
        expected_shape = "[" + ",".join("N" if item is None else str(item) for item in shape) + "]"
        raise ValueError(f"{name} must have shape {expected_shape}")
    if boolean:
        if value.dtype != np.dtype(bool):
            raise ValueError(f"{name} must have boolean dtype")
        return value
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} must have a floating dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_online_observation(observation: Mapping[str, Any], index: int) -> None:
    validate_online_fields(observation)
    missing = _REQUIRED_ONLINE_FIELDS - set(observation)
    if missing:
        raise ValueError(f"online observation {index} is incomplete: missing {sorted(missing)}")
    points = _array(observation["points"], name=f"online observation {index}.points", shape=(None, 3))
    valid = _array(observation["point_valid"], name=f"online observation {index}.point_valid",
                   shape=(points.shape[0],), boolean=True)
    if points.shape[0] == 0 or not valid.any():
        raise ValueError(f"online observation {index} must contain a valid point")
    pose = _array(observation["T_w_e"], name=f"online observation {index}.T_w_e", shape=(4, 4))
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise ValueError(f"online observation {index}.T_w_e must be a valid SE(3) pose")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"online observation {index}.T_w_e must be a valid SE(3) pose")
    grip = observation["grip"]
    if isinstance(grip, (bool, np.bool_)) or not isinstance(grip, Real) or not np.isfinite(grip) or grip not in (0, 1):
        raise ValueError(f"online observation {index}.grip must be exactly 0 or 1")


def _validate_provenance(provenance: Mapping[str, Any]) -> None:
    if not isinstance(provenance, Mapping):
        raise ValueError("episode provenance must be a mapping")
    unknown = set(provenance) - _PROVENANCE_FIELDS
    missing = _PROVENANCE_FIELDS - set(provenance)
    if unknown:
        raise ValueError(f"unknown episode provenance fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"incomplete episode provenance: missing {sorted(missing)}")
    for field in _PROVENANCE_FIELDS - {"split", "observation_origin"}:
        if not isinstance(provenance[field], str) or not provenance[field].strip():
            raise ValueError(f"episode provenance {field} must be a nonempty string")
    if provenance["split"] not in _SPLITS:
        raise ValueError("episode provenance split must be train, dev, or test")
    if provenance["observation_origin"] != "measured":
        raise ValueError("executed episode observation_origin must be measured")


def _validate_transition_observation(
    online: Mapping[str, Any], transition_observation: Any, *, boundary: int
) -> None:
    """Require the online materialization to represent the executed observation.

    P00 observations have no padding mask.  Therefore their cloud must equal the
    valid rows selected by ``point_valid``.  A fixed absolute tolerance of 1e-6
    permits a float32 materialized array to represent P00's float64-owned values;
    relative tolerance is deliberately zero so magnitude cannot widen it.
    """

    expected_points = online["points"][online["point_valid"]]
    actual_points = np.asarray(transition_observation.points)
    if actual_points.shape != expected_points.shape or not np.allclose(
        actual_points, expected_points, rtol=0.0, atol=_OBSERVATION_ATOL
    ):
        raise ValueError(f"transition observation at boundary {boundary} disagrees on points or point_valid")
    actual_pose = np.asarray(transition_observation.T_w_e)
    if actual_pose.shape != (4, 4) or not np.allclose(
        actual_pose, online["T_w_e"], rtol=0.0, atol=_OBSERVATION_ATOL
    ):
        raise ValueError(f"transition observation at boundary {boundary} disagrees on T_w_e")
    if transition_observation.grip != online["grip"]:
        raise ValueError(f"transition observation at boundary {boundary} disagrees on grip")


def _validate_transitions(transitions: Sequence[Any], observations: Sequence[Mapping[str, Any]]) -> None:
    if not transitions:
        raise ValueError("episode transitions must be nonempty")
    if len(observations) != len(transitions) + 1:
        raise ValueError("online observations must cover every transition boundary")
    previous: ExecutedTransition | None = None
    first_boundary: int | None = None
    for index, transition in enumerate(transitions):
        if not isinstance(transition, ExecutedTransition):
            raise ValueError(f"episode transition {index} must be an ExecutedTransition")
        if transition.after.boundary != transition.before.boundary + 1:
            raise ValueError("episode transitions must have adjacent boundaries")
        if previous is not None:
            if transition.before.boundary != previous.after.boundary:
                raise ValueError("episode transitions must have adjacent boundaries")
            for field in (
                "simulator_timestamp",
                "measured_wall_timestamp",
                "sensor_profile_id",
            ):
                if getattr(transition.before, field) != getattr(previous.after, field):
                    raise ValueError(f"shared boundary metadata disagrees on {field}")
        if first_boundary is None:
            first_boundary = transition.before.boundary
        expected_before = first_boundary + index
        if transition.before.boundary != expected_before:
            raise ValueError("episode transitions must correspond to consecutive observation boundaries")
        if transition.command.duration_s < 0 or transition.achieved_duration_s < 0 or transition.physics_substeps < 0:
            raise ValueError("episode transition timing must be nonnegative")
        if transition.after.simulator_timestamp < transition.before.simulator_timestamp:
            raise ValueError("episode transition simulator timing must be nonnegative")
        if transition.after.measured_wall_timestamp < transition.before.measured_wall_timestamp:
            raise ValueError("episode transition wall timing must be nonnegative")
        _validate_transition_observation(
            observations[index], transition.before.observation, boundary=transition.before.boundary
        )
        _validate_transition_observation(
            observations[index + 1], transition.after.observation, boundary=transition.after.boundary
        )
        previous = transition


def validate_episode(record: Mapping[str, Any]) -> None:
    """Validate a causal, executed ``icgs_episode_v1`` materialized record.

    Metadata and command provenance are deliberately isolated under
    ``provenance``.  The only mappings admitted as online model observations use
    the four causal fields approved in the method contract.
    """

    if not isinstance(record, Mapping):
        raise ValueError("episode record must be a mapping")
    unknown = set(record) - _EPISODE_FIELDS
    missing = _EPISODE_FIELDS - set(record)
    if unknown:
        raise ValueError(f"unknown episode fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"incomplete episode record: missing {sorted(missing)}")
    if record["schema_version"] != "icgs_episode_v1":
        raise ValueError("unsupported episode schema_version")
    _validate_provenance(record["provenance"])
    observations = record["online_observations"]
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise ValueError("online_observations must be a sequence")
    for index, observation in enumerate(observations):
        _validate_online_observation(observation, index)
    transitions = record["transitions"]
    if isinstance(transitions, (str, bytes)) or not isinstance(transitions, Sequence):
        raise ValueError("episode transitions must be a sequence")
    _validate_transitions(transitions, observations)


__all__ = ["validate_episode", "validate_online_fields"]
