"""Assemble icgs_episode_v2 records from an AttemptPlan. Simulator-free."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from icgs.data.collection.v3.batch import AttemptPlan, provenance_from_plan
from icgs.data.collection.v3.camera import wrist_camera_pose_world
from icgs.data.collection.v3.perturbations import INVALID_OBSERVATION, SIMULATOR_CRASH, SUCCESS, VALID_FAILURE
from icgs.data.collection.v3.protocol import V3_PROTOCOL
from icgs.data.schemas.episodes_v2 import validate_episode_v2


def transitions_to_v2(transitions: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(transitions):
        if isinstance(item, Mapping):
            achieved = float(item.get("achieved_duration_s", 0.1))
            command = item.get("command") or {}
            rows.append({
                "command": command if isinstance(command, Mapping) else {"duration_s": achieved},
                "achieved_duration_s": achieved,
                "physics_substeps": int(item.get("physics_substeps", 1)),
                "before_boundary": int(item.get("before_boundary", index)),
                "after_boundary": int(item.get("after_boundary", index + 1)),
            })
            continue
        achieved = float(getattr(item, "achieved_duration_s", 0.1))
        command = getattr(item, "command", None)
        cmd = {"duration_s": achieved}
        if command is not None:
            cmd = {
                "T_w_e": getattr(command, "target_w", None),
                "grip": getattr(command, "grip", 0),
                "duration_s": float(getattr(command, "duration_s", achieved)),
            }
        rows.append({
            "command": cmd,
            "achieved_duration_s": achieved,
            "physics_substeps": int(getattr(item, "physics_substeps", 1)),
            "before_boundary": index,
            "after_boundary": index + 1,
        })
    return rows


def empty_snapshot_metadata() -> dict[str, Any]:
    return {
        "sim_time": None,
        "rng_state": None,
        "robot_joint_state": None,
        "object_states": None,
        "articulation_states": None,
        "gripper_state": None,
        "controller_state": None,
        "physics_solver_state": None,
        "scene_asset_digest": None,
        "snapshot_fidelity": None,
        "restore_validation_result": None,
    }


def _terminal_reason(outcome: str) -> str:
    if outcome == SUCCESS:
        return "predicate_satisfied"
    if outcome == VALID_FAILURE:
        return "predicate_failed"
    if outcome == SIMULATOR_CRASH:
        return "simulator_exception"
    return "observation_incomplete"


def assemble_episode_v2(
    *,
    plan: AttemptPlan,
    binding: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    transitions: Sequence[Any],
    outcome: str | None = None,
    result_class: str | None = None,
    robot_states: Sequence[Mapping[str, Any]] | None = None,
    object_states: Sequence[Mapping[str, Any]] | None = None,
    event_states: Sequence[Mapping[str, Any]] | None = None,
    intervention: Mapping[str, Any] | None = None,
    failure_type: str | None = None,
    terminal_reason: str | None = None,
    terminal_t: int | None = None,
) -> dict[str, Any]:
    outcome = outcome or result_class
    if outcome is None:
        raise ValueError("episode outcome is required")
    if outcome in {SIMULATOR_CRASH, INVALID_OBSERVATION}:
        raise ValueError("crash and invalid observation must use assemble_attempt_record")
    v2_transitions = transitions_to_v2(transitions)
    dt = [float(item["achieved_duration_s"]) for item in v2_transitions]
    provenance = provenance_from_plan(plan, binding=binding)
    n = len(v2_transitions)
    last_t = n
    provenance["outcome"] = outcome
    provenance["failure_type"] = None if outcome == SUCCESS else (failure_type or "predicate_unsatisfied")
    provenance["terminal_reason"] = terminal_reason or _terminal_reason(outcome)
    provenance["terminal_t"] = n if terminal_t is None else terminal_t
    provenance["valid_observation_until"] = last_t
    provenance["terminated"] = True
    provenance["truncated"] = int(provenance["terminal_t"]) < n
    merged_intervention = dict(plan.intervention or {})
    if intervention:
        merged_intervention.update(dict(intervention))
    if merged_intervention:
        if merged_intervention.get("application_t") is None and merged_intervention.get("intervention_frame") is not None:
            merged_intervention["application_t"] = merged_intervention.get("intervention_frame")
        provenance["intervention_id"] = merged_intervention.get("intervention_id", provenance.get("intervention_id"))
        provenance["intervention_type"] = merged_intervention.get("intervention_type") or merged_intervention.get("kind")
        provenance["intervention_params"] = merged_intervention.get(
            "intervention_params", provenance.get("intervention_params")
        )
        provenance["intervention_seed"] = merged_intervention.get("intervention_seed")
        provenance["application_scope"] = merged_intervention.get("application_scope")
        provenance["application_t"] = merged_intervention.get("application_t")
        provenance["intervention_frame"] = merged_intervention.get("application_t")
        episode_external = bool(
            merged_intervention.get(
                "episode_has_external_intervention",
                merged_intervention.get("external_intervention"),
            )
        )
        provenance["episode_has_external_intervention"] = episode_external
        provenance["external_intervention"] = episode_external
        provenance["source_episode_id"] = merged_intervention.get("source_episode_id")
        provenance["base_episode_id"] = merged_intervention.get("base_episode_id")
        if merged_intervention.get("application_scope") == "initial_scene":
            provenance["initial_scene_intervention_id"] = merged_intervention.get("intervention_id")
        else:
            provenance["initial_scene_intervention_id"] = None
    robots = list(robot_states) if robot_states is not None else []
    if robot_states is None:
        for item in observations:
            row = {"T_w_e": item.get("T_w_e"), "grip": item.get("grip")}
            if item.get("T_w_e") is not None:
                try:
                    row["T_world_camera"] = wrist_camera_pose_world(item["T_w_e"])
                except Exception:
                    pass
            robots.append(row)
    objects = list(object_states) if object_states is not None else [{} for _ in range(n + 1)]
    record: dict[str, Any] = {
        "schema_version": V3_PROTOCOL.episode_schema_version,
        "provenance": provenance,
        "online_observations": list(observations),
        "transitions": v2_transitions,
        "dt": dt,
        "robot_states": robots,
        "object_states": objects,
        "snapshot": empty_snapshot_metadata(),
    }
    events = binding.get("events") or binding.get("structured_steps")
    if events:
        record["events"] = list(events)
    if event_states is not None:
        record["event_states"] = list(event_states)
    if merged_intervention:
        record["intervention"] = merged_intervention
        record["randomization"] = dict(plan.randomization)
    validate_episode_v2(record)
    return record


def classify_generation_outcome(*, simulator_crash: bool, predicates_ok: bool, observation_valid: bool = True) -> str:
    if simulator_crash:
        return SIMULATOR_CRASH
    if not observation_valid:
        return INVALID_OBSERVATION
    if predicates_ok:
        return SUCCESS
    return VALID_FAILURE


def assemble_attempt_record(
    *,
    attempt_id: str,
    program_id: str,
    outcome: str,
    error: str,
    episode_id: str | None = None,
    episode_kind: str = "nominal",
) -> dict[str, Any]:
    """Crash / invalid observation sidecar. No dynamics training, no episode_id."""
    keep_episode = outcome in {SUCCESS, VALID_FAILURE}
    return {
        "attempt_id": attempt_id,
        "episode_id": episode_id if keep_episode else None,
        "program_id": program_id,
        "episode_kind": episode_kind,
        "outcome": outcome,
        "failure_type": None if outcome == SUCCESS else outcome,
        "terminal_reason": error,
        "terminal_t": None,
        "valid_observation_until": None,
        "terminated": True,
        "truncated": True,
        "status": "failed_attempt",
        "schema_version": V3_PROTOCOL.episode_schema_version,
    }


def quarantine_sidecar(
    *,
    program_id: str,
    outcome: str | None = None,
    result_class: str | None = None,
    error: str,
    attempt_id: str | None = None,
    episode_id: str | None = None,
) -> dict[str, Any]:
    outcome = outcome or result_class or SIMULATOR_CRASH
    return assemble_attempt_record(
        attempt_id=attempt_id or f"att-{episode_id or program_id}",
        program_id=program_id,
        outcome=outcome,
        error=error,
        episode_id=episode_id,
    )
