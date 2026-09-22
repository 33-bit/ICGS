"""Frozen identifiers for the canonical ICGS generation contract.

Phase 1 covers geometry, temporal physical representation, initial dynamics
and basic event/task tracking.  Reference-policy, recovery, branch and
value data are phase 2 and are not declared here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationProtocol:
    dataset_version: str = "icgs-primary-v3"
    program_manifest_version: int = 3
    episode_schema_version: str = "icgs_episode_v2"
    composition_protocol_id: str = "icgs-composition-primary-v2"
    controller_protocol_id: str = "rlbench-timed-ik-v2"
    minimum_execution_modes: int = 1
    train_programs: int = 20
    development_programs: int = 4
    test_programs: int = 12
    train_on: str = "train_core"
    validation: str = "train_val"
    evaluation: tuple[str, ...] = ("development", "test")
    preserves: str = "primary_v2"
    phase: str = "1"
    camera_profile_id: str = "rlbench-wrist-depth-v1"
    lighting_profile_id: str = "rlbench-base-lighting-v1"
    execution_mode: str = "scripted_waypoint_v1"
    execution_modes: tuple[str, ...] = ("scripted_waypoint_v1",)
    layout_version: int = 3
    pilot_program_ids: tuple[str, ...] = (
        "T01",
        "T02",
        "T03",
        "T04",
        "T07",
        "T09",
        "T10",
        "T11",
        "T12",
        "T13",
        "T17",
    )
    required_train_families: tuple[str, ...] = (
        "basic-manipulation",
        "container-access",
        "packing",
        "grasp-fit",
        "orientation",
        "park-retrieve",
        "gated-access",
    )
    warmup_mixture: tuple[float, float] = (0.7, 0.3)
    mixture_measured_on: str = "training_transitions"
    mixture_views: tuple[str, ...] = ("D_temporal", "D_dyn")
    train_nominal_successes_per_program: int = 200
    train_perturbed_attempts_per_program: int = 80
    perturbed_per_kind: int = 16
    eval_contexts_per_composition: int = 100
    eval_perturbed_attempts_per_program: int = 20
    eval_perturbation_seed_offset: int = 1000003
    eval_perturbation_scale: float = 0.85
    perturbation_inner_fraction: float = 0.7
    eval_held_out_position_bin: int = 3

    def position_bin_rule(self) -> dict[str, Any]:
        reserved = min(self.eval_held_out_position_bin, self.position_strata - 1)
        return {
            "n": self.position_strata,
            "train": [index for index in range(self.position_strata) if index != reserved],
            "eval": [reserved],
            "eval_held_out_position_bin": reserved,
        }

    def perturbation_bounds(self) -> dict[str, Any]:
        return {
            "action_pose_offset": {"translation_m": self.target_offset_m, "yaw_deg": self.target_rotation_offset_deg},
            "gripper_timing": {"intervals": self.grip_timing_offset_intervals},
            "object_displacement": {"shift_m": self.object_shift_m},
            "blocker_insertion": {"declared": True},
            "pause_hold": {"intervals": list(self.pause_intervals)},
            "magnitude_buckets": {
                "inner": {"min_inclusive": 0.0, "max_exclusive": self.perturbation_inner_fraction},
                "outer": {"min_inclusive": self.perturbation_inner_fraction, "max_inclusive": 1.0},
            },
            "position_bins": self.position_bin_rule(),
        }
    program_semantics_version: str = "v3"
    action_space_id: str = "ee_pose_xyzw_grip_v1"
    orientation_convention: str = "xyzw"
    gripper_unit: str = "open_1_closed_0"
    simulator_version: str = "coppeliasim-4.1"
    physics_engine_version: str = "rlbench-default"
    predicate_protocol_id: str = "rlbench-nearcondition-v1"
    phase2_reserved_fields: tuple[str, ...] = (
        "policy_id",
        "policy_version",
        "parent_episode_id",
        "anchor_id",
        "snapshot_id",
        "branch_id",
        "candidate_id",
        "candidate_index",
        "continuation_group_id",
        "trial_id",
        "trial_seed",
        "recovery_id",
        "recovery_budget",
        "recovery_outcome",
        "snapshot_fidelity",
        "restore_validation_result",
    )
    quota_attempt_cap_multiplier: int = 5
    train_core_fraction: float = 0.8
    inter_object_gap_m: float = 0.02
    waypoint_delta_m: float = 0.02
    lighting_profile_ids: tuple[str, ...] = (
        "rlbench-base-lighting-v1",
        "rlbench-dim-lighting-v1",
        "rlbench-high-key-lighting-v1",
    )
    views: tuple[str, ...] = ("D_geom", "D_temporal", "D_dyn", "D_task")
    perturbation_kinds: tuple[str, ...] = (
        "action_pose_offset",
        "gripper_timing",
        "object_displacement",
        "blocker_insertion",
        "pause_hold",
    )
    outcome_classes: tuple[str, ...] = (
        "success",
        "valid_failure",
        "simulator_crash",
        "invalid_observation",
    )
    position_strata: int = 4
    yaw_strata: int = 3
    scale_strata: int = 3
    pose_duplicate_threshold_m: float = 0.004
    target_offset_m: float = 0.02
    target_rotation_offset_deg: float = 10.0
    grip_timing_offset_intervals: int = 1
    object_shift_m: float = 0.03
    pause_intervals: tuple[int, int] = (2, 10)
    collection_seed: int = 20260920
    pilot_nominal_per_program: int = 2
    pilot_perturbed_per_program: int = 2
    physics_passed_program_ids: tuple[str, ...] = (
        "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T10",
        "T11", "T12", "T13", "T14", "T15", "T16", "T17", "T18", "T19", "T20",
        "V01", "V02", "V03", "V04",
        "P1", "P2", "P3", "P4",
        "G1", "G2", "G3", "G4",
        "R1", "R2", "R3", "R4",
    )
    place_ik_z_shortfall_m: float = 0.033
    # Distance from the Panda tool-tip command point to the open contact face
    # used by the physical push primitive.
    push_contact_offset_m: float = 0.045
    predicate_success_m: float = 0.01


GENERATION_PROTOCOL = GenerationProtocol()
