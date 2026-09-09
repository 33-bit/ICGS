"""Frozen, typed metadata sourced from the packaged ICGS primary JSON.

This is deliberately separate from the native :mod:`ExperimentConfig`.  It
validates method metadata before any component or artifact is constructed;
declaring a setting does not implement its planned neural or training consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from .method_schema import TypedSection, defaults, json_shape, strict_json, typed_value


NATIVE_ACTION_HORIZON = 8
NATIVE_DEMO_WAYPOINTS = 10
PRIMARY_HORIZON = 512
KNOWN_NATIVE_PROFILES = frozenset(
    {
        "instant-policy-original-65dc94e",
        "instant_policy_original",
        "instant_policy_published_vv19_119fa871",
    }
)
KNOWN_COMPONENT_IDS = {
    "geometry": "geometry",
    "event": "event",
    "memory": "memory",
    "dynamics": "dynamics",
    "evaluator": "evaluator",
    "planning": "planning",
    "control": "control",
    "dataset": "dataset",
    "training": "training",
}


def _json_value(value: Any, name: str) -> None:
    """Reject values that cannot have come from strict JSON."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{name} must contain only finite JSON numbers")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} has a non-string JSON key")
            _json_value(item, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_value(item, f"{name}[{index}]")
        return
    raise ValueError(f"{name} must contain JSON values, not executable objects")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _component_id(value: Any, section: str) -> str:
    expected = KNOWN_COMPONENT_IDS[section]
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"unknown component ID for {section}: {value!r}")
    return value


def _merge_known(base: dict[str, Any], update: Any, prefix: str) -> None:
    if not isinstance(update, Mapping):
        raise ValueError(f"{prefix or 'config'} must be an object")
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            _merge_known(base[key], value, f"{prefix}{key}.")
        else:
            _json_value(value, f"{prefix}{key}")
            base[key] = value


def _resolve_paths(config: dict[str, Any], directory: Path) -> None:
    for key, value in config.items():
        if isinstance(value, dict):
            _resolve_paths(value, directory)
        elif key.endswith("_path") or key == "output_dir":
            _optional_path(value, key)
            if value is not None:
                if Path(value).is_absolute():
                    continue
                path = Path(value).expanduser()
                config[key] = str(path if path.is_absolute() else (directory / path).resolve())
        elif key.endswith("_paths") and value is not None:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{key} must be an array of paths")
            result = []
            for item in value:
                _optional_path(item, key)
                if item is None:
                    raise ValueError(f"{key} cannot contain null paths")
                if Path(item).is_absolute():
                    result.append(item)
                    continue
                path = Path(item).expanduser()
                result.append(str(path if path.is_absolute() else (directory / path).resolve()))
            config[key] = result


def _optional_path(value: Any, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{name} must be a nonempty path or null")


def _require_absolute_paths(config: Mapping[str, Any], prefix="config") -> None:
    """Resolved records cannot depend on the directory used to reload them."""
    for key, value in config.items():
        name = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            _require_absolute_paths(value, name)
        elif value is not None and (key.endswith("_path") or key == "output_dir"):
            _optional_path(value, name)
            if not Path(value).is_absolute():
                raise ValueError(f"{name} must be absolute in a resolved configuration; "
                                 "resolve an input using MethodConfig.from_file or supply absolute paths")
        elif value is not None and key.endswith("_paths"):
            for index, item in enumerate(value):
                _require_absolute_paths({"item_path": item}, f"{name}[{index}]")


def _default_field(path: str):
    def load():
        value = defaults(MethodConfig)
        for key in path.split("."):
            value = value[key]
        return value
    return field(default_factory=load)


@dataclass(frozen=True)
class GeometryConfig(TypedSection):
    component_id: str = _default_field("geometry.component_id")
    voxel_size_m: float = _default_field("geometry.voxel_size_m")
    num_anchors: int = _default_field("geometry.num_anchors")
    width: int = _default_field("geometry.width")
    point_dim: int = _default_field("geometry.point_dim")
    neighbors: int = _default_field("geometry.neighbors")
    ell0_m: float = _default_field("geometry.ell0_m")
    num_points: int = _default_field("geometry.num_points")
    transformer_layers: int = _default_field("geometry.transformer_layers")
    local_hidden_dims: tuple[int, ...] = _default_field("geometry.local_hidden_dims")
    fps_start: str = _default_field("geometry.fps_start")
    tie_break: str = _default_field("geometry.tie_break")

@dataclass(frozen=True)
class NeuralConfig(TypedSection):
    attention_heads: int = _default_field("neural.attention_heads")
    ffn_width: int = _default_field("neural.ffn_width")
    dropout: float = _default_field("neural.dropout")
    layer_norm_eps: float = _default_field("neural.layer_norm_eps")
    geometry_bias_hidden_dim: int = _default_field("neural.geometry_bias_hidden_dim")
    activation: str = _default_field("neural.activation")
    pre_norm: bool = _default_field("neural.pre_norm")

@dataclass(frozen=True)
class DecoderConfig(TypedSection):
    grid_side: int = _default_field("decoder.grid_side")
    grid_min: float = _default_field("decoder.grid_min")
    grid_max: float = _default_field("decoder.grid_max")
    patch_radius_m: float = _default_field("decoder.patch_radius_m")
    hidden_dims: tuple[int, ...] = _default_field("decoder.hidden_dims")

@dataclass(frozen=True)
class EventConfig(TypedSection):
    component_id: str = _default_field("event.component_id")
    width: int = _default_field("event.width")
    max_interactions: int = _default_field("event.max_interactions")
    landmarks: int = _default_field("event.landmarks")
    debounce_frames: int = _default_field("event.debounce_frames")
    translation_boundary_m: float = _default_field("event.translation_boundary_m")
    rotation_boundary_deg: float = _default_field("event.rotation_boundary_deg")
    max_intervals: int = _default_field("event.max_intervals")
    min_segment_intervals: int = _default_field("event.min_segment_intervals")
    transformer_layers: int = _default_field("event.transformer_layers")
    token_hidden_dim: int = _default_field("event.token_hidden_dim")
    overlap_threshold: float = _default_field("event.overlap_threshold")
    stress_token_cap: int = _default_field("event.stress_token_cap")

@dataclass(frozen=True)
class MemoryConfig(TypedSection):
    component_id: str = _default_field("memory.component_id")
    width: int = _default_field("memory.width")
    slots: int = _default_field("memory.slots")
    descriptor_dim: int = _default_field("memory.descriptor_dim")

@dataclass(frozen=True)
class TrackerConfig(TypedSection):
    attention_layers: int = _default_field("tracker.attention_layers")
    memory_slots: int = _default_field("tracker.memory_slots")

@dataclass(frozen=True)
class RouterConfig(TypedSection):
    full_context_probability: float = _default_field("router.full_context_probability")
    probability_epsilon: float = _default_field("router.probability_epsilon")
    fallback_threshold: float = _default_field("router.fallback_threshold")
    neighbor_events_before: int = _default_field("router.neighbor_events_before")
    neighbor_events_after: int = _default_field("router.neighbor_events_after")
    native_sessions: str = _default_field("router.native_sessions")

@dataclass(frozen=True)
class DynamicsConfig(TypedSection):
    component_id: str = _default_field("dynamics.component_id")
    width: int = _default_field("dynamics.width")
    heads: int = _default_field("dynamics.heads")
    H: int = _default_field("dynamics.H")
    transformer_layers: int = _default_field("dynamics.transformer_layers")
    residual_init_std: float = _default_field("dynamics.residual_init_std")
    bootstrap_probability: float = _default_field("dynamics.bootstrap_probability")

@dataclass(frozen=True)
class EvaluatorConfig(TypedSection):
    component_id: str = _default_field("evaluator.component_id")
    width: int = _default_field("evaluator.width")
    heads: int = _default_field("evaluator.heads")
    H: int = _default_field("evaluator.H")
    attention_layers: int = _default_field("evaluator.attention_layers")
    output_hidden_dim: int = _default_field("evaluator.output_hidden_dim")
    terminal_hidden_dims: tuple[int, ...] = _default_field("evaluator.terminal_hidden_dims")

@dataclass(frozen=True)
class LossesConfig(TypedSection):
    cloud_scale_m: float = _default_field("losses.cloud_scale_m")
    translation_scale_m: float = _default_field("losses.translation_scale_m")
    rotation_scale_deg: float = _default_field("losses.rotation_scale_deg")
    grip_weight: float = _default_field("losses.grip_weight")
    reconstruction_weight: float = _default_field("losses.reconstruction_weight")
    outcome_weight: float = _default_field("losses.outcome_weight")
    completion_weight: float = _default_field("losses.completion_weight")
    terminal_weight: float = _default_field("losses.terminal_weight")
    progress_weight: float = _default_field("losses.progress_weight")
    branch_pair_weight: float = _default_field("losses.branch_pair_weight")
    recovery_pair_weight: float = _default_field("losses.recovery_pair_weight")
    suffix_pair_weight: float = _default_field("losses.suffix_pair_weight")
    alignment_weight: float = _default_field("losses.alignment_weight")
    occurrence_weight: float = _default_field("losses.occurrence_weight")
    relation_weight: float = _default_field("losses.relation_weight")
    eligibility_weight: float = _default_field("losses.eligibility_weight")
    moving_point_weight: float = _default_field("losses.moving_point_weight")
    moving_point_weighting_enabled: bool = _default_field("losses.moving_point_weighting_enabled")
    pair_confidence: float = _default_field("losses.pair_confidence")
    pair_beta_prior_alpha: float = _default_field("losses.pair_beta_prior_alpha")
    pair_beta_prior_beta: float = _default_field("losses.pair_beta_prior_beta")
    pair_quadrature_atol: float = _default_field("losses.pair_quadrature_atol")
    pair_quadrature_rtol: float = _default_field("losses.pair_quadrature_rtol")

@dataclass(frozen=True)
class CalibrationConfig(TypedSection):
    initial_temperature: float = _default_field("calibration.initial_temperature")
    optimizer: str = _default_field("calibration.optimizer")
    max_iterations: int = _default_field("calibration.max_iterations")
    line_search: str = _default_field("calibration.line_search")
    precision: str = _default_field("calibration.precision")
    value_temperature_path: str | None = _default_field("calibration.value_temperature_path")
    completion_temperature_path: str | None = _default_field("calibration.completion_temperature_path")
    event_temperature_path: str | None = _default_field("calibration.event_temperature_path")

@dataclass(frozen=True)
class PlanningConfig(TypedSection):
    component_id: str = _default_field("planning.component_id")
    h: int = _default_field("planning.h")
    r: int = _default_field("planning.r")
    L: int = _default_field("planning.L")
    H: int = _default_field("planning.H")
    algorithm: str = _default_field("planning.algorithm")
    widening_coefficient: float = _default_field("planning.widening_coefficient")
    widening_exponent: float = _default_field("planning.widening_exponent")
    uct_exploration: float = _default_field("planning.uct_exploration")
    wall_budgets_s: tuple[float, ...] = _default_field("planning.wall_budgets_s")
    native_call_caps: tuple[int, ...] = _default_field("planning.native_call_caps")
    model_interval_cap: int | None = _default_field("planning.model_interval_cap")
    lookahead_ablation: tuple[int, ...] = _default_field("planning.lookahead_ablation")
    commitment_ablation: tuple[int, ...] = _default_field("planning.commitment_ablation")
    medoid_cloud_scale_m: float = _default_field("planning.medoid_cloud_scale_m")
    medoid_translation_scale_m: float = _default_field("planning.medoid_translation_scale_m")
    medoid_rotation_scale_deg: float = _default_field("planning.medoid_rotation_scale_deg")

@dataclass(frozen=True)
class ControlConfig(TypedSection):
    component_id: str = _default_field("control.component_id")
    dt0: float = _default_field("control.dt0")
    clock_track: str = _default_field("control.clock_track")

@dataclass(frozen=True)
class StoppingConfig(TypedSection):
    threshold: float = _default_field("stopping.threshold")
    consecutive_observations: int = _default_field("stopping.consecutive_observations")
    enabled: bool = _default_field("stopping.enabled")

@dataclass(frozen=True)
class DatasetConfig(TypedSection):
    component_id: str = _default_field("dataset.component_id")
    schema_id: str = _default_field("dataset.schema_id")
    shard_intervals: int = _default_field("dataset.shard_intervals")
    manifest_path: str | None = _default_field("dataset.manifest_path")
    asset_manifest_path: str | None = _default_field("dataset.asset_manifest_path")
    split_manifest_path: str | None = _default_field("dataset.split_manifest_path")
    train_programs: int = _default_field("dataset.train_programs")
    development_programs: int = _default_field("dataset.development_programs")
    test_programs: int = _default_field("dataset.test_programs")
    asset_split_ratios: tuple[float, float, float] = _default_field("dataset.asset_split_ratios")
    context_demos: tuple[int, ...] = _default_field("dataset.context_demos")
    seed_demos_per_program: int = _default_field("dataset.seed_demos_per_program")
    minimum_execution_modes: int = _default_field("dataset.minimum_execution_modes")
    object_scale_range: tuple[float, float] = _default_field("dataset.object_scale_range")
    episode_mixture: tuple[float, float, float] = _default_field("dataset.episode_mixture")
    warmup_episode_mixture: tuple[float, float] = _default_field("dataset.warmup_episode_mixture")
    target_offset_m: float = _default_field("dataset.target_offset_m")
    target_rotation_offset_deg: float = _default_field("dataset.target_rotation_offset_deg")
    grip_timing_offset_intervals: int = _default_field("dataset.grip_timing_offset_intervals")
    object_shift_m: float = _default_field("dataset.object_shift_m")
    pause_intervals: tuple[int, ...] = _default_field("dataset.pause_intervals")
    anchor_strata_ratios: tuple[float, float, float, float] = _default_field("dataset.anchor_strata_ratios")
    matched_suffix_fraction: float = _default_field("dataset.matched_suffix_fraction")
    original_bank_fraction: float = _default_field("dataset.original_bank_fraction")
    branch_prefix_lengths: tuple[int, ...] = _default_field("dataset.branch_prefix_lengths")
    outcome_horizons: tuple[int, ...] = _default_field("dataset.outcome_horizons")

@dataclass(frozen=True)
class CollectionPilotConfig(TypedSection):
    program_ids: tuple[str, ...] = _default_field("collection.pilot.program_ids")
    successful_contexts_per_program: int = _default_field("collection.pilot.successful_contexts_per_program")
    anchor_contexts: int = _default_field("collection.pilot.anchor_contexts")
    candidates_per_anchor: int = _default_field("collection.pilot.candidates_per_anchor")
    continuations_per_branch: int = _default_field("collection.pilot.continuations_per_branch")
    audit_anchors: int = _default_field("collection.pilot.audit_anchors")

@dataclass(frozen=True)
class CollectionPrimaryConfig(TypedSection):
    successful_contexts_per_program: int = _default_field("collection.primary.successful_contexts_per_program")
    anchor_contexts: int = _default_field("collection.primary.anchor_contexts")
    candidates_per_anchor: int = _default_field("collection.primary.candidates_per_anchor")
    continuations_per_branch: int = _default_field("collection.primary.continuations_per_branch")
    calibration_anchors: int = _default_field("collection.primary.calibration_anchors")
    calibration_candidates: int = _default_field("collection.primary.calibration_candidates")
    calibration_trials: int = _default_field("collection.primary.calibration_trials")
    audit_anchors: int = _default_field("collection.primary.audit_anchors")
    audit_candidates: int = _default_field("collection.primary.audit_candidates")
    audit_trials: int = _default_field("collection.primary.audit_trials")

@dataclass(frozen=True)
class CollectionConfig(TypedSection):
    pilot: CollectionPilotConfig = field(default_factory=CollectionPilotConfig)
    primary: CollectionPrimaryConfig = field(default_factory=CollectionPrimaryConfig)
    max_episode_intervals: int = _default_field("collection.max_episode_intervals")
    wall_limit_s: float | None = _default_field("collection.wall_limit_s")
    disk_limit_bytes: int | None = _default_field("collection.disk_limit_bytes")
    generation_attempt_limit: int | None = _default_field("collection.generation_attempt_limit")

@dataclass(frozen=True)
class TrainingConfig(TypedSection):
    component_id: str = _default_field("training.component_id")
    H: int = _default_field("training.H")
    heads: int = _default_field("training.heads")
    output_dir: str | None = _default_field("training.output_dir")

@dataclass(frozen=True)
class OptimizerConfig(TypedSection):
    kind: str = _default_field("optimizer.kind")
    learning_rate: float = _default_field("optimizer.learning_rate")
    betas: tuple[float, float] = _default_field("optimizer.betas")
    weight_decay: float = _default_field("optimizer.weight_decay")
    warmup_updates: int = _default_field("optimizer.warmup_updates")
    minimum_learning_rate: float = _default_field("optimizer.minimum_learning_rate")
    schedule: str = _default_field("optimizer.schedule")
    gradient_clip_norm: float = _default_field("optimizer.gradient_clip_norm")

@dataclass(frozen=True)
class StagesA0Config(TypedSection):
    batch_size: int = _default_field("stages.A0.batch_size")
    max_updates: int = _default_field("stages.A0.max_updates")

@dataclass(frozen=True)
class StagesA1Config(TypedSection):
    batch_size: int = _default_field("stages.A1.batch_size")
    max_updates: int = _default_field("stages.A1.max_updates")
    burnin_intervals: int = _default_field("stages.A1.burnin_intervals")
    supervised_intervals: int = _default_field("stages.A1.supervised_intervals")
    rollout_curriculum: tuple[int, ...] = _default_field("stages.A1.rollout_curriculum")

@dataclass(frozen=True)
class StagesBConfig(TypedSection):
    batch_size: int = _default_field("stages.B.batch_size")
    max_updates: int = _default_field("stages.B.max_updates")
    supervised_intervals: int = _default_field("stages.B.supervised_intervals")

@dataclass(frozen=True)
class StagesD1Config(TypedSection):
    batch_size: int = _default_field("stages.D1.batch_size")
    max_updates: int = _default_field("stages.D1.max_updates")
    supervised_intervals: int = _default_field("stages.D1.supervised_intervals")
    rollout_curriculum: tuple[int, ...] = _default_field("stages.D1.rollout_curriculum")

@dataclass(frozen=True)
class StagesD2Config(TypedSection):
    batch_size: int = _default_field("stages.D2.batch_size")
    max_updates: int = _default_field("stages.D2.max_updates")
    terminal_batch_size: int = _default_field("stages.D2.terminal_batch_size")
    maximum_pairs: int = _default_field("stages.D2.maximum_pairs")

@dataclass(frozen=True)
class StagesConfig(TypedSection):
    A0: StagesA0Config = field(default_factory=StagesA0Config)
    A1: StagesA1Config = field(default_factory=StagesA1Config)
    B: StagesBConfig = field(default_factory=StagesBConfig)
    D1: StagesD1Config = field(default_factory=StagesD1Config)
    D2: StagesD2Config = field(default_factory=StagesD2Config)
    evaluation_interval_updates: int = _default_field("stages.evaluation_interval_updates")
    early_stopping_patience: int = _default_field("stages.early_stopping_patience")
    training_seed_count: int = _default_field("stages.training_seed_count")
    training_seeds: tuple[int, ...] | None = _default_field("stages.training_seeds")
    generator_seed: int | None = _default_field("stages.generator_seed")
    reset_seed: int | None = _default_field("stages.reset_seed")
    action_seed: int | None = _default_field("stages.action_seed")
    precision: str = _default_field("stages.precision")
    mixed_precision_attention: bool = _default_field("stages.mixed_precision_attention")

@dataclass(frozen=True)
class ReactiveConfig(TypedSection):
    transformer_layers: int = _default_field("reactive.transformer_layers")
    head_hidden_dim: int = _default_field("reactive.head_hidden_dim")
    translation_scale_m: float = _default_field("reactive.translation_scale_m")
    rotation_norm_epsilon: float = _default_field("reactive.rotation_norm_epsilon")
    outcome_weight_floor: float = _default_field("reactive.outcome_weight_floor")
    diffusion_train_steps: int = _default_field("reactive.diffusion_train_steps")
    diffusion_inference_steps: int = _default_field("reactive.diffusion_inference_steps")
    beta_schedule: str = _default_field("reactive.beta_schedule")
    prediction_type: str = _default_field("reactive.prediction_type")
    ddim_eta: float = _default_field("reactive.ddim_eta")
    clip_sample: bool = _default_field("reactive.clip_sample")
    batch_size: int = _default_field("reactive.batch_size")
    max_updates: int = _default_field("reactive.max_updates")
    outcome_weighted: bool = _default_field("reactive.outcome_weighted")

@dataclass(frozen=True)
class ActionValueConfig(TypedSection):
    gru_width: int = _default_field("action_value.gru_width")
    head_hidden_dim: int = _default_field("action_value.head_hidden_dim")
    terminal_sample_weight: float = _default_field("action_value.terminal_sample_weight")

@dataclass(frozen=True)
class BenchmarkConfig(TypedSection):
    resets_per_composition: int = _default_field("benchmark.resets_per_composition")
    demonstrations: int = _default_field("benchmark.demonstrations")
    one_demo_resets: int = _default_field("benchmark.one_demo_resets")
    length_track_H: int = _default_field("benchmark.length_track_H")
    length_train_stage_range: tuple[int, int] = _default_field("benchmark.length_train_stage_range")
    primary_train_stage_range: tuple[int, int] = _default_field("benchmark.primary_train_stage_range")
    length_test_stages: tuple[int, ...] = _default_field("benchmark.length_test_stages")
    dependency_spans: tuple[int, ...] = _default_field("benchmark.dependency_spans")
    diagnostic_candidates: int = _default_field("benchmark.diagnostic_candidates")
    minimum_locally_successful_candidates: int = _default_field("benchmark.minimum_locally_successful_candidates")
    consequential_gap: float = _default_field("benchmark.consequential_gap")
    bootstrap_resamples: int = _default_field("benchmark.bootstrap_resamples")
    confidence_level: float = _default_field("benchmark.confidence_level")
    ece_bins: int = _default_field("benchmark.ece_bins")
    multiple_comparison: str = _default_field("benchmark.multiple_comparison")
    bootstrap_seed: int | None = _default_field("benchmark.bootstrap_seed")
    paired_trial_bootstrap_seed: int | None = _default_field("benchmark.paired_trial_bootstrap_seed")
    paired_trial_bootstrap_resamples: int = _default_field("benchmark.paired_trial_bootstrap_resamples")
    object_dimension_range_m: tuple[float, float] = _default_field("benchmark.object_dimension_range_m")
    drawer_width_range_m: tuple[float, float] = _default_field("benchmark.drawer_width_range_m")
    drawer_depth_range_m: tuple[float, float] = _default_field("benchmark.drawer_depth_range_m")
    aperture_clearance_range_m: tuple[float, float] = _default_field("benchmark.aperture_clearance_range_m")
    grasp_lift_m: float = _default_field("benchmark.grasp_lift_m")
    grasp_translation_tolerance_m: float = _default_field("benchmark.grasp_translation_tolerance_m")
    grasp_rotation_tolerance_deg: float = _default_field("benchmark.grasp_rotation_tolerance_deg")
    stable_intervals: int = _default_field("benchmark.stable_intervals")
    placement_speed_m_s: float = _default_field("benchmark.placement_speed_m_s")
    container_margin_m: float = _default_field("benchmark.container_margin_m")
    drawer_closed_tolerance_m: float = _default_field("benchmark.drawer_closed_tolerance_m")
    final_translation_tolerance_m: float = _default_field("benchmark.final_translation_tolerance_m")
    final_rotation_tolerance_deg: float = _default_field("benchmark.final_rotation_tolerance_deg")
    unsafe_force_n: float = _default_field("benchmark.unsafe_force_n")
    unsafe_force_intervals: int = _default_field("benchmark.unsafe_force_intervals")
    unsafe_penetration_m: float = _default_field("benchmark.unsafe_penetration_m")
    unsafe_penetration_intervals: int = _default_field("benchmark.unsafe_penetration_intervals")
    real_robot_resets_per_family: int = _default_field("benchmark.real_robot_resets_per_family")

@dataclass(frozen=True)
class SensorsConfig(TypedSection):
    workspace_bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]] | None = _default_field("sensors.workspace_bounds_m")
    calibration_path: str | None = _default_field("sensors.calibration_path")
    controller_protocol_path: str | None = _default_field("sensors.controller_protocol_path")
    replay_tolerances_path: str | None = _default_field("sensors.replay_tolerances_path")
    bridge_tolerances_path: str | None = _default_field("sensors.bridge_tolerances_path")
    predicate_protocol_path: str | None = _default_field("sensors.predicate_protocol_path")
    gravity_w: tuple[float, float, float] | None = _default_field("sensors.gravity_w")
    foreground_policy: str = _default_field("sensors.foreground_policy")

@dataclass(frozen=True)
class NumericsConfig(TypedSection):
    rotation_training_clip_margin: float = _default_field("numerics.rotation_training_clip_margin")
    mass_sum_atol_float32: float = _default_field("numerics.mass_sum_atol_float32")
    mass_sum_atol_float64: float = _default_field("numerics.mass_sum_atol_float64")
@dataclass(frozen=True)
class MethodConfig(TypedSection):
    schema_version: int = _default_field("schema_version")
    native_profile: str = _default_field("native_profile")
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    event: EventConfig = field(default_factory=EventConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    losses: LossesConfig = field(default_factory=LossesConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    stages: StagesConfig = field(default_factory=StagesConfig)
    reactive: ReactiveConfig = field(default_factory=ReactiveConfig)
    action_value: ActionValueConfig = field(default_factory=ActionValueConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    sensors: SensorsConfig = field(default_factory=SensorsConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)

    def __post_init__(self):
        super().__post_init__()
        self.validate()

    @classmethod
    def from_dict(cls, payload: dict) -> "MethodConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("method config must be a JSON object")
        _json_value(payload, "method config")
        values = typed_value(payload, cls, "method config")
        return values.validate()

    @classmethod
    def from_file(cls, path: str | Path, *, overrides: Mapping[str, Any] | None = None) -> "MethodConfig":
        json_path = Path(path).expanduser().resolve()
        try:
            document = strict_json(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid method config file: {json_path}") from exc
        if not isinstance(document, Mapping):
            raise ValueError("method config document must be an object")
        _json_value(document, "method config document")
        unknown = set(document) - {"schema_version", "config", "config_sha256"}
        if unknown:
            raise ValueError(f"unknown method config document fields: {sorted(unknown)}")
        if type(document.get("schema_version", 1)) is not int or document.get("schema_version", 1) != 1:
            raise ValueError("unsupported method config schema")
        merged = cls().to_dict()
        _merge_known(merged, document.get("config", {}), "config.")
        if "config_sha256" in document:
            typed_value(document.get("config"), cls, "complete resolved config", complete=True)
            _require_absolute_paths(merged)
            _resolve_paths(merged, json_path.parent)
            if cls.from_dict(merged).fingerprint() != document["config_sha256"]:
                raise ValueError("resolved config_sha256 does not match configuration")
        if overrides is not None and not isinstance(overrides, Mapping):
            raise ValueError("overrides must be a mapping of dotted keys")
        for dotted, value in (overrides or {}).items():
            if not isinstance(dotted, str) or not dotted or any(not bit for bit in dotted.split(".")):
                raise ValueError("CLI override keys must be nonempty dotted strings")
            update: dict[str, Any] = value
            for bit in reversed(dotted.split(".")):
                update = {bit: update}
            _merge_known(merged, update, "")
        _resolve_paths(merged, json_path.parent)
        return cls.from_dict(merged)

    def validate(self) -> "MethodConfig":
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("unsupported method config schema_version")
        if not isinstance(self.native_profile, str) or self.native_profile not in KNOWN_NATIVE_PROFILES:
            raise ValueError(f"unknown native profile: {self.native_profile!r}")

        geometry = self.geometry
        if type(geometry) is not GeometryConfig:
            raise ValueError("geometry must be GeometryConfig")
        _component_id(geometry.component_id, "geometry")
        if geometry.num_anchors != 128 or geometry.width != 256 or geometry.point_dim != 3:
            raise ValueError("geometry width/anchors/point dimension are incompatible")
        if geometry.neighbors != 32:
            raise ValueError("geometry neighbor count is incompatible")
        _finite_positive(geometry.voxel_size_m, "geometry.voxel_size_m")
        _finite_positive(geometry.ell0_m, "geometry.ell0_m")
        if geometry.voxel_size_m != 0.005 or geometry.ell0_m != 1.0:
            raise ValueError("geometry preprocessing scale is incompatible")

        event = self.event
        if type(event) is not EventConfig or event.width != 256:
            raise ValueError("event width is incompatible")
        _component_id(event.component_id, "event")
        _positive_int(event.max_interactions, "event.max_interactions")
        if event.max_interactions > 30 or event.landmarks != 2:
            raise ValueError("event interaction/landmark inventory is incompatible")

        memory = self.memory
        if type(memory) is not MemoryConfig or memory.width != 256 or memory.slots != 2:
            raise ValueError("memory width/slot inventory is incompatible")
        _component_id(memory.component_id, "memory")
        if memory.descriptor_dim != 8:
            raise ValueError("memory descriptor width is incompatible")

        for name, section, section_type in (("dynamics", self.dynamics, DynamicsConfig),
                                            ("evaluator", self.evaluator, EvaluatorConfig)):
            if type(section) is not section_type:
                raise ValueError(f"{name} section is invalid")
            _component_id(section.component_id, name)
            if section.width != 256 or section.heads != 3:
                raise ValueError(f"{name} width/head inventory is incompatible")
            if not 1 <= _positive_int(section.H, f"{name}.H") <= PRIMARY_HORIZON:
                raise ValueError(f"{name} horizon coverage is incompatible")

        planning = self.planning
        if type(planning) is not PlanningConfig:
            raise ValueError("planning section is invalid")
        _component_id(planning.component_id, "planning")
        for name, value in (("planning.h", planning.h), ("planning.r", planning.r),
                            ("planning.L", planning.L), ("planning.H", planning.H)):
            _positive_int(value, name)
        if not 1 <= planning.r <= planning.h <= NATIVE_ACTION_HORIZON:
            raise ValueError("commit/edge horizon is incompatible")
        if planning.L < planning.h or planning.H > PRIMARY_HORIZON:
            raise ValueError("planning horizon/lookahead is incompatible")
        if planning.H > self.dynamics.H or planning.H > self.evaluator.H or planning.H > self.training.H:
            raise ValueError("planning horizon exceeds trained horizon coverage")

        if type(self.control) is not ControlConfig:
            raise ValueError("control section is invalid")
        _component_id(self.control.component_id, "control")
        _finite_positive(self.control.dt0, "control.dt0")
        if type(self.dataset) is not DatasetConfig:
            raise ValueError("dataset section is invalid")
        _component_id(self.dataset.component_id, "dataset")
        if self.dataset.schema_id != "icgs_episode_v1":
            raise ValueError("unknown dataset schema")
        _positive_int(self.dataset.shard_intervals, "dataset.shard_intervals")
        _optional_path(self.dataset.manifest_path, "dataset.manifest_path")
        if type(self.training) is not TrainingConfig:
            raise ValueError("training section is invalid")
        _component_id(self.training.component_id, "training")
        _optional_path(self.training.output_dir, "training.output_dir")
        if self.training.heads != 3 or not 1 <= _positive_int(self.training.H, "training.H") <= PRIMARY_HORIZON:
            raise ValueError("training head/horizon inventory is incompatible")
        self._validate_extended()
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-shaped metadata; no executable serialization is used."""
        self.validate()
        return json_shape(self)

    def fingerprint(self) -> str:
        """SHA256 of all resolved metadata, distinct from reference-controller identity."""
        serialized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                                allow_nan=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def resolved_config(self) -> dict[str, Any]:
        """Return a reloadable record; callers own persistence and artifact IO."""
        payload = self.to_dict()
        _require_absolute_paths(payload)
        return {"schema_version": self.schema_version, "config": payload,
                "config_sha256": self.fingerprint()}

    @property
    def decoded_point_count(self) -> int:
        return self.geometry.num_anchors * self.decoder.grid_side ** 2

    @property
    def event_token_capacity(self) -> int:
        """Maximum retained event tokens per demonstration, including landmarks."""
        return self.event.max_interactions + self.event.landmarks

    def derived_dimensions(self) -> dict[str, int]:
        """Compute tensor dimensions from configuration and fixed semantic arities."""
        self.validate()
        width = self.geometry.width
        # Proprioception: translation XYZ, Rot6, observed gripper, gravity XYZ.
        proprioception = self.geometry.point_dim + 2 * self.geometry.point_dim + 1 + self.geometry.point_dim
        physical_tokens = self.geometry.num_anchors + 1 + self.memory.slots
        return {
            "attention_head_width": width // self.neural.attention_heads,
            "physical_tokens": physical_tokens,
            "dynamics_tokens": physical_tokens + 1,
            "patch_points": self.decoder.grid_side ** 2,
            "decoded_points": self.decoded_point_count,
            "physical_input": width + proprioception + self.memory.descriptor_dim,
            "demo_frame_input": width + proprioception,
            "event_input": 3 * width + 2 * self.geometry.point_dim + 2,
            "event_heads_input": 3 * width,
            "terminal_input": 5 * width + self.memory.descriptor_dim,
        }

    def validate_for(self, operation: str) -> "MethodConfig":
        """Check declared setup only, never read empirical artifacts or start jobs.

        A successful check does not certify calibration quality, file existence,
        measured feasibility, model availability, or authorization to launch.
        """
        required = {
            "geometry_training": ("dataset.manifest_path", "training.output_dir", "stages.training_seeds"),
            "task_training": ("dataset.manifest_path", "dataset.asset_manifest_path",
                              "dataset.split_manifest_path", "training.output_dir",
                              "stages.training_seeds", "sensors.gravity_w",
                              "sensors.predicate_protocol_path"),
            "collection": ("dataset.asset_manifest_path", "dataset.split_manifest_path",
                           "sensors.controller_protocol_path", "sensors.replay_tolerances_path",
                           "sensors.predicate_protocol_path", "sensors.gravity_w",
                           "stages.generator_seed", "stages.reset_seed", "stages.action_seed",
                           "collection.wall_limit_s", "collection.disk_limit_bytes",
                           "collection.generation_attempt_limit"),
            "evaluation": ("dataset.manifest_path", "dataset.asset_manifest_path",
                           "dataset.split_manifest_path", "sensors.controller_protocol_path",
                           "sensors.bridge_tolerances_path", "sensors.predicate_protocol_path",
                           "stages.reset_seed", "stages.action_seed", "benchmark.bootstrap_seed",
                           "benchmark.paired_trial_bootstrap_seed"),
        }
        if not isinstance(operation, str) or operation not in required:
            raise ValueError(f"unknown method operation: {operation!r}")
        self.validate()
        missing = []
        for path in ("sensors.workspace_bounds_m", "sensors.calibration_path", *required[operation]):
            value = self
            for key in path.split("."):
                value = getattr(value, key)
            if value is None:
                missing.append(path)
        if missing:
            raise ValueError(f"{operation} requires missing setup: {', '.join(missing)}")
        return self

    def _validate_extended(self) -> None:
        values = {}

        def flatten(obj, prefix=""):
            for key, value in obj.items():
                name = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flatten(value, name)
                else:
                    values[name] = value

        flatten(json_shape(self))
        zero_ints = {"optimizer.warmup_updates", "router.neighbor_events_before",
                     "router.neighbor_events_after", "stages.A1.burnin_intervals",
                     "dataset.grip_timing_offset_intervals"}
        nonnegative = {"neural.dropout", "optimizer.weight_decay", "optimizer.minimum_learning_rate",
                       "reactive.ddim_eta", "planning.uct_exploration"}
        probabilities = {
            "neural.dropout", "event.overlap_threshold", "router.full_context_probability",
            "router.probability_epsilon", "router.fallback_threshold", "dynamics.bootstrap_probability",
            "losses.pair_confidence", "stopping.threshold", "dataset.matched_suffix_fraction",
            "dataset.original_bank_fraction", "reactive.outcome_weight_floor", "benchmark.consequential_gap",
            "benchmark.confidence_level", "planning.widening_exponent",
        }
        signed = {"decoder.grid_min", "decoder.grid_max", "sensors.gravity_w", "sensors.workspace_bounds_m"}
        for name, value in values.items():
            if value is None or isinstance(value, bool):
                continue
            if name.endswith("_path") or name.endswith("output_dir"):
                _optional_path(value, name)
            elif isinstance(value, str):
                if not value.strip():
                    raise ValueError(f"{name} must be nonempty")
            elif type(value) is int:
                minimum = 0 if name in zero_ints or name.endswith("_seed") else 1
                if value < minimum:
                    raise ValueError(f"{name} must be >= {minimum}")
            elif isinstance(value, float):
                if not isfinite(value):
                    raise ValueError(f"{name} must be finite")
                if name in signed:
                    continue
                zero_allowed = name in nonnegative or name.endswith("_weight") or name in probabilities
                if value < 0 or (value == 0 and not zero_allowed):
                    raise ValueError(f"{name} must be {'nonnegative' if zero_allowed else 'positive'}")
                if name in probabilities and not 0 <= value <= 1:
                    raise ValueError(f"{name} must be a probability in [0, 1]")
            elif isinstance(value, list) and name not in signed:
                if not value:
                    raise ValueError(f"{name} must be a nonempty array")
                for item in value:
                    if isinstance(item, str):
                        if not item.strip():
                            raise ValueError(f"{name} requires nonempty values")
                    elif item < 0 or (item == 0 and name != "stages.training_seeds"
                                      and "ratios" not in name and "mixture" not in name
                                      and name != "optimizer.betas"):
                        raise ValueError(f"{name} contains invalid nonpositive values")

        choices = {
            "geometry.fps_start": {"lexicographic"}, "geometry.tie_break": {"ordered_index"},
            "neural.activation": {"gelu"}, "router.native_sessions": {"independent_by_demo_count"},
            "calibration.optimizer": {"lbfgs"}, "calibration.line_search": {"strong_wolfe"},
            "calibration.precision": {"float64"}, "planning.algorithm": {"mcts", "shooting", "rerank"},
            "control.clock_track": {"paused", "live"}, "optimizer.kind": {"adamw"},
            "optimizer.schedule": {"cosine"}, "stages.precision": {"float32"},
            "reactive.beta_schedule": {"squaredcos_cap_v2"}, "reactive.prediction_type": {"epsilon"},
            "benchmark.multiple_comparison": {"holm"},
            "sensors.foreground_policy": {"declared_objects_and_blockers"},
        }
        for name, allowed in choices.items():
            if values[name] not in allowed:
                raise ValueError(f"{name} is unknown; expected one of {sorted(allowed)}")

        # These are accepted architectural invariants, not fallback defaults.
        locked = {"geometry.num_points": 2048, "geometry.transformer_layers": 2,
                  "geometry.local_hidden_dims": [64, 128], "neural.attention_heads": 8,
                  "neural.ffn_width": 1024, "neural.geometry_bias_hidden_dim": 32,
                  "neural.pre_norm": True, "decoder.grid_side": 4,
                  "decoder.hidden_dims": [256, 128]}
        for name, expected in locked.items():
            if values[name] != expected:
                raise ValueError(f"{name} is incompatible with the fixed primary architecture")
        if self.geometry.width % self.neural.attention_heads:
            raise ValueError("neural.attention_heads must divide geometry.width")
        if self.decoded_point_count != self.geometry.num_points:
            raise ValueError("decoder grid/anchor count does not match geometry.num_points")
        if self.decoder.grid_min >= self.decoder.grid_max:
            raise ValueError("decoder grid_min must be less than grid_max")
        if self.event.min_segment_intervals > self.event.max_intervals:
            raise ValueError("event.min_segment_intervals exceeds max_intervals")
        if self.event.stress_token_cap < self.event_token_capacity:
            raise ValueError("event.stress_token_cap does not cover the event token inventory")
        if self.neural.dropout >= 1:
            raise ValueError("neural.dropout must be less than 1")
        for name in ("numerics.rotation_training_clip_margin", "numerics.mass_sum_atol_float32",
                     "numerics.mass_sum_atol_float64"):
            if not 0 < values[name] < 1:
                raise ValueError(f"{name} must be strictly between 0 and 1")
        for name in ("router.probability_epsilon", "router.fallback_threshold", "losses.pair_confidence",
                     "benchmark.confidence_level", "reactive.outcome_weight_floor"):
            if not 0 < values[name] < 1:
                raise ValueError(f"{name} must be strictly between 0 and 1")
        if not all(0 <= beta < 1 for beta in self.optimizer.betas):
            raise ValueError("optimizer.betas must be in [0, 1)")
        if self.optimizer.minimum_learning_rate > self.optimizer.learning_rate:
            raise ValueError("optimizer.minimum_learning_rate exceeds learning_rate")
        for stage in (self.stages.A0, self.stages.A1, self.stages.B, self.stages.D1, self.stages.D2):
            if self.optimizer.warmup_updates >= stage.max_updates:
                raise ValueError("optimizer.warmup_updates must be smaller than every stage.max_updates")
        if self.optimizer.warmup_updates >= self.reactive.max_updates:
            raise ValueError("optimizer.warmup_updates must be smaller than reactive.max_updates")
        for name in ("asset_split_ratios", "episode_mixture", "warmup_episode_mixture", "anchor_strata_ratios"):
            ratio = getattr(self.dataset, name)
            if not all(0 <= value <= 1 for value in ratio) or abs(sum(ratio) - 1) > 1e-12:
                raise ValueError(f"dataset.{name} must sum to 1 and contain probabilities")
        for name, value in values.items():
            if isinstance(value, list) and ("_range" in name or name.endswith("pause_intervals")):
                if len(value) != 2 or value[0] > value[1]:
                    raise ValueError(f"{name} must be an ordered lower/upper pair")
            if name.endswith("rollout_curriculum"):
                if any(left >= right for left, right in zip(value, value[1:])):
                    raise ValueError(f"{name} must be strictly increasing")
                parent = self.stages.A1 if name.startswith("stages.A1") else self.stages.D1
                if max(value) > parent.supervised_intervals:
                    raise ValueError(f"{name} exceeds supervised_intervals")
        if any(demo not in (1, 2) for demo in self.dataset.context_demos) or self.benchmark.demonstrations not in (1, 2):
            raise ValueError("dataset/benchmark demonstration counts must be 1 or 2")
        if any(commit > NATIVE_ACTION_HORIZON for commit in self.planning.commitment_ablation):
            raise ValueError("planning.commitment_ablation exceeds the native horizon")
        seeds = self.stages.training_seeds
        if seeds is not None and (len(seeds) != self.stages.training_seed_count or len(set(seeds)) != len(seeds)):
            raise ValueError("stages.training_seeds must contain training_seed_count distinct seeds")
        if self.reactive.diffusion_inference_steps > self.reactive.diffusion_train_steps:
            raise ValueError("reactive.diffusion_inference_steps exceeds training schedule")
        if self.benchmark.minimum_locally_successful_candidates > self.benchmark.diagnostic_candidates:
            raise ValueError("benchmark.minimum_locally_successful_candidates exceeds candidate count")
        if self.sensors.workspace_bounds_m is not None:
            lower, upper = self.sensors.workspace_bounds_m
            if any(lo >= hi for lo, hi in zip(lower, upper)):
                raise ValueError("sensors.workspace_bounds_m must have lower < upper on every axis")
        if self.sensors.gravity_w is not None and not any(self.sensors.gravity_w):
            raise ValueError("sensors.gravity_w must be nonzero")


__all__ = [
    "ActionValueConfig", "BenchmarkConfig", "CalibrationConfig", "CollectionConfig",
    "CollectionPilotConfig", "CollectionPrimaryConfig", "ControlConfig", "DatasetConfig",
    "DecoderConfig", "DynamicsConfig", "EventConfig", "EvaluatorConfig", "GeometryConfig",
    "LossesConfig", "MemoryConfig", "MethodConfig", "NeuralConfig", "NumericsConfig",
    "OptimizerConfig", "PlanningConfig", "ReactiveConfig", "RouterConfig", "SensorsConfig",
    "StagesConfig", "StagesA0Config", "StagesA1Config", "StagesBConfig", "StagesD1Config",
    "StagesD2Config", "StoppingConfig", "TrackerConfig", "TrainingConfig",
]
