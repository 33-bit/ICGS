# ICGS parameter reference

The single authoritative default-value file is
[icgs_primary.json](../../src/icgs/configuration/profiles/icgs_primary.json).
This page documents ownership and meaning, not a second copy of numeric defaults.
The [override example](../../configs/icgs/example.json) changes only selected values.
Native IP's Python defaults and hash-bound published profile are **unchanged**.
Implementation and actual checks: [configuration delivery record](../plans/completed/icgs-central-configuration.md).

## Editing and loading

Edit a copy/override for an experiment, not the frozen default profile after
collection. The installed package includes its own default JSON; editing the
checkout default affects an editable install, but a built wheel must be rebuilt.
No checkout-relative file is read implicitly. Python keeps typed validation and
immutability; defaults are not independently maintained in dataclass literals.

```python
from icgs.configuration.method import MethodConfig

config = MethodConfig()  # packaged primary defaults; no model construction
config = MethodConfig.from_file(
    "configs/icgs/example.json",
    overrides={"optimizer.learning_rate": 0.00002},
)
print(config.optimizer.learning_rate)
resolved = config.resolved_config()  # complete JSON-ready config plus SHA256
config_id = config.fingerprint()
```

Precedence is packaged defaults → explicit file → explicit dotted overrides.
Relative paths resolve against the supplied file, including overridden paths.
`from_dict` consumes the inner config object; `from_file` consumes the
`schema_version`/`config` envelope. Resolved output retains all values, including
currently unused component settings. The default native profile remains the
existing original-source identity; choose a compatible published profile explicitly.

Missing/malformed packaged defaults fail instead of falling back to Python numbers.
Unknown keys, duplicate JSON keys, invalid types (including booleans passed as
counts), nonfinite numbers and incompatible combinations fail before model IO.
Nested records are frozen and sequence fields are tuples internally; `to_dict()`
returns independent JSON-ready values. There is no YAML/Hydra framework.
An identity-bearing `resolved_config()` record requires absolute paths (or null);
use `from_file` to resolve relative experiment paths first. Direct `from_dict`
metadata may retain relative paths, but cannot be exported as a resolved record
until those paths are explicitly resolved. Signed resolved records require the
complete configuration; deliberate overrides create a new resolved identity.

## What changing a value means

- **Locked:** visible configuration values constrained by the current implementation,
  e.g. feature width/anchor inventory and fixed native compatibility. Moving them
  to JSON does not authorize architecture variants or arbitrary checkpoint shapes.
- **Tunable:** supported numeric/protocol settings subject to validation. A valid
  configuration is not evidence that every current constructor consumes it.
- **Derived:** compute dimensions/limits from named inputs, not separately editable
  competing values. XYZ3, SE(3)4x4, Rot6 and binary grip are semantic invariants.
- **Gated:** null means a physical setup, seed or resource limit has not been supplied.
  No fabricated workspace, calibration, asset identity or measured tolerance.
  `validate_for(operation)` checks operation-specific prerequisites; this never
  launches or authorizes a workload, and presence alone does not prove a protocol
  file is trusted, exists or has passed its physical gate.

## Current consumption and handoff

Implemented now: resource-backed MethodConfig, strict overlays/validation,
configuration metadata/identity and the parameter inventory. Existing native
configuration remains independent. MethodConfig's prior planning/control/artifact
validation remains in place.

Existing P03/P04 physical modules still have their own fixed constructors;
loading `neural.dropout`, `decoder.patch_radius_m` or other newly exposed values
**does not retroactively reconfigure those objects**. P01 materialization already
accepts explicit h/r/duration arguments; the untracked implementation was preserved.
The per-plan addenda require wiring the resolved sections at composition time and
adding nondefault propagation tests before a consumer is called configurable.
P05–P13 missing models/collectors/trainers remain planned, not implemented by this
configuration work. No setting can replace a missing execution capability.

Observability is the one additional packaged section consumed by the outer
logging adapter. Its values live under `observability` in the same primary JSON;
the typed `ObservabilityConfig` validates mode, levels, queue/capture quotas and
the optional W&B envelope. A logging-only file uses the explicit
`{"schema_version": 1, "observability": {...}}` envelope and does not alter the
native `ExperimentConfig` or published reference projection. Library calls remain
no-op unless a caller explicitly starts a run. The full method fingerprint includes
these logging settings, while `reference_fingerprint` intentionally excludes them.

Runtime components receive typed sections or scalar arguments from outer composition;
never parse JSON or read docs inside forward/step loops. Preserve default state-dict
shapes, ordering and numerical behavior when wiring existing consumers. Do not
silently remove locked constraints merely because a parameter is present in JSON.

## Dimensions and aliases

Shared learned width is `geometry.width`; prior `event.width`, `memory.width`,
`dynamics.width` and `evaluator.width` remain compatibility aliases constrained to
agree. The existing `heads` fields keep their old meaning; evaluator outputs
V/S/Phi are three semantic outputs, not independently selectable ensemble outcomes.
`training.H`, `dynamics.H` and `evaluator.H` are coverage; `planning.H` cannot
exceed their minimum. Native action horizon8/demo-waypoints10 remain native
constraints, not method tuning fields. Old `point_dim`/`descriptor_dim` fields
remain checked compatibility metadata rather than freely adjustable dimensions.

Derive attention head width = width/attention_heads; patch points = grid_side²;
physical token count = anchors+1+memory slots; dynamics token count adds one action;
proprioception13 and command8 from representation semantics; physical GRU input =
width+13+8; demo frame input = width+13; event input =3*width+8; event-head input
=3*width; terminal input =5*width+8. Total collection maxima are derived from
anchors*candidates*trials, never independently adjusted to disagree.

## Provenance and readiness

Full resolved-config SHA256 identifies the entire experiment configuration, **not**
the reference controller. Continue using the existing reference fingerprint with
IP/encoder/task/router/preprocessing/cadence/seed provenance; changes to those
inputs require new reference labels. Evaluator/temperature/stopping/search changes
alter full experiment identity but do not silently redefine that frozen reference.
No hash proves consumption, artifact trust, calibrated probabilities or physical
replay equivalence. The future runner must save `resolved_config()` alongside its
existing artifact manifests before execution; no automatic runner is added here.

The profile keeps collection wall/disk/attempt limits, actual seeds, workspace,
calibration and measured protocol paths null. Fill required fields and obtain the
roadmap's resource/feasibility approval before executing. Length/stress/robot values
are analysis targets in the inventory, not unlocked primary runtime coverage.
Fixed safety/contact defaults are proposed simulator predicates pending physical
validation, not a real-robot safety guarantee.

## Operation-readiness declarations

Every operation requires workspace min/max bounds and a calibration path.
Additional declarations below are checked without reading any artifact or
authorizing a workload:

| Operation | Additional required declarations |
| --- | --- |
| `geometry_training` | dataset manifest, output directory, actual training seeds |
| `task_training` | annotated dataset/asset/split manifests, output directory, actual training seeds, gravity, predicate protocol; no future outcome-bank dependency |
| `collection` | asset/split manifests, controller/replay/predicate protocols, gravity, generator/reset/action seeds, wall/disk/attempt limits |
| `evaluation` | dataset/asset/split manifests, controller/bridge/predicate protocols, reset/action seeds and statistical-bootstrap seeds |

These checks are a metadata preflight, not completeness of the eventual runner's
execution authorization or physical evidence. Actual jobs must still check trusted
artifacts, model availability, dataset/reference compatibility and approval gates.

## Parameter inventory

Every leaf below is relative to the JSON `config` object. Numerical values live
only in the linked JSON; consult it for defaults. In the restriction column,
“validated” means metadata type/range checks, not a claim that downstream tuning
is already implemented. Architecture fields may have additional compatibility
checks as documented by the schema. Array index positions are not separate knobs.

### geometry

Consumer: P03 physical preprocessing/encoder.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `geometry.component_id` | identifier / named choice | known identity |
| `geometry.voxel_size_m` | meters | locked compatibility |
| `geometry.num_anchors` | count or dimensionless setting | locked compatibility |
| `geometry.width` | feature channels | locked compatibility |
| `geometry.point_dim` | count or dimensionless setting | locked compatibility |
| `geometry.neighbors` | count or dimensionless setting | locked compatibility |
| `geometry.ell0_m` | meters | locked compatibility |
| `geometry.num_points` | count or dimensionless setting | locked compatibility |
| `geometry.transformer_layers` | count or dimensionless setting | locked compatibility |
| `geometry.local_hidden_dims` | feature channels | locked compatibility |
| `geometry.fps_start` | identifier / named choice | validated |
| `geometry.tie_break` | identifier / named choice | validated |

### neural

Consumer: P03 shared blocks; P05–P09/P12 neural consumers.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `neural.attention_heads` | count or dimensionless setting | locked compatibility |
| `neural.ffn_width` | feature channels | locked compatibility |
| `neural.dropout` | dimensionless | validated |
| `neural.layer_norm_eps` | dimensionless | validated |
| `neural.geometry_bias_hidden_dim` | feature channels | locked compatibility |
| `neural.activation` | identifier / named choice | validated |
| `neural.pre_norm` | boolean | locked compatibility |

### decoder

Consumer: P03 patch decoder.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `decoder.grid_side` | count or dimensionless setting | locked compatibility |
| `decoder.grid_min` | count or dimensionless setting | validated |
| `decoder.grid_max` | count or dimensionless setting | validated |
| `decoder.patch_radius_m` | meters | validated |
| `decoder.hidden_dims` | feature channels | locked compatibility |

### event

Consumer: P05 segmentation/event encoder; P02 overlap labels.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `event.component_id` | identifier / named choice | known identity |
| `event.width` | feature channels | locked compatibility |
| `event.max_interactions` | count or dimensionless setting | validated |
| `event.landmarks` | count or dimensionless setting | locked compatibility |
| `event.debounce_frames` | count or dimensionless setting | validated |
| `event.translation_boundary_m` | meters | validated |
| `event.rotation_boundary_deg` | degrees | validated |
| `event.max_intervals` | control intervals (unless updates in key) | validated |
| `event.min_segment_intervals` | control intervals (unless updates in key) | validated |
| `event.transformer_layers` | count or dimensionless setting | validated |
| `event.token_hidden_dim` | count or dimensionless setting | validated |
| `event.overlap_threshold` | dimensionless | validated |
| `event.stress_token_cap` | count or dimensionless setting | optional track; not enabled |

### memory

Consumer: P04 physical GRUs.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `memory.component_id` | identifier / named choice | known identity |
| `memory.width` | feature channels | locked compatibility |
| `memory.slots` | count or dimensionless setting | locked compatibility |
| `memory.descriptor_dim` | count or dimensionless setting | locked compatibility |

### tracker

Consumer: P06 task tracker.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `tracker.attention_layers` | count or dimensionless setting | validated |
| `tracker.memory_slots` | count or dimensionless setting | validated |

### router

Consumer: P06 routed reference.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `router.full_context_probability` | dimensionless | validated |
| `router.probability_epsilon` | dimensionless | validated |
| `router.fallback_threshold` | dimensionless | validated |
| `router.neighbor_events_before` | count or dimensionless setting | validated |
| `router.neighbor_events_after` | count or dimensionless setting | validated |
| `router.native_sessions` | identifier / named choice | validated |

### dynamics

Consumer: P07 physical prediction.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `dynamics.component_id` | identifier / named choice | known identity |
| `dynamics.width` | feature channels | locked compatibility |
| `dynamics.heads` | count or dimensionless setting | locked compatibility |
| `dynamics.H` | count or dimensionless setting | validated |
| `dynamics.transformer_layers` | count or dimensionless setting | validated |
| `dynamics.residual_init_std` | dimensionless | validated |
| `dynamics.bootstrap_probability` | dimensionless | validated |

### evaluator

Consumer: P09 continuation/terminal heads.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `evaluator.component_id` | identifier / named choice | known identity |
| `evaluator.width` | feature channels | locked compatibility |
| `evaluator.heads` | count or dimensionless setting | locked compatibility |
| `evaluator.H` | count or dimensionless setting | validated |
| `evaluator.attention_layers` | count or dimensionless setting | validated |
| `evaluator.output_hidden_dim` | count or dimensionless setting | validated |
| `evaluator.terminal_hidden_dims` | feature channels | validated |

### losses

Consumer: P06/P07/P09/P11 objectives; P08 pair filtering.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `losses.cloud_scale_m` | meters | validated |
| `losses.translation_scale_m` | meters | validated |
| `losses.rotation_scale_deg` | degrees | validated |
| `losses.grip_weight` | dimensionless | validated |
| `losses.reconstruction_weight` | dimensionless | validated |
| `losses.outcome_weight` | dimensionless | validated |
| `losses.completion_weight` | dimensionless | validated |
| `losses.terminal_weight` | dimensionless | validated |
| `losses.progress_weight` | dimensionless | validated |
| `losses.branch_pair_weight` | dimensionless | validated |
| `losses.recovery_pair_weight` | dimensionless | validated |
| `losses.suffix_pair_weight` | dimensionless | validated |
| `losses.alignment_weight` | dimensionless | validated |
| `losses.occurrence_weight` | dimensionless | validated |
| `losses.relation_weight` | dimensionless | validated |
| `losses.eligibility_weight` | dimensionless | validated |
| `losses.moving_point_weight` | dimensionless | validated |
| `losses.moving_point_weighting_enabled` | boolean | validated |
| `losses.pair_confidence` | dimensionless | validated |
| `losses.pair_beta_prior_alpha` | dimensionless | validated |
| `losses.pair_beta_prior_beta` | dimensionless | validated |
| `losses.pair_quadrature_atol` | dimensionless | validated |
| `losses.pair_quadrature_rtol` | dimensionless | validated |

### calibration

Consumer: P09 calibration; P11 phase E.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `calibration.initial_temperature` | dimensionless | validated |
| `calibration.optimizer` | dimensionless | validated |
| `calibration.max_iterations` | dimensionless | validated |
| `calibration.line_search` | dimensionless | validated |
| `calibration.precision` | dimensionless | validated |
| `calibration.value_temperature_path` | file-relative path | gated / optional; required per operation |
| `calibration.completion_temperature_path` | file-relative path | gated / optional; required per operation |
| `calibration.event_temperature_path` | file-relative path | gated / optional; required per operation |

### planning

Consumer: P10 search; P12 budget/ablation panel.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `planning.component_id` | identifier / named choice | known identity |
| `planning.h` | count or dimensionless setting | validated |
| `planning.r` | count or dimensionless setting | validated |
| `planning.L` | count or dimensionless setting | validated |
| `planning.H` | count or dimensionless setting | validated |
| `planning.algorithm` | identifier / named choice | validated |
| `planning.widening_coefficient` | count or dimensionless setting | validated |
| `planning.widening_exponent` | dimensionless | validated |
| `planning.uct_exploration` | dimensionless | validated |
| `planning.wall_budgets_s` | seconds | validated |
| `planning.native_call_caps` | count or dimensionless setting | validated |
| `planning.model_interval_cap` | control intervals (unless updates in key) | gated / optional; required per operation |
| `planning.lookahead_ablation` | control intervals (unless updates in key) | validated |
| `planning.commitment_ablation` | control intervals (unless updates in key) | validated |
| `planning.medoid_cloud_scale_m` | meters | validated |
| `planning.medoid_translation_scale_m` | meters | validated |
| `planning.medoid_rotation_scale_deg` | degrees | validated |

### control

Consumer: P01/P13 timed execution.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `control.component_id` | identifier / named choice | known identity |
| `control.dt0` | seconds per fixed target-holding interval | validated |
| `control.clock_track` | identifier / named choice | validated |

### stopping

Consumer: P13 deployed stop gate.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `stopping.threshold` | dimensionless | validated |
| `stopping.consecutive_observations` | count or dimensionless setting | validated |
| `stopping.enabled` | boolean | validated |

### dataset

Consumer: P02 episode views/generator; P08 branch bank.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `dataset.component_id` | identifier / named choice | known identity |
| `dataset.schema_id` | identifier / named choice | known identity |
| `dataset.shard_intervals` | control intervals (unless updates in key) | validated |
| `dataset.manifest_path` | file-relative path | gated / optional; required per operation |
| `dataset.asset_manifest_path` | file-relative path | gated / optional; required per operation |
| `dataset.split_manifest_path` | file-relative path | gated / optional; required per operation |
| `dataset.train_programs` | count or dimensionless setting | validated |
| `dataset.development_programs` | count or dimensionless setting | validated |
| `dataset.test_programs` | count or dimensionless setting | validated |
| `dataset.asset_split_ratios` | dimensionless | validated |
| `dataset.context_demos` | count or dimensionless setting | validated |
| `dataset.seed_demos_per_program` | count or dimensionless setting | validated |
| `dataset.minimum_execution_modes` | count or dimensionless setting | validated |
| `dataset.object_scale_range` | count or dimensionless setting | validated |
| `dataset.episode_mixture` | count or dimensionless setting | validated |
| `dataset.warmup_episode_mixture` | count or dimensionless setting | validated |
| `dataset.target_offset_m` | meters | validated |
| `dataset.target_rotation_offset_deg` | degrees | validated |
| `dataset.grip_timing_offset_intervals` | control intervals (unless updates in key) | validated |
| `dataset.object_shift_m` | meters | validated |
| `dataset.pause_intervals` | control intervals (unless updates in key) | validated |
| `dataset.anchor_strata_ratios` | dimensionless | validated |
| `dataset.matched_suffix_fraction` | dimensionless | validated |
| `dataset.original_bank_fraction` | dimensionless | validated |
| `dataset.branch_prefix_lengths` | control intervals (unless updates in key) | validated |
| `dataset.outcome_horizons` | control intervals (unless updates in key) | validated |

### collection

Consumer: P02/P08 collection orchestration.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `collection.pilot.program_ids` | catalog identifiers | validated |
| `collection.pilot.successful_contexts_per_program` | count or dimensionless setting | validated |
| `collection.pilot.anchor_contexts` | count or dimensionless setting | validated |
| `collection.pilot.candidates_per_anchor` | count or dimensionless setting | validated |
| `collection.pilot.continuations_per_branch` | count or dimensionless setting | validated |
| `collection.pilot.audit_anchors` | count or dimensionless setting | validated |
| `collection.primary.successful_contexts_per_program` | count or dimensionless setting | validated |
| `collection.primary.anchor_contexts` | count or dimensionless setting | validated |
| `collection.primary.candidates_per_anchor` | count or dimensionless setting | validated |
| `collection.primary.continuations_per_branch` | count or dimensionless setting | validated |
| `collection.primary.calibration_anchors` | dimensionless | validated |
| `collection.primary.calibration_candidates` | dimensionless | validated |
| `collection.primary.calibration_trials` | dimensionless | validated |
| `collection.primary.audit_anchors` | count or dimensionless setting | validated |
| `collection.primary.audit_candidates` | count or dimensionless setting | validated |
| `collection.primary.audit_trials` | count or dimensionless setting | validated |
| `collection.max_episode_intervals` | control intervals (unless updates in key) | validated |
| `collection.wall_limit_s` | seconds | gated / optional; required per operation |
| `collection.disk_limit_bytes` | bytes | gated / optional; required per operation |
| `collection.generation_attempt_limit` | dimensionless | gated / optional; required per operation |

### training

Consumer: P11 artifact/coverage owner.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `training.component_id` | identifier / named choice | known identity |
| `training.H` | count or dimensionless setting | validated |
| `training.heads` | count or dimensionless setting | locked compatibility |
| `training.output_dir` | file-relative path | gated / optional; required per operation |

### optimizer

Consumer: P11 training; P12 student.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `optimizer.kind` | identifier / named choice | validated |
| `optimizer.learning_rate` | dimensionless | validated |
| `optimizer.betas` | dimensionless | validated |
| `optimizer.weight_decay` | dimensionless | validated |
| `optimizer.warmup_updates` | count or dimensionless setting | validated |
| `optimizer.minimum_learning_rate` | dimensionless | validated |
| `optimizer.schedule` | identifier / named choice | validated |
| `optimizer.gradient_clip_norm` | dimensionless | validated |

### stages

Consumer: P11 stage schedule/seeds.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `stages.A0.batch_size` | count or dimensionless setting | validated |
| `stages.A0.max_updates` | count or dimensionless setting | validated |
| `stages.A1.batch_size` | count or dimensionless setting | validated |
| `stages.A1.max_updates` | count or dimensionless setting | validated |
| `stages.A1.burnin_intervals` | control intervals (unless updates in key) | validated |
| `stages.A1.supervised_intervals` | control intervals (unless updates in key) | validated |
| `stages.A1.rollout_curriculum` | control intervals (unless updates in key) | validated |
| `stages.B.batch_size` | count or dimensionless setting | validated |
| `stages.B.max_updates` | count or dimensionless setting | validated |
| `stages.B.supervised_intervals` | control intervals (unless updates in key) | validated |
| `stages.D1.batch_size` | count or dimensionless setting | validated |
| `stages.D1.max_updates` | count or dimensionless setting | validated |
| `stages.D1.supervised_intervals` | control intervals (unless updates in key) | validated |
| `stages.D1.rollout_curriculum` | control intervals (unless updates in key) | validated |
| `stages.D2.batch_size` | count or dimensionless setting | validated |
| `stages.D2.max_updates` | count or dimensionless setting | validated |
| `stages.D2.terminal_batch_size` | count or dimensionless setting | validated |
| `stages.D2.maximum_pairs` | count or dimensionless setting | validated |
| `stages.evaluation_interval_updates` | control intervals (unless updates in key) | validated |
| `stages.early_stopping_patience` | count or dimensionless setting | validated |
| `stages.training_seed_count` | count or dimensionless setting | validated |
| `stages.training_seeds` | integer seed(s) | gated / optional; required per operation |
| `stages.generator_seed` | integer seed(s) | gated / optional; required per operation |
| `stages.reset_seed` | integer seed(s) | gated / optional; required per operation |
| `stages.action_seed` | integer seed(s) | gated / optional; required per operation |
| `stages.precision` | identifier / named choice | validated |
| `stages.mixed_precision_attention` | boolean | validated |

### reactive

Consumer: P12 separate B3 student.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `reactive.transformer_layers` | count or dimensionless setting | validated |
| `reactive.head_hidden_dim` | count or dimensionless setting | validated |
| `reactive.translation_scale_m` | meters | validated |
| `reactive.rotation_norm_epsilon` | dimensionless | validated |
| `reactive.outcome_weight_floor` | dimensionless | validated |
| `reactive.diffusion_train_steps` | dimensionless | validated |
| `reactive.diffusion_inference_steps` | dimensionless | validated |
| `reactive.beta_schedule` | dimensionless | validated |
| `reactive.prediction_type` | identifier / named choice | validated |
| `reactive.ddim_eta` | dimensionless | validated |
| `reactive.clip_sample` | boolean | validated |
| `reactive.batch_size` | count or dimensionless setting | validated |
| `reactive.max_updates` | count or dimensionless setting | validated |
| `reactive.outcome_weighted` | boolean | validated |

### action_value

Consumer: P12 direct-Q baseline.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `action_value.gru_width` | feature channels | validated |
| `action_value.head_hidden_dim` | count or dimensionless setting | validated |
| `action_value.terminal_sample_weight` | dimensionless | validated |

### benchmark

Consumer: P02 predicates; P12 evaluation/statistics.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `benchmark.resets_per_composition` | count or dimensionless setting | validated |
| `benchmark.demonstrations` | dimensionless | validated |
| `benchmark.one_demo_resets` | count or dimensionless setting | validated |
| `benchmark.length_track_H` | control intervals (unless updates in key) | optional track; not enabled |
| `benchmark.length_train_stage_range` | count or dimensionless setting | validated |
| `benchmark.primary_train_stage_range` | count or dimensionless setting | validated |
| `benchmark.length_test_stages` | count or dimensionless setting | validated |
| `benchmark.dependency_spans` | count or dimensionless setting | validated |
| `benchmark.diagnostic_candidates` | count or dimensionless setting | validated |
| `benchmark.minimum_locally_successful_candidates` | count or dimensionless setting | validated |
| `benchmark.consequential_gap` | dimensionless | validated |
| `benchmark.bootstrap_resamples` | count or dimensionless setting | validated |
| `benchmark.confidence_level` | dimensionless | validated |
| `benchmark.ece_bins` | count or dimensionless setting | validated |
| `benchmark.multiple_comparison` | identifier / named choice | validated |
| `benchmark.bootstrap_seed` | integer seed(s) | gated / optional; required per operation |
| `benchmark.paired_trial_bootstrap_seed` | integer seed(s) | gated / optional; required per operation |
| `benchmark.paired_trial_bootstrap_resamples` | count or dimensionless setting | validated |
| `benchmark.object_dimension_range_m` | meters | validated |
| `benchmark.drawer_width_range_m` | meters | validated |
| `benchmark.drawer_depth_range_m` | meters | validated |
| `benchmark.aperture_clearance_range_m` | meters | validated |
| `benchmark.grasp_lift_m` | meters | validated |
| `benchmark.grasp_translation_tolerance_m` | meters | validated |
| `benchmark.grasp_rotation_tolerance_deg` | degrees | validated |
| `benchmark.stable_intervals` | control intervals (unless updates in key) | validated |
| `benchmark.placement_speed_m_s` | meters/second | validated |
| `benchmark.container_margin_m` | meters | validated |
| `benchmark.drawer_closed_tolerance_m` | meters | validated |
| `benchmark.final_translation_tolerance_m` | meters | validated |
| `benchmark.final_rotation_tolerance_deg` | degrees | validated |
| `benchmark.unsafe_force_n` | simulator newtons; pilot validation | validated |
| `benchmark.unsafe_force_intervals` | control intervals (unless updates in key) | validated |
| `benchmark.unsafe_penetration_m` | meters | validated |
| `benchmark.unsafe_penetration_intervals` | dimensionless | validated |
| `benchmark.real_robot_resets_per_family` | count or dimensionless setting | optional track; not enabled |

### sensors

Consumer: P01 sensor/control protocol; P02/P08/P13 feasibility gates.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `sensors.workspace_bounds_m` | meters; min/max XYZ | gated / optional; required per operation |
| `sensors.calibration_path` | file-relative path | gated / optional; required per operation |
| `sensors.controller_protocol_path` | file-relative path | gated / optional; required per operation |
| `sensors.replay_tolerances_path` | file-relative path | gated / optional; required per operation |
| `sensors.bridge_tolerances_path` | file-relative path | gated / optional; required per operation |
| `sensors.predicate_protocol_path` | file-relative path | gated / optional; required per operation |
| `sensors.gravity_w` | calibrated gravity direction | gated / optional; required per operation |
| `sensors.foreground_policy` | identifier / named choice | validated |

### numerics

Consumer: P03/P07 geometry; P10 mass checks.

| Key | Units / interpretation | Restriction |
| --- | --- | --- |
| `numerics.rotation_training_clip_margin` | dimensionless | validated |
| `numerics.mass_sum_atol_float32` | dimensionless | validated |
| `numerics.mass_sum_atol_float64` | dimensionless | validated |

Top-level `schema_version` is format identity, and `native_profile` selects an
existing native identity. These are preserved, not hyperparameter search axes.

## Proposal-to-configuration audit

| Proposal topic | Configuration coverage |
| --- | --- |
| Shared blocks and geometry | geometry, neural, decoder, numerics |
| Physical and task history/events | memory, event, tracker; derived dimension rules above |
| Prior and observation/action interface | router, control, geometry; native profile remains separate |
| World model and losses | dynamics, losses, numerics; frozen-versus-trainable phase is algorithm semantics |
| Continuation/terminal/progress and calibration | evaluator, losses, calibration, stopping |
| Search/UCT/compute | planning; exact caching/absorbed-mass semantics are not tunable shortcuts |
| Splits, generation, perturbations and label filtering | dataset, benchmark, sensors, losses |
| Pilot/main/continuation/calibration/audit budgets | collection.pilot, collection.primary, dataset.outcome_horizons |
| Optimizers/curricula/burn-in/early-stop/seeds | optimizer, stages, training |
| Equal-data student/direct Q | reactive, action_value; architecture fields remain pending consumer support |
| Benchmark tolerances/statistics/stress | benchmark, planning ablation arrays, sensors protocol paths |

The immutable [proposal](../proposals/README.md) and method equations remain the
scientific source. Quoted data-size arithmetic, numbers of semantic outcomes,
actual historical checkpoint evidence and test fixture values are not independent
hyperparameters. No exact asset/camera/seed/throughput values were invented to fill
missing empirical information.
