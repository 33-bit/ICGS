# Pre-migration classification: generic ICGS vs custom generator

Date: 2026-09-30. Status: dated audit evidence for the
[RoboHiMan backbone migration](../plans/active/robohiman-backbone-migration.md)
and [ADR 0016](../decisions/0016-robohiman-backbone.md). It classifies the
repository at `e76d98e` (plus the user's uncommitted training-export work,
which is left untouched) before any RoboHiMan refactor.

Labels: **A** generic ICGS logic to preserve and reuse; **B** specific to the
custom 36-program generator (T01–T20/V01–V04/P/G/R); **C** mixed, where the
environment-specific part must become replaceable. FACT = read from source;
ASSUMPTION = inferred.

## A. Generic ICGS logic (preserve, reuse)

| Owner | Why generic | Reuse in RoboHiMan path |
| --- | --- | --- |
| `contracts/method.py` (`TimedCommand`, `TimedObservation`, `ExecutedTransition`, `TimedEnvironment`, `TaskMonitor`, `ReplayProvider`) | FACT: numpy-only, no simulator/program IDs | Stage-2 interfaces; D_dyn adapters |
| `contracts/records.py` (`Observation`) | World XYZ, `T_w_e`, grip 0/1 | Online observation target of the camera adapter |
| `environments/rlbench/replay.py` (`validate_replay_report`, `REQUIRED_REPLAY_FIELDS`) | FACT: validates exact/deterministic/approximate claims without restoring | Snapshot/replay gate reports are validated with it |
| `environments/rlbench/timed.py` (`TimedRLBenchAdapter`, `TimedController`) | Controller-seam timed stepping | Stage-2 pi_ref execution through a RoboHiMan controller (future) |
| `data/preprocessing/physical.py` (5 mm voxel / 2048 / FPS) | FACT: takes world points + valid mask | Downstream of RoboHiMan depth→XYZ |
| `data/preprocessing/native.py`, `events.py` | IP-native preprocessing; observable demo segmentation | Unchanged |
| `data/schemas/episodes.py` (`icgs_episode_v1` online-field allowlist) | Keeps privileged fields out of model input | Same allowlist enforced by the Stage-1 view builder |
| `data/datasets/episodes.py` (`validate_split_lineage`, `causal_prefix`) | Pure split/causal helpers | Split-lineage rule reused in the overlap audit |
| `state/*`, `models/*`, `algorithms/*`, `policies/*`, `execution/*` | Method/model runtime, environment independent | Unchanged |
| `observability/*`, `configuration/*`, `artifacts/*` | Infrastructure | Unchanged |

## B. Specific to the custom generator (deprecate from primary path, keep recoverable)

| Owner | Custom content |
| --- | --- |
| `data/collection/programs.py`, `generation/steps.py`, `scenes.py`, `compiler.py`, `primitive_compiler.py`, `layout.py`, `layout_clearance.py` | 36-program catalog, handle-block scenes, layouts, compiled task modules |
| `artifacts/composition/approved_composition_manifest.json`, `approved_manifest.py`, `generation/manifest.py` | 20/4/12 program split |
| `generation/expert.py`, `rlbench_attempt.py`, `attempt_prep.py` | Scripted IK expert and attempt materialization for generated scenes |
| `generation/protocol.py` | Frozen `icgs-primary-v3` identities, quotas, pilot IDs, 20/4/12 counts |
| `generation/quota.py`, `batch.py`, `diversity.py`, `lighting.py`, `simulator_randomization.py`, `camera.py` | Custom quota/seed/randomization; `camera.py` holds an *unmeasured* placeholder intrinsics profile |
| `generation/distributed_*`, `capacity_probe.py`, `scripts/generation_*` | Distributed run/publish control plane for the custom archive |
| `tests/regression/generation_program_audit.py`, `tests/test_36_*`, `test_generation_*` | Custom-suite regression |

## C. Mixed (environment-specific part must become replaceable)

| Owner | Generic core | Custom coupling (FACT) | Migration action |
| --- | --- | --- | --- |
| `generation/task_labels.py` | rho historical / nu current / epsilon prerequisite derivation | Relations are inferred from custom step postcondition *strings* and object-marker distance | Replaced for RoboHiMan by a predicate-trace label deriver that consumes simulator predicates; the custom deriver stays for the legacy suite |
| `generation/perturbations.py` | Keeps perturbation kind separate from outcome class | Kinds and bounds tied to `GENERATION_PROTOCOL` | Vocabulary re-expressed source-neutrally in `icgs.data.stage1` |
| `generation/episode_record.py`, `data/schemas/episode_records.py` (`icgs_episode_v2`) | T+1 observations, T transitions, measured timing, outcome in provenance | Provenance requires `program_id`, `scene_seed`, custom protocol IDs; outcome set lacks timeout/invalid execution | New source-neutral `icgs_stage1_episode_v1` record; v2 unchanged |
| `generation/episode_archive.py`, `datasets/generation_*`, `datasets/training_export.py` | Lossless chunked NPZ, content hashes, pointer views | Manifest identity is the custom dataset/protocol; 2.3k lines bound to v2 fields | Not reused for the smoke store; convergence deferred until the Stage-1 record is accepted |
| `environments/rlbench/controller.py`, `teleop_oracle.py` | Pose/quaternion conversion, workspace filtering | Upstream-RLBench (stepjam 02720bba) scene assumptions | RoboHiMan uses its own pinned fork; not imported by the new adapter |
| `data/training_layout.py`, `data/archives.py` | Atomic writes, hashing | v1/v2 layout | Unchanged |

## Observed facts that shape the migration

- RoboHiMan expert demos come from `TaskEnvironment.get_demos(live_demos=True)`
  → `Scene.get_demo`, which steps a Reflexxes path via
  `arm.set_joint_target_positions(q)` each 0.05 s physics step, actuates the
  gripper by joint target velocities, and *kinematically attaches* graspables
  with `gripper.grasp()`. The `JointVelocity`/`Discrete` action mode passed to
  the environment is not used by demo generation.
- `Scene.get_demo` does not record frames during gripper actuation by default
  (`record_gripper_closing=False`), producing 19–20 physics-step gaps
  (probe on `vps-a`, 2026-09-30).
- The RoboHiMan checkout only vendors 13 RLBench and 5 PyRep overlay files. They
  are byte-identical to RVT's submodule pins (RLBench `buttomnutstoast@587a6a0`,
  PyRep `stepjam@231a1ac`) except one added `print`. The 2026-09-29
  compatibility smoke used a different RLBench (`stepjam@02720bba`) with a
  signature shim; its results are not evidence for the pinned environment.
- `put_in_without_close` success is only “item detected inside drawer”; there
  is no occurrence-order predicate in any RoboHiMan success condition.

## Not changed by this audit

No custom module is deleted. The custom generator stays runnable and documented
as legacy/reference, and its archives keep their identities.
