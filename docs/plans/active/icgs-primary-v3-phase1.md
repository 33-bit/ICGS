# Plan: ICGS primary v3 phase 1 generation

Status: active
Owner: repository owner / assigned implementer
Motivation and observable acceptance criteria: Produce a new `primary_v3` dataset sufficient to train geometry, temporal physical representation, initial dynamics and a basic event/task tracker, without overwriting `primary_v2`. Phase 1 is complete only when train episodes have full provenance/calibration, training programs pass catalog/compiler/event parity, train/test asset overlap is 0, nominal and perturbation transitions exist, and `D_geom` / `D_temporal` / `D_dyn` / `D_task` can be built as pointers.

## Current and target states

Current collection is `scripted_waypoint_v1` expert demos on composition protocol v1. The compiler looks up hardcoded routines instead of parsing catalog steps. Phase 1 target: freeze v3 protocol, compile events from structured steps, independent episode seeds, five perturbation kinds, episode schema `icgs_episode_v2`, and four pointer views.

Out of scope until phase 2: `pi_ref_v1`, recovery bank, anchors/branches/continuations, `D_recovery`, `D_value`.

## Invariants and scope

- Keep `artifacts/composition/approved_composition_manifest.json` and `icgs_primary.json` unchanged.
- Train on `train_core` only. `train_val` is validation. Development and test are evaluation, including held-out perturbed attempts.
- Catalog `ordered_steps` = compiled events = observed events = postcondition labels.
- `T` actions, `T+1` observations and robot/object states, `T` dt values from `achieved_duration_s`.
- Single field `outcome`; crashes/invalid observations are attempt records without `episode_id`.
- Collection quota is 200 successful nominal contexts + 80 valid perturbed attempts. 70/30 is a training sampler on `D_temporal`/`D_dyn` transitions.
- `push` is a contact push, not a silent pick-place remap.
- Train scenes split `train_core` / `train_val` must be disjoint. `D_dyn` `external_intervention` is true only at timestep/event-scoped interventions.

## Phases and progress

- [x] Freeze phase-1 protocol (`dataset_version=icgs-primary-v3`, manifest 3, `icgs_episode_v2`, composition `icgs-composition-primary-v2`, controller `rlbench-timed-ik-v2`, one execution mode).
- [x] Structured catalog + step-driven compiler with mismatch fixes (push, articulation open/close, close predicates, nominal regrasp, future dependencies).
- [x] Independent seeds, stratified sampling, duplicate rejection, five perturbation kinds, four outcome classes.
- [x] Episode v2 schema, camera calibration profile, phase-1 layout and pointer views.
- [x] Local unittest `tests/test_primary_v3.py` and `tests/test_v3_expert.py`.
- [x] Scripted expert plans for push / articulation / grasp / pause_hold; generator `--protocol v3`.
- [x] Pilot attempt plan written (`artifacts/composition/v3_phase1_pilot_attempts.json`, 44 attempts).
- [x] Colab CPU provisioned: CoppeliaSim 4.1 + PyRep + RLBench.
- [x] Built 11 pilot task models (T01–T04, T07, T09–T13, T17).
- [x] 11-program physics pilot complete (no simulator crashes; all T+1 timelines OK).
- [x] Predicate success: T01, T04. Valid failure: T02, T03, T07, T09, T10, T11, T12, T13, T17.
- [ ] Expert place is XY-correct but +3.2 cm in Z vs target (T02 object z=0.823, target z=0.790). Do not scale place programs until Z is inside 1 cm.
- [ ] Dev/test evaluation contexts and view materialization from collected episodes.

## Validation and experiment strategy

Cheap L0: `PYTHONPATH=src python3 -B -m unittest discover -s tests -p 'test_primary_v3.py' -v` and `test_v3_batch.py`.
Reviewer-facing generation description: [docs/audits/2026-09-20-icgs-v3-generation.md](../../audits/2026-09-20-icgs-v3-generation.md).
Colab CPU session `icgs-primary-v3-cpu` is retained. Do not launch the 200-success + 80-attempt scale until asked.

## Compatibility

`primary_v2` archives, `icgs_episode_v1`, and MethodConfig defaults stay as-is. v3 is an additive protocol and layout.

## Risks, open questions and recovery

Articulation is a 1-DoF handle-displacement proxy, not a CoppeliaSim drawer mesh. If measured joint assets are required, T04/T12 become `deferred`. Camera intrinsics are RLBench defaults marked unmeasured until a calibration capture exists.
