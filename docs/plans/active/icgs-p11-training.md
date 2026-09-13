# P11: Staged training, freeze boundaries and resumable artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Train added components in a reproducible phase order without moving the reference value target.

**Architecture:** Training owns optimizer/scheduler/batching/resume state; objective modules own math and models own forward tensors. An explicit phase matrix prevents accidental upstream updates; collection stage C has no optimizer.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Data and training](../../method/data-training.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — Physical repairs (2A), CPU exact resume (2B), and readiness/sampling/gates (2C1) ACCEPTED; Batch 2C2 (final local selection repair) implemented and verified (D1/B/C/D2/E dependency gated; whole-diff audit pending; no overall P11 completion)**. This document is a category C
component of the approved docs-only research migration, not permission to execute it.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00/P02 and each stage's component as it arrives: P03 for A0,
P04/P07 for A1, P05/P06 for B, P08/P09 for D2. Build runner infrastructure
alongside its first consumer, not after search. Actual training requires authorized
pilot data and resource scope; no training launch is implied by this plan.
Read [workflow](../../WORKFLOW.md), [architecture](../../ARCHITECTURE.md),
[baseline](../../baselines/instant_policy.md), [native contract](../../components/policy-data-contract.md),
[research policy](../../RESEARCH.md) and [validation guide](../../../tests/README.md)
before runtime execution. Inspect Git state; preserve unrelated work and user assets.

## Configuration consumption addendum — 2026-09-09

Default values now belong to the [packaged primary JSON](../../../src/icgs/configuration/profiles/icgs_primary.json),
with key/unit/restriction details in the [parameter reference](../../method/parameters.md).
Consumed sections for this plan: **training, optimizer, stages, collection, losses, calibration**.

Use optimizer fields and the current stages.A0/A1/B/D1/D2 record for batches, maximum updates, burn-in and curricula. Use stages seed fields and require actual values at run readiness. Preserve phase trainable-set semantics and save config.resolved_config() beside checkpoints before execution.

Use an explicitly resolved `cfg: MethodConfig` (or injected section) in implementation.
Numeric shapes and test inputs below are baseline examples/compatibility assertions;
they are not a second editable default source. New tunable implementation constants
must be replaced by the matching configuration key. Preserve prior progress and
evidence; this addendum does not certify that the component consumes every new field.

## Global constraints

- Canonical runtime is `src/icgs`; no `ip` shims or wrapped legacy runtime.
- Native public B=1, candidate K, action horizon P=8 and demo waypoints 10 are distinct.
- Added preprocessing is 5 mm/FPS; published native preprocessing stays 10 mm/native.
- Timed method defaults are dt0=0.1 s, h=r=2, L=32 and primary H<=512.
- `pi_ref` is frozen router/IP without learned stopping; deployment stopping is separate.
- All online inputs are causal XYZ/pose/grip plus independent demonstrations.
- Masks exclude invalid padding from pooling/attention/loss; no oracle labels enter models.
- Models own tensors, algorithms own objectives/search, composition owns artifact IO.
- Numerical defaults are IDs, not measured optima. FG evidence cannot be assumed.
- No training, downloads, preprocessing jobs, simulator workloads or robot motion now.

## File and interface ownership

- Create: `src/icgs/training/stages/method.py` — phase/freeze matrix and curricula.
- Create: `src/icgs/training/method.py` — finite runner, sampling and resume.
- Extend: `src/icgs/artifacts/method.py` — stage/optimizer/RNG manifests.
- Test: `tests/test_method_training.py`.

Public capability boundary (local synthetic runner implemented and importable; stages A0 and A1 publicly executable; stages B, C, D1, D2, and E dependency-gated):

```python
trainable_components(stage: str) -> frozenset[str]
rollout_horizon(stage: str, update: int, max_updates: int) -> int
run_method_training(config, components, datasets, *, limits) -> TrainingReport
validate_resume(saved, requested) -> None
```

A0 geometry; A1 geometry+physical GRUs+pilot dynamics; B event/task; C none with frozen reference; D1 dynamics only; D2 evaluators/terminal only; E temperatures only; test none. Changed reference components after C demand new reference ID and recollected outcomes.

### Task 1: Trainable-set enforcement and finite curricula

**Files:** Create `src/icgs/training/stages/method.py`.
**Test owner:** `tests/test_method_training.py`.
**Consumes / produces:** Produces trainable_components and rollout_horizon.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.training.stages.method import trainable_components, rollout_horizon
self.assertEqual(trainable_components('C'), frozenset())
self.assertEqual(trainable_components('D1'), frozenset({'dynamics'}))
self.assertEqual(rollout_horizon('A1', 0, 90), 1)
self.assertEqual(rollout_horizon('A1', 60, 90), 4)
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_training.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
stage_trainable = {'A0': {'geometry'}, 'A1': {'geometry','physical_memory','dynamics'},
                   'B': {'events','task'}, 'C': set(), 'D1': {'dynamics'},
                   'D2': {'evaluation','terminal'}, 'E': {'temperatures'}, 'Test': set()}
for name, module in components.items():
    module.requires_grad_(name in stage_trainable[stage])
# Phase membership above is research semantics, not a hyperparameter.
stage_config = getattr(cfg.stages, stage)  # trainable phases only
optimizer = torch.optim.AdamW(trainable_parameters,
    lr=cfg.optimizer.learning_rate, betas=cfg.optimizer.betas,
    weight_decay=cfg.optimizer.weight_decay)
```

Curricula A1 K1/2/4 in equal update thirds,D1 K2/4/8/16 in quarters; validate range/phase boundaries. Assert actual optimizer parameter IDs equal the permitted set and that frozen tensor hashes remain unchanged after tiny optimizer tests. A1 must train physical memory temporally before B freezes it.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_training.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.


### Task 2: Full-history replay, batches and selected metrics

**Files:** Create `src/icgs/training/method.py`.
**Test owner:** `tests/test_method_training.py`.
**Consumes / produces:** Produces `sample_anchor_horizon(groups, rng)` and `history_slices(boundary, burnin)`; consumes objective APIs.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.training.method import history_slices
prefix, burn, supervised = history_slices(boundary=20, burnin=8)
self.assertEqual(prefix, slice(0, 12))
self.assertEqual(burn, slice(12, 20))
self.assertEqual(supervised.start, 20)
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_training.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
# Replay reset->burn-in start under current weights without gradients.
# Detach hidden state, then process burn-in/supervised segment as declared.
anchor = rng.choice(anchor_context_groups)
H = rng.choice(anchor.valid_horizons)
# Targets detached; frozen modules may transmit continuous input gradients.
# Build u_previous from stored before pose+command, never the target observation.
```

Honor supervised intervals16 and batch8 sequence phases; A0batch64,D2batch128 anchors+64 terminals+<=128 pairs. AdamW1e-4,betas.9/.999,wd.01,warmup2000,cosine1e-5,clip1; evaluate5000,patience5,max50k/100k/100k/200k/100k. Lexicographic validation metric priorities are fixed before training; A1 ranking is NOT RUN until available.

Add a recorded transition whose before pose is identity, target +0.10m and
successor +0.06m; assert the training input descriptor is +0.10m while achieved
translation target is +0.06m. This must match P04/P07 online/imagined semantics.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_training.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.


### Task 3: Exact resume, reference identity and calibration separation

**Files:** Complete runner and extend `src/icgs/artifacts/method.py`.
**Test owner:** `tests/test_method_training.py`.
**Consumes / produces:** Produces validate_resume; persists weights,optimizer,scheduler,stage,update,resolved config,dataset/reference IDs and Python/NumPy/torch/CUDA RNG.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.artifacts.method import validate_resume
with self.assertRaisesRegex(ValueError, 'reference|manifest'):
    saved = dict(schema_version=1, stage='D2', reference_id='a',\
                 dataset_id='d', config_id='c')
    validate_resume(saved, dict(saved, reference_id='b'))
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_training.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
for key in ('schema_version','stage','reference_id','dataset_id','config_id'):
    if saved[key] != requested[key]:
        raise ValueError(f'incompatible resume manifest: {key}')
# Construct permitted optimizer, load all states, restore RNG before next sample.
# Write checkpoint atomically; preserve previous accepted checkpoint.
```

Use tiny deterministic two-step uninterrupted versus1+resume tests to compare weights,next batch,scheduler and RNG, not a real training job. E uses calibration-only temperatures; retain identity on failed fit. Report3 training seeds separately, class-balance inverse inclusion weights and hard-example growth; main-scale launch stays FG gated.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_training.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

## Acceptance, resource limits and evidence

L1 tests parameter membership, gradients, curriculum boundaries, causal history, dataset partition checks and synthetic exact resume. Model training acceptance requires authorized staged pilots and recorded metric/freeze evidence; maximum-update defaults are not granted compute.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_method_training.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

Do not edit native Lightning runner/objective/sampler or reuse its effectively unbounded defaults. No automatic WandB/network logger, primary training or test-set checkpoint selection. Restore previous compatible artifact on interruption, never quietly restart with mismatched labels.

On failure, retain the failed evidence and isolate the new component/config ID.
Do not overwrite native checkpoints, relabel collected outcomes or drop difficult samples.
Resume only from a compatible explicit artifact/manifest; changed reference lineage
requires new outcome labels, not a cache workaround. Return to the preceding accepted
phase if an FG fails and request a scoped protocol decision.

## Execution evidence

- Manager baseline verification (HEAD 273b7b397ff4f3b16b06a46102d97e160843d5d2; tracked tree clean; untracked user `output/` untouched):
  - Host info: macOS 15.7.2 arm64 (as measured by manager).
    - Note: Darwin 25.3.0, Apple M1 Pro, and `.venv` L0 command were worker-reported diagnostic details.
  - Python 3.14.4 (system `python3`):
    - `python3 -B scripts/validate_fast.py` -> PASS (19 executed, 0 skipped, L0 only).
    - `python3 -B -S scripts/validate_fast.py` -> PASS (19 executed, 0 skipped, L0 only).

- Initial 11-test suite (Preserved as limited helper/config coverage only; DOES NOT satisfy runner-level acceptance):
  - Initial tests verified helper functions (`history_slices`, `sample_anchor_horizon`, `build_transition_descriptor_and_achieved`), nominal config defaults, and basic dictionary freeze/matrix lookup.
  - Retraction of false claims:
    - Previous claim of Tasks 1–3 being complete is RETRACTED.
    - Previous runner fell back to dummy parameter decay `sum(p**2)` without calling real objective APIs or performing full history replay.
    - Previous Task 3 RED was a post-implementation test of `ValueError` and is NOT claimed as a pre-implementation RED proof.
    - Previous exact resume test compared parameters with `1e-6` tolerance rather than exact equality, changed total `max_updates` between runs, and omitted scheduler, sampler, and next-batch checks.
    - Previous calibration test passed `datasets=None` and returned a fake scalar without FP64 LBFGS NLL fitting.

- Fix Round 1 (Addressing Sol Audit Essential Findings 1–5):
  - Finding 1: ADDRESSED.
    - Removed dummy `sum(p**2)` and arbitrary callback shortcuts completely from `src/icgs/training/method.py`.
    - Implemented real Stage A1 temporal replay: prefix replay under `torch.no_grad()`, detached hidden state at the boundary, multi-step unrolling using `PhysicalRollout` with `physical_loss` objective.
    - Added regression test `test_runner_a1_temporal_replay_physical_loss_and_gradients`: verifies `physical_loss` computation and non-zero temporal gradients flowing to `physical_memory`.
    - Rejects missing datasets (`datasets=None`) for trainable stages and Stage E with `ValueError`.
    - Added runner seed readiness validation: rejects with `ValueError` when `stages.training_seeds is None` and no explicit `rng` is passed.
    - Enforces configured stage ceiling on `limits["max_updates"]`, strictly rejecting values exceeding stage limits with `ValueError`.
    - Implemented lexicographic evaluation cadence and early stopping with patience counter; verified by `test_runner_lexicographic_evaluation_cadence_and_early_stopping`.
  - Finding 2: ADDRESSED.
    - Enforced freeze boundary and parameter/module pointer checks in `src/icgs/training/stages/method.py`.
    - Validates that all components are `nn.Module` (raises `TypeError` otherwise).
    - Detects parameter and submodule sharing across trainable and frozen roles regardless of dictionary insertion order and raises `ValueError`.
    - Explicitly sets `module.train()` and `requires_grad_(True)` on trainable components; sets `module.eval()` and `requires_grad_(False)` on frozen components (ensuring BatchNorm running statistics and Dropout do not mutate).
    - Validates curriculum division bounds in `rollout_horizon` (`update < max_updates`), strictly rejecting out-of-bounds updates.
    - Regression tests verified RED (4 failures), then GREEN in `MethodTrainingTask1Tests`.
  - Finding 3: ADDRESSED.
    - Extended `validate_resume` in `src/icgs/artifacts/method.py` to require and match `total_updates`.
    - In `src/icgs/training/method.py`, separated configured total schedule (`stage_max_updates` / `limits["max_updates"]`) from temporary pause bound (`stop_after`). Exact resume must execute under the identical total schedule.
    - Checkpoints persist `manifest`, `total_updates`, `update`, `config` (`config.resolved_config()`), component state dicts, optimizer state, scheduler state, complete RNG dict (Python, NumPy, PyTorch, CUDA, runner), `sampler_state`, and `selection_state`.
    - `load_checkpoint` validates state dict keys before mutating live components (raises `KeyError` on missing trainable components without modifying live weights).
    - Verified by `test_uninterrupted_vs_1_plus_resume_exact_identity`: 2 uninterrupted updates vs 1 update + resume to 2 under identical total schedule `max_updates=2` yields matching parameters (`assert_close` at 1e-6, max diff 1.89e-10), identical scheduler state, and identical float32 loss trajectory.
  - Finding 4: ADDRESSED.
    - Stage C: strictly requires `reference_manifest`, validates reference fingerprint via `validate_method_manifest(manifest, config)`, completes 0 updates, does not instantiate an optimizer, and records verified `reference_id` fingerprint.
    - Stage E: strictly requires `datasets`, validates disjointness between calibration and training partitions (raises `ValueError` on overlap), performs deterministic FP64 fitting of 3 positive temperatures ($T_v, T_s, T_e$) using `torch.optim.LBFGS` with `strong_wolfe` line search. On failure or nonfinite inputs, retains initial identity $T_0=1.0$, sets `retained_identity=True`, and records failure reason without crashing.
    - Verified by `test_stage_c_requires_valid_reference_manifest_and_binds_fingerprint`, `test_stage_e_fp64_three_temperatures_lbfgs_fit_and_disjoint_check`, and `test_stage_e_retains_identity_on_nonfinite_or_failed_fit`.
  - Finding 5: ADDRESSED.
    - Accurately recorded manager baseline (macOS 15.7.2 arm64, HEAD 273b7b3), retracted false claims, and documented actual test commands and counts.
    - L0 Fast validators: `python3 -B scripts/validate_fast.py` (19 executed, 0 skipped, PASS) and `python3 -B -S scripts/validate_fast.py` (19 executed, 0 skipped, PASS).
    - L1 Focused test suite: `.venv/bin/python -B -m unittest discover -s tests -p 'test_method_training.py' -v` (22 executed, 0 skipped, PASS).
    - Full method test suite: `.venv/bin/python -B -m unittest discover -s tests -p 'test_method_*.py' -v` (63 executed, 0 skipped, PASS).

- Fix Round 2A (Bounded Physical-Runner Repair; Findings 3 & 4 remain OPEN):
  - Sol Audit 2 Result: Rejected (FAIL / Spec FAIL).
  - Scope: Bounded repair of Findings 1 (physical replay & objectives), 2 (geometry ownership), evaluation mode restoration, and honest evidence. Findings 3 (exact resume) and 4 (reference/calibration) are explicitly DEFERRED and remain OPEN for subsequent dispatches.
  - Finding 1: PARTIALLY ADDRESSED.
    - Implemented real A1 and D1 history reconstruction from reset measured observation (`boundary=0`) and ordered `ExecutedTransition` sequence under current weights.
    - Replay loop re-encodes each measured observation with `encoder` and updates `PhysicalMemory` with descriptor from `action_descriptor(before_pose, trans.command)`. Past observations are NOT predicted from the physical model.
    - [SUPERSEDED by Authority Resolution below] Prefix segment replayed under `torch.no_grad()`; bounded burn-in replayed under `torch.enable_grad()`.
    - Hidden state detached explicitly at the supervision boundary.
    - Subsequent supervised intervals unrolled using `PhysicalRollout` with `physical_loss` (A1) and `rollout_loss` over active `bootstrap_mask` heads (D1).
    - Removed `dyn_val.sum()` fallback completely.
    - Removed invented generic BCE/MSE placeholders for Stages B and D2; early explicit `NotImplementedError` raised until real upstream objectives are available.
    - Documented precise missing upstream objective dependencies:
      - `src/icgs/algorithms/objectives/task.py` (Stage B event/task objectives, owned by P06 Task1C) does not exist.
      - `src/icgs/algorithms/objectives/evaluation.py` (Stage D2 evaluator/terminal objectives, owned by P09) does not exist.
    - [SUPERSEDED by Authority Resolution below] Multi-transition regression test proved indices visited and grad modes; now updated to verify strict `torch.no_grad()` pre-supervision replay and isolated converging prefix influence.
    - Regression test `test_d1_history_replay_frozen_gradient_transmission_and_bootstrap` proves frozen `geometry` and `physical_memory` parameters receive no gradient (`grad is None`) while `dynamics` receives non-zero gradients transmitted through the frozen graph under `rollout_loss` with bootstrap masking.
  - Finding 2: ADDRESSED.
    - Single `geometry` owner provides both `encoder` and `decoder` via `nn.ModuleDict` (or attribute extraction). Removed unknown `geometry_decoder` role.
    - `apply_freeze_boundary` validates all required stage roles before mutating any component, and validates both encoder and decoder submodules for stages A0, A1, D1.
    - Regression test `test_freeze_boundary_optimizer_parameter_ids_and_tensor_invariance` asserts optimizer parameter IDs include both encoder and decoder, asserts non-zero decoder gradients, and verifies parameter updates after `optimizer.step()`.
  - Evaluation: ADDRESSED.
    - `_evaluate_stage` puts all evaluated modules into `.eval()` mode during validation batch processing and restores their previous training modes in a `finally` block even when an exception is raised.
    - Rejects invalid/empty datasets or batches explicitly with `ValueError` (no 0.0 fallback).
    - Note: Scalar `val_loss` does not substitute for full lexicographic selection; remaining lexicographic priorities remain OPEN.
    - Regression test `test_evaluation_temporary_eval_mode_and_restoration_on_success_and_exception` proves eval mode during evaluation, restoration on success and exception, and rejection of invalid/empty datasets.
  - Finding 3 (Exact resume): OPEN (deferred to subsequent dispatch).
  - Finding 4 (Reference/calibration): OPEN (deferred to subsequent dispatch).

- Fix Round 2A Follow-Up & Scoped Physical Runner Repairs:
  - Audit Status: ACCEPTED by Sol auditor. Manager verification: `.venv/bin/python -B -m unittest discover -s tests -p test_method_training.py` PASS 33 executed/0 skipped/5.352s; `git diff --check` PASS; auditor no rerun. Physical 2A repairs closed.
  - Authority Resolution: Sol WITHDREW its request for gradients through burn-in. Per docs/method/data-training.md:219-220 and proposal, boundary detach is preserved; all pre-supervision measured transitions (including burn-in) are replayed strictly under torch.no_grad(); temporal gradients are proven across SUPERVISED intervals, not burn-in.
  - Scoped Repair 1 (Calibrated Gravity Carrier & Fallback Deletion):
    - Reconstruct RESET pose/grip from `transitions[0].before.observation`, but obtain gravity carrier from validated `reset_state.p[:, 10:13]` (no hardcoded `[0, 0, -1]`).
    - Entire missing-reset fallback branch deleted from `_validate_transitions_and_reconstruct_reset`: `reset_state` is strictly REQUIRED as the calibrated gravity carrier.
    - Entry check in `_replay_history_and_detach` and `_validate_transitions_and_reconstruct_reset` requires `isinstance(reset_state, PhysicalState)` with explicit error referencing calibrated gravity carrier before constructing tensors or updating anything.
    - Validates `reset_state.p` as finite 2D tensor with last dimension 13.
    - Replay loop propagates `current_state.p[:, 10:13]` unchanged at each history step.
    - Verified by `test_noncanonical_gravity_preserved_in_reset_and_replay_proprioception` proving noncanonical gravity `[0.25, -0.50, -9.75]` reaches reset and replay proprioception unchanged.
    - Verified by regression test `test_a1_and_d1_reject_missing_reset_state_calibrated_gravity_carrier` proving both A1 and D1 stage paths without `reset_state` (and without state alias) fail immediately with TypeError/ValueError referencing calibrated gravity carrier.
  - Scoped Repair 2 (Strict D1 Provenance & Validity):
    - Removed generic `seed` fallback. Stage D1 accepts only explicit `bootstrap_seed`; generic-only `seed` raises `ValueError`.
    - Requires `episode_id` with `bool(episode_id.strip())`; whitespace-only or empty strings raise `ValueError`.
    - Requires explicitly supplied boolean `valid` mask of shape `[1, supervised_intervals]`; missing or wrong-shape masks raise `ValueError`.
    - Verified by 3 negative assertions added to `test_d1_provenance_and_validity_shape_rejection`.
  - Scoped Repair 3 (A0 Unconditional Chamfer Objective):
    - Removed `hasattr(geometry, "reconstruction_loss")` shortcut completely from `_step_a0_reconstruction`. A0 always executes `encoder -> decoder -> masked_normalized_chamfer_distance`.
    - Verified by `test_a0_always_uses_chamfer_and_ignores_bogus_reconstruction_loss_method` proving that a bogus `reconstruction_loss` method attached to geometry is ignored.
  - Scoped Repair 4 (Preflight & Diagnostics Deduplication; Isolated Prefix Influence):
    - Removed duplicate dataset and seed preflight block from `run_method_training`.
    - Synchronized B/D2 diagnostics in `_execute_training_step` with top-level gates, referencing exact component owners: P06 Task1C deferred `src/icgs/algorithms/objectives/task.py` and P09 `src/icgs/algorithms/objectives/evaluation.py`.
    - Updated `test_history_reconstruction_multi_transition_replay_indices_grad_modes_and_prefix_influence`: uses two continuous histories that converge to identical observations at boundary 4 and share identical supervised transitions 4 and 5, cleanly isolating prefix influence through boundary memory.
  - Honest Status and Evidence:
    - Working tree is DIRTY (modified/untracked P11 files plus user-owned `output/`).
    - Findings 1 (batching/selection details), 3 (exact resume fidelity), 4 (reference/calibration rigor), and P06/P09 integration remain OPEN for subsequent dispatches. No exact resume or full completion is claimed.
    - Real training, pilot datasets, simulator, and robot motion remain NOT RUN.
    - Test Evidence:
      - Fast L0 validators: `python3 -B scripts/validate_fast.py` (19 executed, 0 skipped, PASS in 0.485s) and `python3 -B -S scripts/validate_fast.py` (19 executed, 0 skipped, PASS in 0.485s).
      - L1 Method training suite: `.venv/bin/python -m unittest tests/test_method_training.py` (33 executed, 0 skipped, PASS in 4.661s).
      - L1 World model suite: `.venv/bin/python -m unittest tests/test_world_model.py` (29 executed, 0 skipped, PASS in 1.107s).
      - Full method test suite: `.venv/bin/python -m unittest discover -s tests -p 'test_method_*.py' -v` (74 executed, 0 skipped, PASS in 5.470s).
      - `git diff --check`: PASS (0 whitespace/syntax issues).

- Fix Round 2B & 2B Scoped Repairs — Exact Resume Only (Audit Finding 3 Closed):
  - Audit Status: REVISED per Sol audit findings; all 4 concrete defects addressed; stopped for Sol re-audit.
  - Snapshot Cloning & Full Collaborator Rollback:
    - Addressed defect where module.state_dict() returns tensor references that are mutated in place during load_state_dict.
    - Live module tensors are now cloned ({k: v.clone() for k, v in module.state_dict().items()}), and optimizer, scheduler, sampler state, and RNG are deepcopied before apply begins.
    - On late apply failure (e.g. sampler load_state_dict raises after module/optimizer update), all live module weights/buffers, optimizer state/groups, scheduler, sampler cursor, and next RNG draws are restored bit-exactly.
    - Verified by test_load_checkpoint_apply_failure_rolls_back_all_collaborators using a controlled failure-once sampler after live mutation.
  - Strict Canonical Optimizer ID Sequence & Validated ID Mapping Preflight:
    - In method.py:421-460, compares saved optimizer param_groups params sequences sequences directly against canonical live optimizer.state_dict() group ID/order before loading.
    - Resolves vulnerability where counts/shapes alone allowed same-shaped parameter moments to swap silently across parameters of identical shape (e.g. across distinct layers/modules).
    - Constructs explicit canonical `id_to_param` mapping from `zip(optimizer.param_groups, live_sd_groups)` and iterates via validated actual ID mappings, removing unchecked 0..N iteration.
    - Validates all keys in `saved_opt["state"]` map to known parameter IDs; unknown keys raise KeyError before mutation.
    - Verified by regression check in `test_load_checkpoint_fails_malformed_state_before_mutating_live_components`: swaps same-shaped parameter IDs across distinct modules and proves immediate rejection with ValueError before any mutation of live weights.
  - Unconditional Manifest update and total_updates Enforcement:
    - In method.py:348-368, requires `update` and `total_updates` unconditionally in `saved_manifest`.
    - Enforces strict non-negative/positive integer and range validation (`man_update != saved_update`, `man_total != saved_total_updates`, `man_update <= man_total`).
    - Missing manifest update or total_updates raises KeyError before mutation.
    - Verified by negative regression checks in `test_load_checkpoint_fails_malformed_state_before_mutating_live_components` proving rejection with live weights unmutated.
  - Strict Sampler Preflight & Rejection:
    - Any object implementing sample_batch must provide callable state_dict AND callable load_state_dict before any component or optimizer mutation in both load_checkpoint and run_method_training.
    - Missing state_dict and missing load_state_dict are rejected separately with actionable TypeError. Immutable sequence path is separate and supported.
    - Verified by dedicated negative tests in test_stateful_sampler_contract_and_rejection_of_unrestorable_callable.
  - Required Progress, Manifest Agreement & AdamW Per-Param State Compatibility:
    - Required top-level key "accumulated_loss_history", strictly enforcing all finite values and len(accumulated_loss_history) == update.
    - Manifest update and total_updates must strictly match top-level update and total_updates.
    - Bound AdamW per-parameter preflight: iterates live optimizer param_groups, checks corresponding parameter state in saved_opt["state"], and validates that exp_avg and exp_avg_sq match live parameter shape and dtype, and step is a scalar, before any component or optimizer mutation.
    - Verified by 9 new negative preflight checks in test_load_checkpoint_fails_malformed_state_before_mutating_live_components.
  - Optimizer State Exact Equality in Resume Verification:
    - Added recursive optimizer param_groups and per-parameter state equality (torch.equal on step, exp_avg, exp_avg_sq) between uninterrupted Run A and resumed Run B checkpoint payloads in test_uninterrupted_vs_1_plus_resume_exact_identity.
  - Root Cause Statement on Prior Divergence:
    - Prior numerical divergence (1.89e-10) root cause was NOT DETERMINED / not isolated. No post-hoc cause is invented; the current passing bit-exact test is proof only for the current tested code and environment.
  - CUDA Test Policy:
    - test_cuda_resume_if_available unconditionally raises unittest.SkipTest("CUDA resume continuation fixture is NOT IMPLEMENTED / NOT RUN on this environment") rather than containing a body pass, preventing false-positive test passes.
  - Honest Status and Remaining Limitations:
    - Working tree is DIRTY (modified/untracked P11 files plus user-owned output/).
    - Finding 3 (Exact Resume): REPAIRED and VERIFIED.
    - Findings 1 (batching and lexicographic selection details), 4 (reference/calibration rigor), and upstream P06/P09 objectives remain OPEN for subsequent dispatches.
    - Real training, pilot datasets, simulator, and robot motion remain NOT RUN.
    - Test Evidence:
      - Fast L0 validators: python3 -B scripts/validate_fast.py (19 executed, 0 skipped, PASS in 0.492s) and python3 -B -S scripts/validate_fast.py (19 executed, 0 skipped, PASS in 0.492s).
      - L1 Method training suite: .venv/bin/python -B -m unittest discover -s tests -p test_method_training.py (37 selected, 36 executed PASS, 1 skipped: CUDA in 7.805s).
      - git diff --check: PASS (0 issues).

- Fix Round 2C (Supported-Stage Batching, Selection, Readiness + Honest Upstream Gates) [REJECTED / SUPERSEDED by Round 2C1 Scoped Fix]:
  - Audit Status: REJECTED by Sol auditor. Superseded by Round 2C1 below. Reasons: `limits` permitted `"stop_after"` key instead of enforcing explicit kwarg only; claimed D1 runnable under `pre-reference-D1` when D1 must be gated pending P06/C verified frozen reference; separate episode and lineage ID checks allowed cross-pair adversarial recombination `(ep1, lin2)`; mutable evaluation dataset list was not snapshotted into an immutable sequence; `selected_seed` was not checkpoint-bound.
  - Sol Acceptance Record: 2A physical repairs and 2B CPU exact resume ACCEPTED by Sol auditor. Latest manager baseline verified: 37 selected / 36 executed PASS / 1 CUDA unimplemented skip, 7.805s + diff-check PASS.
  - Upstream Blocked Work Inventory & Gate Failures:
    - Stage B: Raises early `NotImplementedError` naming missing P06 Task 1C (`src/icgs/algorithms/objectives/task.py`), zero mutation.
    - Stage C: Raises early `NotImplementedError` naming missing P06 reference router and model weights binding, zero mutation.
    - Stage D2: Raises early `NotImplementedError` naming missing P09 evaluation objective (`src/icgs/algorithms/objectives/evaluation.py`), zero mutation.
    - Stage E: Raises early `NotImplementedError` naming missing P09 calibration API, zero mutation. Completely deleted premature local calibration fitting implementation (`_fit_calibration_temperatures`); replaced with honest no-success/no-mutation gate test `test_stage_e_raises_not_implemented_dependency_error`.
    - P02 Descendant-Aware Lineage Closure: UNAVAILABLE in P02 (`src/icgs/data/datasets/episodes.py`). Literal ID and split checks only; limitation documented.
    - Full Workloads (C1–C5): NOT RUN; no GPU or remote cluster training.
  - Stage Identity & Dataset Protocol:
    - Removed `dataset-synthetic` and `reference-synthetic` defaults in `run_method_training` and `save_checkpoint`. Requires explicit non-empty `dataset_id`.
    - Stale claim: Pre-reference stages (A0, A1, D1) before Stage C; SUPERSEDED: Stage D1 is gated with early NotImplementedError pending P06/C verified frozen reference binding; only A0 and A1 use pre-reference identity protocol.
  - Supported-Stage Batching & Optimization:
    - Consumes `cfg.stages[stage].batch_size`: samples `batch_size` independent records per optimizer update.
    - Executes physical objectives serially across records in batch and averages valid objective contributions (`avg_loss = total_loss / valid_count`).
    - Single backward, gradient clip (`clip_grad_norm_`), optimizer step, and scheduler step per batch update.
    - Stateful batch sampler supports `batch_size`; sequence dataset supports random-with-replacement under runner RNG (not striding). [SUPERSEDED by 2C1 exact pair set and single sample_batch contract].
    - Verified by `test_batch_size_consumption_multiple_records_and_exact_resume`: proves both records contribute to gradient in `batch_size: 2` and exact resume preserves identical bit-for-bit weights across paused updates.
  - Seed Readiness & Reporting:
    - Strictly requires `stages.training_seeds` (3 distinct non-negative integers); caller-owned `rng` cannot bypass missing seed metadata.
    - Selects explicit run seed (`report.selected_seed`) and reports it separately in `TrainingReport`. [SUPERSEDED by 2C1 checkpoint-bound selected_seed].
    - Verified by `test_seed_readiness_rejects_missing_seeds_and_rng_bypass`.
  - Provenance & Partition Overlap Protection:
    - Enforces record provenance: rejects `test`, `calib`, `calibration`, `audit` records during training updates; requires `dev` partition for evaluation.
    - Stale helper: `_check_partition_overlap` was duplicate/unused and allowed cross-pair recombination; DELETED in 2C1.
    - Verified by `test_provenance_split_leakage_and_partition_overlap_rejection`.
  - Stage-Specific Lexicographic Selection:
    - Stage A0: primary metric is `normalized_reconstruction_loss`; paired native fidelity gating reported `"NOT RUN (unavailable)"`, `bridge_gate_passed: False`.
    - Stages A1 / D1: primary metric is `physical_validation_loss`; secondary is optional explicitly provided `dev_ranking` (reported `"NOT RUN (unavailable)"` when absent).
    - Full metric vector compared lexicographically (`candidate_vector < best_metric_vector`); exact ties favor earlier checkpoint and increment patience.
    - Verified by `test_lexicographic_selection_priority_order_ties_and_patience`.
  - Strict Cadence and Limits Validation:
    - Stale claim: `limits` permitted `stop_after`; SUPERSEDED in 2C1: `limits["stop_after"]` is strictly rejected; only explicit keyword argument `stop_after` is permitted; `limits` permits `max_updates` only.
    - Cadences (`evaluation_interval_updates`, save cadence) must be positive integers in config; boolean and non-positive values strictly rejected.
    - Verified by `test_cadence_and_limits_strict_validation`.
  - Config Persistence (`resolved_config.json`):
    - When `output_dir` is requested, writes `resolved_config.json` beside checkpoint directory before execution begins.
    - Verified by `test_resolved_config_json_written_before_execution_when_output_requested`.
  - Bounded Developer Usage Example:
    ```python
    from icgs.configuration.method import MethodConfig
    from icgs.models.decoders.physical import PhysicalDecoder
    from icgs.models.dynamics.physical import PhysicalDynamics
    from icgs.models.encoders.physical import PhysicalEncoder
    from icgs.models.memories.physical import PhysicalMemory
    from icgs.training.method import run_method_training
    import torch

    cfg = MethodConfig.from_dict({
        "stages": {
            "training_seeds": (101, 102, 103),
            "generator_seed": 104,
            "reset_seed": 105,
            "action_seed": 106,
            "evaluation_interval_updates": 5,
            "early_stopping_patience": 3,
            "A1": {"batch_size": 2, "supervised_intervals": 2, "burnin_intervals": 0, "rollout_curriculum": [2]},
        }
    })
    # Caller owns model initialization seed and initial weights; runtime run_seed does not retroactively reinitialize models
    torch.manual_seed(101)
    components = {
        "geometry": torch.nn.ModuleDict({
            "encoder": PhysicalEncoder(method_config=cfg),
            "decoder": PhysicalDecoder(method_config=cfg),
        }),
        "physical_memory": PhysicalMemory(config=cfg),
        "dynamics": PhysicalDynamics(cfg),
    }
    # Datasets require explicit provenance with split, episode_id, and lineage_id
    train_dataset = [
        {
            "transitions": [...],
            "state": ...,
            "valid": torch.ones((1, 2), dtype=torch.bool),
            "provenance": {"episode_id": "ep_train_1", "lineage_id": "lin_train_1", "split": "train"},
        },
    ]
    dev_dataset = [
        {
            "transitions": [...],
            "state": ...,
            "valid": torch.ones((1, 2), dtype=torch.bool),
            "provenance": {"episode_id": "ep_dev_1", "lineage_id": "lin_dev_1", "split": "dev"},
        },
    ]
    # One run executes one seed; caller specifies dataset_id and optional evaluation dataset
    report = run_method_training(
        config=cfg,
        components=components,
        datasets=train_dataset,
        stage="A1",
        dataset_id="dataset-v1",
        run_seed=101,
        limits={"max_updates": 10},
        evaluation_dataset=dev_dataset,
        output_dir="checkpoints/a1_run",
    )
    ```
  - Test Evidence:
    - Fast L0 validators: `python3 -B scripts/validate_fast.py` (19 executed, 0 skipped, PASS in 0.491s) and `python3 -B -S scripts/validate_fast.py` (19 executed, 0 skipped, PASS in 0.491s).
    - L1 Method training suite: `.venv/bin/python -B -m unittest discover -s tests -p test_method_training.py` (52 selected, 51 executed PASS, 1 skipped: CUDA fixture NOT IMPLEMENTED in 20.140s).
    - L1 World model suite: `.venv/bin/python -B -m unittest discover -s tests -p test_world_model.py` (29 executed PASS in 1.194s).
    - Full method test suite: `.venv/bin/python -B -m unittest discover -s tests -p 'test_method_*.py'` (93 selected, 92 executed PASS, 1 skipped in 23.353s).
    - git diff --check: PASS (0 issues).

- Fix Round 2C1 (Strict Readiness, Sampling, Gates & Provenance Contracts):
  - Status: 2A physical repairs and 2B CPU exact resume ACCEPTED by independent Sol auditor. Earlier 2C rejected and superseded; 2C1 strict readiness/sampling/gates and 3 Sol closure items implemented and verified.
  - Closure Item 1 (Selected Run Seed Checkpoint-Bound & Validated Before Mutation):
    - `selected_seed` is required checkpoint identity/progress metadata in both `manifest["selected_seed"]` and `state["selected_seed"]`.
    - Standalone `save_checkpoint` requires non-negative integer `selected_seed` (no silent default; missing/non-int raises TypeError/ValueError).
    - `load_checkpoint` validates `selected_seed` non-negative integer in top-level state and manifest, and validates agreement.
    - `validate_resume` in `src/icgs/artifacts/method.py` includes `"selected_seed"` as a required key.
    - In `run_method_training`, moved resume identity preflight (`schema_version`, `stage`, `dataset_id`, `reference_id`, `config_id`, `total_updates`, `selected_seed`) before ALL live mutations: before output directory writes (`resolved_config.json`), before ambient RNG handling / reseeding, before freeze mode alterations / `apply_freeze_boundary`, and before optimizer/scheduler construction.
    - Used narrow `_preflight_resume_checkpoint` metadata helper to load state on CPU once and preflight identity before mutation, explicitly validating direct equality between top-level `state["selected_seed"]` and `saved_manifest["selected_seed"]`.
    - Verified by extended `test_resume_rejects_missing_or_mismatched_seed_before_mutation`: proves that on mismatch (both requested vs manifest and corrupted top-level seed 102 vs manifest 101), trainable module weights/modes, frozen-role module (`physical_memory`) initially in train mode with `requires_grad=True` (mode, requires_grad, weights, buffers), ambient RNG (Python, NumPy, PyTorch, CUDA), and existing output config sentinel file remain completely unchanged.
    - Verified by `test_resume_same_seed_bitwise_continuation`: same-seed resume to update 2 preserves bit-for-bit weights and reports bound seed 102.
  - Closure Item 2 (Exact Provenance Pair Set, Immutable Eval Snapshot & Duplicate Helper Deletion):
    - Replaced separate episode/lineage ID membership sets with exact `train_provenance_pairs: set[tuple[str, str]]`.
    - Rejects adversarial cross-pair recombination `(ep1, lin2)` when only `(ep1, lin1)` and `(ep2, lin2)` were declared. Verified by `test_sampled_records_reject_adversarial_recombination_pairs`.
    - Snapshots finite evaluation sequence into an immutable tuple (`eval_snapshot = tuple(evaluation_dataset)`) once during preflight before updates start and uses it throughout.
    - Promises membership snapshot (elements and order), not deep immutable tensors. Verified by `test_evaluation_snapshot_immutable_to_list_mutation` proving caller list mutations (`clear()`, `append()`) after start cannot alter evaluation membership or order.
    - Completely deleted unused `_check_partition_overlap` duplicate preflight helper.
  - Closure Item 3 (Plan & Usage Corrections):
    - Marked whole earlier 2C section rejected and superseded.
    - Corrected stale claims regarding limits `stop_after` (now strictly rejected from `limits`, only explicit kwarg accepted; `limits` permits `max_updates` only) and runnable D1 (`D1` strictly gated with early `NotImplementedError` pending P06/C; public runner supports `A0`/`A1` only).
    - Corrected developer usage example: explicit `torch.manual_seed(101)` before component construction; caller initialization responsibility explicit; runtime run_seed does not retroactively reinitialize models.
  - Scope Item 1 (Public D1 Gating & Frozen-Reference Prerequisite):
    - Public runner for Stage D1 raises early `NotImplementedError` naming dependency on AFTER-C verified frozen reference and unavailable P06 router/weights binding.
    - Removed `pre-reference-D1`; pre-reference identity protocol is restricted to (`pre-reference-A0`, `pre-reference-A1`).
    - Documented that Stage D1 multi-record global selected-head/step normalization remains unresolved and uncertified.
    - Private synthetic D1 objective tests (`test_d1_history_replay_frozen_gradient_transmission_and_bootstrap`, `test_d1_provenance_and_validity_shape_rejection`) retained and passing.
    - Upstream gates for B, C, D2, E remain strictly enforced with early `NotImplementedError`. Only A0/A1 public synthetic runner supported.
  - Scope Item 2 (Seed Readiness, Ambient RNG Invariance & Injected RNG Contract):
    - Requires actual `generator_seed`, `reset_seed`, and `action_seed` metadata plus configured 3 distinct `training_seeds` and selected `run_seed` membership in `stages.training_seeds`.
    - Fresh run seeds Python `random`, global NumPy, PyTorch, and available CUDA RNG from `selected_seed`, plus owned runner `np.random.Generator`.
    - Documented in `run_method_training` docstring that models must be passed already constructed; caller owns model initialization seed and initial weights.
    - Injected RNG contract: supports only `numpy.random.Generator`, which is reseeded to `selected_seed` on fresh run, and restored from checkpoint on resume. Unsupported injected RNG types (e.g. `random.Random`, strings) rejected with actionable `TypeError`.
    - Verified by `test_nonzero_update_repeatability_ambient_rng_dropout_and_resume`: two runs with different ambient Python/NumPy/Torch seeds produce bit-for-bit identical weights and losses under active `Dropout(p=0.5)`; checkpoint resume preserves identical RNG state and matches uninterrupted run.
  - Scope Item 3 (Provenance, Stateful Sampler Contract & Disjoint Partition Lineage):
    - Every training update record requires explicit `split="train"` and non-blank `episode_id` and `lineage_id`. Selection records require explicit `split="dev"`.
    - Stateful training samplers must expose inspectable `.records` Sequence for preflight provenance; opaque samplers missing `.records` rejected with `TypeError`.
    - Evaluation dataset must be an immutable Sequence only; stateful evaluation datasets rejected with `TypeError`.
    - Disjointness between training and evaluation partitions verified before any update begins: requires zero overlapping episode IDs and zero overlapping lineage IDs.
    - Employs P02 `validate_split_lineage` helper from `icgs.data.datasets.episodes` for literal lineage split validation.
    - Sampled records are verified in `_sample_records` to belong to declared training provenance before backward; undeclared records raise `ValueError`.
    - Verified by `test_red_2c1_stateful_sampler_requires_inspectable_records`, `test_red_2c1_provenance_requires_lineage_id`, `test_provenance_split_leakage_and_partition_overlap_rejection`, and `test_sampled_records_must_belong_to_declared_train_provenance`.
  - Scope Item 4 (Single sample_batch Invocation Contract):
    - Strict single `sample_batch(stage, update, curriculum_k, *, rng, batch_size)` invocation contract.
    - Requires `state_dict` and `load_state_dict` before any component mutation in both `run_method_training` and `load_checkpoint`.
    - Removed catch-TypeError retry fallback; internal errors propagate directly once.
    - Enforces exact requested record count (`len(records) == batch_size`); silent truncation or over-sampling rejected with `ValueError`.
    - Updated existing sampler fixtures to match keyword-only `*, rng, batch_size` signature and verified exact resume proof.
    - Verified by `test_sample_batch_error_propagates_without_retry_and_exact_count_enforced` and `test_stateful_sampler_contract_and_rejection_of_unrestorable_callable`.
  - Scope Item 5 (Strict Limits & Cadence Ownership):
    - `limits` strictly rejects `"stop_after"` key in favor of explicit keyword argument `stop_after`.
    - `limits["max_updates"]` is the only allowed finite execution schedule binding.
    - Evaluation cadence is owned by `cfg.stages.evaluation_interval_updates` and cannot be overridden by limits.
    - Verified by `test_red_2c1_limits_rejects_stop_after_key` and `test_cadence_and_limits_strict_validation`.
  - Scope Item 6 (Config Persistence Envelope):
    - Writes `resolved_config.json` with exact `cfg.resolved_config()` envelope (including schema, config, and content hash) before first sample is drawn when `output_dir` is specified.
    - Verified by `test_red_2c1_resolved_config_json_matches_resolved_config_envelope` and `test_resolved_config_json_written_before_execution_when_output_requested`.
  - Item 7 (Documentation & Sequence Sampling):
    - Corrected developer usage example with `config=cfg`, declared seeds (`generator_seed`, `reset_seed`, `action_seed`, `training_seeds`), and explicit record provenance.
    - Sequence datasets sample random-with-replacement under runner RNG (not striding).
    - Documented honest status: 2A/2B/2C1 accepted; 2C2 implemented; D1/B/C/D2/E dependency gated; no overall P11 completion.
  - Sol Closure: Batch 2C1 accepted by Sol auditor. Preflight ordering defect resolved: resume preflight validates manifest, config, and selected_seed before output writes, RNG, freeze, and optimizer.
  - Verification Evidence:
    - Fast L0 validators: `python3 -B scripts/validate_fast.py` (19 executed, 0 skipped, PASS in 0.472s) and `python3 -B -S scripts/validate_fast.py` (19 executed, 0 skipped, PASS in 0.478s).
    - L1 Method training suite: `.venv/bin/python -B -m unittest discover -s tests -p test_method_training.py` (56 selected, 55 executed PASS, 1 skipped: CUDA fixture NOT IMPLEMENTED in 21.978s).
    - L1 World model suite: `.venv/bin/python -B -m unittest discover -s tests -p test_world_model.py` (29 executed PASS in 1.146s).
    - Full method test suite: `.venv/bin/python -B -m unittest discover -s tests -p 'test_method_*.py'` (97 selected, 96 executed PASS, 1 skipped in 22.868s).
    - `git diff --check`: PASS (0 issues).


- Fix Round 2C2 (Final Local Selection Repair & Checkpoint Integrity) & 2C2 Final Fix:
  - Status: Implemented and verified locally; stopped for Sol 2C2 closure review.
  - Baseline Evidence: 103 selected, 102 executed PASS, 1 CUDA skip in 22.355s / world 29 PASS in 1.287s / diffcheck PASS.
  - Item 1 (Strict Selection State Validator & Documented Status Vocabulary):
    - Enforces documented `selection_status` vocabulary: `ALLOWED_SELECTION_STATUSES = frozenset({"no_evaluation", "certified", "not_certified", "diagnostic_external", "not_certified (bridge gate unavailable)"})`.
    - Stage consistency: Stage A0 and Stage A1 with external `dev_ranking` diagnostic cannot have status `"certified"`.
    - Exact primary equality: when `best_metric` is present, requires exact equality `float(best_metric) == float(best_metric_vector[0])` with zero tolerance (no `math.isclose`).
    - Bounds enforcement: `best_update` must be `None` for no-evaluation, or integer `1 <= best_update <= checkpoint update`.
    - No-evaluation invariants: when `selection_schema` is `None`, status must be `"no_evaluation"`, `patience_counter == 0`, `eval_history == []`, and `best_metric_vector`, `best_update`, and `best_metric` must be `None`.
    - Structured history: every entry in `eval_history` must be a Mapping with finite float `metric_vector` matching the exact schema arity.
    - Validated in `save_checkpoint` before saving publishes and in `_preflight_resume_checkpoint` before any live mutation.
    - Verified by negative checks in `test_2c2_selection_state_validator_strict_invariants_and_negative_checks`.
  - Item 2 (Supplied dev_ranking Strict Uniformity & Evaluator Attribution):
    - Evaluation records may carry finite `dev_ranking` diagnostic only if ALL records have the exact same value; varying or partial values are rejected before updates start (no first-record wins).
    - Labeled as `selection_status: "diagnostic_external"` and `evaluator: "NOT RUN (diagnostic external supplied)"`, never certified.
    - Real `_evaluate_stage` verified for sequence reordering invariance (`[rec_a, rec_b]` vs `[rec_b, rec_a]` produces identical metric vector) and varying/partial values rejection in `test_2c2_dev_ranking_sequence_reorder_invariance_and_varying_values_rejection`.
  - Item 1 (Immutable Predeclared Selection Schema):
    - Predeclares an immutable `selection_schema` tuple before any updates begin:
      - Stage A0: strictly `("normalized_reconstruction_loss",)` with minimization direction; `dev_ranking` in evaluation records is strictly rejected with `ValueError`.
      - Stage A1: `("physical_validation_loss", "dev_ranking")` if `dev_ranking` is uniformly supplied with identical value across all evaluation records; `("physical_validation_loss",)` if omitted; partial/mixed presence, varying values, or non-finite float values are rejected with `ValueError`.
    - Evaluation candidate vectors are validated to match the exact arity and names of `selection_schema` on every evaluation interval, preventing dynamic arity changes.
    - Verified by `test_2c2_selection_schema_predeclaration_and_rejection_of_mixed_or_invalid_metrics`.
  - Item 2 (Selection State Persistence & Early Preflight):
    - `selection_state` records `selection_schema`, `best_metric_vector`, `best_update`, `patience_counter`, `eval_history`, and `selection_status`.
    - Checkpoints without evaluation explicitly save `selection_state` with `None` fields (`best_metric: None`, `best_metric_vector: None`, `selection_schema: None`, `best_update: None`, `patience_counter: 0`, `eval_history: []`, `selection_status: "no_evaluation"`); removed `float("inf")` sentinels.
    - `_preflight_resume_checkpoint` validates `selection_state` structure, finiteness, and agreement with the expected stage and run schema before any live mutation. Non-finite values, arity mismatches, and stage/schema mismatches are rejected immediately with `ValueError`.
    - Verified by `test_2c2_checkpoint_selection_state_preflight_rejection_of_corrupted_vectors_and_schemas`.
  - Item 3 (Lexicographic Comparison, Tie-Breaking & Patience Lifecycle):
    - Selection evaluates candidates strictly lexicographically under element-wise minimization.
    - Exact numerical ties (`candidate_vector == best_metric_vector`) evaluate to non-improvement: keeps earlier checkpoint, increments patience counter, and does not advance `best_update`.
    - Genuine improvement (`candidate_vector < best_metric_vector`) updates `best_metric_vector`, sets `best_update = current_update + 1`, and resets `patience_counter = 0`.
    - When `patience_counter >= patience_limit`, early stopping triggers.
    - Real run lifecycle verified by `test_2c2_lexicographic_selection_lifecycle_primary_dominates_secondary_ties_and_patience` proving primary dominates secondary, tie-breaking by secondary, exact ties increment patience, and early stopping triggers at cfg patience.
    - Exact bit-for-bit resume continuation with active evaluation verified by `test_2c2_real_optimizer_evaluation_resume_bit_for_bit_continuation`.
  - Item 4 (Stage A0 Bridge Gate Certification Refusal):
    - Stage A0 fidelity gate is reported `"NOT RUN (unavailable)"`, `bridge_gate_passed: False`.
    - Because the bridge gate is unavailable, `checkpoint_A0_best.pt` is NEVER written, even when reconstruction loss improves.
    - Periodic diagnostic checkpoints are saved with explicit status `selection_status: "not_certified (bridge gate unavailable)"`.
    - Verified by `test_2c2_a0_refuses_to_certify_best_checkpoint_and_saves_diagnostic`.
  - Item 5 (Early Stopping Progress & Already-Stopped Resume):
    - When early stopping triggers off periodic save intervals, the runner saves the final state to `checkpoint_{stage}_{u}.pt`.
    - When resuming from an already-stopped checkpoint (`patience_counter >= patience_limit`), the runner performs 0 updates silently, reports `updates_completed=0`, `total_updates_completed=start_update`, `stopped_early=True`, `status="stopped_early"`.
    - `TrainingReport` exposes `updates_completed` (this run), `total_updates_completed` (overall progress), `best_update`, `selection_schema`, and `selection_status`.
    - Verified by `test_2c2_early_stopping_saves_off_interval_and_already_stopped_resume_does_not_advance`.
  - Item 6 (Edge Fix — Finite Run Ending Before Eval Cadence):
    - When `evaluation_dataset` and schema are predeclared but run ceiling ends before the first evaluation interval (e.g. `eval_interval > max_updates` or `stop_after < eval_interval`), final checkpoint is saved with consistent schema-present non-certified status `selection_status: "not_certified"` (or `"not_certified (bridge gate unavailable)"` for A0), `best_metric_vector: None`, `best_update: None`, `patience_counter: 0`, and `eval_history: []`.
    - Preflight resume from this checkpoint succeeds, allowing continuation to later updates where evaluation executes normally; verified `checkpoint_A1_5.pt` exists, payload `selection_status == "certified"`, `best_update == 5`, and `eval_history` is non-empty.
    - Verified by `test_2c2_eval_interval_greater_than_max_updates_checkpoint_and_resume_succeeds` (PASS).
  - Verification Evidence:
    - Fast L0 validators: `python3 -B scripts/validate_fast.py` (19 executed, 0 skipped, PASS in 0.494s) and `python3 -B -S scripts/validate_fast.py` (19 executed, 0 skipped, PASS in 0.484s).
    - L1 Method training suite: `.venv/bin/python -B -m unittest discover -s tests -p test_method_training.py` (65 selected, 64 executed PASS, 1 skipped: CUDA fixture NOT IMPLEMENTED in 20.700s).
    - L1 World model suite: `.venv/bin/python -B -m unittest discover -s tests -p test_world_model.py` (29 executed PASS in 1.112s).
    - Full method test suite: `.venv/bin/python -B -m unittest discover -s tests -p 'test_method_*.py'` (106 selected, 105 executed PASS, 1 skipped in 22.811s).
    - `git diff --check`: PASS (0 issues).
  - Item 6 (Edge Fix — Finite Run Ending Before Eval Cadence):
## Current Canonical State Summary (P11 Local Training Runner)

1. **Supported Public Stages**: Only stages `A0` (geometry autoencoder) and `A1` (geometry + physical memory + dynamics rollout) are executable on the public synthetic runner.
2. **Upstream Gated Stages**: Stages `B` (event/task objectives, P06 Task 1C), `C` (reference router / model weights binding, P06), `D1` (AFTER-C verified frozen reference binding, P06/C, and multi-record global loss normalization), `D2` (evaluator/terminal objectives, P09), and `E` (temperature calibration API, P09) raise early `NotImplementedError` with zero live mutation.
3. **Freeze Boundaries**: Strict parameter ID, buffer, and mode invariance enforced. Frozen components set to `.eval()` with `requires_grad=False`; trainable components set to `.train()` with `requires_grad=True`. Shared/aliased parameters across trainable and frozen roles strictly rejected.
4. **Exact Resume & Preflight**: Checkpoints store exact model state dicts, optimizer state, scheduler state, Python/NumPy/Torch/CUDA RNG states, sampler state, config envelope, and `selection_state`. Resume preflight validates manifest identity (`schema_version`, `stage`, `dataset_id`, `reference_id`, `config_id`, `total_updates`, `selected_seed`) and `selection_state` BEFORE any output writes, RNG reseeding, freeze boundary alteration, or optimizer construction. Bit-for-bit identical parameters, next batches, and scheduler states proven across paused and resumed runs.
5. **Dataset & Provenance**: Records require explicit `split="train"` (or `split="dev"` for evaluation), non-blank `episode_id`, and non-blank `lineage_id`. Stateful training samplers must expose inspectable `.records` sequence. Evaluation requires an immutable Sequence (snapshotted into an immutable tuple on preflight). Train and eval partitions must have disjoint episode and literal lineage IDs (via P02 `validate_split_lineage`). Sampled records must belong to declared `(episode_id, lineage_id)` pair sets before backward.
6. **Sampling Contract**: Strict single `sample_batch(stage, update, curriculum_k, *, rng, batch_size)` invocation; requires `state_dict` and `load_state_dict`; internal errors propagate immediately once without catch-TypeError retry; exact requested record count enforced. Sequence datasets sample random-with-replacement under runner RNG.
7. **Selection & Early Stopping**: Predeclared immutable `selection_schema` per stage. Lexicographic comparison with tie-breaking favoring earlier checkpoints. Stage A0 refuses to certify best checkpoint (`checkpoint_A0_best.pt` never written) while bridge gate is unavailable. Early stopping saves final checkpoint off-interval; resume on stopped checkpoint does not advance updates silently.
8. **Remaining Open Items / Upstream Blockers**:
   - Stage B objectives (`src/icgs/algorithms/objectives/task.py`, P06 Task 1C).
   - Stage C reference router and frozen model weights binding (P06).
   - Stage D1 multi-record global selected-head/step loss normalization and frozen reference binding (P06/C).
   - Stage D2 evaluator objectives (`src/icgs/algorithms/objectives/evaluation.py`, P09).
   - Stage E temperature calibration API (P09).
   - P02 descendant-aware lineage closure (literal IDs validated).
   - Native action bridge gate certification for Stage A0 checkpoint selection.
   - Mandatory published benchmark feasibility gates C1–C5 remain NOT RUN.

## Observability integration addendum — 2026-09-09

The native training logger defect for `record=True` with W&B disabled is fixed by
initializing an explicit logger variable and using the optional local Lightning
bridge. It preserves native metric names and optimizer-step axes; the lazy W&B
mirror is rank-0/queue bounded and never starts unless explicitly requested.
Real model training, pilot workloads, published checkpoint generation, and remote transport remain **NOT RUN** here; bounded synthetic tests do generate local checkpoints for resume and diagnostic verification.
