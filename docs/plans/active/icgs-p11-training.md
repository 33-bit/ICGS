# P11: Staged training, freeze boundaries and resumable artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Train added components in a reproducible phase order without moving the reference value target.

**Architecture:** Training owns optimizer/scheduler/batching/resume state; objective modules own math and models own forward tensors. An explicit phase matrix prevents accidental upstream updates; collection stage C has no optimizer.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Data and training](../../method/data-training.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — planned runtime NOT IMPLEMENTED**. This document is a category C
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

Public capability boundary (planned; not currently importable):

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

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.training.stages.method import trainable_components, rollout_horizon
self.assertEqual(trainable_components('C'), frozenset())
self.assertEqual(trainable_components('D1'), frozenset({'dynamics'}))
self.assertEqual(rollout_horizon('A1', 0, 90), 1)
self.assertEqual(rollout_horizon('A1', 60, 90), 4)
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_training.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

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

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_training.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 2: Full-history replay, batches and selected metrics

**Files:** Create `src/icgs/training/method.py`.
**Test owner:** `tests/test_method_training.py`.
**Consumes / produces:** Produces `sample_anchor_horizon(groups, rng)` and `history_slices(boundary, burnin)`; consumes objective APIs.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.training.method import history_slices
prefix, burn, supervised = history_slices(boundary=20, burnin=8)
self.assertEqual(prefix, slice(0, 12))
self.assertEqual(burn, slice(12, 20))
self.assertEqual(supervised.start, 20)
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_training.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

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

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_training.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 3: Exact resume, reference identity and calibration separation

**Files:** Complete runner and extend `src/icgs/artifacts/method.py`.
**Test owner:** `tests/test_method_training.py`.
**Consumes / produces:** Produces validate_resume; persists weights,optimizer,scheduler,stage,update,resolved config,dataset/reference IDs and Python/NumPy/torch/CUDA RNG.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.artifacts.method import validate_resume
with self.assertRaisesRegex(ValueError, 'reference|manifest'):
    saved = dict(schema_version=1, stage='D2', reference_id='a',
                 dataset_id='d', config_id='c')
    validate_resume(saved, dict(saved, reference_id='b'))
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_training.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
for key in ('schema_version','stage','reference_id','dataset_id','config_id'):
    if saved[key] != requested[key]:
        raise ValueError(f'incompatible resume manifest: {key}')
# Construct permitted optimizer, load all states, restore RNG before next sample.
# Write checkpoint atomically; preserve previous accepted checkpoint.
```

Use tiny deterministic two-step uninterrupted versus1+resume tests to compare weights,next batch,scheduler and RNG, not a real training job. E uses calibration-only temperatures; retain identity on failed fit. Report3 training seeds separately, class-balance inverse inclusion weights and hard-example growth; main-scale launch stays FG gated.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_training.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
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

- Documentation drafting: this plan specifies future work only.
- Component RED/GREEN commands: **NOT RUN** — runtime/test files are not implemented.
- L1 model assertions: **NOT RUN** — future installed supported environment required.
- L2/C1–C5: **NOT RUN** by this plan — preserve mandatory integration acceptance.
- L3/L4, collection and training: **NOT RUN** — separate resource authorization required.
- Remaining risk: Training cost, optimizer stability and scientific performance remain unknown before authorized pilots.

## Observability integration addendum — 2026-09-09

The native training logger defect for `record=True` with W&B disabled is fixed by
initializing an explicit logger variable and using the optional local Lightning
bridge. It preserves native metric names and optimizer-step axes; the lazy W&B
mirror is rank-0/queue bounded and never starts unless explicitly requested.
Training, checkpoint generation and remote transport remain **NOT RUN** here.
