# P01: Fixed-interval execution and replay capabilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Execute explicit absolute commands for measured 0.1-second intervals and expose honest replay fidelity.

**Architecture:** Add an independently selected timed RLBench adapter and execution materializer beside the native rollout. Low-level controller time, measured state and privileged monitoring are separate capabilities.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Contracts and cadence](../../method/contracts.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — P01 tasks 1–3 runtime boundary and correction round 1 implemented;
prior scoped correction and centralization follow-up accepted by Sol audit; empirical
timing/replay feasibility and restoration remain NOT IMPLEMENTED**. This document is a category C
component of the approved docs-only research migration, not permission to execute it.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00; approved FG timing/replay pilot protocol before any simulator work.
Read [workflow](../../WORKFLOW.md), [architecture](../../ARCHITECTURE.md),
[baseline](../../baselines/instant_policy.md), [native contract](../../components/policy-data-contract.md),
[research policy](../../RESEARCH.md) and [validation guide](../../../tests/README.md)
before runtime execution. Inspect Git state; preserve unrelated work and user assets.

## Configuration consumption addendum — 2026-09-09

Default values now belong to the [packaged primary JSON](../../../src/icgs/configuration/profiles/icgs_primary.json),
with key/unit/restriction details in the [parameter reference](../../method/parameters.md).
Propagation scope for this follow-up: **planning.h, planning.r and control.dt0**
only. Pass those scalars into `materialize_prefix`. `control.clock_track` and all
`sensors` entries remain declarations for measured controller/calibration protocols;
P01 does not consume them, infer a simulator clock from them, or treat them as
measured evidence. Resolve the immutable `cfg: MethodConfig` once outside step/
forward loops; do not read JSON inside those loops. This follow-up keeps the
explicit-argument seam and adds propagation evidence without adding adapter or
P13 composition.

Use an explicitly resolved `cfg: MethodConfig` (or injected section) at the
composition boundary. Read measured controller/calibration protocol separately.
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

- Create: `src/icgs/execution/timed.py` — prefix materialization and interval protocol.
- Create: `src/icgs/environments/rlbench/timed.py` — target-hold stepping capability.
- Create: `src/icgs/environments/rlbench/replay.py` — capability/report validation only; P08 implements restoration.
- Test: `tests/test_timed_execution.py`.

Public capability boundary (materializer, timed stepping and report validation are
importable; engine restoration remains P08-owned):

```python
materialize_prefix(candidate, *, h: int, r: int, duration_s: float) -> CommandPrefix
TimedEnvironment.reset(seed) -> TimedObservation
TimedEnvironment.advance(command) -> ExecutedTransition
ReplayProvider.restore(anchor_ref) -> tuple[TimedObservation, ReplayReport]
```

Every native relative target is multiplied by the same proposal root. Native grip maps with int(g_native > 0), and each r-group holds its first grip. Store raw and materialized commands; achieved pose/duration never substitutes for a command.

### Task 1: Materialize the physical command once

**Files:** Create `src/icgs/execution/timed.py`.
**Test owner:** `tests/test_timed_execution.py`.
**Consumes / produces:** Consumes method Candidate wrapper/native trajectory; produces `materialize_prefix`; internal `materialize_grips(grips, r)` returns binary sequence.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.execution.timed import materialize_grips
self.assertEqual(materialize_grips([-1, 1, 0, 1], 2), (0, 0, 0, 0))
self.assertEqual(materialize_grips([1, -1, -1, 1], 2), (1, 1, 0, 0))
with self.assertRaises(ValueError):
    materialize_grips([1], 0)
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_timed_execution.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
result = []
for start in range(0, len(grips), r):
    held = int(grips[start] > 0)
    result.extend([held] * min(r, len(grips) - start))
return tuple(result)
```

Add a noncommuting rotated-root fixture proving targets are root@A[j], not A[j]@root or cumulative multiplication. Enforce B=1, 1<=r<=h<=P, positive finite duration and owned command arrays. Same materializer is mandatory in collection, reference and all added controls.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_timed_execution.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 2: Bounded interval stepping with separate clocks

**Files:** Create `src/icgs/environments/rlbench/timed.py`.
**Test owner:** `tests/test_timed_execution.py`.
**Consumes / produces:** Produces `interval_substeps(duration_s, physics_dt)` and timed reset/advance/close; consumes explicit controller capability.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.environments.rlbench.timed import interval_substeps
self.assertEqual(interval_substeps(0.1, 0.005), 20)
with self.assertRaisesRegex(ValueError, 'interval'):
    interval_substeps(0.1, 0.03)
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_timed_execution.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
ratio = duration_s / physics_dt
count = round(ratio)
if count < 1 or not math.isclose(ratio, count, rel_tol=0, abs_tol=1e-9):
    raise ValueError('interval must match an integer physics-step count')
return count
```

Set one target at interval start, hold it while stepping exactly the configured substeps, then read measured observation and increment boundary once. Detect unsupported low-level IK/controller stepping instead of wrapping pose-reaching task.step as fixed time. Use bounded reset attempts, controller-declared `safe_hold()` followed by close on failure/cleanup, irreversible invalidation after an operational failure, controller-status diagnostics and explicit simulated versus wall timestamps. Cleanup diagnostics preserve safe-hold and close errors without claiming a robot safety policy.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_timed_execution.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 3: Replay does not claim inaccessible hidden state

**Files:** Create `src/icgs/environments/rlbench/replay.py`.
**Test owner:** `tests/test_timed_execution.py`.
**Consumes / produces:** Produces `ReplayReport` and `validate_replay_report(report)`; consumes snapshots or reset+recorded-command history, never online task features.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.environments.rlbench.replay import validate_replay_report
with self.assertRaisesRegex(ValueError, 'report'):
    validate_replay_report(True)
with self.assertRaisesRegex(ValueError, 'mode|fields'):
    validate_replay_report({'mode': 'exact-snapshot'})
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_timed_execution.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
# P01 defines the report schema and restoration capability only.
# P08 implements engine restoration and produces measured discrepancies.
report = dict(mode=mode, stored_fields=stored_fields,
              unavailable_fields=unavailable_fields,
              restored_fields=restored_fields, discrepancies=measurements,
              protocol_id=protocol_id)
# approximate-reset additionally reports repetitions >= 2 and finite uncertainty.
validate_replay_report(report)
```

Report joints/velocities, articulations, constraints/attachments, integrators, monitor history, RNG and solver coverage explicitly. If exact restore is unavailable use deterministic history replay; if equivalence still fails tag approximate-reset with repeated-reset uncertainty. Never relabel an approximate counterfactual exact.

This task validates the capability/report boundary without implementing or calling
restoration. P08 owns the engine-specific restoration implementation and replay
pilot; P01 owns target-hold stepping and sensor/timing traces.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_timed_execution.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

## Acceptance, resource limits and evidence

L1 uses an in-process deterministic stepping collaborator; it proves orchestration only. FG timing requires measured target-hold duration/substeps, pose/cloud/grip drift and paused/live-clock latency. FG replay requires duplicate-prefix pose/cloud/outcome discrepancies against tolerances approved before the pilot; L3 only under explicit caps.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_timed_execution.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

Keep native RLBenchAdapter, reward cadence and unbounded legacy behavior unchanged; added paths must be bounded. No robot adapter, controller safety claim, asset download or automatic simulator launch.

On failure, retain the failed evidence and isolate the new component/config ID.
Do not overwrite native checkpoints, relabel collected outcomes or drop difficult samples.
Resume only from a compatible explicit artifact/manifest; changed reference lineage
requires new outcome labels, not a cache workaround. Return to the preceding accepted
phase if an FG fails and request a scoped protocol decision.

## Execution evidence

### Configuration centralization follow-up — 2026-09-09

- Baseline central commit: `782eb683d85757a8d2450f789024e7c6b8abff41`.
  The canonical defaults remain `src/icgs/configuration/profiles/icgs_primary.json`;
  `MethodConfig` supplies the immutable typed sections and strict validation. No
  shared configuration/schema or P04/P13 file was changed by this follow-up.
- The existing explicit seam is sufficient. Composition resolves configuration
  once and passes scalars before any execution loop:

  ```python
  cfg = MethodConfig()  # or one explicit, validated MethodConfig overlay
  h = cfg.planning.h
  r = cfg.planning.r
  duration_s = cfg.control.dt0
  prefix = materialize_prefix(candidate, h=h, r=r, duration_s=duration_s)
  ```

  `control.clock_track` and `sensors.*` are not consumed by this invocation;
  measured controller clocks and calibration/replay protocol evidence remain
  separate runtime/feasibility inputs.
- Installed supported-L1 characterization command (cwd
  `/Users/33bit/AI/Research/VLA/ICGS`, no `PYTHONPATH`):
  `.venv/bin/python -B -m unittest discover -s tests -p
  'test_timed_execution.py' -v` — **23 selected, 23 executed, 23 PASS, 0
  skipped**. The test covers canonical default parity, nondefaults
  `(h=4, r=1, dt0=0.2)` and `(h=4, r=2, dt0=0.15)`, prefix length, exact
  `root @ A[j]` targets including a noncommuting fixture, grip cadence,
  per-command durations, and typed/shape rejection.
- This was characterization/coverage rather than a fabricated RED cycle: the
  existing explicit-argument materializer already accepted the resolved scalars,
  so no runtime implementation was added merely to create a failure. No config
  is read inside timed stepping or model-forward loops.
- Environment for this supported-L1 characterization: `.venv/bin/python`,
  Python **3.11.15**, NumPy **1.26.4**, PyTorch **2.2.0**, editable `icgs` from
  `src/icgs`; RLBench and PyRep are absent. No simulator, model, download,
  training, collection or robot workload was run.
- Centralization review status: **accepted by Sol Medium audit, no blockers or
  corrective changes found** (2026-09-09). Follow-up changed tests/documentation
  only; no runtime adapter was needed. P01 remains active; FG
  timing/replay feasibility and mandatory C1–C5 remain unmet and are not passed.

### Correction round 1 — 2026-09-09

- Replay reports now require explicit `unavailable_fields`; stored and unavailable
  inventories are known-field, duplicate-free and disjoint, and every required
  inventory category is accounted for. Required categories include joints,
  velocities, articulations, constraints, attachments, integrators, monitor
  history, RNG, solver, object poses, grip, controller state and task history.
  Restored fields must be a subset of stored fields. Exact snapshots require all
  required fields restored and no required field unavailable, while permitting
  known extra restored fields such as command history. No unavailable state is
  represented as a fake stored field.
- Replay validation requires finite scalar `pose`, `cloud` and `outcome`
  discrepancy entries. Approximate-reset reports additionally require integer
  `repetitions >= 2` (booleans rejected) and a nonempty finite uncertainty
  mapping. This is reported-structure validation only: it chooses no tolerance,
  does not claim equivalence or G5, and implements no restoration. P08 produces
  measured replay reports.
- The timed adapter now requires the narrow controller-declared `safe_hold()`
  capability. On interval/reset failure it invalidates before cleanup and then
  delegates `safe_hold()` before always attempting `close()`. Failed close leaves
  the adapter invalidated but permits an explicit bounded retry; successful close
  is idempotent. `TimedLifecycleError` retains the operational, safe-hold and
  close exceptions without `ExceptionGroup` or `add_note`; no pose, grip or
  physical safety policy is selected by P01.
- Correction RED diagnostic (source-root only):
  `PYTHONPATH=src python3 -B -m unittest discover -s tests -p
  'test_timed_execution.py' -v` selected/executed **21/21**, skipped **0**, and
  failed with **9 failure records and 3 error records** before the correction.
  The failures were the new replay schema and timed lifecycle/capability
  expectations, not an unrelated import setup failure.
- Correction GREEN diagnostics (source-root only): the replay subset selected/
  executed **5/5**, passed **5**, skipped **0**; the lifecycle subset selected/
  executed **7/7**, passed **7**, skipped **0**; the complete owner selected/
  executed **21/21**, passed **21**, skipped **0**. These deterministic source-root
  diagnostics are not supported-environment L1 acceptance.
- Review status: **Sol Medium re-audit accepted scoped code after one correction
  round; no remaining code blockers found** (2026-09-09). Luna Max implemented;
  the manager constrained corrections to replay evidence and timed cleanup, with
  no restoration implementation or speculative framework. The centralization
  follow-up is separately **accepted by Sol audit**, with installed focused L1
  evidence above. P01 remains active; full supported
  L1, measured FG timing/replay feasibility and mandatory C1–C5 remain unmet and
  are not passed by these diagnostics.

- Repository state: implemented directly in `/Users/33bit/AI/Research/VLA/ICGS` on
  `main`; no commit, stage, reset, checkout, worktree, or out-of-scope repository
  edit was made by the P01 workers. P01-owned changed paths are the three named
  runtime modules, `tests/test_timed_execution.py`, and this evidence update.
  Concurrent unrelated checkout changes were preserved and are not P01 work.
- P00 reuse: `materialize_prefix` returns the existing `CommandPrefix` and
  `TimedCommand` records; timed stepping returns existing `TimedObservation` and
  `ExecutedTransition` records. Native `RLBenchAdapter`, native preprocessing,
  action conversion, checkpoints, and reward cadence were not changed.
- RED evidence was retained. The exact plan command initially selected 1 test but
  failed with `ModuleNotFoundError: icgs` because the package is not installed;
  this was not counted as feature RED. Source-root diagnostic RED runs used
  `PYTHONPATH=src` only to execute local code and are not supported L1 claims:
  Task 1 selected/executed 4/4 with 4 missing-module errors; Task 2 8/8 with
  4 prior passes and 4 missing-adapter errors; Task 3 12/12 with 8 prior passes
  and 4 missing-replay-module errors.
- Component GREEN diagnostic command:
  `PYTHONPATH=src python3 -B -m unittest discover -s tests -p
  'test_timed_execution.py' -v`. Task 1: 4 selected, 4 executed, 4 passed,
  0 skipped. Task 2: 8 selected, 8 executed, 8 passed, 0 skipped. Task 3/final:
  12 selected, 12 executed, 12 passed, 0 skipped. These are tiny deterministic
  source-root diagnostics, not installed-environment L1 acceptance.
- L0 PASS: `python3 -B scripts/validate_fast.py` — 19/19 harness self-tests
  passed; higher tiers NOT RUN. L0 PASS:
  `python3 -B -S scripts/validate_fast.py` — 19/19 harness self-tests passed;
  higher tiers NOT RUN.
- Relevant native source-root regressions (diagnostic only, same explicit
  `PYTHONPATH=src`): `test_candidates.py` 2/2 PASS, 0 skipped;
  `test_method_contracts.py` 15/15 PASS, 0 skipped;
  `test_evaluation.py` 6/6 executed assertions PASS, 1 SKIPPED because RLBench
  is not installed; `test_inference_contract.py` 2/2 executed assertions PASS,
  1 SKIPPED because CUDA is unavailable. Skipped checks are not counted as PASS.
- Environment: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`,
  Python 3.14.4; NumPy and PyTorch are present, `icgs` is not installed and
  RLBench is absent. Declared supported range is Python `>=3.10,<3.13`, so
  installed supported L1 is **NOT RUN**. No fake modules or path injection were
  used to claim L1.
- L2/C1–C5, L3/L4, replay restoration, model execution, downloads, preprocessing
  jobs, collection, training, simulator workloads, and robot motion are **NOT
  RUN** and are not passed by this component. The earlier no-commit instruction
  was superseded by the owner's explicit request to commit the five P01-owned
  paths after review; unrelated changes remain excluded.
- Remaining risk: RLBench/PyRep low-level target-hold capability, measured
  controller timing, sensor/pose/cloud drift, and replay equivalence remain
  feasibility questions. P01 validates only the explicit collaborator seam and
  report vocabulary; P08 still owns engine restoration and replay pilots.

## Observability integration addendum — 2026-09-09

The existing `materialize_prefix` and `TimedRLBenchAdapter` boundaries accept an
optional recorder and emit only existing command identity, requested/achieved
duration, physics-substep, boundary and controller-status metadata. Reset,
advance and close are host spans; safe-hold/close failures emit bounded
lifecycle-error type information while preserving the original control error.
The recorder does not infer timing, alter target-root math or claim controller
success. Engine-specific restoration remains a P08 producer.
