# P01: Fixed-interval execution and replay capabilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Execute explicit absolute commands for measured 0.1-second intervals and expose honest replay fidelity.

**Architecture:** Add an independently selected timed RLBench adapter and execution materializer beside the native rollout. Low-level controller time, measured state and privileged monitoring are separate capabilities.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Contracts and cadence](../../method/contracts.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — planned runtime NOT IMPLEMENTED**. This document is a category C
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
Consumed sections for this plan: **control, planning, sensors**.

Pass config.planning.h, config.planning.r and config.control.dt0 into materialize_prefix. Read measured controller/calibration protocol separately. Do not change the in-progress implementation or its evidence in this configuration delivery; verify nondefault argument propagation when adding composition.

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

- Create: `src/icgs/execution/timed.py` — prefix materialization and interval protocol.
- Create: `src/icgs/environments/rlbench/timed.py` — target-hold stepping capability.
- Create: `src/icgs/environments/rlbench/replay.py` — capability/report validation only; P08 implements restoration.
- Test: `tests/test_timed_execution.py`.

Public capability boundary (planned; not currently importable):

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

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.execution.timed import materialize_grips
self.assertEqual(materialize_grips([-1, 1, 0, 1], 2), (0, 0, 0, 0))
self.assertEqual(materialize_grips([1, -1, -1, 1], 2), (1, 1, 0, 0))
with self.assertRaises(ValueError):
    materialize_grips([1], 0)
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_timed_execution.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
result = []
for start in range(0, len(grips), r):
    held = int(grips[start] > 0)
    result.extend([held] * min(r, len(grips) - start))
return tuple(result)
```

Add a noncommuting rotated-root fixture proving targets are root@A[j], not A[j]@root or cumulative multiplication. Enforce B=1, 1<=r<=h<=P, positive finite duration and owned command arrays. Same materializer is mandatory in collection, reference and all added controls.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_timed_execution.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 2: Bounded interval stepping with separate clocks

**Files:** Create `src/icgs/environments/rlbench/timed.py`.
**Test owner:** `tests/test_timed_execution.py`.
**Consumes / produces:** Produces `interval_substeps(duration_s, physics_dt)` and timed reset/advance/close; consumes explicit controller capability.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.environments.rlbench.timed import interval_substeps
self.assertEqual(interval_substeps(0.1, 0.005), 20)
with self.assertRaisesRegex(ValueError, 'interval'):
    interval_substeps(0.1, 0.03)
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_timed_execution.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
ratio = duration_s / physics_dt
count = round(ratio)
if count < 1 or not math.isclose(ratio, count, rel_tol=0, abs_tol=1e-9):
    raise ValueError('interval must match an integer physics-step count')
return count
```

Set one target at interval start, hold it while stepping exactly the configured substeps, then read measured observation and increment boundary once. Detect unsupported low-level IK/controller stepping instead of wrapping pose-reaching task.step as fixed time. Use bounded reset attempts, finally-close, controller-status diagnostics and explicit simulated versus wall timestamps.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_timed_execution.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 3: Replay does not claim inaccessible hidden state

**Files:** Create `src/icgs/environments/rlbench/replay.py`.
**Test owner:** `tests/test_timed_execution.py`.
**Consumes / produces:** Produces `ReplayReport` and `validate_replay_report(report)`; consumes snapshots or reset+recorded-command history, never online task features.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.environments.rlbench.replay import validate_replay_report
with self.assertRaisesRegex(ValueError, 'report'):
    validate_replay_report(True)
with self.assertRaisesRegex(ValueError, 'mode|fields'):
    validate_replay_report({'mode': 'exact-snapshot'})
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_timed_execution.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
# P01 defines the report schema and restoration capability only.
# P08 implements engine restoration and produces measured discrepancies.
report = dict(mode=mode, stored_fields=stored_fields,
              restored_fields=restored_fields, discrepancies=measurements,
              protocol_id=protocol_id)
validate_replay_report(report)
```

Report joints/velocities, articulations, constraints/attachments, integrators, monitor history, RNG and solver coverage explicitly. If exact restore is unavailable use deterministic history replay; if equivalence still fails tag approximate-reset with repeated-reset uncertainty. Never relabel an approximate counterfactual exact.

This task validates the capability/report boundary without implementing or calling
restoration. P08 owns the engine-specific restoration implementation and replay
pilot; P01 owns target-hold stepping and sensor/timing traces.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_timed_execution.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
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

- Documentation drafting: this plan specifies future work only.
- Component RED/GREEN commands: **NOT RUN** — runtime/test files are not implemented.
- L1 model assertions: **NOT RUN** — future installed supported environment required.
- L2/C1–C5: **NOT RUN** by this plan — preserve mandatory integration acceptance.
- L3/L4, collection and training: **NOT RUN** — separate resource authorization required.
- Remaining risk: RLBench/PyRep low-level stepping and snapshot completeness remain measured feasibility questions.
