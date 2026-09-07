# P13: Method composition, observed-root lifecycle and acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Connect approved components into explicit method entry points while preserving installed native inference and mandatory published fidelity.

**Architecture:** Outer composition loads versioned artifacts and exposes method controls; execution owns real boundary updates and separate stopping. Existing native build_policy/commands remain behaviorally unchanged and independently installed.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Method contracts and roadmap](../../method/contracts.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — planned runtime NOT IMPLEMENTED**. This document is a category C
component of the approved docs-only research migration, not permission to execute it.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00–P12 implementations, required FG reports and declared supported environment; no missing gate may be replaced by a skip.
Read [workflow](../../WORKFLOW.md), [architecture](../../ARCHITECTURE.md),
[baseline](../../baselines/instant_policy.md), [native contract](../../components/policy-data-contract.md),
[research policy](../../RESEARCH.md) and [validation guide](../../../tests/README.md)
before runtime execution. Inspect Git state; preserve unrelated work and user assets.

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

- Modify: `src/icgs/composition.py` — additive build_method only.
- Create: `src/icgs/execution/method.py` and `src/icgs/execution/stopping.py`.
- Create: `src/icgs/cli/method.py`; Modify: `src/icgs/cli/main.py` — explicit new command registration.
- Extend: `src/icgs/artifacts/method.py` — full compatibility manifest.
- Test: `tests/test_method_integration.py`.

Public capability boundary (planned; not currently importable):

```python
build_method(config: MethodConfig, factories=None) -> MethodRuntime
MethodRuntime.prepare_context(raw_demos) -> MethodContext
MethodRuntime.observe(observation: TimedObservation, previous_command=None) -> tuple[PhysicalState, TaskState]
MethodRuntime.plan(*, H: int, budget: PlanningBudget) -> PlanningResult
CompletionGate.update(boundary: int, calibrated_completion: float) -> bool
```

One real observation boundary updates b then q once, stop counter once, and planner reads cached real root. Execute up to r and remaining deadline, updating history after each interval; imagined states never replace measured root. Reference collection uses ungated action policy.

### Task 1: Explicit composition and pre-load compatibility rejection

**Files:** Add build_method to composition; complete artifact manifest.
**Test owner:** `tests/test_method_integration.py`.
**Consumes / produces:** Produces build_method and `validate_component_ids(requested, available)`; consumes MethodConfig and all P00 capability adapters.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.artifacts.method import validate_component_ids
with self.assertRaisesRegex(ValueError, 'unknown'):
    validate_component_ids({'dynamics': 'missing'}, {'dynamics': {'three-head-v1'}})
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_integration.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
# Validate schema, component IDs, shapes, coverage and reference/evaluator IDs first.
# Construct modules through immutable explicit factory maps; then load tensors strictly.
# Construct D1/D2 native sessions once from the same immutable published profile.
# Expose only protocols to execution/planning; no neural module performs artifact IO.
```

Test loader spy gets zero reads on invalid config, mismatched native hash/reference ID,missing component/horizon. Verify native factories and build_policy state-dict keys unchanged. Separate geometry/event/dynamics/evaluator/student/stop artifact IDs, training manifest and resolved config; reject stale cached state before online entry.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_integration.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 2: Observed-boundary lifecycle and separate learned stopping

**Files:** Create `src/icgs/execution/method.py` and `stopping.py`.
**Test owner:** `tests/test_method_integration.py`.
**Consumes / produces:** Produces CompletionGate and MethodRuntime.observe/plan using timed environment/capabilities.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.execution.stopping import CompletionGate
g = CompletionGate(threshold=.95, required=3)
self.assertFalse(g.update(0, .96))
self.assertFalse(g.update(0, .96))
self.assertFalse(g.update(1, .96))
self.assertTrue(g.update(2, .96))
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_integration.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
if boundary == last_boundary:
    return stopped
if boundary != last_boundary + 1:
    raise ValueError('stop gate requires consecutive observed boundaries')
streak = streak + 1 if math.isfinite(S) and S >= threshold else 0
stopped = streak >= required
last_boundary = boundary
```

Initialize last_boundary=-1,streak0; invalid observation resets streak and terminates added rollout with invalid-input tag. After each advance update real b/q; replanning after r reads state without encoding last boundary again. Context swap replays q from full same physical history. Fallback allowed only for valid real/context; reject imagined-origin root writes. Stop gate is B1–B7 only,never collection reference.

MethodRuntime.observe may accept previous_command at the orchestration boundary,
but computes its descriptor against the retained BEFORE pose before calling
physical_update(encoded, observation, previous_descriptor, memory). Never rebuild
the descriptor from the newly observed successor pose.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_integration.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 3: Installed CLI, bounded dry validation and full acceptance matrix

**Files:** Create method CLI; add explicit registration in `src/icgs/cli/main.py` and integration tests.
**Test owner:** `tests/test_method_integration.py`.
**Consumes / produces:** Produces `validate_method_request(mode, config_path, limits)` and installed method validate/infer/collect/train/evaluate modes with explicit paths.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.cli.method import validate_method_request
with self.assertRaisesRegex(ValueError, 'limits|scope'):
    validate_method_request('collect', 'method.json', limits=None)
with self.assertRaisesRegex(ValueError, 'limits|scope'):
    validate_method_request('train', 'method.json', limits=None)
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_integration.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
# Parse lightweight help without importing simulator or loading weights.
# validate mode checks config/manifests only; it does not prepare data or run inference.
# Mutating/compute modes require explicit resource limits and prerequisite FG manifests.
# Always close environment in finally; write audit results atomically.
```

Use tiny installed subprocess fixtures outside checkout for help/config failure and capability integration. Add deterministic2-interval end-to-end public-seam test for real-root/candidate/three-head/search/execute/new-root and no oracle leakage. Preserve native installed infer/evaluate semantics and C1 metadata,C2 strict load,C3 inference,C4 external reference fidelity,C5 standalone execution.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_integration.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

## Acceptance, resource limits and evidence

L0 plus full installed L1 discovery must report counts/skips. Mandatory C1–C5 use exact documented published commands in tests/README.md and existing acceptance evidence under pinned Linux Python3.10/CUDA11.8; no substitute checkpoint or fake modules. Capped L3 method pilot must pass all prerequisite FG reports before L4; documentation alone never closes runtime roadmap.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_method_integration.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

Keep native path selectable and byte-preserved; additive method failure cannot silently fall back to a different scientific protocol. No runtime import of docs/tests/scripts, no package relocation or new aliases. Roll back only new component selection/artifacts, preserving user files and failed evidence.

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
- Remaining risk: Native compatibility and end-to-end learned behavior require actual supported-environment and simulator acceptance, not documentation or local L0.
