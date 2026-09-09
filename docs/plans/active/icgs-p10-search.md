# P10: Expected-return belief search and matched planners Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Implement reranking, shooting and progressive-widening MCTS over common commands with honest budgets.

**Architecture:** Planning algorithms depend on capabilities, not concrete IP or environment. A root-local belief node tracks absorbed terminal mass and three fixed-head states; exact caches reuse transitions without inventing visits.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Planning and evaluation](../../method/planning-evaluation.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — planned runtime NOT IMPLEMENTED**. This document is a category C
component of the approved docs-only research migration, not permission to execute it.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00/P04/P06/P07/P09; deterministic capability fixtures first, trained artifact pilot only after FG bridge/dynamics/value.
Read [workflow](../../WORKFLOW.md), [architecture](../../ARCHITECTURE.md),
[baseline](../../baselines/instant_policy.md), [native contract](../../components/policy-data-contract.md),
[research policy](../../RESEARCH.md) and [validation guide](../../../tests/README.md)
before runtime execution. Inspect Git state; preserve unrelated work and user assets.

## Configuration consumption addendum — 2026-09-09

Default values now belong to the [packaged primary JSON](../../../src/icgs/configuration/profiles/icgs_primary.json),
with key/unit/restriction details in the [parameter reference](../../method/parameters.md).
Consumed sections for this plan: **planning, control, numerics**.

Use config.planning widening/UCT, medoid scales, budget panels and horizons. Use numerics tolerances for mass checks. Derived counts and terminal-mass rules are not independent tuneable settings. Record all effective budgets with the full resolved configuration.

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

- Create: `src/icgs/algorithms/planning/belief.py` — hypotheses, nodes, masses and medoid.
- Create: `src/icgs/algorithms/planning/mcts.py`, `shooting.py`, `rerank.py`.
- Create: `src/icgs/algorithms/planning/budget.py` — clocks/counters/fallback.
- Test: `tests/test_search.py`.

Public capability boundary (planned; not currently importable):

```python
plan(state: PhysicalState, task: TaskState, context: MethodContext, *, H: int, budget: PlanningBudget) -> PlanningResult
propagate_mass(U, F, weights, event_probabilities) -> tuple
leaf_return(U, weights, active_values, remaining) -> float
```

Root tau0,U=S,F0,w=(1-U)/3. Each edge advances min(h,H_root-tau,L-tau). Every interval predicts b,tracks q,predicts terminal,then updates root-relative masses. No max-over-head outcomes, leaf completion bonus, approximate merging or previous-root W reuse.

### Task 1: Mass conservation, deadline return and medoid commands

**Files:** Create `src/icgs/algorithms/planning/belief.py`.
**Test owner:** `tests/test_search.py`.
**Consumes / produces:** Produces propagate_mass, leaf_return, Hypothesis, BeliefNode and representative selection.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import numpy as np
from icgs.algorithms.planning.belief import propagate_mass, leaf_return
u, f, w = propagate_mass(.1, .0, np.array([.3, .3, .3]),
                         np.array([[.2,.1,.7]]*3))
self.assertAlmostEqual(u + f + w.sum(), 1.)
self.assertAlmostEqual(leaf_return(u, w, np.ones(3), 0), u)
self.assertAlmostEqual(leaf_return(u, w, np.ones(3), 3), u+w.sum())
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_search.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
  *Verified: 7 ran, 1 failed, 6 errored (`NotImplementedError: propagate_mass is not implemented yet`).*
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
success, failure, cont = event_probabilities.T
next_U = U + (weights*success).sum()
next_F = F + (weights*failure).sum()
next_weights = weights*cont
# Validate probabilities and total mass; never renormalize an invalid prediction.
```

FP32 mass tolerance1e-5,FP64 tests1e-12; zero active mass means exactly zero. Medoid minimizes active-head pairwise CD/.01²+translation/.01²+angle²/(5deg)², unweighted and tie lower head. Materialize candidate once from medoid root and apply identical absolute bytes to every head.
*Implemented in `src/icgs/algorithms/planning/belief.py` with `propagate_mass`, `leaf_return`, `Hypothesis`, `BeliefNode`, and `select_representative`.*

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_search.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
  *Verified: 13 ran, 13 passed, 0 failed, 0 skipped in 0.217s. L0 fast validation passed 19/19 (selected=19, executed=19, skipped=0) on both python3 -B and python3 -B -S scripts/validate_fast.py. Historical full suite regression results at pre-fix commit fc80203: selected=302, executed=293, skipped=9; all 293 executed tests passed (9 skipped due to optional external dependencies: 4 differential baseline tests requiring IP_LEGACY_SOURCE_ROOT, 1 RLBench test requiring RLBench, 1 CUDA RNG test requiring CUDA, 1 opt-in published checkpoint test, and 2 PyTorch Lightning tests requiring lightning); full suite was NOT RERUN after 6966fcf or Fix Round 2.*
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 2: Progressive widening, selection, backup and exact cache

**Files:** Create `src/icgs/algorithms/planning/mcts.py`.
**Test owner:** `tests/test_search.py`.
**Consumes / produces:** Produces `widening_limit(visits)`, `uct(total_return, visits, parent_visits)` and MCTS plan capability.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.algorithms.planning.mcts import widening_limit, uct
self.assertEqual(widening_limit(0), 1)
self.assertEqual(widening_limit(3), 3)
with self.assertRaisesRegex(ValueError, 'unvisited'):
    uct(0., 0, 1)
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_search.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
  *Verified RED: 14 ran, 13 passed, 1 errored (`NotImplementedError: widening_limit is not implemented yet`) in 0.230s via `.venv/bin/python -B -m unittest discover -s tests -p 'test_search.py' -v`.*
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
limit = max(1, math.floor(cfg.planning.widening_coefficient *
                          (1+node.visits)**cfg.planning.widening_exponent))
# Expand one edge if under limit; otherwise unvisited first, then Q+UCT bonus.
G = leaf_return(leaf.U, leaf.weights, values, H_root-leaf.tau)
for edge in path:
    edge.total_return += G
    edge.visits += 1
# Increment each visited node once; cache lookup alone adds no visit.
```

Tie selection by insertion ID; final choice visits,then Q,then earlier ID. Cache requires byte-identical canonical commands/durations plus parent/context/head/model/time IDs, root-local only. Keep duplicate samples in audit. Tests use a hand-enumerated two-depth tree to check one backup, visit accounting, partial edges and task-history cache separation.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_search.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
  *Verified GREEN: 23 ran, 23 passed, 0 failed, 0 skipped in 0.485s via `.venv/bin/python -B -m unittest discover -s tests -p 'test_search.py' -v`. L0 fast validation passed 19/19 on both `python3 -B scripts/validate_fast.py` and `python3 -B -S scripts/validate_fast.py`.*
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.
  *Verified exact source diff in mcts.py, budget.py, tests/test_search.py, active plan. Progressive widening, UCT, root-scoped exact cache, one G backup, node/edge visit semantics, and deterministic tie breaking all pass.*
  - **Task 2 Cumulative Audit & Fix Trail:**
    - **Original Implementation (`72ba6e8`):** Progressive widening MCTS + root-scoped exact cache (implemented then; accepted only at `cfe6693` after 3 audit fixes). Evidence: 23 ran, 23 passed via `.venv/bin/python -B -m unittest discover -s tests -p 'test_search.py' -v`. L0 passed 19/19 on both `python3 -B scripts/validate_fast.py` and `python3 -B -S scripts/validate_fast.py`.
    - **Fix Round 1 (`0317f75`):** Addressed six initial audit findings (composite models, task identity, state identity across 13 fields, operation-level budget checks, structured recorder events). Evidence: 29 ran, 29 passed (0.524s) via `.venv`. Both L0 passed 19/19.
    - **Fix Round 2 (`e0d5a0e`):** Addressed six re-audit findings (ExactCache task branch copy on all representations, wall/native_call budget invariant, full native Observation, MethodConfig requirement without dt0 fallback, scalar calibration [0, 1] validation, depth-2 tree & real tie tests). Evidence: 29 ran, 29 passed (0.550s) via `.venv`. Both L0 passed 19/19.
    - **Fix Round 3 (`cfe6693`):** Addressed two new findings from re-audit 2 (early-root evaluation sets completed=True independent of U, zero_horizon ordered before absorbed_root; explicit FG pilot NOT RUN/not accepted; durable active plan evidence updated without rewriting historical runs). Accepted at `cfe6693`. Evidence: 29 ran, 29 passed (0.515s) via `.venv/bin/python -B -m unittest discover -s tests -p 'test_search.py' -v`. Both L0 passed 19/19.

### Task 3: Matched baselines, completed-only timing and fallback

**Files:** Create shooting/rerank/budget modules.
**Test owner:** `tests/test_search.py`.
**Consumes / produces:** Produces `eligible_completion(finished_at, deadline)` and matched plan adapters; consumes monotonic clock and capability counters.

- [x] **Step 1 \u2014 RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.algorithms.planning.budget import eligible_completion
self.assertTrue(eligible_completion(0.5, 0.5))
self.assertFalse(eligible_completion(0.5001, 0.5))
```

- [x] **Step 2 \u2014 Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_search.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
  *Verified RED: 30 ran, 29 passed, 1 errored (`NotImplementedError: eligible_completion is not implemented yet`) in 0.521s via `.venv/bin/python -B -m unittest discover -s tests -p 'test_search.py' -v`.*
- [x] **Step 3 \u2014 GREEN:** Implement the boundary using this algorithm/code sketch.

```python
if clock() >= deadline or counters.at_cap():
    break
result = noninterruptible_operation()
finished = clock()
audit.record(result, finished, overshoot=max(0., finished-deadline))
if finished <= deadline:
    completed.append(result)
```

Count encoding/native preprocessing/diffusion/transfer/synchronization/decode/reencode/evaluation. Rerank h2/T8; shooting resamples prior through L and ties first completed. If no eligible evaluation reuse earliest prior sample or draw exactly one new reference fallback, record extra latency; invalid input aborts. Match wall0.1/.5/2s,L and diagnostic native caps16/64/256 plus independent model counters.
*Implemented: Completed `src/icgs/algorithms/planning/budget.py` with `eligible_completion`, `synchronize_device`, `NonfiniteModelError`, and `execute_fallback`; completed B5 rerank in `src/icgs/algorithms/planning/rerank.py` (configured h and native T8 rollouts); completed B6 shooting in `src/icgs/algorithms/planning/shooting.py` (sequential resampling through L); integrated monotonic wall deadline gating, completed-only timing, device synchronization, overshoot recording, and fallback handling across MCTS, rerank, and shooting.*

- [x] **Step 4 \u2014 Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_search.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
  *Verified GREEN: 39 ran, 39 passed, 0 failed, 0 skipped in 0.783s via `.venv/bin/python -B -m unittest discover -s tests -p 'test_search.py' -v`. L0 fast validation passed 19/19 on both `python3 -B scripts/validate_fast.py` and `python3 -B -S scripts/validate_fast.py`. Full test suite: 333 tests selected/executed: 324 passed, 9 skipped (optional dependencies: 4 differential baseline, 1 RLBench, 1 CUDA RNG, 1 opt-in published checkpoint, 2 PyTorch Lightning), 0 failed in 14.700s.*
- [x] **Step 5 \u2014 Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.
  *Verified exact diff: completed budget.py, rerank.py, shooting.py, mcts.py completed-only wall timing and fallback integration, comprehensive deterministic test suite in tests/test_search.py, clean L0/full test results.*

## Acceptance, resource limits and evidence

L1 hand-computed trees validate conservation/return/visits/ties/cache/caps/fallback and root/branch isolation. FG planner pilot reports completed candidates, actual boundaries crossed, optimism, support, latency/memory and B6/B7 comparisons under approved caps. Correct MCTS arithmetic is not evidence of H2.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_search.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

No simulator import, privileged goal, diffusion density/PUCT assumption, risk coefficient or stochastic outcome cherry-picking. Discard tree values after real execution; keep failed model traces and reference fallback counts.

On failure, retain the failed evidence and isolate the new component/config ID.
Do not overwrite native checkpoints, relabel collected outcomes or drop difficult samples.
Resume only from a compatible explicit artifact/manifest; changed reference lineage
requires new outcome labels, not a cache workaround. Return to the preceding accepted
phase if an FG fails and request a scoped protocol decision.

## Execution evidence

- Documentation drafting: complete.
- Component RED/GREEN commands: Task 1 (belief.py) accepted at a430cbf; Task 2 (mcts.py, budget.py seam) implemented at 72ba6e8 (23 passed), accepted only cfe6693 after 3 audit fixes (29 passed in 0.515s); Task 3 (budget.py, rerank.py, shooting.py, mcts.py completed-only wall timing and fallback) completed (39 passed in 0.783s); Task 3 Fix Round 1 (BASE b0d7ccb) addressed 6 audit findings (46 passed in 1.421s); Task 3 Fix Round 2 (BASE b1d152c) addressed 3 regressions and 3 residual audit findings (52 passed in 2.156s); Task 3 Fix Round 3 (BASE f9bb329) resolved remaining audit findings: parameterized fallback timed_operation(permit_beyond_deadline=True) with uniform start/attempt/sync/error machinery beyond deadline, preserved primary op error on op+sync double failure with sync_error audit, nonmutating root and predicted state validation via dataclasses.replace() preserving tensor data_ptr, observation points mask filtering strictly excluding padding, multi-planner behavioral coverage across mcts/rerank/shooting, and per-planner algorithm_params envelope audit (52 passed in 2.892s). Affected contract and observability integration tests passed (20 passed in 0.552s). L0 passed 19/19 on scripts/validate_fast.py (both invocations).
- L1 search assertions: 52 executed, 52 passed in tests/test_search.py (2.892s) using .venv/bin/python (historical 23-test, 29-test, 39-test, 46-test, and 52-test runs preserved).
- Full regression suite: 333 selected, 324 executed, 9 skipped (4 differential baseline requiring legacy source root, 1 RLBench not installed, 1 CUDA RNG requiring CUDA, 1 opt-in published checkpoint, 2 Lightning not installed), 0 failed in 14.700s (historical provenance at b0d7ccb).
- L2/C1–C5: **NOT RUN** by this component task — required Feasibility Gate (FG) remains active and FG pilot is **NOT RUN / not accepted**.
- L3/L4, collection and training: **NOT RUN** — separate resource authorization required.
- Remaining risk: Nonpreemptible IP may exceed small budgets and deeper model rollout may exploit prediction errors.

## Observability integration addendum — 2026-09-09

The current sequential candidate seam emits `candidate.created` with supplied
index, seed, source/artifact IDs, batch/horizon and measured seconds. It does not
emit U/F masses, visits, ranking, tie breaks or fallback claims. Those records
remain P10-owned and are not inferred from the selected action.
