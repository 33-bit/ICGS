# P08: Executed branch bank and exact-reference outcomes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Collect audited branch/continuation labels for the frozen reference without learned-stop contamination.

**Architecture:** Collection composes timed execution, replay and external monitors; it never fabricates a successor from model prediction. Dataset algorithms derive horizons and uncertainty-qualified disjoint pair pools from retained physical trial traces.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Data and training](../../method/data-training.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — planned runtime NOT IMPLEMENTED**. This document is a category C
component of the approved docs-only research migration, not permission to execute it.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P01/P02/P06; frozen reference manifest, FG replay/assets/timing and explicitly approved six-program pilot caps.
Read [workflow](../../WORKFLOW.md), [architecture](../../ARCHITECTURE.md),
[baseline](../../baselines/instant_policy.md), [native contract](../../components/policy-data-contract.md),
[research policy](../../RESEARCH.md) and [validation guide](../../../tests/README.md)
before runtime execution. Inspect Git state; preserve unrelated work and user assets.

## Configuration consumption addendum — 2026-09-09

Default values now belong to the [packaged primary JSON](../../../src/icgs/configuration/profiles/icgs_primary.json),
with key/unit/restriction details in the [parameter reference](../../method/parameters.md).
Consumed sections for this plan: **dataset, collection, losses, sensors, benchmark**.

Use named pilot/primary quotas, prefix/horizon lists and pair confidence/prior/quadrature settings. Require seed/replay/resource metadata before collection. Keep first-terminal/coverage semantics invariant and full config hash separate from frozen reference ID.

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

- Create: `src/icgs/data/collection/outcomes.py` — anchors, branches and reference trials.
- Modify: `src/icgs/environments/rlbench/replay.py` — implement the P01 restoration capability.
- Create: `src/icgs/data/collection/pairs.py` — suffix/recovery grouping and uncertainty.
- Create: `src/icgs/data/datasets/outcomes.py` — horizon labels and trial dependence.
- Test: `tests/test_outcome_collection.py`.

Public capability boundary (planned; not currently importable):

```python
collect_branch(anchor_ref, context, candidate, *, replay, environment, reference, monitor, limits) -> BranchRecord
counts_at_horizon(trials: list[dict], H: int) -> tuple[int, int]
pair_probability(k_a: int, n_a: int, k_b: int, n_b: int) -> float
```

Reference collections terminate only on external first terminal or deadline; calibrated S never stops these trials. Terminal prefixes get realized1/0 but no active successor V label. Trials contain exact reference ID, seeds, actual commands/timing/replay mode and first terminal times.

### Task 1: First-terminal outcomes, deadlines and censoring

**Files:** Create `src/icgs/data/datasets/outcomes.py`.
**Test owner:** `tests/test_outcome_collection.py`.
**Consumes / produces:** Produces counts_at_horizon; consumes trial terminal/censor records with relative interval times.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.data.datasets.outcomes import counts_at_horizon
trials = [dict(first_success=3, first_failure=None, censored=False, observed_through=3),
          dict(first_success=None, first_failure=2, censored=False, observed_through=2),
          dict(first_success=None, first_failure=None, censored=True, observed_through=1)]
self.assertEqual(counts_at_horizon(trials, 4), (1, 2))
self.assertEqual(counts_at_horizon(trials, 2), (0, 2))
timeout = dict(first_success=None, first_failure=None, censored=False, observed_through=32)
self.assertEqual(counts_at_horizon([timeout], 32), (0, 1))
self.assertEqual(counts_at_horizon([timeout], 512), (0, 0))
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_outcome_collection.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
eligible = [t for t in trials if terminal_observed(t) or H <= t['observed_through']]
k = sum(t['first_success'] is not None and t['first_success'] <= H
        and (t['first_failure'] is None or t['first_success'] < t['first_failure'])
        for t in eligible)
return k, len(eligible)
```

Implement terminal_observed in the same file for physical first success/failure;
failure wins simultaneous first event. Store observation-end boundary and collection
deadline explicitly. Timeout gives return0 only through observed coverage, not an
absorbing physical failure or permission to extrapolate to a longer H. Infrastructure
interruption is censored beyond observed coverage; an already observed physical
terminal determines longer horizons. Compute n(H) per eligible coverage, retain
n=0 groups but exclude them from value loss. Store shared trial-group IDs for
H32/64/128/256/512; never add horizons as independent trials.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_outcome_collection.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 2: Replay and physically execute shared prefixes

**Files:** Create `src/icgs/data/collection/outcomes.py`; implement restoration in `src/icgs/environments/rlbench/replay.py`.
**Test owner:** `tests/test_outcome_collection.py`.
**Consumes / produces:** Produces `require_reference(reference_id, manifest_id)` and collect_branch; consumes P01 ReplayReport and P06 sample_prior.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.data.collection.outcomes import require_reference
require_reference('ref-a', 'ref-a')
with self.assertRaisesRegex(ValueError, 'reference'):
    require_reference('ref-a', 'ref-b')
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_outcome_collection.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
observation, replay_report = replay.restore(anchor_ref)
# Reconstruct b,q from the complete recorded history and verify anchor boundary.
for command in candidate.commands:
    transition = environment.advance(command)
    annotation = monitor.annotate(transition)
    writer.append(transition, annotation, replay_report)
    if annotation.first_terminal is not None:
        break
```

At active successors restore and run n independent reference trials; no learned stop, V-ranking or expert mixing. Primary K8,n4,prefix2/8/16 uniformly chosen feasible length per anchor; pilot K4,n2. Long prefixes resample IP from actual observed groups, then store commands for exact reuse. Caps terminate collection cleanly and retain partial/censored manifests.

Implement engine snapshot restore for actually exposed fields, otherwise reset
and deterministic action-history replay; produce P01's ReplayReport. Execute the
same prefix twice only in the authorized G5 pilot and record pose/cloud/outcome
discrepancies. No inaccessible solver field is reported restored.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_outcome_collection.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 3: Suffix/recovery evidence and pair uncertainty

**Files:** Create `src/icgs/data/collection/pairs.py`.
**Test owner:** `tests/test_outcome_collection.py`.
**Consumes / produces:** Produces pair_probability, `pair_pool(is_suffix, is_recovery)` and confidence filtering; consumes independent branch counts.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.data.collection.pairs import pair_probability, pair_pool
self.assertAlmostEqual(pair_probability(2, 4, 2, 4), 0.5, places=7)
self.assertEqual(pair_pool(True, True), 'suffix')
self.assertEqual(pair_pool(False, True), 'recovery')
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_outcome_collection.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
prior_a, prior_b = cfg.losses.pair_beta_prior_alpha, cfg.losses.pair_beta_prior_beta
p = scipy.integrate.quad(lambda x: scipy.stats.beta.pdf(x, k_a+prior_a, n_a-k_a+prior_b)
                         * scipy.stats.beta.cdf(x, k_b+prior_a, n_b-k_b+prior_b),
                         0., 1., epsabs=cfg.losses.pair_quadrature_atol,
                         epsrel=cfg.losses.pair_quadrature_rtol)[0]
keep, weight = max(p, 1-p) >= cfg.losses.pair_confidence, 2*abs(p-.5)
```

Shared random numbers require paired bootstrap, not independent Beta. Recompute q by full-history replay for each suffix and collect separate continuations; do not force reversal. Pool precedence suffix→recovery→general; empty pools remain valid. Preserve all-zero groups and local failures. Recovery tag requires executed successful recovery, not distance.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_outcome_collection.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

## Acceptance, resource limits and evidence

L1 verifies count arithmetic, censoring, terminal-prefix exclusion, reference mismatch, pair independence and context-swapped label nonreuse. Authorized pilot ceiling200 anchor-contexts×4 branches×2 continuations=1600 maximum trials, not automatic launch. FG outcome reports replay fidelity, support, all-zero rate, reversals, runtime/storage and owner-approved scaling decision.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_outcome_collection.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

No model-generated physical outcome labels, reference changes, discarded failures or rewritten posterior thresholds. Approximate replay stays tagged; failed feasibility returns to P01/P06 without expanding collection.

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
- Remaining risk: Weak reference support, scarce confident reversal pairs and expensive replay may block meaningful primary-scale evidence.

## Observability integration addendum — 2026-09-09

P08 remains a future producer of attempt, branch, replay discrepancy and
observed-through records. The current recorder can store explicit links supplied
by an eventual collector, but it does not restore simulators, deserialize replay
artifacts or assert outcome causes.
