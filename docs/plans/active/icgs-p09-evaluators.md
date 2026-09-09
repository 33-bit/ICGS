# P09: Continuation value, completion and first-terminal evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Estimate calibrated finite-deadline reference success while keeping completion and progress semantically separate.

**Architecture:** Shared-weight but independent single-query attention streams read all valid event requirements and physical/task state. A separate transition head predicts mutually exclusive first hazards; objective algorithms consume executed labels only.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Neural and evaluation losses](../../method/neural.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — planned runtime NOT IMPLEMENTED**. This document is a category C
component of the approved docs-only research migration, not permission to execute it.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00/P04–P08, frozen reference path and valid outcome/terminal annotations.
Read [workflow](../../WORKFLOW.md), [architecture](../../ARCHITECTURE.md),
[baseline](../../baselines/instant_policy.md), [native contract](../../components/policy-data-contract.md),
[research policy](../../RESEARCH.md) and [validation guide](../../../tests/README.md)
before runtime execution. Inspect Git state; preserve unrelated work and user assets.

## Configuration consumption addendum — 2026-09-09

Default values now belong to the [packaged primary JSON](../../../src/icgs/configuration/profiles/icgs_primary.json),
with key/unit/restriction details in the [parameter reference](../../method/parameters.md).
Consumed sections for this plan: **evaluator, neural, losses, calibration, numerics**.

Parameterize MLP/block configuration, loss weights and scalar-temperature fitting from these sections. Three terminal outcomes, active V0=0 and no double completion remain semantic invariants. Keep learned stopping outside reference collection.

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

- Create: `src/icgs/models/evaluators/continuation.py` — V/S/progress queries.
- Create: `src/icgs/models/evaluators/terminal.py` — first-event head.
- Create: `src/icgs/algorithms/objectives/evaluation.py` — count/pair/masked losses.
- Create: `src/icgs/algorithms/calibration.py`; Test: `tests/test_evaluators.py`.

Public capability boundary (planned; not currently importable):

```python
evaluate_state(state: PhysicalState, task: TaskState, events: EventMemory, H: int) -> EvaluationOutput
predict_terminal(before: PhysicalState, q_before: TaskState, after: PhysicalState, q_after: TaskState, events: EventMemory, command: TimedCommand) -> TerminalProbabilities
outcome_loss(logits, k, n) -> Tensor
```

V_H estimates frozen pi_ref success for active states; V0 exactly0. S estimates already complete and Phi is auxiliary only. Completion/progress cannot read H indirectly; no query-to-query attention, no hard mask of completed requirements and no extra S added at search leaves.

### Task 1: Independent deadline-conditioned and deadline-free queries

**Files:** Create `src/icgs/models/evaluators/continuation.py`.
**Test owner:** `tests/test_evaluators.py`.
**Consumes / produces:** Produces `horizon_features(H, Hmax)` and ContinuationEvaluator implementing evaluate_state.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import torch
from icgs.models.evaluators.continuation import horizon_features
h = horizon_features(torch.tensor([0, 512]), 512)
torch.testing.assert_close(h, torch.tensor([[0., 0.], [1., 1.]]))
with self.assertRaises(ValueError):
    horizon_features(torch.tensor([513]), 512)
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_evaluators.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
keys = events.tokens + status_mlp(torch.stack((alpha, rho, nu, eligible), -1))
qv, qs, qp = e_v + Wr(r) + horizon_mlp(h), e_s + Wr(r), e_phi + Wr(r)
# Apply the same three block weights separately to each one-query stream.
V = torch.where(H > 0, torch.sigmoid(logit_v / Tv), 0.)
```

Keys include all valid events plus physical tokens/task r; status MLP4→256→256, query heads256→128→1. Test changing H leaves S/Phi bitwise or numerically unchanged while V may change; reject negative/noninteger/uncovered H. Expose logits and calibrated outputs; nondecreasing V versus H is diagnostic, not falsely guaranteed.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_evaluators.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 2: Active first-terminal probabilities and count likelihood

**Files:** Create terminal module and evaluation objectives.
**Test owner:** `tests/test_evaluators.py`.
**Consumes / produces:** Produces `terminal_features` of1288 dimensions, predict_terminal and outcome_loss.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import torch
from icgs.algorithms.objectives.evaluation import outcome_loss
logits = torch.zeros(2, requires_grad=True)
loss = outcome_loss(logits, torch.tensor([1., 0.]), torch.tensor([2., 1.]))
self.assertAlmostEqual(loss.item(), 0.69314718, places=6)
loss.backward()
self.assertTrue(torch.isfinite(logits.grad).all().item())
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_evaluators.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
eta = torch.cat((mean_S_before, r_before, mean_S_after, r_after, mean_events, u), -1)
raw_event_logits = terminal_mlp(eta)
probabilities = torch.softmax(raw_event_logits / Te, -1)
# D2 CE consumes raw_event_logits; inference output records the Te artifact ID.
loss = (n * F.binary_cross_entropy_with_logits(logits, k/n, reduction='none')).sum()/n.sum()
```

MLP1288→512→256→3; validate finite normalized S/F/C and active before-state. Future fields are one executed successor for training or model successor for search, never real-root future input. Require n>0 and 0<=k<=n; terminal prefix is legal Q target but not active successor V target.

Test Te=1 versus Te=2 changes only calibrated probabilities, keeps raw logits
unchanged and preserves probability sum1; reject missing/incompatible Te lineage.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_evaluators.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 3: Auxiliary masks, separate calibration and freeze checks

**Files:** Complete evaluation objectives and `src/icgs/algorithms/calibration.py`.
**Test owner:** `tests/test_evaluators.py`.
**Consumes / produces:** Produces `weighted_pair_loss(delta_logits, labels, weights)` and `fit_temperature(logits, targets, weights)`.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import torch
from icgs.algorithms.objectives.evaluation import weighted_pair_loss
x = torch.empty(0, requires_grad=True)
loss = weighted_pair_loss(x, torch.empty(0), torch.empty(0))
self.assertEqual(loss.item(), 0.)
loss.backward()
self.assertIsNotNone(x.grad)
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_evaluators.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
if weights.numel() == 0 or weights.sum() == 0:
    return delta_logits.sum()*0.
pair = (weights * F.binary_cross_entropy_with_logits(delta_logits, labels,
                                                    reduction='none')).sum()/weights.sum()
# Total = outcome+completion+event+.1*progress+.2*(branch+recovery+suffix).
```

Progress pairs require strict satisfied-set inclusion without losses, not timestamps. Calibrate Tv/Ts/Te independently with exp(logT),FP64 deterministic LBFGS,max100,strong-Wolfe; nonfinite fit retains identity with failure report. Train D2 only evaluator/hazard weights; test upstream hashes/grad absence and calibration split isolation.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_evaluators.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

## Acceptance, resource limits and evidence

L1 checks architecture H isolation, masks, stable count/pair losses, finite gradients and frozen upstream modules. FG value uses development binomial NLL/trial Brier/ECE10, finite-pool regret and all-zero support; calibration and imagined-state optimism reported separately. No achieved prediction quality is implied by unit tests.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_evaluators.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

No progress reward, expert-target mixing, hard suffix-reversal labels or learned stop inside reference. Recollect outcomes after any reference lineage change; keep evaluator/reference mismatch a hard failure.

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
- Remaining risk: Low-sample labels and planner distribution shift can yield miscalibration even under correct likelihood semantics.

## Observability integration addendum — 2026-09-09

P09 value/completion/terminal producers are not implemented. The recorder has no
authority to infer V/S/progress/terminal probabilities or calibration IDs from
an action; those fields remain future evaluator-owned evidence.
