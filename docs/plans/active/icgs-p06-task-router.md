# P06: Task memory, event routing and frozen reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Track full causal task history and route frozen IP between full context and valid event windows.

**Architecture:** A task neural module owns recurrent tensors; task state owns context/boundary lineage. Router algorithms consume native proposal capability through separately owned D1/D2 sessions; they never mutate native graph configuration.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Neural method](../../method/neural.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — Task 1A tensor tracker COMPLETE; P06 remains PARTIAL because
TaskState, objectives, routing and native sessions are NOT IMPLEMENTED**. This
document is a category C component of the approved research migration; it does not
authorize simulator or training workloads.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00/P03–P05; P02 supervised labels; FG native one-demo/window
compatibility before reference freeze. P05 Task 3A ownership is available at
`64402db`.
Read [workflow](../../WORKFLOW.md), [architecture](../../ARCHITECTURE.md),
[baseline](../../baselines/instant_policy.md), [native contract](../../components/policy-data-contract.md),
[research policy](../../RESEARCH.md) and [validation guide](../../../tests/README.md)
before runtime execution. Inspect Git state; preserve unrelated work and user assets.

## Configuration consumption addendum — 2026-09-09

Default values now belong to the [packaged primary JSON](../../../src/icgs/configuration/profiles/icgs_primary.json),
with key/unit/restriction details in the [parameter reference](../../method/parameters.md).
Consumed sections for this plan: **tracker, router, event, memory, neural, control, planning**.

Consume tracker layers, router mixture/epsilon/fallback/window settings and passed native sessions. Reference fingerprints include every behavior-affecting resolved section plus weights/protocol IDs. Do not put stopping or evaluator options into the reference fingerprint.

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

- Create: `src/icgs/models/memories/task.py` — two attention blocks, GRU and event heads.
- Create: `src/icgs/state/task.py` — TaskState and context replay validation.
- Create: `src/icgs/algorithms/planning/router.py` — window selection and route RNG.
- Create: `src/icgs/algorithms/objectives/task.py`; Test: `tests/test_task_router.py`.

Public capability boundary (planned; not currently importable):

```python
track_task(previous: TaskState | None, state: PhysicalState, events: EventMemory) -> TaskState
sample_prior(observation, task: TaskState, context: MethodContext, *, seed: int) -> Candidate
router_probabilities(alpha, eligible, valid_windows) -> Tensor
select_window_indices(a, b, neighbors, grips, count: int=10) -> tuple[int, ...]
```

TaskState r[B,W],alpha[B,Lc+1] including null,rho/nu/eligible[B,Lc],boundary/context/tracker lineage. Nonmonotonic recovery is legal. Reference pi_ref is exactly one routed sample plus r2 execution, with no V,H-conditioned choice or learned stop.

### Task 1A: Tensor-only recurrent tracker and raw prediction heads

**Files:** Create `src/icgs/models/memories/task.py`; extend
`src/icgs/models/layers/method.py` with generic masked cross-attention.
**Test owner:** `tests/test_task_router.py`.
**Consumes / produces:** Produces `task_event_features(events, r)`, transient
`TaskEncoding`, and tensor-only `TaskTracker`. TaskState construction, probabilities,
boundary/context lineage, replay and supervised objectives remain Tasks 1B/1C.

**Accepted boundary record:** `TaskTracker` requires one explicit resolved
`MethodConfig` and consumes only physical tokens/mask, event tokens/mask and
`previous_r`. Physical tokens are the complete P04 set—geometry, proprioception and
physical-memory rows—not only geometry anchors. Valid inputs must be finite; masked
padding may be nonfinite but is sanitized before LayerNorm, pooling or attention.
Cross-attention uses separate pre-normalized query and key/value tensors, no geometry
bias, and zeros masked query rows after residual boundaries. Output remains raw,
finite and differentiable: invalid event alignment logits equal
`torch.finfo(dtype).min`, the last null logit is finite, and invalid event-head rows
are exact zero. Zero event logits are not labels; Task 1C must still apply independent
supervision masks. Demo-block permutation leaves `r` and null invariant within
numerical tolerance and permutes event/alignment rows correspondingly. The model
module must not import TaskState, MethodContext, SegmentRef, native IP, program
labels, objectives or routing.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import torch
from icgs.models.memories.task import task_event_features
M = torch.ones(1, 4, 256)
r = torch.zeros(1, 256)
y = task_event_features(M, r)
self.assertEqual(tuple(y.shape), (1, 4, 768))
torch.testing.assert_close(y[..., 512:], torch.zeros(1, 4, 256))
```

- [x] **Step 2 — Verify RED:** The supported `.venv` command selected/executed 8
  tests with 0 skips and produced 8 expected errors: seven for the absent
  `icgs.models.memories.task` module and one for the absent generic
  `MaskedCrossAttentionBlock`. Test collection and installed dependencies succeeded;
  no unrelated failure was counted as RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
z = (r_previous + scene_projection(masked_mean(S, S_valid, 1)))[:, None]
query_valid = ones([B,1], bool)
for block in blocks:
    z = block(z, query_valid, keys, key_valid)
r_input = z[:, 0]
r = gru(r_input, r_previous)
event_features = torch.cat((M, r[:, None].expand_as(M), M*r[:, None]), -1)
```

Compute dot-product event alignment plus one learned null logit and raw
`3*width→width→3` event logits. Do not apply sigmoid or softmax in Task 1A; Task 1B
materializes probabilities and Task 1C consumes logits. Sanitize invalid rows before
all computation, then write finite mask sentinels/zeros into output. Landmarks remain
tracker inputs. Derive dimensions from the resolved configuration and reject
incompatible geometry/memory/event widths, tracker memory slots other than one,
nonpositive attention-layer count, or a width not divisible by neural attention
heads. Use every physical token in the scene mean. Do not update causal boundaries
or implement context replay here.

- [x] **Step 4 — Verify GREEN:** The supported `.venv` command passed 10/10
  selected/executed tests with 0 failures/errors/skips. Refinement coverage locks
  shape/dtype/all-invalid-physical diagnostics, all-invalid-event null behavior,
  zero padding gradients, complete finite parameter gradients, and confirms no
  early Sigmoid/Softmax module.
- [x] **Step 5 — Review:** Inspected the exact source/test/plan diff, explicit
  configuration and width checks, masks before LayerNorm/pooling/attention,
  gradients, demo-block permutation, source boundary, `py_compile`, and
  `git diff --check`. No native IP, TaskState, objective, routing, simulator,
  preprocessing, collection or training owner was changed. No commit was made.

Task 1A completion wording: **Task 1A tensor tracker — COMPLETE; TaskState,
supervised objectives, routing and native sessions — deferred.** P06 remains PARTIAL.

### Task 1B: Owned TaskState, causal boundary and context lineage

**Files:** Create `src/icgs/state/task.py`.
**Test owner:** `tests/test_task_router.py`.
**Status:** Deferred. Convert transient logits into masked probabilities, own `r`,
alignment including null, rho/nu/eligibility and exact tracker/context/boundary
lineage. Reset initializes `r=0`; each real boundary updates once. A context change
requires explicit full-history replay by its orchestration owner rather than an
in-place lineage swap.

### Task 1C: Independently masked supervised objectives

**Files:** Create `src/icgs/algorithms/objectives/task.py`.
**Test owner:** `tests/test_task_router.py`.
**Status:** Deferred. Consume raw Task 1A logits and separate P02 alignment,
rho/nu/eligibility targets plus validity masks. Free-space invalid `nu` must not mask
valid eligibility; no privileged target enters TaskTracker or TaskState inference.

### Task 2: Valid windows and 50/50 routed mixture

**Files:** Create `src/icgs/algorithms/planning/router.py`.
**Test owner:** `tests/test_task_router.py`.
**Consumes / produces:** Produces `router_probabilities` returning full-context followed by event-window probabilities.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import torch
from icgs.algorithms.planning.router import router_probabilities
p = router_probabilities(torch.tensor([.4, .6]), torch.ones(2),
                         torch.tensor([True, False]))
torch.testing.assert_close(p, torch.tensor([.5, .5, 0.]))
f = router_probabilities(torch.zeros(2), torch.zeros(2), torch.ones(2,dtype=torch.bool))
torch.testing.assert_close(f, torch.tensor([1., 0., 0.]))
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_task_router.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
v = torch.where(valid_windows, (alpha + cfg.router.probability_epsilon)*eligible, 0.)
if v.sum() < cfg.router.fallback_threshold:
    return torch.cat((v.new_ones(1), v.new_zeros(v.numel())))
p_full = cfg.router.full_context_probability
return torch.cat((v.new_tensor([p_full]), (1-p_full)*v/v.sum()))
```

Window includes same-demo event and immediate previous/next interactions. Preserve unique endpoints/grip transitions; >10 mandatory or <10 unique frames makes invalid. Fill evenly spaced unused ranks, tie earlier, sort chronologically. Invalid window falls back through mixture; invalid full context aborts setup.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_task_router.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 3: Reference sessions, seeds and immutable fingerprint

**Files:** Complete router and `src/icgs/state/task.py`.
**Test owner:** `tests/test_task_router.py`.
**Consumes / produces:** Produces `split_route_seed(seed)` returning route/diffusion seeds; sample_prior consumes native predict/prepare capabilities.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.algorithms.planning.router import split_route_seed
route, diffusion = split_route_seed(17)
self.assertNotEqual(route, diffusion)
self.assertEqual((route, diffusion), split_route_seed(17))
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_task_router.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
children = np.random.SeedSequence(seed).spawn(2)
route_seed, diffusion_seed = [int(c.generate_state(1)[0]) for c in children]
# Draw route only with route_seed; execute one native scoped-seed call with diffusion_seed.
# Persist chosen full/window ID, both seeds, raw candidate and native artifact IDs.
```

Independently construct D1 and D2 native sessions from identical frozen weights once per method; use correctly owned PreparedContexts, no branch-time loads or concurrent mutable scratch. Preserve native RNG draw order and global restoration even on failure. Freeze physical/event/task path, calibration and cadence before C; changing any reference field invalidates outcomes.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_task_router.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

## Acceptance, resource limits and evidence

Task 1A L1 checks raw tensor features, explicit configuration, query/key masks,
finite sentinels/zeros, full physical-token pooling, gradients and demo-block
permutation. Later P06 L1 adds route probabilities, all-invalid fallback, window
boundary counts, context replay and no mutation of native config. L2 FG one-demo
verifies strict published load/inference for D1 and independent D2; capped L3 pilot
measures stage-aware behavior before outcome collection.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_task_router.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

No learned stopping or evaluator in pi_ref; no deadline input to routed sampling. Native IP stays frozen; event tokens never enter its graph. Preserve unsuccessful routing outcomes and invalid-window counts.

On failure, retain the failed evidence and isolate the new component/config ID.
Do not overwrite native checkpoints, relabel collected outcomes or drop difficult samples.
Resume only from a compatible explicit artifact/manifest; changed reference lineage
requires new outcome labels, not a cache workaround. Return to the preceding accepted
phase if an FG fails and request a scoped protocol decision.

## Execution evidence

- Environment: repository `.venv/bin/python`, CPython 3.10.20, editable install of
  the current checkout; cwd repository root.
- Task 1A RED: `.venv/bin/python -B -m unittest discover -s tests -p
  'test_task_router.py' -v` — **FAIL as expected**, 8 selected/executed, 8 missing
  planned-surface errors, 0 skips. Seven named the absent task-memory module and one
  named the absent cross-attention export.
- Task 1A GREEN: the same command after implementation and the final coverage
  refinement — **PASS**, 10 selected/executed/passed, 0 failures/errors/skips.
- One proposed `memory_slots=True` negative fixture failed inside the existing
  typed `TrackerConfig` constructor before reaching TaskTracker. It was removed as
  duplicate schema coverage and was not counted as Task 1A RED evidence.
- Related L1 regressions: `test_event_memory.py` **PASS** 29/29,
  `test_physical_memory.py` **PASS** 17/17, `test_method_config.py` **PASS** 24/24,
  `test_method_contracts.py` **PASS** 17/17, `test_architecture.py` **PASS** 3/3,
  and `test_policy.py` **PASS** 6/6; 96/96 total with 0 skips.
- L0: `.venv/bin/python -B scripts/validate_fast.py` and `.venv/bin/python -B -S
  scripts/validate_fast.py` — **FAIL** overall with no new failure versus `64402db`.
  Both variants passed Python syntax, static harness boundary, and 19/19 harness
  self-tests with 0 skips; the only failures remain the same five pre-existing
  missing evidence-log links under `docs/experiments/vv19-validation`.
- L2/C1–C5: **NOT RUN** by this plan — preserve mandatory integration acceptance.
- L3/L4, simulator, preprocessing, collection and training: **NOT RUN** — separate
  resource authorization required.
- Remaining risk: Task 1A has synthetic CPU evidence only. Owned task state,
  probability materialization, masked objectives, GPU/mixed-precision behavior,
  context replay, routing, native sessions and measured reference support remain
  unverified.
