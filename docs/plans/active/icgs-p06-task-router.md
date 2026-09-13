# P06: Task memory, event routing and frozen reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Track full causal task history and route frozen IP between full context and valid event windows.

**Architecture:** A task neural module owns recurrent tensors; task state owns context/boundary lineage. Router algorithms consume native proposal capability through separately owned D1/D2 sessions; they never mutate native graph configuration.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Neural method](../../method/neural.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — Task 1A tensor tracker, Task 1B owned TaskState, Task 1C
masked objectives and Task 2 deterministic routing/window selection COMPLETE;
P06 remains PARTIAL because native context materialization, sessions, seeded
sampling and reference calls are NOT IMPLEMENTED**. This
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
Consumed sections for this plan: **tracker, router, event, memory, neural, losses,
control, planning**.

Consume tracker layers, task-loss weights, router mixture/epsilon/fallback/window
settings and passed native sessions. Reference fingerprints include every
behavior-affecting resolved section plus weights/protocol IDs. Do not put stopping
or evaluator options into the reference fingerprint.

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

Implemented/planned capability vocabulary (Task 3 surfaces remain unavailable):

```python
track_task(previous: TaskState | None, state: PhysicalState, events: EventMemory) -> TaskState
sample_prior(observation, task: TaskState, context: MethodContext, *, seed: int) -> Candidate
router_probabilities(event_alpha, eligible, native_window_valid, config) -> Tensor
select_window_indices(target, demo_interactions, grip_transition_indices, config,
                      *, native_waypoint_count) -> tuple[int, ...] | None
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
**Status:** COMPLETE. Convert transient logits into masked
probabilities, own `r`, alignment including null, rho/nu/eligible and exact
tracker/context/boundary lineage. Reset initializes `r=0`; each real boundary
updates once. A context change requires explicit full-history replay by its
orchestration owner rather than an in-place lineage swap.

**Accepted boundary record:** `TaskState` owns cloned-but-not-detached tensors and
contains no second event-valid mask. Its intrinsic validation covers compatible
`[B,W]`, `[B,Lc+1]` and `[B,Lc]` shapes, finite probabilities in `[0,1]`,
nonnegative alignment whose rows sum to one within a dtype-derived numerical
tolerance, a nonnegative causal boundary, ordered context fingerprints and a
tracker identifier. `build_task_state` alone validates the mask-dependent
contract against `EventMemory.valid`, derives ordered context fingerprints from
the event-memory owner, writes invalid event probabilities as exact zero and
materializes the learned null/event probabilities from raw Task 1A logits.
`rho`, `nu` and `eligible` remain independent; recovery and `rho=1, nu=0` are
legal. `validate_task_update` requires unchanged B/W/L shape, event-memory
fingerprints and tracker lineage followed by exactly one boundary. Context replay
is deliberately deferred to orchestration.

- [x] **Step 1 — RED:** Add focused allowed and forbidden fixtures for reset,
  probability materialization, null fallback, intrinsic shapes/ranges, exact
  padding zeros, derived lineage, tensor ownership/autograd, branch-copy isolation
  and update continuity.
- [x] **Step 2 — Verify RED:** Run `.venv/bin/python -B -m unittest discover -s
  tests -p 'test_task_router.py' -v`; existing Task 1A tests must remain green and
  every new Task 1B failure must identify only the absent planned module/surface.
- [x] **Step 3 — GREEN:** Implement `TaskState`, `initial_task_memory`,
  `build_task_state` and `validate_task_update` without orchestration, objectives,
  routing or native-policy dependencies.
- [x] **Step 4 — Verify GREEN:** Repeat the focused suite, related state/model
  regressions, `py_compile`, `git diff --check`, and both L0 variants. Report
  pre-existing documentation-link failures separately from new regressions.
- [x] **Step 5 — Review:** Inspect the exact source/test/contract/plan diff and
  leave Task 1C, router and native sessions deferred.

Task 1B completion wording: **Task 1B owned TaskState — COMPLETE; supervised
objectives, routing, replay orchestration and native sessions — deferred.** P06
remains PARTIAL.

### Task 1C: Independently masked supervised objectives

**Files:** Create `src/icgs/algorithms/objectives/task.py`.
**Test owner:** `tests/test_task_router.py`.
**Status:** COMPLETE. Consume raw Task 1A logits and separate P02
alignment, rho/nu/eligibility targets plus validity masks. Free-space invalid `nu`
must not mask valid eligibility; no privileged target enters TaskTracker or
TaskState inference.

**Accepted boundary record:** `task_loss` consumes a caller-supplied soft alignment
distribution `[B,Lc+1]`, with null last, and boolean rho/nu/eligibility targets
`[B,Lc]`. It validates and uses a valid multi-support distribution exactly; it
does not infer corresponding events, enforce uniform semantic mass, or tensorize
P02 annotations. Valid alignment rows are finite, nonnegative, sum to one within
a dtype-derived numerical guard and assign exact-zero mass to invalid event rows;
masked alignment rows are the all-zero placeholder. Each auxiliary target uses an
independent boolean supervision mask contained by `TaskEncoding.event_valid`, and
every masked target uses the canonical false placeholder. Alignment CE and the
three BCE-with-logits terms average their valid entries independently before the
four explicit `MethodConfig.losses` weights are applied. An empty pool contributes
a differentiable zero through its own logits. Alignment targets may use a different
floating dtype from logits but every input tensor shares one device. The objective
knows only tensors, `MethodConfig` and raw `TaskEncoding`; TaskState, event-memory
lineage, data tensorization, routing, native policy and training orchestration are
outside this task.

- [x] **Step 1 — RED:** Add lazy-import tests for the exact four-term formula,
  multi-support/null alignment, independent masks and denominators, canonical
  placeholders, empty-pool zero, gradients and strict tensor validation.
- [x] **Step 2 — Verify RED:** Run `.venv/bin/python -B -m unittest discover -s
  tests -p 'test_task_router.py' -v`; all 23 Task 1A/1B tests must pass and every
  new Task 1C test must fail only because the objective module is absent.
- [x] **Step 3 — GREEN:** Implement the narrow objective and package export without
  a target builder, TaskState conversion, dataset adapter, runner or router.
- [x] **Step 4 — Verify GREEN:** Repeat the focused suite, P02/config/world-model
  and existing P05/P06 regressions, `py_compile`, `git diff --check`, and both L0
  variants. Record selected/executed/skipped counts and pre-existing link failures.
- [x] **Step 5 — Review:** Inspect the exact source/test/plan diff and leave P02
  task-view tensorization, Stage B training, routing and native sessions deferred.

Task 1C completion wording: **Task 1C independently masked objectives — COMPLETE;
P02 task-view tensorization, Stage B training integration, routing and native
sessions — deferred.** P06 remains PARTIAL.

### Task 2: Valid windows and 50/50 routed mixture

**Status:** COMPLETE. Batched route arithmetic and deterministic structural
window indices only; native context materialization and routed policy execution
remain Task 3.

**Files:** Create `src/icgs/algorithms/planning/router.py`.
**Test owner:** `tests/test_task_router.py`.
**Consumes / produces:** `router_probabilities` consumes event-only alignment
probabilities (`TaskState.alpha[..., :-1]`), eligibility and the final derived
`MethodContext.native_window_valid`; it returns full-context followed by
event-window probabilities. `select_window_indices` consumes one target plus the
complete interaction partition for that target's single demo and P05-owned
debounced grip-transition indices; it returns exact native frame indices or
`None` when a well-formed window cannot fit the native waypoint budget.

Task 2A is batched routing arithmetic only. Task 2B is deterministic structural
index selection only. Neither task constructs `PreparedContext`, calls native
preprocessing or Instant Policy, owns native sessions, or splits seeds.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import torch
from icgs.algorithms.planning.router import router_probabilities
from icgs.configuration.method import MethodConfig

cfg = MethodConfig()
p = router_probabilities(
    torch.tensor([[.4, .6]]),
    torch.ones(1, 2),
    torch.tensor([[True, False]]),
    cfg,
)
torch.testing.assert_close(p, torch.tensor([[.5, .5, 0.]]))
f = router_probabilities(
    torch.zeros(1, 2),
    torch.zeros(1, 2),
    torch.ones(1, 2, dtype=torch.bool),
    cfg.router,
)
torch.testing.assert_close(f, torch.tensor([[1., 0., 0.]]))
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_task_router.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
v = torch.where(
    native_window_valid,
    (event_alpha + cfg.probability_epsilon) * eligible,
    0.0,
)
mass = v.sum(dim=-1, keepdim=True)
fallback = mass < cfg.fallback_threshold
safe_mass = torch.where(fallback, torch.ones_like(mass), mass)
normalized = v / safe_mass
routed = (1.0 - cfg.full_context_probability) * normalized
probs = torch.cat(
    (torch.full_like(mass, cfg.full_context_probability), routed), dim=-1
)
full_fallback = torch.cat((torch.ones_like(mass), torch.zeros_like(v)), dim=-1)
return torch.where(fallback, full_fallback, probs)
```

`event_alpha`, `eligible` and `native_window_valid` have shape `[B,L]`; the
returned tensor has shape `[B,L+1]` with full context at slot zero. Fallback is
per row, the threshold comparison is strict `<`, invalid-window probabilities
and gradients are exact zero, and configuration is always an explicitly resolved
`MethodConfig` or injected `RouterConfig`. Both `probability_epsilon` and
`fallback_threshold` are strictly positive, matching central `MethodConfig`
validation; this is not a router-local fallback default.
Normalize before multiplying by the routed-mixture mass so subnormal FP16 values
are not rounded or underflowed before their common mass is divided out.

For Task 2B, every `demo_interactions` entry must be an interaction with the
target's exact `demo_content_hash`; target occurs exactly once. Entries are in
chronological order and form a contiguous shared-endpoint partition:
`current.b == next.a`. Gaps, interior overlap and zero-length interactions are
malformed and reject. Grip-transition indices are sorted unique nonnegative
integers, reject booleans, lie in the full demo range and come from P05's
`debounced_grip_boundaries`; P06 does not reimplement debounce.

The window includes target plus configured immediate previous/next interactions.
Mandatory indices are every selected interaction endpoint plus every supplied
debounced grip-transition index inside the selected span, deduplicated. The
native waypoint count is a required positive integer injected from the selected
native graph configuration, never a method default. More mandatory indices than
the budget, or fewer unique span frames than the budget, returns `None`; malformed
input raises.

Let sorted unused indices be `U` of length `M`, and let `K` slots remain after
mandatory indices. For `K=1`, choose rank `(M-1)//2`. For `K>1`, rank `i` is the
nearest integer to `i*(M-1)/(K-1)` with exact half ties going to the lower rank:

```python
lower, remainder = divmod(i * (M - 1), K - 1)
rank = lower + int(2 * remainder > K - 1)
```

Canonical examples are `U=[2,4,6,8], K=1 -> [4]`,
`U=[1,3,5,7,9,11,13], K=4 -> [1,5,9,13]`, and
`U=[2,4,6,8], K=3 -> [2,4,8]`. The final tuple is unique and chronological.
Invalid windows fall back through the mixture; invalid full context aborts setup.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_task_router.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

Task 2 completion wording: **Task 2 deterministic routing math and structural
window selection — COMPLETE; native `PreparedContext` materialization, D1/D2
sessions, seed splitting, frozen-IP calls, candidate/audit construction and
end-to-end final-availability derivation — deferred.** P06 remains PARTIAL.

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

Task 1A/1B/1C L1 checks raw tensor features, explicit configuration, query/key
masks, finite sentinels/zeros, full physical-token pooling, gradients, demo-block
permutation, owned task probabilities, causal/context lineage and independently
masked task objectives. Later P06 L1 adds route probabilities, window boundary
counts, context replay and no mutation of native config. L2 FG one-demo verifies
strict published
load/inference for D1 and independent D2; capped L3 pilot measures stage-aware
behavior before outcome collection.

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
- Task 1B RED: the same focused command — **FAIL as expected**, 23 selected and
  executed: all 10 existing Task 1A tests passed and all 13 new Task 1B tests
  raised only `ModuleNotFoundError` for the absent `icgs.state.task`; 0 skips.
- Task 1B GREEN: the same command after implementation and final numerical guard
  refinement — **PASS**, 23/23 selected/executed/passed, 0 failures/errors/skips.
  Coverage includes allowed and forbidden mask/shape/lineage cases, exact padding
  zeros, independent rho/nu/eligible, all-invalid null normalization, derived
  fingerprints, storage ownership, branch copying, nonzero weighted-alignment
  gradient, all three event-head gradient paths, and exact-zero padding gradients.
- The first focused run after the gradient refinement passed 22/23 and exposed an
  indexing error in the new assertion: an `[B,L]` mask was applied to `[B,L+1]`
  alignment logits. Restricting that assertion to the event portion `[:, :-1]`
  corrected the test harness; the subsequent focused run passed 23/23. No runtime
  implementation changed during this correction.
- Task 1C RED: the focused command selected/executed 31 tests — **FAIL as
  expected**: all 23 Task 1A/1B tests passed and all 8 new Task 1C tests raised
  only `ModuleNotFoundError` for absent `icgs.algorithms.objectives.task`; 0 skips.
- Task 1C initial GREEN: the same command after implementation and refinement —
  **PASS**, 31/31 selected/executed/passed with 0 failures/errors/skips. Coverage includes
  exact configured soft-CE plus three-BCE arithmetic with four independent
  denominators, multi-support/null targets, canonical placeholders, free-space
  nu versus eligibility separation, differentiable empty pools, device/dtype/shape
  diagnostics, four supervised gradient paths and exact-zero masked gradients.
- The first Task 1C GREEN run passed 30/31 and exposed an overbroad static-test
  substring: `annotations` matched Python's `from __future__ import annotations`.
  The guard was narrowed to the forbidden data-module path while its AST import
  allowlist was expanded to cover both `import` and `from ... import`; the next
  run passed 31/31. No objective arithmetic changed during this correction.
- Task 1C pre-commit refinement: the focused command — **PASS**, 32/32
  selected/executed/passed with 0 failures/errors/skips. The added malformed raw
  `TaskEncoding` test covers event-logit shape, event-valid dtype, valid nonfinite
  alignment/event logits, padding sentinel and padding-zero violations. Formula
  fixtures now use distinct auxiliary denominator counts 4/3/2, and one mixed
  alignment batch row is masked with an all-zero target and exact-zero gradient.
  Objective arithmetic was unchanged.
- Task 2 RED: the focused command selected/executed 45 tests — **FAIL as
  expected**: all 32 Task 1A/1B/1C tests passed and all 13 new Task 2 tests raised
  only `ModuleNotFoundError` for absent `icgs.algorithms.planning.router`; 0
  skips. This established the missing implementation surface without weakening
  earlier task evidence.
- Task 2 initial GREEN: the focused command passed 44/45. All numerical,
  validation and structural behavior passed; the only failure was the static
  dependency fixture omitting standard-library `math` from its allowlist while
  the implementation used `math.isfinite` for scalar config validation. Adding
  `math` to that test allowlist corrected the harness; no router behavior changed.
- Task 2 final GREEN after adding the P05/P06 boundary fixture: the focused
  command — **PASS**, 46/46 selected/executed/passed with 0
  failures/errors/skips. Coverage includes per-row fallback, strict threshold,
  exact invalid-window probabilities/gradients, malformed tensor/config inputs,
  shared-endpoint partitions, gap/interior-overlap/zero-length rejection,
  same-demo target ownership, exact uniform-rank tie breaking, impossible-window
  `None`, edge-bounded neighbors and consumption of P05 debounce confirmation
  indices without a second debounce implementation.
- Task 2 pre-commit numerical review added a CPU-FP16 subnormal-mass fixture,
  explicit zero-epsilon rejection and direct input non-mutation proof. The first
  focused run selected/executed 48 tests: 47 passed and the new FP16 fixture
  failed as intended because clamping mass to `torch.float16.tiny` changed a
  non-fallback denominator. Replacing the fallback-only divisor with
  `where(fallback, 1, mass)` exposed a second FP16 operation-order issue:
  multiplying subnormal `v` by the mixture mass before division still rounded
  the window probability to 0.4707. Normalizing `v/mass` first and multiplying
  the resulting distribution second fixed the invariant. The final focused
  command — **PASS**, 48/48 selected/executed/passed with 0
  failures/errors/skips. `probability_epsilon > 0` remains deliberate and now
  test-locked because central `MethodConfig` already requires it.
- One proposed `memory_slots=True` negative fixture failed inside the existing
  typed `TrackerConfig` constructor before reaching TaskTracker. It was removed as
  duplicate schema coverage and was not counted as Task 1A RED evidence.
- Related L1 regressions: `test_event_memory.py` **PASS** 29/29,
  `test_physical_memory.py` **PASS** 17/17, `test_method_config.py` **PASS** 24/24,
  `test_method_contracts.py` **PASS** 17/17, `test_architecture.py` **PASS** 3/3,
  and `test_policy.py` **PASS** 6/6; 96/96 total with 0 skips.
- Task 1C also ran `test_episode_data.py` **PASS** 13/13 and
  `test_world_model.py` **PASS** 28/28. Together with the 96 tests above, related
  regressions passed 137/137 with 0 skips.
- Task 2 related regressions reran `test_episode_data.py` **PASS** 13/13,
  `test_world_model.py` **PASS** 28/28, `test_event_memory.py` **PASS** 29/29,
  `test_physical_memory.py` **PASS** 17/17, `test_method_config.py` **PASS**
  24/24, `test_method_contracts.py` **PASS** 17/17, `test_architecture.py`
  **PASS** 3/3 and canonical-discovery `test_policy.py` **PASS** 6/6: 137/137
  related tests passed with 0 skips. A combined positional unittest invocation
  first passed 131 assertions but could not collect `test_policy.py` because its
  sibling `test_composition` import was outside that invocation's module path;
  rerunning that owner through canonical discovery passed 6/6. This collection
  error is not counted as runtime regression evidence.
- A noncanonical positional unittest invocation executed the first 90 related
  assertions successfully, then failed to collect `test_policy.py` because its
  sibling `test_composition` import was not on the discovery path. The canonical
  `discover -s tests -p 'test_policy.py'` correction passed 6/6; the invocation
  error is not counted as a runtime regression.
- `.venv/bin/python -B -m py_compile src/icgs/state/task.py
  tests/test_task_router.py` and `git diff --check` — **PASS**.
- `.venv/bin/python -B -m py_compile src/icgs/algorithms/objectives/task.py
  src/icgs/algorithms/objectives/__init__.py tests/test_task_router.py`,
  `git diff --check`, and the changed-file trailing-whitespace scan — **PASS**.
- Task 2 `.venv/bin/python -B -m py_compile
  src/icgs/algorithms/planning/router.py tests/test_task_router.py`, `git diff
  --check`, and the changed-file trailing-whitespace scan — **PASS**.
- L0: `.venv/bin/python -B scripts/validate_fast.py` and `.venv/bin/python -B -S
  scripts/validate_fast.py` — **FAIL** overall with no new failure versus `273b7b3`.
  Both variants passed Python syntax, static harness boundary, and 19/19 harness
  self-tests with 0 skips; the only failures remain the same five pre-existing
  missing evidence-log links under `docs/experiments/vv19-validation`.
- Task 2 L0 rerun: both commands remain **FAIL** overall with exactly those same
  five pre-existing missing evidence-log targets. Both passed Python syntax,
  static harness boundary and 19/19 harness self-tests with 0 skips; no new L0
  failure was introduced relative to `bc019c9`.
- L2/C1–C5: **NOT RUN** by this plan — preserve mandatory integration acceptance.
- L3/L4, simulator, preprocessing, collection and training: **NOT RUN** — separate
  resource authorization required.
- Remaining risk: Tasks 1A/1B/1C/2 have primarily synthetic CPU/float32 evidence;
  Task 2 additionally has one CPU-FP16 subnormal normalization regression. In
  particular, the dtype-derived TaskState alpha and Task 1C alignment-target
  normalization tolerances have not been validated for float16/bfloat16. Strict
  validation also uses host-reading `.item()` checks that may synchronize each GPU
  batch; before real Stage B training, decide from measurement whether those checks
  remain on the hot path or move partly to dataset/debug validation. P02 task-view
  tensorization, Stage B training, GPU/mixed-precision behavior, context replay
  orchestration, native window materialization/final-availability integration,
  routed policy execution, native sessions and measured reference support remain
  unverified.
