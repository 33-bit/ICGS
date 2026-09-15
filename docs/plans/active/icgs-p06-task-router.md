# P06: Task memory, event routing and frozen reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Track full causal task history and route frozen IP between full context and valid event windows.

**Architecture:** A task neural module owns recurrent tensors; task state owns context/boundary lineage. Router algorithms consume native proposal capability through separately owned D1/D2 sessions; they never mutate native graph configuration.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Neural method](../../method/neural.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — Task 1A tensor tracker, Task 1B owned TaskState, Task 1C
masked objectives, Task 2 deterministic routing/window selection, Task 3A route
RNG protocol and Task 3B.0 context-preparation RNG/provenance foundation
and Task 3B.1 exact-index/full-demo materialization COMPLETE; P06 remains PARTIAL
because Task 3B.2a injected-session validation and Task 3B.2b authoritative
profile/outer D1-D2 construction are COMPLETE under synthetic L1, while Task
3B.3 session/context ownership and Task 3C routed reference calls are NOT
IMPLEMENTED**. This
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
- Create for Task 3B/3C: `src/icgs/policies/reference.py` — exact-index
  materialization orchestration, injected D1/D2 session ownership, MethodContext
  assembly and routed reference proposals. This module may consume the pure
  router, InstantPolicy capability and method state; `router.py` must not import
  concrete policies, contexts or sessions.
- Extend only at the outer construction boundary: `src/icgs/composition.py` builds
  and loads D1/D2 policies once from the same frozen artifact. Extend
  `src/icgs/artifacts/method.py` rather than creating another reference hash.
- New Task 3B/3C test owner: `tests/test_reference_policy.py`; keep pure arithmetic
  and dependency tests in `tests/test_task_router.py`.

Implemented/planned capability vocabulary (Task 3B.2b profile/session factory is
available; Task 3B.3/3C surfaces remain unavailable):

```python
track_task(previous: TaskState | None, state: PhysicalState, events: EventMemory) -> TaskState
router_probabilities(event_alpha, eligible, native_window_valid, config) -> Tensor
select_window_indices(target, demo_interactions, grip_transition_indices, config,
                      *, native_waypoint_count) -> tuple[int, ...] | None
split_route_seed(seed: int) -> tuple[int, int]  # route seed, diffusion seed
draw_route(probabilities: Tensor, *, route_seed: int) -> int
split_context_seeds(context_seed: int, *, demo_count: int,
                    event_count: int) -> tuple[tuple[int, ...], tuple[int, ...]]
ContextPreparationRecord(...)  # immutable validated preparation provenance
materialize_indexed_native_demo(demo, boundary_indices, *, point_seed,
                                native_waypoint_count,
                                native_point_count) -> Mapping[str, tuple]
materialize_full_native_demo(demo, *, point_seed, native_waypoint_count,
                             native_point_count) -> Mapping[str, tuple]
ReferenceSessions(...)  # frozen validation-only D1/D2 ownership record
resolve_native_profile(profile_id)  # exact published profile; no aliases
build_reference_sessions(checkpoint, *, reference_id, native_profile, device)

# Planned and unavailable until Task 3C; P10 treats proposals as opaque and the
# P06-owned materializer unwraps proposal.candidate:
sample_prior(observation, task: TaskState, context: MethodContext,
             *, seed: int) -> ReferenceProposal
materialize_prefix(proposal: ReferenceProposal, *, h: int, r: int,
                   duration_s: float) -> CommandPrefix
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
sessions, frozen-IP calls, candidate/audit construction and end-to-end
final-availability derivation — deferred.** Deterministic route RNG belongs to
Task 3A rather than Task 2. P06 remains PARTIAL.

### Task 3A: Deterministic route RNG protocol

**Status:** COMPLETE. This task contains seed splitting and a local categorical
route draw only. It does not materialize native contexts or call a policy.

**Files:** Extend `src/icgs/algorithms/planning/router.py`.
**Test owner:** `tests/test_task_router.py`.
**Consumes / produces:** `split_route_seed(seed)` returns the exact route and
diffusion child integers produced by `SeedSequence(seed).spawn(2)`.
`draw_route(probabilities, *, route_seed)` consumes one non-batched
`[full, window_0, ...]` distribution and returns one route index under protocol
`pcg64-v1`.

The seed input is a nonnegative integer with booleans rejected. Child streams are
not post-processed to force pairwise inequality: the standard `SeedSequence`
outputs are authoritative. `draw_route` requires a nonempty one-dimensional
floating tensor with finite nonnegative entries and sum one within the same
dtype-derived numerical guard used for task probability state. Invalid input is
rejected outside that guard. Accepted near-unit distributions are canonicalized
on a float64 copy for categorical sampling; their relative and zero masses remain
unchanged and the caller's tensor is never mutated.

Categorical sampling uses a local
`numpy.random.Generator(numpy.random.PCG64(route_seed))`. It computes the CDF on
a detached, canonicalized CPU float64 copy and sets only `cdf[-1] = 1.0` after
validation and canonicalization so roundoff cannot produce an out-of-range index.
`searchsorted(..., side="right")` skips zero-mass entries, including entries at
an exact CDF boundary. Neither success nor validation failure mutates global
Python, NumPy or Torch RNG state.
Changing PCG64 or categorical boundary semantics requires a new behavior-affecting
route RNG protocol ID and, once Task 3C owns reference integration, new reference
lineage.

- [x] **Step 1 — RED:** Add focused seed, categorical, malformed-input,
  zero-mass, global-RNG and dependency-boundary fixtures.
- [x] **Step 2 — Verify RED:** Run `.venv/bin/python -B -m unittest discover -s
  tests -p 'test_task_router.py' -v`. Existing 48 Task 1/2 tests must remain
  green; every Task 3A failure must name only the absent planned surface.
- [x] **Step 3 — GREEN:** Implement only `ROUTE_RNG_PROTOCOL`,
  `split_route_seed` and `draw_route`; do not add session/context/candidate APIs.
- [x] **Step 4 — Verify GREEN:** Repeat the focused suite and record exact
  selected/executed/skipped counts, then run related regressions and L0.
- [x] **Step 5 — Review:** Inspect source/test diff and update evidence. Make a
  focused Task 3A commit only with explicit authorization.

Task 3A completion wording: **Task 3A deterministic root-seed splitting and local
categorical route draw — COMPLETE; exact-index native context preparation, D=1/D=2
ownership, routed native inference and candidate provenance — deferred.** P06
remains PARTIAL.

### Task 3B: Exact native contexts and separately owned D1/D2 sessions

**Status:** ACTIVE — Task 3B.0 and Task 3B.1 exact-index/full-demo materializers
and Task 3B.2a injected-session validation plus Task 3B.2b authoritative
profile/outer construction COMPLETE under synthetic L1; Task 3B.3 NOT STARTED.
Task 3B constructs contexts only; it does not choose a route or call native
inference.

**Files:** Create `src/icgs/policies/reference.py`; extend
`src/icgs/composition.py` only for outer D1/D2 construction/loading. Do not add
context/session/policy imports to `router.py` and do not modify the behavior of
the existing `sample_to_cond_demo()` or `InstantPolicy.prepare_context()` paths.
**Test owner:** `tests/test_reference_policy.py`.

#### Task 3B.0 — Context-preparation RNG foundation and provenance

Context preparation has a separate lifecycle from action sampling:

```python
split_context_seeds(
    context_seed: int,
    *,
    demo_count: int,
    event_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]  # full-demo, window-slot seeds
```

```text
context_seed
    -> SeedSequence(context_seed)
    -> full-demo seeds in raw-demo order
    -> one reserved window seed for every EventMemory slot in slot order

action_seed
    -> split_route_seed(action_seed)
    -> route_seed + diffusion_seed
```

Protocol `seedsequence-native-choice-v1` spawns `D+L` children in that fixed
order. A seed is reserved for every event slot, including invalid or
unmaterializable slots, so changing which windows succeed cannot shift later
streams. Each returned integer is exactly
`int(child.generate_state(1)[0])`; children are not post-processed or forced
unequal. `context_seed` is a nonnegative integer; `demo_count` is positive and
`event_count` is nonnegative, with booleans rejected for all three.

Every native demo materialization uses the existing
`icgs.state.randomness.scoped_seed(point_seed, device="cpu")` around all of that
demo's sequential `subsample_pcd` calls. Do not implement another RNG scope.
The existing scope restores Python/NumPy/Torch global RNG state on success or
failure; construction remains serial because the native global NumPy path is not
concurrency safe. No context seed or child may be replaced with an action route/
diffusion seed.

The outer episode/run owner supplies `context_seed` explicitly and persists it;
P06 never invents a default. Its lifetime is one `MethodContext` materialization:
cache reuse retains the same preparation record, while rebuilding a context
requires an explicitly chosen seed and produces new context provenance.

```python
@dataclass(frozen=True)
class ContextPreparationRecord:
    context_seed: int
    full_demo_seeds: tuple[int, ...]
    window_slot_seeds: tuple[int, ...]
    rng_protocol: str
    full_context_id: str
    window_context_ids: tuple[str | None, ...]
```

The record's intrinsic consistency requires `full_demo_seeds` to be nonempty and
the two window tuples to have equal length. It intentionally carries neither D
nor L: Task 3B.3 validates the full-demo count against `raw_demos` and both
window-tuple lengths against the EventMemory slot count before assembly. All three
collection fields require exact tuple inputs; lists are rejected rather than
normalized. Every seed accepts a built-in `int` or NumPy integer, must be
nonnegative, and rejects built-in or NumPy booleans. `rng_protocol` is exactly
`seedsequence-native-choice-v1`; `full_context_id` is a nonempty string, and each
window context ID is either `None` or a nonempty string corresponding to the same
slot's materialization result. Identifier whitespace is used only to reject an
all-whitespace value and is otherwise preserved exactly. The record validates
and preserves values without normalization. It is provenance outside online
model state. Per-context seeds are not a second reference fingerprint; the
protocol ID is part of the existing P00 reference payload.

- [x] **Step 1 — RED:** Lock exact SeedSequence child values/order, reserved-slot
  stability, validation, global-RNG restoration and route/diffusion-seed
  separation.
- [x] **Step 2 — GREEN:** Implement only the context seed plan and immutable
  record validation; do not enter the scoped RNG boundary before materialization,
  and add no materialization, route draw or native prediction.
- [x] **Step 3 — Verify:** Run the focused owner and RNG regressions with exact
  counts and 0 hidden skips.

Task 3B.0 completion wording: **Task 3B.0 deterministic context child-seed
allocation and immutable preparation provenance — COMPLETE; Task 3B.1 native
materialization and Task 3B.2 D1/D2 session ownership are also COMPLETE under
synthetic L1, while Task 3B.3 MethodContext assembly remains NOT IMPLEMENTED.**
P06 remains PARTIAL.

#### Task 3B.1 — Exact-index native demonstration materialization

The additive materializer consumes measured raw-demo boundary observations and
the exact tuple produced by `select_window_indices`:

```python
materialize_indexed_native_demo(
    demo: TimedDemoInput,
    boundary_indices: tuple[int, ...],
    *,
    point_seed: int,
    native_waypoint_count: int,
    native_point_count: int,
) -> Mapping[str, tuple]  # obs, grips, T_w_es

materialize_full_native_demo(
    demo: TimedDemoInput,
    *,
    point_seed: int,
    native_waypoint_count: int,
    native_point_count: int,
) -> Mapping[str, tuple]  # obs, grips, T_w_es
```

`boundary_indices` is a nonempty tuple of sorted unique nonnegative integers,
with booleans rejected, and every index addresses the `N+1` measured observations
of the contiguous `TimedDemoInput`. `native_waypoint_count` and
`native_point_count` are positive integers with booleans rejected, and
`len(boundary_indices) == native_waypoint_count`. The caller passes
`sessions.d1.graph_config.traj_horizon` explicitly; the helper does not import or
inspect a session and never hard-codes 10.

`native_point_count` is resolved by the outer native-profile/artifact factory,
stored on `ReferenceSessions`, and must match the pinned reference payload's
`preprocessing.native_point_count`. It is distinct from
`MethodConfig.geometry.num_points` even when both currently equal 2048. There is
no helper-local or Python-signature fallback. Inputs are validated but never
normalized or mutated.

For each selected boundary, reuse the existing native outlier filter,
`subsample_pcd` point selection and world-to-end-effector transform on that
measured observation inside the Task 3B.0
`scoped_seed(point_seed, device="cpu")` boundary. Do not read `TimedCommand`,
call `extract_waypoints`, insert or repeat a waypoint, or reinterpret
`SegmentRef.a/b`. The returned native-shaped mapping is supplied to
`InstantPolicy.prepare_context(..., prepared=True)` so the exact event-window
indices are not selected a second time.

Full contexts deliberately retain the native waypoint-selection algorithm. The
one-demo `materialize_full_native_demo` helper converts raw measured observations
to the native raw-demo mapping and calls existing
`sample_to_cond_demo(raw_native_demo, native_waypoint_count,
num_points=native_point_count)` exactly once under its assigned point seed. It
never relies on `sample_to_cond_demo`'s default 2048. The helper validates demo
type, point seed and both native counts before calling the collaborator. After
the call it requires a mapping with exactly the logical `obs`, `grips` and
`T_w_es` fields, equal cardinalities and exactly `native_waypoint_count` entries;
malformed output raises and is never repaired, padded or truncated. It does not
deeply revalidate measured SE(3), grip or point values already owned by upstream
contracts. Thus full context uses native selection exactly once, while an event
window uses the Task 2 indices exactly once.

This helper owns one demo and one point seed only. Raw-demo order,
`len(full_demo_seeds) == len(raw_demos)` and assignment of child seeds to demos
remain cross-object checks owned by Task 3B.3. A function-scoped source/AST guard
keeps `materialize_full_native_demo` independent of `ReferenceSessions`,
`InstantPolicy`, `PreparedContext`, `MethodContext`, the exact-index materializer
and composition. It intentionally does not impose those restrictions on the
whole `reference.py` module, which later Task 3B.2–3C surfaces extend.

- [x] **Step 1a — Exact-index RED:** Add allowed exact-index fixtures plus
  duplicate, unordered, bool, out-of-range, wrong-count, command-access and
  mutation negatives. Include indices that native `extract_waypoints` would not
  choose and assert the exact poses/grips survive. Same point seed must reproduce
  identical prepared content; changing only point seed may change sampled points
  but never indices/poses/grips. Exercise global RNG restoration after successful
  materialization and an injected preprocessing failure.
- [x] **Step 2a — Verify exact-index RED:** Run only the new test owner. Existing
  router tests must remain green; each new failure must name the absent materializer
  surface.
- [x] **Step 3a — Exact-index GREEN:** Implement the additive bridge using Task
  3B.0's existing scoped RNG owner without changing native preprocessing defaults
  or public InstantPolicy behavior.
- [x] **Step 4a — Verify exact-index GREEN:** Run the focused owner and
  router/event/policy regressions, `py_compile`, diff/whitespace checks and both
  L0 variants.
- [x] **Step 1b — Full-context RED:** Lock explicit native counts, native
  waypoint selection exactly once, measured-state-only conversion, pre-call input
  validation, post-call result validation, one-demo seed behavior, immutability,
  RNG restoration and the function-scoped dependency boundary. Multi-demo order
  and seed-to-demo assignment remain Task 3B.3 ownership.
- [x] **Step 2b — Verify full-context RED:** Run the focused owner and require all
  exact-index tests to remain green while only the new full-context surface fails.
- [x] **Step 3b — Full-context GREEN:** Implement only the full-context raw-demo
  conversion and native selection path; do not start session/context assembly.
- [x] **Step 4b — Verify full-context GREEN:** Run the focused owner plus native
  preprocessing/policy regressions and both L0 variants before Task 3B.2.

#### Task 3B.2 — D1/D2 session ownership

`ReferenceSessions` receives two already constructed policies from the outer
composition/artifact boundary. It does not read checkpoints itself. D1 and D2
must have the same frozen IP checkpoint SHA256 and native profile except for
`graph.num_demos`; they are separately constructed, have distinct
`InstantPolicy.context_owner` objects and independent mutable graph/session
scratch, and are loaded once per constructed reference runtime rather than per
context, candidate or search branch. They execute sequentially, never
concurrently.

```python
@dataclass(frozen=True)
class ReferenceSessions:
    d1: InstantPolicy
    d2: InstantPolicy
    d1_config: ExperimentConfig
    d2_config: ExperimentConfig
    reference_id: str
    native_profile: str
    checkpoint_sha256: str
    native_point_count: int
    d1_session_id: str
    d2_session_id: str
```

`d1_config.graph.num_demos` is exactly 1. The canonical comparison is:

```python
expected_d2 = replace(
    d1_config,
    graph=replace(d1_config.graph, num_demos=2),
)
```

and `d2_config == expected_d2`; no allowlist loop silently omits a section.
Each policy must agree with its config through `graph_config`, `runtime`,
`sampler.config == config.sampling`, `sampler.diffusion == config.diffusion` and
`objective.config == config.diffusion`. Its sampler and objective must both use
that policy's exact `network.codec`. Both verified `policy.artifact_sha256` values
equal the record's canonical lowercase SHA-256.

Across D1/D2, policy, `context_owner`, network, `network.graph`,
`network.graph.graph`, `network.codec`, sampler, sampler `noise_scheduler` and
objective objects are distinct. Cross-policy parameter or registered-buffer
storage aliasing is rejected for the independently constructed default: any
positive-byte storage intervals on the same device must not overlap; adjacent
non-overlapping intervals and zero-byte storage are allowed. Within one policy,
sampler and objective deliberately share that policy's scheduler; this is native
behavior and is allowed. Frozen-weight sharing remains a future optimization
requiring an explicit adapter and separate equivalence tests.
`ReferenceSessions(...)` snapshots already constructed objects only for
validation and never mutates graph config or scratch, performs IO, prepares a
context or calls inference.

Session IDs are canonical lineage identifiers derived by composition, never
caller-chosen labels or Python object IDs. For role `"d1"` or `"d2"`:

```python
payload = {
    "domain": "icgs.reference-session",
    "schema": 1,
    "reference_id": reference_id,
    "role": role,
}
encoded = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
session_id = hashlib.sha256(encoded).hexdigest()
```

`build_reference_sessions()` does not accept session IDs; it derives them and
passes them to `ReferenceSessions`, whose constructor validates and preserves its
intrinsic scalar/config/session ownership without reconstructing a policy.

The authoritative versioned native artifact profile adds exact metadata:

```json
{
  "schema_version": 1,
  "profile_id": "instant_policy_published_vv19_119fa871",
  "artifact_sha256": "...",
  "preprocessing": {
    "native_point_count": 2048
  }
}
```

`resolve_native_profile(native_profile)` is the single artifact owner for this
metadata and for D1/D2 config derivation. `load_published_policy()` consumes that
same resolver instead of independently reconstructing another config. The first
real adapter supported by Task 3B.2b is published vv19. Primary MethodConfig
currently names `instant-policy-original-65dc94e`; it is not silently aliased to
published vv19. Supporting it requires its own versioned metadata owner. P00's
later canonical payload validation checks its `native_point_count` against the
resolved profile and `ReferenceSessions`; `MethodConfig.geometry.num_points` is
never used as a proxy.

The resolver's concrete return class is not contractual. Its required capability
surface exposes `profile_id`, `artifact_sha256`, `native_point_count` and
`config_for(*, num_demos, device)`. For the outer factory path,
`load_published_policy()` accepts the exact resolved `config=` plus the explicit
`native_profile`; existing published convenience arguments remain compatible.
Composition owns:

```python
build_reference_sessions(
    checkpoint,
    *,
    reference_id: str,
    native_profile: str,
    device: str,
) -> ReferenceSessions
```

It resolves once, requests D1 then D2 configs, loads two distinct policies in
that order, derives both canonical session IDs and constructs the validation
record only after both loads succeed. Reading or passing around that returned
record performs no further load. This does not require caching or idempotence when
`build_reference_sessions()` itself is called again.

Task 3B.2 owns sessions only. Every `PreparedContext` placement, full/window
policy-owner check, context alias check and `native_window_valid` consequence
remains Task 3B.3 ownership.

##### Task 3B.2a — Injected session validation

- [x] **Step 1a — RED:** Add fake-session tests for canonical D1/D2 config
  comparison, symmetric D1/D2 policy/config/checksum correspondence, intrinsic
  identifier/positive-integer count validation, per-policy sampler/objective
  scheduler sharing, exact cross-policy mutable-owner/scheduler/codec/tensor-
  storage non-aliasing and constructor non-mutation across ownership identities
  plus parameter/buffer contents. Add allowed and forbidden class/function-scoped
  dependency fixtures; do not scan all of `reference.py`.
- [x] **Step 2a — Verify RED:** Require every existing reference-policy test to
  remain green and every new test to fail only at the absent `ReferenceSessions`
  surface.
- [x] **Step 3a — GREEN:** Implement only the frozen validation record in
  `policies/reference.py`; do not add profile resolution, loading, context
  preparation or inference.
- [x] **Step 4a — Verify GREEN:** Run the focused owner, policy/config/architecture
  regressions, direct syntax/diff checks and both L0 variants.

##### Task 3B.2b — Authoritative profile and outer construction

- [x] **Step 1b — RED:** Lock profile JSON schema/ID and authoritative positive
  `preprocessing.native_point_count` without forbidding additional preprocessing
  metadata; unknown and original profile IDs must reach the resolver unchanged
  and be rejected before checkpoint/model IO. Lock shared config resolution,
  canonical session-ID bytes, exact profile metadata propagation, exactly two
  loads with D=1 then D=2, construction only after both loads succeed and no
  further loads when the returned session record is reused.
- [x] **Step 2b — Verify RED:** Keep 3B.2a green; new failures must identify only
  the absent resolver/factory or missing profile metadata.
- [x] **Step 3b — GREEN:** Implement the resolver in `artifacts/published.py`, make
  `load_published_policy()` consume it, and implement outer
  `build_reference_sessions()` in `composition.py`. `reference.py` performs no
  artifact IO.
- [x] **Step 4b — Verify GREEN:** Run focused, configuration, composition,
  checkpoint, loading and policy owners plus L0. Real shared-checksum D1/D2
  loading/inference and resident-memory evidence remain G4/L2.

#### Task 3B.3 — MethodContext assembly

```python
build_method_context(
    raw_demos: tuple[TimedDemoInput, ...],
    events: EventMemory,
    sessions: ReferenceSessions,
    *,
    context_seed: int,
    reference_id: str,
    config: MethodConfig,
) -> tuple[MethodContext, ContextPreparationRecord]
```

The builder validates online B=1 and exact raw-demo/hash order, materializes the
full context or fails setup, attempts only structurally eligible interaction
windows, and stores `None` for a valid-but-unmaterializable window. It delegates
final availability exclusively to `MethodContext.native_window_valid`; it does
not add another mask or treat structural eligibility as native feasibility.
Before context assembly it validates
`len(record.full_demo_seeds) == len(raw_demos)` and both
`len(record.window_slot_seeds)` and `len(record.window_context_ids)` against the
EventMemory slot count. These are builder-owned cross-object checks, not intrinsic
`ContextPreparationRecord` validation.

- [ ] **Step 1 — RED:** Cover one/two-demo full contexts, mixed available/absent
  windows, exact event-slot alignment, raw-hash order, owner mismatch, full-context
  failure and derived final validity.
- [ ] **Step 2 — GREEN:** Assemble existing `MethodContext` plus the external
  immutable preparation record; do not extend online task/context tensors.
- [ ] **Step 3 — Verify:** Focused and EventMemory/MethodContext regressions.

Task 3B completion wording: **Task 3B exact-index/full native context
materialization, separate context RNG provenance, owned D1/D2 session boundary
and MethodContext assembly — COMPLETE under synthetic L1; real native D1/D2
compatibility and memory cost — NOT RUN until G4/L2.** P06 remains PARTIAL.

### Task 3C: Exactly-one routed reference proposal and canonical lineage

**Status:** PLANNED — contract frozen for RED after Task 3B. Task 3C performs one
route draw and at most one native prediction. It is not candidate selection.

#### Task 3C.1 — Immutable routed proposal contract

Do not add method provenance fields to generic `Candidate`. Define:

```python
@dataclass(frozen=True)
class ReferenceProposal:
    candidate: Candidate
    reference_id: str
    route_index: int              # 0 full; 1..L event windows
    event_index: int | None       # None iff route_index == 0
    route_seed: int
    diffusion_seed: int
    native_context_id: str        # selected PreparedContext.source_id
    native_session_id: str        # stable D1/D2 session lineage, never id(...)
```

Validate route/event consistency, nonnegative integer seeds with booleans
rejected, exact context/session/reference IDs, `candidate.context_id ==
native_context_id`, and `candidate.seed == diffusion_seed`. P10's capability
boundary already treats proposals as opaque; the P06-owned `materialize_prefix`
adapter accepts `ReferenceProposal` and unwraps `.candidate`, so P10 algorithms
do not import this concrete policy type.

- [ ] **Step 1 — RED:** Add immutable ownership, malformed provenance, full/window
  indexing and opaque materialization fixtures; update the canonical planned
  method contract before runtime implementation.
- [ ] **Step 2 — GREEN:** Add the wrapper and P06 materializer adapter only.

#### Task 3C.2 — `sample_prior`: exactly one route, exactly one inference

`StageAwareReferencePolicy` is constructed with one validated
`ReferenceSessions`, one resolved `MethodConfig` and the same `reference_id`; it
does not load artifacts or construct sessions. Its public `sample_prior` and
`materialize_prefix` methods satisfy the existing opaque P10 capability seam.
It executes this fixed sequence:

```text
validate TaskState B=1 and exact EventMemory context fingerprints
validate MethodContext.reference_id and ReferenceSessions lineage
router_probabilities(task.alpha[..., :-1], task.eligible,
                     context.native_window_valid, config)
split_route_seed(action_seed) -> route_seed, diffusion_seed
draw_route(probabilities[0], route_seed=route_seed)
resolve exactly one (session, PreparedContext):
    route 0 -> full context and D1/D2 matching raw-demo count
    route k>0 -> native_windows[k-1] and D1
validate the selected context owner
propose_candidates(
    selected_session,
    observation,
    selected_context,
    count=1,
    seeds=[diffusion_seed],
)
Candidate -> immutable ReferenceProposal
```

Reuse the existing `propose_candidates(..., count=1,
seeds=[diffusion_seed])` path exactly. It already owns scoped RNG restoration,
timing, CUDA synchronization, context/artifact IDs and generic Candidate
metadata. Do not wrap `policy.predict()` in another RNG scope and do not duplicate
that proposal bookkeeping in `policies/reference.py`.

**No route retry. No alternate-window retry. No best-of-N native samples.** A
post-draw validation or inference failure fails that proposal attempt and retains
the selected route/seeds in failure evidence; it never turns the reference into
a search policy. A missing full context is a setup error. A window with no native
context has exact zero route probability and therefore cannot be selected.

- [ ] **Step 1 — RED:** Add deterministic full/window draws, lineage/owner/device
  negatives, exactly-one call counters, injected predict failure and explicit
  no-retry proofs. Verify route and diffusion streams independently.
- [ ] **Step 2 — GREEN:** Implement the B=1 reference policy against injected
  sessions and the existing single-candidate proposal seam.
- [ ] **Step 3 — Verify:** Run focused P06/P10 capability regressions; no real
  model, simulator, preprocessing job or training workload in L1.

#### Task 3C.3 — Existing reference payload and manifest integration

P06 produces metadata for the existing P00 `reference_fingerprint()` and
`validate_method_manifest()` owners; it never computes a parallel reference ID.
Extend the existing payload/schema with these exact keys; aliases or alternative
nesting reject:

```python
"preprocessing": {
    # all existing preprocessing fields remain
    "voxel_size_m": ...,
    "num_anchors": ...,
    "num_points": ...,             # existing method geometry lineage
    "neighbors": ...,
    "ell0_m": ...,
    "fps_start": ...,
    "tie_break": ...,
    "native_point_count": ...,     # pinned native-profile metadata
    "exact_window_protocol": "exact-boundary-v1",
},
"rng_protocol": {
    "route": "pcg64-v1",
    "context_points": "seedsequence-native-choice-v1",
    "native_predict": "scoped-seed-v1",
    "config": {
        "generator_seed": ...,
        "reset_seed": ...,
        "action_seed": ...,
    },
},
```

`native_predict` is deliberately not named `diffusion`: the seed passed to one
native `policy.predict()` also controls live point-cloud subsampling before the
diffusion noise draw. The outer native-profile/artifact owner resolves
`native_point_count` from versioned pinned profile metadata and injects the same
value into `ReferenceSessions`; it is never projected from
`MethodConfig.geometry.num_points` or accepted from an unverified caller. P00
validation requires the profile metadata, payload and session value to agree
before artifact/model use.

The reference payload continues to include IP checksum, native profile,
physical/event/task weights, router/config, calibration/workspace/camera/gravity
and cadence, and continues to exclude world model, evaluator, search and learned
stopping.

Actual context/action seeds and selected route/window IDs belong to immutable
per-context/proposal evidence. Protocol IDs and behavior-affecting configuration
belong to the canonical reference payload. Any change to exact-index, point
sampling, route-boundary or diffusion RNG behavior changes its protocol metadata
and therefore the existing `reference_id`; outcome labels must then be recollected.

- [ ] **Step 1 — RED:** Extend `test_method_contracts.py` with required nested
  protocol metadata, behavior-change fingerprint, excluded-field invariance and
  validate-before-artifact-read negatives. Reject missing/unknown protocol keys,
  the stale `rng_protocol.diffusion` spelling, and native point-count disagreement
  between pinned profile metadata, payload and ReferenceSessions.
- [ ] **Step 2 — GREEN:** Extend only `src/icgs/artifacts/method.py` canonical
  payload validation/projection; reuse `reference_fingerprint` and manifest schema.
- [ ] **Step 3 — Verify:** Run P00/P06/P10 plus artifact/config regressions and L0.

Task 3C completion wording: **Task 3C immutable routed ReferenceProposal,
exactly-one stage-aware native inference and canonical P00 reference-lineage
integration — COMPLETE under synthetic L1; G4/L2 native D1/D2 fidelity and
reference freeze remain required.** P06 remains PARTIAL until G4 passes and the
reference manifest is frozen.

### Parallel Stage B dependency path

Task 3B/3C runtime/reference semantics do not absorb training. In parallel, P02
must provide versioned task-view tensorization before P11 can wire real Stage B
batches to the already implemented P06 Task 1C objective. The current P11 gate
text saying `src/icgs/algorithms/objectives/task.py` is absent is stale after the
P06 merge; correct that diagnostic in a separate P11-scoped change, but do not
remove the gate until the batch adapter and data view exist.

Synthetic Task 3B/3C tests may use fabricated TaskState, EventMemory, contexts
and fake sessions without trained weights. Reference freeze requires both paths:
trained/frozen event/task/router artifacts and passing G4 D1/D2 native evidence.
Only after the canonical manifest is frozen may Stage C/P08 outcome collection
begin; P09/D2/E and real P10/P13 integration follow those labels.

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
- L1 pure tracker/router: `python3 -B -m unittest discover -s tests -p
  'test_task_router.py' -v` in an installed supported environment.
- L1 reference context/policy: `python3 -B -m unittest discover -s tests -p
  'test_reference_policy.py' -v`; this owner must exist and pass before claiming
  Task 3B or 3C complete.
- L1 canonical reference schema after Task 3C.3: `python3 -B -m unittest discover
  -s tests -p 'test_method_contracts.py' -v`.
  These use tiny deterministic fixtures only; no data loader workers, networking,
  real model load, preprocessing job or simulator startup.
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
- 2026-09-14 Task 3B/3C contract refinement: documentation only — no RED tests or
  runtime implementation were started. The plan now fixes exact-index/full-demo
  materialization, separate context/action RNG lifecycles, D1/D2 ownership,
  MethodContext assembly, immutable ReferenceProposal semantics, exactly-one
  inference and existing P00 fingerprint integration. `git diff --check` and the
  changed-file whitespace scan passed. Both L0 variants remain **FAIL** overall
  only for the same five pre-existing missing evidence-log links; Python syntax,
  static harness boundary and 19/19 harness self-tests passed with 0 skips.
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
- Task 3A RED: the focused command selected/executed 54 tests — **FAIL as
  expected**: all 48 existing Task 1/2 tests passed and all 6 new Task 3A tests
  raised only import errors for the absent `ROUTE_RNG_PROTOCOL`, `draw_route`
  and `split_route_seed` surface; 0 skips.
- Task 3A initial GREEN: the same focused command after implementation — **PASS**,
  54/54 selected/executed/passed with 0 failures/errors/skips. Coverage includes
  exact `SeedSequence.spawn(2)` child values, deterministic local PCG64 draws,
  exact-CDF-boundary and zero-mass behavior, malformed distributions and seeds,
  input immutability, global Python/NumPy/Torch RNG preservation on success and
  failure, and the no-native-context/session/policy dependency boundary.
- Task 3A pre-commit correctness review added an accepted-near-unit distribution
  fixture. Its targeted RED run selected/executed 7 Task 3A tests: 6 passed and
  the new test failed as intended because an undersummed distribution's closing
  residual selected a zero-mass final route (`actual=2`, `expected=1`). After
  canonicalizing the already-validated distribution on a float64 copy, the final
  focused command — **PASS**, 55/55 selected/executed/passed with 0
  failures/errors/skips. The same fixture also covers a slightly oversummed
  distribution so its CDF remains monotonic, and the seed API test now locks the
  second child as the diffusion seed rather than context-preparation RNG.
- 2026-09-15 Task 3B.0 RED: `.venv/bin/python -B -m unittest discover -s tests
  -p 'test_reference_policy.py' -v` selected/executed 9 top-level tests — **FAIL
  as expected**. All 9 failed only because `icgs.policies.reference` was absent;
  parameterized subtests expanded this to 29 missing-module error records, with
  0 skips. The pre-GREEN `test_task_router.py` regression remained **PASS**
  55/55 with 0 skips.
- Task 3B.0 initial GREEN selected/executed 9 tests: 8 passed and one test-harness
  assertion incorrectly rejected `None` in optional `window_context_ids`.
  Correcting that fixture to retain the specified optional-ID behavior required
  no runtime change. The final focused command and direct `py_compile` both
  **PASS**, 9/9 selected/executed/passed with 0 failures/errors/skips. Coverage
  locks exact ordered `SeedSequence` children, all-slot reservation, builtin and
  NumPy integer validation with boolean rejection, exact tuple inputs, frozen
  non-normalizing provenance, exact protocol identity, identifier validity,
  global Python/NumPy/Torch RNG preservation on success and malformed input, and
  the Task 3B.0 static dependency boundary.
- Task 3B.0 related regressions: `test_task_router.py` **PASS** 55/55; the
  canonical episode/world-model/event-memory/physical-memory/config/contracts/
  architecture owners **PASS** 132/132; and canonical-discovery `test_policy.py`
  **PASS** 6/6. These related owners total 138/138 with 0 skips. The separate
  RNG owner `test_inference_contract.py` executed 3 tests: 2 passed and the CUDA
  RNG test was **SKIPPED** because CUDA is unavailable; the skip is not counted
  as a pass.
- Pre-3B.1 review scoped the Task 3B.0 dependency guard to the source/AST of
  `split_context_seeds` and `ContextPreparationRecord`. Both real surfaces pass,
  while an isolated fixture referencing `scoped_seed` fails with the intended
  ownership diagnostic; the focused suite remains **PASS** 9/9 with 0 skips.
  This direct static check does not claim complete detection of dynamic or
  alias-obscured dependencies. The same review clarified that the record enforces
  only intrinsic tuple consistency; Task 3B.3 owns D/L cross-object cardinality.
- Task 3B.1 initial RED selected/executed 15 tests: all 9 Task 3B.0 tests passed
  and all 6 new tests failed only at the absent materializer import. Review found
  that the alternating-grip fixture's `(1,2)` indices were also selected by the
  native extractor and that its command guard covered fields rather than
  `transition.command` itself, so that run is retained but not accepted as final
  RED proof.
- Task 3B.1 refined RED: the focused reference-policy command selected/executed
  16 tests — **FAIL as expected**. All 9 Task 3B.0 tests passed; all 7 exact-index
  materialization tests errored only because
  `materialize_indexed_native_demo` is absent, with 0 skips. A constant-grip
  fixture directly proves native extraction returns `(0,4)` while the exact path
  requires `(1,3)`, and a guarded `ExecutedTransition` rejects any access to
  `transition.command`. Remaining fixtures lock measured pose/grip preservation,
  validation before preprocessing, explicit waypoint/point counts, exact
  sequential native point draws under one seed, input immutability, and Python/
  NumPy/Torch RNG restoration on success and injected preprocessing failure. The
  existing `test_task_router.py` owner remained **PASS** 55/55 with 0 skips. No
  Task 3B.1 runtime surface was added. Both L0 variants remain **FAIL** overall
  only for the same five pre-existing missing evidence-log links; Python syntax,
  static harness boundary and 19/19 harness self-tests passed with 0 skips.
- Task 3B.1a GREEN: the focused reference-policy command selected/executed/passed
  16/16 tests with 0 failures/errors/skips: the accepted 9 Task 3B.0 tests plus 7
  exact-index materializer tests. The implementation validates all inputs before
  preprocessing, reads only measured boundary observations, preserves exact
  selected poses/grips, delegates point selection to native `subsample_pcd`
  inside the existing scoped RNG owner, and never calls `extract_waypoints` or
  accesses `transition.command`. The first architecture regression exposed that
  the ADR 0002 Open3D allowlist named only the older concrete InstantPolicy. The
  guard was updated with an exact `icgs.policies.reference` entry, not a package
  exemption; its negative fixture still treats an unapproved policy as pure.
  `test_architecture.py` then passed 3/3.
- Task 3B.1a related regressions: `test_task_router.py` **PASS** 55/55,
  `test_event_memory.py` **PASS** 29/29 and `test_policy.py` **PASS** 6/6, all
  with 0 skips. `test_inference_contract.py` executed 3 tests: 2 passed and its
  CUDA RNG test was **SKIPPED** because CUDA is unavailable; that skip is not
  counted as a pass. Direct `py_compile` of the changed runtime/test files and
  `git diff --check` both **PASS**.
- Task 3B.1b RED: the focused reference-policy command selected/executed 24 tests
  — **FAIL as expected**. All 16 prior Task 3B.0/3B.1a tests passed and all 8
  full-demo tests errored only because `materialize_full_native_demo` was absent,
  with 0 skips. The RED surface separates pre-collaborator scalar validation from
  post-collaborator mapping/cardinality validation, uses a real native-selection
  fixture separately from a consumer-namespace spy, and includes allowed and
  forbidden function-scoped dependency fixtures.
- Task 3B.1b GREEN: the focused reference-policy command **PASS** 24/24 with 0
  failures/errors/skips. The one-demo helper reads only measured observations,
  calls native `sample_to_cond_demo` exactly once with explicit waypoint and point
  counts inside one scoped RNG boundary, validates the returned three-field
  mapping without repairing it, and emits tuple-valued native content. Related
  owners passed `test_data.py` 7/7 including the actual Open3D outlier/voxel path,
  `test_policy.py` 6/6, `test_architecture.py` 3/3, `test_event_memory.py` 29/29
  and `test_task_router.py` 55/55: 100/100 with 0 skips. The separate
  `test_inference_contract.py` selected 3 tests: 2 passed and the CUDA RNG test
  was **SKIPPED** because CUDA is unavailable; it is not counted as a pass.
  Direct `py_compile` and `git diff --check` both **PASS**.
- Task 3B.2a RED: the focused reference-policy command selected/executed 30
  tests — **FAIL as expected**. All 24 accepted Task 3B.0/3B.1 tests passed and
  all 6 injected-session validation tests errored only because
  `ReferenceSessions` is absent, with 0 skips. The RED surface locks the frozen
  field layout, exact D1/D2 config derivation, symmetric policy/config/checkpoint
  correspondence, canonical lineage/session IDs, full positive-integer native
  point-count negatives, per-policy scheduler sharing, constructor non-mutation,
  and cross-policy mutable-owner plus parameter/buffer-storage non-aliasing. Its
  class-scoped positive and isolated negative source/AST fixtures define the
  validation-only dependency check but cannot execute until the import exists;
  GREEN must exercise both before the guard is claimed as passing, and even then
  it will not detect dynamic or alias-obscured dependencies. Existing
  `test_policy.py` and `test_architecture.py` regressions remained **PASS** 6/6
  and 3/3 respectively, with 0 skips. Direct `py_compile` and `git diff --check`
  both **PASS**. No Task 3B.2 runtime surface was added.
- Task 3B.2a initial GREEN selected/executed 30 tests: 28 passed and two isolated
  cross-policy sampler/objective alias subtests failed because those fixtures also
  broke the earlier within-policy scheduler-sharing invariant. Refining each
  negative fixture to preserve its authorized internal sharing required no
  runtime change. The final focused command **PASS** 30/30 with 0 failures,
  errors or skips. `ReferenceSessions` is frozen and validation-only; it preserves
  injected objects, validates exact D1/D2 config and policy correspondence,
  canonical checkpoint/session lineage, positive native point count, internal
  scheduler ownership, cross-policy mutable owners and parameter/buffer storage.
  The class-scoped dependency guard's allowed surface and isolated forbidden
  `load_published_policy` fixture both executed with the intended result; it does
  not claim detection of dynamic, transitive or alias-obscured dependencies.
- Task 3B.2a pre-commit hardening RED kept 30 test methods and produced five
  intended assertion failures: four symmetric D1/D2 sampler/objective codec
  correspondence subcases and one offset, positive-byte storage-overlap case.
  An adjacent non-overlapping pair over the same NumPy backing remained accepted.
  GREEN now requires each collaborator to use its policy network's exact codec
  and compares cross-policy storage address intervals on the same device; the
  final focused command returned to **PASS** 30/30 with 0 failures/errors/skips.
- Task 3B.2a related regressions: `test_policy.py` **PASS** 6/6,
  `test_architecture.py` **PASS** 3/3, `test_config.py` **PASS** 5/5 and
  `test_v5_config.py` **PASS** 4/4, totalling 18/18 with 0 skips. Direct
  `py_compile` and `git diff --check` both **PASS**. Profile resolution, artifact
  loading, outer session construction, context preparation and inference remain
  absent from this Task 3B.2a runtime change.
- Task 3B.2b RED: the focused reference-policy command selected/executed 37
  tests — **FAIL as expected**. All 30 accepted Task 3B.0–3B.2a tests passed;
  the seven new methods produced two intended assertion failures for missing
  profile metadata and loader delegation plus six expected error records across
  five methods for the absent loader API, resolver and factory surfaces, with 0
  skips. The loader-level unknown/original-profile subcases both fail at the
  absent `native_profile` argument before checkpoint/model IO. Fixtures lock
  exact published profile identity and point count while permitting additional
  preprocessing metadata, unchanged unknown/original profile IDs at the resolver,
  exact D1/D2 resolved configs, function-scoped loader/factory boundaries,
  canonical session IDs, exact profile metadata propagation, two distinct
  D1-then-D2 loads with exact config objects, construction only after both loads,
  policy/config non-mutation and no load triggered by reading the returned record.
  Factory collaborator and constructor spies patch both authoritative modules and
  any existing consumer aliases, so the RED contract does not prescribe import
  syntax. The tests do not require a second builder call to be cached. Static and
  runtime negative fixtures that need the missing imports are defined but cannot
  all execute until GREEN; GREEN must exercise them before claiming the guards
  pass.
- Task 3B.2b RED related regressions: `test_v5_config.py` **PASS** 4/4,
  `test_composition.py` **PASS** 6/6, `test_loading.py` **PASS** 2/2 and
  `test_policy.py` **PASS** 6/6, totalling 18/18 with 0 skips. Direct
  `py_compile` and `git diff --check` both **PASS**. No profile, resolver, loader
  or composition runtime was changed in this RED phase. An initial combined
  module-path unittest invocation was not a valid repository test entry point:
  the latter two modules could not resolve their discovery-style sibling import.
  It was replaced by the four canonical per-file discovery commands above; this
  invocation error is not counted as a regression result.
- Task 3B.2b GREEN: `test_reference_policy.py` **PASS** 37/37 with 0
  failures/errors/skips. The packaged vv19 profile now owns its exact ID and
  positive native point count; the resolver rejects unknown/original IDs without
  aliases, and the published loader resolves before checkpoint/model IO while
  passing the exact injected config object to `build_policy()` without
  loader-side reconstruction. The outer factory resolves once, derives D1 then
  D2 configs, performs exactly two distinct loads, derives canonical session IDs
  and constructs `ReferenceSessions` only after both loads succeed.
  Function-scoped allowed/forbidden guards and both loader pre-IO rejection
  subcases executed successfully. These synthetic collaborators prove ordering,
  lineage and ownership contracts, not real model loading or fidelity.
- Task 3B.2b GREEN related regressions: `test_v5_config.py` **PASS** 4/4,
  `test_composition.py` **PASS** 6/6, `test_checkpoints.py` **PASS** 13/13,
  `test_loading.py` **PASS** 2/2, `test_policy.py` **PASS** 6/6,
  `test_cli.py` **PASS** 7/7 and `test_architecture.py` **PASS** 3/3, totalling
  41/41 with 0 skips. Direct `py_compile` and `git diff --check` both **PASS**.
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
- Task 3A related regressions reran the same canonical owners: the combined
  episode/world-model/event-memory/physical-memory/config/contracts/architecture
  invocation passed 131/131, and canonical-discovery `test_policy.py` passed
  6/6; 137/137 related assertions passed with 0 skips.
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
- Task 3A `.venv/bin/python -B -m py_compile
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
- Task 3A L0 rerun: both commands remain **FAIL** overall with exactly the same
  five pre-existing missing evidence-log targets. Both passed Python syntax,
  static harness boundary and 19/19 harness self-tests with 0 skips; no new L0
  failure was introduced relative to Task 2 commit `1375865`.
- Task 3B.0 L0 rerun: both commands remain **FAIL** overall with exactly the same
  five pre-existing missing evidence-log targets. Both passed Python syntax,
  static harness boundary and 19/19 harness self-tests with 0 skips; no new L0
  failure was introduced relative to Task 3A commit `9cca46e`.
- Task 3B.1a L0 rerun: both commands remain **FAIL** overall only for those same
  five pre-existing missing evidence-log targets. Both pass Python syntax, the
  static harness boundary and 19/19 harness self-tests with 0 skips; the initial
  Open3D-ownership regression was resolved before this final evidence run.
- Task 3B.1b L0 rerun: both commands remain **FAIL** overall only for the same
  five pre-existing missing evidence-log targets. Both pass Python syntax, the
  static harness boundary and 19/19 harness self-tests with 0 skips; no new L0
  regression was introduced.
- Task 3B.2a final L0 rerun: both commands remain **FAIL** overall only for those
  same five pre-existing missing evidence-log targets. Both pass Python syntax,
  the static harness boundary and 19/19 harness self-tests with 0 skips; no new
  L0 regression was introduced by the Task 3B.2a runtime/test/plan changes.
- Task 3B.2b RED L0 rerun: both commands remain **FAIL** overall only for the
  same five pre-existing missing evidence-log targets. Both pass Python syntax,
  the static harness boundary and 19/19 harness self-tests with 0 skips; no new
  L0 regression was introduced by the Task 3B.2b test/plan-only changes.
- Task 3B.2b GREEN L0 rerun: both commands remain **FAIL** overall only for the
  same five pre-existing missing evidence-log targets. Both pass Python syntax,
  the static harness boundary and 19/19 harness self-tests with 0 skips; no new
  L0 regression was introduced by the Task 3B.2b runtime/profile/plan changes.
- L2/C1–C5: **NOT RUN** by this plan — preserve mandatory integration acceptance.
- L3/L4, simulator, preprocessing, collection and training: **NOT RUN** — separate
  resource authorization required.
- Remaining risk: Tasks 1A/1B/1C/2/3A/3B.0/3B.1 have primarily synthetic CPU
  evidence;
  Task 2 additionally has one CPU-FP16 subnormal normalization regression and
  Task 3A exercises float32/float64 categorical inputs. In particular, the
  dtype-derived TaskState alpha, Task 1C alignment-target and Task 3A route-sum
  tolerances have not been validated broadly for float16/bfloat16. Strict
  validation also uses host-reading `.item()` checks that may synchronize each
  GPU batch; before real Stage B training, decide from measurement whether those
  checks remain on the hot path or move partly to dataset/debug validation. P02
  task-view tensorization, Stage B training, GPU/mixed-precision behavior,
  context replay orchestration, final-availability integration, routed policy
  execution, real shared-checksum native sessions and measured reference support
  remain unverified.
  Task 3B.1 materializers exercise native point selection and RNG ownership with
  synthetic outlier-filter fixtures; the native Open3D primitives pass their
  independent L1 owner, but full materializer-to-PreparedContext integration
  remains NOT RUN.
