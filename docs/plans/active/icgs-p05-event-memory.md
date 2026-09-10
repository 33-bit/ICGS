# P05: Observable event segmentation and immutable demonstration memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Turn one or two independent successful demos into bounded event tokens with raw-window references.

**Architecture:** A deterministic data algorithm segments observable motion/grip; a separate neural event encoder consumes physical features. State owns immutable demo content and token/cache lineage, without task memory.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Neural method](../../method/neural.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — P05 Tasks 1–2 and Task 3A COMPLETE; P05 remains PARTIAL because
raw-demo tensor orchestration, native-window construction/session validation,
routing, and measured feasibility gates are NOT IMPLEMENTED**. This document is a category C
component of the approved research migration; it does not authorize simulator,
collection, or training workloads.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00/P02/P03/P04; segmentation uses raw timed demos, not resampled native10-waypoint contexts.
Read [workflow](../../WORKFLOW.md), [architecture](../../ARCHITECTURE.md),
[baseline](../../baselines/instant_policy.md), [native contract](../../components/policy-data-contract.md),
[research policy](../../RESEARCH.md) and [validation guide](../../../tests/README.md)
before runtime execution. Inspect Git state; preserve unrelated work and user assets.

## Configuration consumption addendum — 2026-09-09

Default values now belong to the [packaged primary JSON](../../../src/icgs/configuration/profiles/icgs_primary.json),
with key/unit/restriction details in the [parameter reference](../../method/parameters.md).
Consumed sections for this plan: **event, geometry, neural, router**.

Segmentation uses event debounce/motion/time/merge/cap fields; event MLPs and blocks consume event/neural settings. Native action windows consume router neighbor counts and native waypoint inventory. Keep literal numeric fixtures as independent expected results, not default sources.

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

- Create: `src/icgs/data/preprocessing/events.py` — observable segmentation.
- Create: `src/icgs/models/encoders/event.py` — frame/event encoders.
- Create: `src/icgs/state/method_context.py` — EventMemory and MethodContext.
- Test: `tests/test_event_memory.py`.

Public capability boundary (partially implemented):

```python
segment_demo(raw_demo, segmentation_config) -> tuple[SegmentRef, ...]
encode_events(raw_demos, segments) -> EventMemory
MethodContext(raw_demos, events, native_full, native_windows, reference_id)
```

`segment_demo`, pure `build_event_memory`, `EventMemory`, and `MethodContext` are
implemented. P13 still owns the public `encode_events` orchestration facade; it is
not implemented by this plan.

SegmentRef stores content hash,a,b,kind and structural action-window eligibility.
At most 30 interaction events plus start/end per demo; overflow is explicit. No demo-ID
embedding or global concatenation-index embedding; context hash binds raw data,
preprocessing, segmentation and weights.

### Task 1: Deterministic motion/grip boundaries and protected merges

**Files:** Create `src/icgs/data/preprocessing/events.py`.
**Test owner:** `tests/test_event_memory.py`.
**Consumes / produces:** Produces `debounced_grip_boundaries(grips, event_config)` and `segment_demo`.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.configuration.method import MethodConfig
from icgs.data.preprocessing.events import debounced_grip_boundaries
cfg = MethodConfig().event
self.assertEqual(debounced_grip_boundaries([0, 1, 0, 1, 1], cfg), (4,))
self.assertEqual(debounced_grip_boundaries([1, 1, 0, 0], cfg), (3,))
```

- [x] **Step 2 — Verify RED:** The supported `.venv` run selected 9 tests and
  produced 9 expected missing-module errors before `events.py` existed. No
  unrelated import or dependency failure was counted as RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
stable = grips[0]
boundaries = []
for t in range(1, len(grips)):
    if grips[t] != stable and grips[t] == grips[t-1]:
        stable = grips[t]
        boundaries.append(t)
return tuple(boundaries)
```

Mark at confirmation t, never backdate. Sum consecutive translation/geodesic angles;
split at >0.12m,>30deg or >=20 intervals. Retain endpoints. Process short
segments shortest-first; merge with the adjacent segment having shorter duration,
tie to the earlier segment, and never erase a grip boundary. For capacity, merge
the adjacent legal pair having minimum combined duration, tie to the earlier pair;
emit overflow rather than truncate when protected boundaries cannot fit.

Implemented `TimedDemoInput` as a frozen validation-only boundary over a nonempty
tuple of adjacent, contiguous `ExecutedTransition` records and an unchanged
injected demo-content hash. Segmentation extracts N+1 measured observations from
before/after records without reading `TimedCommand`; `SegmentRef.a/b` are local
indices into that sequence and every reference preserves the injected hash.

- [x] **Step 4 — Verify GREEN:** The supported `.venv` suite passed: 9 selected,
  9 executed/passed, 0 failures/errors/skips. A review refinement first proved
  that a missing explicit config was accepted (**FAIL**, 1 executed, 1 failure),
  then rejected it and restored the 9/9 GREEN result. A follow-up separately
  proved the debounce helper still accepted implicit config (**FAIL**, 1 executed,
  1 failure), removed that fallback, and again restored the 9/9 GREEN result.
- [x] **Step 5 — Review:** Inspected the exact source/test diff, command-access
  audit, trailing whitespace, and `git diff --check`. No command fields, native IP,
  P01 execution/replay, simulator, collection, or training path was changed. No
  commit was made.

Task 1 status: **COMPLETE** — deterministic measured-state segmentation and
synthetic contract evidence are implemented. Overall P05 remains **PARTIAL**:
raw-demo tensor orchestration, native-window construction/session validation,
routing, and measured suitability/overflow gates remain deferred.

### Task 2: Encode event geometry and per-demo order

**Files:** Create `src/icgs/models/encoders/event.py`; modify
`src/icgs/models/layers/method.py` with a generic masked self-attention block.
**Test owner:** `tests/test_event_memory.py`.
**Consumes / produces:** Produces strict
`event_features(d_start,d_end,d_mean,xi,g_start,g_end)`, transient
`EventEncoding(tokens, valid)`, and a tensor-only `EventEncoder`. Public
`encode_events(raw_demos, segments) -> EventMemory`, `SegmentRef` tensorization,
token/reference correspondence, and owned context composition remain Task 3.

**Accepted boundary record:** The P05 neural specification and owner-approved
Task 2 contract authorize only synthetic tensor inputs to `EventEncoder` and
`EventEncoding` output. `START_KIND=0`, `INTERACTION_KIND=1`, and `END_KIND=2`
are the tensor kind vocabulary; no demo ID/global concatenation position is an
embedding. `SegmentRef`, content hashes, `EventMemory`, `MethodContext`, routing,
simulator and training behavior are forbidden from this task. Valid entries must
be finite. Masked anchor/event padding may contain sentinel or nonfinite values,
but must be sanitized before pooling, gather, division, feature construction or
attention. Valid landmarks must have `a == b` and input twist L2 norm at most
`LANDMARK_TWIST_ATOL=1e-6`; the encoder validates and never rewrites twist. This
constant is a fixed numerical validation guard with `rtol=0`, not a configurable
segmentation threshold or LayerNorm epsilon. Violations reject the named tensor
or invariant so the Task 3 tensorization caller can correct its input.

- [x] **Step 1 — RED:** Add focused assertions to the test owner for the complete
  tensor boundary, including the primary-profile compatibility assertion below.

```python
import torch
from icgs.models.encoders.event import event_features
v = torch.zeros(1, 256)
f = event_features(v, v, v, torch.zeros(1, 6), torch.zeros(1, 1), torch.ones(1, 1))
self.assertEqual(tuple(f.shape), (1, 776))
```

- [x] **Step 2 — Verify RED:** `.venv/bin/python -B -m unittest discover -s
  tests -p 'test_event_memory.py' -v` selected/executed 18 tests with 0 skips:
  the 9 retained Task 1 tests passed, while all 9 new Task 2 tests produced the
  expected errors for absent `icgs.models.encoders.event` or absent
  `MaskedSelfAttentionBlock`. No dependency, collection, or syntax failure was
  counted as RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
@dataclass(frozen=True)
class EventEncoding:
    tokens: Tensor  # [B,L,event.width]
    valid: Tensor   # bool [B,L]


class EventEncoder(nn.Module):
    def forward(
        self, frame_tokens, anchor_valid, proprio, segment_start, segment_end,
        twist, grip_start, grip_end, event_kind, event_valid,
        local_index, local_count,
    ) -> EventEncoding:
        ...
```

The processing order is fixed: validate dtype/outer shape; validate finite values
only on valid entries; validate valid-landmark constraints; validate valid-row
indices/counts; sanitize every masked row; then gather/inclusive segment mean,
compute local order, call strict `event_features`, apply MLP/embeddings and generic
masked attention, zero invalid output rows, and construct `EventEncoding`.

Require `segment_start/segment_end` to be `torch.long` and
`anchor_valid/event_valid` to be `torch.bool`. Only valid event rows must satisfy
`0 <= start <= end < F`, `local_count > 0`, and
`0 <= local_index < local_count`; sanitize padding indices/counts before gather or
division. `event_features` has no mask and rejects every NaN/Inf. The encoder mask
permits nonfinite padding only by checking valid entries first and replacing all
masked padding before calling that strict helper. Landmark twist is validated,
not replaced.

Derive frame input width as `geometry.width + 13` and event input width as
`3 * event.width + 8`; 269 and 776 are primary-profile acceptance values rather
than implementation literals. Frame MLP is
`(geometry.width+13)->event.width->event.width`; event MLP is
`(3*event.width+8)->event.token_hidden_dim->event.width`; order MLP is
`2->event.width->event.width`. `local_index/local_count` supplies
`[j/J,1/J]`, where j starts at zero and J includes landmarks.

Implemented `MaskedSelfAttentionBlock` using the existing pre-LN/head/FFN/dropout/mask
conventions without `GeometryBlock`'s relative-3D bias. Instantiate exactly
`event.transformer_layers` blocks. `EventEncoder` requires either one explicit
resolved `MethodConfig`, or all three explicit `GeometryConfig`, `EventConfig`,
and `NeuralConfig` sections; it never constructs defaults internally.

RED coverage must distinguish allowed poisoned padding from forbidden valid
nonfinite values, verify padding is sanitized before any intermediate operation,
lock inclusive `[a,b]` frame means and unchanged accepted landmark twist, assert
kind/dtype/index/count validation, and prove demo-block permutation equivariance
plus valid-token pooled invariance within numerical tolerance. A focused source
boundary assertion forbids Task 3 ownership types from `event.py`; it is not a
complete dynamic dependency proof.

- [x] **Step 4 — Verify GREEN:** `.venv/bin/python -B -m unittest discover -s
  tests -p 'test_event_memory.py' -v` passed all 18 selected/executed tests with
  0 failures/errors/skips. The retained Task 1 tests remained unchanged and green.
- [x] **Step 5 — Review:** Inspected the exact source/test diff, explicit config
  propagation, gradient finiteness, forbidden owner names, padding sanitization,
  and `git diff --check`. No native IP, Task 3, simulator, collection, routing, or
  training owner was changed. Committed as `37cca4c feat(events): add masked event
  encoder`.

Task 2 status: **Task 2 tensor encoder — COMPLETE.** Task 3A now owns pure
`EventEncoding` to `EventMemory` composition; public `encode_events` orchestration
remains deferred to P13. P05 remains PARTIAL until the remaining integration owners
and required measured feasibility gates are complete.

### Task 3A: State ownership and lineage remain distinct

**Files:** Create `src/icgs/state/method_context.py`.
**Test owner:** `tests/test_event_memory.py`.
**Consumes / produces:** Produces
`context_fingerprint(raw_hashes, encoder_id, segmentation_id)`, pure
`build_event_memory(encoding, segment_refs, raw_hashes, encoder_id,
segmentation_id)`, batched `EventMemory`, and online-B=1 `MethodContext`
ownership. P13 owns raw-demo tensorization/orchestration behind the public
`encode_events` facade. P06 owns native-window construction, routing and native
session compatibility; this task only stores already prepared contexts.

**Accepted boundary record:** EventMemory aligns refs as an exact nested `[B,L]`
tuple and ordered raw hashes as `[B,D]`. It clones tokens without detaching their
autograd graph, clones validity, validates finite tokens and exact-zero invalid
rows, and computes one domain-separated fingerprint per batch row. Fingerprints
are derived, never caller input. Within each row, raw hashes are unique and every
declared hash owns one nonempty contiguous valid-ref block; those blocks occur in
exactly the declared raw-hash order. MethodContext requires EventMemory B=1, an exact
ordered raw-demo hash match, one mandatory injected full `PreparedContext`, and an
injected `PreparedContext | None` tuple of length L. It stores these objects
unchanged, rejects object or mutable-buffer aliases, and derives
`native_window_valid`; it never branches, copies, prepares, mutates or infers
D1/D2 owner compatibility. A structurally eligible interaction with no materialized
window is valid state with derived false. No TaskState, router, artifact IO,
calibration, simulator or model orchestration enters this module.

- [x] **Step 1 — RED:** Add focused fingerprint, batched alignment/autograd,
  ownership, alias and derived-window assertions to the test owner, including the
  following basic identity check.

```python
from icgs.state.method_context import context_fingerprint
a = context_fingerprint(('demo-a',), 'encoder-a', 'segments-a')
b = context_fingerprint(('demo-a',), 'encoder-a', 'segments-b')
self.assertNotEqual(a, b)
self.assertEqual(a, context_fingerprint(('demo-a',), 'encoder-a', 'segments-a'))
```

- [x] **Step 2 — Verify RED:** `.venv/bin/python -B -m unittest discover -s tests
  -p 'test_event_memory.py' -v` selected/executed 25 tests with 0 skips: all 18
  retained Task 1–2 tests passed and all 7 new Task 3A tests produced the expected
  missing-`icgs.state.method_context` errors. Test collection and installed
  dependencies succeeded; no unrelated failure was counted as RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
payload = dict(raw_hashes=tuple(raw_hashes), encoder_id=encoder_id,
               segmentation_id=segmentation_id)
return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
```

Use canonical JSON with domain `icgs.event-memory`, schema 1, ordered raw hashes,
opaque injected encoder lineage and segmentation lineage, then SHA-256. Build
EventMemory from `EventEncoding` without detaching. Require `valid=True` exactly
when the aligned ref is non-None, every ref hash to belong to that row's raw hashes,
every declared raw hash to be represented exactly once at the block level, and
those unique demo blocks to follow declared order. Invalid token rows must be
finite exact zero rather than being sanitized.

MethodContext stores existing immutable `TimedDemoInput` records and exact injected
PreparedContext objects. Require `len(native_windows) == L`; any non-None window
must align with a valid structurally eligible interaction. None is permitted for
such an interaction and derives false. Reject aliases among full/windows and their
mutable embeddings/positions; do not enforce owners equal because P06/P13 own D1/D2
session validation. Full context is mandatory. Context has no TaskState; context
swap replay remains P06.

- [x] **Step 4 — Verify GREEN:** The supported `.venv` command passed all 25
  selected/executed tests with 0 failures/errors/skips. All 18 retained Task 1–2
  tests remained green.
- [x] **Step 5 — Review:** Inspected the exact source/test/documentation diff,
  ownership and source-boundary guards, `git diff --check`, focused compilation,
  related regressions, and both L0 variants. No P06/P13 owner, native IP, simulator,
  collection, preprocessing, or training path was changed. No commit was made.

Task 3A completion wording: **Task 3A state/lineage ownership — COMPLETE; raw-demo
tensor orchestration, native-window construction/session validation, routing and
MethodRuntime composition — deferred.** P05 remains PARTIAL until those owners and
the required measured feasibility gates are complete.

## Acceptance, resource limits and evidence

L1 fixtures cover debounce, strict thresholds, endpoints, one-frame rejection, protected overflow, deterministic ties, permutation and ownership. Segmentation suitability/overflow frequency and identifiable labels require a declared development pilot; threshold changes get protocol IDs.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_event_memory.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

Do not feed new64-token event memory into native IP or compress raw demos before segmentation. Stress cap64 is separate; no oracle segmentation and no truncation of suffix requirements.

On failure, retain the failed evidence and isolate the new component/config ID.
Do not overwrite native checkpoints, relabel collected outcomes or drop difficult samples.
Resume only from a compatible explicit artifact/manifest; changed reference lineage
requires new outcome labels, not a cache workaround. Return to the preceding accepted
phase if an FG fails and request a scoped protocol decision.

## Execution evidence

- Environment: repository `.venv/bin/python`, CPython 3.10.20, editable
  no-dependency install of the current checkout; cwd repository root.
- Initial RED: `.venv/bin/python -B -m unittest discover -s tests -p
  'test_event_memory.py' -v` — **FAIL**, 9 selected/executed, 9 expected
  `ModuleNotFoundError` errors for the absent planned module, 0 skips.
- GREEN: the same command — **PASS**, 9 selected, 9 executed/passed, 0
  failures/errors/skips. Fixtures cover validation-only ownership, command
  independence, exact injected hashes, debounce, cumulative strict motion/time
  thresholds, canonical short/cap merging, protected boundaries, and overflow.
- Task 2 RED: the same command after adding the final tensor contract tests —
  **FAIL as expected**, 18 selected/executed, 9 existing Task 1 tests passed and
  9 new Task 2 tests errored, 0 skips. Eight errors were the missing planned
  `icgs.models.encoders.event` module and one was the missing generic
  `MaskedSelfAttentionBlock`; the installed dependencies and test collection
  succeeded. GREEN implementation was not started in this phase.
- Task 2 GREEN: the same command — **PASS**, 18 selected/executed/passed, 0
  failures/errors/skips. The tensor tests cover strict feature finiteness,
  allowed poisoned padding, exact mask zeroing, valid-row dtype/index/order and
  landmark guards, inclusive segment pooling, explicit configuration, generic
  non-geometric attention, finite gradients, and demo-block permutation behavior.
- Task 3A RED: the same command after adding the final ownership tests — **FAIL as
  expected**, 25 selected/executed, 18 retained Task 1–2 tests passed and all 7 new
  tests errored only because `icgs.state.method_context` did not yet exist; 0 skips.
- Task 3A GREEN: the same command — **PASS**, 25 selected/executed/passed, 0
  failures/errors/skips. One preceding GREEN attempt passed 24 tests and failed one
  assertion because the diagnostic capitalization did not match its regex; the
  runtime rejection itself was correct, the diagnostic was clarified, and the
  complete suite then passed.
- Task 3A lineage refinement RED: after adding unique/completeness/order coverage,
  the same command selected/executed 29 tests with 0 skips: 25 passed and exactly
  the four new duplicate, missing, reordered and split demo-lineage cases failed
  because the runtime did not yet reject them. The added valid-row nonfinite case
  already passed and confirmed the existing finiteness guard.
- Task 3A lineage refinement GREEN: the same command — **PASS**, 29
  selected/executed/passed, 0 failures/errors/skips.
- Related L1 regressions: `test_method_contracts.py` **PASS** 17/17,
  `test_method_config.py` **PASS** 24/24, `test_physical_geometry.py` **PASS**
  20/20, `test_episode_data.py` **PASS** 13/13, and `test_architecture.py`
  **PASS** 3/3, plus `test_policy.py` **PASS** 6/6; all had 0 skips.
- L0: `.venv/bin/python -B scripts/validate_fast.py` and `.venv/bin/python -B -S
  scripts/validate_fast.py` — **FAIL** overall. At this run both variants passed
  Python syntax, static harness boundary, and 19/19 harness self-tests with 0 skips;
  the only failures were the same five pre-existing missing evidence-log links
  under `docs/experiments/vv19-validation`. Both variants were rechecked after
  the Task 3A implementation with the same result. A newly introduced Markdown
  reference ambiguity was found and fixed before this final result. No evidence
  file/link was changed.
- L2/C1–C5: **NOT RUN** by this task — mandatory final integration acceptance
  remains outstanding.
- L3/L4, simulator, collection and training: **NOT RUN** — outside authorized scope.
- Remaining risk: Observable segmentation need not recover symbolic interactions;
  real-demo overflow/suitability and native window validity remain empirical.
  Tasks 2–3A have synthetic CPU evidence only; public raw-demo tensorization,
  `encode_events` orchestration, native-window/session construction, GPU or
  mixed-precision numerical behavior, and measured demonstrations remain untested.
