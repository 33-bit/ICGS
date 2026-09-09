# P06: Task memory, event routing and frozen reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Track full causal task history and route frozen IP between full context and valid event windows.

**Architecture:** A task neural module owns recurrent tensors; task state owns context/boundary lineage. Router algorithms consume native proposal capability through separately owned D1/D2 sessions; they never mutate native graph configuration.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Neural method](../../method/neural.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — planned runtime NOT IMPLEMENTED**. This document is a category C
component of the approved docs-only research migration, not permission to execute it.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00/P03–P05; P02 supervised labels; FG native one-demo/window compatibility before reference freeze.
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

TaskState r[B,256],alpha[B,Lc+1] including null,rho/nu/eligible[B,Lc],boundary/context/tracker lineage. Nonmonotonic recovery is legal. Reference pi_ref is exactly one routed sample plus r2 execution, with no V,H-conditioned choice or learned stop.

### Task 1: Recurrent tracker, null alignment and masked supervision

**Files:** Create task memory/state/objective files.
**Test owner:** `tests/test_task_router.py`.
**Consumes / produces:** Produces `task_event_features(events, r)` and `track_task`; masked CE/BCE consume separate program-derived label tensors.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
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

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_task_router.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
z = r_previous + scene_projection(masked_mean(S, S_valid, 1))
z = block2(block1(z[:, None], keys, key_valid), keys, key_valid)[:, 0]
r = gru(z, r_previous)
event_features = torch.cat((M, r[:, None].expand_as(M), M*r[:, None]), -1)
```

Compute dot-product alignment plus learned null logit, sigmoid768→256→3 event heads. Mask invalid events before softmax; landmarks remain tracker inputs. Include rho1/nu0 displacement, uniform matching across demos, null recovery and no monotonic clamp. Update boundary once; a context change replays all physical history through new M_C.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_task_router.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

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

L1 checks route probabilities, all-invalid fallback, window boundary counts, context replay and no mutation of native config. L2 FG one-demo verifies strict published load/inference for D1 and independent D2; capped L3 pilot measures stage-aware behavior before outcome collection.

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

- Documentation drafting: this plan specifies future work only.
- Component RED/GREEN commands: **NOT RUN** — runtime/test files are not implemented.
- L1 model assertions: **NOT RUN** — future installed supported environment required.
- L2/C1–C5: **NOT RUN** by this plan — preserve mandatory integration acceptance.
- L3/L4, collection and training: **NOT RUN** — separate resource authorization required.
- Remaining risk: Reference may have weak support or low completion rates, which require measured pilot diagnosis rather than relabeling.
