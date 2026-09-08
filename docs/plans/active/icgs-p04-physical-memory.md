# P04: Causal physical memory and branch-local state Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Maintain once-per-boundary physical history independently of task context and imagined branches.

**Architecture:** Tensor modules produce physical GRU updates; state owns causal boundaries, array/tensor ownership and lineage. Execution later orchestrates real updates; dynamics consumes branch copies through the same update capability.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Neural method](../../method/neural.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — P04 modules/tests implemented; supported validation and integration gates pending**. This document is a category C
component of the approved docs-only research migration; it does not authorize expensive workloads.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00/P03; P02 causal histories for training and replay.
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

- Create: `src/icgs/models/memories/physical.py` — proprioception/action encoding and two GRUs.
- Create: `src/icgs/state/physical.py` — PhysicalState and causal lineage.
- Test: `tests/test_physical_memory.py`.

Public capability boundary (planned; not currently importable):

```python
physical_update(encoded, observation, previous_descriptor, memory) -> PhysicalState
PhysicalState.branch_copy() -> PhysicalState
PhysicalMemory.physical_tokens(state: PhysicalState) -> tuple[Tensor, Tensor]
validate_next_boundary(previous: int, current: int) -> None
```

PhysicalState has X[B,128,256],x[B,128,3],valid[B,128],p[B,13],memory[B,2,256],pose/grip, cached world cloud/mask,boundary,origin and encoder/memory lineage. No context or task tensor is accepted by physical_update.

### Task 1: Proprioception and previous command features

**Files:** Create `src/icgs/models/memories/physical.py`.
**Test owner:** `tests/test_physical_memory.py`.
**Consumes / produces:** Produces `proprioception(T_w_e, grip, gravity)` and `action_descriptor(T_w_e, command)`; consumes P03 SE(3).

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import torch
from icgs.models.memories.physical import proprioception
p = proprioception(torch.eye(4)[None], torch.zeros(1, 1),
                   torch.tensor([[0., 0., -1.]]))
self.assertEqual(tuple(p.shape), (1, 13))
torch.testing.assert_close(p[0, 3:9], torch.tensor([1., 0., 0., 0., 1., 0.]))
import numpy as np
from icgs.contracts.method import TimedCommand
from icgs.models.memories.physical import action_descriptor
target = np.eye(4); target[0, 3] = .10
command = TimedCommand(target, 0, .1)
before = torch.eye(4)[None]
successor = before.clone(); successor[0, 0, 3] = .06
u_before = action_descriptor(before, command)
self.assertAlmostEqual(u_before[0, 0].item(), .10, places=6)
self.assertNotAlmostEqual(u_before[0, 0].item(),
                          action_descriptor(successor, command)[0, 0].item())
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_physical_memory.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
rot6 = torch.cat((T_w_e[..., :3, 0], T_w_e[..., :3, 1]), dim=-1)
p = torch.cat((T_w_e[..., :3, 3] / ell0, rot6, grip, gravity), -1)
u = torch.cat((scaled_se3_log(torch.linalg.inv(T_w_e) @ command.target_w),
               commanded_grip, log_duration_ratio), -1)
```

Use planned duration not measured completion time; reset u_previous is zero.
Compute previous_descriptor [B,8] from the BEFORE-transition achieved pose and
absolute command, then pass it unchanged into the successor memory update.
For an identity before pose, a target translated +0.10m and achieved successor
translated +0.06m, assert previous_descriptor[0,0] == 0.10, not 0.04.
For different before-pose hypotheses under the same command, descriptors differ
while target_w remains identical. Add joint-yaw and invalid pose/gravity/grip tests.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_physical_memory.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 2: Two-layer temporal update and masked token set

**Files:** Create physical memory module and state record.
**Test owner:** `tests/test_physical_memory.py`.
**Consumes / produces:** Produces `PhysicalMemory.forward(X, valid, p, previous_u, memory)` returning memory; `PhysicalMemory.physical_tokens(state)` returns 131 rows and validity.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import torch
from icgs.models.memories.physical import PhysicalMemory
net = PhysicalMemory()
m = net(torch.ones(1, 128, 256), torch.ones(1, 128, dtype=torch.bool),
        torch.zeros(1, 13), torch.zeros(1, 8), torch.zeros(1, 2, 256))
self.assertEqual(tuple(m.shape), (1, 2, 256))
m.sum().backward()
self.assertTrue(any(p.grad is not None for p in net.parameters()))
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_physical_memory.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
v = input_mlp(torch.cat((masked_mean(X, valid, 1), p, previous_u), -1))
m1 = gru1(v, memory[:, 0])
m2 = gru2(m1, memory[:, 1])
return torch.stack((m1, m2), dim=1)
```

Input MLP277→256→256; PyTorch GRUCell reset-after-hidden-affine semantics. Physical tokens concatenate geometry/proprioception/two memories with type embeddings; invalid geometry rows remain masked. Ensure A1 temporal loss reaches these GRUs before reference freeze; geometry-only warm-up is insufficient.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_physical_memory.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 3: Boundary uniqueness and mutation isolation

**Files:** Create `src/icgs/state/physical.py`.
**Test owner:** `tests/test_physical_memory.py`.
**Consumes / produces:** Produces `validate_next_boundary`, branch_copy and real-update origin/lineage validation.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.state.physical import validate_next_boundary
validate_next_boundary(3, 4)
for boundary in (3, 5):
    with self.assertRaisesRegex(ValueError, 'boundary'):
        validate_next_boundary(3, boundary)
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_physical_memory.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
if current != previous + 1:
    raise ValueError('physical update requires exactly the next boundary')
# branch_copy clones each mutable tensor; shares only owned immutable metadata.
# Mark the returned branch imagined without changing the real root object.
```

Reset boundary0 is encoded once; replanning at an existing boundary is a read, not another GRU update. Reject imagined state as a measured-history update. Test two branches mutate independently, context swap leaves physical state reusable only under matching physical lineage, and full-history replay equals sequential updates.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_physical_memory.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

## Acceptance, resource limits and evidence

L1 requires causal prefix, boundary double-update rejection, copy isolation and nonzero temporal gradients with tiny tensors. A1 memory usefulness is an authorized training/pilot question, not established by forward/backward shape tests.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_physical_memory.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

Never extend native PreparedContext to hold physical history or clone model/graph/scheduler per branch. No claimed sufficient Markov state, contact observability or hidden simulator information.

On failure, retain the failed evidence and isolate the new component/config ID.
Do not overwrite native checkpoints, relabel collected outcomes or drop difficult samples.
Resume only from a compatible explicit artifact/manifest; changed reference lineage
requires new outcome labels, not a cache workaround. Return to the preceding accepted
phase if an FG fails and request a scoped protocol decision.

## Execution evidence

- Documentation drafting: the P04 component modules/tests were implemented in the planned owners after exact source/test-diff review; no native integration was added.
- Component RED/GREEN commands: source-path diagnostic RED/GREEN evidence is recorded in the controller ledger and implementation report. Final diagnostic GREEN before the ownership fix was 13 selected/13 executed/0 skipped; the fix-round GREEN is 14 selected/14 executed/0 skipped in Python 3.14.4 with torch 2.11.0. The exact installed-environment command was attempted but remains unavailable because `icgs` is not installed in the host interpreter; it is not counted as PASS.
- Fix round 1/5: after exact source/test-diff review, the learned token projector is registered inside each `PhysicalMemory`; `physical_tokens` is now the instance method `PhysicalMemory.physical_tokens(state)`, with no global learned module or cache. The ownership RED was feature-specific (`AttributeError` for the absent method); GREEN covered registration in `parameters()`/`state_dict()`, nonzero token gradients, per-instance parameter identity, and zero invalid geometry rows.
- L1 model assertions: **SKIPPED** for the supported installed environment — no Python >=3.10,<3.13 environment with the pinned ICGS installation is available. The source-path run is diagnostic smoke evidence only, not supported L1 acceptance.
- L0 PASS: `python3 -B scripts/validate_fast.py`, 19/19 harness self-tests, syntax/links/boundaries pass; L1/L2/L3/L4 not run by this command.
- L0 PASS: `python3 -B -S scripts/validate_fast.py`, 19/19 harness self-tests, syntax/links/boundaries pass; L1/L2/L3/L4 not run by this command.
- L2/C1–C5: **NOT RUN** by this plan — preserve mandatory integration acceptance.
- L3/L4, collection and training: **NOT RUN** — separate resource authorization required.
- Remaining risk: supported-environment L1 compatibility, temporal training usefulness, physical ambiguity sufficiency, native integration, and mandatory C1–C5 remain unverified. No training, downloads, preprocessing jobs, simulator workloads, robot motion, or network work was run.
