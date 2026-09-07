# P07: Action-conditioned physical dynamics and recursive imagination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Predict task-independent physical transitions with three fixed bootstrap heads and closed decode–reencode rollout.

**Architecture:** The neural dynamics trunk consumes physical tokens and one command descriptor only. The algorithm recursively decodes, re-encodes and updates branch-local physical state; task tracking remains outside predict_step.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Neural method](../../method/neural.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — planned runtime NOT IMPLEMENTED**. This document is a category C
component of the approved docs-only research migration, not permission to execute it.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00/P03/P04; P02 executed transitions; P06 only for external q update; FG geometry bridge before imagined IP use.
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

- Create: `src/icgs/models/dynamics/physical.py` — trunk and three residual heads.
- Create: `src/icgs/algorithms/rollout/physical.py` — predict_step and closed rollout.
- Create: `src/icgs/algorithms/objectives/physical.py` — geometry/dynamics losses.
- Test: `tests/test_world_model.py`.

Public capability boundary (planned; not currently importable):

```python
predict_step(state: PhysicalState, command: TimedCommand, *, head_id: int) -> PhysicalPrediction
bootstrap_mask(episode_ids, seed: int) -> Tensor
physical_loss(prediction, target, valid) -> Tensor
```

Four width256 attention blocks over132 rows; head256→256→259 gives residual feature/anchor, head256→256→7 gives achieved pose twist/grip logit. Three heads share trunk; fixed head ID remains attached across every interval. No task/context argument is allowed.

### Task 1: Physical-only trunk and residual achieved state

**Files:** Create `src/icgs/models/dynamics/physical.py`.
**Test owner:** `tests/test_world_model.py`.
**Consumes / produces:** Produces `PhysicalDynamics.forward(S, valid, u, head_id)` and `validate_head_id(head_id)`.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.models.dynamics.physical import validate_head_id
for head in range(3):
    validate_head_id(head)
for head in (-1, 3, True):
    with self.assertRaises(ValueError):
        validate_head_id(head)
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
Y = trunk(torch.cat((S, action_mlp(u)[:, None] + action_type), 1), valid132)
delta = geometry_heads[head_id](Y[:, :128])
pose_grip = pose_heads[head_id](Y[:, 128])
T_next = T_current @ se3_exp(unscale_twist(pose_grip[:, :6]))
```

Initialize geometry and pose residual final weights normal std1e-4,bias0; do not apply this override to grip output row. Keep valid anchor mask, hard grip>=0, FP32 geometry and achieved versus commanded pose. Tests inspect parameter independence and identical physics predictions under different external contexts.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 2: Closed rollout and branch-local causal update

**Files:** Create `src/icgs/algorithms/rollout/physical.py`.
**Test owner:** `tests/test_world_model.py`.
**Consumes / produces:** Produces predict_step; consumes decode_cloud,encode_cloud,physical_update and P04 branch_copy.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import inspect
from icgs.algorithms.rollout.physical import predict_step
self.assertEqual(tuple(inspect.signature(predict_step).parameters),
                 ('state', 'command', 'head_id'))
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
decoded = decode_cloud(residual_encoded)
encoded_next = encode_cloud(decoded.points_w, decoded.point_valid)
u_before = action_descriptor(state.T_w_e, command)
next_state = physical_update(encoded_next, predicted_observation, u_before,
                             state.memory)
# Store decoded.points_w directly as the native imagined cloud cache.
return PhysicalPrediction(next_state, grip_logits, head_id)
```

Count exactly one decode and one re-encode per interval; use no measured future frame or target memory after root. Maintain same head and common absolute command but recompute u per achieved pose. Update q separately after physical result; cache context/head/model/boundary IDs and reject nonfinite predictions.

The descriptor is rooted at each hypothesis's BEFORE pose, not its predicted
successor. In the +0.10m target/+0.06m achieved fixture, spy on physical_update
and assert the passed descriptor translation is +0.10m, not remaining +0.04m.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 3: Recursive executed-data losses and bootstrap support

**Files:** Create `src/icgs/algorithms/objectives/physical.py`.
**Test owner:** `tests/test_world_model.py`.
**Consumes / produces:** Produces `bootstrap_mask(episode_ids, seed)`, physical_loss and rollout_loss with teacher targets detached.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.algorithms.objectives.physical import bootstrap_mask
m = bootstrap_mask(['episode-a', 'episode-b'], seed=9)
self.assertEqual(tuple(m.shape), (2, 3))
self.assertTrue(m.any(dim=1).all().item())
self.assertTrue((m == bootstrap_mask(['episode-a', 'episode-b'], seed=9)).all().item())
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
loss_phys = cd / 0.01**2 + translation_mse / 0.01**2
loss_phys = loss_phys + rotation_angle_squared / (5*math.pi/180)**2 + grip_bce
loss = (mask[:, :, None]*loss_by_episode_head_step).sum() / (Kroll*mask.sum())
if encoder_decoder_trainable:
    loss = loss + 0.1*reconstruction_loss
```

Bootstrap episode Bernoulli0.8 with deterministic episode-hash modulo3 forced head if all zero; persist seed. Unroll predicted inputs from step2, keep fixed head, supervise each prefix. Frozen encoder/decoder parameters still allow input gradients; do not wrap continuous path in no_grad. Mask padded points and use stable training rotation clipping1e-6 versus exact reporting angle.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

## Acceptance, resource limits and evidence

L1 checks task-input exclusion, residual shapes, common commands, fixed head, decode count, gradients through frozen continuous modules and bootstrap masks. FG dynamics requires measured multihorizon geometry/pose/grip/ranking and head correlation on executed validation data; L1 is not contact-mode fidelity.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

No MCGS, posterior calibration claim, independent trunks or joint stochastic contact-mode modeling in primary. World-model refinement cannot update reference encoders/memories; its artifact may change without changing pi_ref only under that freeze.

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
- Remaining risk: Correlated deterministic heads and unrealizable decoded states may induce optimistic search despite correct bookkeeping.
