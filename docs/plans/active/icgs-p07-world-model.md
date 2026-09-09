# P07: Action-conditioned physical dynamics and recursive imagination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Predict task-independent physical transitions with three fixed bootstrap heads and closed decode–reencode rollout.

**Architecture:** The neural dynamics trunk consumes physical tokens and one command descriptor only. The algorithm recursively decodes, re-encodes and updates branch-local physical state; task tracking remains outside predict_step.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Neural method](../../method/neural.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — P07 runtime/test implementation present; measured FG and C1–C5
gates remain unmet**. This document is a category C component of the approved
research migration; implementation was authorized directly on `main`, but this
status does not authorize training, collection, simulator/robot work, or efficacy
claims.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00/P03/P04; P02 executed transitions; P06 only for external q update; FG geometry bridge before imagined IP use.
Read [workflow](../../WORKFLOW.md), [architecture](../../ARCHITECTURE.md),
[baseline](../../baselines/instant_policy.md), [native contract](../../components/policy-data-contract.md),
[research policy](../../RESEARCH.md) and [validation guide](../../../tests/README.md)
before runtime execution. Inspect Git state; preserve unrelated work and user assets.

## Configuration consumption addendum — 2026-09-09

Default values now belong to the [packaged primary JSON](../../../src/icgs/configuration/profiles/icgs_primary.json),
with key/unit/restriction details in the [parameter reference](../../method/parameters.md).
Consumed sections for this plan: **dynamics, geometry, decoder, neural, losses, numerics**.

Build fixed-compatible dimensions from config and derive concatenated/output sizes. Use dynamics.bootstrap_probability/residual_init_std and losses scales/weights. Do not turn freeze phase or physical/task separation into optional flags.

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

- Create: `src/icgs/models/dynamics/physical.py` — trunk and three residual heads.
- Create: `src/icgs/algorithms/rollout/physical.py` — bound predict_step capability and closed rollout.
- Create: `src/icgs/algorithms/objectives/physical.py` — geometry/dynamics losses.
- Test: `tests/test_world_model.py`.

Normative capability boundary:

```python
predict_step(state: PhysicalState, command: TimedCommand, *, head_id: int) -> PhysicalPrediction
bootstrap_mask(episode_ids, seed: int, *, config: MethodConfig) -> Tensor
physical_loss(prediction, target, valid, *, config: MethodConfig) -> Tensor
rollout_loss(predictions, targets, bootstrap, *, valid, config: MethodConfig) -> Tensor
```

Implementation seam: `PhysicalRollout` owns the explicitly injected dynamics,
encoder, decoder and P04 `PhysicalMemory` collaborators. `bind_predict_step(...)`
returns its bound `predict_step(state, command, *, head_id)` method, whose visible
parameter names are exactly `state`, `command`, `head_id`. There is deliberately no
module-level free function with a process-global default model and no collaborator
stored in `PhysicalState`. This is the smallest safe composition seam found in the
existing runtime. Manager ruling recorded for Fix Round 1: this bound-capability
form is approved for P07/P13 composition; do not replace it with global model state
or collaborators stored in `PhysicalState`.

With the primary config, four width256 attention blocks operate over132 rows;
head256→256→259 gives residual feature/anchor, and head256→256→7 gives achieved
pose twist/grip logit. Counts and slices are derived from the explicit
`MethodConfig`; three heads share the trunk, fixed head ID remains attached across
every interval, and no task/context argument is allowed.

### Task 1: Physical-only trunk and residual achieved state

**Files:** Create `src/icgs/models/dynamics/physical.py`.
**Test owner:** `tests/test_world_model.py`.
**Consumes / produces:** Produces `PhysicalDynamics.forward(S, valid, u, head_id)` and `validate_head_id(head_id)`.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.models.dynamics.physical import validate_head_id
for head in range(3):
    validate_head_id(head)
for head in (-1, 3, True):
    with self.assertRaises(ValueError):
        validate_head_id(head)
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
cfg = MethodConfig()
Y = trunk(torch.cat((S, action_mlp(u)[:, None] + action_type), 1), valid)
delta = geometry_heads[head_id](Y[:, :cfg.geometry.num_anchors])
pose_grip = pose_heads[head_id](Y[:, cfg.geometry.num_anchors])
T_next = T_current @ se3_exp(unscale_twist(pose_grip[:, :6]))
```

Initialize geometry and pose residual final weights normal std1e-4,bias0; do not apply this override to grip output row. Keep valid anchor mask, hard grip>=0, FP32 geometry and achieved versus commanded pose. Tests inspect parameter independence and identical physics predictions under different external contexts.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 2: Closed rollout and branch-local causal update

**Files:** Create `src/icgs/algorithms/rollout/physical.py`.
**Test owner:** `tests/test_world_model.py`.
**Consumes / produces:** Produces a bound predict_step capability and closed rollout;
consumes explicitly injected decode/encode/P04-memory collaborators and P04
`branch_copy`.

- [x] **Step 1 — RED:** The literal free-function signature check was used as the
initial boundary probe. Existing composition had no safe way to inject four model
collaborators into that exact free function without global state or `PhysicalState`
ownership, so the final owner test checks the returned bound capability instead.
The bound method exposes the same visible parameter names. The check remains a
named `unittest.TestCase` method in the test owner:

```python
import inspect
from icgs.algorithms.rollout.physical import bind_predict_step
step = bind_predict_step(dynamics, encoder, decoder, physical_memory, config=cfg)
self.assertEqual(tuple(inspect.signature(step).parameters),
                 ('state', 'command', 'head_id'))
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

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

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 3: Recursive executed-data losses and bootstrap support

**Files:** Create `src/icgs/algorithms/objectives/physical.py`.
**Test owner:** `tests/test_world_model.py`.
**Consumes / produces:** Produces `bootstrap_mask(episode_ids, seed, config)`,
`physical_loss` and `rollout_loss` with teacher targets detached. Every P07 public
boundary receives an explicit resolved `MethodConfig`; mapping/`None` resolution
is not part of this component.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.algorithms.objectives.physical import bootstrap_mask
cfg = MethodConfig()
m = bootstrap_mask(['episode-a', 'episode-b'], seed=9, config=cfg)
self.assertEqual(tuple(m.shape), (2, 3))
self.assertTrue(m.any(dim=1).all().item())
self.assertTrue((m == bootstrap_mask(['episode-a', 'episode-b'], seed=9, config=cfg)).all().item())
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
loss_phys = cd / 0.01**2 + translation_mse / 0.01**2
loss_phys = loss_phys + rotation_angle_squared / (5*math.pi/180)**2 + grip_bce
selected = bootstrap_mask[:, :, None] * supervision_mask[:, None, :]
loss = (selected * loss_by_episode_head_step).sum() / selected.sum()
if encoder_decoder_trainable:
    loss = loss + cfg.losses.reconstruction_weight*reconstruction_loss
```

Bootstrap episode Bernoulli0.8 with deterministic episode-hash modulo3 forced head if all zero; persist seed. Unroll predicted inputs from step2, keep fixed head, supervise each prefix. Frozen encoder/decoder parameters still allow input gradients; do not wrap continuous path in no_grad. Mask padded points and use stable training rotation clipping1e-6 versus exact reporting angle.

`rollout_loss` accepts concrete `PhysicalPrediction` records and native
`ExecutedTransition`/`TimedObservation`/`Observation` targets. Its optional
`valid` mask is exactly boolean `[episodes, rollout_steps]`; false padding is
excluded from both numerator and denominator, and zero supervised steps are
rejected. Each selected prediction must carry the matching head slot. The physical
CD is the sum of the two directed means from Eq.\ \ref{eq:cd}; translation remains
the configured MSE and rotation remains the configured geodesic unit.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

## Fix Round 1 audit record — 2026-09-09

The first Sol audit returned **REQUEST CHANGES**. The manager verified the accepted
findings against the implementation and the source CD equation. This record keeps
that failed audit visible; the changes below do not certify a trained artifact,
foreground fidelity, or efficacy.

1. **CRITICAL — tensor gradient ownership:** rollout grip logits were converted to
   NumPy before `PhysicalPrediction`, so BCE could not reach pose output row 6.
   `PhysicalPrediction` now clones torch logits without detaching while retaining
   its copied/read-only NumPy path. Regression tests are
   `test_physical_prediction_owns_numpy_and_tensor_logits`,
   `test_real_rollout_grip_bce_preserves_pose_grip_gradient`, and
   `test_real_two_step_rollout_loss_reaches_earlier_prediction_through_frozen_bridge`.
2. **HIGH — masked rollout count:** `rollout_loss` now requires boolean
   `[episodes, rollout_steps]` supervision, excludes false padding from numerator
   and denominator, and rejects zero supervised steps. Tests are
   `test_rollout_loss_is_invariant_to_masked_padding` and
   `test_rollout_loss_rejects_invalid_supervision_mask_shapes_and_empty_supervision`.
3. **HIGH — directed CD:** the implementation now sums both directed means, with
   no extra `/2`. `test_physical_loss_uses_sum_of_directed_cd_terms` isolates a
   one-point 1 cm contribution of exactly `2.0` with grip weight zero.
4. **HIGH — empty geometry:** supervised physical samples with no predicted or
   target geometry are rejected; dynamics rejects a mixed batch row with no valid
   geometry anchors even when nongeometry rows are valid. Tests are
   `test_physical_loss_rejects_valid_samples_without_geometry` and
   `test_physical_dynamics_rejects_mixed_batch_without_geometry_anchor`.
5. **MEDIUM — prediction contract/head identity:** loss inputs are narrowed to
   concrete `PhysicalPrediction` plus native executed/observation targets, and
   every selected step checks its `head_id` against its slot. Test:
   `test_rollout_loss_rejects_selected_prediction_head_mismatch`.
6. **MEDIUM SPEC — explicit configuration:** P07 dynamics, rollout binding and
   losses require a resolved `MethodConfig`; dimensions/slices come from
   `derived_dimensions()` and typed sections. Tests are
   `test_p07_boundaries_require_explicit_method_config` and
   `test_nondefault_p07_config_reaches_bootstrap_depth_loss_and_weights`.

## Fix Round 2 numerical audit record — 2026-09-09

Sol's scoped re-audit confirmed closure of all six Fix Round 1 findings. The
remaining request was a numerical training-objective correction, not a new
architecture or contract: `_rotation_angle_squared` had been zeroing the
identity interval after the hard clip. The manager ruling is that the training
objective is exactly
`acos(cosine.clamp(-1 + margin, 1 - margin)).square()`; exact identity zero is
reporting-only. The hard clip therefore permits zero gradient inside the clip,
and no straight-through estimator or alternate geodesic formula is allowed.

The approved change removes only that interval-zeroing branch. Real
`PhysicalDynamics` → `PhysicalEncoder` → `PhysicalDecoder` → `PhysicalMemory`
→ `PhysicalRollout` BCE and recursive gradients remain covered by the Round 1
tests. New real `physical_loss` regressions use `MethodConfig`'s configured
margin and isolate rotation with identical one-point geometry, zero translation,
and `grip_weight=0`: identity, 0.0005 rad, and 0.001 rad equal the positive
clip-floor; 0.005 rad has finite nonzero gradient; near-π loss and gradient are
finite. Existing exact-CD, padding, real BCE, and nondefault-weight baselines
explicitly include the same identity floor and retain strict equality checks.

TDD evidence in the supported `.venv` (Python 3.11.15, PyTorch 2.2.0, NumPy
1.26.4): before the production change, the focused command
`./.venv/bin/python -B -m unittest discover -s tests -p 'test_world_model.py' -v`
ran **28 tests: 5 expected failures, 0 errors, 0 skipped**, all failing on the
old zeroed identity contribution or its dependent baselines. After the minimal
change, the same command ran **28 passed, 0 skipped**. This is numerical
contract evidence only; it is not measured FG evidence or an efficacy claim.

Downstream integration gaps remain explicit: any artifact initialization must
record its seed, and any cached context must carry the context, head, model, and
boundary IDs needed to reject stale or cross-branch reuse. These are downstream
composition requirements to wire through existing records, not permission to
add a metadata/cache framework in P07.

## Acceptance, resource limits and evidence

L1 checks task-input exclusion, residual shapes, common commands, fixed head, decode count, gradients through frozen continuous modules and bootstrap masks. FG dynamics requires measured multihorizon geometry/pose/grip/ranking and head correlation on executed validation data; L1 is not contact-mode fidelity.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_world_model.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- C1–C5 remain hard limits for this implementation turn: no published artifact,
  pinned Linux/CUDA reference environment, or authorized reference comparison was
  available/run. Therefore C1 load, C2 strict load, C3 real inference, C4 reference
  fidelity and C5 independent installed execution are all **NOT RUN**, not PASS.
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

- Baseline/workspace: began from `88b1043927d8263dc15880b68c5006a4d195e583`
  (`main`); direct-main edits were explicitly authorized, with no worktree/branch
  creation and no commit. Scope includes the P07 runtime modules, the minimal
  tensor-preserving `PhysicalPrediction` contract extension, focused tests and
  this plan.
- Initial implementation checkpoint before the first Sol audit: the focused L1
  command `./.venv/bin/python -B -m unittest discover -s tests -p
  'test_world_model.py' -v` reported **15 passed, 0 skipped**, but the audit
  remained failed because it exposed the six contract/arithmetic gaps recorded
  below. No efficacy was inferred from that checkpoint.
- Fix Round 1 TDD RED after adding regression assertions: the same focused command
  ran **25 tests: 7 failures, 3 errors**; the independent method-contract command
  ran **17 passed, 0 skipped**. The seven failures were expected missing behavior;
  the three errors were the new fixtures still passing the episode-step mask in
  the bootstrap slot and were corrected before the GREEN implementation.
- Fix Round 1 GREEN focused L1: the same P07 command ran **25 passed, 0 skipped**.
  It includes the real dynamics→encoder→decoder→memory BCE gradient path on pose
  output row 6, the real two-step frozen-bridge rollout gradient, native executed
  targets, padding invariance, shape/empty-supervision rejection, head matching,
  exact directed-CD contribution, geometry-anchor rejection, and nondefault config
  propagation.
- Required focused commands in the supported `.venv` all passed with zero skips:
  `./.venv/bin/python -B -m unittest discover -s tests -p 'test_method_contracts.py' -v`
  (**17**), `./.venv/bin/python -B -m unittest discover -s tests -p
  'test_method_config.py' -v` (**24**), `./.venv/bin/python -B -m unittest discover
  -s tests -p 'test_physical_geometry.py' -v` (**20**), and
  `./.venv/bin/python -B -m unittest discover -s tests -p
  'test_physical_memory.py' -v` (**17**).
- Supported L1 environment: `.venv/bin/python`, Python **3.11.15**, PyTorch
  **2.2.0**, NumPy **1.26.4**, editable `icgs` import from `src/icgs`.
- Focused prerequisite regressions: `test_physical_memory.py` **17/17 PASS** and
  `test_physical_geometry.py` **20/20 PASS** in the same environment.
- L0: `python3 -B scripts/validate_fast.py` **PASS, 19/19 harness tests**;
  `python3 -B -S scripts/validate_fast.py` **PASS, 19/19 harness tests**.
  These are syntax/link/boundary checks only; Python `3.14.4` is the host L0
  interpreter and did not execute model code.
- Fix Round 2 final focused L1: after the numerical correction,
  `./.venv/bin/python -B -m unittest discover -s tests -p 'test_world_model.py' -v`
  reported **28 passed, 0 skipped**. The same supported environment reran method
  contracts/config (**17/17**, **24/24**), P03 geometry (**20/20**) and P04 memory
  (**17/17**), all PASS with zero skips.
- Full installed discovery rerun: **217 passed, 8 skipped**. Skips are the absent
  differential source, optional RLBench/simulator dependency, unavailable CUDA,
  opt-in published-checkpoint fixture, and optional Lightning; none is counted as
  PASS. No training, download, preprocessing job, simulator, or robot workload ran.
- `git diff --check`: **PASS**. The working tree remains uncommitted on `main`.
- Final review: GPT 5.6 Luna Max implemented and completed two scoped fix rounds;
  GPT 5.6 Sol Medium independently accepted isolated P07 L1 specification and
  implementation quality, with **no blocking findings**. The six first-round
  findings and second-round training rotation correction are verified closed.
  No generic registry, global model binding, or unrelated attention refactor was
  introduced. This acceptance does not close the overall P07 plan.
- Manager verification from `/Users/33bit/AI/Research/VLA/ICGS`:
  `.venv/bin/python -B -m unittest discover -s tests -p test_world_model.py -v`
  **28/28 PASS, 0 skipped**; both L0 commands above **19/19 PASS** and
  `git diff --check` **PASS**. Sol independently reran full discovery with
  **217 PASS, 8 SKIPPED** in the same environment.
- Commit/review state: final independent review accepted; changes remain
  uncommitted on `main` for owner handoff.
- Manager ruling for this round: the minimal bound-capability `bind_predict_step`
  seam is approved for later P13/composition wiring. Do not add a global model, put
  collaborators in `PhysicalState`, or invent a larger registry/framework. The
  Round 2 ruling also rejects any below-threshold nonzero-gradient request and
  any straight-through or alternate geodesic implementation; the hard clipped
  training objective is the specified one.
- L2/C1–C5, L3/L4, collection and training: **NOT RUN**. No measured FG dynamics,
  bridge, ranking, efficacy, or contact-mode claim is made. Remaining risk:
  correlated deterministic heads and unrealizable decoded states may induce
  optimistic search despite correct bookkeeping, and no trained P07 artifact exists.

## Observability integration addendum — 2026-09-09

`PhysicalRollout` now accepts an optional outer recorder and emits nested
`physical.memory.read`, `physical.dynamics`, `physical.decode`,
`physical.encode` and `physical.memory.update` spans. These observe host-side
boundary identity only and preserve collaborator order/math. No model-forward
producer, replay snapshot or imagined-state capture is fabricated.
