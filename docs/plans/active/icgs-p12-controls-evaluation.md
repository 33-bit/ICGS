# P12: Mandatory B0–B7 controls and statistical evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Implement the complete matched baseline ladder and evidence protocol including a separate equal-data diffusion student.

**Architecture:** Control policies compose shared frozen representations with independent student/Q heads or existing planners. Benchmark owners keep privileged predicates outside online state; statistics operate on immutable per-episode outcome panels.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Planning and evaluation](../../method/planning-evaluation.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — planned runtime NOT IMPLEMENTED**. This document is a category C
component of the approved docs-only research migration, not permission to execute it.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00–P11, locked independent split/reset/context panels; FG predicate/assets/cadence and explicit evaluation/training budgets.
Read [workflow](../../WORKFLOW.md), [architecture](../../ARCHITECTURE.md),
[baseline](../../baselines/instant_policy.md), [native contract](../../components/policy-data-contract.md),
[research policy](../../RESEARCH.md) and [validation guide](../../../tests/README.md)
before runtime execution. Inspect Git state; preserve unrelated work and user assets.

## Configuration consumption addendum — 2026-09-09

Default values now belong to the [packaged primary JSON](../../../src/icgs/configuration/profiles/icgs_primary.json),
with key/unit/restriction details in the [parameter reference](../../method/parameters.md).
Consumed sections for this plan: **reactive, action_value, benchmark, planning, stages, dataset, losses**.

Use reactive schedule/scale/weight and action_value architecture controls; read matched panel/statistics/predicate settings from benchmark and proposal splits from dataset. Bind named comparison metrics to resolved config; do not treat optional-track values as permission to execute.

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

- Create: `src/icgs/models/denoisers/reactive.py` — separate B3 diffusion student.
- Create: `src/icgs/algorithms/objectives/reactive.py` and `src/icgs/policies/reactive.py`.
- Create: `src/icgs/models/evaluators/action_value.py` and `src/icgs/policies/controls.py`.
- Consume P02 `src/icgs/evaluation/dependencies.py`; Create: `src/icgs/evaluation/method.py`, `statistics.py`.
- Test: `tests/test_method_evaluation.py`.

Public capability boundary (planned; not currently importable):

```python
build_control(control_id, components, config) -> MethodControl
ReactiveStudent.sample(state, task, events, *, seed: int) -> ActionTrajectory
ActionValue.forward(state, task, events, commands, H: int) -> Tensor
summarize_panel(episodes, protocol) -> EvaluationReport
```

B0 native; B1 full context plus method cadence; B2 routed reference action policy; B3 separate diffusion student; B4 directQ; B5 WM+V rerank; B6shooting; B7MCTS. B1–B7 share learned stopping plus matched disabled control. B3 is not IP fine-tuning or the same action prior.

### Task 1: Separate outcome-weighted B3 diffusion student

**Files:** Create reactive denoiser/policy/objective files.
**Test owner:** `tests/test_method_evaluation.py`.
**Consumes / produces:** Produces `outcome_weight(returns)`, B3 codec and ReactiveStudent.sample with no H/online IP ranking.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import torch
from icgs.algorithms.objectives.reactive import outcome_weight
w = outcome_weight(torch.tensor([0., .5, 1.]))
torch.testing.assert_close(w, torch.tensor([.1, .55, 1.]))
from icgs.models.denoisers.reactive import ReactiveDenoiser
net = ReactiveDenoiser()
x, t = torch.randn(1, 8, 10), torch.tensor([50])
S, K, r = torch.randn(1, 131, 256), torch.randn(1, 2, 256), torch.zeros(1, 256)
sv, kv = torch.ones(1, 131, dtype=torch.bool), torch.tensor([[True, False]])
av = torch.tensor([[True, True, False, False, False, False, False, False]])
y = net(x, t, S, K, r, physical_valid=sv, event_valid=kv, action_valid=av)
changed = K.clone(); changed[:, 1] = 1000.
y_masked = net(x, t, S, changed, r, physical_valid=sv, event_valid=kv, action_valid=av)
self.assertEqual(tuple(y.shape), (1, 8, 10))
torch.testing.assert_close(y, y_masked)
fixed_noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(7), dtype=x.dtype)
perturbed = x.clone(); perturbed[~av] = 1000.
def predict_with_fixed_noise(values):
    clean = torch.where(av[..., None], values, torch.zeros_like(values))
    noise = torch.where(av[..., None], fixed_noise, torch.zeros_like(fixed_noise))
    noisy = .8*clean + .6*noise
    return net(noisy, t, S, K, r, physical_valid=sv, event_valid=kv, action_valid=av)
a, b = predict_with_fixed_noise(x), predict_with_fixed_noise(perturbed)
torch.testing.assert_close(a[av], b[av])
torch.testing.assert_close((a-fixed_noise).square()[av].mean(),
                           (b-fixed_noise).square()[av].mean())
raw_perturbed = x.clone(); raw_perturbed[~av] = 1000.
direct = net(raw_perturbed, t, S, K, r, physical_valid=sv, event_valid=kv, action_valid=av)
torch.testing.assert_close(y[av], direct[av])
self.assertTrue((direct[~av] == 0).all().item())
with self.assertRaises(ValueError):
    net(x, t, S, K, r, physical_valid=sv, event_valid=kv, action_valid=torch.zeros_like(av))
y.square().mean().backward()
self.assertTrue(any(p.grad is not None for p in net.parameters()))
self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_evaluation.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
targets = torch.cat((root_relative_translation / cfg.reactive.translation_scale_m, root_relative_rot6,
                     commanded_grip_pm1), -1)  # [B,8,10], executed prefix mask
if not action_valid.any(dim=1).all():
    raise ValueError('each example requires at least one valid action target')
targets = torch.where(action_valid[..., None], targets, torch.zeros_like(targets))
noise = torch.where(action_valid[..., None], torch.randn_like(targets), torch.zeros_like(targets))
noisy_targets = alpha_bar.sqrt()*targets + (1-alpha_bar).sqrt()*noise
epsilon = student(noisy_targets, diffusion_time, physical_tokens, event_keys, task_r,
                  physical_valid=physical_valid, event_valid=event_valid, action_valid=action_valid)
per_example = masked_mean((epsilon-noise).square(), action_valid, dims=(1,2))
floor = cfg.reactive.outcome_weight_floor
weights = floor + (1-floor)*returns if cfg.reactive.outcome_weighted else torch.ones_like(returns)
loss = (weights*per_example).sum()/weights.sum()
```

Eight256-wide action tokens with horizon positions and sinusoidal timestep embedding256;4 decoder blocks self/cross attention to[S;K;r],8heads,FFN1024,dropout0;head256→256→10. Separate100-step squared-cosine DDPM epsilon schedule,DDIM eta0 four inference steps,no clipping. Decode Rot6 Gram-Schmidt; norm<1e-6/nonfinite rejects with counted reference fallback. Use executed commands only, masked tails; terminal return1/0 or active k(H)/n at sampled recorded H, no H inference input. Batch128,max100k,shared optimizer defaults; unweighted retrained control uses same bank.

The tested student module signature is `ReactiveDenoiser.forward(noisy_actions,
diffusion_time, physical_tokens, event_keys, task_r, *, physical_valid,
event_valid, action_valid) -> Tensor[B,8,10]`. The policy adapter alone owns
schedule/decode and sets all eight action positions valid at inference.
Sanitize invalid values before noising and zero invalid token rows after adding
position/time embeddings, before their first normalization. Mask invalid action
self-attention keys and zero invalid query rows after every decoder block and
output projection. Loss uses the same [B,8] mask, not only attention masking.
Also assert masked loss is unchanged for the fixed-noise tail perturbation;
absent tail commands must neither condition valid predictions nor become zero-action
supervision. Reject examples with no valid target before computing attention/loss.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_evaluation.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 2: Direct Q and exact control mapping

**Files:** Create action-value evaluator and controls policy.
**Test owner:** `tests/test_method_evaluation.py`.
**Consumes / produces:** Produces `control_kind(control_id)` and ActionValue.forward with prefix return labels.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.policies.controls import control_kind
self.assertEqual(control_kind('B3'), 'reactive-diffusion')
self.assertEqual(control_kind('B4'), 'direct-q')
self.assertEqual(control_kind('B7'), 'mcts')
with self.assertRaises(ValueError):
    control_kind('B8')
from icgs.models.evaluators.action_value import prefix_horizon
self.assertEqual(prefix_horizon(8, 32, 512), 40)
with self.assertRaisesRegex(ValueError, 'horizon'):
    prefix_horizon(8, 512, 512)
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_evaluation.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
descriptors = [action_descriptor(observed_root_pose, c) for c in commands]
a = command_gru(pack_valid(descriptors)).last_valid_state
logit_Q = mlp_512_256_1(torch.cat((value_query, a), -1))
# BCE prefix return: active records weighted n, terminal prefixes weighted1.
```

Define GRUwidth256 on all descriptors relative to observed root, never predicted future pose. Same h/H/candidate pool, own calibration logits, same data/reference. B5 Lh2/T8 and B6/B7 matchedL8/16/32/64,r<=h; B3 has no online IP/dynamics call except failure fallback. B8 stays unimplemented optional competitor until official reproduction.

Implement `prefix_horizon(prefix_intervals, successor_H, Hmax)` with
H_root=prefix_intervals+successor_H, rejecting H_root>Hmax rather than clipping.
Assert prefix_horizon(8,32,512)==40: successor k(32)/n trains Q40, not Q32.
Record this same root/successor horizon provenance for B3 outcome weights even
though B3 receives no H at inference. Terminal prefix labels retain their root
deadline and observed first-terminal time; prefixes beyond that deadline reject.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_evaluation.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 3: External custom predicates and locked experiment panels

**Files:** Consume P02 `src/icgs/evaluation/dependencies.py`; create `src/icgs/evaluation/method.py`.
**Test owner:** `tests/test_method_evaluation.py`.
**Consumes / produces:** Consumes P02 `first_terminal(success, failure, absorbed_success)` and external TaskMonitor; produces panel validation and E1–E9 runner.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.evaluation.dependencies import first_terminal
self.assertEqual(first_terminal(True, True, False), 'failure')
self.assertEqual(first_terminal(False, True, True), 'success')
self.assertEqual(first_terminal(False, False, False), 'continue')
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_evaluation.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
if absorbed_success:
    return 'success'
if failure:
    return 'failure'
return 'success' if success else 'continue'
```

Reuse P02's exact catalog predicates, five-boundary stability/history and engine-validated force/penetration; do not implement a second monitor. Online models never receive monitor output. Lock12 compositions×100 resets×2 demos,H512,3 trained seeds; one-demo50 resets. E2/E3 realized-state diagnostics precede E4 learned-world claims. Include independent-chain/memory controls, suffix ambiguity and feasible J/delta cells without test filtering.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_evaluation.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 4: Matched statistics, failure accounting and compute fairness

**Files:** Create `src/icgs/evaluation/statistics.py` and complete method runner.
**Test owner:** `tests/test_method_evaluation.py`.
**Consumes / produces:** Produces `trial_brier(probabilities, trial_labels)` and paired hierarchical panel summaries.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.evaluation.statistics import trial_brier
self.assertAlmostEqual(trial_brier([.5, .5], [0, 1]), .25)
with self.assertRaises(ValueError):
    trial_brier([], [])
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_evaluation.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
brier = sum((p-y)**2 for p,y in zip(probabilities, trial_labels))/len(trial_labels)
# Average matched learned-seed outcomes within reset/context before suite contrast.
# Bootstrap compositions then reset/context IDs with shared indices (10000 resamples).
# Retain per-seed SR/spread, paired effect sizes, 95% intervals and Holm family IDs.
```

Report macro/unconditional/conditional success,NLL,trial Brier,ECE10,regret with trial uncertainty,actual boundaries,call/latency/memory/support/false-stop/invalid counts. Primary E4 B7–B6; E3 mechanistic. Never replicate B0 as independent trained seeds. Predetermined failure order allows secondary/unknown tags. Retrain changed architecture/loss ablations; preserve negative MCTS-vs-shooting findings.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_evaluation.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

## Acceptance, resource limits and evidence

L1 validates codec/weights/forward masks/Q targets/control mapping,predicate precedence,matched panels and small deterministic bootstrap arithmetic. B3 tiny forward/backward is not trained control. L3/L4 primary suite and published regression require explicit resource scope; optional public transfer/B8/robot tracks remain NOT RUN without their gates.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_method_evaluation.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

No IP fine-tuning, extra data only for favored controls, oracle value on arbitrary decoded clouds or hidden test threshold tuning. Stop incomplete panels honestly; preserve requested-run denominator and missing/invalid accounting. Benchmark success does not establish all-task generality.

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
- Remaining risk: Suite feasibility, equal-data training and matched compute measurements remain unexecuted; B3 can generate outside the IP pool.
