# P03: Physical geometry encoder, decoder and native bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Represent full foreground geometry and reconstruct world clouds for imagined native-policy queries.

**Architecture:** Add independent 5-mm/FPS preprocessing, metric SE(3) helpers and width256 geometry modules. Native live preprocessing and graph/sampling math remain untouched; bridges pass world clouds exactly once.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Neural method](../../method/neural.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — planned runtime NOT IMPLEMENTED**. This document is a category C
component of the approved docs-only research migration, not permission to execute it.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00/P02, fixed sensor/crop profile and native checkpoint availability for later FG bridge validation.
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

- Create: `src/icgs/geometry/se3.py` — translation-first Log/Exp.
- Create: `src/icgs/data/preprocessing/physical.py` — crop/voxel/FPS/masks.
- Create: `src/icgs/models/encoders/physical.py` and `src/icgs/models/decoders/physical.py`.
- Create: `src/icgs/models/layers/method.py` — masked pre-LN attention/MLPs.
- Create: `src/icgs/execution/observation_bridge.py`; Test: `tests/test_physical_geometry.py`.

Public capability boundary (planned; not currently importable):

```python
encode_cloud(points_w, point_valid) -> EncodedCloud
decode_cloud(encoded: EncodedCloud) -> DecodedCloud
se3_exp(xi) -> Tensor
se3_log(transform) -> Tensor
imagined_observation(state: PhysicalState) -> Observation
```

2048 sampled points,128 anchors,32 neighbors,width256; repeat sparse points with invalid duplicate-anchor masks. p uses translation/ell0, first two rotation columns, measured grip and gravity. All geometry is in meters/radians, ell0=1m; empty valid clouds reject.

### Task 1: Stable metric SE(3) and preprocessing masks

**Files:** Create `src/icgs/geometry/se3.py` and `src/icgs/data/preprocessing/physical.py`.
**Test owner:** `tests/test_physical_geometry.py`.
**Consumes / produces:** Produces `se3_exp`, `se3_log` and `prepare_physical_cloud(points, valid, config)`.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import torch
from icgs.geometry.se3 import se3_exp, se3_log
xi = torch.tensor([[0.02, -0.01, 0.03, 1e-7, 0., 0.]], dtype=torch.float64)
torch.testing.assert_close(se3_log(se3_exp(xi)), xi, atol=1e-9, rtol=1e-7)
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_physical_geometry.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
# xi = [rho, omega]; translation is J(omega) @ rho, not rho.
# Use Taylor coefficients near zero; near pi recover a stable rotation axis.
R = so3_exp(omega)
t = so3_left_jacobian(omega) @ rho.unsqueeze(-1)
T = homogeneous(R, t)
```

Implement private SO(3)/Jacobian/homogeneous helpers in the same file with finite inverse/roundtrip tests, including near-pi rotations and noncommuting translation. Deterministic tie handling for voxel/FPS/kNN, declared crop and joint yaw augmentation of cloud/pose/commands; index selection has no gradient.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_physical_geometry.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 2: Masked geometry tokens and patch reconstruction

**Files:** Create physical encoder/decoder and `src/icgs/models/layers/method.py`.
**Test owner:** `tests/test_physical_geometry.py`.
**Consumes / produces:** Produces `masked_mean(values, valid, dim)`, `PhysicalEncoder`, `PhysicalDecoder` and encode/decode adapters.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import torch
from icgs.models.layers.method import masked_mean
x = torch.tensor([[[2.], [6.], [999.]]])
m = torch.tensor([[True, True, False]])
torch.testing.assert_close(masked_mean(x, m, dim=1), torch.tensor([[4.]]))
with self.assertRaises(ValueError):
    masked_mean(x, torch.zeros_like(m), dim=1)
from icgs.models.encoders.physical import PhysicalEncoder
points = torch.zeros(1, 2048, 3)
valid = torch.zeros(1, 2048, dtype=torch.bool); valid[:, 0] = True
encoded = PhysicalEncoder()(points, valid)
self.assertTrue(torch.isfinite(encoded.X).all().item())
self.assertTrue((encoded.X[~encoded.anchor_valid] == 0).all().item())
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_physical_geometry.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
local = mlp_6_64_128_256(torch.cat((neighbors - anchors, anchors_expanded), -1))
X = local.masked_fill(~neighbor_valid[..., None], -torch.inf).amax(-2)
X = torch.where(anchor_valid[..., None], X, torch.zeros_like(X))
X = geometry_block_2(geometry_block_1(X, anchors, anchor_valid), anchors, anchor_valid)
patches = anchors[..., None, :] + 0.10 * torch.tanh(decoder(torch.cat((ln_X, grid16), -1)))
```

Use width256,8 heads,FFN1024,pre-LN epsilon1e-5,dropout0; geometric attention bias MLP3→32→8. Decoder MLP258→256→128→3 uses fixed4x4 grid in[-1,1]^2, removes invalid patches and returns world cloud/mask. Test invalid padding cannot alter output and gradients remain finite.

Zero invalid anchor rows BEFORE any LayerNorm/attention, not only after blocks:
an all-invalid padded neighborhood otherwise creates -inf at max aggregation.
Keep invalid keys masked and re-zero invalid query rows after each block. Reject
the fully empty sample instead of treating every anchor as zero valid geometry.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_physical_geometry.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 3: Measured-root and decoded-imagination bridge

**Files:** Create `src/icgs/execution/observation_bridge.py`.
**Test owner:** `tests/test_physical_geometry.py`.
**Consumes / produces:** Produces `bridge_source(origin)` and `imagined_observation(state)`; consumes cached decoded world cloud and achieved pose/grip.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.execution.observation_bridge import bridge_source
self.assertEqual(bridge_source('real'), 'measured')
self.assertEqual(bridge_source('imagined'), 'cached-decoded-world')
with self.assertRaises(ValueError):
    bridge_source('oracle')
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_physical_geometry.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
if state.origin != 'imagined':
    raise ValueError('imagined bridge requires imagined state')
return Observation(points=state.cached_world_cloud,
                   T_w_e=state.T_w_e[0], grip=state.grip[0].item())
```

Native policy already performs world→end-effector conversion; adapter must not pre-transform again. Never decode twice after re-encoding. Pair measured/reconstructed real queries under identical diffusion seeds and report CD plus action translation/rotation and locally executed success; reconstruction CD alone is not bridge acceptance.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_physical_geometry.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

## Acceptance, resource limits and evidence

L1 covers shapes/masks/SE(3)/gradient and exactly-once frame conversion. FG bridge needs actual frozen published inference on validation observations, paired action discrepancies and capped execution-fidelity pilot; owner approves tolerances before measuring. L2 required for native action fidelity, L3 for execution fidelity.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_physical_geometry.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

No modification of native codec/transforms, point sampler, IP scene encoder or published graph. Isolate geometry artifact/profile on drift; keep measured-root inference available. Disocclusion and sparse foreground are explicit limitations.

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
- Remaining risk: Reconstruction can be geometrically close yet change IP actions; no policy fidelity has been measured.
