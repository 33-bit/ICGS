# P03: Physical geometry encoder, decoder and native bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Represent full foreground geometry and reconstruct world clouds for imagined native-policy queries.

**Architecture:** Add independent 5-mm/FPS preprocessing, metric SE(3) helpers and width256 geometry modules. Native live preprocessing and graph/sampling math remain untouched; bridges pass world clouds exactly once.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Neural method](../../method/neural.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — P03 Tasks 1–3 implemented; supported validation and FG gates pending**.
This document records the bounded runtime component of the approved research migration;
later state/lineage integration remains owned by P04.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00/P02, fixed sensor/crop profile and native checkpoint availability for later FG bridge validation.
Read [workflow](../../WORKFLOW.md), [architecture](../../ARCHITECTURE.md),
[baseline](../../baselines/instant_policy.md), [native contract](../../components/policy-data-contract.md),
[research policy](../../RESEARCH.md) and [validation guide](../../../tests/README.md)
before runtime execution. Inspect Git state; preserve unrelated work and user assets.

## Configuration consumption addendum — 2026-09-09

Default values now belong to the [packaged primary JSON](../../../src/icgs/configuration/profiles/icgs_primary.json),
with key/unit/restriction details in the [parameter reference](../../method/parameters.md).
Consumed sections for this plan: **geometry, neural, decoder, sensors, numerics**.

Wire the typed geometry/preprocessing and block/decoder sections when extending existing constructors. Preserve fixed shape locks and original default initialization order; tests must show a supported nondefault scalar reaches its consumer. Existing fixed constructors are not retroactively controlled by merely loading MethodConfig.

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

- Repository state: correction was performed in the shared checkout
  `/Users/33bit/AI/Research/VLA/ICGS` on `main`; no worktree, subagent,
  commit, stage, reset, checkout, download, training, preprocessing job,
  simulator, or robot action was used. P00 paths remain byte-for-byte
  unchanged. Correction-round files are the existing P03 runtime modules and
  `tests/test_physical_geometry.py`; the justified lazy preprocessing
  initializer remains unchanged.
- Interpreter/environment: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`,
  Python 3.14.4, repository-root cwd. This is outside the declared supported
  `>=3.10,<3.13` range and `icgs` is not installed.
- Correction RED: `PYTHONPATH=src python3 -B -m unittest discover -s tests
  -p 'test_physical_geometry.py' -v` ran 13 selected tests with 5 existing
  behaviors passing and 8 expected failures: same-voxel/config-identity
  subcases, bridge batch/padding behavior, direct decoder empty-input
  rejection, and non-XYZ input rejection. No production correction was made
  before this RED run.
- Prior correction GREEN: the source-path command ran 13 selected/executed
  tests, 13 passed, 0 failures, 0 errors, 0 skips. It covers same-voxel
  averaging, fixed 5-mm/2048/128/32/ell0 identity, XYZ/empty-input rejection,
  direct decoder empty/nonfinite rejection, bridge mask/sentinel/B=1 behavior,
  finite backward flow through encoder/attention/decoder, and the original
  P03 geometry/bridge checks. This is diagnostic execution outside the
  supported installed environment, not a supported L1 acceptance claim.
- Final bridge correction RED: the same source-path command ran 16 selected
  tests with 13 existing tests passing and 3 expected failures for scaled/
  nonorthogonal rotation, malformed homogeneous bottom row, and grip=2.0.
  No bridge validation correction was made before this RED run.
- Intermediate GREEN diagnostic: the first guard implementation produced 5
  errors because `torch.allclose` returned a Python bool and the guard called
  `.item()`; removing those two calls was the minimal bridge-only correction.
- Final bridge correction GREEN / focused source-path diagnostic: the command
  ran 16 selected/executed tests, 16 passed, 0 failures, 0 errors, 0 skips.
  It preserves the valid bridge case while covering the three new rejection
  regressions. This remains diagnostic execution outside the supported installed
  environment, not a supported L1 acceptance claim.
- L0 PASS: `python3 -B scripts/validate_fast.py` from the repository root —
  syntax, local links, static harness boundary, and 19/19 harness self-tests
  passed; L1/L2/L3/L4 were not run by this command.
- L0 PASS: `python3 -B -S scripts/validate_fast.py` from the repository root —
  the same 19/19 harness self-tests passed.
- Supported installed L1 **SKIPPED**: the exact plan command
  `python3 -B -m unittest discover -s tests -p 'test_physical_geometry.py' -v`
  was attempted from the repository root after the bridge correction and ran
  16 selected tests, all 16 ending in `ModuleNotFoundError: No module named 'icgs'`.
  No supported installed environment is available; this is not PASS.
- L2/C1–C5, L3/L4, collection, training and benchmark validation: **NOT RUN**
  by explicit task authorization. Native codec, transforms, graph, sampler,
  published profile, and native preprocessing behavior were not modified.
- Remaining risk: P04 still owns PhysicalState/lineage enforcement; no cache
  lineage field or alternate state schema was added. Supported-environment
  import/tensor compatibility, checkpoint interaction, paired action drift,
  local success, and broader feasibility evidence remain unverified. The
  crop profile is still explicit but optional because no calibrated numeric
  workspace was supplied; GPU floating-point tie ordering is not claimed
  bitwise identical.
