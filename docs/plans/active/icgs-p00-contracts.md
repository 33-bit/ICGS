# P00: Method contracts, configuration and provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Define separate validated method records/configuration without changing native inference semantics.

**Architecture:** Keep lightweight capability records in contracts and frozen configuration in configuration. Neural state ownership stays in state; only outer composition/artifact code resolves or loads component weights.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Contracts](../../method/contracts.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — P00 runtime boundary implemented; downstream target runtime remains planned**.
This document is a category C component of the approved docs-only research migration;
the completed P00 surface does not authorize downstream workloads.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: Approved target boundaries and roadmap; no learned weights or simulator required.
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

- Create: `src/icgs/contracts/method.py` — timed records, capabilities and output validation.
- Create: `src/icgs/configuration/method.py` — frozen MethodConfig and strict JSON conversion.
- Create: `src/icgs/artifacts/method.py` — canonical reference fingerprint and manifest checks.
- Test: `tests/test_method_contracts.py`.

Public capability boundary (planned; not currently importable):

```python
TimedCommand(target_w, grip: int, duration_s: float)
CommandPrefix(commands, proposal_root, raw_candidate_id)
MethodConfig.from_dict(payload: dict) -> MethodConfig
reference_fingerprint(payload: dict) -> str
```

Copy numeric inputs into owned read-only backing; reject bool-as-integer counters, invalid SE(3), empty commands, nonfinite numbers, unknown schema keys and unsupported horizons. Record the exact contracts.md field inventory; records never carry program IDs, snapshot handles or scores.

### Task 1: Owned timed records and explicit capability surfaces

**Files:** Create `src/icgs/contracts/method.py`.
**Test owner:** `tests/test_method_contracts.py`.
**Consumes / produces:** Consumes native `Observation`; produces `TimedCommand`, `TimedObservation`, `ExecutedTransition`, `CommandPrefix` and the normative protocol signatures.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
import numpy as np
from icgs.contracts.method import TimedCommand
pose = np.eye(4)
cmd = TimedCommand(pose, 1, 0.1)
pose[0, 3] = 7
self.assertEqual(cmd.target_w[0, 3], 0)
self.assertFalse(cmd.target_w.flags.writeable)
with self.assertRaises(ValueError):
    TimedCommand(np.eye(4), 1, float('nan'))
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_contracts.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
owned = np.array(target_w, dtype=np.float64, copy=True)
# Validate shape, finite entries, homogeneous row, SO(3), grip and duration.
backing = np.frombuffer(owned.tobytes(), dtype=owned.dtype).reshape(4, 4)
object.__setattr__(self, 'target_w', backing)
```

Define all planned output records in the shared inventory, using TYPE_CHECKING for state types. Validate terminal probability sums and finite PlanningResult returns; use protocols, not concrete policy imports. Test zero duration, negative boundary, empty prefixes and mutation through caller aliases.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_contracts.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. Focused commit: **NOT RUN** per the explicit no-commit instruction.

### Task 2: Frozen config validates before tensor IO

**Files:** Create `src/icgs/configuration/method.py`.
**Test owner:** `tests/test_method_contracts.py`.
**Consumes / produces:** Produces `MethodConfig.from_dict(payload)` and `MethodConfig.validate()`; consumes immutable native profile ID.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.configuration.method import MethodConfig
with self.assertRaisesRegex(ValueError, 'unknown'):
    MethodConfig.from_dict({'surprise': 1})
with self.assertRaisesRegex(ValueError, 'horizon|commit'):
    MethodConfig.from_dict({'planning': {'h': 2, 'r': 8}})
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_contracts.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
allowed = set(field.name for field in dataclasses.fields(section_type))
unknown = set(payload) - allowed
if unknown:
    raise ValueError(f'unknown config fields: {sorted(unknown)}')
if not 1 <= planning.r <= planning.h <= native.pred_horizon:
    raise ValueError('commit/edge horizon is incompatible')
```

Implement geometry/event/memory/dynamics/evaluator/planning/control/dataset/training sections explicitly. Enforce width 256, 128 anchors, three fixed heads and trained horizon coverage; keep native P=8 versus demo waypoints10 distinct. Preserve defaults < explicit JSON < explicit CLI, resolve paths relative to JSON and reject executable serialization.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_contracts.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. Focused commit: **NOT RUN** per the explicit no-commit instruction.

### Task 3: Reference and deployment artifact lineage

**Files:** Create `src/icgs/artifacts/method.py`.
**Test owner:** `tests/test_method_contracts.py`.
**Consumes / produces:** Produces `reference_fingerprint(payload)` and `validate_method_manifest(manifest, config)`; no load inside neural modules.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.artifacts.method import reference_fingerprint
with self.assertRaisesRegex(ValueError, 'reference'):
    reference_fingerprint({})
with self.assertRaisesRegex(ValueError, 'reference'):
    reference_fingerprint({'ip_checksum': 'not-a-sha256'})
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_method_contracts.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [x] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
# Select exactly the reference fields specified in contracts.md.
canonical = json.dumps(reference_fields, sort_keys=True, separators=(',', ':'),
                       allow_nan=False).encode('utf-8')
reference_id = hashlib.sha256(canonical).hexdigest()
```

Require IP hash/profile, geometry and physical/event/task weights, segmentation/router, preprocessing, calibration/camera/gravity/workspace, cadence/r and RNG IDs. Exclude dynamics, evaluator and learned stopping from the reference hash; record them separately. Test reordering invariance, one-field mutation sensitivity and rejected evaluator/reference mismatch.

- [x] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_method_contracts.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [x] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. Focused commit: **NOT RUN** per the explicit no-commit instruction.

## Acceptance, resource limits and evidence

L1 acceptance requires full invalid-field/ownership matrix and an artifact-loader spy proving rejection occurs before any tensor read. No empirical gate is satisfied by config construction.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_method_contracts.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

Do not enlarge native ExperimentConfig, native Candidate, NPZ schema or checkpoint key vocabulary. No plugin discovery, Hydra, executable configuration or speculative runtime package moves.

On failure, retain the failed evidence and isolate the new component/config ID.
Do not overwrite native checkpoints, relabel collected outcomes or drop difficult samples.
Resume only from a compatible explicit artifact/manifest; changed reference lineage
requires new outcome labels, not a cache workaround. Return to the preceding accepted
phase if an FG fails and request a scoped protocol decision.

## Execution evidence

- Repository state: implementation was run in `/Users/33bit/AI/Research/VLA/ICGS` on
  `main`; no commit, stage, reset, checkout, or out-of-scope repository edit was
  made. The only changed repository paths are the three created runtime modules,
  `tests/test_method_contracts.py`, and this factual evidence/checklist update.
- Interpreter: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`,
  Python 3.14.4. This is outside the declared supported `>=3.10,<3.13` range and
  `icgs` is not installed.
- Audit repair RED evidence: the focused diagnostic command
  `PYTHONPATH=src python3 -B -m unittest discover -s tests -p
  'test_method_contracts.py' -v` meaningfully failed for the forbidden concrete
  records, missing outer loader seam, missing lineage checks, missing config file
  overlay/IDs, and invalid horizon/type checks before each corresponding fix.
- Focused diagnostic PASS: the same command from the repository root ran 15
  selected tests, 15 passed, 0 failures/errors/skips. It covers the contract-only
  boundary, genuine post-validation loader callback (zero calls on rejected
  evaluator/reference lineage), component IDs, strict JSON file-relative paths,
  defaults < JSON < explicit dotted overrides, bool/type rejection, artifact
  lineage, and sensor/temperature lineage IDs. This is diagnostic execution with
  `PYTHONPATH=src`, not supported installed L1 acceptance.
- L0 PASS: `python3 -B scripts/validate_fast.py` from the repository root —
  Python syntax, local links, static boundary and 19 harness self-tests passed;
  command also reported higher tiers NOT RUN. L0 PASS: `python3 -B -S
  scripts/validate_fast.py` from the repository root — same 19/19 harness
  self-tests passed.
- Supported installed L1: **NOT RUN** — no supported ICGS environment is
  available (Python 3.14.4 is outside the supported range and `icgs` is not
  installed). No local import failure is counted as RED or L1 PASS.
  L2/C1–C5, L3/L4, model/artifact execution, downloads, preprocessing,
  simulator workloads, training and robot motion are **NOT RUN**.
- Remaining risk: supported-environment imports and real tensor/checkpoint
  compatibility remain unverified; no empirical feasibility or C1–C5 gate is
  claimed. The active roadmap gates remain unchanged.
