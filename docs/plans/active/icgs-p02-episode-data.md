# P02: Executed episode repository and data views Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Store causal executed episodes, independent contexts and split-safe supervision in a versioned repository.

**Architecture:** Create method-specific numeric archives plus JSON manifests without touching native NPZ/PyG. Annotation and replay metadata remain outside online records; datasets form views over unique transition IDs.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Data and training](../../method/data-training.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — planned runtime NOT IMPLEMENTED**. This document is a category C
component of the approved docs-only research migration, not permission to execute it.
The [master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00–P01 contracts; FG assets/calibration and split manifest approval before collection.
Read [workflow](../../WORKFLOW.md), [architecture](../../ARCHITECTURE.md),
[baseline](../../baselines/instant_policy.md), [native contract](../../components/policy-data-contract.md),
[research policy](../../RESEARCH.md) and [validation guide](../../../tests/README.md)
before runtime execution. Inspect Git state; preserve unrelated work and user assets.

## Configuration consumption addendum — 2026-09-09

Default values now belong to the [packaged primary JSON](../../../src/icgs/configuration/profiles/icgs_primary.json),
with key/unit/restriction details in the [parameter reference](../../method/parameters.md).
Consumed sections for this plan: **dataset, collection, benchmark, sensors**.

Use dataset split/mix/perturbation settings and the chosen collection.pilot/primary budget. Task monitors use benchmark tolerances only after predicate-protocol approval. No literal quotas or scalar thresholds in new generator/annotation code.

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

- Create: `src/icgs/data/schemas/episodes.py` — method episode schema and validation.
- Create: `src/icgs/data/datasets/episodes.py` — causal views, splits and masks.
- Create: `src/icgs/data/collection/attempts.py` — bounded executed generation bookkeeping.
- Create: `src/icgs/data/collection/programs.py` — exact program catalog and split manifest.
- Create: `src/icgs/data/collection/annotations.py` — event/primitive overlap and causal label masks.
- Create: `src/icgs/evaluation/dependencies.py` — sole external TaskMonitor/predicate implementation, also consumed by P12.
- Test: `tests/test_episode_data.py`.

Public capability boundary (planned; not currently importable):

```python
validate_episode(record: dict) -> None
validate_split_lineage(rows: list[dict]) -> None
causal_prefix(episode, boundary: int) -> tuple
build_view(manifest, view: str) -> EpisodeView
```

Persist episode/source/asset/program/split IDs, commands, before/after observations, calibration, actual timing, interventions, causal pointers, replay reports and separate annotation masks. Never serialize pickle/object arrays; versioned JSON plus numeric arrays are the implementation default.

### Task 1: Separate storage from online capabilities

**Files:** Create `src/icgs/data/schemas/episodes.py`.
**Test owner:** `tests/test_episode_data.py`.
**Consumes / produces:** Produces `validate_episode` and `validate_online_fields(mapping)`; consumes P00 records.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.data.schemas.episodes import validate_online_fields
validate_online_fields({'points': [], 'T_w_e': [], 'grip': 0})
with self.assertRaisesRegex(ValueError, 'privileged'):
    validate_online_fields({'points': [], 'program_id': 'T01'})
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_episode_data.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
online = {'points', 'T_w_e', 'grip', 'point_valid'}
for key in mapping:
    if key not in online:
        raise ValueError(f'privileged or unknown online field: {key}')
```

Validate shapes/dtypes/finite values separately from field allowlisting. Store variable clouds via offsets and validity, calibration IDs per episode, observation origin and schema version. Verify transition boundary adjacency, nonnegative durations and raw/materialized provenance; reject unknown or incomplete schema.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_episode_data.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 2: Split by ancestry and expose only causal history

**Files:** Create `src/icgs/data/datasets/episodes.py`.
**Test owner:** `tests/test_episode_data.py`.
**Consumes / produces:** Produces `validate_split_lineage`, `causal_prefix`, and geom/dyn/task/terminal/value/pair/calib/audit views.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.data.datasets.episodes import validate_split_lineage
with self.assertRaisesRegex(ValueError, 'lineage'):
    validate_split_lineage([
        {'lineage_id': 'seed-a', 'split': 'train'},
        {'lineage_id': 'seed-a', 'split': 'test'}])
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_episode_data.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
splits_by_lineage = {}
for row in rows:
    previous = splits_by_lineage.setdefault(row['lineage_id'], row['split'])
    if previous != row['split']:
        raise ValueError('source lineage crosses splits')
```

Keep mesh-family and augmentation descendants together; lock 20 train/4 development/12 test skeletons before generation. Split development selection/calibration by lineage. Query episode must differ from every context episode; horizons share trial-group IDs. Count overlapping views as unique transitions, not new data.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_episode_data.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

### Task 3: Retain failed attempts and intervention provenance

**Files:** Create `src/icgs/data/collection/attempts.py`, `programs.py`, `annotations.py` and `src/icgs/evaluation/dependencies.py`.
**Test owner:** `tests/test_episode_data.py`.
**Consumes / produces:** Produces `attempt_counts(records)` and bounded attempt collector consuming TimedEnvironment and separate TaskMonitor.

- [ ] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.data.collection.attempts import attempt_counts
counts = attempt_counts([{'status': 'success'}, {'status': 'timeout'},
                         {'status': 'invalid-input'}])
self.assertEqual(counts['attempts'], 3)
self.assertEqual(counts['successes'], 1)
self.assertEqual(counts['invalid'], 1)
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_episode_data.py' -v`.
  Expect the new test to fail because its new implementation is absent or violates
  the stated assertion; record that failure. An unrelated import failure is not RED proof.
- [ ] **Step 3 — GREEN:** Implement the boundary using this algorithm/code sketch.

```python
counts = {'attempts': len(records), 'successes': 0, 'invalid': 0}
for row in records:
    counts['successes'] += int(row['status'] == 'success')
    counts['invalid'] += int(row['status'] == 'invalid-input')
return counts
```

Run object-frame waypoint adaptations and free-space connectors only through physics; retain valid failed attempts, pauses/retries and bounded perturbations with intervention IDs. Warm-up mixture70/30 attempts/perturbed; frozen-reference mixture50/30/20 attempts/reference/recovery. Preserve generator limits, source meshes and empirical acceptance rates.

Transcribe the archived proposal's T01–T20, V01–V04 and P/G/R held-out skeletons
into an explicit program catalog; no filename inference. Pilot selects
T06/T08/T09/T11/T13/T14. Bind actual assets/ranges/seeds only after G2 approval.
Implement `map_event_annotation(segment, primitive_intervals)` by greatest temporal
overlap with >=50% coverage, ties to earlier occurrence, otherwise mask. Compute
rho from observed completion history, nu from current predicates and eligibility
from current prerequisites; free-space nu is invalid. Align uniformly across
corresponding demo events, with null for unrepresented recovery. Use the external
TaskMonitor capability implemented here and consumed by P12, never an online
program-ID tensor. Implement `first_terminal(success, failure, absorbed_success)`:
absorbed success wins, otherwise failure wins a simultaneous first event, else
success or continue. Implement the exact predicate/stability defaults in
data-training.md; unavailable force/penetration measurements block G2.

- [ ] **Step 4 — Verify GREEN:** Repeat `python3 -B -m unittest discover -s tests -p 'test_episode_data.py' -v`.
  Expect every selected assertion to execute and pass; record selected/executed/skipped counts.
- [ ] **Step 5 — Review:** Inspect the exact source/test diff and update this plan's
  evidence. At authorized execution time, make a focused commit only after that review.

## Acceptance, resource limits and evidence

L1 roundtrips tiny numeric fixtures with allow_pickle=False and checks leakage, masks and unique counts. FG assets/calibration requires concrete versioned assets/cameras/workspace, expert feasibility and split certification before six-program pilot collection; quotas are ceilings/targets, not permission.

- L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`; syntax/links/boundaries only.
- L1: `python3 -B -m unittest discover -s tests -p 'test_episode_data.py' -v` in an installed supported environment.
  Tiny deterministic fixtures only; no data loader workers, networking or simulator startup.
- Preserve native regression coverage and strict published C1–C5 in the final
  integration acceptance; this component's synthetic tests do not replace those gates.
- Any authorized L2/L3/L4 job must record immutable inputs, command, interpreter,
  hardware, seeds, limits and its named FG report before downstream work proceeds.
- Stop a pilot at its approved interval/episode/wall/disk caps; never turn a failed
  feasibility gate into a broader automatic run. SKIPPED is never PASS.

## Compatibility, exclusions and recovery

No reuse of kinematic IP pseudo-demo transforms as dynamics truth. Native NPZ/PyG loaders remain untouched. Reject corrupt records to an auditable quarantine manifest without deletion or relabeling.

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
- Remaining risk: Asset availability, sensor fidelity, controllable generation and storage cost need a capped pilot.

## Observability integration addendum — 2026-09-09

No P02 episode collector or replay producer exists, so this implementation emits
no fabricated lineage, rejection, censoring or terminal records. Offline run
inspection can link current outer-boundary records and explicit capture IDs, but it
does not replace P02 schema validation, ancestry splits or executed evidence.
