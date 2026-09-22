# P02: Executed episode repository and data views Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Agent delegation requires the execution session's authority.

**Goal:** Store causal executed episodes, independent contexts and split-safe supervision in a versioned repository.

**Architecture:** Create method-specific numeric archives plus JSON manifests without touching native NPZ/PyG. Annotation and replay metadata remain outside online records; datasets form views over unique transition IDs.

**Tech Stack:** Python >=3.10,<3.13, installed ICGS, NumPy, PyTorch and the repository's pinned runtime dependencies. Flat standard-library unittest discovery; no new framework.

**Spec:** [Shared contracts](../../method/contracts.md), [Data and training](../../method/data-training.md), [method index](../../method/README.md).

## Status, authority and prerequisites

Status: **ACTIVE — initial geom/dyn software slice implemented: in-memory
contracts, bounded attempts, versioned archives, lineage validation, initial
views/adapters and bounded Python runner. Initial software slice accepted after
manager verification and scoped Sol audit;
physical simulator collection and later task/outcome views remain incomplete**.
This document is a category C component of the approved research migration; its
gates still do not authorize simulator collection. The
[master roadmap](../../plans/active/icgs-method-implementation.md) owns phase
ordering, feasibility gates and workload authorization.

Prerequisites: P00–P01 contracts; [ADR0014](../../decisions/0014-canonical-generation.md)
authorizes G2 certification work, but measured assets/calibration and split
manifest approval remain required before collection.
Read [workflow](../../WORKFLOW.md), [architecture](../../ARCHITECTURE.md),
[baseline](../../baselines/instant_policy.md), [native contract](../../components/policy-data-contract.md),
[research policy](../../RESEARCH.md) and [validation guide](../../../tests/README.md)
before runtime execution. Inspect Git state; preserve unrelated work and user assets.

## Configuration consumption addendum — 2026-09-09

Default values now belong to the [packaged primary JSON](../../../src/icgs/configuration/profiles/method.json),
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

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.data.schemas.episodes import validate_online_fields
validate_online_fields({'points': [], 'T_w_e': [], 'grip': 0})
with self.assertRaisesRegex(ValueError, 'privileged'):
    validate_online_fields({'points': [], 'program_id': 'T01'})
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_episode_data.py' -v`.
  The original RED command was not run in a supported installed environment, so
  it remains unchecked. A later supported-Python regression RED for duplicated
  online/transition observations is recorded below.
- [ ] **Step 3 — GREEN (partial):** Implement the boundary using this algorithm/code sketch.

```python
online = {'points', 'T_w_e', 'grip', 'point_valid'}
for key in mapping:
    if key not in online:
        raise ValueError(f'privileged or unknown online field: {key}')
```

Validation now covers field allowlisting, shapes/dtypes/finite values, metadata
separation, transition adjacency/timing, and consistency between the materialized
online observation and its P00 `ExecutedTransition` observation. Consecutive
transitions also require exact equality of simulator timestamp, measured wall
timestamp, and sensor-profile ID at their shared boundary. It does **not**
implement JSON manifest/NPZ shard storage, ragged cloud offsets, checksums, atomic
publication, or quarantine. The approved documents do not define the necessary
manifest/shard key inventory or checksum/offset contract, so that storage work
remains blocked rather than being invented here.

- [x] **Step 4 — Verify GREEN (validation boundary only):** The supported installed
  Python 3.12 targeted suite executed the current Task1/2 assertions with no skips.
  This does not validate the unimplemented archive roundtrip acceptance.
- [x] **Step 5 — Review:** Inspected the new schema and test files plus
  `git diff --check`; no native source, loader, or P01 path was changed. No commit
  was made.

Task 1 status: **PARTIAL** — the in-memory validation boundary is implemented;
versioned manifest/NPZ persistence and ragged-offset roundtrips require an approved
concrete storage protocol.

### Task 2: Split by ancestry and expose only causal history

**Files:** Create `src/icgs/data/datasets/episodes.py`.
**Test owner:** `tests/test_episode_data.py`.
**Consumes / produces:** Produces `validate_split_lineage`, `causal_prefix`, and geom/dyn/task/terminal/value/pair/calib/audit views.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.data.datasets.episodes import validate_split_lineage
with self.assertRaisesRegex(ValueError, 'lineage'):
    validate_split_lineage([
        {'lineage_id': 'seed-a', 'split': 'train'},
        {'lineage_id': 'seed-a', 'split': 'test'}])
```

- [ ] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_episode_data.py' -v`.
  The original RED command was not run in a supported installed environment, so
  it remains unchecked.
- [ ] **Step 3 — GREEN (partial):** Implement the boundary using this algorithm/code sketch.

```python
splits_by_lineage = {}
for row in rows:
    previous = splits_by_lineage.setdefault(row['lineage_id'], row['split'])
    if previous != row['split']:
        raise ValueError('source lineage crosses splits')
```

`validate_split_lineage` implements the explicit literal-`lineage_id` rule and
`causal_prefix` implements boundary-limited transition history. The declared
`build_view(manifest, view) -> EpisodeView` and the geom/dyn/task/terminal/value/
pair/calib/audit selections are **not** implemented: no approved manifest payload,
`EpisodeView` type, transition-ID field, context/query selection representation,
or view-specific mask/unique-count contract exists. Query/context independence and
overlap accounting therefore cannot be enforced without inventing an API.

No existing source/document contract defines `root_lineage_id`, parent lineage, or
ancestry closure. Descendant-aware mesh-family and augmentation split validation is
blocked on that manifest/lineage protocol decision; literal duplicate lineage IDs
remain guarded.

- [x] **Step 4 — Verify GREEN:** Implemented `validate_dataset_manifest`,
  lineage closure/acyclicity, `build_view` for `geom` and `dyn` views, and
  `adapter_a0` / `adapter_a1` consumer adapters per ADR0011. Tested with 6
  deterministic tests in `tests/test_episode_views.py`.
- [x] **Step 5 — Review:** Inspected the dataset helper and deterministic tests;
  no simulator, collection, native data loader, or P01 source was changed.

Task 2 status: **RESOLVED FOR INITIAL GEOM/DYN VIEWS** (ADR0011) — manifest
lineage closure, acyclicity, asset family uniqueness, `geom` and `dyn` views,
and consumer adapters implemented; future task outcome views (P08) deferred.

### Task 3: Retain failed attempts and intervention provenance

**Files:** Create `src/icgs/data/collection/attempts.py`, `programs.py`, `annotations.py` and `src/icgs/evaluation/dependencies.py`.
**Test owner:** `tests/test_episode_data.py`.
**Consumes / produces:** Produces `attempt_counts(records)` and bounded attempt collector consuming TimedEnvironment and separate TaskMonitor.

- [x] **Step 1 — RED:** Add the following assertion body to a named
  `unittest.TestCase` method in the test owner, with the shown imports.

```python
from icgs.data.collection.attempts import attempt_counts
counts = attempt_counts([{'status': 'success'}, {'status': 'timeout'},
                         {'status': 'invalid-input'}])
self.assertEqual(counts['attempts'], 3)
self.assertEqual(counts['successes'], 1)
self.assertEqual(counts['invalid'], 1)
```

- [x] **Step 2 — Verify RED:** Run `python3 -B -m unittest discover -s tests -p 'test_episode_data.py' -v`.
  The supported Python 3.12 run failed with six expected missing-module errors
  after the seven pre-existing assertions passed; no unrelated failure was used
  as RED evidence.
- [x] **Step 3 — GREEN (deterministic foundation only):** Implement the boundary
  using this algorithm/code sketch.

```python
counts = {'attempts': len(records), 'successes': 0, 'invalid': 0}
for row in records:
    counts['successes'] += int(row['status'] == 'success')
    counts['invalid'] += int(row['status'] == 'invalid-input')
return counts
```

Implemented `attempt_counts`, an immutable explicit proposal catalog, temporal
event/primitive mapping, causal rho/nu/eligibility labels with masks, first-terminal
resolution, and an injected external `TaskMonitor` capability. The monitor contains
no predicate tolerances and requires an approved capability/configuration injection.
It does not instantiate a timed environment or write an episode record.

Run object-frame waypoint adaptations and free-space connectors only through physics; retain valid failed attempts, pauses/retries and bounded perturbations with intervention IDs. Warm-up mixture70/30 attempts/perturbed; frozen-reference mixture50/30/20 attempts/reference/recovery. Preserve generator limits, source meshes and empirical acceptance rates.

Transcribe the archived proposal's T01–T20, V01–V04 and P/G/R held-out skeletons
into an explicit program catalog; no filename inference. Pilot selects
T06/T08/T09/T11/T13/T14. Bind actual assets/ranges/seeds only after G2
certification evidence and the resulting manifest approval; owner authorization
alone is not a physical G2 PASS.
Implement `map_event_annotation(segment, primitive_intervals)` by greatest temporal
overlap with >=50% coverage, ties to earlier occurrence, otherwise mask. Compute
rho from observed completion history, nu from current predicates and eligibility
from current prerequisites; free-space nu is invalid. Align uniformly across
corresponding demo events, with null for unrepresented recovery. Use the external
TaskMonitor capability implemented here and consumed by P12, never an online
program-ID tensor. Implement `first_terminal(success, failure, absorbed_success)`:
absorbed success wins, otherwise failure wins a simultaneous first event, else
success or continue. Concrete predicate/stability defaults must be supplied by
the injected resolved configuration only after predicate-protocol approval;
unavailable force/penetration measurements block G2 rather than receiving a
local substitute threshold.

- [x] **Step 4 — Verify GREEN:** The supported Python 3.12 targeted suite passed:
  13 executed, 13 passed, 0 failures/errors/skips. It covers synthetic fixtures
  only and does not certify simulator collection or G2 bindings.
- [x] **Step 5 — Review:** Inspected the exact source/test diff and `git diff
  --check`. No native IP, P01 execution/replay, simulator, or training path was
  changed. No commit was made.

Task 3 status: **PARTIAL** — deterministic accounting/catalog/annotation/monitor
foundations and the in-memory bounded attempt collector (`collect_attempt`) are
implemented. Concrete simulator execution, concrete assets/seeds/tolerances,
dataset collection workloads, and G1/G2 physical feasibility gates remain
deferred/unauthorized.

### Executable program-binding preflight — 2026-09-18

`icgs.data.collection.bindings.load_binding_manifest` now provides the minimal
fail-closed preflight for executable program specifications. Each binding must
declare a catalog-matching scene, asset family/version, source-lineage root,
workspace bounds, translation/yaw/scale/camera/lighting randomization, unique
seed IDs, expert and waypoint protocols, controller/predicate protocol IDs,
predicate tolerances and calibration identity. The loader checks catalog split
agreement and rejects asset-family or source-lineage leakage before a caller
constructs a simulator.

This is metadata enforcement, not physical certification: scene files, expert
executions, calibration measurements and tolerance sensitivity still need to be
supplied and recorded for G2.

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

- Historical scope audit (pre-P01 implementation): P00 timed records and protocols
  existed while P01's runtime stepping and materializer boundaries were still in
  development. Task 3 implemented deterministic accounting, catalog, annotation,
  and external-monitor boundaries without simulator collection.
- Current P01 dependency status (2026-09-15): Runtime stepping and materialization
  seams are implemented: `src/icgs/execution/timed.py` provides `materialize_prefix`
  and grip grouping; `src/icgs/environments/rlbench/timed.py` provides the injected
  `TimedRLBenchAdapter` reset/advance/close stepping boundary; and
  `src/icgs/environments/rlbench/replay.py` provides replay report validation.
  Engine restoration (P08), empirical G1/G2 physical certification, and
  durable/physical collector integration remain incomplete.
- Historical correctness RED: supported installed Python 3.12 ran the targeted suite after
  mismatch tests were added and before correspondence validation. **FAIL**, 6
  executed, 1 failure, 0 skips: a changed online point cloud was accepted despite
  disagreeing with its transition observation.
- Historical correctness GREEN: supported installed Python 3.12 ran the targeted suite after
  the online/transition fix. **PASS**, 6 executed, 6 passed, 0 failures/errors/skips.
  This validates
  only the in-memory helper boundary, not archive storage or view construction.
- Historical shared-boundary metadata RED: supported installed Python 3.12 ran the targeted
  suite after adding simulator-time, wall-time, and sensor-profile mismatches and
  before their validator. **FAIL**, 7 executed, 3 failures, 0 skips; all three
  inconsistent shared boundaries were accepted.
- Historical shared-boundary metadata GREEN: the same supported suite after the exact metadata
  checks. **PASS**, 7 executed, 7 passed, 0 failures/errors/skips.
- Historical supported targeted command: `PATH=/home/hunganh/miniconda3/envs/a0_py312/bin:$PATH
  python3 -B -m unittest discover -s tests -p 'test_episode_data.py' -v` — **PASS**,
  13 executed, 13 passed, 0 failures/errors/skips. The host-default `python3` was
  Python 3.13.12 and unsupported; no `PYTHONPATH` diagnostic is counted as
  acceptance evidence. Task 3 RED in the same environment was **FAIL**, 13
  executed with 6 expected missing-module errors and 0 skips before its source
  files existed.
- Historical Task 3 label/status refinement used the repository `.venv/bin/python`, CPython
  3.10.20, after an editable no-dependency install of the current checkout. RED
  was **FAIL**, 13 executed with exactly 2 assertion failures: noncanonical padded
  status was accepted and unidentifiable postcondition masked an independently
  observable eligibility label. GREEN was **PASS**, 13 executed, 13 passed, 0
  failures/errors/skips. The same environment passed `test_architecture.py` 3/3
  and `compileall` for the new collection/evaluation modules.
- Historical Linux L0: `PATH=/home/hunganh/miniconda3/envs/a0_py312/bin:$PATH python3 -B
  scripts/validate_fast.py` and `python3 -B -S scripts/validate_fast.py` — **FAIL**
  overall. Both variants passed Python syntax, static harness boundary, and 19
  harness tests with zero skips. Five pre-existing missing evidence-log links under
  `docs/experiments/vv19-validation` failed local-link validation; repeating both
  commands through `.venv/bin/python` reproduced the exact same failure.
- Manager baseline evidence (2026-09-15, macOS 15.7.2 arm64):
  - `.venv/bin/python -B -m unittest discover -s tests -p 'test_episode_data.py' -v`:
    **PASS**, 13 selected/executed/passed, 0 skips. CPython 3.11.15, NumPy 1.26.4;
    installed `icgs` resolves to this checkout's `src/icgs`.
  - `python3 -B scripts/validate_fast.py`: **PASS with two pre-existing warnings**, syntax/links/boundaries and
    19/19 harness tests, no skips; CPython 3.14.4 (L0 only).
  - `python3 -B -S scripts/validate_fast.py`: **PASS with two pre-existing warnings**, same checks/counts;
    CPython 3.14.4 (L0 only).
  - Both L0 variants emit unrelated invalid-escape `SyntaxWarning`s in
    `tests/test_task_router.py` (pre-existing; no change authorized there).
  - Full L1 / L2 C1–C5 / L3 / L4: **NOT RUN**; preflight uses only bounded synthetic
    P02 tests and stdlib L0. No simulator, training, downloads, preprocessing,
    GPU or robot workloads.
- Task 3 in-memory bounded attempt collector: `src/icgs/data/collection/attempts.py`
  implements `collect_attempt`, `AttemptResult`, and `AttemptExecutionError`.
  - API: `collect_attempt(environment, commands, *, config: MethodConfig | CollectionConfig, monitor=None, seed=None, clock=time.monotonic) -> AttemptResult`.
  - Semantics: Single reset with caller seed; sequential execution of supplied
    materialized `TimedCommand`s bounded by `config.collection.max_episode_intervals`
    without eager iterator overconsumption; requires explicit typed `MethodConfig` or
    `CollectionConfig`; halts on command exhaustion (`completed`) or resource limits
    (`timeout`); physical terminal classification is omitted and deferred until an
    approved specification exists; inter-step wall-limit evaluation via injected `clock`;
    full causal correspondence verification across boundary index, simulator/wall
    timestamps, sensor profile, grip, points, pose, and submitted vs recorded command;
    consolidated single-path lifecycle management guarantees `environment.close()` is called
    exactly once after ownership begins; operational failures in iterator, clock,
    stepping, or annotation retain executed transitions, detached annotation snapshots,
    and any returned-but-invalid transition as `rejected_transition` on
    `AttemptExecutionError` without suppressing primary cause or cleanup diagnostics;
    `AttemptResult` is a frozen outer record containing detached mutable annotation snapshots;
    `annotations[i]` corresponds to `transitions[i]` for the recorded successful prefix,
    is empty when no monitor is provided, and is shorter than `transitions` if annotation fails.
  - Limitations: Bounded in-memory execution only; no disk/JSON IO; no outer quotas;
    terminal classification is omitted; does not launch simulator or certify G1/G2 physical feasibility.
- Targeted P02 test execution (2026-09-15, macOS 15.7.2 arm64):
  - `.venv/bin/python -B -m unittest discover -s tests -p 'test_episode_data.py' -v`:
    Fix Round 2 regression suite **PASS**, 25 selected, 25 executed, 25 passed,
    0 failures/errors/skips (0.219s). Covered initial clock, iterator creation, reset,
    and stream `next()` failures with exactly-once close, 11 parameterized boundary/timestamp/sensor/observation/command
    mismatches retaining rejected transitions, and annotation prefix / detachment semantics.
    CPython 3.11.15, NumPy 1.26.4; installed `icgs` resolves to this checkout's `src/icgs`.
  - `.venv/bin/python -B -m unittest discover -s tests -p 'test_architecture.py' -v`:
    **PASS**, 3 executed, 3 passed, 0 failures/errors/skips (0.179s).
  - `python3 -B scripts/validate_fast.py` and `python3 -B -S scripts/validate_fast.py`:
    **PASS with two pre-existing warnings**, 19/19 harness tests OK (0.524s, 0.528s).
- Audit and manager validation note (2026-09-15, macOS arm64):
  Gemini 3.8 Flash High implementer, GPT 5.6 Sol Medium reviewer. All five runtime
  findings resolved; documentation and regression follow-through accepted. Manager
  independently verified PASS: 25/25 P02, 3/3 architecture, and both L0 variants 19/19
  zero skips (two existing SyntaxWarnings on host 3.14.4; supported P02 CPython 3.11.15).
  Full L1/C1–C5/L3/L4 NOT RUN; no physical workloads. P02 remains ACTIVE/PARTIAL with
  named persistence, dataset view, lineage closure, and full-runner gaps.
- Remaining protocol blockers:
  1. Archive persistence protocol: RESOLVED by ADR0011 and implemented in Batch A (`src/icgs/data/archives.py`).
  2. Dataset view contract: RESOLVED for geom/dyn views by ADR0011 and implemented in Batch B (`src/icgs/data/datasets/episodes.py`).
  3. Lineage closure: RESOLVED by ADR0011 and implemented in Batch B (`validate_dataset_manifest`).
  4. Durable episode/attempt linkage: RESOLVED by ADR0011 and implemented in Batch A (`persist_attempt`) and Batch C (`run_collection`).
  5. G1/G2 physical feasibility and workload authorization: concrete asset/seed
     catalog bindings, measured force/penetration observability and predicate
     tolerances, and physical simulator execution permissions (still blocked
     under repository research policy; not self-certified).

## Initial-data completion batch — 2026-09-15

Owner delegated concrete contract decisions to the manager after requesting
Gemini 3.8 Flash High implementation followed by GPT 5.6 Sol Medium audit/fix
loops. Category C; [ADR0011](../../decisions/0011-episode-archive-and-initial-views.md)
is the accepted specification for this batch. This resolves storage/lineage/view
design authority, not missing physical measurements. Preserve existing evidence.

Goal: lossless durable attempts, split-safe geom/dyn datasets and a bounded Python
collection entry point. No native IP edits, new dependencies, generic frameworks,
task/outcome views, model training, simulator launches or invented assets/tolerances.

### Batch A: archive and attempt persistence

Files: create `src/icgs/data/archives.py`; extend
`src/icgs/data/collection/attempts.py` only for the small persistence adapter if
needed; add `tests/test_episode_archives.py`. Keep validation-only schema APIs
compatible. Interface: `write_episode_archive(root, record, *, config, metadata)`
returns manifest Path; `read_episode_archive(manifest_path)` returns the original
in-memory record. Separate report metadata has a documented reader.

- [x] RED: test a two-transition ragged roundtrip with online float32 vs measured
  float64 clouds differing within the existing tolerance. Assert both survive
  independently with exact values/dtypes. Test 257-transition shard boundaries,
  one canonical shared observation, digest/offset/key corruption, NaN/object
  rejection, path escape, symlinks, duplicate publication and failed writes.
- [x] Run supported installed `python -B -m unittest discover -s tests -p
  'test_episode_archives.py' -v`; record actual expected assertion failures.
- [x] GREEN: implement atomic lossless numeric archives and strict loader per
  ADR0011; preserve valid failed attempts, partial evidence, zero-transition
  reports, annotations and cleanup errors with no silent success classification.
- [x] Repeat targeted tests and existing `test_episode_data.py`; inspect diff and
  record commands/environment/counts. No commit/staging during shared checkout work.
  Passed 12/12 `test_episode_archives.py` and 25/25 `test_episode_data.py`.

### Batch B: manifest lineage, views and existing consumer handoff

Files: extend `src/icgs/data/datasets/episodes.py`; add
`tests/test_episode_views.py`. Interface: `build_view(dataset_manifest_path, view,
*, split, config)`; finite manifest and sample payloads follow ADR0011. Keep the
old `causal_prefix` and `validate_split_lineage` APIs intact.

- [x] RED: test cycles/dangling parents, shared ancestors/assets across splits,
  unknown program/split mismatch, duplicate episode IDs/digests, and corrupted
  archived inputs. Test unique boundary counts across shards and causal dynamics
  windows from reset. Test a short failure retained for geom but excluded/countable
  for A1 windows lacking configured supervised intervals.
- [x] Run supported installed `python -B -m unittest discover -s tests -p
  'test_episode_views.py' -v`; record RED.
- [x] GREEN: implement manifest validation, finite views and narrow P11
  batch conversion without changing model objectives or existing contracts.
  Current views load episode data into memory; large-dataset lazy loading is not
  an implemented capability or part of this slice's completion claim.
- [x] Repeat tests; prove A0/A1 batches satisfy the actual consumer boundary with
  tiny deterministic seam tests, not a training workload; record the result.
  Passed 6/6 `test_episode_views.py`.

### Batch C: bounded run entry point and operational handoff

Files: create `src/icgs/data/collection/runner.py`, add
`tests/test_collection_runner.py`, update CLI/data component documentation with
the actual Python API and validation example. Do not add fictitious CLI collect.
Interface: `run_collection` composes explicit attempt specs, injected factories,
validated manifests, config/limits and output root; returns a finite run report.
Choose and document concrete parameter names before implementing its first test.

- [x] RED: spy factory proves incomplete protocol/limit/split inputs are rejected
  before environment construction. Test bounded attempts/interval/wall/disk,
  no silent retries, successful/failed/empty/interrupted attempts, shared total
  budget, cleanup, collision-safe restart and run-summary consistency.
- [x] Run supported installed `python -B -m unittest discover -s tests -p
  'test_collection_runner.py' -v`; record RED.
- [x] GREEN: implement the small explicit runner, failure retention and a
  copyable data-generation handoff showing external controller/scene bindings
  as prerequisites, not invented working implementations.
- [x] Repeat targeted tests plus both L0 variants and architecture tests. Report
  real external readiness checks and remaining physical blockers separately.
  Passed 9/9 `test_collection_runner.py`.

### Manager acceptance

- [x] Gemini reports exact diff, commands/counts and gaps; manager inspects scope.
- [x] Sol Medium audits read-only after implementation, independently tests edge
  cases, and returns severity/source evidence plus minimal corrections.
- [x] Gemini fixes confirmed findings; Sol re-audits; manager independently reruns
  focused tests before accepting the batch. Do not expand to stylistic rewrites.
- [x] Preserve full P02 active status and distinguish software slice completion
  from physical generation readiness and mandatory C1–C5 migration acceptance.

## Historical manager verification hold — 2026-09-16

At this checkpoint the Gemini report was **not accepted**, pending correction
of the following manager-reproduced defects. These failures are retained as
historical evidence; see the latest verification checkpoint below for status.

- Environment: repository root, installed editable ICGS in `.venv`, CPython
  3.11.15, macOS arm64. No model training or physical environment was run.
- `.venv/bin/python -B -m unittest discover -s tests -p 'test_episode*.py' -v`
  — PASS, 56/56, zero skips.
- `.venv/bin/python -B -m unittest discover -s tests -p
  'test_collection_runner.py' -v` — PASS, 16/16, zero skips.
- Bounded inline probe using the existing `CollectionRunnerTests` temporary
  fixture: three attempts each record one valid transition then raise an
  operational error, with global interval cap two. **FAIL:** three valid
  transitions execute, report says zero intervals, all three attempts run.
  Failed-attempt physical work must consume the shared budget.
- Successful one-attempt archive -> geom view -> existing P11
  `_validate_record_provenance(..., is_eval=False)`. **FAIL:** the view sample
  lacks `lineage_id`; `adapter_a0` output lacks `provenance` entirely. Actual P11
  public-runner readiness must work, not only the private reconstruction helper.
- Full P02, C1–C5 and physical generation remain incomplete/NOT RUN. No new
  authority for simulator assets, training, or expanding the data architecture
  follows from this verification hold.

## Historical disk-budget verification hold — 2026-09-16

Manager latest disk-budget regression, 2026-09-16: 92 selected focused/regression
tests pass with zero skips under installed `.venv` CPython 3.11.15, but the
implementation remains **NOT ACCEPTED**. A one-transition archive with 100000
characters of metadata and a 32768-byte staging cap reached 107166 bytes on disk
before rejection (measured after flushing the manifest write). An injected reset
failure with a 10000-character exception wrote 21080 bytes of quarantine evidence
under a 629-byte dataset cap. Both probes used temporary directories and existing
test fixtures, not physical execution. Fixed overhead guesses and uncapped failure
reports do not establish peak-disk enforcement; Sol has the reproductions.

## Latest manager verification checkpoint — 2026-09-16

The implementer remains Gemini 3.8 Flash High and independent reviewer remains
GPT 5.6 Sol Medium. After disk fixes passed re-audit, manager found an additional
recovery-path defect: `reconcile_episode` could attach an unknown lineage/family
and invalidate the dataset index. Gemini added six regression tests and corrected
only recovery validation/path handling/budgeted publication. The manager's original
probe now rejects the candidate and preserves a byte-identical valid index.

Fresh manager evidence, repository-root cwd, `.venv` CPython 3.11.15 and installed
editable ICGS:

- `.venv/bin/python -B -m unittest tests/test_episode_archives.py
  tests/test_collection_runner.py tests/test_episode_views.py
  tests/test_episode_data.py tests/test_architecture.py` — **PASS**, 106 executed,
  zero failures/errors/skips (32 archive, 37 runner, 9 views, 25 prior P02,
  3 architecture).
- `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py` — **PASS**, 19/19 each, zero skips;
  two existing invalid-escape SyntaxWarnings in `test_task_router.py` on host
  Python 3.14.4. L0 is not model execution.
- `git diff --check` — **PASS** after removing introduced trailing blank lines.
- Final scoped Sol recovery/documentation re-audit: **PASS**, 106/106 tests with
  zero skips, matching manager execution. The reviewer also independently probed
  intermediate-directory symlink rejection. Unknown lineage/family/split/program
  candidates reject before write; successful recovery is visible to views, and
  disk exhaustion leaves the original index unchanged. API documentation matches
  actual signatures and honestly describes eager loading and physical limitations.
- Manager accepts the **initial-data software slice only**. Earlier failed
  checkpoints above remain historical evidence, superseded by these corrections
  and final scoped verification; no physical or model gate is reclassified.

Owner `.gitignore` edit is preserved; nothing staged or committed. Runtime changes
are confined to `src/icgs/data`, with focused tests and owning docs. No changes to
native IP or P11 model/training objectives were made.

Full L1 model coverage, fresh C1–C5, physical G1/G2 measurements, L3/L4 and training
are **NOT RUN** in this software task. Local `.venv` has no RLBench/PyRep; concrete
scene/expert assets and calibration remain external prerequisites. This initial
slice does not close all P02 task/outcome views or the migration's mandatory gates.

## Observability integration addendum — 2026-09-09 (historical)

The original observability addendum preceded durable P02 collection. The current
initial slice persists explicit attempt/archive metadata but adds no fictitious
physical replay or simulator evidence. Offline run inspection is not a substitute
for measured collection, lineage manifests, or protocol certification.
