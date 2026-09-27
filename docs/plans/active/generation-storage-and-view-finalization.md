# HF Source-of-Truth Generation Storage and View Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the JSON-heavy generation archive with a lossless, HF-authoritative binary archive, bounded local staging, and revision-bound final training views without changing generation quotas or episode semantics.

**Architecture:** A new archive contract stores all information-bearing episode data in compressed binary chunks plus compact manifests in Hugging Face. Workers may emit provisional per-episode pointers, but a finalizer creates reproducible `D_geom`, `D_temporal`, `D_dyn`, and `D_task` snapshots from a frozen HF dataset-manifest revision. Training reads chunks lazily and derives the fixed 2,048-point representation; local disks retain only bounded staging and receipts after remote verification.

**Tech Stack:** Python 3.10–3.12, NumPy compressed NPZ with `allow_pickle=False`, existing ICGS distributed queue/coordinator/publisher, JSON manifests, existing pytest/L0 validation. No new storage dependency in the first migration.

**Spec:** [ADR0015 — HF source-of-truth generation archives and frozen view snapshots](../../decisions/0015-generation-storage-and-view-snapshots.md)

## Global Constraints

- Canonical runtime remains `src/icgs`; no `ip` shims, wrapped legacy subtree, or harness-driven runtime move.
- Hugging Face is the complete source of truth; full raw information may not exist only on local disk or an undocumented external tier.
- Existing `icgs_episode_v2` artifacts, protocol identity strings, quotas, split rules, action/graph semantics, and valid-failure classification remain immutable.
- New serialization uses a new dataset/archive/schema identity and a new HF prefix; no in-place rewrite of historical v2 artifacts.
- `success` and `valid_failure` are retained episodes; `simulator_crash` and `invalid_observation` remain attempt records excluded from training views.
- Lossless binary storage is the first migration. Point quantization, depth-only replacement, and camera-profile correction are separate experiments.
- View mixture is measured on training transitions, not collection quota. Final 70/30 selection requires a frozen dataset-manifest snapshot.
- Local validation must not launch training, downloads, preprocessing workloads, simulator workloads, robot motion, or HF publication.
- Every validation report uses `PASS`, `FAIL`, `SKIPPED`, and `NOT RUN`; `SKIPPED` never counts as `PASS`.

---

## File and interface map

The implementation must keep these ownership boundaries:

| Responsibility | Planned owner | Notes |
|---|---|---|
| Archive contract and array schema | `src/icgs/data/collection/generation/episode_archive.py` | New semantic module; no version-coded source filename |
| Archive write/read | `src/icgs/data/collection/generation/episode_archive.py` | Lossless chunks, manifest, aliases, bounded staging |
| Archive validation | `src/icgs/data/collection/generation/distributed_validation.py` plus archive validator | Identity, hashes, timeline, array schema, no dense JSON arrays |
| Worker materialization | `scripts/generation_episode_worker.py`, `src/icgs/data/collection/generation/rlbench_attempt.py` | Both paths call one canonical writer |
| Runtime profile | `src/icgs/data/collection/generation/distributed_contracts.py`, `src/icgs/configuration/profiles/generation_runtime.json` | Archive profile and local-retention policy |
| Publication/resume | `distributed_publication.py`, `distributed_queue.py`, coordinator/launcher | HF-first, receipt-only local mode after verification |
| Lazy training reader | `src/icgs/data/datasets/generation_archive.py` | Manifest index, chunk cache, A0/A1 adapters |
| View indexes/finalizer | `src/icgs/data/datasets/generation_view_index.py`, `scripts/generation_finalize_views.py` | Provisional per-episode pointers and final revision-bound snapshots |
| Migration/profile tools | `scripts/generation_archive_migrate.py`, `scripts/generation_capacity_probe.py` | Bounded, explicit, no automatic production run |
| Tests | `tests/test_generation_archive.py`, `tests/test_generation_view_finalization.py`, existing generation tests | Pure fixtures locally; live gates separately authorized |

## Phase 0 — Contract freeze and implementation identities

### Task 0.1: Record the accepted storage/view decision

**Files:**
- Read: `docs/decisions/0015-generation-storage-and-view-snapshots.md`
- Modify: `docs/README.md`, `docs/components/generation.md`, `docs/components/generation-artifacts.md`

**Interfaces:**
- Produces the durable `archive_format_id`, new episode schema identity, HF source-of-truth rule, provisional/final view status vocabulary, and local-retention modes used by later tasks.

- [x] Re-read ADR0015 and list its required invariants in the implementation issue/branch record.
- [x] Add the ADR and active plan to the documentation map.
- [x] Update generation lifecycle documentation to distinguish materialization-time provisional pointers from quota-complete final view snapshots.
- [x] Update artifact inventory to mark v2 artifacts as legacy and document the new binary inventory; explicitly state that valid failures remain full episodes.
- [x] Run `git diff --check` and the documentation-link portion of `python3 -B scripts/validate_fast.py`.

### Task 0.2: Add explicit archive profile configuration

**Files:**
- Modify: `src/icgs/data/collection/generation/distributed_contracts.py`
- Modify: `src/icgs/configuration/profiles/generation_runtime.json`
- Test: `tests/test_generation_config.py`, `tests/test_generation_archive.py`

**Interfaces:**
- Add `ArchiveProfileConfig` with fields:
  - `archive_format_id: str`
  - `episode_schema_version: str`
  - `chunk_boundaries: int`
  - `retain_full_cloud: bool`
  - `publish_debug_metadata: bool`
  - `local_artifact_retention: Literal["keep", "receipt_only"]`
  - `view_status: Literal["provisional", "final"]`
- `GenerationRuntimeConfig.archive_profile` returns the validated immutable profile.

- [x] Write rejection tests for unknown fields, nonpositive chunk sizes, unsupported archive IDs, invalid retention, and `retain_full_cloud=False` when the profile claims HF source-of-truth raw retention.
- [x] Write acceptance tests for the compact lossless profile and bounded validation profile.
- [x] Add the profile to the no-secret runtime example without changing the existing legacy profile defaults.
- [x] Run `PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation_config.py tests/test_generation_archive.py`.

## Phase 1 — Lossless canonical episode archive

### Task 1.1: Define manifest and chunk schemas

**Files:**
- Create: `src/icgs/data/collection/generation/episode_archive.py`
- Test: `tests/test_generation_archive.py`

**Interfaces:**
- `ArchiveManifest.from_dict(payload) -> ArchiveManifest`
- `ArchiveManifest.as_dict() -> dict[str, Any]`
- `EpisodeArchiveWriter(profile: ArchiveProfileConfig)`
- `EpisodeArchiveReader(manifest_path: str | Path)`
- `EpisodeArchiveReader.observation(boundary: int) -> Mapping[str, Any]`
- `EpisodeArchiveReader.transition(index: int) -> Mapping[str, Any]`
- `EpisodeArchiveReader.iter_boundaries() -> Iterator[int]`
- `validate_archive_manifest(manifest_path: str | Path) -> dict[str, Any]`

- [x] Define the manifest identity fields: archive format, episode schema, dataset identity, episode/attempt IDs, program/split/subset, outcome, source run, code revision, preprocessing identity, and chunk inventory.
- [x] Define array specs with name, dtype, shape, semantic role, chunk reference, byte count, and SHA256; reject object/pickle arrays.
- [x] Define required arrays for valid episodes: online points/offsets/validity, poses, grips, commands, command grips, `dt`, substeps, robot/object state references, and task-label arrays when present.
- [x] Define attempt archive requirements for crash/invalid records, including measured prefix arrays and explicit null episode identity.
- [x] Define alias entries for byte-identical online/measured arrays so deduplication is explicit and verifiable.
- [x] Add fixtures for success, valid failure, simulator crash, invalid observation, missing optional modality, and distinct online/measured arrays.
- [x] Add RED tests for manifest identity mismatch, offsets that do not end at point count, nonmonotonic offsets, missing `T+1/T/T` arrays, object dtype, pickle loading, and alias target mismatch.
- [x] Run the archive test file and record the expected failures before implementation.

### Task 1.2: Implement bounded chunk writing and lossless reading

**Files:**
- Modify: `src/icgs/data/collection/generation/episode_archive.py`
- Test: `tests/test_generation_archive.py`

**Interfaces:**
- `EpisodeArchiveWriter.write_episode(record, *, raw_arrays, debug_metadata, output_dir) -> ArchiveManifest`
- `EpisodeArchiveWriter.write_attempt(attempt, *, prefix_arrays, debug_metadata, output_dir) -> ArchiveManifest`
- `EpisodeArchiveReader.to_episode_record() -> dict[str, Any]` for test/compatibility reconstruction only.

- [x] Serialize one bounded NPZ chunk at a time with `np.savez_compressed`; reject object arrays before writing and read with `allow_pickle=False`, never buffering an entire dataset.
- [x] Preserve float32/float64 dtypes exactly for the first migration; do not quantize or convert matrices silently.
- [x] Store ragged cloud values plus zero-based offsets and pack boolean validity masks without changing semantic values.
- [x] Write each chunk to a temporary sibling, fsync/rename it, calculate SHA256/bytes, then write the manifest last through an atomic rename.
- [x] Store only compact scalar/string debug metadata in `debug.json`; route all numeric arrays through chunk inventory.
- [x] Reconstruct a v2-compatible in-memory record in tests and run the existing episode validator to prove timeline/provenance parity.
- [x] Add corruption tests for altered chunk bytes, altered manifest hash, truncated chunk, path traversal, symlink, missing alias target, and partial manifest.
- [x] Run `PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation_archive.py` and require all selected tests to execute.

### Task 1.3: Converge both materialization paths on the canonical writer

**Files:**
- Modify: `scripts/generation_episode_worker.py:56-293`
- Modify: `src/icgs/data/collection/generation/rlbench_attempt.py:247-351`
- Test: `tests/test_generation_attempt.py`, `tests/test_generation_archive.py`

**Interfaces:**
- Both paths call `EpisodeArchiveWriter.write_episode` or `write_attempt`; neither path writes dense point arrays to `episode.json`.

- [x] Replace the distributed writer’s dense `episode.json`, duplicate layout arrays, duplicate telemetry arrays, and array-bearing `execution.json` with one archive manifest, chunk files, and compact debug metadata.
- [x] Preserve all currently captured debug fields, including plan, routine, predicate distances, sensor-randomization receipt, object/robot states, errors and tracebacks.
- [x] Keep valid failures on the episode path and crash/invalid results on the attempt path.
- [x] Keep unavailable RGB/depth/mask/joint/object modalities omitted unless they are actually captured; never synthesize them.
- [x] Ensure per-episode provisional pointers contain only archive references and do not copy observations.
- [x] Add tests that assert no dense numeric arrays occur in JSON manifests/debug metadata and that valid failures have the same archive completeness as successes.
- [x] Run focused generation tests plus `git diff --check`.

## Phase 2 — Validation, publication, and HF-first retention

### Task 2.1: Validate the new inventory before ingestion

**Files:**
- Modify: `src/icgs/data/collection/generation/distributed_validation.py`
- Test: `tests/test_generation_validation.py`, `tests/test_generation_archive.py`

**Interfaces:**
- `validate_closed_result()` accepts the new archive profile and returns an immutable `ValidatedResult` whose `file_sha256` covers every chunk/manifest/debug file.
- `validate_archive_manifest()` is called before ingestion and before publication.

- [x] Require the new archive manifest and reject a v2 dense-JSON result under the new production profile.
- [x] Validate archive identity against `GenerationJob`, including episode/attempt/program/outcome/plan identity.
- [x] Validate timeline cardinalities, array schemas, chunk hashes, aliases, partial-modality boundary declarations, and provenance fields.
- [x] Keep simulator crashes and invalid observations out of episode manifests and training view inputs.
- [x] Add tests for valid success, valid failure, crash, invalid observation, malformed chunk inventory, dense JSON regression, and immutable re-ingestion across a different local mount.
- [x] Run `PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation_validation.py tests/test_generation_archive.py`.

### Task 2.2: Publish complete HF archives and finalize local retention

**Files:**
- Modify: `src/icgs/data/collection/generation/distributed_publication.py`
- Modify: `src/icgs/data/collection/generation/distributed_queue.py`
- Modify: `scripts/generation_coordinator.py`, `scripts/generation_launch.py`
- Test: `tests/test_generation_publication.py`, `tests/test_generation_queue.py`, `tests/test_generation_worker.py`

**Interfaces:**
- `HuggingFaceBatchPublisher` publishes all canonical chunks/manifests/debug metadata and records the remote revision.
- `FilesystemJobQueue.mark_published(job_id, *, retention: Literal["keep", "receipt_only"]) -> Path` retains receipts/hashes and optionally prunes only after verified remote publication.

- [x] Upload the complete archive inventory for each success/valid-failure episode and complete attempt inventory for crash/invalid results.
- [x] Keep HF `dataset_manifest.json`, `resume_receipt.json`, publication receipt, per-episode manifests, and artifact hashes consistent with the new archive identity.
- [x] Mark prefix views as `PROVISIONAL` until finalization; never label cumulative batch IDs as final training samples.
- [x] Add an explicit receipt-only retention path that refuses to prune before remote hash verification and keeps local queue receipts, remote commit OID, artifact hashes, and source run identity.
- [x] Add idempotency tests for retry after remote commit, remote verification failure, local result already pruned, duplicate publication, and resume from remote manifest with no local episode payload.
- [x] Keep default bounded validation in `keep` mode; enable `receipt_only` only in the new production archive profile.
- [x] Run focused publication/queue/worker tests without network or HF credentials.

### Task 2.3: Extend resume bootstrap for HF-only recovery

**Files:**
- Modify: `src/icgs/data/collection/generation/distributed_planner.py`
- Modify: `scripts/generation_coordinator.py`, `scripts/generation_launch.py`
- Test: `tests/test_generation_control.py`, `tests/test_generation_planner.py`, `tests/test_generation_publication.py`

**Interfaces:**
- `load_remote_manifest()` validates the new dataset/archive identity and returns immutable episode/attempt rows plus archive manifest references.
- `DistributedPlanner.from_manifest()` restores uniqueness/quota state without local episode binaries.

- [x] Validate new archive/schema identity, source run disjointness, episode/attempt uniqueness, plan identity, catalog split/kind/asset constraints, split/subset, and per-episode manifest references.
- [x] Prove resume can reconstruct planner state after local `result_dir` payloads are absent but pinned HF manifest/receipts and the complete archive inventory are available (local fake HF tree).
- [x] Preserve valid failures and failure attempts during resume; never reset quota because a local payload was pruned.
- [x] Add tests for malformed remote archive manifest, conflicting immutable row, missing chunk reference, matching remote hash, disjoint new run ID, receipts, and self-consistent malformed chunk/array inventories.
- [x] Run the resume/control test subset and L0.

Local evidence (macOS, Python 3.14.4, fixture-only): `PYTHONPATH=src python3 -B -m
pytest -q tests/test_generation_control.py tests/test_generation_planner.py
tests/test_generation_publication.py tests/test_generation_queue.py` PASS 170/170;
`PYTHONPATH=src python3 -B -m pytest -q tests/test_generation*.py
tests/test_capacity_probe.py` PASS 419/419; `python3 -B scripts/validate_fast.py`
PASS (22 harness tests); `git diff --check` PASS. No live HF, simulator, or full
generation run was selected.
The three rehashed profile-mismatch fixtures were RED before the production
guard (3 failures: resume returned without raising), then GREEN after it (3/3).

### Task 2.4: Bound interrupted-publication recovery downloads

**Files:**
- Modify: `scripts/generation_coordinator.py`
- Test: `tests/test_generation_publication.py`, `tests/test_generation_control.py`

**Interfaces:**
- Archive-profile HF control-file downloads during interrupted publication
  reconciliation and initial manifest discovery use owned, temporary cache
  directories and preserve commit-OID discovery from the returned snapshot path.

- [x] Add RED fake-HF tests proving archive recovery and initial manifest
      discovery supply an owned cache, reject returned files outside it, and
      remove cache bytes after success or error.
- [x] Keep legacy non-archive download/retry behavior unchanged.
- [x] Preserve the existing fail-closed data/receipt reconciliation and exact
      local/remote hash checks; no remote write or pruning before verification.
- [x] Run focused control/publication tests and L0 without contacting HF.

**Local acceptance (2026-09-26):** Commits `4a521b1` and `cc1f75b` use
disposable owned caches for archive recovery and initial manifest discovery.
The scoped review closed a fail-open non-resume startup path, returned-directory
misclassification, and an external symlink alias that could spoof a snapshot
OID. Fresh parent CPython 3.11.15: control/publication **133 PASS**,
generation/capacity **478 PASS**, system L0 **22 PASS** (two pre-existing
invalid-escape warnings), `git show --check` **PASS**. No selected skips;
live HF behavior is **NOT RUN** and remains a Task 5.3 gate.

## Phase 3 — Lazy HF archive reader and training views

### Task 3.1: Implement lazy archive access

**Files:**
- Create: `src/icgs/data/datasets/generation_archive.py`
- Modify: `src/icgs/data/datasets/generation_views.py`
- Leave `src/icgs/data/datasets/episodes.py` unchanged; P02/A0/A1 behavior remains separate and is covered by compatibility tests.
- Test: `tests/test_generation_archive.py`, `tests/test_generation_views.py`

**Interfaces:**
- `ArchiveDatasetIndex.from_manifest(dataset_manifest_path) -> ArchiveDatasetIndex`
- `ArchiveDatasetIndex.sample_refs(view: str, role: str) -> Iterator[SampleRef]`
- `EpisodeArchiveReader.observation(boundary)`, `.transition(index)`, and `.task_labels(boundary)` provide lazy values.
- `build_generation_view(records_or_index, view, *, role="train", mix=None)` supports archive-backed references without loading all episodes.

- [x] Index manifests and sample references without loading full point arrays.
- [x] Add bounded LRU chunk caching with an explicit dataset byte cap and deterministic cache keys including archive/preprocessing identity.
- [x] Make `D_geom`, `D_temporal`, `D_dyn`, and `D_task` use archive references; preserve causal histories and intervention flags.
- [x] Derive the 2,048-point training input lazily with a strict preprocessing identity/hash and deterministic sample seed; expose a compact metadata payload for callers.
- [x] Keep `adapter_a0`/`adapter_a1` semantics and native PyG loaders separate from the new archive reader.
- [x] Test random boundary access, cross-chunk windows, cache eviction, missing optional modalities, causal masking, and valid-failure inclusion.
- [x] Run the archive/view tests with tiny fixtures only.

**Integration boundary:** Persisting the metadata payload automatically in a run
record and teaching `icgs train` to consume the archive-backed index are **NOT
IMPLEMENTED**. The current trainer reads native PyG sample directories and has no
archive-backed metadata seam; this task does not add one. The real Open3D filter
execution is also **NOT RUN** when Open3D is unavailable; the fixture test isolates
the deterministic sampling/frame path while production delegates to the unchanged
native filter.

**Local validation record (2026-09-25):** Environment was CPython 3.14.4,
NumPy 2.4.4, PyTorch 2.11.0, PyG 2.5.0, SciPy 1.17.1, pytest 8.4.2; Open3D was
not installed.

- **PASS** — `PYTHONPATH=src python3 -B -m pytest -q tests/test_generation_archive.py tests/test_generation_views.py tests/test_generation.py` (87 passed; no skips after fix rounds 1–2; review round 1 requested archive hash/role/modality corrections, round 2 requested legacy sequence-role parity).
- **PASS** — `PYTHONPATH=src python3 -B -m pytest -q tests/test_generation*.py tests/test_capacity_probe.py` (436 passed; one Python multiprocessing fork deprecation warning).
- **PASS** — `python3 -B scripts/validate_fast.py` (22 harness self-tests passed; L0 syntax, local links, boundaries, and generation naming passed; two existing invalid-escape `SyntaxWarning`s from `tests/test_task_router.py`).
- **NOT RUN** — actual Open3D statistical-outlier filtering; unavailable in this environment. No HF access/publication, training, simulator, full preprocessing, or C1–C5 execution was run for this bounded task. Archive-backed `icgs train` integration and automatic run metadata persistence remain NOT IMPLEMENTED.

### Task 3.2: Add provisional and final view snapshots

**Files:**
- Create: `src/icgs/data/datasets/generation_view_index.py`
- Create: `scripts/generation_finalize_views.py`
- Modify: `src/icgs/data/collection/generation/distributed_publication.py`
- Test: `tests/test_generation_view_finalization.py`, `tests/test_generation_publication.py`

**Interfaces:**
- `build_provisional_episode_pointers(episode_manifest) -> tuple[dict[str, Any], ...]`
- `finalize_generation_views(dataset_manifest, *, source_revision, role_specs, mixture_seed) -> ViewFinalizationReceipt`
- `ViewFinalizationReceipt` contains source manifest SHA256, source HF revision, view IDs, counts, mixture identity, and status.

- [x] Generate per-episode provisional pointers with `role="all"` and `mix=False`; mark them `PROVISIONAL`.
- [x] Build final role-specific snapshots from the complete HF dataset manifest: train=`train_core`, validation=`train_val`, evaluation=`development`+`test`.
- [x] Apply the 70/30 nominal/perturbed selection only to final `D_temporal` and `D_dyn` snapshots using the recorded collection seed/mixture seed.
- [x] Exclude crash/invalid attempts; retain valid failures when their split/subset is eligible.
- [x] Write view manifests with source revision/SHA256, archive identity, preprocessing identity, role, mixture, sample counts, and finalization timestamp.
- [x] Make finalization idempotent: same source manifest/seed produces the same bytes; a source revision/manifest mismatch fails closed only when reusing the same existing output target, while a later source revision uses its own `<output-prefix>/<source-revision>/seed-<mixture-seed>/` directory.
- [x] Test partial/provisional versus final status, changing dataset snapshot, 70/30 counts, split leakage, valid-failure retention, and deterministic re-run.
- [x] Run the view-finalization test subset; do not contact HF.

**Local acceptance (2026-09-26):** Commits `701e809` and `5b35f6e` implement
provisional pointers plus twelve role/view snapshots at a pinned source revision.
An independent scoped review accepted the 7:3 transition-count correction,
source-prefix binding, pinned pointer verification, owned finalizer caches, and
post-commit byte verification. Fresh parent CPython 3.11.15 results: archive/view/
publication subset **118 PASS**, generation/capacity **450 PASS**, L0 **22 PASS**
(two pre-existing invalid-escape warnings), `git show --check` **PASS**.
Live HF publication/finalization, simulator, training, Open3D SOR, C1–C5 and
full generation are **NOT RUN** here; Task 5.3 remains the live gate. A `FINAL`
mixed train view now requires at least one complete 7 nominal : 3 perturbed
transition unit in each mixed view, otherwise finalization fails before writing.

## Phase 4 — Migration tooling and bounded capacity evidence

### Task 4.1: Convert existing v2 artifacts without rewriting them

**Files:**
- Create: `scripts/generation_archive_migrate.py`
- Test: `tests/test_generation_archive_migration.py`
- Modify: `docs/components/generation-artifacts.md`

**Interfaces:**
- `migrate_episode(source_reader, target_writer, *, source_identity, target_profile) -> MigrationReceipt`
- `MigrationReceipt` records source artifact hashes, target archive hashes, field/timeline counts, and conversion warnings.

- [x] Read one v2 episode/attempt at a time from an explicitly supplied source revision/prefix; never scan arbitrary HF paths.
- [x] Convert dense JSON arrays to lossless chunks and preserve provenance/debug fields.
- [x] Reject incomplete/malformed source artifacts rather than silently repairing them; write a migration failure receipt.
- [x] Verify reconstructed v2 semantics against the source before publishing the new target artifact.
- [x] Publish converted artifacts only to an explicitly supplied new prefix; never delete or overwrite v2 data.
- [x] Add tests for success, valid failure, crash attempt, invalid observation, malformed source, hash mismatch, and idempotent re-run.
- [x] Run migration tests against tiny local fixtures only; a remote migration job remains separately authorized.

**Local acceptance (2026-09-26):** Commits `0184be5`, `e950efc`, and
`0037a93` added the explicit one-record converter, typed-sidecar reconciliation,
source/target receipts, decoded-sidecar budget, redaction-safe failure behavior,
and offline fake-HF publication checks. Two scoped review rounds closed a source
semantic-parity defect, nested source-inventory omission, decoded-size/secret
handling gaps, and a lossy-existing-target reuse hole. Fresh parent CPython
3.11.15 results: migration **21 PASS**, generation/capacity **471 PASS**, L0
**22 PASS** (two pre-existing invalid-escape warnings), `git show --check`
**PASS**. Live HF migration is **NOT RUN** and requires separate authorization;
JSON-only v2 dtype ambiguity is reported, not silently repaired. The converter
is 1,936 lines after the fixes and remains a maintenance risk to revisit before
any broad historical conversion job.

### Task 4.2: Add size instrumentation to the bounded capacity probe

**Files:**
- Modify: `scripts/generation_capacity_probe.py`
- Modify: `src/icgs/data/collection/generation/capacity_probe.py`
- Test: `tests/test_capacity_probe.py`
- Modify: `docs/experiments/generation-validation/capacity-probe-tpu-v6e1-20260924.md`

**Interfaces:**
- `capacity_receipt.json` adds per-result `bytes_by_category`, `boundaries`, `raw_points`, `archive_profile`, and `local_peak_bytes`.

- [x] Categorize bytes into archive chunks, manifests, debug metadata, views, and receipts.
- [x] Record point counts and boundary counts without loading all prior episodes.
- [x] Enforce a bounded per-result and total staging cap before launching a probe stage.
- [x] Add tests for byte-cap rejection, category accounting, compact-profile validation, and no-worker preflight behavior.
- [x] Run only configuration/preflight and fixture tests locally; live capacity measurement remains explicit.

**Local acceptance (2026-09-26):** Commits `8660299`, `6779b37`, and
`ae9d06f` add per-result category/timeline/point accounting and writer-reported
local-peak upper-bound labels. Archive-profile probes now require explicit
per-result and global staging caps; product and cumulative retained-stage
budgets are checked before worker launch. A Grok 4.7 High task review accepted
the cap fixes. Fresh parent CPython 3.11.15 results: capacity **31 PASS**,
generation/capacity **497 PASS**, L0 **22 PASS** (two pre-existing invalid-escape
warnings); task-range `git diff --check` **PASS**. Live capacity measurement,
simulator, HF, training, C1–C5, and full generation are **NOT RUN**.

## Phase 5 — Documentation and acceptance closure

### Task 5.1: Update operational and validation documentation

**Files:**
- Modify: `docs/components/generation.md`
- Modify: `docs/components/generation-artifacts.md`
- Modify: `docs/audits/2026-09-20-generation-contract.md`
- Modify: `docs/plans/active/resumable-multihost-generation.md`
- Modify: `tests/README.md`
- Modify: `docs/README.md`

- [x] Document the new HF archive tree, SOT rule, archive/schema identities, valid-failure retention, provisional/final view lifecycle, and receipt-only local retention.
- [x] Document exact clean-machine HF reader/resume commands without embedding credentials.
- [x] Document that final view snapshots bind to an HF revision and are not regenerated from remembered local state.
- [x] Add validation commands for archive roundtrip, view finalization, remote-resume contract fixtures, and size budgets.
- [x] Record the current v2 artifact format as legacy historical evidence and preserve links to old manifests.
- [x] Run link validation and `git diff --check`.

### Task 5.2: Local acceptance gates

**Files:**
- Test: all focused generation/archive/view tests; no runtime code changes in this task.

- [x] Run a superset of the listed generation tests, including archive, finalization, migration, and capacity checks, with the shared CPython 3.11 virtual environment because this worktree has no `.venv`.
- [x] Run `python3 -B scripts/validate_fast.py`.
- [x] Confirm no dense observation arrays exist in new JSON manifests/debug files.
- [x] Confirm valid failures have complete binary archives and appear in eligible view snapshots.
- [x] Confirm local receipt-only retention tests prune only after verified remote identity.
- [x] Report L1/L2/L3/L4, simulator, HF publication, full quota, and training as `NOT RUN` unless separately authorized.

**Local fixture gate execution (2026-09-26, commit `58b1d7a`):** A scoped Grok
4.7 High re-review accepted both HF-completeness fixes: semantic redaction of
v2 execution/sidecar/artifact-manifest metadata now fails before fresh or reuse
migration, and the production archive-profile worker writes captured wrist-depth
frames with source-boundary indices to episode and attempt archives. Fresh parent
CPython 3.11.15 / NumPy 1.26.4 / pytest 8.4.2:
`PYTHONPATH=src /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -B -m
pytest -q tests/test_generation*.py tests/test_capacity_probe.py` — **PASS,
507 passed, 0 skipped**. A focused six-case selection covering compact debug
JSON, success/valid-failure archive parity, final train mixture and verified-only
receipt pruning — **PASS, 6 passed, 0 skipped**. System CPython 3.14.4
`python3 -B scripts/validate_fast.py` — **PASS, 22 L0 tests**, with two
pre-existing invalid-escape warnings. `git show --check 58b1d7a` and
`git diff --check 6f917a1..58b1d7a` — **PASS**. No selected tests were skipped.
Live HF publication, clean-machine resume, simulator capture, actual Open3D
SOR, native archive-backed training, full preprocessing, C1–C5 and full quota
generation are **NOT RUN** by these local checks. The crash-prefix hardening
identified during the scoped review is recorded below; Task 5.3 remains
required before any full-generation authorization.

**Task 5.3 view-metadata finding (2026-09-27):** Read-only review of source
manifest revision `f3ee930e366a7ff8132a6412bd06d59c07ebdbc2` and immutable view
output HEAD `8a3416a` found that `train/D_task` listed 841 samples but reported
all 841 as nominal, while source-row identity gives 586 nominal and 255
perturbed. The D_task archive refs omitted `episode_kind`, so finalization's
per-sample kind counter defaulted every ref to nominal. The local correction
propagates the already-validated archive provenance kind into D_task refs and
tests final snapshot counts, membership, label pointers, and role separation.
The published output remains unchanged; any corrected remote finalization must
use a distinct output prefix after review. HF access and remote re-finalization
were **NOT RUN** at this local-fix checkpoint. The subsequent bounded live
readback and corrected-prefix publication are recorded in the
[2026-09-27 archive acceptance record](../../experiments/generation-validation/archive-aa869b3-20260927/README.md).

**Crash-prefix follow-up (2026-09-26, commit `959aca6`):** The production
archive-profile exception path now carries already captured observations,
commands and state into a `simulator_crash` attempt, including the case where
one command was issued but its post-action observation failed. A pre-capture
exception still has an empty prefix. RED tests reproduced the former zero-prefix
archive; GREEN tests validate the canonical attempt archive and unchanged v2
behavior. Fresh parent CPython 3.11.15 generation/capacity — **PASS, 511 passed,
0 skipped**; system CPython 3.14.4 L0 — **PASS, 22 tests** with the same two
pre-existing warnings; whitespace checks — **PASS**. Grok 4.7 High approved the
scoped handoff with no Critical, Important or Minor findings. Live HF, simulator
and training remain **NOT RUN**.

### Task 5.3: Bounded remote acceptance

**Files:**
- Evidence: `docs/experiments/generation-validation/<new-archive-acceptance>/`

- [x] Start from a fresh committed revision and a new isolated HF validation prefix.
- [ ] Run a bounded set containing success, valid failure, simulator-crash/invalid-attempt, publication, resume, and workers-only attach cases.
- [x] Verify every declared remote chunk/manifest/debug hash for the nine bounded
      episode archives from owned scratch; separately restore six and nine
      episodes from disjoint roots with zero local result payloads. Failure
      attempts remain a separate open case in the preceding checkbox.
- [x] Finalize twelve views against a pinned nine-row manifest, correct the
      observed `D_task` episode-kind metadata defect, and independently verify
      exact 7:3 transition units, split separation, immutable output bytes and
      same-target idempotency under a new corrected prefix.
- [ ] Record bytes per episode, peak local staging bytes, upload throughput, publication commits, resume revision/SHA, and all PASS/FAIL/SKIPPED/NOT RUN states.
- [ ] Establish a hard live-writer staging bound and repeat a finite capacity
      probe; sampled tree detection alone does not satisfy this gate.
- [ ] Prove that a cap-triggered worker stop preserves scratch and prevents a
      new claim during the still-live lease as well as after lease expiry.
- [ ] Do not authorize the 7,520-attempt production run until archive quality, HF-only recovery, final views, and storage budget gates pass.

**Partial live acceptance (2026-09-27):** The
[durable validation record](../../experiments/generation-validation/archive-aa869b3-20260927/README.md)
contains exact code revisions, VPS environment, run roots, HF prefixes,
per-episode bytes, pinned data/receipt/HEAD identities, full archive readbacks,
HF-only resume and corrected FINAL-view receipts. Nine bounded episodes grew
the HF manifest 1→2→3→4→5→6→7→8→9 across restarts; the complete nine-row
readback and zero-payload resume **PASS**. The first immutable FINAL prefix is
preserved as a **FAIL** for `D_task` kind counts; corrected output from reviewed
`8a2b811` **PASS** under a distinct prefix. A second corrected prefix was
accidentally published, its thirteen files were checked byte-identical, and
both outputs were preserved. A separate one-job `receipt_only` run at reviewed
`8a2b811` then **PASS**ed pinned HF byte verification, post-verification local
payload pruning with a durable per-job receipt, and disjoint HF-only resume
with zero payloads; its data/receipt/HEAD OIDs are in the same record. Task 5.3
stays open: live failure-attempt HF retention, workers-only attach, a hard
live-writer staging bound/capacity retest, native training ingestion, and full
quota remain open. At reviewed `8a2b811`, a new no-HF, one-job G1 probe with
80,000,000-byte result/stage/total caps returned **FAIL** after its one-second sampler
observed at least 81,804,171 staged bytes (81,795,696 of writer scratch) and
terminated the worker. No closed artifact was produced; the claimed job,
52 NPY spool files and write marker remain in the preserved run root. A live
lease still hid that scratch from orphan discovery at inspection, so immediate
fail-closed behavior for another claim is not established; post-expiry
discovery is a code-path inference, **NOT RUN** on this root. Exact receipt
hashes, queue state and limitations are in the durable validation record.
The observed breach is detection, not a hard writer-side staging bound. No
7,520-attempt authorization follows from this partial gate.

**Data-generation-only retest (2026-09-27):** Fresh local and `vps-a` focused
generation/capacity suites each **PASS**ed 577 tests with zero skips. The
generation environment verifier **PASS**ed its selected CPU/simulator-path
checks, and one kept real archive observation **PASS**ed Open3D SOR and
2,048-point local-frame derivation. A no-job, no-HF archive-profile workers-only
attach **PASS**ed on `vps-a` with a separate host identity and credential-free
idle worker; this is not a second-physical-host or live claim result. The
80 MB G1 capacity **FAIL**, missing hard cumulative writer limit, and live-lease
claim window remain unchanged. No additional simulator attempt or HF publication
was run. Exact commands, failed test-root preparation, hashes, and the broader
pre-clarification suite results are in the durable validation record. Per the
user's narrowed scope, GPU C1–C5 and training were not selected as generation
readiness checks. Full generation is **NOT AUTHORIZED**.

**Writer-cap live retest (2026-09-27):** At committed `d793b39`, a fresh
one-job, no-HF G1 probe on `vps-a` hit `ArchiveWriterCapExceeded` before a
numeric write would have raised the retry-root inventory from 78,649,839 to
80,222,703 bytes against an 80,000,000-byte result cap. The 51 NPY spool files
and write marker were preserved and a durable queue safety stop exists; no ready
result was published. This is positive evidence for the hard *per-result*
pre-write limit, but the probe itself exited with a generic worker-exit error
and wrote no stage receipt. The stop receipt also lists the same orphan
inventory twice, inflating its top-level byte total. The original and new
failed roots remain untouched. The exact command, hashes and limitations are
in the [validation record](../../experiments/generation-validation/archive-aa869b3-20260927/README.md).
The 80 MB limit cannot hold this G1 archive, a successful closed-result peak
and aggregate storage budget are still unmeasured, and live failure-attempt HF
retention remains **NOT RUN**. Full generation remains **NOT AUTHORIZED**.

**Reporting fix and generation-only fixture check (2026-09-27):** Reviewed
commits `3fec036` and `d194bbe` deduplicate an identical worker orphan-stop
inventory and emit a `FAIL` capacity stage receipt naming an existing worker
stop, including the reproduced worker-exit/deadline/low-memory races. Fresh
local generation/capacity fixtures **PASS**ed 611 tests with zero skips; a
clean, separate `vps-a` checkout at `d194bbe` **PASS**ed the same 611 tests
and L0 **PASS**ed 22. The `d793b39` live G1 failure was **not rerun** after
these reporting fixes. The exact commands, review caveat and Git-bundle hash
are in the [validation record](../../experiments/generation-validation/archive-aa869b3-20260927/README.md).
The failed G1 root remains preserved, 80 MB is insufficient for a closed G1
archive, and full generation remains **NOT AUTHORIZED**.

### Task 5.3 continuation — `vps-a` generation-only acceptance

**Goal:** Close the measured G1 storage and live failure-attempt HF retention
gaps without starting the 7,520-attempt collection. This is a bounded
component-validation continuation of the accepted archive migration; protocol,
quota, episode labels, and published prefixes remain unchanged. All new run
roots and the HF validation prefix must be disjoint from earlier evidence.

- [ ] **Capacity, no HF:** Pin one clean `vps-a` checkout to code revision
      `53bc3e66de4c279cf7565eca4cd5ca611d0027b3`. Preflight free disk,
      memory, simulator paths, absent run root, and no generation processes.
      Use one G1 job, one worker/slot, a 600-second worker timeout, 900-second
      global deadline, 536,870,912-byte `max_result_bytes`, and
      1,073,741,824-byte stage/total staging caps. Execute
      `scripts/generation_capacity_probe.py --execute` with the approved
      composition manifest under a fresh root. On `FAIL`, preserve every byte
      and stop; on `PASS`, independently validate the closed archive and record
      per-category compressed bytes, writer-reported upper bound, sampled stage
      lower bound, queue counts, process exit, config hashes, and root identity.
- [ ] **Two controlled failure attempts, fresh HF prefix:** Use only an
      operator-owned validation shim around the existing RLBench
      `TaskEnvironment.step` boundary, never a production protocol change:
      induce one non-IK exception after captured T01 frames and one empty
      T02 wrist point-cloud frame. Enqueue exactly those two nominal jobs in
      a fresh single-host run with a positive per-result writer cap; start
      one credential-free worker and no refill loop. Require closed
      `simulator_crash` and `invalid_observation` attempt archives with null
      `episode_id`, measured prefixes, depth when captured, and immutable
      inventories. If a stop/invalid artifact appears, do not publish it.
      Otherwise publish through the canonical coordinator to a new
      `validation/` HF prefix, hash every declared file at one pinned commit,
      and restore both attempts from a separate disjoint root with zero local
      result payloads. Preserve all receipts and existing HF prefixes.
- [ ] **Storage decision and handoff:** Use the successful G1 writer upper
      bound plus the measured attempt sizes and chosen production worker/slot
      count to calculate a worst-case concurrent reservation and a queue
      backlog limit. Prove that a still-live lease cannot claim past the
      configured reservation, and that verified receipt-only pruning releases
      capacity. If production host count or hard aggregate backpressure is
      undecided, keep full generation **NOT AUTHORIZED**; `vps-a` testing
      cannot certify a second physical host. Record commands, environment,
      PASS/FAIL/SKIPPED/NOT RUN, hashes, and residual risks in the existing
      generation validation record. No GPU, training, C1–C5 or full-quota run
      is part of this continuation.

**Continuation stop (2026-09-27):** The first fresh `vps-a` G1 run at
`53bc3e6` did not reach either byte cap; it **FAIL**ed at its 600-second
worker timeout while spooling thousands of small numeric metadata arrays.
The durable safety stop preserves 5,825 hashed files (102,163,510 bytes),
and the stage receipt records one claimed, zero ready/published results.
Per the checklist, no controlled failure-attempt HF run followed this failed
probe. The exact command, hashes, file-size distribution and read-only
diagnosis are in the [validation record](../../experiments/generation-validation/archive-aa869b3-20260927/README.md).
The probe's 600-second schema maximum is shorter than the production runtime
example's 1,200-second timeout; any revised time bound or writer-performance
change requires a separately reviewed finite rerun. Full generation remains
**NOT AUTHORIZED**.

## Compatibility, rollback, and recovery

- Existing v2 artifacts and HF prefixes remain immutable and readable.
- The new profile uses a new dataset/archive/schema identity and new HF prefix.
- If archive validation or view finalization fails, stop the new run, retain its HF receipts, and resume only with a new disjoint run ID after correcting the source revision/profile.
- If remote verification fails, retain local staging and do not prune; retry publication from the immutable local result.
- If remote verification succeeds, receipt-only pruning may remove only validated result directories under the configured run root; queue receipts and remote identities remain.
- If lazy training reads disagree with v2 reconstruction, disable the new training profile and use the existing v2 reader; do not alter native preprocessing or relabel episodes.
- No destructive HF cleanup or history squashing is part of this plan.

## Validation strategy and current evidence

Local plan gates are pure fixture/unit/L0 checks. No simulator, download, preprocessing workload, training, robot motion, or HF publication is incidental to this plan.

## Implementation progress

- [x] Phase 0.2 — Added an opt-in immutable archive profile. Legacy runtime configs still parse with `archive_profile=None`; the checked-in bounded validation example selects the archive format with `keep` retention. Profiles also bound each NPZ chunk to 256 MiB of total uncompressed ZIP member bytes, including `.npy` headers; the writer streams `.npy` scratch pieces and records a local staging byte upper bound that includes spool plus final files.
- [x] Recorded implementation rulings: `numpy.savez_compressed` has no `allow_pickle` argument, so archive writes reject object arrays and readers use `numpy.load(..., allow_pickle=False)`; dated audits remain historical evidence and current storage behavior belongs in current owner docs.
- [x] Phase 1 — Canonical archive writer/reader and both materialization paths. Archive core, both opt-in materializers, archive-manifest worker handoff/detection, and scoped review corrections are locally tested.
- [x] Phase 2 — Local implementation of new-profile validation, complete HF publication, receipt-only retention, and HF-only resume; live HF acceptance remains a later gate.
- [x] Phase 3 — Lazy archive readers and revision-bound provisional/final views (local implementation only; Task 5.3 live gate remains).
- [x] Phase 4 — Local migration and capacity instrumentation; live migration and capacity measurement remain NOT RUN.
- [ ] Phase 5 — Current owner documentation and Task 5.2 local acceptance are complete; Task 5.3 bounded remote acceptance remains open.

Tasks 2.1, 2.2, and 2.3 are locally implemented. `validate_closed_result(..., archive_profile=...)`
uses the canonical archive validator before ingestion and emits episode/attempt
rows with relative HF archive references, source/profile identities, and every
file hash. Profile-aware publisher operation construction revalidates the queue
result before upload. Legacy configs continue through the original dense-JSON
publication path. Task 2.2 publishes the complete file inventory and dataset
manifest hash, binds provisional views and receipts to archive identity, verifies
each remote file in an owned per-file cache at a pinned revision, and prunes only
after a verified receipt is durably bound to the job/run/source/hash inventory.
Receipt-only queue receipts retain the manifest row needed to rebuild local
planner state after payload pruning. Task 2.3 now requires a pinned lowercase HF
revision, checks both remote control receipts against the manifest and latest batch,
downloads and hashes every row file through owned bounded scratch, invokes the
canonical archive validator, and checks AttemptPlan against catalog split/kind/asset
rules before planner construction. The full per-archive profile must also equal
the dataset/runtime profile, including chunk limits, retention, and view status.
Local fake HF tests cover allowed recovery and
targeted malformed cases; live HF acceptance and production run authorization
remain later gates.

Task 2.2 rulings: validation mode rejects `receipt_only`; the checked-in bounded
validation profile remains `keep`. A data-commit timeout before an OID is known
retries the same ingested batch; after a successful data commit, retries reuse
its recorded data revision and, after the receipt commit, retry byte verification
against the same returned receipt-commit OID. Verification failures leave the
local result payload in place. Archive files, dataset manifest, resume receipt,
four provisional views and the exact receipt snapshot are downloaded sequentially
into distinct temporary `cache_dir` directories; each cache is removed before the
next file, and no shared HF cache participates. After a process interruption, the
coordinator reconciles the latest remote manifest/receipt revision before another
archive upload is allowed.

Phase 0.2 TDD evidence (`/tmp/icgs-generation-hf-archive.koAE1D`, system CPython 3.14.4, NumPy 2.4.4, pytest 8.4.2):

- RED: `PYTHONPATH=src python3 -B -m pytest -q tests/test_generation_config.py` — 10 new profile assertions failed because the profile API did not exist; 32 legacy tests passed.
- GREEN: `PYTHONPATH=src python3 -B -m pytest -q tests/test_generation_config.py` — **PASS, 42 passed** after initial implementation.
- RED/GREEN for the checked-in example profile: the targeted profile test first failed because the example had no archive profile, then the full config file passed with **42 passed** after the example was updated.
- Environment limit: this isolated worktree has no `.venv`; the system interpreter is CPython 3.14.4 rather than the documented 3.10–3.12 validation environment.

Post-review profile hardening evidence (`PYTHONPATH=src python3 -B -m pytest -q tests/test_generation_config.py -k 'unhashable_enum_values or literal_retention'`): **RED, 5 failed**, exposing unhashable enum values leaking `TypeError` and missing `Literal` annotations. After adding string-type checks, plan `Literal` types and `max_chunk_bytes`, config acceptance was rerun on supported CPython 3.11.15 and passed **48 tests**.

Phase 1 archive-core evidence (read-only use of `/Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python`, CPython 3.11.15; `PYTHONPATH=/tmp/icgs-generation-hf-archive.koAE1D/src`):

- RED: `PYTHONPATH=src python3 -B -m pytest -q tests/test_generation_archive.py` — **FAIL as expected**, 1 API-availability failure and 7 skipped because the module did not exist yet.
- RED/GREEN for lazy task labels: the targeted test first failed because `task_labels()` called `to_episode_record()` and materialized every point cloud; it then passed after boundary-row chunk reads were added.
- RED/GREEN for symlinked chunks: the targeted test first exposed inventory mismatch masking the symlink error; path preflight now rejects symlinks explicitly.
- GREEN focused core: `PYTHONPATH=/tmp/icgs-generation-hf-archive.koAE1D/src /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -B -m pytest -q /tmp/icgs-generation-hf-archive.koAE1D/tests/test_generation_archive.py` — **PASS, 15 passed**.
- Required paired command: the same interpreter and `PYTHONPATH` with `test_generation_config.py test_generation_archive.py` — **PASS, 63 passed**.
- `git diff --check` — **PASS**, no output.
- `python3 -B scripts/validate_fast.py` — **PASS, 22 L0 tests**, local links and static boundaries passed; two existing `SyntaxWarning` notices in `tests/test_task_router.py` were emitted.
- Archive core uses bounded per-NPZ chunks, on-disk `.npy` spool, a 256 MiB uncompressed chunk cap, `np.load(..., allow_pickle=False)`, explicit bit-packed online validity, streamed archive validation, and LRU chunk caching. Materializer/handoff integration is not yet included in this evidence.

Archive boundary review hardening (writer/reader/validator only; worker/materializer integration is intentionally held pending scoped re-review):

- Alias identity now includes semantic role and logical range layout. Only exact-role arrays alias automatically, except the explicit byte-identical `raw_arrays/measured_points` → `online_observations/points` case; aliases record semantic role and target pieces/ranges.
- Attempt metadata and debug strings are recursively bounded/redacted before any attempt copies are written. Prefix counts are inferred before encoding; action/observation arrays chunk against their respective counts, and ragged points require offsets.
- Manifest validation now checks array references and their semantic roles, piece range continuity/coverage, point offsets and packed masks, required T/T+1 arrays, optional state/label references and boundary alignment, and attempt prefix counts/identity. It rejects truncated chunks and checks NPZ ZIP member sizes before decompression in both validation and read paths.
- `max_chunk_bytes` is defined against the sum of uncompressed `.npy` ZIP member sizes, including headers; writer preflight accounts for those headers.
- Targeted RED/GREEN evidence was collected for attempt-token and sensitive-key redaction, cross-role rho/point aliasing, differing logical ranges, long attempt prefixes, missing arrays/references, malformed ranges/masks, and bounded streaming validation. Review-round-2 fixes also require a measured-prefix reference for non-null `valid_observation_until` and compare the source attempt value to the timeline. The current suite is **PASS: 88 paired config/archive tests (40 archive tests)** on CPython 3.11.15 / NumPy 1.26.4; `python3 -B scripts/validate_fast.py` is **PASS: 22 L0 tests**, with two pre-existing `SyntaxWarning`s in `tests/test_task_router.py`; `git diff --check` is **PASS**. VPS, HF, simulator, training, and L1–L4 gates are **NOT RUN** by scope. Task 1.3 integration remains pending scoped re-review.

Task 2.2 local evidence (2026-09-25; `/tmp/icgs-generation-hf-archive.koAE1D`; venv CPython 3.11.15, NumPy 1.26.4, pytest 8.4.2):

- RED/GREEN: receipt-only queue tests first failed because `mark_published` did not accept a retention mode; validation-profile tests first failed because `receipt_only` was accepted in validation mode; remote-verification tests first failed because downloads had no owned `cache_dir`; pruned-manifest recovery first failed because it required local payload. Each now passes against the real queue/publisher/coordinator with only HF calls replaced by local fakes.
- `PYTHONPATH=src /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -B -m pytest -q tests/test_generation_publication.py tests/test_generation_queue.py tests/test_generation_config.py tests/test_generation_worker.py` — **PASS, 118 passed**.
- `PYTHONPATH=src /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -B -m pytest -q tests/test_generation*.py tests/test_capacity_probe.py` — **PASS, 351 passed**.
- `python3 -B scripts/validate_fast.py` — **PASS, 22 L0 tests**; interpreter was CPython 3.14.4 and emitted two existing invalid-escape `SyntaxWarning`s from `tests/test_task_router.py`.
- `git diff --check` — **PASS**, no output. HF network/API, simulator, preprocessing workload, training, robot motion, and C1–C5 live published-checkpoint gates — **NOT RUN**; tests used local fake download/API seams only.
- Residual scope: full remote-manifest resume validation is Task 2.3; remote HF acceptance and production run authorization remain later gates. Local fixtures do not establish a live HF repo’s download/commit behavior.

Task 2.2 reviewer-fix verification (2026-09-25; same worktree/environment; these
counts supersede the earlier 351-test generation-suite run above):

- RED/GREEN: the pinned verifier test first failed because it omitted resume/view/publication receipt files; it now checks archive artifacts, dataset manifest, resume receipt, publication receipt snapshot and each of the four views at one OID, using one fresh temporary cache per download.
- RED/GREEN: queue tests first accepted a missing outside-run result path when a per-job receipt already existed and accepted mutated planner-facing row fields. Receipt-only transitions now validate result-root containment even for already-pruned payloads; row reconstruction checks plan-derived split/subset/kind/scene/asset/intervention identities plus a canonical manifest-row digest.
- RED/GREEN: interrupted data-commit and receipt-commit tests simulate an accepted remote commit followed by local OID-write interruption. Recovery reuses the recorded dataset revision or exact receipt snapshot revision; retry does not reupload archive payloads.
- `PYTHONPATH=src /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -B -m pytest -q tests/test_generation_publication.py` — **PASS, 43 passed**; queue — **PASS, 39 passed**; control — **PASS, 45 passed**.
- `PYTHONPATH=src /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -B -m pytest -q tests/test_generation*.py tests/test_capacity_probe.py` — **PASS, 382 passed**.
- `PYTHONPATH=src /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -q` — **721 passed, 10 skipped, 0 failures/errors**; not a complete full-suite PASS. Skips: four original-source differential tests (no `IP_LEGACY_SOURCE_ROOT`), RLBench unavailable, CUDA RNG and CUDA-resume fixture not available/implemented, published-checkpoint integration opt-in, and two Lightning tests unavailable.
- `python3 -B scripts/validate_fast.py` — **PASS, 22 L0 tests**, system CPython 3.14.4; two pre-existing invalid-escape `SyntaxWarning`s in `tests/test_task_router.py`.
- `git diff --check` — **PASS**. Live HF/API calls, simulator, preprocessing workloads, training, robot motion, and C1–C5 — **NOT RUN**. All remote behavior in tests used local fakes.
- Legacy non-archive receipt retry behavior is intentionally unchanged; if verification fails after a known legacy receipt commit, that path may recommit the receipt. The new archive profile retries verification at the pinned known OID without reuploading archive payloads.

Task 1.3 local integration evidence: both materializers route through `EpisodeArchiveWriter` only when `ICGS_GENERATION_ARCHIVE_PROFILE` is present; legacy configs preserve their prior JSON/layout paths. Archive-profile success/valid-failure episodes use `episode.manifest.json`, crash/invalid results use `attempt.manifest.json`, and worker result detection rejects legacy JSON in archive mode. Optional RGB/depth/mask/joint/object modalities are omitted when unavailable. Focused generation integration tests are recorded in the implementation report; distributed archive validation/publication/resume/view work remains pending.

At plan creation time:

- Git state: clean `main` checkout.
- Focused existing generation/layout/publication tests: **PASS — 41 passed**.
- `python3 -B scripts/validate_fast.py`: **PASS — 22 L0 tests**.
- New archive/view/migration implementation: **NOT RUN — not implemented**.
- L1 model execution, L2/C1–C5, L3 simulator, L4 benchmark, full collection, and training: **NOT RUN**.

Move this plan to `docs/plans/completed/` only after the new archive has passed bounded remote acceptance and all required evidence/remaining risks are recorded.
