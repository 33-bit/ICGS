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

- [ ] Re-read ADR0015 and list its required invariants in the implementation issue/branch record.
- [ ] Add the ADR and active plan to the documentation map.
- [ ] Update generation lifecycle documentation to distinguish materialization-time provisional pointers from quota-complete final view snapshots.
- [ ] Update artifact inventory to mark v2 artifacts as legacy and document the new binary inventory; explicitly state that valid failures remain full episodes.
- [ ] Run `git diff --check` and the documentation-link portion of `python3 -B scripts/validate_fast.py`.

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

- [ ] Write rejection tests for unknown fields, nonpositive chunk sizes, unsupported archive IDs, invalid retention, and `retain_full_cloud=False` when the profile claims HF source-of-truth raw retention.
- [ ] Write acceptance tests for the compact lossless profile and bounded validation profile.
- [ ] Add the profile to the no-secret runtime example without changing the existing legacy profile defaults.
- [ ] Run `PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation_config.py tests/test_generation_archive.py`.

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

- [ ] Define the manifest identity fields: archive format, episode schema, dataset identity, episode/attempt IDs, program/split/subset, outcome, source run, code revision, preprocessing identity, and chunk inventory.
- [ ] Define array specs with name, dtype, shape, semantic role, chunk reference, byte count, and SHA256; reject object/pickle arrays.
- [ ] Define required arrays for valid episodes: online points/offsets/validity, poses, grips, commands, command grips, `dt`, substeps, robot/object state references, and task-label arrays when present.
- [ ] Define attempt archive requirements for crash/invalid records, including measured prefix arrays and explicit null episode identity.
- [ ] Define alias entries for byte-identical online/measured arrays so deduplication is explicit and verifiable.
- [ ] Add fixtures for success, valid failure, simulator crash, invalid observation, missing optional modality, and distinct online/measured arrays.
- [ ] Add RED tests for manifest identity mismatch, offsets that do not end at point count, nonmonotonic offsets, missing `T+1/T/T` arrays, object dtype, pickle loading, and alias target mismatch.
- [ ] Run the archive test file and record the expected failures before implementation.

### Task 1.2: Implement bounded chunk writing and lossless reading

**Files:**
- Modify: `src/icgs/data/collection/generation/episode_archive.py`
- Test: `tests/test_generation_archive.py`

**Interfaces:**
- `EpisodeArchiveWriter.write_episode(record, *, raw_arrays, debug_metadata, output_dir) -> ArchiveManifest`
- `EpisodeArchiveWriter.write_attempt(attempt, *, prefix_arrays, debug_metadata, output_dir) -> ArchiveManifest`
- `EpisodeArchiveReader.to_episode_record() -> dict[str, Any]` for test/compatibility reconstruction only.

- [ ] Serialize one bounded NPZ chunk at a time with `np.savez_compressed` and `allow_pickle=False`; never buffer an entire dataset.
- [ ] Preserve float32/float64 dtypes exactly for the first migration; do not quantize or convert matrices silently.
- [ ] Store ragged cloud values plus zero-based offsets and pack boolean validity masks without changing semantic values.
- [ ] Write each chunk to a temporary sibling, fsync/rename it, calculate SHA256/bytes, then write the manifest last through an atomic rename.
- [ ] Store only compact scalar/string debug metadata in `debug.json`; route all numeric arrays through chunk inventory.
- [ ] Reconstruct a v2-compatible in-memory record in tests and run the existing episode validator to prove timeline/provenance parity.
- [ ] Add corruption tests for altered chunk bytes, altered manifest hash, truncated chunk, path traversal, symlink, missing alias target, and partial manifest.
- [ ] Run `PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation_archive.py` and require all selected tests to execute.

### Task 1.3: Converge both materialization paths on the canonical writer

**Files:**
- Modify: `scripts/generation_episode_worker.py:56-293`
- Modify: `src/icgs/data/collection/generation/rlbench_attempt.py:247-351`
- Test: `tests/test_generation_attempt.py`, `tests/test_generation_archive.py`

**Interfaces:**
- Both paths call `EpisodeArchiveWriter.write_episode` or `write_attempt`; neither path writes dense point arrays to `episode.json`.

- [ ] Replace the distributed writer’s dense `episode.json`, duplicate layout arrays, duplicate telemetry arrays, and array-bearing `execution.json` with one archive manifest, chunk files, and compact debug metadata.
- [ ] Preserve all currently captured debug fields, including plan, routine, predicate distances, sensor-randomization receipt, object/robot states, errors and tracebacks.
- [ ] Keep valid failures on the episode path and crash/invalid results on the attempt path.
- [ ] Keep unavailable RGB/depth/mask/joint/object modalities omitted unless they are actually captured; never synthesize them.
- [ ] Ensure per-episode provisional pointers contain only archive references and do not copy observations.
- [ ] Add tests that assert no dense numeric arrays occur in JSON manifests/debug metadata and that valid failures have the same archive completeness as successes.
- [ ] Run focused generation tests plus `git diff --check`.

## Phase 2 — Validation, publication, and HF-first retention

### Task 2.1: Validate the new inventory before ingestion

**Files:**
- Modify: `src/icgs/data/collection/generation/distributed_validation.py`
- Test: `tests/test_generation_validation.py`, `tests/test_generation_archive.py`

**Interfaces:**
- `validate_closed_result()` accepts the new archive profile and returns an immutable `ValidatedResult` whose `file_sha256` covers every chunk/manifest/debug file.
- `validate_archive_manifest()` is called before ingestion and before publication.

- [ ] Require the new archive manifest and reject a v2 dense-JSON result under the new production profile.
- [ ] Validate archive identity against `GenerationJob`, including episode/attempt/program/outcome/plan identity.
- [ ] Validate all timeline cardinalities, array schemas, chunk hashes, aliases, optional-modality declarations, and provenance fields.
- [ ] Keep simulator crashes and invalid observations out of episode manifests and training view inputs.
- [ ] Add tests for valid success, valid failure, crash, invalid observation, malformed chunk inventory, dense JSON regression, and immutable re-ingestion.
- [ ] Run `PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation_validation.py tests/test_generation_archive.py`.

### Task 2.2: Publish complete HF archives and finalize local retention

**Files:**
- Modify: `src/icgs/data/collection/generation/distributed_publication.py`
- Modify: `src/icgs/data/collection/generation/distributed_queue.py`
- Modify: `scripts/generation_coordinator.py`, `scripts/generation_launch.py`
- Test: `tests/test_generation_publication.py`, `tests/test_generation_queue.py`, `tests/test_generation_worker.py`

**Interfaces:**
- `HuggingFaceBatchPublisher` publishes all canonical chunks/manifests/debug metadata and records the remote revision.
- `FilesystemJobQueue.mark_published(job_id, *, retention: Literal["keep", "receipt_only"]) -> Path` retains receipts/hashes and optionally prunes only after verified remote publication.

- [ ] Upload the complete archive inventory for each success/valid-failure episode and complete attempt inventory for crash/invalid results.
- [ ] Keep HF `dataset_manifest.json`, `resume_receipt.json`, publication receipt, per-episode manifests, and artifact hashes consistent with the new archive identity.
- [ ] Mark prefix views as `PROVISIONAL` until finalization; never label cumulative batch IDs as final training samples.
- [ ] Add an explicit receipt-only retention path that refuses to prune before remote hash verification and keeps local queue receipts, remote commit OID, artifact hashes, and source run identity.
- [ ] Add idempotency tests for retry after remote commit, remote verification failure, local result already pruned, duplicate publication, and resume from remote manifest with no local episode payload.
- [ ] Keep default bounded validation in `keep` mode; enable `receipt_only` only in the new production archive profile.
- [ ] Run focused publication/queue/worker tests without network or HF credentials.

### Task 2.3: Extend resume bootstrap for HF-only recovery

**Files:**
- Modify: `src/icgs/data/collection/generation/distributed_planner.py`
- Modify: `scripts/generation_coordinator.py`, `scripts/generation_launch.py`
- Test: `tests/test_generation_control.py`, `tests/test_generation_planner.py`, `tests/test_generation_publication.py`

**Interfaces:**
- `load_remote_manifest()` validates the new dataset/archive identity and returns immutable episode/attempt rows plus archive manifest references.
- `DistributedPlanner.from_manifest()` restores uniqueness/quota state without local episode binaries.

- [ ] Validate new archive/schema identity, source run disjointness, episode/attempt uniqueness, plan identity, split/subset, and per-episode manifest references.
- [ ] Prove resume can reconstruct planner state after local `result_dir` payloads are absent but HF manifest/receipts are available.
- [ ] Preserve valid failures and failure attempts during resume; never reset quota because a local payload was pruned.
- [ ] Add tests for malformed remote archive manifest, conflicting immutable row, missing chunk reference, matching remote hash, and disjoint new run ID.
- [ ] Run the resume/control test subset and L0.

## Phase 3 — Lazy HF archive reader and training views

### Task 3.1: Implement lazy archive access

**Files:**
- Create: `src/icgs/data/datasets/generation_archive.py`
- Modify: `src/icgs/data/datasets/generation_views.py`
- Modify: `src/icgs/data/datasets/episodes.py`
- Test: `tests/test_generation_archive.py`, `tests/test_generation_views.py`

**Interfaces:**
- `ArchiveDatasetIndex.from_manifest(dataset_manifest_path) -> ArchiveDatasetIndex`
- `ArchiveDatasetIndex.sample_refs(view: str, role: str) -> Iterator[SampleRef]`
- `EpisodeArchiveReader.observation(boundary)`, `.transition(index)`, and `.task_labels(boundary)` provide lazy values.
- `build_generation_view(records_or_index, view, *, role="train", mix=None)` supports archive-backed references without loading all episodes.

- [ ] Index manifests and sample references without loading full point arrays.
- [ ] Add bounded LRU chunk caching with explicit cache byte cap and deterministic cache keys including archive/preprocessing identity.
- [ ] Make `D_geom`, `D_temporal`, `D_dyn`, and `D_task` use archive references; preserve causal histories and intervention flags.
- [ ] Derive the 2,048-point training input lazily with the existing preprocessing identity; record the resolved preprocessing hash in the training run metadata.
- [ ] Keep `adapter_a0`/`adapter_a1` semantics and native PyG loaders separate from the new archive reader.
- [ ] Test random boundary access, cross-chunk windows, cache eviction, missing optional modalities, causal masking, and valid-failure inclusion.
- [ ] Run the archive/view tests with tiny fixtures only.

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

- [ ] Generate per-episode provisional pointers with `role="all"` and `mix=False`; mark them `PROVISIONAL`.
- [ ] Build final role-specific snapshots from the complete HF dataset manifest: train=`train_core`, validation=`train_val`, evaluation=`development`+`test`.
- [ ] Apply the 70/30 nominal/perturbed selection only to final `D_temporal` and `D_dyn` snapshots using the recorded collection seed/mixture seed.
- [ ] Exclude crash/invalid attempts; retain valid failures when their split/subset is eligible.
- [ ] Write view manifests with source revision/SHA256, archive identity, preprocessing identity, role, mixture, sample counts, and finalization timestamp.
- [ ] Make finalization idempotent: same source manifest/seed produces the same bytes; conflicting source revision fails closed.
- [ ] Test partial/provisional versus final status, changing dataset snapshot, 70/30 counts, split leakage, valid-failure retention, and deterministic re-run.
- [ ] Run the view-finalization test subset; do not contact HF.

## Phase 4 — Migration tooling and bounded capacity evidence

### Task 4.1: Convert existing v2 artifacts without rewriting them

**Files:**
- Create: `scripts/generation_archive_migrate.py`
- Test: `tests/test_generation_archive_migration.py`
- Modify: `docs/components/generation-artifacts.md`

**Interfaces:**
- `migrate_episode(source_reader, target_writer, *, source_identity, target_profile) -> MigrationReceipt`
- `MigrationReceipt` records source artifact hashes, target archive hashes, field/timeline counts, and conversion warnings.

- [ ] Read one v2 episode/attempt at a time from an explicitly supplied source revision/prefix; never scan arbitrary HF paths.
- [ ] Convert dense JSON arrays to lossless chunks and preserve provenance/debug fields.
- [ ] Reject incomplete/malformed source artifacts rather than silently repairing them; write a migration failure receipt.
- [ ] Verify reconstructed v2 semantics against the source before publishing the new target artifact.
- [ ] Publish converted artifacts only to an explicitly supplied new prefix; never delete or overwrite v2 data.
- [ ] Add tests for success, valid failure, crash attempt, invalid observation, malformed source, hash mismatch, and idempotent re-run.
- [ ] Run migration tests against tiny local fixtures only; a remote migration job remains separately authorized.

### Task 4.2: Add size instrumentation to the bounded capacity probe

**Files:**
- Modify: `scripts/generation_capacity_probe.py`
- Modify: `src/icgs/data/collection/generation/capacity_probe.py`
- Test: `tests/test_capacity_probe.py`
- Modify: `docs/experiments/generation-validation/capacity-probe-tpu-v6e1-20260924.md`

**Interfaces:**
- `capacity_receipt.json` adds per-result `bytes_by_category`, `boundaries`, `raw_points`, `archive_profile`, and `local_peak_bytes`.

- [ ] Categorize bytes into archive chunks, manifests, debug metadata, views, and receipts.
- [ ] Record point counts and boundary counts without loading all prior episodes.
- [ ] Enforce a bounded per-result and total staging cap before launching a probe stage.
- [ ] Add tests for byte-cap rejection, category accounting, compact-profile validation, and no-worker preflight behavior.
- [ ] Run only the configuration/preflight tests locally; live capacity measurement remains explicit.

## Phase 5 — Documentation and acceptance closure

### Task 5.1: Update operational and validation documentation

**Files:**
- Modify: `docs/components/generation.md`
- Modify: `docs/components/generation-artifacts.md`
- Modify: `docs/audits/2026-09-20-generation-contract.md`
- Modify: `docs/plans/active/resumable-multihost-generation.md`
- Modify: `tests/README.md`
- Modify: `docs/README.md`

- [ ] Document the new HF archive tree, SOT rule, archive/schema identities, valid-failure retention, provisional/final view lifecycle, and receipt-only local retention.
- [ ] Document exact clean-machine HF reader/resume commands without embedding credentials.
- [ ] Document that final view snapshots bind to an HF revision and are not regenerated from remembered local state.
- [ ] Add validation commands for archive roundtrip, view finalization, remote-resume contract fixtures, and size budgets.
- [ ] Record the current v2 artifact format as legacy historical evidence and preserve links to old manifests.
- [ ] Run link validation and `git diff --check`.

### Task 5.2: Local acceptance gates

**Files:**
- Test: all focused generation/archive/view tests; no runtime code changes in this task.

- [ ] Run `PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation*.py tests/test_generation_archive.py tests/test_generation_view_finalization.py tests/test_generation_archive_migration.py`.
- [ ] Run `python3 -B scripts/validate_fast.py`.
- [ ] Confirm no dense observation arrays exist in new JSON manifests/debug files.
- [ ] Confirm valid failures have complete binary archives and appear in eligible view snapshots.
- [ ] Confirm local receipt-only retention tests prune only after verified remote identity.
- [ ] Report L1/L2/L3/L4, simulator, HF publication, full quota, and training as `NOT RUN` unless separately authorized.

### Task 5.3: Bounded remote acceptance

**Files:**
- Evidence: `docs/experiments/generation-validation/<new-archive-acceptance>/`

- [ ] Start from a fresh committed revision and a new isolated HF validation prefix.
- [ ] Run a bounded set containing success, valid failure, simulator-crash/invalid-attempt, publication, resume, and workers-only attach cases.
- [ ] Verify every remote chunk/manifest/debug hash from a clean machine with no local result payloads.
- [ ] Finalize views against a pinned remote dataset-manifest revision and verify deterministic sample counts/mixtures.
- [ ] Record bytes per episode, peak local staging bytes, upload throughput, publication commits, resume revision/SHA, and all PASS/FAIL/SKIPPED/NOT RUN states.
- [ ] Do not authorize the 7,520-attempt production run until archive quality, HF-only recovery, final views, and storage budget gates pass.

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

- [x] Phase 0.2 — Added an opt-in immutable archive profile. Legacy runtime configs still parse with `archive_profile=None`; the checked-in bounded validation example selects the archive format with `keep` retention. Profiles also bound each uncompressed NPZ chunk with `max_chunk_bytes` (default 256 MiB); the writer streams `.npy` scratch pieces and records a local staging byte upper bound that includes spool plus final files.
- [x] Recorded implementation rulings: `numpy.savez_compressed` has no `allow_pickle` argument, so archive writes reject object arrays and readers use `numpy.load(..., allow_pickle=False)`; dated audits remain historical evidence and current storage behavior belongs in current owner docs.
- [ ] Phase 1 — Canonical archive writer/reader and both materialization paths. Archive core is implemented and its focused tests pass; worker/materializer and distributed-validation integration remain open.
- [ ] Phase 2 — New-profile validation, complete HF publication, receipt-only retention, and HF-only resume.
- [ ] Phase 3 — Lazy archive readers and revision-bound provisional/final views.
- [ ] Phase 4 — Local migration and capacity instrumentation.
- [ ] Phase 5 — Current owner documentation and local acceptance.

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

At plan creation time:

- Git state: clean `main` checkout.
- Focused existing generation/layout/publication tests: **PASS — 41 passed**.
- `python3 -B scripts/validate_fast.py`: **PASS — 22 L0 tests**.
- New archive/view/migration implementation: **NOT RUN — not implemented**.
- L1 model execution, L2/C1–C5, L3 simulator, L4 benchmark, full collection, and training: **NOT RUN**.

Move this plan to `docs/plans/completed/` only after the new archive has passed bounded remote acceptance and all required evidence/remaining risks are recorded.
