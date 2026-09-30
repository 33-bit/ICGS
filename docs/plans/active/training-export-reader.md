# Training Export and Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Publish and read a compact, revision-bound `training/` index over the
lossless HF generation archive without duplicating numeric chunks.

**Architecture:** Reuse final role/view snapshots as the authoritative sample
selection. A new export layer binds those snapshots to one pinned archive
revision; a lazy reader downloads only the manifest/chunk needed by a sample and
delegates decoding/preprocessing to the existing archive reader.

**Tech Stack:** Python 3.10–3.12, dataclasses, JSON/SHA256, NumPy, existing
`EpisodeArchiveReader`, `huggingface-hub`.

**Spec:** `docs/superpowers/specs/2026-09-29-training-export-reader-design.md`

## Global Constraints

- HF target prefix is exactly `training`.
- The source archive and final views remain immutable and are never rewritten.
- Export files contain references and metadata only; raw numeric chunks stay at
  the source archive prefix.
- Reader validates source revision, manifest/view hashes, archive identity and
  preprocessing identity before returning data.
- No training, simulator, full generation or HF call runs during implementation.
- Preserve unrelated dirty worker/compiler/expert changes in the shared checkout.

### Task 1: Define export manifest and local materialization

**Files:**
- Create: `src/icgs/data/datasets/training_export.py`
- Test: `tests/test_training_export.py`

**Interfaces:**
- `TRAINING_EXPORT_SCHEMA = "icgs_training_export_v1"`
- `TRAINING_PREFIX = "training"`
- `TrainingExportReceipt`
- `export_training_views(view_root, *, repo_id, source_revision, views_revision, source_prefix, output_dir)`
- `TrainingExportReader.from_local(path, ...)`

- [x] Write tests for exact layout, source binding, view hashes, causal windows,
  source membership, corruption, bounded-cache and no copied chunk files.
- [ ] Run the focused test file and confirm the missing API failure (**NOT RUN by
  explicit user instruction**).
- [x] Implement canonical JSON export and immutable/idempotent writes.
- [ ] Run the focused tests and confirm they pass (**NOT RUN by explicit user
  instruction**).

### Task 2: Add pinned lazy HF reader

**Files:**
- Modify: `src/icgs/data/datasets/training_export.py`
- Test: `tests/test_training_export.py`

**Interfaces:**
- `TrainingExportReader.from_hf(repo_id, revision, prefix="training", cache_dir=..., cache_bytes=...)`
- `.sample_refs(view, role="train")`
- `.reader_for(ref)`
- `.preprocess_observation(ref)`

- [x] Add fake-HF tests for pinned manifest/view download and chunk hash
  verification.
- [ ] Run them RED (**NOT RUN by explicit user instruction**).
- [x] Implement the bounded revision-scoped on-demand file store and
  archive-reader bridge.
- [ ] Run them GREEN, including eviction and source-revision mismatch cases
  (**NOT RUN by explicit user instruction**).

### Task 3: Add the operator CLI

**Files:**
- Create: `scripts/generation_training_export.py`
- Test: `tests/test_training_export.py`

**Interfaces:**
- `--repo-id`
- `--source-prefix`
- `--source-revision`
- `--views-prefix`
- `--views-revision`
- `--output-prefix training`
- `--cache-dir`

- [x] Implement parser/validation for the exact unsuffixed target prefix and
  disjoint source/target prefixes.
- [x] Implement local export assembly plus HF commit/byte verification using the
  existing credential boundary.
- [ ] Run parser and fake-client tests (**NOT RUN by explicit user instruction**);
  leave the live validation command for the next user-approved step.

### Task 4: Document the consumer boundary

**Files:**
- Modify: `docs/components/generation.md`
- Modify: `docs/components/generation-artifacts.md`
- Modify: `tests/README.md`

- [x] Document `training/manifest.json`, lazy reader usage, and the exact live
  validation command boundary.
- [x] State clearly that native `icgs train` still requires separate integration.
- [ ] Run documentation/static checks (**NOT RUN by explicit user instruction**).

## Final evidence

Implementation stops before tests and live HF validation per the user's explicit
instruction. The next turn must run the focused RED/GREEN suite, L0, and the
real validation archive read/export/readback before claiming completion.
