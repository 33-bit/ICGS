# Archive Publication Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make bounded HF publication concurrency explicit and resumable, then benchmark it before the full archive run.

**Architecture:** Add `publication_batch_size` and `publication_upload_threads` to the immutable runtime/run contracts. The coordinator constructs `PublicationConfig` from those values; workers remain credential-free and archive semantics remain unchanged.

**Tech Stack:** Python dataclasses, Hugging Face Hub commit API, pytest, existing filesystem queue and archive validation.

**Spec:** Approved chat design for archive-backed generation and v2-style training export.

## Global Constraints

- HF prefix for the production run is exactly `raw`.
- Archive profile remains lossless and keeps full clouds/debug metadata.
- Resume identity, pinned revisions, receipts, and worker/coordinator separation remain fail-closed.
- No full generation starts until focused tests and a bounded publication benchmark pass.

### Task 1: Runtime publication contract

**Files:**
- Modify: `src/icgs/data/collection/generation/distributed_contracts.py`
- Modify: `scripts/generation_launch.py`
- Modify: `scripts/generation_coordinator.py`
- Test: `tests/test_generation_config.py`
- Test: `tests/test_generation_control.py`

**Interfaces:**
- `RuntimeRunConfig.publication_batch_size: int = 1`
- `RuntimeRunConfig.publication_upload_threads: int = 1`
- `RunConfig` carries the same two immutable values.
- `HuggingFaceBatchPublisher` receives `PublicationConfig(batch_size=..., upload_threads=...)`.

- [ ] Add failing round-trip and run-receipt tests.
- [ ] Verify the tests fail because the fields are not accepted/persisted.
- [ ] Add strict validation and runtime/run binding.
- [ ] Verify focused tests pass.

### Task 2: Bounded benchmark and production config

**Files:**
- Add: `outputs/generation-raw-runtime.json` (ignored operational artifact)
- Use: existing `scripts/generation_launch.py`, `scripts/generation_coordinator.py`

- [ ] Run a bounded publication benchmark with batch/thread settings on the selected host.
- [ ] Keep the fastest passing configuration with no archive or receipt conflicts.
- [ ] Set production prefix to `raw`, resume enabled, and receipt-only retention.
- [ ] Verify preflight resume and launch receipts before starting the full run.

### Task 3: Training export/reader after archive completion

**Files:**
- Add: archive-to-v2 training export/materializer and reader modules.
- Modify: final-view integration only where required.
- Test: A0/A1/B sample loading, split/lineage, masks, normalization, and deterministic re-read.

- [ ] Run only after the production archive is frozen at a verified HF revision.
- [ ] Materialize compact v2-style metadata and binary arrays without duplicating canonical raw payloads in HF.
- [ ] Produce a training-readiness receipt for A0, A1, and B.
