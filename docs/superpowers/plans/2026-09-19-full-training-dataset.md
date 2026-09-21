# Full Training Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a lossless, versioned training/debug layout for successful and failed RLBench attempts.

**Architecture:** Preserve the existing archive as canonical and add a v2 sidecar layout plus root metadata. Collection captures optional simulator supervision; validation rejects misalignment and false exact-snapshot claims.

**Tech Stack:** Python, NumPy, RLBench/PyRep, JSON/NPZ, pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-full-training-dataset-design.md`

## Global Constraints

- Do not change native Instant Policy observation/action semantics.
- Failed attempts never increment successful episode targets.
- Missing modalities are omitted and declared unavailable.
- Snapshot quality defaults to `approximate_replay`.
- Test data remains evaluation-only.

---

### Task 1: Versioned dataset metadata and episode sidecars

**Files:**
- Create: `src/icgs/data/training_layout.py`
- Test: `tests/test_training_layout.py`

- [ ] Write failing tests for root metadata, aligned episode arrays, failure results, integrity inventory, and snapshot-quality validation.
- [ ] Run the focused test and confirm RED.
- [ ] Implement root initialization, episode writer, cache manifest writer, and validator.
- [ ] Run the focused test and confirm PASS.

### Task 2: RLBench supervision capture

**Files:**
- Modify: `scripts/colab_g2_dataset_generator.py`
- Test: `tests/test_g2_training_layout_capture.py`

- [ ] Write failing source/fixture tests for masks, depth, object state, collision events, labels, and shared success/failure envelopes.
- [ ] Run the focused test and confirm RED.
- [ ] Capture only available simulator fields and materialize the v2 sidecar.
- [ ] Run the focused test and confirm PASS.

### Task 3: Archive integration and validation

**Files:**
- Modify: `src/icgs/data/archives.py`
- Modify: `src/icgs/data/__init__.py`
- Test: `tests/test_training_layout.py`

- [ ] Write failing round-trip and corruption tests.
- [ ] Run the focused test and confirm RED.
- [ ] Attach the layout inventory to the canonical manifest without changing shard semantics.
- [ ] Run focused tests, L0, and the applicable data-collection tests.
