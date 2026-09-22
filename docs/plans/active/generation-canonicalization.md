# Generation Canonicalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Make `icgs.data.collection.generation` and semantic `generation_*` entry points the only supported data-generation runtime.

**Architecture:** Move the live execution/materialization seam out of the compatibility collector into the canonical generation package, rename the package and operational scripts, then remove obsolete branches and generated one-off tooling. Preserve serialized payload schema values, checkpoints, manifests and historical provenance.

**Tech Stack:** Python 3.10–3.12, setuptools package under `src/`, RLBench/Colab operational scripts, pytest/unittest, Markdown decision records.

**Spec:** `docs/superpowers/specs/2026-09-22-primary-generation-canonicalization-design.md`

## Global Constraints

- Canonical runtime remains under `src/icgs`.
- No worker/coordinator import may reference a legacy collector or obsolete generation suffix.
- Serialized episode payload compatibility remains unchanged; only source names and Python APIs are renamed.
- Do not delete checkpoints, manifests, episodes or historical baseline evidence.
- Do not launch simulator, training, or full generation as validation.
- Keep valid failures and failure attempts in the generation contract.

### Task 1: Rename the generation package and protocol symbols

**Files:**
- Rename: `src/icgs/data/collection/v3/` → `src/icgs/data/collection/generation/`
- Modify: all imports under `src/icgs`, `scripts`, and generation tests
- Test: `tests/test_v3_batch.py`, `tests/test_v3_distributed_*.py`, `tests/test_v3_rlbench_attempt.py`

**Interfaces:**
- Produce `icgs.data.collection.generation.GENERATION_PROTOCOL` and `GenerationProtocol`.
- Preserve the payload schema value and all protocol field values.

- [ ] Rename package files and update package-data configuration.
- [ ] Rename `V3_PROTOCOL`/`V3Protocol` and update all call sites.
- [ ] Rename semantic functions ending `_v2` in the generation package without changing record output.
- [ ] Run focused collection and distributed tests.
- [ ] Commit the package rename separately.

### Task 2: Extract the live execution seam from the compatibility collector

**Files:**
- Create: `src/icgs/data/collection/generation/execution.py`
- Modify: `scripts/colab_g2_dataset_generator.py`
- Modify: `scripts/colab_v3_distributed_worker.py` and its renamed successor
- Test: `tests/test_v3_rlbench_attempt.py`, `tests/test_colab_generator_manifest_gate.py`, `tests/test_batch_commit_protocol.py`

**Interfaces:**
- `generation.execution.load_task_specs`
- `generation.execution.execute_raw_attempt`
- `generation.execution.collect_single_attempt`

- [ ] Move only the functions required by the distributed worker and their direct helpers.
- [ ] Update tests and worker imports to the new module.
- [ ] Prove the compatibility collector is no longer imported by runtime generation.
- [ ] Commit the extraction before deleting the collector.

### Task 3: Rename operational entry points

**Files:**
- Rename worker/coordinator/watchdog/launch/episode-worker scripts to `generation_*.py`.
- Modify: launch command construction, docs and tests.
- Test: `tests/test_v3_distributed_control.py`, `tests/test_v3_distributed_worker.py`, `tests/test_v3_expert.py`.

- [ ] Rename scripts and update every command string.
- [ ] Keep one semantic launch/status surface; do not retain duplicate debug wrappers.
- [ ] Run operational script contract tests.
- [ ] Commit the entry-point rename.

### Task 4: Normalize manifests and documentation

**Files:**
- Rename: `artifacts/composition/approved_composition_manifest_v3.json` → `approved_composition_manifest.json`
- Modify: `docs/README.md`, generation audits, generation plans/specs, `artifacts/composition/README.md`, package data metadata.
- Archive/delete: obsolete v2 operational docs and sidecar instructions after reference scan.

- [ ] Replace operational `v2/v3/g2/primary` names with “generation” terminology.
- [ ] Mark historical records as provenance and preserve their original identifiers inside historical sections.
- [ ] Update links and run a Markdown link scan.
- [ ] Commit docs/artifact naming separately.

### Task 5: Remove obsolete and generated tooling

**Files:**
- Delete: untracked Colab debug/drain/recovery/alias/bulk scripts and receipts identified by the audit.
- Delete: tracked obsolete v1/v2/g2 generation branches after Tasks 1–3.
- Modify: tests that only protect removed legacy branches.

- [ ] Re-run `rg` for legacy generation imports and operational paths.
- [ ] Remove only files with no remaining runtime/test/documentation references.
- [ ] Keep baseline, checkpoint, model and historical evidence files.
- [ ] Commit cleanup separately.

### Task 6: Full verification and handoff

- [ ] Run `python3 -B scripts/validate_fast.py`.
- [ ] Run focused generation tests and report exact counts.
- [ ] Run `git diff --check` and a final reference scan.
- [ ] Record PASS/FAIL/SKIPPED/NOT RUN and remaining risk in this plan.
- [ ] Move this plan to `docs/plans/completed/` only if all required source/doc cleanup is complete.
