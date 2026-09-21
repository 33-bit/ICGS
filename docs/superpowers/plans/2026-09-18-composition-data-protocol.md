# Composition Data Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the approved 20/4/12 composition decision into one executable manifest and a cheap validator that gates every future episode.

**Architecture:** Keep the existing immutable program skeleton catalog as the source for IDs and ordered semantic steps. Add a declarative JSON manifest for executable bindings and a focused validator that checks cardinality, split closure, lineage/asset separation, seed banks, and pilot status without launching a simulator. Existing G2 pilot bindings remain compatible inputs; no broad runtime refactor is introduced.

**Tech Stack:** Python 3.10+, JSON, existing `icgs.data.collection.programs` and `bindings` modules, pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-composition-data-protocol.md`

## Global Constraints

- The primary catalog is exactly 20 train IDs `T01`–`T20`, 4 development IDs `V01`–`V04`, and 12 test IDs `P1`–`P4`, `G1`–`G4`, `R1`–`R4`.
- `E01` remains exploratory and is excluded from the primary cardinality.
- Split, asset family, source lineage, seed IDs, ranges, controller, and predicates are manifest inputs, never inferred from filenames or episode IDs.
- A planned program cannot publish measured episodes.
- No Colab simulator workload is part of manifest validation.
- Existing user/agent changes must be preserved.

### Task 1: Add the approved manifest fixture

**Files:**
- Create: `artifacts/composition/approved_composition_manifest.json`
- Create: `artifacts/composition/README.md`
- Test: `tests/test_approved_composition_manifest.py`

**Interfaces:**
- Consumes: `icgs.data.collection.programs.PROGRAM_CATALOG`.
- Produces: JSON object with `manifest_version`, `catalog`, `asset_split`, and one binding per primary program.

- [ ] **Step 1: Write the failing cardinality and field tests**

```python
def test_manifest_has_exact_primary_cardinality():
    manifest = load_approved_manifest()
    assert len(manifest["catalog"]) == 36
    assert {x["split"] for x in manifest["catalog"]} == {"train", "development", "test"}
    assert sum(x["split"] == "train" for x in manifest["catalog"]) == 20
    assert sum(x["split"] == "development" for x in manifest["catalog"]) == 4
    assert sum(x["split"] == "test" for x in manifest["catalog"]) == 12

def test_every_binding_has_required_protocol_fields():
    for entry in load_approved_manifest()["catalog"]:
        assert {"program_id", "split", "family", "scene_id", "asset_family_id",
                "asset_version", "source_lineage_id", "workspace", "randomization",
                "seed_demos", "execution_modes", "controller", "predicates",
                "execution_status"} <= entry.keys()
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_approved_composition_manifest.py`

Expected: FAIL because the approved manifest and loader do not exist.

- [ ] **Step 3: Write the manifest and minimal loader fixture**

Use the exact IDs and steps from `PROGRAM_CATALOG`. Give every entry five fixed seed IDs (`<program>-seed-0001` through `0005`), explicit workspace/range objects, `rlbench-timed-ik-v1`, predicate tolerance objects, and `execution_status` equal to `pilot_certified` only for `T06`, `T08`, `T09`, `T11`, `T13`, and `T14`; use `planned` for the remaining primary programs.

- [ ] **Step 4: Run the focused tests**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_approved_composition_manifest.py`

Expected: PASS.

### Task 2: Implement the pure manifest validator

**Files:**
- Create: `src/icgs/data/collection/approved_manifest.py`
- Test: `tests/test_approved_manifest_validator.py`

**Interfaces:**
- Consumes: `Path` to the approved JSON manifest.
- Produces: `ApprovedManifest` dataclass and `validate_approved_manifest(path) -> tuple[Diagnostic, ...]`; invalid manifests raise `ValueError` with program ID and field path.

- [ ] **Step 1: Write failing validator tests**

Cover duplicate IDs, wrong 20/4/12 counts, cross-split asset family, cross-split source lineage, fewer than five training seeds, missing required fields, and a planned entry marked generation-certified.

- [ ] **Step 2: Run the validator tests and confirm failure**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_approved_manifest_validator.py`

Expected: FAIL because the validator module is absent.

- [ ] **Step 3: Implement the smallest pure checks**

Load JSON, compare IDs/steps against `PROGRAM_CATALOG`, normalize development to the canonical `development` label, enforce split closure, enforce five seeds for training entries, and reject any test seed or asset family overlap with train/dev. Do not import RLBench or start external processes.

- [ ] **Step 4: Run focused validator tests**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_approved_manifest_validator.py tests/test_approved_composition_manifest.py`

Expected: PASS.

### Task 3: Connect episode provenance to the approved manifest

**Files:**
- Modify: `src/icgs/data/collection/runner.py`
- Modify: `src/icgs/data/datasets/episodes.py`
- Test: `tests/test_episode_provenance_manifest.py`

**Interfaces:**
- Consumes: approved manifest digest and episode provenance mapping.
- Produces: rejection of episodes whose program, split, lineage, seed, or execution status is not authorized by the approved manifest.

- [ ] **Step 1: Write failing provenance tests**

Test one valid pilot provenance, one unknown program, one planned program, one wrong split, and one seed not in the declared seed bank.

- [ ] **Step 2: Run the tests and confirm failure**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_episode_provenance_manifest.py`

Expected: FAIL on missing manifest gate.

- [ ] **Step 3: Add an opt-in manifest gate**

Require the gate when `approved_manifest_path` is supplied; preserve existing exploratory behavior when it is omitted. Reject before simulator execution and include the exact manifest field in the error.

- [ ] **Step 4: Run targeted regression tests**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_episode_provenance_manifest.py tests/test_collection_runner.py tests/test_program_bindings.py`

Expected: PASS.

### Task 4: Delegate only mechanical validation and pilot execution

**Files:**
- No protocol decision files may be changed by the delegated agent.
- Read-only audit: `scripts/colab_g2_dataset_generator.py`, `artifacts/g2-pilot/`, and approved manifest.

**Interfaces:**
- Consumes: the approved manifest and its SHA-256 digest.
- Produces: validator output, explicitly named pilot batch receipts, and HF readback receipts only.

- [ ] **Step 1: Ask `6c827e6` to run the manifest validator only**

The prompt must forbid changing IDs, split, assets, seeds, tolerances, or acceptance rules.

- [ ] **Step 2: If validator passes, authorize one named pilot batch**

Run only the six certified pilot IDs, with explicit episode count and start index. Do not launch planned programs.

- [ ] **Step 3: Audit receipts**

Check manifest digest, per-episode provenance, archive checksums, controller statuses, and HF commit/readback. Never count `SKIPPED` as `PASS`.

### Task 5: Repository validation and handoff

**Files:**
- Modify: `artifacts/composition/README.md` with commands and evidence.

- [ ] **Step 1: Run cheap validation**

Run: `python3 -B scripts/validate_fast.py` and
`PYTHONPATH=src python3 -m pytest -q tests/test_approved_composition_manifest.py tests/test_approved_manifest_validator.py tests/test_episode_provenance_manifest.py tests/test_collection_runner.py tests/test_program_bindings.py`.

- [ ] **Step 2: Record PASS/FAIL/SKIPPED explicitly**

The README must include command output summaries, simulator/HF coverage, and remaining risks. A successful local validator does not imply simulator generation success.

- [ ] **Step 3: Review the diff without cleaning unrelated work**

Run: `git status --short` and inspect only the files named by this plan before handoff.
