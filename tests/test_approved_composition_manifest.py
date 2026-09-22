import json
from pathlib import Path

import pytest

from icgs.data.collection.approved_manifest import (
    load_approved_manifest,
    validate_approved_manifest,
)


MANIFEST = Path("artifacts/composition/approved_composition_manifest.json")


def test_manifest_has_exact_generation_cardinality():
    data = load_approved_manifest(MANIFEST)
    rows = data["catalog"]
    assert len(rows) == 36
    assert sum(row["split"] == "train" for row in rows) == 20
    assert sum(row["split"] == "development" for row in rows) == 4
    assert sum(row["split"] == "test" for row in rows) == 12


def test_manifest_is_valid_and_excludes_exploratory_track():
    data = load_approved_manifest(MANIFEST)
    assert validate_approved_manifest(data) == ()
    assert data["excluded_tracks"] == [{
        "program_id": "E01",
        "track": "exploratory",
        "reason": "not part of generation 20/4/12 cardinality",
    }]


def test_every_entry_has_required_protocol_fields():
    rows = load_approved_manifest(MANIFEST)["catalog"]
    required = {
        "program_id", "split", "family", "ordered_steps", "scene_id",
        "asset_family_id", "asset_version", "source_lineage_id", "workspace",
        "randomization", "seed_demos", "execution_modes", "controller",
        "predicates", "execution_status",
    }
    assert all(required <= row.keys() for row in rows)


def test_validator_rejects_cross_split_asset_family(tmp_path):
    data = json.loads(MANIFEST.read_text())
    data["catalog"][20]["asset_family_id"] = data["catalog"][0]["asset_family_id"]
    errors = validate_approved_manifest(data)
    assert any("asset_family_id" in error for error in errors)


def test_validator_rejects_short_training_seed_bank(tmp_path):
    data = json.loads(MANIFEST.read_text())
    data["catalog"][0]["seed_demos"] = ["only-one"]
    errors = validate_approved_manifest(data)
    assert any("seed_demos" in error for error in errors)
