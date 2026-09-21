import copy
import json
from pathlib import Path

from icgs.data.collection.approved_manifest import (
    load_approved_manifest,
    validate_episode_provenance,
)


MANIFEST = Path("artifacts/composition/approved_composition_manifest.json")


def pilot_provenance():
    row = next(item for item in load_approved_manifest(MANIFEST)["catalog"] if item["program_id"] == "T06")
    return {
        "program_id": "T06",
        "split": row["split"],
        "source_lineage_id": row["source_lineage_id"],
        "asset_family_id": row["asset_family_id"],
        "seed_id": row["seed_demos"][0],
        "execution_mode": row["execution_modes"][0],
    }


def test_valid_pilot_provenance_passes():
    manifest = load_approved_manifest(MANIFEST)
    manifest["catalog"] = [
        {**row, "execution_status": "pilot_certified"}
        if row["program_id"] == "T06" else row
        for row in manifest["catalog"]
    ]
    assert validate_episode_provenance(manifest, pilot_provenance()) == ()


def test_unknown_program_is_rejected():
    provenance = pilot_provenance()
    provenance["program_id"] = "T99"
    errors = validate_episode_provenance(load_approved_manifest(MANIFEST), provenance)
    assert any("not in approved catalog" in error for error in errors)


def test_wrong_seed_is_rejected():
    provenance = pilot_provenance()
    provenance["seed_id"] = "unapproved-seed"
    errors = validate_episode_provenance(load_approved_manifest(MANIFEST), provenance)
    assert any("seed_id" in error for error in errors)


def test_planned_program_is_rejected():
    manifest = load_approved_manifest(MANIFEST)
    manifest["catalog"] = [
        {**row, "execution_status": "planned"}
        if row["program_id"] == "T01" else row
        for row in manifest["catalog"]
    ]
    provenance = pilot_provenance()
    provenance.update({
        "program_id": "T01",
        "source_lineage_id": next(item for item in manifest["catalog"] if item["program_id"] == "T01")["source_lineage_id"],
        "asset_family_id": next(item for item in manifest["catalog"] if item["program_id"] == "T01")["asset_family_id"],
        "seed_id": next(item for item in manifest["catalog"] if item["program_id"] == "T01")["seed_demos"][0],
        "execution_mode": next(item for item in manifest["catalog"] if item["program_id"] == "T01")["execution_modes"][0],
    })
    errors = validate_episode_provenance(manifest, provenance)
    assert any("planned program" in error for error in errors)
