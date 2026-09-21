import copy
import json
from pathlib import Path

from icgs.data.collection.approved_manifest import validate_approved_manifest


MANIFEST = Path("artifacts/composition/approved_composition_manifest.json")


def base_manifest():
    return json.loads(MANIFEST.read_text())


def test_rejects_duplicate_program_id():
    data = base_manifest()
    data["catalog"].append(copy.deepcopy(data["catalog"][0]))
    errors = validate_approved_manifest(data)
    assert any("duplicate program_id" in error for error in errors)


def test_rejects_wrong_ordered_steps():
    data = base_manifest()
    data["catalog"][0]["ordered_steps"] = ["invented action"]
    errors = validate_approved_manifest(data)
    assert any("ordered_steps" in error for error in errors)


def test_rejects_generation_certified_development_entry():
    data = base_manifest()
    data["catalog"][20]["execution_status"] = "generation_certified"
    errors = validate_approved_manifest(data)
    assert any("generation_certified" in error for error in errors)


def test_rejects_bad_randomization_ranges():
    data = base_manifest()
    data["catalog"][0]["randomization"]["yaw_deg"] = [30, -30]
    errors = validate_approved_manifest(data)
    assert any("yaw_deg" in error for error in errors)
