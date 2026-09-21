import importlib.util
from pathlib import Path

import pytest


def load_generator():
    path = Path("scripts/colab_g2_dataset_generator.py")
    spec = importlib.util.spec_from_file_location("icgs_g2_generator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generator_blocks_planned_program_before_simulator_launch():
    generator = load_generator()
    import json
    import tempfile
    data = json.loads(Path("artifacts/composition/approved_composition_manifest.json").read_text())
    for row in data["catalog"]:
        if row["program_id"] == "T01":
            row["execution_status"] = "planned"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "manifest.json"
        path.write_text(json.dumps(data))
        with pytest.raises(RuntimeError, match="planned programs"):
            generator.load_approved_generation_manifest(path, ["T01"])


def test_generator_rejects_unknown_program_before_simulator_launch():
    generator = load_generator()
    with pytest.raises(ValueError, match="absent"):
        generator.load_approved_generation_manifest(
            "artifacts/composition/approved_composition_manifest.json",
            ["T99"],
        )


def test_generator_allows_planned_program_only_for_explicit_smoke():
    generator = load_generator()
    manifest, digest = generator.load_approved_generation_manifest(
        "artifacts/composition/approved_composition_manifest.json",
        ["T01"],
        allow_planned=True,
    )
    assert manifest["protocol_id"] == "icgs-composition-primary-v1"
    assert len(digest) == 64


def test_generator_v3_protocol_rejects_v1_manifest():
    generator = load_generator()
    with pytest.raises(ValueError, match="v3 composition manifest"):
        generator.load_approved_generation_manifest(
            "artifacts/composition/approved_composition_manifest.json",
            ["T01"],
            protocol="v3",
        )


def test_generator_v3_rejects_allow_planned():
    generator = load_generator()
    with pytest.raises(RuntimeError, match="cannot use --allow-planned"):
        generator.load_approved_generation_manifest(
            "artifacts/composition/approved_composition_manifest_v3.json",
            ["T01"],
            protocol="v3",
            allow_planned=True,
        )


def test_generator_v3_protocol_accepts_v3_manifest():
    generator = load_generator()
    manifest, digest = generator.load_approved_generation_manifest(
        "artifacts/composition/approved_composition_manifest_v3.json",
        ["T01"],
        protocol="v3",
    )
    assert manifest["composition_protocol_id"] == "icgs-composition-primary-v2"
    assert manifest["manifest_version"] == 3
    assert len(digest) == 64
