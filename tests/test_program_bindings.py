from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from icgs.data.collection.bindings import load_binding_manifest


def _binding(program_id: str = "T06", split: str = "train") -> dict:
    return {
        "program_id": program_id,
        "split": split,
        "scene_id": "custom-scene-v1",
        "asset_family_id": "custom-pack-v1",
        "asset_version": "mesh-pack-2026-09-18",
        "source_lineage_root": f"seed-{program_id.lower()}",
        "workspace_bounds_m": [[-0.4, 0.4], [-0.4, 0.4], [0.75, 1.25]],
        "randomization": {
            "translation_bounds_m": [[-0.03, 0.03], [-0.03, 0.03], [0.0, 0.0]],
            "yaw_deg": [-30.0, 30.0],
            "scale": [0.8, 1.2],
            "camera_profile_id": "cam-v1",
            "lighting_profile_id": "light-v1",
        },
        "seed_ids": ["seed-0001", "seed-0002", "seed-0003", "seed-0004", "seed-0005"],
        "expert_id": "waypoint-expert-v1",
        "waypoint_protocol_id": "object-relative-waypoints-v1",
        "controller_protocol_id": "rlbench-timed-v1",
        "predicate_protocol_id": "predicate-v1",
        "predicate_tolerances": {"position_m": 0.01, "rotation_deg": 10.0},
        "calibration_id": "calib-v1",
    }


class ProgramBindingTests(unittest.TestCase):
    def _write(self, bindings: list[dict]) -> Path:
        temp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self.addCleanup(lambda: Path(temp.name).unlink(missing_ok=True))
        json.dump({"manifest_version": 1, "bindings": bindings}, temp)
        temp.close()
        return Path(temp.name)

    def test_loads_complete_binding_and_exposes_required_runtime_fields(self):
        manifest = self._write([_binding()])

        loaded = load_binding_manifest(manifest, required_program_ids=("T06",))

        self.assertEqual(loaded["T06"].scene_id, "custom-scene-v1")
        self.assertEqual(loaded["T06"].seed_ids[-1], "seed-0005")
        self.assertEqual(loaded["T06"].randomization["scale"], (0.8, 1.2))

    def test_rejects_missing_executable_binding_fields(self):
        value = _binding()
        del value["expert_id"]
        manifest = self._write([value])

        with self.assertRaisesRegex(ValueError, "expert_id"):
            load_binding_manifest(manifest, required_program_ids=("T06",))

    def test_rejects_catalog_split_mismatch_and_missing_required_program(self):
        manifest = self._write([_binding("T06", split="dev")])

        with self.assertRaisesRegex(ValueError, "catalog split"):
            load_binding_manifest(manifest, required_program_ids=("T06",))

    def test_rejects_cross_split_asset_family_and_lineage(self):
        first = _binding("T06")
        second = _binding("T08")
        second["asset_family_id"] = first["asset_family_id"]
        second["source_lineage_root"] = first["source_lineage_root"]
        second["split"] = "dev"
        manifest = self._write([first, second])

        with self.assertRaisesRegex(ValueError, "catalog split"):
            load_binding_manifest(manifest, required_program_ids=("T06", "T08"))


if __name__ == "__main__":
    unittest.main()
