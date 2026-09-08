"""Method metadata tests; these do not exercise neural/training consumers."""

import hashlib
import dataclasses
import json
from pathlib import Path
from importlib import resources
import tempfile
import unittest
from unittest.mock import patch

from icgs.configuration.method import MethodConfig


class MethodConfigTests(unittest.TestCase):
    def test_existing_defaults_survive_configuration_source_migration(self):
        # An accidental source/default migration must not alter existing metadata.
        previous = {
            "schema_version": 1,
            "native_profile": "instant-policy-original-65dc94e",
            "geometry": {"component_id": "geometry", "voxel_size_m": 0.005,
                         "num_anchors": 128, "width": 256, "point_dim": 3,
                         "neighbors": 32, "ell0_m": 1.0},
            "event": {"component_id": "event", "width": 256,
                      "max_interactions": 30, "landmarks": 2},
            "memory": {"component_id": "memory", "width": 256, "slots": 2,
                       "descriptor_dim": 8},
            "dynamics": {"component_id": "dynamics", "width": 256, "heads": 3, "H": 512},
            "evaluator": {"component_id": "evaluator", "width": 256, "heads": 3, "H": 512},
            "planning": {"component_id": "planning", "h": 2, "r": 2, "L": 32, "H": 512},
            "control": {"component_id": "control", "dt0": 0.1},
            "dataset": {"component_id": "dataset", "schema_id": "icgs_episode_v1",
                        "shard_intervals": 256, "manifest_path": None},
            "training": {"component_id": "training", "H": 512, "heads": 3, "output_dir": None},
        }
        actual = MethodConfig().to_dict()
        for key, value in previous.items():
            with self.subTest(key=key):
                selected = ({field: actual[key][field] for field in value}
                            if isinstance(value, dict) else actual[key])
                self.assertEqual(selected, value)

    def test_resolved_identity_roundtrip_and_override_changes_hash(self):
        config = MethodConfig.from_dict({"planning": {"H": 128}})
        self.assertTrue(callable(getattr(config, "fingerprint", None)),
                        "MethodConfig must expose its complete resolved identity")
        expected = hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True,
                                            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(config.fingerprint(), expected)
        self.assertNotEqual(config.fingerprint(), MethodConfig().fingerprint())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved.json"
            path.write_text(json.dumps(config.resolved_config()))
            loaded = MethodConfig.from_file(path)
            self.assertEqual(loaded.fingerprint(), expected)

    def test_file_rejects_duplicate_keys_instead_of_last_value_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"config":{"planning":{"h":1,"h":2}}}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                MethodConfig.from_file(path)

    def test_non_integer_schema_and_bool_dimensions_fail(self):
        for payload in ({"schema_version": 1.0}, {"geometry": {"point_dim": True}},
                        {"event": {"landmarks": 2.0}}, {"training": {"heads": 3.0}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                MethodConfig.from_dict(payload)

    def test_metadata_defaults_do_not_claim_operation_readiness(self):
        config = MethodConfig()
        self.assertTrue(callable(getattr(config, "validate_for", None)),
                        "method metadata requires explicit operation-readiness validation")
        for operation in ("geometry_training", "task_training", "collection", "evaluation"):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "required|missing"):
                config.validate_for(operation)
        with self.assertRaisesRegex(ValueError, "unknown"):
            config.validate_for("typo")

    def test_missing_or_malformed_packaged_defaults_have_no_python_fallback(self):
        # Replace only resource discovery: parsing/default construction stays real.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("importlib.resources.files", return_value=root):
                with self.assertRaisesRegex(ValueError, "canonical|packaged|defaults"):
                    MethodConfig()
                (root / "profiles").mkdir()
                (root / "profiles/icgs_primary.json").write_text('{"config":{}}')
                with self.assertRaisesRegex(ValueError, "canonical|packaged|defaults"):
                    MethodConfig()

    def test_optimizer_and_stage_defaults_overlay_and_frozen_sequences(self):
        config = MethodConfig()
        self.assertTrue(hasattr(config, "optimizer"), "canonical optimizer metadata is missing")
        self.assertEqual(config.optimizer.learning_rate, 0.0001)
        self.assertEqual(config.optimizer.betas, (0.9, 0.999))
        self.assertEqual(config.stages.A1.rollout_curriculum, (1, 2, 4))
        source = [0.8, 0.95]
        changed = MethodConfig.from_dict({"optimizer": {"betas": source}})
        source[0] = 0.1
        self.assertEqual(changed.optimizer.betas, (0.8, 0.95))
        self.assertEqual(changed.to_dict()["optimizer"]["betas"], [0.8, 0.95])

    def test_new_ranges_types_combinations_and_unknown_keys_rejected(self):
        for payload in (
            {"optimizer": {"learning_rate": 0}},
            {"optimizer": {"learning_rate": 0.000001}},
            {"optimizer": {"betas": [0.9, 1.0]}},
            {"optimizer": {"kind": "typo"}},
            {"stages": {"A0": {"batch_size": True}}},
            {"stages": {"A1": {"rollout_curriculum": [4, 2]}}},
            {"dataset": {"asset_split_ratios": [0.7, 0.2, 0.2]}},
            {"sensors": {"workspace_bounds_m": [[1, 0, 0], [0, 1, 1]]}},
            {"stages": {"training_seeds": [1, 2]}},
            {"stages": {"A0": {"surprise": 2}}},
            {"neural": {"attention_heads": 7}},
            {"collection": {"disk_limit_bytes": 0}},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                MethodConfig.from_dict(payload)

    def test_new_paths_resolve_relative_to_file_before_fingerprinting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "method.json"
            path.write_text(json.dumps({"config": {
                "sensors": {"calibration_path": "calibration.json"},
                "optimizer": {"learning_rate": 0.0002}}}))
            config = MethodConfig.from_file(path, overrides={"optimizer.learning_rate": 0.0003})
            self.assertEqual(config.optimizer.learning_rate, 0.0003)
            self.assertEqual(config.sensors.calibration_path,
                             str((Path(directory) / "calibration.json").resolve()))
            self.assertIsNone(config.sensors.replay_tolerances_path)

    def test_derived_dimensions_follow_semantic_composition(self):
        config = MethodConfig()
        self.assertTrue(callable(getattr(config, "derived_dimensions", None)))
        self.assertEqual(config.derived_dimensions(), {
            "attention_head_width": 32, "physical_tokens": 131, "dynamics_tokens": 132,
            "patch_points": 16, "decoded_points": 2048, "physical_input": 277,
            "demo_frame_input": 269, "event_input": 776, "event_heads_input": 768,
            "terminal_input": 1288,
        })
        self.assertEqual(config.event_token_capacity, 32)

    def test_stage_b_readiness_does_not_require_future_outcome_bank(self):
        config = MethodConfig.from_dict({
            "planning": {"H": 128}, "training": {"H": 256, "output_dir": "/run"},
            "dataset": {"manifest_path": "/episodes", "asset_manifest_path": "/assets",
                        "split_manifest_path": "/splits"},
            "stages": {"training_seeds": [0, 1, 2]},
            "sensors": {"workspace_bounds_m": [[-1, -1, 0], [1, 1, 1]],
                        "calibration_path": "/calibration", "gravity_w": [0, 0, -9.81],
                        "predicate_protocol_path": "/predicate"},
        })
        self.assertIs(config.validate_for("task_training"), config)

    def test_invalid_numeric_margins_and_whitespace_paths_rejected(self):
        for payload in ({"numerics": {"rotation_training_clip_margin": 2}},
                        {"numerics": {"mass_sum_atol_float32": 1}},
                        {"sensors": {"calibration_path": "   "}},
                        {"optimizer": {"learning_rate": 10 ** 400}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                MethodConfig.from_dict(payload)

    def test_resource_edits_drive_defaults_and_invalid_source_fails_closed(self):
        document = json.loads(resources.files("icgs.configuration")
                              .joinpath("profiles/icgs_primary.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profiles").mkdir()
            path = root / "profiles/icgs_primary.json"
            document["config"]["optimizer"]["learning_rate"] = 0.0002
            path.write_text(json.dumps(document))
            with patch("importlib.resources.files", return_value=root):
                self.assertEqual(MethodConfig().optimizer.learning_rate, 0.0002)
                document["config"]["optimizer"]["learning_rate"] = -1
                path.write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError, "learning_rate"):
                    MethodConfig()

    def test_packaged_schema_requires_all_keys_and_rejects_unknown_keys(self):
        original = json.loads(resources.files("icgs.configuration")
                              .joinpath("profiles/icgs_primary.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profiles").mkdir()
            path = root / "profiles/icgs_primary.json"
            for mutation in ("missing", "unknown", "wrong_type", "nonfinite"):
                document = json.loads(json.dumps(original))
                if mutation == "missing":
                    del document["config"]["optimizer"]["betas"]
                elif mutation == "unknown":
                    document["config"]["optimizer"]["typo"] = 1
                elif mutation == "wrong_type":
                    document["config"]["optimizer"]["warmup_updates"] = False
                else:
                    document["config"]["optimizer"]["learning_rate"] = float("inf")
                path.write_text(json.dumps(document))
                with self.subTest(mutation=mutation), patch("importlib.resources.files", return_value=root):
                    with self.assertRaisesRegex(ValueError, "canonical defaults"):
                        MethodConfig()

    def test_direct_sections_freeze_nested_arrays_and_keep_old_constructor_positions(self):
        from icgs.configuration.method import (
            GeometryConfig, EventConfig, MemoryConfig, DynamicsConfig, EvaluatorConfig,
            PlanningConfig, ControlConfig, DatasetConfig, TrainingConfig, OptimizerConfig,
        )
        config = MethodConfig(1, "instant-policy-original-65dc94e", GeometryConfig(),
                              EventConfig(), MemoryConfig(), DynamicsConfig(), EvaluatorConfig(),
                              PlanningConfig(), ControlConfig(), DatasetConfig(), TrainingConfig())
        self.assertEqual(config.optimizer.learning_rate, 0.0001)
        section = OptimizerConfig(betas=[0.8, 0.99])
        self.assertEqual(section.betas, (0.8, 0.99))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            section.betas = (0.1, 0.2)
        exported = config.to_dict()
        exported["optimizer"]["betas"][0] = 0.1
        self.assertEqual(config.optimizer.betas, (0.9, 0.999))

    def test_json_nonfinite_and_non_json_values_fail_before_construction(self):
        for value in (float("nan"), float("inf"), lambda: 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MethodConfig.from_dict({"optimizer": {"learning_rate": value}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for text in ('{"config":{"optimizer":{"learning_rate":NaN}}}',
                         '{"config":{"optimizer":{"learning_rate":1e999}}}',
                         '{"schema_version":1.0}', '{"config":{},"config":{}}'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    MethodConfig.from_file(path)

    def test_short_legacy_training_coverage_still_loads_metadata(self):
        config = MethodConfig.from_dict({"planning": {"H": 128}, "training": {"H": 256}})
        self.assertEqual(config.training.H, 256)

    def test_readiness_accepts_explicit_geometry_setup_without_reading_artifacts(self):
        config = MethodConfig.from_dict({
            "dataset": {"manifest_path": "/not-present/manifest.json"},
            "training": {"output_dir": "/not-present/run"},
            "stages": {"training_seeds": [0, 1, 2]},
            "sensors": {"workspace_bounds_m": [[-1, -1, 0], [1, 1, 1]],
                        "calibration_path": "/not-present/calibration.json"},
        })
        self.assertIs(config.validate_for("geometry_training"), config)
        self.assertEqual(config.sensors.workspace_bounds_m, ((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)))

    def test_resolved_identity_rejects_tampering_but_allows_explicit_override(self):
        config = MethodConfig()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved.json"
            document = config.resolved_config()
            path.write_text(json.dumps(document))
            changed = MethodConfig.from_file(path, overrides={"optimizer.learning_rate": 0.0002})
            self.assertNotEqual(changed.fingerprint(), config.fingerprint())
            document["config"]["optimizer"]["learning_rate"] = 0.0002
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "sha256"):
                MethodConfig.from_file(path)

    def test_relative_metadata_cannot_claim_a_resolved_identity(self):
        from icgs.configuration.method import DatasetConfig
        for config in (MethodConfig.from_dict({"dataset": {"manifest_path": "episodes/manifest.json"}}),
                       MethodConfig(dataset=DatasetConfig(manifest_path="episodes/manifest.json"))):
            with self.subTest(config=config.dataset.manifest_path):
                with self.assertRaisesRegex(ValueError, "absolute|from_file"):
                    config.resolved_config()

    def test_hashed_envelope_requires_complete_absolute_metadata(self):
        config = MethodConfig()
        relative = config.to_dict()
        relative["dataset"]["manifest_path"] = "episodes/manifest.json"
        relative_hash = hashlib.sha256(json.dumps(relative, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved.json"
            for label, document in (
                ("sparse", {"schema_version": 1, "config": {}, "config_sha256": config.fingerprint()}),
                ("relative", {"schema_version": 1, "config": relative, "config_sha256": relative_hash}),
            ):
                path.write_text(json.dumps(document))
                with self.subTest(label=label), self.assertRaisesRegex(ValueError, "complete|missing|absolute"):
                    MethodConfig.from_file(path)

    def test_hashed_envelope_path_override_resolves_and_recomputes_identity(self):
        config = MethodConfig()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved.json"
            path.write_text(json.dumps(config.resolved_config()))
            changed = MethodConfig.from_file(path, overrides={"dataset.manifest_path": "episodes/manifest.json"})
            self.assertEqual(changed.dataset.manifest_path,
                             str((Path(directory) / "episodes/manifest.json").resolve()))
            self.assertNotEqual(changed.fingerprint(), config.fingerprint())
            path.write_text(json.dumps(changed.resolved_config()))
            self.assertEqual(MethodConfig.from_file(path).fingerprint(), changed.fingerprint())

    def test_reactive_training_schedule_must_extend_beyond_warmup(self):
        with self.assertRaisesRegex(ValueError, "warmup|max_updates"):
            MethodConfig.from_dict({"reactive": {"max_updates": 1}})

    def test_absolute_metadata_path_spelling_is_identity_preserving(self):
        config = MethodConfig.from_dict({"dataset": {"manifest_path": "/not-present/./manifest.json"}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved.json"
            path.write_text(json.dumps(config.resolved_config()))
            self.assertEqual(MethodConfig.from_file(path).fingerprint(), config.fingerprint())


if __name__ == "__main__":
    unittest.main()
