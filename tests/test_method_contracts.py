import dataclasses
import json
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np


class MethodContractTests(unittest.TestCase):
    @staticmethod
    def _reference_payload():
        return {
            "ip_checksum": "a" * 64,
            "native_profile": "instant-policy-original-65dc94e",
            "geometry": {"voxel_size_m": 0.005, "anchors": 128},
            "physical_weights": "physical-v1",
            "event_weights": "event-v1",
            "task_weights": "task-v1",
            "segmentation": {"id": "segments-v1"},
            "router": {"id": "router-v1"},
            "preprocessing": {"voxel_size_m": 0.005},
            "calibration": {"id": "cal-v1"},
            "camera": {"id": "camera-v1"},
            "gravity": [0.0, 0.0, -1.0],
            "workspace": {"id": "workspace-v1"},
            "cadence": {"dt0": 0.1, "h": 2, "r": 2, "H": 512},
            "rng_protocol": {"route": "route-v1", "diffusion": "diffusion-v1"},
        }

    def test_timed_command_owns_read_only_pose_and_rejects_nan_duration(self):
        from icgs.contracts.method import TimedCommand

        pose = np.eye(4)
        cmd = TimedCommand(pose, 1, 0.1)
        pose[0, 3] = 7
        self.assertEqual(cmd.target_w[0, 3], 0)
        self.assertFalse(cmd.target_w.flags.writeable)
        with self.assertRaises(ValueError):
            TimedCommand(np.eye(4), 1, float("nan"))

    def test_timed_command_validates_pose_grip_and_duration(self):
        from icgs.contracts.method import TimedCommand

        for pose in (np.zeros((3, 3)), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                TimedCommand(pose, 0, 0.1)
        bad_pose = np.eye(4)
        bad_pose[3, 3] = 2
        with self.assertRaises(ValueError):
            TimedCommand(bad_pose, 0, 0.1)
        for grip in (True, -1, 2):
            with self.assertRaises(ValueError):
                TimedCommand(np.eye(4), grip, 0.1)
        for duration in (0, -0.1, float("inf")):
            with self.assertRaises(ValueError):
                TimedCommand(np.eye(4), 0, duration)

    def test_prefix_and_transition_validate_ownership_and_boundaries(self):
        from icgs.contracts.method import (
            CommandPrefix,
            ExecutedTransition,
            TimedCommand,
            TimedObservation,
        )
        from icgs.contracts.records import Observation

        command = TimedCommand(np.eye(4), 0, 0.1)
        with self.assertRaises(ValueError):
            CommandPrefix((), np.eye(4), "candidate")
        root = np.eye(4)
        prefix = CommandPrefix((command,), root, "candidate")
        root[0, 3] = 3
        self.assertEqual(prefix.proposal_root[0, 3], 0)
        observation = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
        before = TimedObservation(observation, 0, 1.0, 2.0, "sensor")
        after = TimedObservation(observation, 1, 1.1, 2.1, "sensor")
        with self.assertRaises(ValueError):
            TimedObservation(observation, -1, 1.0, 2.0, "sensor")
        with self.assertRaisesRegex(ValueError, "sensor"):
            TimedObservation(observation, 0, 1.0, 2.0, "")
        transition = ExecutedTransition(before, after, command, 0.1, 4, "ok")
        self.assertEqual(transition.after.boundary, 1)

    def test_timed_observation_owns_native_numeric_inputs(self):
        from icgs.contracts.method import TimedObservation
        from icgs.contracts.records import Observation

        points = np.zeros((1, 3))
        pose = np.eye(4)
        timed = TimedObservation(Observation(points, pose, 1.0), 0, 1.0, 2.0, "sensor")
        points[0, 0] = 9
        pose[0, 3] = 9
        self.assertEqual(timed.observation.points[0, 0], 0)
        self.assertEqual(timed.observation.T_w_e[0, 3], 0)
        self.assertFalse(timed.observation.points.flags.writeable)
        self.assertFalse(timed.observation.T_w_e.flags.writeable)

    def test_probabilities_and_planning_result_reject_invalid_values(self):
        from icgs.contracts.method import PlanningResult, TerminalProbabilities

        with self.assertRaises(ValueError):
            TerminalProbabilities(np.array([[0.2, 0.2, 0.2]]), "temperature")
        probabilities = TerminalProbabilities(
            np.array([[0.2, 0.3, 0.5]], dtype=np.float64), "temperature"
        )
        self.assertFalse(probabilities.probabilities.flags.writeable)
        with self.assertRaises(ValueError):
            TerminalProbabilities(np.array([[np.nan, 0.0, 1.0]]), "temperature")
        with self.assertRaisesRegex(ValueError, "temperature"):
            TerminalProbabilities(np.array([[0.2, 0.3, 0.5]]), "")
        from icgs.contracts.method import EvaluationOutput
        with self.assertRaisesRegex(ValueError, "temperature"):
            EvaluationOutput(np.zeros(1), np.zeros(1), np.zeros(1),
                             np.zeros(1), np.zeros(1), "")
        with self.assertRaises(ValueError):
            PlanningResult(None, False, "fallback", float("nan"), 0, {}, {}, ())

    def test_protocol_surfaces_are_importable_without_concrete_runtime_modules(self):
        from icgs.contracts.method import (
            MethodCapabilities,
            ReplayProvider,
            TaskMonitor,
            TimedEnvironment,
        )

        for protocol in (TimedEnvironment, TaskMonitor, ReplayProvider, MethodCapabilities):
            self.assertTrue(getattr(protocol, "_is_protocol", False))

    def test_contracts_do_not_implement_state_or_planning_records(self):
        import icgs.contracts.method as method

        for name in ("PhysicalState", "EventMemory", "TaskState", "MethodContext",
                     "Hypothesis", "BeliefNode"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(method, name))

    def test_method_config_rejects_unknown_fields_and_commit_horizon(self):
        try:
            from icgs.configuration.method import MethodConfig
        except ModuleNotFoundError as exc:
            self.fail(f"MethodConfig boundary is missing: {exc}")

        with self.assertRaisesRegex(ValueError, "unknown"):
            MethodConfig.from_dict({"surprise": 1})
        with self.assertRaisesRegex(ValueError, "horizon|commit"):
            MethodConfig.from_dict({"planning": {"h": 2, "r": 8}})

    def test_method_config_validates_sections_defaults_and_native_distinctions(self):
        from icgs.configuration.method import MethodConfig
        from icgs.configuration.schema import ExperimentConfig

        config = MethodConfig.from_dict({})
        self.assertEqual(config.geometry.num_anchors, 128)
        self.assertEqual(config.geometry.width, 256)
        self.assertEqual(config.planning.H, 512)
        self.assertEqual(config.control.dt0, 0.1)
        self.assertEqual(json.loads(json.dumps(config.to_dict()))["schema_version"], 1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.planning = config.planning

        native = ExperimentConfig().validate()
        self.assertEqual(native.graph.pred_horizon, 8)
        self.assertEqual(native.graph.traj_horizon, 10)
        self.assertNotEqual(config.planning.H, native.graph.traj_horizon)

    def test_method_config_rejects_each_unknown_section_field_and_invalid_values(self):
        from icgs.configuration.method import MethodConfig

        for section in ("geometry", "event", "memory", "dynamics", "evaluator",
                        "planning", "control", "dataset", "training"):
            with self.subTest(section=section):
                with self.assertRaisesRegex(ValueError, "unknown"):
                    MethodConfig.from_dict({section: {"typo": 1}})
        for payload in (
            {"schema_version": True},
            {"native_profile": "invented"},
            {"geometry": {"voxel_size_m": 0.01}},
            {"planning": {"h": True}},
            {"control": {"dt0": float("nan")}},
            {"planning": {"h": lambda: None}},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    MethodConfig.from_dict(payload)

    def test_method_config_uses_component_ids_and_file_overlay_path_precedence(self):
        from icgs.configuration.method import MethodConfig

        document = {
            "schema_version": 1,
            "config": {
                "geometry": {"component_id": "geometry"},
                "dataset": {
                    "component_id": "dataset",
                    "manifest_path": "manifests/train.json",
                },
                "planning": {"H": 256, "h": 2, "r": 2},
                "training": {"component_id": "training", "output_dir": "runs"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "method.json"
            path.write_text(json.dumps(document))
            config = MethodConfig.from_file(
                path,
                overrides={"planning.H": 128, "planning.h": 1, "planning.r": 1},
            )
            self.assertEqual(config.geometry.component_id, "geometry")
            self.assertEqual(config.planning.H, 128)
            self.assertEqual(config.planning.h, 1)
            self.assertEqual(config.dataset.manifest_path,
                             str((Path(directory) / "manifests/train.json").resolve()))
            self.assertEqual(config.training.output_dir,
                             str((Path(directory) / "runs").resolve()))

            path.write_text(json.dumps({"schema_version": 1, "surprise": {}}))
            with self.assertRaisesRegex(ValueError, "unknown"):
                MethodConfig.from_file(path)

        with self.assertRaisesRegex(ValueError, "component"):
            MethodConfig.from_dict({"geometry": {"component_id": "unknown"}})

    def test_method_config_rejects_bool_horizons_and_cross_section_types(self):
        from icgs.configuration.method import DynamicsConfig, EvaluatorConfig, MethodConfig

        for section in ("dynamics", "evaluator", "training"):
            with self.subTest(section=section):
                with self.assertRaises(ValueError):
                    MethodConfig.from_dict({section: {"H": True}})
        with self.assertRaises(ValueError):
            MethodConfig(dynamics=EvaluatorConfig()).validate()
        with self.assertRaises(ValueError):
            MethodConfig(evaluator=DynamicsConfig()).validate()

    def test_reference_fingerprint_requires_reference_identity(self):
        try:
            from icgs.artifacts.method import reference_fingerprint
        except ModuleNotFoundError as exc:
            self.fail(f"reference fingerprint boundary is missing: {exc}")

        with self.assertRaisesRegex(ValueError, "reference"):
            reference_fingerprint({})
        with self.assertRaisesRegex(ValueError, "reference"):
            reference_fingerprint({"ip_checksum": "not-a-sha256"})

    def test_reference_fingerprint_is_ordered_sensitive_and_excludes_stopping(self):
        from icgs.artifacts.method import reference_fingerprint

        payload = self._reference_payload()
        reference_id = reference_fingerprint(payload)
        reordered = dict(reversed(tuple(payload.items())))
        reordered["geometry"] = {"anchors": 128, "voxel_size_m": 0.005}
        self.assertEqual(reference_fingerprint(reordered), reference_id)

        changed = copy.deepcopy(payload)
        changed["router"]["id"] = "router-v2"
        self.assertNotEqual(reference_fingerprint(changed), reference_id)
        excluded = copy.deepcopy(payload)
        excluded["dynamics"] = {"id": "dynamics-v2"}
        excluded["evaluator"] = {"temperature": "te-v1"}
        excluded["learned_stopping"] = {"threshold": 0.95}
        self.assertEqual(reference_fingerprint(excluded), reference_id)

    def test_method_manifest_checks_reference_and_evaluator_lineage_before_reads(self):
        from icgs.artifacts.method import reference_fingerprint, validate_method_manifest
        from icgs.configuration.method import MethodConfig

        payload = self._reference_payload()
        reference_id = reference_fingerprint(payload)
        manifest = {
            "schema_version": 1,
            "reference_id": reference_id,
            "reference_payload": payload,
            "evaluator_reference_id": reference_id,
            "dynamics_artifact_id": "dynamics-v1",
            "evaluator_artifact_id": "evaluator-v1",
            "learned_stopping_artifact_id": "stopping-v1",
        }
        config = MethodConfig.from_dict({})

        class ArtifactLoader:
            def __init__(self):
                self.calls = []

            def __call__(self, role, artifact_id):
                self.calls.append((role, artifact_id))
                return {"role": role, "artifact_id": artifact_id}

        loader = ArtifactLoader()
        try:
            validated = validate_method_manifest(manifest, config, artifact_loader=loader)
        except TypeError as exc:
            self.fail(f"outer artifact loader seam is missing: {exc}")
        self.assertEqual(validated, reference_id)
        self.assertEqual(
            loader.calls,
            [
                ("dynamics", "dynamics-v1"),
                ("evaluator", "evaluator-v1"),
                ("learned_stopping", "stopping-v1"),
            ],
        )

        loader.calls.clear()
        with self.assertRaisesRegex(ValueError, "reference"):
            validate_method_manifest(
                dict(manifest, evaluator_reference_id="b" * 64),
                config,
                artifact_loader=loader,
            )
        self.assertEqual(loader.calls, [])
        mismatched_config = MethodConfig.from_dict({"planning": {"h": 1, "r": 1}})
        with self.assertRaisesRegex(ValueError, "reference|cadence"):
            validate_method_manifest(manifest, mismatched_config)
        incomplete = dict(manifest)
        incomplete.pop("learned_stopping_artifact_id")
        with self.assertRaisesRegex(ValueError, "reference"):
            validate_method_manifest(incomplete, config)


if __name__ == "__main__":
    unittest.main()
