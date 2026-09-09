import json
import random
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
import torch

from icgs.contracts.records import ActionTrajectory, Observation


class IntegrationObservabilityTests(unittest.TestCase):
    def _recorder(self, directory):
        from icgs.observability.config import from_mapping
        from icgs.observability.recorder import RunRecorder
        return RunRecorder.start(from_mapping({"output_dir": directory,
                                               "flush_interval_s": 0.01,
                                               "shutdown_timeout_s": 1.0}), command="integration")

    def test_candidate_proposals_emit_existing_timing_and_identity_without_changing_rng(self):
        from icgs.algorithms.planning.candidates import propose_candidates

        class Policy:
            runtime = SimpleNamespace(device="cpu")
            artifact_sha256 = "artifact"

            def validate_context(self, context):
                return None

            def predict(self, observation, context):
                transforms = torch.eye(4).repeat(1, 2, 1, 1)
                transforms[:, :, 0, 3] = torch.rand(1, 2)
                return ActionTrajectory(transforms, torch.ones(1, 2, 1))

        with tempfile.TemporaryDirectory() as directory:
            recorder = self._recorder(directory)
            observation = Observation(np.ones((3, 3)), np.eye(4), 1.0)
            before = torch.get_rng_state().clone()
            candidates = propose_candidates(Policy(), observation,
                                            SimpleNamespace(source_id="demo"), count=2,
                                            seeds=[1, 2], recorder=recorder)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            recorder.close()
            records = []
            for path in sorted((Path(recorder.path) / "events").glob("*.jsonl")):
                records.extend(json.loads(line) for line in path.read_text().splitlines())
            created = [item for item in records if item["event"] == "candidate.created"]
            self.assertEqual([item["fields"]["index"] for item in created], [0, 1])
            self.assertEqual(created[0]["fields"]["source_id"], "demo")

    def test_evaluation_boundary_links_episode_and_preserves_step_exception_behavior(self):
        from icgs.execution.rollout import evaluate_policy
        from tests.test_evaluation import CallablePolicy, FakeEnvironment

        with tempfile.TemporaryDirectory() as directory:
            recorder = self._recorder(directory)
            environment = FakeEnvironment([[RuntimeError("step failed")]])
            result = evaluate_policy(CallablePolicy(), environment, num_rollouts=1,
                                     recorder=recorder)
            self.assertEqual(result, 0.0)
            recorder.close()
            records = []
            for path in sorted((Path(recorder.path) / "events").glob("*.jsonl")):
                records.extend(json.loads(line) for line in path.read_text().splitlines())
            self.assertTrue(any(item["event"] == "evaluation.rollout.end" for item in records))
            self.assertTrue(any(item["event"] == "environment.step.error" for item in records))
            self.assertTrue(any(item["episode_id"] == "0" for item in records))

    def test_enabled_and_disabled_recorder_preserve_real_component_rng_order_keys_and_output(self):
        from icgs.configuration.method import MethodConfig
        from icgs.models.memories.physical import PhysicalMemory
        from icgs.observability.config import from_mapping
        from icgs.observability.recorder import RunRecorder

        def run_once(recorder):
            random.seed(1729)
            np.random.seed(1729)
            torch.manual_seed(1729)
            component = PhysicalMemory(MethodConfig())
            component.eval()
            calls = []
            X = torch.zeros(1, 128, 256)
            valid = torch.ones(1, 128, dtype=torch.bool)
            proprioception = torch.zeros(1, 13)
            previous_action = torch.zeros(1, 8)
            memory = torch.zeros(1, 2, 256)
            with recorder.span("physical.memory.update", component="physical"):
                calls.append("before")
                output = component(X, valid, proprioception, previous_action, memory)
                calls.append("after")
            rng_probe = (random.random(), np.random.random(), torch.rand(3))
            return calls, tuple(component.state_dict()), output.detach().clone(), rng_probe

        with tempfile.TemporaryDirectory() as disabled_directory, tempfile.TemporaryDirectory() as enabled_directory:
            disabled_recorder = RunRecorder.start(from_mapping({
                "output_dir": disabled_directory,
                "mode": "off",
            }), command="component-disabled")
            disabled = run_once(disabled_recorder)
            disabled_recorder.close()
            enabled_recorder = self._recorder(enabled_directory)
            try:
                enabled = run_once(enabled_recorder)
            finally:
                enabled_recorder.close()

        self.assertEqual(enabled[0], disabled[0])
        self.assertEqual(enabled[1], disabled[1])
        torch.testing.assert_close(enabled[2], disabled[2], rtol=0, atol=0)
        self.assertEqual(enabled[3][0], disabled[3][0])
        self.assertEqual(enabled[3][1], disabled[3][1])
        torch.testing.assert_close(enabled[3][2], disabled[3][2], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
