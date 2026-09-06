"""CPU evaluation-contract tests; no simulator or model construction."""

import importlib.util
import unittest

import numpy as np
import torch

from icgs.contracts.records import ActionTrajectory, Observation
from icgs.state.context_cache import PreparedContext


class CallablePolicy:
    def __init__(self, horizon=3):
        self.horizon = horizon
        self.owner = object()
        self.eval_calls = 0
        self.prepare_calls = []
        self.reset_calls = 0
        self.cache_before_predict = []
        self.observations = []

    def eval(self):
        self.eval_calls += 1
        return self

    def prepare_context(self, demos, *, prepared=False):
        self.prepare_calls.append((demos, prepared))
        return PreparedContext(list(demos), self.owner)

    def reset_context(self, context):
        self.reset_calls += 1
        context.embeddings = None
        context.positions = None

    def predict(self, observation, context):
        self.cache_before_predict.append(context.embeddings)
        context.embeddings = torch.tensor([len(self.observations) + 1.0])
        self.observations.append(observation)
        transforms = torch.eye(4).repeat(1, self.horizon, 1, 1)
        grips = torch.ones(1, self.horizon, 1)
        return ActionTrajectory(transforms, grips)


class FakeEnvironment:
    def __init__(self, rollout_steps):
        self.rollout_steps = rollout_steps
        self.launched = False
        self.shutdown_called = False
        self.collect_args = None
        self.reset_calls = 0
        self.observe_calls = 0
        self.encoded = []
        self.step_index = 0

    def launch(self):
        self.launched = True

    def collect_demos(self, num_demos, num_traj_wp):
        self.collect_args = (num_demos, num_traj_wp)
        return [dict(obs=[np.zeros((2, 3))], grips=[1.0], T_w_es=[np.eye(4)])
                for _ in range(num_demos)]

    def reset(self):
        self.rollout_index = self.reset_calls
        self.reset_calls += 1
        self.step_index = 0

    def observe(self):
        transform = np.eye(4)
        transform[0, 3] = self.observe_calls + 1
        self.observe_calls += 1
        return Observation(np.full((4, 3), self.observe_calls), transform, 1.0)

    def encode_action(self, observation, trajectory, index):
        self.encoded.append((observation, trajectory, index))
        return index

    def step(self, command):
        result = self.rollout_steps[self.rollout_index][self.step_index]
        self.step_index += 1
        if isinstance(result, Exception):
            raise result
        return result

    def shutdown(self):
        self.shutdown_called = True


class EvaluationTests(unittest.TestCase):
    def test_success_metric_termination_anchor_and_context_reset(self):
        from icgs.execution.rollout import evaluate_policy

        policy = CallablePolicy()
        environment = FakeEnvironment([
            [(0.0, False), (1.0, True)],
            [(0.0, True)],
        ])

        result = evaluate_policy(policy, environment, num_demos=2, num_rollouts=2,
                                 max_execution_steps=4, execution_horizon=3,
                                 num_traj_wp=10)

        self.assertEqual(result, 0.5)
        self.assertTrue(environment.launched)
        self.assertTrue(environment.shutdown_called)
        self.assertEqual(environment.collect_args, (2, 10))
        self.assertEqual(policy.eval_calls, 1)
        self.assertTrue(policy.prepare_calls[0][1])
        self.assertEqual(policy.reset_calls, 2)
        self.assertEqual(policy.cache_before_predict, [None, None])
        self.assertIs(environment.encoded[0][0], environment.encoded[1][0])
        self.assertEqual([entry[2] for entry in environment.encoded], [0, 1, 0])

    def test_step_exception_terminates_unsuccessful_rollout(self):
        from icgs.execution.rollout import evaluate_policy

        policy = CallablePolicy()
        environment = FakeEnvironment([[RuntimeError("step failed")]])

        result = evaluate_policy(policy, environment, num_rollouts=1)

        self.assertEqual(result, 0.0)
        self.assertEqual(environment.observe_calls, 1)
        self.assertEqual(len(environment.encoded), 1)
        self.assertTrue(environment.shutdown_called)

    def test_execution_and_replanning_horizons_are_independent(self):
        from icgs.execution.rollout import evaluate_policy

        policy = CallablePolicy(horizon=3)
        environment = FakeEnvironment([[(0.0, False)] * 6])

        result = evaluate_policy(policy, environment, num_rollouts=1,
                                 max_execution_steps=2, execution_horizon=3)

        self.assertEqual(result, 0.0)
        self.assertEqual(environment.observe_calls, 2)
        self.assertEqual([entry[2] for entry in environment.encoded], [0, 1, 2, 0, 1, 2])
        self.assertIsNot(environment.encoded[0][0], environment.encoded[3][0])

    def test_encode_failure_propagates_without_failure_cleanup(self):
        from icgs.execution.rollout import evaluate_policy

        class EncodeFailure(FakeEnvironment):
            def encode_action(self, observation, trajectory, index):
                raise ValueError("encode failed")

        environment = EncodeFailure([[]])
        with self.assertRaisesRegex(ValueError, "encode failed"):
            evaluate_policy(CallablePolicy(), environment, num_rollouts=1)
        self.assertFalse(environment.shutdown_called)

    def test_rlbench_adapter_converts_public_trajectory_at_boundary(self):
        from icgs.environments.rlbench.adapter import RLBenchAdapter

        adapter = RLBenchAdapter("plate_out")
        anchor = np.eye(4)
        anchor[0, 3] = 1.0
        relative = torch.eye(4).repeat(1, 2, 1, 1)
        relative[0, 1, 1, 3] = 0.25
        trajectory = ActionTrajectory(relative, torch.tensor([[[-0.2], [0.1]]]))

        command = adapter.encode_action(Observation(np.zeros((1, 3)), anchor, 0.0),
                                        trajectory, 1)

        np.testing.assert_allclose(command[:3], [1.0, 0.25, 0.0])
        np.testing.assert_allclose(command[3:7], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(command[7], 1.0)

    def test_rlbench_imports_are_lazy_and_task_mapping_is_available(self):
        from icgs.environments.rlbench import adapter as rlbench

        self.assertEqual(len(rlbench.TASK_NAMES), 17)
        self.assertIn("plate_out", rlbench.TASK_NAMES)

    @unittest.skipUnless(importlib.util.find_spec("rlbench"),
                         "SKIPPED: RLBench is not installed")
    def test_rlbench_task_alias_resolves_to_real_task_class(self):
        from rlbench.tasks import TakePlateOffColoredDishRack
        from icgs.environments.rlbench.adapter import TASK_NAMES

        self.assertIs(TASK_NAMES["plate_out"], TakePlateOffColoredDishRack)


if __name__ == "__main__":
    unittest.main()
