from dataclasses import replace
import importlib.util
import unittest
import torch
import test_composition


class ConstantSampler:
    def __init__(self, sampling, diffusion, codec, schedule):
        self.codec = codec

    def sample(self, network, data):
        from icgs.contracts.records import ActionTrajectory
        return ActionTrajectory(torch.eye(4).repeat(data.actions.shape[0], self.codec.pred_horizon, 1, 1),
                                torch.ones(data.actions.shape[0], self.codec.pred_horizon, 1))


class LinearSchedule:
    def __init__(self, config):
        self.calls = []

    def add_noise(self, values, noise, timesteps):
        return values + noise * 0.01

    def set_timesteps(self, count):
        self.count = count

    def step(self, model_output, sample, timestep):
        from types import SimpleNamespace
        self.calls.append(timestep)
        return SimpleNamespace(prev_sample=model_output)


class PolicyTests(unittest.TestCase):
    sample = test_composition.CompositionTests.sample

    def setUp(self):
        test_composition.CompositionTests.setUp(self)
        self.assertIsNotNone(importlib.util.find_spec('icgs.policies.instant_policy'), 'Missing public policy')

    def policy(self, custom=True):
        from icgs.composition import build_policy
        cfg = replace(self.c, sampling=replace(self.c.sampling, kind='constant' if custom else 'original', steps=2),
                      diffusion=replace(self.c.diffusion, scheduler_kind='linear'))
        from icgs.algorithms.diffusion.process import OriginalDiffusionObjective
        factories = replace(self.factories, sampler={'constant': ConstantSampler},
                            scheduler={'linear': LinearSchedule})
        return build_policy(cfg, factories)

    def test_sampler_swap_uses_same_public_batch_api(self):
        policy = self.policy()
        data = self.sample()
        before = data.actions.clone()
        result = policy.predict_batch(data)
        torch.testing.assert_close(result.transforms, torch.eye(4).repeat(1,2,1,1))
        torch.testing.assert_close(data.actions, before)
        self.assertEqual(result.grips.shape, (1,2,1))

    def test_original_sampler_loop_and_objective_run_with_real_network(self):
        policy = self.policy(custom=False)
        result = policy.predict_batch(self.sample())
        self.assertTrue(torch.isfinite(result.transforms).all())
        self.assertEqual(result.transforms.shape, (1,2,4,4))
        self.assertEqual(policy.sampler.noise_scheduler.calls, [1,1,0,0])
        loss = policy.objective(policy.network, self.sample())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_context_rejects_cross_policy_and_resets_cache(self):
        from icgs.state.context_cache import PreparedContext
        a, b = self.policy(), self.policy()
        context = PreparedContext([{'obs':[None,None],'grips':[1.,1.],'T_w_es':[None,None]}], a.context_owner, embeddings=torch.ones(1), positions=torch.ones(1))
        with self.assertRaisesRegex(ValueError, 'another policy'):
            b.validate_context(context)
        a.reset_context(context)
        self.assertIsNone(context.embeddings)
        self.assertIsNone(context.positions)
        self.assertIsNot(a.sampler, b.sampler)

    def test_observation_api_caches_real_embeddings_per_episode(self):
        import numpy as np
        from unittest.mock import patch
        from icgs.contracts.records import Observation
        from icgs.data.preprocessing import native as preprocessing
        policy = self.policy()
        demos = [dict(obs=[np.zeros((4,3)), np.ones((4,3))], grips=[1.,1.],
                      T_w_es=[np.eye(4), np.eye(4)])]
        context = policy.prepare_context(demos, prepared=True)
        observation = Observation(np.arange(12,dtype=float).reshape(4,3), np.eye(4), 1.)
        with patch.object(preprocessing, 'remove_statistical_outliers',
                          side_effect=lambda points, **kwargs: (points, np.arange(len(points)))):
            result = policy.predict(observation, context)
            cached = context.embeddings
            policy.predict(observation, context)
            self.assertIs(context.embeddings, cached)
            policy.reset_context(context)
            policy.predict(observation, context)
            self.assertIsNot(context.embeddings, cached)
        self.assertEqual(result.transforms.shape, (1,2,4,4))
        self.assertEqual(context.embeddings.shape, (1,1,2,2,128))

    def test_prepared_demo_content_is_owned_and_readonly(self):
        import numpy as np
        policy=self.policy()
        cloud=np.ones((4,3))
        demo=dict(obs=[cloud,cloud],grips=[1.,1.],T_w_es=[np.eye(4),np.eye(4)])
        context=policy.prepare_context([demo],prepared=True)
        identity=context.source_id
        cloud.fill(9)
        np.testing.assert_array_equal(context.demos[0]['obs'][0],np.ones((4,3)))
        with self.assertRaises((ValueError,TypeError)):context.demos[0]['obs'][0][0,0]=5
        with self.assertRaises(TypeError):context.demos[0]['obs']=[]
        self.assertEqual(context.source_id,identity)
        context.embeddings=torch.ones(2)
        branch=context.branch_copy()
        branch.embeddings.zero_()
        self.assertTrue(torch.equal(context.embeddings,torch.ones(2)))
        self.assertIs(branch.demos,context.demos)

    def test_real_policy_runs_generic_evaluator_without_model_internal_calls(self):
        import numpy as np
        from unittest.mock import patch
        from test_evaluation import FakeEnvironment
        from icgs.execution.rollout import evaluate_policy
        from icgs.data.preprocessing import native as preprocessing
        self.c = replace(self.c, graph=replace(self.c.graph, traj_horizon=1))
        policy = self.policy()
        environment = FakeEnvironment([[(1., True)]])
        with patch.object(preprocessing, 'remove_statistical_outliers',
                          side_effect=lambda points, **kwargs: (points, np.arange(len(points)))):
            score = evaluate_policy(policy, environment, num_demos=1, num_rollouts=1, num_traj_wp=1,
                                    execution_horizon=2)
        self.assertEqual(score, 1.)
        self.assertTrue(environment.shutdown_called)


if __name__ == '__main__':
    unittest.main()
