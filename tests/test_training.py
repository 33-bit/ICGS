import importlib.util
import unittest
from dataclasses import replace
import torch


@unittest.skipIf(importlib.util.find_spec('lightning') is None, 'SKIPPED: Lightning unavailable')
class TrainingTests(unittest.TestCase):
    def test_training_adapter_registers_legacy_keys_and_uses_built_policy(self):
        from test_policy import PolicyTests
        helper = PolicyTests()
        helper.setUp()
        from ip.training import GraphDiffusion
        policy = helper.policy()
        model = GraphDiffusion(helper.c, policy=policy)
        self.assertIs(model.model, policy.network)
        keys = model.state_dict()
        self.assertIn('model.scene_encoder.weight', keys)
        self.assertIn('scene_encoder.weight', keys)
        torch.testing.assert_close(keys['scene_encoder.weight'], keys['model.scene_encoder.weight'])


if __name__ == '__main__':
    unittest.main()
