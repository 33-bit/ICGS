import dataclasses
import importlib.util
import unittest


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('ip.configs.original'), 'Missing structured baseline')
        from ip.configs.original import instant_policy_original, from_legacy, to_legacy, profile
        self.original, self.from_legacy, self.to_legacy, self.profile = instant_policy_original, from_legacy, to_legacy, profile

    def test_baseline_roundtrip_and_fresh_tensor_limits(self):
        from ip.configs.base_config import config
        import torch
        resolved = self.from_legacy(config)
        result = self.to_legacy(resolved)
        self.assertEqual(set(result), set(config))
        for key in config:
            if isinstance(config[key], torch.Tensor):
                torch.testing.assert_close(result[key], config[key])
            else:
                self.assertEqual(result[key], config[key], key)
        result['min_actions'].zero_()
        self.assertLess(self.to_legacy(resolved)['min_actions'][0], 0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            resolved.runtime.device = 'cpu'

    def test_profiles_do_not_mutate_source_and_preserve_override_differences(self):
        original = self.original()
        evaluation = self.profile(original, 'eval', num_demos=3)
        self.assertEqual(original.sampling.steps, 8)
        self.assertEqual(evaluation.sampling.steps, 4)
        self.assertEqual(evaluation.runtime.batch_size, 1)
        self.assertEqual(evaluation.graph.num_demos, 3)
        self.assertEqual(original.graph.num_demos, 2)

    def test_legacy_missing_unknown_and_inconsistent_config_fail(self):
        from ip.configs.base_config import config
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.from_legacy({})
        with self.assertRaisesRegex(ValueError, 'unknown'):
            self.from_legacy(dict(config, typo=1))
        invalid = dataclasses.replace(self.original(), graph=dataclasses.replace(self.original().graph, num_demos=0))
        with self.assertRaises(ValueError):
            invalid.validate()

    def test_resolved_json_preserves_component_identity(self):
        import json
        from ip.configs.original import resolved_config, from_resolved
        c = dataclasses.replace(self.original(), scene=dataclasses.replace(self.original().scene, kind='experiment'))
        restored = from_resolved(json.loads(json.dumps(resolved_config(c))))
        self.assertEqual(restored, c)
        self.assertIsInstance(restored.action.minimum, tuple)
        with self.assertRaises(ValueError):
            from_resolved({'schema_version': 2, 'config': {}})

    def test_custom_graph_is_not_subject_to_original_gripper_width_limit(self):
        c = self.original()
        c = dataclasses.replace(c, scene=dataclasses.replace(c.scene, kind='custom', embd_dim=32),
                                graph=dataclasses.replace(c.graph, kind='custom', embd_dim=32))
        self.assertIs(c.validate(), c)


if __name__ == '__main__':
    unittest.main()
