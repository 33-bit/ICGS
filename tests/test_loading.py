from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import test_composition
from test_policy import LinearSchedule


class LoadingTests(unittest.TestCase):
    def test_direct_encoder_builder_rejects_unknown_selection(self):
        from ip.composition import build_scene_encoder
        from ip.configs.structured import SceneConfig
        with self.assertRaisesRegex(ValueError, 'scene.*unknown'):
            build_scene_encoder(SceneConfig(kind='unknown', pretrained=False))

    def test_resolved_demo_override_survives_load_and_observation_prediction(self):
        from ip.composition import build_policy, load_policy
        from ip.configs.original import resolved_config
        from ip.types import Observation
        from ip.data import preprocessing
        helper = test_composition.CompositionTests(); helper.setUp()
        config = replace(helper.c, diffusion=replace(helper.c.diffusion, scheduler_kind='linear'))
        factories = replace(helper.factories, scheduler={'linear': LinearSchedule})
        source = build_policy(config, factories)
        with tempfile.TemporaryDirectory(prefix='ip-load-') as directory:
            root = Path(directory)
            (root/'resolved_config.json').write_text(json.dumps(resolved_config(config)))
            torch.save({'state_dict': source.network.state_dict()}, root/'model.pt')
            # The default eval num_demos=2 must not override this later explicit override.
            policy = load_policy(directory, factories=factories,
                                 overrides=lambda c: replace(c, graph=replace(c.graph,num_demos=1)))
        self.assertEqual(policy.network.num_demos, 1)
        demos = [dict(obs=[np.zeros((4,3)),np.ones((4,3))], grips=[1.,1.], T_w_es=[np.eye(4),np.eye(4)])]
        context = policy.prepare_context(demos, prepared=True)
        with patch.object(preprocessing, 'remove_statistical_outliers',
                          side_effect=lambda points, **kwargs: (points,np.arange(len(points)))):
            trajectory = policy.predict(Observation(np.zeros((4,3)), np.eye(4), 1.), context)
        self.assertEqual(trajectory.transforms.shape, (1,2,4,4))
        self.assertTrue(torch.isfinite(trajectory.transforms).all())


if __name__ == '__main__':
    unittest.main()
