"""Opt-in full research stack/asset checks; no download or simulator side effects."""
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('IP_RUN_MODEL_TESTS') == '1', 'SKIPPED: opt-in L2 requires IP_RUN_MODEL_TESTS=1')
class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        required = ['torch', 'torch_geometric', 'torch_cluster', 'diffusers', 'lightning', 'open3d']
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            self.skipTest('SKIPPED: missing real model dependencies: ' + ', '.join(missing))

    def test_published_checkpoint_load_and_real_synthetic_inference(self):
        import torch
        from ip.composition import load_policy
        directory = os.environ.get('IP_CHECKPOINT_DIR')
        if not directory:
            self.skipTest('SKIPPED: trusted IP_CHECKPOINT_DIR not supplied')
        required = ('config.pkl', 'model.pt', 'scene_encoder.pt')
        absent = [name for name in required if not (Path(directory) / name).is_file()]
        if absent:
            self.skipTest('SKIPPED: missing checkpoint assets: ' + ', '.join(absent))
        if not torch.cuda.is_available():
            self.skipTest('SKIPPED: original CUDA model integration requires CUDA')
        from torch_geometric.data import Data
        # Absolute encoder path is an explicit artifact-location override only.
        policy = load_policy(directory, overrides=lambda c: replace(c, scene=replace(c.scene, checkpoint=str(Path(directory) / 'scene_encoder.pt'))))
        d, t, p = policy.graph_config.num_demos, policy.graph_config.traj_horizon, policy.graph_config.pred_horizon
        torch.manual_seed(123)
        data = Data(pos_demos=torch.randn(d*t*2048,3)*.05,
                    batch_demos=torch.arange(d*t).repeat_interleave(2048),
                    pos_obs=torch.randn(2048,3)*.05, batch_pos_obs=torch.zeros(2048,dtype=torch.long),
                    graps_demos=torch.ones(1,d,t,1), demo_T_w_es=torch.eye(4).repeat(1,d,t,1,1),
                    current_grip=torch.ones(1), actions=torch.eye(4).repeat(1,p,1,1),
                    actions_grip=torch.ones(1,p), T_w_e=torch.eye(4).unsqueeze(0))
        result = policy.predict_batch(data)
        self.assertEqual(result.transforms.shape, (1,p,4,4))
        self.assertTrue(torch.isfinite(result.transforms).all())
        self.assertEqual(result.grips.shape, (1,p,1))
        torch.testing.assert_close(result.transforms[...,3,:], torch.tensor([0.,0.,0.,1.], device=result.transforms.device).expand(1,p,4))


if __name__ == '__main__':
    unittest.main()
