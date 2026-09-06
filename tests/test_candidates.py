from types import SimpleNamespace
import unittest
import numpy as np
import torch
from icgs.contracts.records import Observation,ActionTrajectory


class CandidateTests(unittest.TestCase):
    def test_absolute_prefix_uses_single_nontrivial_root(self):
        from icgs.execution.commands import absolute_prefix
        root=np.eye(4);root[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];root[:3,3]=[1,2,3]
        actions=torch.eye(4).repeat(1,2,1,1);actions[0,0,0,3]=.1;actions[0,1,0,3]=.2
        result=absolute_prefix(ActionTrajectory(actions,torch.ones(1,2,1)),root,2)
        np.testing.assert_allclose(result.targets[0,:,:3,3],[[1,2.1,3],[1,2.2,3]],atol=1e-6)
        self.assertIsNone(result.command_duration_seconds)
        with self.assertRaises(ValueError):absolute_prefix(ActionTrajectory(actions,torch.ones(1,2,1)),root,3)

    def test_sequential_k_and_rng_restore(self):
        from icgs.algorithms.planning.candidates import propose_candidates
        class Policy:
            runtime=SimpleNamespace(device='cpu')
            artifact_sha256='fixture'
            def validate_context(self,c): pass
            def predict(self,o,c):
                t=torch.eye(4).repeat(1,2,1,1);t[:,:,:3,3]=torch.rand(1,2,3)
                return ActionTrajectory(t,torch.ones(1,2,1))
        policy=Policy();observation=Observation(np.ones((3,3)),np.eye(4),1.)
        context=SimpleNamespace(source_id='demo-fixture')
        before=torch.get_rng_state().clone()
        candidates=propose_candidates(policy,observation,context,count=3,seeds=[1,2,3])
        self.assertEqual([c.index for c in candidates],[0,1,2])
        self.assertEqual([c.seed for c in candidates],[1,2,3])
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        one=propose_candidates(policy,observation,context,count=1,seeds=[1])
        torch.testing.assert_close(one[0].trajectory.transforms,candidates[0].trajectory.transforms)
        self.assertEqual(candidates[0].batch_size,1);self.assertEqual(candidates[0].horizon,2)
        with self.assertRaises(ValueError):propose_candidates(policy,observation,context,count=2,seeds=[1])


if __name__=='__main__':unittest.main()
