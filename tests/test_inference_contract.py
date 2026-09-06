import unittest
import tempfile
from pathlib import Path
import numpy as np


class InferenceContractTests(unittest.TestCase):
    def test_safe_fixture_load_validates_frames_counts_and_shapes(self):
        from icgs.data.schemas.inference import load_input
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.npz'
            points=np.arange(72,dtype=float).reshape(2,3,4,3)/100
            poses=np.tile(np.eye(4),(2,3,1,1))
            poses[:,:,0,3]=.4
            values=dict(schema_version=np.array(1),demo_points=points,demo_poses=poses,
                        demo_grips=np.ones((2,3)),points=points[0,0],root_pose=poses[0,0],grip=np.array(1.))
            np.savez(path,**values)
            observation,demos=load_input(path)
            self.assertEqual(len(demos),2)
            self.assertEqual(len(demos[0]['pcds']),3)
            self.assertEqual(observation.T_w_e[0,3],.4)
            values['root_pose']=np.zeros((4,4));np.savez(path,**values)
            with self.assertRaisesRegex(ValueError,'pose'): load_input(path)
            values['root_pose']=np.eye(4)[None];np.savez(path,**values)
            with self.assertRaisesRegex(ValueError,'root pose'):load_input(path)

    def test_rng_scope_restores_even_after_exception(self):
        import torch,random
        from icgs.state.randomness import scoped_seed
        a=random.getstate();b=np.random.get_state();c=torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with scoped_seed(17,device='cpu'):
                random.random();np.random.rand();torch.rand(3);raise RuntimeError('test')
        self.assertEqual(random.getstate(),a)
        np.testing.assert_array_equal(np.random.get_state()[1],b[1])
        self.assertTrue(torch.equal(torch.get_rng_state(),c))

    @unittest.skipUnless(__import__('torch').cuda.is_available(),'CUDA RNG test requires CUDA')
    def test_cpu_scope_does_not_mutate_cuda_rng(self):
        import torch
        from icgs.state.randomness import scoped_seed
        state=torch.cuda.get_rng_state().clone()
        with scoped_seed(18,device='cpu'):torch.rand(3)
        self.assertTrue(torch.equal(state,torch.cuda.get_rng_state()))


if __name__=='__main__': unittest.main()
