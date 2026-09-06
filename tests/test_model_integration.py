"""Opt-in published native artifact acceptance smoke; no fabricated sidecars."""
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('ICGS_RUN_PUBLISHED')=='1', 'SKIPPED: opt-in published checkpoint integration')
class ModelIntegrationTests(unittest.TestCase):
    def test_published_checkpoint_load_and_real_inference(self):
        from icgs.artifacts.published import load_published_policy
        from icgs.data.schemas.inference import load_input
        from icgs.state.randomness import scoped_seed
        import torch
        checkpoint=os.environ['ICGS_CHECKPOINT']
        fixture=os.environ['ICGS_FIXTURE']
        policy=load_published_policy(checkpoint,device='cuda')
        observation,demos=load_input(fixture)
        with scoped_seed(17,device='cuda'):
            context=policy.prepare_context(demos)
            result=policy.predict(observation,context)
        self.assertEqual(result.transforms.shape,(1,8,4,4))
        self.assertTrue(torch.isfinite(result.transforms).all())
        self.assertFalse(result.transforms.requires_grad)
        self.assertFalse(policy.checkpoint_report.missing)
        self.assertFalse(policy.checkpoint_report.unexpected)


if __name__=='__main__':unittest.main()
