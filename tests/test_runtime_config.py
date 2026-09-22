from pathlib import Path
import tempfile
import json
import unittest


class V5ConfigTests(unittest.TestCase):
    def test_file_profile_override_precedence_and_relative_paths(self):
        from icgs.configuration.loader import load_config
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'experiment.json'
            path.write_text(json.dumps({'schema_version':1,'profile':'eval','config':{
                'sampling':{'steps':6},'scene':{'checkpoint':'encoder.pt'},'runtime':{'device':'cpu'}}}))
            c=load_config(path, overrides={'sampling.steps':3})
            self.assertEqual(c.sampling.steps,3)
            self.assertEqual(c.runtime.batch_size,1)
            self.assertEqual(c.scene.checkpoint,str(Path(d).resolve()/'encoder.pt'))

    def test_unknown_keys_and_ids_fail_without_model_construction(self):
        from icgs.configuration.loader import load_config
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json'
            for payload in [{'config':{'graph':{'typo':1}}}, {'config':{'scene':{'kind':'unknown'}}}]:
                p.write_text(json.dumps(dict(schema_version=1,**payload)))
                with self.assertRaises(ValueError): load_config(p)

    def test_invalid_runtime_and_evaluation_values_fail(self):
        from icgs.configuration.loader import load_config
        for overrides in ({'runtime.cache_context':'false'},{'runtime.live_voxel_size':-1},
                          {'evaluation.num_rollouts':0},{'evaluation.execution_horizon':9}):
            with self.assertRaises(ValueError):load_config(overrides=overrides)

    def test_published_profile_does_not_require_encoder_sidecar(self):
        from icgs.artifacts.published import published_config
        c=published_config(device='cpu')
        self.assertFalse(c.scene.pretrained)
        self.assertEqual(c.runtime.batch_size,1)
        self.assertEqual(c.graph.num_demos,2)
        self.assertEqual(c.runtime.live_voxel_size,.01)
        self.assertFalse(c.runtime.cache_context)


if __name__=='__main__': unittest.main()
