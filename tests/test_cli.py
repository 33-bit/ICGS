import subprocess
import sys
import unittest


class CliTests(unittest.TestCase):
    def test_all_help_paths_work_outside_checkout(self):
        for command in ('infer','train','evaluate','prepare-data'):
            r=subprocess.run([sys.executable,'-m','icgs',command,'--help'],cwd='/tmp',capture_output=True,text=True,timeout=20)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('usage:',r.stdout)

    def test_published_evaluation_rejects_runtime_override_before_loading(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from dataclasses import replace
        from icgs.artifacts.published import published_config
        from icgs.cli.evaluate import run
        config=published_config(device='cpu')
        config=replace(config,runtime=replace(config.runtime,cache_context=True))
        args=SimpleNamespace(config='unused',device=None,num_demos=None,num_rollouts=None,checkpoint='unused')
        with patch('icgs.configuration.loader.load_config',return_value=config):
            with self.assertRaisesRegex(ValueError,'runtime'):run(args)


if __name__=='__main__':unittest.main()
