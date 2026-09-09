import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class _LifecycleSpan:
    def __init__(self, recorder, name):
        self.recorder = recorder
        self.name = name

    def __enter__(self):
        self.recorder.order.append(f"span:{self.name}:enter")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.recorder.order.append(f"span:{self.name}:exit")
        return False


class _LifecycleRecorder:
    run_id = "cli-test-run"

    def __init__(self):
        self.order = []
        self.resolved_calls = []
        self.close_calls = []

    def span(self, name, **kwargs):
        return _LifecycleSpan(self, name)

    def write_resolved_config(self, config, **kwargs):
        self.order.append("resolved")
        self.resolved_calls.append((config, kwargs))

    def close(self, **kwargs):
        self.order.append("close")
        self.close_calls.append(kwargs)
        return SimpleNamespace(status=kwargs["status"])


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

    def test_infer_success_lifecycle_starts_before_work_and_closes_once(self):
        from icgs.cli.infer import run

        recorder = _LifecycleRecorder()
        output = StringIO()
        policy = SimpleNamespace(
            artifact_sha256="checkpoint-hash",
            prepare_context=lambda demos: recorder.order.append("prepare") or "context",
            predict=lambda observation, context: recorder.order.append("predict") or "result",
        )
        args = SimpleNamespace(
            logging_config=None, log_dir=None, log_level=None, trace_mode=None,
            device="cpu", seed=7, input="input.npz", checkpoint="checkpoint.pt",
            num_demos=2, diffusion_steps=4, output="output.npz",
        )
        with patch("icgs.cli.observability.start_cli_run",
                   side_effect=lambda *a, **k: recorder.order.append("startup") or recorder), \
             patch("icgs.data.schemas.inference.load_input", return_value=("observation", ["d1", "d2"])), \
             patch("icgs.artifacts.published.load_published_policy", return_value=policy), \
             patch("icgs.artifacts.published.published_config", return_value="resolved-config"), \
             patch("icgs.data.schemas.inference.save_output",
                   side_effect=lambda *a, **k: recorder.order.append("save")), \
             redirect_stdout(output):
            run(args)
        self.assertEqual(recorder.order[0], "startup")
        self.assertLess(recorder.order.index("startup"), recorder.order.index("span:input.validate:enter"))
        self.assertLess(recorder.order.index("span:input.validate:enter"), recorder.order.index("resolved"))
        self.assertLess(recorder.order.index("resolved"), recorder.order.index("prepare"))
        self.assertLess(recorder.order.index("predict"), recorder.order.index("save"))
        self.assertEqual(len(recorder.resolved_calls), 1)
        self.assertEqual(recorder.close_calls, [{"status": "succeeded", "error": None}])
        self.assertIn('"status": "PASS"', output.getvalue())

    def test_train_success_lifecycle_preserves_result_and_closes_once(self):
        from icgs.cli.train import run

        recorder = _LifecycleRecorder()
        config = SimpleNamespace(runtime=SimpleNamespace(device="cpu"))
        args = SimpleNamespace(
            logging_config=None, log_dir=None, log_level=None, trace_mode=None,
            device="cpu", run_name="test", use_wandb=False,
            config="config.json", data_train="train", data_val="val", checkpoint=None,
        )
        order = recorder.order

        def build(_config):
            order.append("build")
            return "module"

        def training(*args, **kwargs):
            order.append("training")
            return "trained"

        with patch("icgs.cli.observability.start_cli_run",
                   side_effect=lambda *a, **k: recorder.order.append("startup") or recorder), \
             patch("icgs.configuration.loader.load_config", return_value=config), \
             patch("icgs.composition.build_training_module", side_effect=build), \
             patch("icgs.training.runner.run_training", side_effect=training):
            result = run(args)
        self.assertEqual(result, "trained")
        self.assertEqual(order[0], "startup")
        self.assertLess(order.index("resolved"), order.index("build"))
        self.assertLess(order.index("build"), order.index("training"))
        self.assertEqual(len(recorder.resolved_calls), 1)
        self.assertEqual(recorder.close_calls, [{"status": "succeeded", "error": None}])

    def test_prepare_data_success_lifecycle_preserves_save_order(self):
        from icgs.cli.prepare_data import run

        recorder = _LifecycleRecorder()
        config = SimpleNamespace(
            runtime=SimpleNamespace(device="cpu"),
            graph=SimpleNamespace(num_demos=2, traj_horizon=10, pred_horizon=8),
        )
        args = SimpleNamespace(
            logging_config=None, log_dir=None, log_level=None, trace_mode=None,
            device="cpu", config="config.json", input="input.npz", output=None,
            offset=0, cache_embeddings=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            args.output = str(Path(directory) / "prepared")
            demos = ["demo-0", "demo-1", "live"]

            def cond(value, horizon):
                recorder.order.append(f"cond:{value}")
                return f"conditioned-{value}"

            def live(value, *args, **kwargs):
                recorder.order.append(f"live:{value}")
                return "live-sample"

            with patch("icgs.cli.observability.start_cli_run",
                       side_effect=lambda *a, **k: recorder.order.append("startup") or recorder), \
                 patch("icgs.configuration.loader.load_config", return_value=config), \
                 patch("icgs.data.schemas.inference.load_input", return_value=(None, demos)), \
                 patch("icgs.data.preprocessing.native.sample_to_cond_demo", side_effect=cond), \
                 patch("icgs.data.preprocessing.native.sample_to_live", side_effect=live), \
                 patch("icgs.data.preprocessing.native.save_sample",
                       side_effect=lambda *a, **k: recorder.order.append("save")):
                run(args)
        self.assertEqual(recorder.order[0], "startup")
        self.assertLess(recorder.order.index("resolved"), recorder.order.index("cond:demo-0"))
        self.assertLess(recorder.order.index("live:live"), recorder.order.index("save"))
        self.assertEqual(len(recorder.resolved_calls), 1)
        self.assertEqual(recorder.close_calls, [{"status": "succeeded", "error": None}])

    def test_evaluate_success_lifecycle_preserves_stdout_and_work_order(self):
        from icgs.cli.evaluate import run
        from icgs.artifacts.published import published_config

        recorder = _LifecycleRecorder()
        config = published_config(device="cpu")
        policy = SimpleNamespace(artifact_sha256="checkpoint-hash")
        args = SimpleNamespace(
            logging_config=None, log_dir=None, log_level=None, trace_mode=None,
            device="cpu", config="config.json", checkpoint="checkpoint.pt",
            num_demos=None, num_rollouts=None,
        )
        order = recorder.order

        class Environment:
            def __init__(self, *args, **kwargs):
                order.append("environment")

        def evaluate(*args, **kwargs):
            order.append("evaluate")
            return 0.5

        output = StringIO()
        with patch("icgs.cli.observability.start_cli_run",
                   side_effect=lambda *a, **k: recorder.order.append("startup") or recorder), \
             patch("icgs.configuration.loader.load_config", return_value=config), \
             patch("icgs.artifacts.published.published_config", return_value=config), \
             patch("icgs.artifacts.published.load_published_policy", side_effect=lambda *a, **k: order.append("policy") or policy), \
             patch("icgs.environments.rlbench.adapter.RLBenchAdapter", Environment), \
             patch("icgs.evaluation.runner.evaluate_policy", side_effect=evaluate), \
             redirect_stdout(output):
            run(args)
        self.assertEqual(order[0], "startup")
        self.assertLess(order.index("policy"), order.index("resolved"))
        self.assertLess(order.index("resolved"), order.index("environment"))
        self.assertLess(order.index("policy"), order.index("environment"))
        self.assertLess(order.index("environment"), order.index("evaluate"))
        self.assertEqual(len(recorder.resolved_calls), 1)
        self.assertEqual(recorder.close_calls, [{"status": "succeeded", "error": None}])
        self.assertIn("Success rate: 0.5", output.getvalue())


if __name__=='__main__':unittest.main()
