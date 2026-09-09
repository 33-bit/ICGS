import importlib
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


class TrainingObservabilityTests(unittest.TestCase):
    def test_local_lightning_bridge_preserves_metric_names_and_axes(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.lightning import RecorderLightningLogger
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            config = from_mapping({"output_dir": directory, "metric_every_steps": 1,
                                   "flush_interval_s": 0.01, "shutdown_timeout_s": 1.0})
            recorder = RunRecorder.start(config, command="train")
            logger = RecorderLightningLogger(recorder)
            logger.log_metrics({"Train_Loss": 1.25}, step=3)
            recorder.close()
            self.assertEqual(logger.name, "icgs-local")
            self.assertEqual(logger.version, recorder.run_id)
            self.assertTrue(any(path.read_text().find("Train_Loss") >= 0
                                for path in (recorder.path / "metrics").glob("*.jsonl")))

    def test_disabled_wandb_does_not_import_sdk(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.wandb import start

        config = from_mapping({"wandb": {"enabled": False}}).wandb
        with patch("importlib.import_module", side_effect=AssertionError("must stay lazy")):
            self.assertIsNone(start(config, run_id="run"))

    def test_requested_missing_wandb_is_a_startup_error(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.wandb import start

        config = from_mapping({"wandb": {"enabled": True}}).wandb
        with patch("importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(RuntimeError, "W&B"):
                start(config, run_id="run")

    def test_non_rank0_wandb_skips_sdk_and_preserves_local_recorder(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.recorder import RunRecorder
        from icgs.observability.wandb import start

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"RANK": "1"}):
            config = from_mapping({"output_dir": directory, "wandb": {"enabled": True}})
            recorder = RunRecorder.start(config, command="train")
            try:
                with patch("importlib.import_module", side_effect=AssertionError("rank 1 must not import W&B")):
                    self.assertIsNone(start(config.wandb, run_id=recorder.run_id, recorder=recorder))
            finally:
                recorder.close()

    def test_wandb_allowlist_preserves_native_training_metric_names(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.wandb import WandbAdapter

        received = []
        delivered = threading.Event()

        class Run:
            def log(self, values, **kwargs):
                received.append(dict(values))
                delivered.set()

            def finish(self):
                return None

        config = from_mapping({"wandb": {"enabled": True}}).wandb
        adapter = WandbAdapter(Run(), config)
        adapter.log({"Train_Loss": 1.0, "Val_Grip_Loss": 2.0,
                     "train.loss": 3.0, "unapproved": 4.0}, step=2)
        self.assertTrue(delivered.wait(timeout=2))
        adapter.close()
        self.assertEqual(received, [{"Train_Loss": 1.0, "Val_Grip_Loss": 2.0,
                                     "train.loss": 3.0}])

    def test_wandb_mirror_disables_after_first_transport_failure(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.wandb import WandbAdapter

        class Recorder:
            def __init__(self):
                self.events = []

            def event(self, *args, **kwargs):
                self.events.append((args, kwargs))

        class FailingRun:
            def __init__(self):
                self.log_calls = 0
                self.finish_calls = 0

            def log(self, values, **kwargs):
                self.log_calls += 1
                raise OSError("transport down")

            def finish(self):
                self.finish_calls += 1

        run = FailingRun()
        recorder = Recorder()
        adapter = WandbAdapter(run, from_mapping({"wandb": {"enabled": True}}).wandb,
                                recorder=recorder)
        adapter.log({"train.loss": 1.0})
        deadline = time.monotonic() + 2
        while not adapter.disabled and time.monotonic() < deadline:
            time.sleep(0.01)
        adapter.log({"train.loss": 2.0})
        adapter.close()
        self.assertTrue(adapter.disabled)
        self.assertEqual(run.log_calls, 1)
        self.assertTrue(any(args and args[0] == "wandb.mirror.error" for args, _ in recorder.events))

    def test_wandb_close_bounds_a_blocked_log_and_preserves_operational_error(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.wandb import WandbAdapter

        entered = threading.Event()
        release = threading.Event()

        class Run:
            def log(self, values, **kwargs):
                entered.set()
                release.wait(timeout=1.0)

            def finish(self):
                return None

        recorder = type("Recorder", (), {"events": [], "event": lambda self, *a, **k: self.events.append((a, k))})()
        adapter = WandbAdapter(Run(), from_mapping({"wandb": {"enabled": True}}).wandb,
                               recorder=recorder, timeout_s=0.02)
        adapter.log({"train.loss": 1.0})
        self.assertTrue(entered.wait(timeout=1.0))
        original = RuntimeError("controller failure")
        started = time.monotonic()
        with self.assertRaises(RuntimeError) as raised:
            try:
                raise original
            finally:
                adapter.close()
        elapsed = time.monotonic() - started
        self.assertIs(raised.exception, original)
        self.assertLess(elapsed, 0.5)
        self.assertTrue(adapter.disabled)
        release.set()
        adapter._thread.join(timeout=1.0)

    def test_wandb_finish_runs_in_worker_after_logs_and_close_is_bounded(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.wandb import WandbAdapter

        finish_entered = threading.Event()
        finish_release = threading.Event()
        log_entered = threading.Event()
        order = []

        class Run:
            def log(self, values, **kwargs):
                order.append("log")
                log_entered.set()

            def finish(self):
                order.append("finish")
                finish_entered.set()
                finish_release.wait(timeout=1.0)

        adapter = WandbAdapter(Run(), from_mapping({"wandb": {"enabled": True}}).wandb,
                               timeout_s=0.2)
        adapter.log({"train.loss": 1.0})
        self.assertTrue(log_entered.wait(timeout=1.0))
        started = time.monotonic()
        adapter.close()
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.5)
        self.assertTrue(finish_entered.is_set())
        self.assertEqual(order, ["log", "finish"])
        self.assertTrue(adapter.disabled)
        finish_release.set()
        adapter._thread.join(timeout=1.0)

    @unittest.skipIf(importlib.util.find_spec("lightning") is None,
                     "SKIPPED: Lightning unavailable for native Trainer construction")
    def test_native_record_use_wandb_combinations_construct_trainer_without_fit(self):
        import lightning.pytorch as L

        from icgs.observability.config import from_mapping
        from icgs.observability.recorder import NoopRecorder, RunRecorder
        from icgs.training.runner import build_training_logger

        class Remote:
            def close(self):
                return None

        for record in (False, True):
            for use_wandb in (False, True):
                with self.subTest(record=record, use_wandb=use_wandb):
                    with tempfile.TemporaryDirectory() as directory:
                        recorder = (RunRecorder.start(from_mapping({
                            "output_dir": directory,
                            "flush_interval_s": 0.01,
                            "shutdown_timeout_s": 1.0,
                        }), command="trainer-construction") if record else NoopRecorder())
                        remote = None
                        try:
                            with patch("icgs.observability.wandb.start",
                                       return_value=Remote()) as start_wandb:
                                logger, remote = build_training_logger(
                                    record=record,
                                    use_wandb=use_wandb,
                                    recorder=recorder,
                                    run_name="trainer-construction",
                                )
                                trainer = L.Trainer(
                                    accelerator="cpu",
                                    devices=1,
                                    max_steps=1,
                                    logger=logger,
                                    enable_checkpointing=False,
                                    enable_progress_bar=False,
                                )
                            self.assertIsNotNone(trainer)
                            self.assertEqual(logger is not None, record)
                            self.assertEqual(start_wandb.called, record and use_wandb)
                        finally:
                            if remote is not None:
                                remote.close()
                            if record:
                                recorder.close()


if __name__ == "__main__":
    unittest.main()
