import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch


class ObservabilityConfigurationTests(unittest.TestCase):
    def test_cli_dotenv_enables_wandb_without_overwriting_exported_credentials(self):
        """A project-local opt-in config must not replace shell credentials."""
        from types import SimpleNamespace

        from icgs.cli.observability import start_cli_run

        args = SimpleNamespace(logging_config=None, log_dir=None, log_level=None,
                               trace_mode=None)
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"WANDB_API_KEY": "exported-key"}, clear=False), \
             patch("icgs.cli.observability.RunRecorder.start") as start:
            Path(directory, ".env").write_text(
                "ICGS_WANDB_ENABLED=true\n"
                "ICGS_WANDB_MODE=online\n"
                "ICGS_WANDB_PROJECT=icgs-local\n"
                "WANDB_API_KEY=dotenv-key\n",
                encoding="utf-8",
            )
            try:
                os.chdir(directory)
                start_cli_run(args, "infer")
                credential = os.environ["WANDB_API_KEY"]
            finally:
                os.chdir(previous_cwd)

        config = start.call_args.args[0]
        self.assertTrue(config.wandb.enabled)
        self.assertEqual(config.wandb.mode, "online")
        self.assertEqual(config.wandb.project, "icgs-local")
        self.assertEqual(credential, "exported-key")

    def test_primary_defaults_are_typed_and_logging_is_separate_from_reference_identity(self):
        from icgs.configuration.method import MethodConfig
        from icgs.artifacts.method import reference_fingerprint

        config = MethodConfig()
        self.assertEqual(config.observability.mode, "normal")
        self.assertFalse(config.observability.wandb.enabled)
        self.assertEqual(config.observability.queue_capacity, 8192)

        payload = {
            "ip_checksum": "a" * 64,
            "native_profile": config.native_profile,
            "geometry": {"num_anchors": config.geometry.num_anchors},
            "physical_weights": {"config": {"geometry": {"num_anchors": config.geometry.num_anchors}}},
            "event_weights": {"id": "event"},
            "task_weights": {"id": "task"},
            "segmentation": {"id": "seg"},
            "router": {"config": config.router.to_dict() if hasattr(config.router, "to_dict") else config.to_dict()["router"]},
            "preprocessing": {"num_anchors": config.geometry.num_anchors, "voxel_size_m": config.geometry.voxel_size_m,
                              "num_points": config.geometry.num_points, "neighbors": config.geometry.neighbors,
                              "ell0_m": config.geometry.ell0_m, "fps_start": config.geometry.fps_start,
                              "tie_break": config.geometry.tie_break},
            "calibration": {"id": "cal"},
            "camera": {"id": "camera"},
            "gravity": {"id": "gravity"},
            "workspace": {"id": "workspace"},
            "cadence": {"dt0": config.control.dt0, "h": config.planning.h,
                        "r": config.planning.r, "H": config.planning.H},
            "rng_protocol": {"config": {"generator_seed": config.stages.generator_seed,
                                          "reset_seed": config.stages.reset_seed,
                                          "action_seed": config.stages.action_seed}},
        }
        baseline = reference_fingerprint(payload)
        changed = MethodConfig.from_dict({"observability": {"metric_every_steps": 101}})
        self.assertNotEqual(changed.fingerprint(), config.fingerprint())
        self.assertEqual(reference_fingerprint(payload), baseline)

    def test_invalid_observability_values_fail_closed(self):
        from icgs.configuration.method import MethodConfig

        for update in (
            {"mode": "unsupported"},
            {"console_level": "TRACE"},
            {"queue_capacity": 0},
            {"reserved_critical_slots": 8192},
            {"capture_total_bytes": 1, "capture_max_bytes_each": 2},
            {"wandb": {"mode": "remote"}},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                MethodConfig.from_dict({"observability": update})


class ObservabilityContextTests(unittest.TestCase):
    def test_nested_bind_and_span_restore_parent_and_do_not_cross_threads(self):
        from icgs.observability.context import current_context
        from icgs.observability.recorder import NoopRecorder

        recorder = NoopRecorder()
        self.assertIsNone(current_context().episode_id)
        with recorder.bind(episode_id="e1", decision_id="d2") as bound:
            self.assertEqual(current_context().episode_id, "e1")
            with bound.span("policy.propose", component="proposal") as span:
                self.assertEqual(current_context().decision_id, "d2")
                self.assertEqual(current_context().span_id, span.span_id)
            self.assertIsNone(current_context().span_id)
            seen = []
            thread = threading.Thread(target=lambda: seen.append(current_context().episode_id))
            thread.start()
            thread.join()
            self.assertEqual(seen, [None])
        self.assertIsNone(current_context().episode_id)

    def test_exception_from_span_is_reraised_and_parent_context_is_restored(self):
        from icgs.observability.context import current_context
        from icgs.observability.recorder import NoopRecorder

        recorder = NoopRecorder()
        with recorder.bind(branch_id="b1"):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with recorder.span("failure", component="test"):
                    raise RuntimeError("boom")
            self.assertEqual(current_context().branch_id, "b1")
            self.assertIsNone(current_context().span_id)

    def test_span_logging_failure_does_not_block_work_or_mask_original_exception(self):
        from icgs.observability.context import current_context
        from icgs.observability.config import from_mapping
        from icgs.observability.recorder import RunRecorder

        class UnprintableError(RuntimeError):
            def __str__(self):
                raise OSError("formatting failed")

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory, "flush_interval_s": 0.01,
                "shutdown_timeout_s": 1.0,
            }), command="fault-isolation")
            operational = UnprintableError()
            with patch.object(recorder, "_emit", side_effect=OSError("writer failed")):
                with self.assertRaises(UnprintableError) as raised:
                    with recorder.bind(branch_id="branch").span("operation", component="test"):
                        raise operational
            self.assertIs(raised.exception, operational)
            self.assertEqual(current_context().branch_id, None)
            self.assertIsNone(current_context().span_id)
            recorder.close(status="failed", error=operational)

    def test_span_entry_failure_does_not_block_work_after_startup(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory,
                "flush_interval_s": 0.01,
                "shutdown_timeout_s": 1.0,
            }), command="span-entry-fault")
            completed = []
            with patch("icgs.observability.recorder._new_id",
                       side_effect=OSError("uuid source failed")):
                with recorder.span("operation", component="test"):
                    completed.append("controller-work")
            self.assertEqual(completed, ["controller-work"])
            self.assertGreaterEqual(recorder.writer.stats["sink_errors"], 1)
            recorder.close()

    def test_exception_formatting_failure_is_recorded_without_replacing_error(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.recorder import RunRecorder

        class UnprintableError(RuntimeError):
            def __str__(self):
                raise OSError("formatting failed")

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory, "flush_interval_s": 0.01,
                "shutdown_timeout_s": 1.0,
            }), command="exception-format")
            error = UnprintableError()
            recorder.exception(error, component="test")
            summary = recorder.close(status="failed", error=error)
            self.assertEqual(summary.status, "failed")
            self.assertTrue((Path(recorder.path) / "summary.json").exists())

    def test_capture_failure_does_not_replace_operational_error(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.recorder import RunRecorder

        class FailingCapture:
            def capture(self, *args, **kwargs):
                raise OSError("capture disk full")

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory, "mode": "capture", "capture_arrays": True,
                "flush_interval_s": 0.01, "shutdown_timeout_s": 1.0,
            }), command="capture-fault")
            recorder._capture_manager = FailingCapture()
            original = RuntimeError("control failure")
            try:
                with self.assertRaises(RuntimeError) as raised:
                    try:
                        raise original
                    except RuntimeError as error:
                        recorder.capture("failure", {}, metadata={"kind": "test"})
                        raise
                self.assertIs(raised.exception, original)
            finally:
                recorder.close(status="failed", error=original)

    def test_noop_span_does_not_change_context_or_touch_lazy_fields(self):
        from icgs.observability.context import current_context
        from icgs.observability.recorder import NoopRecorder

        class ExplodingFields:
            def items(self):
                raise AssertionError("lazy fields must not be inspected")

        recorder = NoopRecorder()
        before = current_context()
        with recorder.bind(episode_id="episode").span(
                "ignored", component="test", fields=ExplodingFields()) as span:
            self.assertIsNone(span.span_id)
            self.assertEqual(current_context(), before)
        self.assertEqual(current_context(), before)

    def test_file_threshold_skips_debug_fields_before_serialization(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.recorder import RunRecorder

        class LazyFields:
            def __init__(self):
                self.touched = False

            def items(self):
                self.touched = True
                return (("value", 1),)

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory, "file_level": "INFO",
                "flush_interval_s": 0.01, "shutdown_timeout_s": 1.0,
            }), command="threshold")
            fields = LazyFields()
            recorder.event("debug.detail", level="DEBUG", component="test", fields=fields)
            recorder.close()
            self.assertFalse(fields.touched)
            records = []
            for path in (Path(recorder.path) / "events").glob("*.jsonl"):
                records.extend(json.loads(line) for line in path.read_text().splitlines())
            self.assertFalse(any(item["event"] == "debug.detail" for item in records))

    def test_resolved_runtime_record_is_full_sanitized_and_single_write(self):
        from icgs.configuration.method import MethodConfig
        from icgs.observability.config import from_mapping
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory, "flush_interval_s": 0.01,
                "shutdown_timeout_s": 1.0,
            }), command="identity")
            runtime = MethodConfig()
            recorder.write_resolved_config(
                runtime,
                metadata={"request": {"token": "hidden"},
                          "identities": {"checkpoint_sha256": "a" * 64,
                                          "dataset_id": "dataset-fixture"}},
                unavailable={"reference": "not supplied by this boundary"},
            )
            payload = json.loads((Path(recorder.path) / "resolved_config.json").read_text())
            self.assertIn("geometry", payload["config"])
            self.assertEqual(payload["identities"]["dataset_id"], "dataset-fixture")
            self.assertEqual(payload["unavailable"]["reference"], "not supplied by this boundary")
            self.assertEqual(payload["metadata"]["request"]["token"], "<redacted>")
            with self.assertRaisesRegex(RuntimeError, "already"):
                recorder.write_resolved_config(runtime)
            recorder.close()

    def test_failed_resolved_runtime_write_is_not_replaced_by_logging_fallback(self):
        from icgs.configuration.method import MethodConfig
        from icgs.observability.config import from_mapping
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory,
                "flush_interval_s": 0.01,
                "shutdown_timeout_s": 1.0,
            }), command="resolved-write-failure")
            with patch("icgs.observability.recorder._atomic_write",
                       side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    recorder.write_resolved_config(MethodConfig())
            summary = recorder.close(status="failed", error=RuntimeError("operational"))
            self.assertFalse(summary.logs_complete)
            self.assertFalse((Path(recorder.path) / "resolved_config.json").exists())


class SerializationTests(unittest.TestCase):
    def test_redacts_sensitive_descendants_and_url_queries_without_repr(self):
        from icgs.observability.serialization import safe_value

        value = safe_value({"token": "secret", "nested": {"api_key": "hidden"},
                            "url": "https://example.invalid/path?token=secret&x=1"})
        self.assertEqual(value["token"], "<redacted>")
        self.assertEqual(value["nested"]["api_key"], "<redacted>")
        self.assertIn("token=%3Credacted%3E", value["url"])
        json.dumps(value, allow_nan=False)

    def test_nonfinite_metric_is_explicitly_invalid(self):
        from icgs.observability.serialization import metric_value

        self.assertEqual(metric_value(float("nan")), {"value": None, "nonfinite": True})
        self.assertEqual(metric_value(2.5), {"value": 2.5, "nonfinite": False})


class LoggingConfigAdapterTests(unittest.TestCase):
    def test_logging_only_envelope_resolves_paths_and_rejects_unknown_fields(self):
        from icgs.observability.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logging.json"
            path.write_text(json.dumps({"schema_version": 1, "observability": {
                "mode": "debug", "output_dir": "runs", "metric_every_steps": 3,
            }}))
            config = load_config(path)
            self.assertEqual(config.mode, "debug")
            self.assertEqual(config.metric_every_steps, 3)
            self.assertEqual(config.output_dir, str((Path(directory) / "runs").resolve()))
            path.write_text(json.dumps({"schema_version": 1, "observability": {"unknown": True}}))
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
