import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


class LocalRecorderTests(unittest.TestCase):
    def _config(self, root, **updates):
        from icgs.observability.config import from_mapping
        base = {
            "mode": "normal", "output_dir": str(root), "queue_capacity": 8,
            "reserved_critical_slots": 2, "flush_interval_s": 0.01,
            "shutdown_timeout_s": 1.0, "event_chunk_bytes": 512,
            "max_event_chunks_per_process": 2, "metrics_total_bytes_per_process": 2048,
            "max_record_bytes": 2048, "recent_event_capacity": 4,
        }
        base.update(updates)
        return from_mapping(base)

    def test_run_is_exclusive_and_summary_separates_status_from_completeness(self):
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            first = RunRecorder.start(config, command="test", metadata={"seed": 17})
            second = RunRecorder.start(config, command="test", metadata={"seed": 29})
            self.assertNotEqual(first.run_id, second.run_id)
            first.event("one", component="test")
            first.close(status="succeeded")
            second.close(status="failed", error=RuntimeError("expected"))
            for recorder, status in ((first, "succeeded"), (second, "failed")):
                summary = json.loads((Path(recorder.path) / "summary.json").read_text())
                self.assertEqual(summary["status"], status)
                self.assertIn("logs_complete", summary)
                self.assertTrue((Path(recorder.path) / "run.json").exists())
                self.assertTrue((Path(recorder.path) / "resolved_config.json").exists())

    def test_nested_span_records_parent_and_duration(self):
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(
                directory, event_chunk_bytes=4096, file_level="DEBUG"), command="trace")
            with recorder.bind(episode_id="e1", decision_id="d1").span("outer", component="proposal") as outer:
                outer.event("candidate.created", component="proposal", fields={"index": 0})
            recorder.close()
            records = []
            for path in sorted((Path(recorder.path) / "events").glob("*.jsonl")):
                records.extend(json.loads(line) for line in path.read_text().splitlines())
            candidate = next(item for item in records if item["event"] == "candidate.created")
            end = next(item for item in records if item["event"] == "span.end")
            self.assertEqual(candidate["episode_id"], "e1")
            self.assertEqual(candidate["decision_id"], "d1")
            self.assertEqual(candidate["span_id"], outer.span_id)
            self.assertGreaterEqual(end["fields"]["duration_ns"], 0)

    def test_mode_off_does_not_touch_files_or_inspect_lazy_fields(self):
        from icgs.observability.recorder import RunRecorder

        class Explodes:
            def __repr__(self):
                raise AssertionError("must not be inspected")

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(directory, mode="off"), command="off")
            recorder.event("ignored", component="test", fields={"lazy": Explodes()})
            self.assertIsNone(recorder.run_id)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_queue_saturation_drops_detail_but_keeps_failure_and_original_exception(self):
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(directory, queue_capacity=2,
                                                       reserved_critical_slots=1,
                                                       flush_interval_s=10), command="saturation")
            for index in range(20):
                recorder.event(f"detail.{index}", component="test")
            with self.assertRaisesRegex(RuntimeError, "original"):
                try:
                    raise RuntimeError("original")
                except RuntimeError as error:
                    recorder.exception(error, component="test")
                    raise
            summary = recorder.close(status="failed", error=RuntimeError("original"))
            self.assertGreater(summary.dropped_count, 0)
            self.assertFalse(summary.logs_complete)

    def test_short_write_is_accounted_as_an_incomplete_sink(self):
        from icgs.observability.local import LocalWriter, PendingRecord

        class ShortWrite:
            closed = False

            def write(self, payload):
                return max(0, len(payload) - 1)

            def flush(self):
                return None

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, event_chunk_bytes=4096)
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            writer = LocalWriter(run_dir, config, process_id=1, rank=0)
            writer._event_stream.close()
            writer._event_stream = ShortWrite()
            writer._write_line(PendingRecord("event", {"event": "short"}, False))
            stats = writer.stats
            self.assertEqual(stats["short_writes"], 1)
            self.assertEqual(stats["sink_errors"], 1)
            self.assertTrue(writer.sink_disabled)
            writer.close(0.1)

    def test_disk_full_write_is_accounted_as_an_incomplete_sink(self):
        from icgs.observability.local import LocalWriter, PendingRecord

        class DiskFull:
            closed = False

            def write(self, payload):
                raise OSError("disk full")

            def flush(self):
                return None

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, event_chunk_bytes=4096)
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            writer = LocalWriter(run_dir, config, process_id=1, rank=0)
            writer._event_stream.close()
            writer._event_stream = DiskFull()
            writer._write_line(PendingRecord("event", {"event": "disk-full"}, False))
            stats = writer.stats
            self.assertTrue(writer.sink_disabled)
            self.assertEqual(stats["sink_errors"], 1)
            self.assertEqual(stats["short_writes"], 0)
            writer.close(0.1)

    def test_flush_and_close_failures_are_accounted_without_killing_shutdown(self):
        from icgs.observability.local import LocalWriter

        class FlushAndCloseFailure:
            closed = False

            def write(self, payload):
                return len(payload)

            def flush(self):
                raise OSError("flush failed")

            def close(self):
                self.closed = True
                raise OSError("close failed")

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            writer = LocalWriter(run_dir, config, process_id=1, rank=0)
            writer._event_stream.close()
            writer._metric_stream.close()
            writer._event_stream = FlushAndCloseFailure()
            writer._metric_stream = FlushAndCloseFailure()
            writer._flush()
            self.assertTrue(writer.sink_disabled)
            self.assertGreaterEqual(writer.stats["flush_errors"], 1)
            writer.close(0.1)
            self.assertGreaterEqual(writer.stats["close_errors"], 1)

    def test_periodic_flush_failure_is_accounted_and_writer_thread_still_stops(self):
        from icgs.observability.local import LocalWriter, PendingRecord

        class PeriodicFlushFailure:
            def __init__(self, stream, flushed):
                self.stream = stream
                self.flushed = flushed

            @property
            def closed(self):
                return self.stream.closed

            def write(self, payload):
                return self.stream.write(payload)

            def flush(self):
                self.flushed.set()
                raise OSError("periodic flush failed")

            def close(self):
                return self.stream.close()

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, flush_interval_s=0.01)
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            writer = LocalWriter(run_dir, config, process_id=1, rank=0)
            flushed = threading.Event()
            event_stream = writer._event_stream
            metric_stream = writer._metric_stream
            writer._event_stream = PeriodicFlushFailure(event_stream, flushed)
            writer._metric_stream = PeriodicFlushFailure(metric_stream, flushed)
            writer._thread.start()
            self.assertTrue(writer.enqueue("event", {"event": "periodic"}))
            self.assertTrue(flushed.wait(timeout=2))
            self.assertGreaterEqual(writer.stats["flush_errors"], 1)
            self.assertTrue(writer.close(0.5))

    def test_metric_cap_loss_makes_summary_incomplete(self):
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(
                directory, metric_every_steps=1, metrics_total_bytes_per_process=1),
                command="metric-cap")
            recorder.metrics({"loss": 1.0}, axis="optimizer_step", step=0, component="test")
            summary = recorder.close()
            self.assertFalse(summary.logs_complete)
            self.assertGreaterEqual(summary.capture_skipped_count, 0)
            payload = json.loads((Path(recorder.path) / "summary.json").read_text())
            self.assertGreater(payload["dropped"]["metrics_skipped"], 0)
            self.assertFalse(payload["logs_complete"])

    def test_critical_output_has_a_finite_emergency_bound(self):
        from icgs.observability.local import LocalWriter, PendingRecord

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, event_chunk_bytes=128,
                                  max_event_chunks_per_process=1,
                                  max_record_bytes=256, reserved_critical_slots=1)
            run_dir = Path(directory) / "run"
            writer = LocalWriter(run_dir, config, process_id=1, rank=0)
            for index in range(20):
                writer._write_line(PendingRecord(
                    "event", {"event": "failure", "index": index,
                              "message": "x" * 32}, True))
            total = sum(path.stat().st_size for path in (run_dir / "events").glob("*.jsonl"))
            emergency = (config.max_record_bytes + 1) * config.reserved_critical_slots
            self.assertLessEqual(total, config.event_chunk_bytes * config.max_event_chunks_per_process + emergency)
            self.assertGreater(writer.stats["dropped_critical"], 0)
            writer.close(0.1)

    def test_summary_write_failure_does_not_replace_operational_error(self):
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(directory), command="summary-fault")
            original = RuntimeError("operational")
            with patch("icgs.observability.recorder._atomic_write",
                       side_effect=OSError("disk full")):
                summary = recorder.close(status="failed", error=original)
            self.assertFalse(summary.logs_complete)
            self.assertEqual(summary.status, "failed")
            self.assertFalse((Path(recorder.path) / "summary.json").exists())

    def test_unstarted_writer_close_is_bounded_when_stream_close_blocks(self):
        from icgs.observability.local import LocalWriter

        class BlockingClose:
            def __init__(self):
                self.release = threading.Event()
                self.closed = False

            def flush(self):
                return None

            def close(self):
                self.release.wait(timeout=2.0)
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            writer = LocalWriter(Path(directory) / "run", self._config(directory),
                                 process_id=1, rank=0)
            blocking = BlockingClose()
            writer._event_stream.close()
            writer._event_stream = blocking
            started = time.monotonic()
            completed = writer.close(0.01)
            elapsed = time.monotonic() - started
            self.assertFalse(completed)
            self.assertLess(elapsed, 0.5)
            blocking.release.set()


if __name__ == "__main__":
    unittest.main()
