import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np


class CaptureTests(unittest.TestCase):
    def _config(self, root, **updates):
        from icgs.observability.config import from_mapping
        values = {
            "mode": "capture", "output_dir": str(root), "capture_arrays": True,
            "capture_max_count": 2, "capture_max_bytes_each": 4096,
            "capture_total_bytes": 8192, "flush_interval_s": 0.01,
            "shutdown_timeout_s": 1.0,
        }
        values.update(updates)
        return from_mapping(values)

    def test_numeric_capture_is_copied_and_roundtrips_without_pickle(self):
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(directory), command="capture")
            submitted = np.arange(8, dtype=np.float32)
            result = recorder.capture("fixture", {"points": submitted}, metadata={"kind": "test"})
            submitted[:] = -1
            self.assertTrue(result.captured)
            recorder.close()
            manifest = json.loads((Path(result.path) / "manifest.json").read_text())
            self.assertEqual(manifest["arrays"]["points"]["dtype"], "float32")
            with np.load(Path(result.path) / "arrays.npz", allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["points"], np.arange(8, dtype=np.float32))

    def test_capture_rejects_object_and_non_numpy_arrays_without_device_transfer(self):
        from icgs.observability.recorder import RunRecorder

        class GPUValue:
            def cpu(self):
                raise AssertionError("capture must not transfer a device value")

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(directory), command="capture")
            self.assertEqual(recorder.capture("bad", {"object": np.array([object()], dtype=object)}).reason,
                             "object_or_non_numeric_array")
            self.assertEqual(recorder.capture("bad", {"gpu": GPUValue()}).reason,
                             "cpu_numpy_arrays_required")
            recorder.close()

    def test_oversized_numpy_capture_is_rejected_before_copying(self):
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(
                directory, capture_max_bytes_each=16, capture_total_bytes=32),
                command="capture-size")
            oversized = np.arange(32, dtype=np.float32)
            with patch("icgs.observability.capture.np.array",
                       side_effect=AssertionError("oversized input must not be copied")):
                result = recorder.capture("oversized", {"x": oversized})
            self.assertFalse(result.captured)
            self.assertEqual(result.reason, "capture_size_quota")
            recorder.close()

    def test_capture_quotas_preserve_old_artifacts_and_skip_later_ones(self):
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(directory, capture_max_count=2,
                                                       capture_total_bytes=8192), command="capture")
            first = recorder.capture("one", {"x": np.arange(4, dtype=np.float32)})
            second = recorder.capture("two", {"x": np.arange(4, dtype=np.float32)})
            third = recorder.capture("three", {"x": np.arange(4, dtype=np.float32)})
            recorder.close()
            self.assertTrue(first.captured)
            self.assertTrue(second.captured)
            self.assertFalse(third.captured)
            self.assertTrue((Path(first.path) / "manifest.json").exists())

    def test_capture_mode_without_raw_arrays_writes_recent_metadata_bundle(self):
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(
                directory, capture_arrays=False, capture_max_bytes_each=4096,
                capture_total_bytes=8192), command="metadata-capture")
            recorder.event("before.failure", component="test", fields={"step": 1})
            result = recorder.capture("failure", {}, metadata={"reason": "fixture"})
            self.assertTrue(result.captured)
            recorder.close()
            manifest = json.loads((Path(result.path) / "manifest.json").read_text())
            self.assertEqual(manifest["arrays"], {})
            self.assertIsNone(manifest["archive"])
            self.assertTrue(any(item["event"] == "before.failure"
                                for item in manifest["recent_events"]))

    def test_concurrent_capture_admission_never_exceeds_count_quota(self):
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(
                directory, capture_max_count=1, capture_total_bytes=4096), command="capture-race")
            barrier = threading.Barrier(8)
            results = []

            def submit(index):
                barrier.wait(timeout=2)
                results.append(recorder.capture(
                    f"capture-{index}", {"x": np.arange(4, dtype=np.float32)}))

            original_savez = np.savez_compressed

            def slow_savez(*args, **kwargs):
                time.sleep(0.01)
                return original_savez(*args, **kwargs)

            with patch("icgs.observability.capture.np.savez_compressed",
                       side_effect=slow_savez):
                threads = [threading.Thread(target=submit, args=(index,)) for index in range(8)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=3)
            recorder.close()
            self.assertEqual(sum(result.captured for result in results), 1)

    def test_run_and_capture_artifacts_are_private_under_permissive_umask(self):
        from icgs.observability.recorder import RunRecorder

        previous_umask = os.umask(0)
        try:
            with tempfile.TemporaryDirectory() as directory:
                recorder = RunRecorder.start(self._config(
                    directory, capture_max_bytes_each=4096, capture_total_bytes=8192),
                    command="permissions")
                result = recorder.capture("fixture", {"x": np.arange(4, dtype=np.float32)})
                recorder.close()
                run_dir = Path(recorder.path)
                self.assertEqual(stat.S_IMODE(run_dir.stat().st_mode), 0o700)
                for name in ("events", "metrics", "captures"):
                    self.assertEqual(stat.S_IMODE((run_dir / name).stat().st_mode), 0o700)
                for path in (run_dir / "run.json", run_dir / "resolved_config.json",
                             run_dir / "summary.json"):
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                self.assertTrue(result.captured)
                capture_dir = Path(result.path)
                self.assertEqual(stat.S_IMODE(capture_dir.stat().st_mode), 0o700)
                for path in (capture_dir / "arrays.npz", capture_dir / "manifest.json"):
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        finally:
            os.umask(previous_umask)

    def test_capture_quota_includes_archive_and_manifest_overhead_for_empty_arrays(self):
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(
                directory, capture_max_bytes_each=128, capture_total_bytes=128),
                command="capture-overhead")
            result = recorder.capture("empty", {"x": np.empty(0, dtype=np.uint8)})
            self.assertFalse(result.captured)
            self.assertEqual(result.reason, "capture_size_quota")
            self.assertEqual(recorder._capture_manager.count, 0)
            self.assertEqual(recorder._capture_manager.total_bytes, 0)
            self.assertEqual(list((Path(recorder.path) / "captures").iterdir()), [])
            recorder.close()

    def test_capture_entry_count_and_key_length_are_rejected_before_copying(self):
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(
                directory, capture_max_bytes_each=4096, capture_total_bytes=8192),
                command="capture-entry-bounds")
            many = {f"array-{index}": np.empty(0, dtype=np.uint8) for index in range(1000)}
            with patch("icgs.observability.capture.np.array",
                       side_effect=AssertionError("entry quota must precede copying")):
                result = recorder.capture("many", many)
            self.assertFalse(result.captured)
            self.assertEqual(result.reason, "capture_entry_quota")

            long_key = {"x" * 4096: np.empty(0, dtype=np.uint8)}
            with patch("icgs.observability.capture.np.array",
                       side_effect=AssertionError("key quota must precede copying")):
                result = recorder.capture("long-key", long_key)
            self.assertFalse(result.captured)
            self.assertEqual(result.reason, "invalid_array_name")
            recorder.close()

    def test_capture_result_and_summary_count_complete_stored_bundle_bytes(self):
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(self._config(
                directory, capture_max_bytes_each=4096, capture_total_bytes=8192),
                command="capture-byte-accounting")
            result = recorder.capture("small", {"x": np.arange(8, dtype=np.float32)})
            self.assertTrue(result.captured)
            capture_dir = Path(result.path)
            stored_bytes = sum(path.stat().st_size for path in capture_dir.iterdir())
            self.assertEqual(result.bytes_written, stored_bytes)
            summary = recorder.close()
            self.assertEqual(summary.capture_count, 1)
            payload = json.loads((Path(recorder.path) / "summary.json").read_text())
            self.assertEqual(payload["captures"]["bytes"], stored_bytes)


if __name__ == "__main__":
    unittest.main()
