import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class InspectionTests(unittest.TestCase):
    def test_inspect_and_trace_are_offline_json_commands(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            config = from_mapping({"output_dir": directory, "flush_interval_s": 0.01,
                                   "shutdown_timeout_s": 1.0})
            recorder = RunRecorder.start(config, command="inspection")
            with recorder.bind(episode_id="episode-1", candidate_id="candidate-0").span(
                    "policy.propose", component="proposal"):
                recorder.event("candidate.created", component="proposal")
            recorder.close()
            run_dir = str(recorder.path)
            inspect = subprocess.run([sys.executable, "-m", "icgs", "logs", "inspect",
                                      run_dir, "--json"], cwd="/tmp", capture_output=True,
                                     text=True, timeout=20)
            self.assertEqual(inspect.returncode, 0, inspect.stderr)
            inspection = json.loads(inspect.stdout)
            self.assertEqual(inspection["status"], "succeeded")
            self.assertTrue(inspection["logs_complete"])
            trace = subprocess.run([sys.executable, "-m", "icgs", "logs", "trace",
                                    run_dir, "--episode", "episode-1", "--json"], cwd="/tmp",
                                   capture_output=True, text=True, timeout=20)
            self.assertEqual(trace.returncode, 0, trace.stderr)
            self.assertTrue(json.loads(trace.stdout)["events"])

    def test_partial_final_line_is_warning_but_interior_corruption_fails(self):
        from icgs.observability.inspect import read_records

        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events"
            events.mkdir()
            path = events / "p1-r0-000001.jsonl"
            path.write_text('{"ok": 1}\n{"partial":')
            records, warnings = read_records(directory)
            self.assertEqual(records, [{"ok": 1}])
            self.assertTrue(warnings)
            path.write_text('{"ok": 1}\nnot-json\n')
            with self.assertRaises(ValueError):
                read_records(directory)

    def test_capture_path_traversal_is_rejected(self):
        from icgs.observability.inspect import load_capture_manifest

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "captures").mkdir()
            with self.assertRaises(ValueError):
                load_capture_manifest(directory, "../outside")

    def test_inspection_bounds_manifest_reads_before_allocation(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.inspect import inspect_run
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory, "flush_interval_s": 0.01,
                "shutdown_timeout_s": 1.0,
            }), command="bounded-inspect")
            recorder.close()
            with self.assertRaisesRegex(ValueError, "read limit"):
                inspect_run(recorder.path, max_bytes=1)

    def test_inspection_rejects_an_oversized_jsonl_line_before_json_decode(self):
        from icgs.observability.inspect import read_records

        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events"
            events.mkdir()
            path = events / "p1-r0-000001.jsonl"
            path.write_bytes(b'{"payload":"' + b"x" * (1024 * 1024 + 1) + b'"}\n')
            with self.assertRaisesRegex(ValueError, "line|read limit"):
                read_records(directory, max_bytes=2 * 1024 * 1024)

    def test_capture_symlink_references_are_rejected_even_inside_run_root(self):
        from icgs.observability.inspect import load_capture_manifest

        with tempfile.TemporaryDirectory() as directory:
            captures = Path(directory) / "captures"
            real = captures / "real"
            captures.mkdir()
            real.mkdir()
            (real / "manifest.json").write_text(json.dumps({"archive": None}))
            (captures / "link").symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                load_capture_manifest(directory, "link")

            (real / "arrays.npz").write_bytes(b"fixture")
            (real / "alias.npz").symlink_to(real / "arrays.npz")
            (real / "manifest.json").write_text(json.dumps({"archive": "alias.npz"}))
            with self.assertRaisesRegex(ValueError, "symlink"):
                load_capture_manifest(directory, "real")

    def test_inspection_reports_corrupt_capture_evidence_without_crashing(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.inspect import inspect_run
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory,
                "mode": "capture",
                "capture_arrays": False,
                "flush_interval_s": 0.01,
                "shutdown_timeout_s": 1.0,
            }), command="corrupt-capture")
            result = recorder.capture("failure", {}, metadata={"fixture": True})
            recorder.close()
            self.assertTrue(result.captured)
            manifest = Path(result.path) / "manifest.json"
            manifest.write_text("{corrupt", encoding="utf-8")

            inspected = inspect_run(recorder.path)
            self.assertFalse(inspected["logs_complete"])
            self.assertIn("captures:summary-count-mismatch", inspected["missing_evidence"])
            self.assertTrue(any("invalid capture evidence" in warning
                                for warning in inspected["warnings"]))

    def test_inspection_reports_unexpected_capture_evidence_against_zero_summary(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.inspect import inspect_run
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory,
                "flush_interval_s": 0.01,
                "shutdown_timeout_s": 1.0,
            }), command="unexpected-capture")
            recorder.close()
            capture_dir = Path(recorder.path) / "captures" / "unexpected"
            capture_dir.mkdir(parents=True)
            (capture_dir / "manifest.json").write_text(
                json.dumps({"archive": None}), encoding="utf-8")

            inspected = inspect_run(recorder.path)
            self.assertFalse(inspected["logs_complete"])
            self.assertIn("captures:summary-count-mismatch", inspected["missing_evidence"])

    def test_inspection_marks_summary_incomplete_when_partial_evidence_is_observed(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.inspect import inspect_run
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory, "flush_interval_s": 0.01,
                "shutdown_timeout_s": 1.0,
            }), command="partial-inspect")
            recorder.event("complete", component="test")
            recorder.close()
            event_path = next((Path(recorder.path) / "events").glob("*.jsonl"))
            with event_path.open("ab") as stream:
                stream.write(b'{"partial":')
            result = inspect_run(recorder.path)
            self.assertFalse(result["logs_complete"])
            self.assertTrue(result["warnings"])

    def test_inspection_rejects_corrupted_archive_bytes(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.inspect import inspect_run
        from icgs.observability.recorder import RunRecorder
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory, "mode": "capture", "capture_arrays": True,
                "capture_max_bytes_each": 4096, "capture_total_bytes": 8192,
                "flush_interval_s": 0.01, "shutdown_timeout_s": 1.0,
            }), command="corrupt-archive")
            result = recorder.capture("archive", {"x": np.arange(8, dtype=np.float32)})
            recorder.close()
            self.assertTrue(result.captured)
            (Path(result.path) / "arrays.npz").write_bytes(b"not-an-npz")

            inspected = inspect_run(recorder.path)
            self.assertFalse(inspected["logs_complete"])
            self.assertIn("captures:invalid-entry", inspected["missing_evidence"])
            self.assertTrue(any("invalid capture evidence" in warning
                                for warning in inspected["warnings"]))

    def test_inspection_marks_unexpected_malformed_zero_capture_evidence_incomplete(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.inspect import inspect_run
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory, "flush_interval_s": 0.01,
                "shutdown_timeout_s": 1.0,
            }), command="malformed-zero-capture")
            recorder.close()
            unexpected = Path(recorder.path) / "captures" / "unexpected"
            unexpected.mkdir(parents=True)
            (unexpected / "manifest.json").write_text("{bad", encoding="utf-8")

            inspected = inspect_run(recorder.path)
            self.assertFalse(inspected["logs_complete"])
            self.assertIn("captures:invalid-entry", inspected["missing_evidence"])
            self.assertTrue(inspected["warnings"])

    def test_record_directory_enumeration_is_bounded_including_empty_files(self):
        from icgs.observability.inspect import read_records

        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events"
            events.mkdir()
            for index in range(5):
                (events / f"empty-{index}.jsonl").touch()
            calls = []
            original_iterdir = Path.iterdir

            def bounded_iterdir(path):
                iterator = original_iterdir(path)
                if path != events:
                    return iterator

                def observed():
                    for item in iterator:
                        calls.append(item)
                        yield item
                return observed()

            with patch.object(Path, "iterdir", bounded_iterdir):
                with self.assertRaisesRegex(ValueError, "directory|entry|record"):
                    read_records(directory, max_records=1)
            self.assertLessEqual(len(calls), 2)

    def test_capture_directory_enumeration_is_bounded_before_sorting(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.inspect import inspect_run
        from icgs.observability.recorder import RunRecorder

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory, "flush_interval_s": 0.01,
                "shutdown_timeout_s": 1.0,
            }), command="capture-entry-limit")
            recorder.close()
            captures = Path(recorder.path) / "captures"
            captures.mkdir()
            for index in range(5):
                (captures / f"extra-{index}").mkdir()
            calls = []
            original_iterdir = Path.iterdir

            def bounded_iterdir(path):
                iterator = original_iterdir(path)
                if path != captures:
                    return iterator

                def observed():
                    for item in iterator:
                        calls.append(item)
                        yield item
                return observed()

            with patch.object(Path, "iterdir", bounded_iterdir):
                inspected = inspect_run(recorder.path, max_records=1)
            self.assertFalse(inspected["logs_complete"])
            self.assertIn("captures:read-limit", inspected["missing_evidence"])
            self.assertLessEqual(len(calls), 2)

    def test_inspection_uses_one_aggregate_budget_for_multiple_capture_bundles(self):
        from icgs.observability.config import from_mapping
        from icgs.observability.inspect import inspect_run
        from icgs.observability.recorder import RunRecorder
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(from_mapping({
                "output_dir": directory, "mode": "capture", "capture_arrays": True,
                "capture_max_count": 3, "capture_max_bytes_each": 4096,
                "capture_total_bytes": 8192, "flush_interval_s": 0.01,
                "shutdown_timeout_s": 1.0,
            }), command="aggregate-inspection-budget")
            first = recorder.capture("one", {"x": np.arange(16, dtype=np.float32)})
            second = recorder.capture("two", {"x": np.arange(16, dtype=np.float32)})
            recorder.close()
            self.assertTrue(first.captured)
            self.assertTrue(second.captured)
            root = Path(recorder.path)
            base_bytes = sum(path.stat().st_size for path in root.glob("run.json"))
            base_bytes += sum(path.stat().st_size for path in root.glob("summary.json"))
            base_bytes += sum(path.stat().st_size for path in (root / "events").iterdir())
            base_bytes += sum(path.stat().st_size for path in (root / "metrics").iterdir())
            bundles = []
            for result in (first, second):
                capture_dir = Path(result.path)
                bundles.append(sum(path.stat().st_size for path in capture_dir.iterdir()))
            with self.assertRaisesRegex(ValueError, "read limit"):
                inspect_run(root, max_bytes=base_bytes + min(bundles))


if __name__ == "__main__":
    unittest.main()
