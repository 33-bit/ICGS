import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from icgs.configuration.method import MethodConfig
from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
from icgs.contracts.records import Observation
from icgs.data.schemas.episodes import validate_episode


class EpisodeArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_episode(self, num_transitions: int = 2, *, episode_id: str = "ep-test-01"):
        # Generate boundary observations first so before/after at adjacent transitions match exactly
        sim_time_start = 1.0
        wall_time_start = 2.0
        sensor_id = "sensor-v1"

        online_obs = []
        timed_obs = []
        for i in range(num_transitions + 1):
            pts_count = 10 + (i % 5)  # ragged variation across boundaries
            # Measured float64
            pts_measured = np.linspace(0.1, 0.9, pts_count * 3, dtype=np.float64).reshape(pts_count, 3) + i * 0.001
            pose = np.eye(4, dtype=np.float64)
            pose[0, 3] = 0.01 * i
            meas_obs = Observation(pts_measured, pose, 0.0)
            t_obs = TimedObservation(
                meas_obs,
                i,
                sim_time_start + 0.1 * i,
                wall_time_start + 0.1 * i,
                sensor_id,
            )
            timed_obs.append(t_obs)

            # Online float32 representation (differs within atol 1e-6 from float64)
            pts_online = pts_measured.astype(np.float32)
            valid = np.ones(pts_count, dtype=bool)
            online_obs.append({
                "points": pts_online,
                "point_valid": valid,
                "T_w_e": pose.copy(),
                "grip": 0,
            })

        transitions = []
        for i in range(num_transitions):
            before_obs = timed_obs[i]
            after_obs = timed_obs[i + 1]
            cmd = TimedCommand(after_obs.observation.T_w_e, 0, 0.1)
            transitions.append(
                ExecutedTransition(before_obs, after_obs, cmd, 0.1, 20, "ok")
            )

        record = {
            "schema_version": "icgs_episode_v1",
            "provenance": {
                "episode_id": episode_id,
                "program_id": "T01",
                "source_lineage_id": "seed-001",
                "asset_family_id": "family-alpha",
                "split": "train",
                "calibration_id": "calib-v1",
                "observation_origin": "measured",
                "raw_commands_id": "raw-01",
                "materialized_commands_id": "mat-01",
            },
            "online_observations": online_obs,
            "transitions": transitions,
        }
        validate_episode(record)
        return record

    def test_lossless_roundtrip_two_transitions_ragged_float32_vs_float64(self):
        from icgs.data.archives import read_episode_archive, write_episode_archive

        record = self._make_episode(num_transitions=2, episode_id="ep-lossless-01")
        manifest_path = write_episode_archive(self.root, record)
        self.assertTrue(manifest_path.is_file())
        self.assertEqual(manifest_path.name, "manifest.json")

        restored = read_episode_archive(manifest_path)
        validate_episode(restored)

        self.assertEqual(restored["schema_version"], record["schema_version"])
        self.assertEqual(restored["provenance"], record["provenance"])
        self.assertEqual(len(restored["online_observations"]), len(record["online_observations"]))
        self.assertEqual(len(restored["transitions"]), len(record["transitions"]))

        # Check online dtypes and bit-exact values preserved
        for orig_o, rest_o in zip(record["online_observations"], restored["online_observations"]):
            self.assertEqual(rest_o["points"].dtype, np.float32)
            self.assertEqual(orig_o["points"].dtype, np.float32)
            np.testing.assert_array_equal(rest_o["points"], orig_o["points"])
            self.assertEqual(rest_o["point_valid"].dtype, bool)
            np.testing.assert_array_equal(rest_o["point_valid"], orig_o["point_valid"])
            self.assertEqual(rest_o["T_w_e"].dtype, np.float64)
            np.testing.assert_array_equal(rest_o["T_w_e"], orig_o["T_w_e"])
            self.assertEqual(rest_o["grip"], orig_o["grip"])

        # Check measured dtypes and bit-exact values preserved
        for orig_t, rest_t in zip(record["transitions"], restored["transitions"]):
            self.assertEqual(rest_t.before.observation.points.dtype, np.float64)
            np.testing.assert_array_equal(rest_t.before.observation.points, orig_t.before.observation.points)
            self.assertEqual(rest_t.after.observation.points.dtype, np.float64)
            np.testing.assert_array_equal(rest_t.after.observation.points, orig_t.after.observation.points)
            self.assertEqual(rest_t.before.boundary, orig_t.before.boundary)
            self.assertEqual(rest_t.after.boundary, orig_t.after.boundary)
            self.assertEqual(rest_t.command.grip, orig_t.command.grip)
            self.assertEqual(rest_t.achieved_duration_s, orig_t.achieved_duration_s)
            self.assertEqual(rest_t.physics_substeps, orig_t.physics_substeps)
            self.assertEqual(rest_t.controller_status, orig_t.controller_status)

    def test_measured_and_online_clouds_and_poses_are_independent(self):
        record = self._make_episode(num_transitions=1, episode_id="ep-independent-streams")
        for obs in record["online_observations"]:
            obs["points"] = obs["points"] + np.float32(0.25)
            obs["T_w_e"] = obs["T_w_e"].copy()
            obs["T_w_e"][0, 3] += 0.25
        validate_episode(record)

        from icgs.data.archives import read_episode_archive, write_episode_archive
        restored = read_episode_archive(write_episode_archive(self.root, record))
        np.testing.assert_array_equal(restored["online_observations"][0]["points"], record["online_observations"][0]["points"])
        np.testing.assert_array_equal(restored["transitions"][0].before.observation.points, record["transitions"][0].before.observation.points)

    def test_257_transitions_shard_boundary_and_shared_observation(self):
        from icgs.data.archives import read_episode_archive, write_episode_archive

        config = MethodConfig(dataset={"shard_intervals": 256})
        record = self._make_episode(num_transitions=257, episode_id="ep-257-shards")
        manifest_path = write_episode_archive(self.root, record, config=config)

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = manifest_data["shards"]
        self.assertEqual(len(shards), 2)
        self.assertEqual(shards[0]["interval_count"], 256)
        self.assertEqual(shards[0]["start_boundary"], 0)
        self.assertEqual(shards[0]["end_boundary"], 256)
        self.assertEqual(shards[1]["interval_count"], 1)
        self.assertEqual(shards[1]["start_boundary"], 256)
        self.assertEqual(shards[1]["end_boundary"], 257)

        # Reconstructed record must agree on shared boundary observation
        restored = read_episode_archive(manifest_path)
        self.assertEqual(len(restored["transitions"]), 257)
        self.assertEqual(restored["transitions"][255].after.boundary, 256)
        self.assertEqual(restored["transitions"][256].before.boundary, 256)
        np.testing.assert_array_equal(
            restored["transitions"][255].after.observation.points,
            restored["transitions"][256].before.observation.points,
        )

    def test_duplicate_publication_refused(self):
        from icgs.data.archives import write_episode_archive

        record = self._make_episode(num_transitions=1, episode_id="ep-duplicate")
        write_episode_archive(self.root, record)
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            write_episode_archive(self.root, record)

    def test_unsafe_episode_id_path_traversal_refused(self):
        from icgs.data.archives import write_episode_archive

        for unsafe_id in ("../escape", "sub/dir", ".", "..", "", "   "):
            with self.subTest(unsafe_id=unsafe_id):
                record = self._make_episode(num_transitions=1, episode_id="safe")
                record["provenance"]["episode_id"] = unsafe_id
                with self.assertRaisesRegex(ValueError, "safe single path component|unsafe episode_id|nonempty string"):
                    write_episode_archive(self.root, record)

    def test_checksum_corruption_detected(self):
        from icgs.data.archives import read_episode_archive, write_episode_archive

        record = self._make_episode(num_transitions=1, episode_id="ep-corrupt-hash")
        manifest_path = write_episode_archive(self.root, record)

        # Corrupt one shard byte
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_path = manifest_path.parent / manifest_data["shards"][0]["filename"]
        data = bytearray(shard_path.read_bytes())
        data[20] = (data[20] + 1) % 256
        shard_path.write_bytes(bytes(data))

        with self.assertRaisesRegex(ValueError, "checksum mismatch|sha256"):
            read_episode_archive(manifest_path)

    def test_quarantine_records_failures(self):
        from icgs.data.archives import quarantine_attempt

        q_path = quarantine_attempt(
            self.root,
            "attempt-fail-01",
            diagnostics={"error": "transition rejected", "code": 42},
            evidence={"bad_step": 3},
        )
        self.assertTrue(q_path.is_file())
        self.assertIn("quarantine", str(q_path))
        data = json.loads(q_path.read_text(encoding="utf-8"))
        self.assertEqual(data["diagnostics"]["error"], "transition rejected")

    def test_symlink_shard_refused(self):
        from icgs.data.archives import read_episode_archive, write_episode_archive

        record = self._make_episode(num_transitions=1, episode_id="ep-symlink")
        manifest_path = write_episode_archive(self.root, record)

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_filename = manifest_data["shards"][0]["filename"]
        shard_file = manifest_path.parent / shard_filename
        target_file = self.root / "external.npz"
        shard_file.replace(target_file)
        os.symlink(target_file, shard_file)

        with self.assertRaisesRegex(ValueError, "symlink"):
            read_episode_archive(manifest_path)

    def test_nan_metadata_rejected(self):
        from icgs.data.archives import write_episode_archive

        record = self._make_episode(num_transitions=1, episode_id="ep-nan-meta")
        with self.assertRaisesRegex(ValueError, "Non-finite float"):
            write_episode_archive(self.root, record, metadata={"bad_value": float("nan")})

    def test_corrupt_offsets_rejected(self):
        from icgs.data.archives import read_episode_archive, write_episode_archive

        record = self._make_episode(num_transitions=1, episode_id="ep-bad-offsets")
        manifest_path = write_episode_archive(self.root, record)

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_file = manifest_path.parent / manifest_data["shards"][0]["filename"]

        with np.load(shard_file) as npz:
            arrays = dict(npz)
        arrays["online_point_offsets"] = np.array([0, 100, 5], dtype=np.int64)  # non-monotonic

        np.savez_compressed(shard_file, **arrays)
        # Update manifest sha256 to isolate offset check
        from icgs.data.archives import _compute_sha256
        manifest_data["shards"][0]["sha256"] = _compute_sha256(shard_file)
        manifest_path.write_text(json.dumps(manifest_data, indent=2))

        with self.assertRaisesRegex(ValueError, "monotonic|Invalid online_point_offsets"):
            read_episode_archive(manifest_path)

    def test_empty_attempt_report(self):
        from icgs.data.archives import read_attempt_report, write_attempt_report

        report_path = write_attempt_report(
            self.root,
            "attempt-empty-01",
            {"status": "empty", "reason": "environment reset timeout", "step_count": 0},
        )
        self.assertTrue(report_path.is_file())
        restored = read_attempt_report(report_path)
        self.assertEqual(restored["status"], "empty")
        self.assertEqual(restored["step_count"], 0)

    def test_persist_attempt_adapter_successful_and_empty(self):
        from icgs.contracts.method import TimedCommand
        from icgs.data.collection.attempts import AttemptResult, persist_attempt
        from icgs.data.archives import read_attempt_report, read_episode_archive

        ep = self._make_episode(num_transitions=2, episode_id="ep-adapter-01")
        prov = ep["provenance"]

        # Attempt with transitions
        result_with_trans = AttemptResult(
            initial_observation=ep["transitions"][0].before,
            transitions=tuple(ep["transitions"]),
            annotations=({"step": 0}, {"step": 1}),
            status="completed",
            online_observations=tuple(ep["online_observations"]),
        )
        path = persist_attempt(self.root, result_with_trans, provenance=prov)
        self.assertTrue(path.is_file())
        restored = read_episode_archive(path)
        self.assertEqual(len(restored["transitions"]), 2)

        # Empty attempt
        prov_empty = dict(prov, episode_id="ep-adapter-empty")
        result_empty = AttemptResult(
            initial_observation=ep["transitions"][0].before,
            transitions=(),
            annotations=(),
            status="failed",
        )
        empty_path = persist_attempt(self.root, result_empty, provenance=prov_empty)
        self.assertTrue(empty_path.is_file())
        report = read_attempt_report(empty_path)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["transition_count"], 0)

    def test_persist_attempt_retains_float64_and_protects_recorder_metadata(self):
        from icgs.data.collection.attempts import AttemptResult, persist_attempt
        from icgs.data.archives import read_episode_archive

        ep = self._make_episode(num_transitions=1, episode_id="ep-meta-guard")
        prov = ep["provenance"]

        # Ensure observation points are float64
        obs = ep["transitions"][0].before.observation
        self.assertEqual(obs.points.dtype, np.float64)

        # Online observation also with float64 to verify preservation without downcasting
        online_obs = (
            {
                "points": obs.points.copy(),
                "point_valid": np.ones(len(obs.points), dtype=bool),
                "T_w_e": ep["transitions"][0].before.observation.T_w_e.copy(),
                "grip": ep["transitions"][0].before.observation.grip,
            },
            {
                "points": ep["transitions"][0].after.observation.points.copy(),
                "point_valid": np.ones(len(ep["transitions"][0].after.observation.points), dtype=bool),
                "T_w_e": ep["transitions"][0].after.observation.T_w_e.copy(),
                "grip": ep["transitions"][0].after.observation.grip,
            },
        )

        result = AttemptResult(
            initial_observation=ep["transitions"][0].before,
            transitions=tuple(ep["transitions"]),
            annotations=({"step": 0},),
            status="completed",
            online_observations=online_obs,
        )

        # Attempt to override recorder-owned metadata key: must be rejected!
        with self.assertRaisesRegex(ValueError, "caller metadata may not override recorder-owned key"):
            persist_attempt(self.root, result, provenance=prov, metadata={"attempt_status": "injected"})

        with self.assertRaisesRegex(ValueError, "caller metadata may not override recorder-owned key"):
            persist_attempt(self.root, result, provenance=prov, metadata={"annotations": []})

        # When persisted without collision, verify float64 points are retained without float32 quantization
        path = persist_attempt(self.root, result, provenance=prov, metadata={"custom_note": "ok"})
        restored = read_episode_archive(path)
        self.assertEqual(restored["online_observations"][0]["points"].dtype, np.float64)
        np.testing.assert_array_equal(restored["online_observations"][0]["points"], obs.points)

    def test_exact_npz_key_inventory_rejects_extra_keys(self):
        from icgs.data.archives import _compute_sha256, read_episode_archive, write_episode_archive

        record = self._make_episode(num_transitions=1, episode_id="ep-extra-keys")
        manifest_path = write_episode_archive(self.root, record)

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_file = manifest_path.parent / manifest_data["shards"][0]["filename"]

        with np.load(shard_file) as npz:
            arrays = dict(npz)
        arrays["extra_unauthorized_key"] = np.array([1, 2, 3])

        np.savez_compressed(shard_file, **arrays)
        manifest_data["shards"][0]["sha256"] = _compute_sha256(shard_file)
        manifest_path.write_text(json.dumps(manifest_data, indent=2))

        with self.assertRaisesRegex(ValueError, "unexpected_extras"):
            read_episode_archive(manifest_path)

    def test_per_array_shape_dtype_finiteness_validation(self):
        from icgs.data.archives import _compute_sha256, read_episode_archive, write_episode_archive

        record = self._make_episode(num_transitions=1, episode_id="ep-bad-array")
        manifest_path = write_episode_archive(self.root, record)

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_file = manifest_path.parent / manifest_data["shards"][0]["filename"]

        with np.load(shard_file) as npz:
            arrays = dict(npz)
        # Inject non-finite coordinate into meas_points
        arrays["meas_points"][0, 0] = np.nan

        np.savez_compressed(shard_file, **arrays)
        manifest_data["shards"][0]["sha256"] = _compute_sha256(shard_file)
        manifest_path.write_text(json.dumps(manifest_data, indent=2))

        with self.assertRaisesRegex(ValueError, "non-finite coordinates"):
            read_episode_archive(manifest_path)

    def test_shared_measured_boundary_disagreement_across_shards_rejected(self):
        from icgs.data.archives import _compute_sha256, read_episode_archive, write_episode_archive

        cfg = MethodConfig(dataset={"shard_intervals": 1})
        record = self._make_episode(num_transitions=2, episode_id="ep-shared-meas-mismatch")
        manifest_path = write_episode_archive(self.root, record, config=cfg)

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(manifest_data["shards"]), 2)
        shard1_file = manifest_path.parent / manifest_data["shards"][1]["filename"]

        with np.load(shard1_file) as npz:
            arrays = dict(npz)
        # Tamper with measured pose at the shared boundary (index 0 of shard 1)
        arrays["meas_T_w_e"][0, 0, 3] += 999.0

        np.savez_compressed(shard1_file, **arrays)
        manifest_data["shards"][1]["sha256"] = _compute_sha256(shard1_file)
        manifest_path.write_text(json.dumps(manifest_data, indent=2))

        with self.assertRaisesRegex(ValueError, "Shared boundary pose T_w_e mismatch"):
            read_episode_archive(manifest_path)

    def test_online_observations_and_masks_lossless_roundtrip(self):
        from icgs.data.collection.attempts import AttemptResult, persist_attempt
        from icgs.data.archives import read_episode_archive

        prov = {
            "episode_id": "ep-lossless-online",
            "program_id": "T01",
            "source_lineage_id": "root-01",
            "asset_family_id": "family-01",
            "split": "train",
            "calibration_id": "calib-01",
            "observation_origin": "measured",
            "raw_commands_id": "raw-01",
            "materialized_commands_id": "mat-01",
        }

        # Online observation with specific float32 dtypes and custom point_valid mask containing False
        online_pts = np.array([[0.1, 0.2, 0.3], [99.0, 99.0, 99.0]], dtype=np.float32)
        online_mask = np.array([True, False], dtype=bool)
        online_obs = (
            {
                "points": online_pts,
                "point_valid": online_mask,
                "T_w_e": np.eye(4, dtype=np.float32),
                "grip": 0,
            },
            {
                "points": online_pts,
                "point_valid": online_mask,
                "T_w_e": np.eye(4, dtype=np.float32),
                "grip": 1,
            },
        )

        # Transition observation points must equal online_pts[online_mask]
        meas_pts = online_pts[online_mask].astype(np.float64)
        meas_obs0 = Observation(meas_pts, np.eye(4, dtype=np.float64), 0.0)
        meas_obs1 = Observation(meas_pts, np.eye(4, dtype=np.float64), 1.0)
        t_obs0 = TimedObservation(meas_obs0, 0, 1.0, 2.0, "sensor-01")
        t_obs1 = TimedObservation(meas_obs1, 1, 1.1, 2.1, "sensor-01")
        trans = ExecutedTransition(
            t_obs0, t_obs1, TimedCommand(np.eye(4, dtype=np.float64), 1, 0.1), 0.1, 10, "ok"
        )

        res = AttemptResult(
            initial_observation=t_obs0,
            transitions=(trans,),
            annotations=(),
            status="completed",
            online_observations=online_obs,
        )

        path = persist_attempt(self.root, res, provenance=prov)
        restored = read_episode_archive(path)

        # Ensure online_observations roundtrip losslessly without forced casts
        restored_online = restored["online_observations"]
        self.assertEqual(restored_online[0]["points"].dtype, np.float32)
        np.testing.assert_array_equal(restored_online[0]["points"], online_pts)
        self.assertEqual(restored_online[0]["point_valid"].dtype, bool)
        np.testing.assert_array_equal(restored_online[0]["point_valid"], online_mask)
        self.assertEqual(restored_online[0]["grip"], 0)

    def test_staging_byte_cap_enforced(self):
        from icgs.data.archives import write_episode_archive

        record = self._make_episode(num_transitions=2, episode_id="ep-staging-cap")
        # Enforce an unrealistically small staging cap (e.g. 100 bytes)
        with self.assertRaisesRegex(ValueError, "staging_byte_cap"):
            write_episode_archive(self.root, record, staging_byte_cap=100)

    def test_unsupported_json_object_rejected_no_silent_str(self):
        from icgs.data.archives import _json_safe

        class CustomArbitraryObject:
            pass

        with self.assertRaises(TypeError):
            _json_safe(CustomArbitraryObject())

    def test_read_archive_metadata(self):
        from icgs.data.archives import read_archive_metadata, write_episode_archive

        record = self._make_episode(num_transitions=1, episode_id="ep-read-meta")
        manifest_path = write_episode_archive(
            self.root,
            record,
            metadata={"collector": "test_run", "status_override_blocked": True},
        )
        meta = read_archive_metadata(manifest_path)
        self.assertEqual(meta["collector"], "test_run")
        self.assertTrue(meta["status_override_blocked"])

    def test_med1_exact_npz_dtype_validation(self):
        from icgs.data.archives import _compute_sha256, read_episode_archive, write_episode_archive

        record = self._make_episode(num_transitions=1, episode_id="ep-med1-dtypes")
        manifest_path = write_episode_archive(self.root, record)

        manifest_data = json.loads(manifest_path.read_text())
        shard0_file = manifest_path.parent / manifest_data["shards"][0]["filename"]

        with np.load(shard0_file) as npz:
            orig_arrays = dict(npz)

        # 1. Tamper online_point_offsets to int32 (shape compatible, wrong dtype)
        bad1 = dict(orig_arrays)
        bad1["online_point_offsets"] = bad1["online_point_offsets"].astype(np.int32)
        mdata1 = json.loads(manifest_path.read_text())
        mdata1["shards"][0]["array_schema"]["online_point_offsets"]["dtype"] = "int32"
        np.savez_compressed(shard0_file, **bad1)
        mdata1["shards"][0]["sha256"] = _compute_sha256(shard0_file)
        manifest_path.write_text(json.dumps(mdata1, indent=2))

        with self.assertRaisesRegex(TypeError, "online_point_offsets in .* must be int64"):
            read_episode_archive(manifest_path)

        # 2. Tamper meas_points to float32
        bad2 = dict(orig_arrays)
        bad2["meas_points"] = bad2["meas_points"].astype(np.float32)
        mdata2 = json.loads(manifest_path.read_text())
        mdata2["shards"][0]["array_schema"]["online_point_offsets"]["dtype"] = str(orig_arrays["online_point_offsets"].dtype)
        mdata2["shards"][0]["array_schema"]["meas_points"]["dtype"] = "float32"
        np.savez_compressed(shard0_file, **bad2)
        mdata2["shards"][0]["sha256"] = _compute_sha256(shard0_file)
        manifest_path.write_text(json.dumps(mdata2, indent=2))

        with self.assertRaisesRegex(TypeError, "meas_points in .* must have float64"):
            read_episode_archive(manifest_path)

        # 3. Tamper cmd_grip to int64
        bad3 = dict(orig_arrays)
        bad3["cmd_grip"] = bad3["cmd_grip"].astype(np.int64)
        mdata3 = json.loads(manifest_path.read_text())
        mdata3["shards"][0]["array_schema"]["meas_points"]["dtype"] = str(orig_arrays["meas_points"].dtype)
        mdata3["shards"][0]["array_schema"]["cmd_grip"]["dtype"] = "int64"
        np.savez_compressed(shard0_file, **bad3)
        mdata3["shards"][0]["sha256"] = _compute_sha256(shard0_file)
        manifest_path.write_text(json.dumps(mdata3, indent=2))

        with self.assertRaisesRegex(TypeError, "cmd_grip in .* must have int32"):
            read_episode_archive(manifest_path)

    def test_med2_array_schema_stored_and_verified(self):
        from icgs.data.archives import _compute_sha256, read_episode_archive, write_episode_archive

        record = self._make_episode(num_transitions=1, episode_id="ep-med2-schema")
        manifest_path = write_episode_archive(self.root, record)

        manifest_data = json.loads(manifest_path.read_text())
        shard0_meta = manifest_data["shards"][0]
        self.assertIn("array_schema", shard0_meta)
        self.assertIn("meas_points", shard0_meta["array_schema"])
        self.assertEqual(shard0_meta["array_schema"]["meas_points"]["dtype"], "float64")

        shard0_file = manifest_path.parent / shard0_meta["filename"]
        with np.load(shard0_file) as npz:
            arrays = dict(npz)

        # Tamper array in NPZ without updating schema -> rejected by MED2
        bad_arrays = dict(arrays)
        bad_arrays["meas_points"] = bad_arrays["meas_points"].astype(np.float32)
        np.savez_compressed(shard0_file, **bad_arrays)
        manifest_data["shards"][0]["sha256"] = _compute_sha256(shard0_file)
        manifest_path.write_text(json.dumps(manifest_data, indent=2))

        with self.assertRaisesRegex(TypeError, "Array meas_points in .* expected float64 from schema"):
            read_episode_archive(manifest_path)

    def test_high1_persist_attempt_refuses_fallback_reconstruction(self):
        from icgs.data.collection.attempts import AttemptResult, persist_attempt

        record = self._make_episode(num_transitions=1, episode_id="ep-high1-refuse")
        trans = record["transitions"][0]
        res = AttemptResult(
            initial_observation=trans.before,
            transitions=(trans,),
            annotations=(),
            status="completed",
            online_observations=None,  # Missing online observations!
        )

        prov = {
            "episode_id": "ep-high1-refuse",
            "program_id": "T01",
            "source_lineage_id": "root-01",
            "asset_family_id": "fam-01",
            "split": "train",
            "calibration_id": "cal-01",
            "observation_origin": "measured",
            "raw_commands_id": "raw-01",
            "materialized_commands_id": "mat-01",
        }

        # Fallback reconstruction is refused
        with self.assertRaisesRegex(ValueError, "persist_attempt refuses fallback reconstruction"):
            persist_attempt(self.root, res, provenance=prov)

        # Providing explicit online_provider succeeds
        def dummy_provider(step):
            obs = step.observation if hasattr(step, "observation") else step.after.observation
            return {
                "points": obs.points.astype(np.float32),
                "point_valid": np.ones(len(obs.points), dtype=bool),
                "T_w_e": obs.T_w_e.astype(np.float32),
                "grip": float(obs.grip),
            }

        out_path = persist_attempt(self.root, res, provenance=prov, online_provider=dummy_provider)
        self.assertTrue(out_path.is_file())

    def test_missing_array_schema_or_incomplete_keys_rejected(self):
        """Issue 9: Missing array_schema or incomplete keys rejected even with valid NPZ/checksum."""
        from icgs.data.archives import _compute_sha256, read_episode_archive, write_episode_archive

        record = self._make_episode(num_transitions=1, episode_id="ep-issue9-schema")
        manifest_path = write_episode_archive(self.root, record)

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

        # 1. Missing array_schema entirely
        mdata_no_schema = json.loads(manifest_path.read_text(encoding="utf-8"))
        mdata_no_schema["shards"][0].pop("array_schema", None)
        manifest_path.write_text(json.dumps(mdata_no_schema, indent=2), encoding="utf-8")

        with self.assertRaises((KeyError, ValueError)):
            read_episode_archive(manifest_path)

        # 2. Incomplete keys in array_schema (e.g. missing meas_points)
        mdata_incomplete = json.loads(manifest_path.read_text(encoding="utf-8"))
        mdata_incomplete["shards"][0]["array_schema"] = {
            k: v for k, v in manifest_data["shards"][0]["array_schema"].items()
            if k != "meas_points"
        }
        manifest_path.write_text(json.dumps(mdata_incomplete, indent=2), encoding="utf-8")

        with self.assertRaises((KeyError, ValueError)):
            read_episode_archive(manifest_path)

    def test_online_buffer_provider_mutation_isolation_and_immediate_detachment(self):
        """Issue 3: Mutable buffer provider values/dtypes/masks detached immediately per boundary."""
        from icgs.contracts.method import TimedCommand
        from icgs.data.collection.attempts import collect_attempt

        # Mock environment that yields 2 transitions
        class StepEnv:
            def __init__(self):
                self.step = 0
            def reset(self, *, seed=None):
                obs = Observation(np.zeros((4, 3), dtype=np.float64), np.eye(4, dtype=np.float64), 0.0)
                return TimedObservation(obs, 0, 0.0, 0.0, "mock")
            def advance(self, cmd):
                self.step += 1
                obs1 = Observation(np.zeros((4, 3), dtype=np.float64), np.eye(4, dtype=np.float64), 0.0)
                obs2 = Observation(np.zeros((4, 3), dtype=np.float64), np.eye(4, dtype=np.float64), 0.0)
                t1 = TimedObservation(obs1, self.step - 1, 0.1 * (self.step - 1), 0.1 * (self.step - 1), "mock")
                t2 = TimedObservation(obs2, self.step, 0.1 * self.step, 0.1 * self.step, "mock")
                return ExecutedTransition(t1, t2, cmd, cmd.duration_s, 10, "ok")
            def close(self):
                pass

        # Provider that returns the SAME mutable mapping and buffer, mutating in place
        buffer_pts = np.zeros((4, 3), dtype=np.float32)
        buffer_mask = np.ones(4, dtype=bool)
        buffer_pose = np.eye(4, dtype=np.float32)
        shared_mapping = {
            "points": buffer_pts,
            "point_valid": buffer_mask,
            "T_w_e": buffer_pose,
            "grip": 0.0,
        }

        def mutating_provider(step):
            # Mutate buffer in place for each call
            step_idx = getattr(step, "step_count", 0) if hasattr(step, "step_count") else 1
            buffer_pts[0, 0] = float(step_idx)
            return shared_mapping

        env = StepEnv()
        commands = [TimedCommand(np.eye(4), 0, 0.1), TimedCommand(np.eye(4), 0, 0.1)]
        config = MethodConfig()
        res = collect_attempt(env, commands, config=config, online_provider=mutating_provider)

        # If detached immediately, online_observations must NOT share the mutated array references
        self.assertIsNotNone(res.online_observations)
        self.assertEqual(len(res.online_observations), 3)

        # Verify mutation after collection does not change collected observations
        buffer_pts[:] = 999.0
        self.assertNotEqual(res.online_observations[0]["points"][0, 0], 999.0)
        self.assertNotEqual(res.online_observations[1]["points"][0, 0], 999.0)

    def test_invalid_online_fields_rejected_at_collection_boundary(self):
        """Issue 3: Invalid online fields rejected before another step; no evidence lost."""
        from icgs.contracts.method import TimedCommand
        from icgs.data.collection.attempts import AttemptExecutionError, collect_attempt

        call_count = 0
        def bad_provider(step):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                # Return invalid unapproved field
                return {
                    "points": np.zeros((4, 3), dtype=np.float32),
                    "point_valid": np.ones(4, dtype=bool),
                    "T_w_e": np.eye(4, dtype=np.float32),
                    "grip": 0.0,
                    "unapproved_privileged_field": "leak",
                }
            return {
                "points": np.zeros((4, 3), dtype=np.float32),
                "point_valid": np.ones(4, dtype=bool),
                "T_w_e": np.eye(4, dtype=np.float32),
                "grip": 0.0,
            }

        class CountingEnv:
            def __init__(self):
                self.steps_advanced = 0
            def reset(self, *, seed=None):
                obs = Observation(np.zeros((4, 3), dtype=np.float64), np.eye(4, dtype=np.float64), 0.0)
                return TimedObservation(obs, 0, 0.0, 0.0, "mock")
            def advance(self, cmd):
                self.steps_advanced += 1
                obs = Observation(np.zeros((4, 3), dtype=np.float64), np.eye(4, dtype=np.float64), 0.0)
                t1 = TimedObservation(obs, self.steps_advanced - 1, 0.0, 0.0, "mock")
                t2 = TimedObservation(obs, self.steps_advanced, 0.1, 0.1, "mock")
                return ExecutedTransition(t1, t2, cmd, cmd.duration_s, 10, "ok")
            def close(self):
                pass

        env = CountingEnv()
        commands = [TimedCommand(np.eye(4), 0, 0.1), TimedCommand(np.eye(4), 0, 0.1)]
        config = MethodConfig()
        with self.assertRaises(AttemptExecutionError) as ctx:
            collect_attempt(env, commands, config=config, online_provider=bad_provider)

        # Must reject before taking another step
        self.assertEqual(env.steps_advanced, 1)
        # Accepted evidence (initial observation + 1 transition) must NOT be lost
        self.assertIn("initial_observation", ctx.exception.evidence)


    def test_high5_write_attempt_report_staging_cap(self):
        from icgs.data.archives import write_attempt_report

        report = {"attempt_id": "ep-report-cap", "status": "completed"}
        with self.assertRaisesRegex(OSError, "exceeds staging cap"):
            write_attempt_report(self.root, "ep-report-cap", report, staging_byte_cap=5)

    def test_sol_fix2_write_episode_archive_prevents_any_file_write_on_projected_cap_exceeded(self):
        """Sol Fix 2: Staging byte cap exceeded must raise BEFORE writing any shard/manifest file to disk."""
        from icgs.data.archives import StorageLimitExceeded, write_episode_archive

        record = self._make_episode(num_transitions=2, episode_id="ep-sol-proj-cap")
        # staging_byte_cap is smaller than the array bytes (which is several KB)
        with self.assertRaises(StorageLimitExceeded):
            write_episode_archive(self.root, record, staging_byte_cap=100)

        # Confirm NO staging directory or episode directory was left or created with files
        ep_dir = self.root / "episodes" / "ep-sol-proj-cap"
        self.assertFalse(ep_dir.exists())
        staging_dirs = list(self.root.glob(".staging*")) + list((self.root / "episodes").glob(".staging*"))
        self.assertEqual(len(staging_dirs), 0)

    def test_exact_cap_equality_acceptance_for_helper(self):
        """check_and_write_bytes accepts exact equality (len(data) == remaining_bytes)."""
        from icgs.data.archives import StorageLimitExceeded, check_and_write_bytes

        test_file = self.root / "exact_cap.bin"
        data = b"exact_bytes_1234"
        exact_len = len(data)

        # Exact equality accepted
        written = check_and_write_bytes(test_file, data, remaining_bytes=exact_len)
        self.assertEqual(written, exact_len)
        self.assertEqual(test_file.read_bytes(), data)

        # len > remaining rejected
        with self.assertRaises(StorageLimitExceeded):
            check_and_write_bytes(test_file, data, remaining_bytes=exact_len - 1)

    def test_repro1a_oversized_metadata_prevents_peak_exceeding_cap(self):
        """Repro 1a: Archive 1 transition cap 32768 with metadata note='x'*100000 must not exceed peak 32768 on disk."""
        import builtins
        import os
        from unittest import mock
        from icgs.data.archives import write_episode_archive

        record = self._make_episode(num_transitions=1, episode_id="ep-repro-1a")
        metadata = {"note": "x" * 100000}

        peak_bytes = 0

        def sample():
            nonlocal peak_bytes
            curr = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            if curr > peak_bytes:
                peak_bytes = curr

        orig_open = builtins.open

        def hooked_open(*args, **kwargs):
            f = orig_open(*args, **kwargs)
            sample()
            orig_close = f.close

            def hooked_close():
                sample()
                orig_close()
                sample()

            f.close = hooked_close
            return f

        orig_replace = os.replace

        def hooked_replace(src, dst):
            sample()
            res = orig_replace(src, dst)
            sample()
            return res

        with mock.patch("builtins.open", side_effect=hooked_open), mock.patch("os.replace", side_effect=hooked_replace):
            try:
                write_episode_archive(self.root, record, metadata=metadata, staging_byte_cap=32768)
            except Exception:
                pass
            sample()

        self.assertLessEqual(peak_bytes, 32768, f"Actual peak disk bytes {peak_bytes} exceeded cap 32768")

    def test_repro1b_oversized_sensor_and_controller_status_prevents_peak_exceeding_cap(self):
        """Repro 1b: Archive 1 transition cap 32768 with long 100k controller/sensor strings must not exceed peak 32768 on disk."""
        import builtins
        import os
        from unittest import mock
        from icgs.contracts.method import ExecutedTransition, TimedObservation
        from icgs.data.archives import write_episode_archive

        record = self._make_episode(num_transitions=1, episode_id="ep-repro-1b")
        # Inject 100k sensor_profile_id and controller_status
        old_trans = record["transitions"][0]
        long_sensor = "s" * 100000
        new_before = TimedObservation(
            old_trans.before.observation,
            old_trans.before.boundary,
            old_trans.before.simulator_timestamp,
            old_trans.before.measured_wall_timestamp,
            long_sensor,
        )
        new_after = TimedObservation(
            old_trans.after.observation,
            old_trans.after.boundary,
            old_trans.after.simulator_timestamp,
            old_trans.after.measured_wall_timestamp,
            long_sensor,
        )
        new_trans = ExecutedTransition(
            new_before,
            new_after,
            old_trans.command,
            old_trans.achieved_duration_s,
            old_trans.physics_substeps,
            "c" * 100000,
        )
        record["transitions"] = [new_trans]

        peak_bytes = 0

        def sample():
            nonlocal peak_bytes
            curr = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            if curr > peak_bytes:
                peak_bytes = curr

        orig_open = builtins.open

        def hooked_open(*args, **kwargs):
            f = orig_open(*args, **kwargs)
            sample()
            orig_close = f.close

            def hooked_close():
                sample()
                orig_close()
                sample()

            f.close = hooked_close
            return f

        orig_replace = os.replace

        def hooked_replace(src, dst):
            sample()
            res = orig_replace(src, dst)
            sample()
            return res

        with mock.patch("builtins.open", side_effect=hooked_open), mock.patch("os.replace", side_effect=hooked_replace):
            try:
                write_episode_archive(self.root, record, staging_byte_cap=32768)
            except Exception:
                pass
            sample()

        self.assertLessEqual(peak_bytes, 32768, f"Actual peak disk bytes {peak_bytes} exceeded cap 32768")

    def test_incompressible_points_exact_byte_sizing(self):
        """Incompressible random points exceeding cap must raise StorageLimitExceeded and write 0 bytes."""
        from icgs.data.archives import StorageLimitExceeded, write_episode_archive

        rng = np.random.RandomState(1337)
        # 1000 random points is incompressible float32 data
        pts = rng.randn(1000, 3).astype(np.float32)
        record = self._make_episode(num_transitions=1, episode_id="ep-incompress")
        t0 = TimedObservation(Observation(pts.astype(np.float64), np.eye(4), 0.0), 0, 1.0, 2.0, "s1")
        t1 = TimedObservation(Observation(pts.astype(np.float64), np.eye(4), 0.0), 1, 1.1, 2.1, "s1")
        cmd = TimedCommand(np.eye(4), 0, 0.1)
        record["transitions"] = [ExecutedTransition(t0, t1, cmd, 0.1, 1, "ok")]
        record["online_observations"] = [
            {"points": pts, "point_valid": np.ones(1000, dtype=bool), "T_w_e": np.eye(4), "grip": 0},
            {"points": pts, "point_valid": np.ones(1000, dtype=bool), "T_w_e": np.eye(4), "grip": 0},
        ]

        with self.assertRaises(StorageLimitExceeded):
            write_episode_archive(self.root, record, staging_byte_cap=500)

        # Confirm 0 files created on disk
        self.assertFalse((self.root / "episodes" / "ep-incompress").exists())
        self.assertEqual(len(list(self.root.glob("**/*.*"))), 0)

    def test_many_small_shards_exact_byte_staging_cleanup(self):
        """Many small shards staged sequentially clean up on limit exhaustion; peak <= cap."""
        import os
        from unittest import mock
        from icgs.configuration.method import MethodConfig
        from icgs.data.archives import StorageLimitExceeded, write_episode_archive

        # 5 transitions, 1 interval per shard -> 5 shards
        cfg = MethodConfig(dataset={"shard_intervals": 1})
        record = self._make_episode(num_transitions=5, episode_id="ep-many-shards")

        peak_bytes = 0

        def sample():
            nonlocal peak_bytes
            total = 0
            for r, _, files in os.walk(self.root):
                for f_name in files:
                    try:
                        total += (Path(r) / f_name).stat().st_size
                    except OSError:
                        pass
            if total > peak_bytes:
                peak_bytes = total

        orig_open = open

        def hooked_open(*args, **kwargs):
            sample()
            f = orig_open(*args, **kwargs)
            orig_close = f.close

            def hooked_close():
                sample()
                orig_close()
                sample()

            f.close = hooked_close
            return f

        cap = 4000  # Will fit ~1-2 shards, then fail on subsequent shard
        with mock.patch("builtins.open", side_effect=hooked_open):
            with self.assertRaises(StorageLimitExceeded):
                write_episode_archive(self.root, record, config=cfg, staging_byte_cap=cap)
            sample()

        self.assertLessEqual(peak_bytes, cap, f"Peak bytes {peak_bytes} exceeded cap {cap}")
        # Cleaned up on error
        self.assertFalse((self.root / "episodes" / "ep-many-shards").exists())
        staging_dirs = list((self.root / "episodes").glob(".tmp_ep_*"))
        self.assertEqual(len(staging_dirs), 0)

