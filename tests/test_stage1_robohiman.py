"""Stage-1 record/label/view contracts and the RoboHiMan camera math (no simulator)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from icgs.data.stage1.labels import derive_event_labels, label_consistency_report
from icgs.data.stage1.schema import SCHEMA_VERSION, perturbation_families, validate_manifest
from icgs.data.stage1.store import read_episode, write_episode
from icgs.data.stage1.views import ReferencePolicyRequired, build_views
from icgs.environments.robohiman.camera import (
    camera_cloud,
    decode_mask_handles,
    depth_to_world,
    metric_depth,
    project_world,
)
from icgs.environments.robohiman.pins import level_for, task_family


PREDICATES = ["drawer_open", "drawer_closed", "item_grasped", "item_in_drawer"]
SPECS = [
    {"event_id": "drawer_opened", "relation": "drawer_open"},
    {"event_id": "item_grasped", "relation": "item_grasped"},
    {"event_id": "item_placed", "relation": "item_in_drawer", "prerequisites": ["item_grasped"],
     "current_requirements": ["drawer_open"]},
    {"event_id": "drawer_closed_after", "relation": "drawer_closed", "prerequisites": ["item_placed"],
     "count_only_when_eligible": True},
]


def _trace(rows):
    return np.array(rows, dtype=bool)


class LabelTests(unittest.TestCase):
    def test_history_is_not_current_relation(self):
        trace = _trace([
            [0, 1, 0, 0],  # closed drawer at reset: not an occurrence of the close event
            [1, 0, 0, 0],  # opened
            [1, 0, 1, 0],  # grasped
            [1, 0, 1, 1],  # placed while open
            [1, 0, 0, 1],  # released: rho stays, nu drops
            [0, 1, 0, 1],  # closed after placement
        ])
        labels = derive_event_labels(trace, PREDICATES, SPECS)
        self.assertEqual(labels["first_occurrence"].tolist(), [1, 2, 3, 5])
        self.assertFalse(labels["rho"][0, 3], "closed-at-reset must not count as the close event")
        self.assertTrue(labels["rho"][4, 1] and not labels["nu"][4, 1], "history true, relation false")
        self.assertEqual(labels["alpha"].tolist(), [0, 1, 2, 3, 3, 4])
        report = label_consistency_report(labels, SPECS)
        self.assertTrue(report["occurrences_in_spec_order"])
        self.assertTrue(report["events"][1]["undone_after_occurrence"])

    def test_ineligible_pending_event_aligns_to_null(self):
        trace = _trace([[0, 1, 0, 0], [0, 1, 1, 0]])  # grasped but drawer never opened
        labels = derive_event_labels(trace, PREDICATES, SPECS)
        self.assertEqual(labels["alpha"].tolist(), [0, 0])
        self.assertFalse(labels["epsilon"][1, 2])

    def test_unknown_predicate_and_order_rejected(self):
        with self.assertRaises(ValueError):
            derive_event_labels(_trace([[0, 0, 0, 0]]), PREDICATES,
                                [{"event_id": "x", "relation": "not_recorded"}])
        with self.assertRaises(ValueError):
            derive_event_labels(_trace([[0, 0, 0, 0]]), PREDICATES,
                                [{"event_id": "a", "relation": "drawer_open", "prerequisites": ["b"]},
                                 {"event_id": "b", "relation": "drawer_open"}])


def _episode(episode_id="run-open_drawer-v0-s0-0000", status="success", execution=()):
    steps = 4
    arrays = {
        "step_sim_time": np.arange(steps + 1) * 0.05,
        "step_joint_positions": np.zeros((steps + 1, 7)),
        "step_joint_velocities": np.zeros((steps + 1, 7)),
        "step_tip_pose": np.tile([0, 0, 1, 0, 0, 0, 1.0], (steps + 1, 1)),
        "step_gripper_joint_positions": np.zeros((steps + 1, 2)),
        "step_predicates": np.zeros((steps + 1, 1), dtype=bool),
        "step_task_success": np.array([0, 0, 0, 1, 1], dtype=bool),
        "cmd_arm_joint_target": np.ones((steps, 7)),
        "cmd_arm_joint_target_valid": np.array([1, 1, 0, 0], dtype=bool),
        "cmd_arm_joint_velocity_valid": np.zeros(steps, dtype=bool),
        "cmd_gripper_joint_velocity": np.zeros((steps, 2)),
        "cmd_gripper_joint_velocity_valid": np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=bool),
        "cmd_grasp_event": np.zeros(steps, dtype=np.int8),
        "cmd_arm_teleport_calls": np.array([1, 0, 0, 0], dtype=np.int16),
        "cmd_phase": np.array([1, 1, 2, 2], dtype=np.int8),
        "frame_step": np.array([0, 2, 4]),
        "label_alpha": np.array([0, 0, 0, 1, 1]),
    }
    perturbation = {"benchmark_factors": [{"type": "camera_pose", "name": "any", "enabled": False}],
                    "execution": list(execution)}
    perturbation["families"] = list(perturbation_families(perturbation))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "source": {"benchmark": "robohiman", "benchmark_revision": "x", "task": "open_drawer",
                   "task_family": "atomic", "level": "A", "variation_index": 0},
        "perturbation": perturbation,
        "outcome": {"status": status, "recoverability": "unknown", "task_success_final": status == "success"},
        "lineage": {"collection_run_id": "run", "parent_episode_id": None},
        "environment": {}, "controller": {"physics_dt": 0.05, "native_command": "joint targets"},
        "code": {}, "cameras": {"names": ["front"]},
        "predicates": {"names": ["drawer_open"]},
        "events": {"specs": [{"event_id": "drawer_opened"}]},
        "arrays": {"sha256": "pending"},
        "counts": {"steps": steps, "frames": 3},
    }
    return manifest, arrays


class StoreAndViewTests(unittest.TestCase):
    def test_roundtrip_views_and_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, arrays = _episode()
            write_episode(tmp, manifest, arrays)
            failure = [{"family": "object_displacement", "external_intervention": True, "applied_at_step": 2}]
            manifest2, arrays2 = _episode("run-open_drawer-v0-s0-0001", "failure", failure)
            write_episode(tmp, manifest2, arrays2)
            stored, loaded = read_episode(Path(tmp) / "episodes" / manifest["episode_id"])
            self.assertEqual(stored["arrays"]["members"]["cmd_phase"]["dtype"], "int8")
            np.testing.assert_array_equal(loaded["frame_step"], arrays["frame_step"])
            summary = build_views(tmp)
            self.assertEqual(summary["views"]["D_geom"]["rows"], 6)
            self.assertEqual(summary["views"]["D_dyn"]["rows"], 8)
            self.assertEqual(summary["views"]["D_task"]["rows"], 10)
            rows = [json.loads(line) for line in (Path(tmp) / "views" / "D_dyn.jsonl").read_text().splitlines()]
            self.assertTrue(any(row["external_intervention"] for row in rows))
            self.assertTrue(any(row["kinematic_arm_excursion"] for row in rows))
            self.assertEqual({row["outcome"] for row in rows}, {"success", "failure"})
            with self.assertRaises(ReferencePolicyRequired):
                build_views(tmp, view_names=("D_value",))
            with self.assertRaises(FileExistsError):
                write_episode(tmp, manifest, arrays)
            (Path(tmp) / "episodes" / manifest["episode_id"] / "arrays.npz").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                read_episode(Path(tmp) / "episodes" / manifest["episode_id"])

    def test_manifest_rejects_conflated_or_unsupported_claims(self):
        manifest, _ = _episode()
        manifest["perturbation"]["families"] = ["execution_pose_offset"]
        with self.assertRaisesRegex(ValueError, "families"):
            validate_manifest(manifest)
        manifest, _ = _episode(status="failure")
        manifest["outcome"]["recoverability"] = "recoverable"
        with self.assertRaisesRegex(ValueError, "evidence"):
            validate_manifest(manifest)
        manifest, _ = _episode(status="success")
        manifest["outcome"]["task_success_final"] = False
        with self.assertRaisesRegex(ValueError, "success"):
            validate_manifest(manifest)
        manifest, arrays = _episode()
        arrays["step_sim_time"][2] = arrays["step_sim_time"][1]
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "increase"):
            write_episode(tmp, manifest, arrays)
        self.assertEqual(list(Path(tmp).glob("episodes/*")) if Path(tmp).exists() else [], [])


def _pyrep_reference(depth, extrinsics, intrinsics):
    """Literal transcription of PyRep 231a1ac pointcloud_from_depth_and_camera_params."""
    h, w = depth.shape
    x = np.reshape(np.tile(np.arange(w), [h]), (h, w, 1)).astype(np.float32)
    y = np.transpose(np.reshape(np.tile(np.arange(h), [w]), (w, h, 1)).astype(np.float32), (1, 0, 2))
    upc = np.concatenate((x, y, np.ones_like(x)), -1)
    pc = upc * np.expand_dims(depth, -1)
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R_inv = extrinsics[:3, :3].T
    proj = np.matmul(intrinsics, np.concatenate((R_inv, -np.matmul(R_inv, C)), -1))
    inv = np.linalg.inv(np.concatenate([proj, [np.array([0, 0, 0, 1])]]))[0:3]
    coords = np.concatenate([pc, np.ones((h, w, 1))], -1).reshape(h * w, -1).T
    return (inv @ coords).T.reshape(h, w, 3)


class CameraTests(unittest.TestCase):
    def setUp(self):
        angle = np.radians(30)
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        self.extrinsics = np.eye(4)
        self.extrinsics[:3, :3] = rotation
        self.extrinsics[:3, 3] = [1.2, -0.3, 1.6]
        self.intrinsics = np.array([[-110.8, 0, 64.0], [0, -110.8, 64.0], [0, 0, 1.0]])
        rng = np.random.default_rng(0)
        self.depth = rng.uniform(0.2, 0.9, size=(12, 16)).astype(np.float32)

    def test_matches_literal_pyrep_algorithm(self):
        metres = metric_depth(self.depth, 0.01, 4.5)
        ours = depth_to_world(metres, self.extrinsics, self.intrinsics)
        reference = _pyrep_reference(metres, self.extrinsics, self.intrinsics)
        self.assertLess(np.abs(ours - reference).max(), 1e-9)

    def test_projection_inverts_backprojection_and_depth_is_metric(self):
        metres = metric_depth(self.depth, 0.01, 4.5)
        world = depth_to_world(metres, self.extrinsics, self.intrinsics).reshape(-1, 3)
        uv, z = project_world(world, self.extrinsics, self.intrinsics)
        v, u = np.divmod(np.arange(world.shape[0]), self.depth.shape[1])
        self.assertLess(np.abs(uv - np.stack([u, v], 1)).max(), 1e-6)
        self.assertLess(np.abs(z - metres.reshape(-1)).max(), 1e-9)
        # Depth is distance along the optical axis: camera-frame z equals metric depth.
        camera = (world - self.extrinsics[:3, 3]) @ self.extrinsics[:3, :3]
        self.assertLess(np.abs(camera[:, 2] - metres.reshape(-1)).max(), 1e-9)

    def test_far_plane_is_invalid_and_mask_decoding_rounds(self):
        depth = self.depth.copy()
        depth[0, 0] = 1.0
        _, valid = camera_cloud(depth, 0.01, 4.5, self.extrinsics, self.intrinsics)
        self.assertFalse(valid[0])
        handle = 70000
        rgb = np.array([[[handle % 256, (handle // 256) % 256, handle // 65536]]], dtype=np.float64) / 255.0
        self.assertEqual(int(decode_mask_handles(rgb)[0, 0]), handle)


class PinsTests(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(level_for("open_drawer", 0), "A")
        self.assertEqual(level_for("open_drawer", 3), "AP")
        self.assertEqual(level_for("put_in_and_close", 0), "C")
        self.assertEqual(task_family("transfer_box"), "compositional")
        with self.assertRaises(KeyError):
            task_family("T01")


if __name__ == "__main__":
    unittest.main()
