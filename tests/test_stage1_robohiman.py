"""Stage-1 record/label/view contracts and the RoboHiMan camera math (no simulator)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from icgs.data.stage1.labels import (
    context_alignment,
    context_event_tokens,
    derive_monitor_states,
    label_consistency_report,
)
from icgs.data.stage1.schema import validate_arrays, validate_manifest
from icgs.data.stage1.store import read_episode, write_episode
from icgs.environments.robohiman.camera import (
    camera_cloud,
    decode_mask_handles,
    depth_to_world,
    metric_depth,
    project_world,
)
from icgs.environments.robohiman.pins import level_for, task_family
from stage1_fixtures import make_episode


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


def _slow(rows, repeat=5):
    """Hold every row ``repeat`` boundaries so validity margins leave stable interiors."""
    return np.repeat(_trace(rows), repeat, axis=0)


class MonitorStateTests(unittest.TestCase):
    def test_history_is_not_current_relation(self):
        trace = _trace([
            [0, 1, 0, 0],  # closed drawer at reset: not an occurrence of the close event
            [1, 0, 0, 0],  # opened
            [1, 0, 1, 0],  # grasped
            [1, 0, 1, 1],  # placed while open
            [1, 0, 0, 1],  # released: rho stays, nu drops
            [0, 1, 0, 1],  # closed after placement
        ])
        states = derive_monitor_states(trace, PREDICATES, SPECS, margin=0)
        self.assertEqual(states["first_occurrence"].tolist(), [1, 2, 3, 5])
        self.assertFalse(states["rho"][0, 3], "closed-at-reset must not count as the close event")
        self.assertTrue(states["rho"][4, 1] and not states["nu"][4, 1], "history true, relation false")
        self.assertEqual(states["event_id"].tolist(), [0, 1, 2, 3, 3, 4])
        report = label_consistency_report(states, SPECS)
        self.assertTrue(report["occurrences_in_spec_order"])
        self.assertTrue(report["events"][1]["undone_after_occurrence"])

    def test_ineligible_pending_event_is_not_eligible(self):
        trace = _trace([[0, 1, 0, 0], [0, 1, 1, 0]])  # grasped but drawer never opened
        states = derive_monitor_states(trace, PREDICATES, SPECS, margin=0)
        self.assertEqual(states["event_id"].tolist(), [0, 0])
        self.assertFalse(states["epsilon"][1, 2])

    def test_validity_masks_near_changes(self):
        trace = _slow([[0, 1, 0, 0], [1, 0, 0, 0]])  # drawer opens at boundary 5
        states = derive_monitor_states(trace, PREDICATES, SPECS, margin=2)
        self.assertEqual(states["nu_valid"][:, 0].tolist(), [1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
        self.assertEqual(states["rho_valid"][:, 0].tolist(), [1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
        self.assertTrue(states["nu_valid"][:, 2].all(), "unchanged relations stay valid")
        self.assertFalse(states["event_id_valid"][4], "event id is uncertain at its transition")

    def test_unobservable_event_gets_no_fabricated_postcondition(self):
        names = ["near_handle", "grasped"]
        specs = [{"event_id": "reach", "relation": "near_handle", "observable": False},
                 {"event_id": "grasp", "relation": "grasped"}]
        trace = np.repeat(np.array([[0, 0], [1, 0], [1, 1]], dtype=bool), 5, axis=0)
        states = derive_monitor_states(trace, names, specs)
        self.assertFalse(states["nu_valid"][:, 0].any())
        self.assertFalse(states["rho_valid"][:, 0].any())
        self.assertFalse(states["rho"][:, 0].any(), "free-space event never claims an occurrence")
        self.assertEqual(states["first_occurrence"][0], -1)
        self.assertTrue(states["epsilon_valid"][:, 0].all(), "prerequisites stay observable")
        self.assertFalse(states["event_id_valid"][states["event_id"] == 0].any())

    def test_unordered_pending_events_make_event_id_invalid(self):
        names = ["a", "b"]
        specs = [{"event_id": "a", "relation": "a"}, {"event_id": "b", "relation": "b"}]
        states = derive_monitor_states(np.zeros((6, 2), dtype=bool), names, specs)
        self.assertEqual(states["event_id"].tolist(), [0] * 6)
        self.assertFalse(states["event_id_valid"].any(), "two unordered pending events: no single next event")
        ordered = [{"event_id": "a", "relation": "a"}, {"event_id": "b", "relation": "b", "prerequisites": ["a"]}]
        self.assertTrue(derive_monitor_states(np.zeros((6, 2), dtype=bool), names, ordered)["event_id_valid"].all())

    def test_conjunctive_relation_requires_release(self):
        names = ["in_region", "grasped"]
        trace = np.array([[0, 1], [1, 1], [1, 0]], dtype=bool)  # detected while carried, then released
        states = derive_monitor_states(trace, names, [{"event_id": "placed", "relation": "in_region&!grasped"}])
        self.assertEqual(states["first_occurrence"].tolist(), [2])
        with self.assertRaises(ValueError):
            derive_monitor_states(trace, names, [{"event_id": "x", "relation": "in_region&!missing"}])

    def test_unknown_predicate_and_order_rejected(self):
        with self.assertRaises(ValueError):
            derive_monitor_states(_trace([[0, 0, 0, 0]]), PREDICATES,
                                  [{"event_id": "x", "relation": "not_recorded"}])
        with self.assertRaises(ValueError):
            derive_monitor_states(_trace([[0, 0, 0, 0]]), PREDICATES,
                                  [{"event_id": "a", "relation": "drawer_open", "prerequisites": ["b"]},
                                   {"event_id": "b", "relation": "drawer_open"}])


class ContextAlignmentTests(unittest.TestCase):
    """alpha is built from a query prefix and an independent context, not stored per episode."""

    def setUp(self):
        rows = [[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]]
        self.query = derive_monitor_states(_slow(rows), PREDICATES, SPECS, margin=1)
        self.events = [spec["event_id"] for spec in SPECS]
        self.boundaries = [2, 7, 12, 17, 22, 27, 29]  # interiors of the held rows

    def test_target_advances_and_ends_null(self):
        tokens = context_event_tokens({"drawer_opened": 3, "item_grasped": 9, "item_placed": 14,
                                       "drawer_closed_after": 20})
        labels = context_alignment(self.query, self.events, tokens, self.boundaries)
        self.assertEqual(labels["alignment_target"].tolist(), [0, 1, 2, 3, 3, 4, 4])
        self.assertEqual(int(labels["alignment_target"][-1]), len(tokens), "null index = number of tokens")
        self.assertTrue(labels["alignment_valid"].all())

    def test_alignment_depends_on_the_context(self):
        # A context that never closed the drawer (e.g. a *_without_close demo) has three tokens,
        # so the same query prefix aligns to null as soon as the item is placed.
        short = context_event_tokens({"drawer_opened": 3, "item_grasped": 9, "item_placed": 14,
                                      "drawer_closed_after": -1})
        full = context_event_tokens({"drawer_opened": 3, "item_grasped": 9, "item_placed": 14,
                                     "drawer_closed_after": 20})
        a = context_alignment(self.query, self.events, short, self.boundaries)["alignment_target"]
        b = context_alignment(self.query, self.events, full, self.boundaries)["alignment_target"]
        self.assertEqual((len(short), int(a[4])), (3, 3), "short context: null after placement")
        self.assertEqual((len(full), int(b[4])), (4, 3), "full context: closing is still pending")

    def test_order_disagreement_and_uncertain_history_are_invalid(self):
        # The context grasped before opening; the query opened first.
        tokens = context_event_tokens({"item_grasped": 2, "drawer_opened": 6})
        labels = context_alignment(self.query, self.events, tokens, [7, 12])
        self.assertEqual(labels["alignment_target"].tolist(), [0, 2])
        self.assertEqual(labels["alignment_valid"].tolist(), [False, True])
        near = context_alignment(self.query, self.events, context_event_tokens({"drawer_opened": 1}), [5])
        self.assertFalse(bool(near["alignment_valid"][0]), "a token whose history is uncertain is invalid")
        with self.assertRaises(ValueError):
            context_alignment(self.query, self.events, [("not_in_query", 0)], [0])


class StoreAndSchemaTests(unittest.TestCase):
    def setUp(self):
        self.id = "ep-open_drawer-a81f2c3d-0000"

    def test_roundtrip_is_atomic_immutable_and_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, arrays = make_episode(self.id, provenance=None, seed=1)
            path = write_episode(tmp, manifest, arrays)
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "episodes").iterdir()), [self.id])
            for item in path.iterdir():
                self.assertEqual(item.stat().st_mode & 0o777, 0o444, item.name)
            stored, loaded = read_episode(path)
            self.assertEqual(stored["arrays"]["members"]["cmd_phase"]["dtype"], "int8")
            np.testing.assert_array_equal(loaded["frame_step"], arrays["frame_step"])
            with self.assertRaises(FileExistsError):
                write_episode(tmp, manifest, arrays)
            path.joinpath("arrays.npz").chmod(0o644)
            path.joinpath("arrays.npz").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                read_episode(path)

    def test_failed_write_leaves_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, arrays = make_episode(self.id, provenance=None, seed=1)
            arrays["label_alpha"] = np.zeros(13, dtype=np.int64)
            with self.assertRaisesRegex(ValueError, "retired"):
                write_episode(tmp, manifest, arrays)
            self.assertEqual(list((Path(tmp) / "episodes").iterdir()), [], "no partial or temporary directory")

    def test_episode_id_format(self):
        manifest, _ = make_episode(self.id, provenance=None, seed=1)
        validate_manifest(manifest)
        for bad in ("run-open_drawer-v0-s0-0000", "ep-open_drawer-v0-s0-a81f2c3d-0000",
                    "ep-open_drawer-A81F2C3D-0000", "ep-close_drawer-a81f2c3d-0000", "ep-open_drawer-a81f2c3d-7"):
            manifest["episode_id"] = bad
            with self.subTest(bad), self.assertRaises(ValueError):
                validate_manifest(manifest)

    def test_timing_semantics(self):
        manifest, arrays = make_episode(self.id, provenance=None, seed=1)
        validate_arrays(manifest, arrays)
        jumped = dict(arrays, step_sim_time=arrays["step_sim_time"].copy())
        jumped["step_sim_time"][5:] += 0.05  # one row spans two physics steps
        with self.assertRaisesRegex(ValueError, "one physics step"):
            validate_arrays(manifest, jumped)
        with self.assertRaisesRegex(ValueError, "retired"):
            validate_arrays(manifest, dict(arrays, cmd_wall_s=np.zeros(12)))
        for key, value in (("model_dt_s", 0.05), ("transition", "one model step"), ("physics_dt_s", 0.1)):
            broken, _ = make_episode(self.id, provenance=None, seed=1)
            broken["timing"][key] = value
            with self.subTest(key), self.assertRaises(ValueError):
                validate_manifest(broken)

    def test_frame_arrays_follow_frame_step(self):
        manifest, arrays = make_episode(self.id, provenance=None, seed=1)
        with self.assertRaisesRegex(ValueError, "F="):
            validate_arrays(manifest, dict(arrays, cam_front_depth=arrays["cam_front_depth"][:-1]))

    def test_mask_legend_required_when_masks_recorded(self):
        manifest, _ = make_episode(self.id, provenance=None, seed=1)
        del manifest["cameras"]["mask_legend"]
        with self.assertRaises(ValueError):
            validate_manifest(manifest)
        manifest, _ = make_episode(self.id, provenance=None, seed=1)
        manifest["cameras"]["mask_legend"]["17"].pop("role")
        with self.assertRaisesRegex(ValueError, "role"):
            validate_manifest(manifest)

    def test_manifest_rejects_conflated_or_unsupported_claims(self):
        manifest, _ = make_episode(self.id, provenance=None, seed=1)
        manifest["perturbation"]["families"] = ["execution_pose_offset"]
        with self.assertRaisesRegex(ValueError, "families"):
            validate_manifest(manifest)
        manifest, _ = make_episode(self.id, provenance=None, seed=1, status="failure")
        manifest["outcome"]["recoverability"] = "recoverable"
        with self.assertRaisesRegex(ValueError, "evidence"):
            validate_manifest(manifest)
        manifest, _ = make_episode(self.id, provenance=None, seed=1)
        manifest["outcome"]["task_success_final"] = False
        with self.assertRaisesRegex(ValueError, "success"):
            validate_manifest(manifest)


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
