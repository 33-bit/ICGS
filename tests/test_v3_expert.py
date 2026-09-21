"""Simulator-free tests for v3 expert motions."""

from __future__ import annotations

import unittest

from icgs.data.collection.v3.compiler import compile_v3_catalog
from icgs.data.collection.v3.scenes import get_scene_layout
from icgs.data.collection.v3.expert import (
    ARTICULATION_TYPES,
    CONTACT_PUSH_TYPES,
    ROUTINE_TYPES,
    assert_routine_supported,
    plan_step,
)
from icgs.data.collection.v3.protocol import V3_PROTOCOL


class ExpertPlanningTests(unittest.TestCase):
    def test_all_compiled_routines_are_supported(self):
        compiled = compile_v3_catalog()
        used = set()
        for spec in compiled.values():
            assert_routine_supported(spec.routine)
            used.update(step["type"] for step in spec.routine)
        self.assertTrue(used <= ROUTINE_TYPES)
        self.assertIn("push", used)
        self.assertTrue(ARTICULATION_TYPES & used)
        self.assertIn("grasp", used)

    def test_place_targets_sit_at_support_rest_height(self):
        layout = get_scene_layout("T02")
        pad = layout["objects"]["pad"]
        obj = layout["objects"]["object_a"]
        target = layout["objects"]["target_a"]
        rest = pad["pos"][2] + pad["size"][2] / 2 + obj["size"][2] / 2
        self.assertAlmostEqual(target["pos"][2], rest)
        self.assertGreater(target["pos"][2], pad["pos"][2])

    def test_push_does_not_grasp(self):
        poses = {"blocker": [0.25, -0.05, 0.775], "push_target": [0.25, 0.08, 0.775]}
        motions = plan_step({"type": "push", "obj": "blocker", "target": "push_target"}, poses)
        self.assertTrue(all(not item.get("grasp") for item in motions))
        self.assertTrue(any(item["kind"] == "move" for item in motions))
        self.assertNotIn("pick_place", [item.get("kind") for item in motions])
        final_contact = motions[-2]["xyz"]
        self.assertGreater(final_contact[1], poses["push_target"][1])

    def test_worker_does_not_remap_push_to_pick_place(self):
        from pathlib import Path

        text = Path("scripts/colab_v3_pilot_episodes_worker.py").read_text(encoding="utf-8")
        self.assertNotIn('exec_step["type"] = "pick_place"', text)
        self.assertIn('retry = {"type": "push"', text)

    def test_open_close_slide_along_axis_not_free_place(self):
        poses = {
            "drawer_handle": [0.25, 0.04, 0.775],
            "open_target": [0.25, -0.06, 0.775],
            "close_target": [0.25, 0.04, 0.775],
        }
        opened = plan_step(
            {"type": "open_articulation", "obj": "drawer_handle", "target": "open_target", "axis": "y"},
            poses,
        )
        self.assertTrue(any(item["kind"] == "slide" for item in opened))
        self.assertEqual(opened[-2]["kind"], "grip")
        self.assertEqual(opened[-2]["grip"], 1.0)

    def test_place_waypoints_are_object_frame_without_extra_height(self):
        poses = {"object_a": [0.25, -0.10, 0.775], "target_a": [0.25, 0.15, 0.775]}
        motions = plan_step({"type": "place", "obj": "object_a", "target": "target_a"}, poses)
        grasped = [item for item in motions if item.get("grasp")]
        self.assertTrue(grasped)
        self.assertEqual(grasped[-1]["xyz"][2], 0.775)
        self.assertEqual(grasped[-1].get("frame"), "object")

    def test_pause_hold_is_not_a_timestamp_edit(self):
        motions = plan_step({"type": "pause_hold", "intervals": 4}, {})
        self.assertEqual(motions, [{"kind": "pause", "intervals": 4, "grasp": False}])

    def test_pilot_programs_have_executable_plans(self):
        compiled = compile_v3_catalog()
        for pid in V3_PROTOCOL.pilot_program_ids:
            spec = compiled[pid]
            poses = {name: geom[0] for name, geom in spec.objects.items()}
            for step in spec.routine:
                motions = plan_step(step, poses)
                self.assertGreaterEqual(len(motions), 1)
