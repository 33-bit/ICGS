"""Simulator-free tests for canonical generation expert motions."""

from __future__ import annotations

import unittest

from icgs.data.collection.generation.compiler import compile_generation_catalog
from icgs.data.collection.generation.scenes import get_scene_layout
from icgs.data.collection.generation.expert import (
    ARTICULATION_TYPES,
    CONTACT_PUSH_TYPES,
    ROUTINE_TYPES,
    assert_routine_supported,
    plan_step,
)
from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL


class ExpertPlanningTests(unittest.TestCase):
    def test_all_compiled_routines_are_supported(self):
        compiled = compile_generation_catalog()
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

    def test_push_stops_before_target_for_open_contact_face(self):
        poses = {"blocker": [0.25, -0.05, 0.775], "push_target": [0.25, 0.08, 0.775]}
        motions = plan_step({"type": "push", "obj": "blocker", "target": "push_target"}, poses)
        final_contact = [item for item in motions if item.get("push_contact")][-1]["xyz"]
        self.assertAlmostEqual(
            final_contact[1],
            poses["push_target"][1] - GENERATION_PROTOCOL.push_contact_offset_m,
            places=3,
        )

    def test_push_contacts_at_object_centre_height_on_its_centre_line(self):
        poses = {"object_a": [0.24, -0.10, 0.77], "gate_wp": [0.25, 0.05, 0.775], "target_a": [0.25, 0.16, 0.77]}
        step = {
            "type": "push", "obj": "object_a", "target": "target_a", "via": "gate_wp", "push_z": 0.0,
            "contact_offset_m": 0.04, "approach_waypoint": {"dx_m": 0.01, "dy_m": -0.01, "dz_m": 0.005},
        }
        motions = plan_step(step, poses)
        low = [item for item in motions if item["xyz"][2] < 0.8]
        self.assertTrue(all(abs(item["xyz"][2] - 0.77) < 1e-9 for item in low))
        pre = motions[1]["xyz"]
        # The pre-push point lies behind the object on the line to the via point.
        direction = [0.25 - 0.24, 0.05 + 0.10]
        cross = (pre[0] - 0.24) * direction[1] - (pre[1] + 0.10) * direction[0]
        self.assertAlmostEqual(cross, 0.0, places=9)
        contacts = [item["xyz"] for item in motions if item.get("push_contact")]
        self.assertEqual(len(contacts), 2)
        self.assertAlmostEqual(contacts[-1][1], 0.16 - 0.04, places=6)
        # Only the last contact is marked for closed-loop completion, with its direction.
        finals = [item for item in motions if item.get("push_final")]
        self.assertEqual(len(finals), 1)
        self.assertIs(finals[0], [item for item in motions if item.get("push_contact")][-1])
        direction = finals[0]["push_direction"]
        self.assertAlmostEqual(direction[0] ** 2 + direction[1] ** 2, 1.0, places=9)
        self.assertGreater(direction[1], 0.99)

    def test_worker_does_not_remap_push_to_pick_place(self):
        from pathlib import Path

        text = Path("scripts/generation_episode_worker.py").read_text(encoding="utf-8")
        self.assertNotIn('exec_step["type"] = "pick_place"', text)
        # Push retries keep the push primitive (see test_step_retry_* in the worker tests).
        self.assertIn('if kind in {"grasp", "push", "reach"} | ARTICULATION_STEP_TYPES:', text)

    def test_episode_worker_has_no_machine_specific_content_paths(self):
        from pathlib import Path

        text = Path("scripts/generation_episode_worker.py").read_text(encoding="utf-8")
        self.assertNotIn("/content", text)
        self.assertIn("Path(__file__).resolve()", text)

    def test_pilot_writer_materializes_training_layout(self):
        from pathlib import Path

        text = Path("scripts/generation_episode_worker.py").read_text(encoding="utf-8")
        self.assertIn("write_training_episode_layout", text)
        self.assertIn('write_dir / "layout"', text)
        self.assertIn('"attempt_id": f"att-{plan.episode_id}"', text)

    def test_pilot_persists_only_outcome_not_result_class(self):
        from pathlib import Path

        text = Path("scripts/generation_episode_worker.py").read_text(encoding="utf-8")
        self.assertIn('execution["outcome"] = result_class', text)
        self.assertIn('"result_class"}', text.replace(", ", ""))

    def test_pilot_invalid_attempt_retains_safe_prefix_arrays(self):
        from pathlib import Path

        text = Path("scripts/generation_episode_worker.py").read_text(encoding="utf-8")
        self.assertIn('point_offsets=point_offsets', text)
        self.assertIn('T_w_e=np.asarray([item["T_w_e"] for item in timed]', text)
        self.assertNotIn('dtype=object', text)

    def test_pilot_public_receipt_excludes_all_numeric_private_fields(self):
        from pathlib import Path

        text = Path("scripts/generation_episode_worker.py").read_text(encoding="utf-8")
        self.assertIn('if not k.startswith("_")', text)

    def test_pilot_uses_measured_wrist_points_not_tip_repetition(self):
        from pathlib import Path

        text = Path("scripts/generation_episode_worker.py").read_text(encoding="utf-8")
        self.assertIn("wrist_point_cloud", text)
        self.assertNotIn("np.repeat(pos.reshape(1, 3), 8", text)
        self.assertIn("obs_config.wrist_camera.depth = True", text)

    def test_pilot_does_not_relabel_invalid_observation_as_valid_failure(self):
        from pathlib import Path

        text = Path("scripts/generation_episode_worker.py").read_text(encoding="utf-8")
        self.assertIn('observation_valid=row.get("result_class") != "invalid_observation"', text)

    def test_distributed_worker_propagates_approved_binding(self):
        from pathlib import Path

        text = Path("scripts/generation_worker.py").read_text(encoding="utf-8")
        self.assertIn("ICGS_GENERATION_BINDING_JSON", text)
        self.assertIn("approved_manifest", text)

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
        compiled = compile_generation_catalog()
        for pid in GENERATION_PROTOCOL.pilot_program_ids:
            spec = compiled[pid]
            poses = {name: geom[0] for name, geom in spec.objects.items()}
            for step in spec.routine:
                motions = plan_step(step, poses)
                self.assertGreaterEqual(len(motions), 1)


class HeldContinuationTests(unittest.TestCase):
    """The scripted expert must not open the gripper mid-program on a held object."""

    def _motions(self, spec):
        poses = {name: list(geom[0]) for name, geom in spec.objects.items()}
        for step in spec.routine:
            for motion in plan_step(step, poses):
                yield step, motion

    def test_no_compiled_program_opens_the_gripper_while_holding(self):
        for program_id, spec in compile_generation_catalog().items():
            holding = None
            for step, motion in self._motions(spec):
                if motion["kind"] == "grip":
                    holding = motion.get("grasp_obj") if motion["grip"] < 0.5 else None
                    continue
                if motion["kind"] in {"move", "slide"} and motion.get("grip", 1.0) > 0.5:
                    self.assertIsNone(
                        holding,
                        f"{program_id} {step['type']} opens the gripper while holding {holding}",
                    )

    def test_every_grasp_is_released_by_an_explicit_grip_or_program_end(self):
        for program_id, spec in compile_generation_catalog().items():
            grips = [motion for _step, motion in self._motions(spec) if motion["kind"] == "grip"]
            for index, item in enumerate(grips[:-1]):
                if item["grip"] < 0.5:
                    self.assertGreater(grips[index + 1]["grip"], 0.5, f"{program_id} double close")

    def test_retrieve_then_place_is_one_grasp(self):
        compiled = compile_generation_catalog()
        for program_id in ("T18", "T20", "V03"):
            types = [(step["type"], step.get("obj"), bool(step.get("held"))) for step in compiled[program_id].routine]
            self.assertIn(("lift", "object_a", True), types, program_id)
            self.assertEqual(
                sum(1 for kind, obj, _held in types if obj == "object_a" and kind == "pick_place"), 0, program_id,
            )

    def test_transport_then_place_keeps_the_object(self):
        compiled = compile_generation_catalog()
        for program_id in ("G1", "G2", "G3", "G4"):
            transport = next(step for step in compiled[program_id].routine if step["type"] == "transport_through_aperture")
            self.assertTrue(transport.get("held"), program_id)
            self.assertIs(transport.get("release"), False, program_id)
            motions = plan_step(transport, {name: geom[0] for name, geom in compiled[program_id].objects.items()})
            self.assertFalse([item for item in motions if item["kind"] == "grip"], program_id)
