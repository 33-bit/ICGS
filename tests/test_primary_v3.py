"""Phase-1 primary v3 protocol: catalog/compiler parity, diversity, schema, views."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from icgs.data.collection.programs import get_program, program_catalog
from icgs.data.collection.v3.camera import CAMERA_PROFILES, get_camera_profile
from icgs.data.collection.v3.compiler import compile_v3_catalog, events_from_steps
from icgs.data.collection.v3.diversity import (
    episode_seed,
    is_duplicate_episode,
    sample_randomization,
    scene_signature,
)
from icgs.data.collection.v3.layout import initialize_v3_dataset_layout
from icgs.data.collection.v3.manifest import build_v3_manifest
from icgs.data.collection.v3.perturbations import (
    SIMULATOR_CRASH,
    SUCCESS,
    VALID_FAILURE,
    apply_perturbation,
    classify_attempt_outcome,
    perturbation_quota,
)
from icgs.data.collection.v3.protocol import V3_PROTOCOL
from icgs.data.collection.v3.report import observed_semantics_report, program_parity_report, training_eligible_ids
from icgs.data.collection.v3.steps import PROGRAM_FAMILIES, structured_steps
from icgs.data.datasets.v3_views import SUPPORTED_V3_VIEWS, build_v3_view
from icgs.data.schemas.episodes_v2 import validate_episode_v2


V2_MANIFEST = Path("artifacts/composition/approved_composition_manifest.json")
V2_PROFILE = Path("src/icgs/configuration/profiles/icgs_primary.json")
V3_PROFILE = Path("src/icgs/configuration/profiles/icgs_primary_v3.json")

PILOT_IDS = (
    "T01",
    "T02",
    "T03",
    "T04",
    "T07",
    "T09",
    "T10",
    "T11",
    "T12",
    "T13",
    "T17",
)


class ProtocolFreezeTests(unittest.TestCase):
    def test_v3_protocol_ids_are_frozen(self):
        self.assertEqual(V3_PROTOCOL.dataset_version, "icgs-primary-v3")
        self.assertEqual(V3_PROTOCOL.program_manifest_version, 3)
        self.assertEqual(V3_PROTOCOL.episode_schema_version, "icgs_episode_v2")
        self.assertEqual(V3_PROTOCOL.composition_protocol_id, "icgs-composition-primary-v2")
        self.assertEqual(V3_PROTOCOL.controller_protocol_id, "rlbench-timed-ik-v2")
        self.assertEqual(V3_PROTOCOL.minimum_execution_modes, 1)
        self.assertEqual(V3_PROTOCOL.execution_modes, ("scripted_waypoint_v1",))
        self.assertEqual(V3_PROTOCOL.phase, "1")
        self.assertEqual(V3_PROTOCOL.train_programs, 20)
        self.assertEqual(V3_PROTOCOL.development_programs, 4)
        self.assertEqual(V3_PROTOCOL.test_programs, 12)
        self.assertEqual(V3_PROTOCOL.pilot_program_ids, PILOT_IDS)
        self.assertEqual(V3_PROTOCOL.views, ("D_geom", "D_temporal", "D_dyn", "D_task"))
        self.assertEqual(V3_PROTOCOL.mixture_measured_on, "training_transitions")
        self.assertEqual(V3_PROTOCOL.mixture_views, ("D_temporal", "D_dyn"))
        self.assertEqual(V3_PROTOCOL.train_on, "train_core")
        self.assertEqual(V3_PROTOCOL.validation, "train_val")
        self.assertEqual(V3_PROTOCOL.train_nominal_successes_per_program, 200)
        self.assertEqual(V3_PROTOCOL.train_perturbed_attempts_per_program, 80)
        self.assertEqual(V3_PROTOCOL.eval_perturbed_attempts_per_program, 20)
        self.assertEqual(len(V3_PROTOCOL.physics_passed_program_ids), 36)
        self.assertNotIn("D_recovery", V3_PROTOCOL.views)
        self.assertNotIn("D_value", V3_PROTOCOL.views)
        self.assertEqual(len(V3_PROTOCOL.perturbation_kinds), 5)

    def test_primary_v2_profile_and_manifest_are_untouched(self):
        profile = json.loads(V2_PROFILE.read_text(encoding="utf-8"))
        self.assertEqual(profile["config"]["dataset"]["schema_id"], "icgs_episode_v1")
        self.assertEqual(profile["config"]["dataset"]["minimum_execution_modes"], 2)
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["protocol_id"], "icgs-composition-primary-v1")
        self.assertEqual(manifest["manifest_version"], 1)

    def test_v3_profile_does_not_replace_canonical_defaults(self):
        payload = json.loads(V3_PROFILE.read_text(encoding="utf-8"))
        self.assertEqual(payload["dataset_version"], "icgs-primary-v3")
        self.assertEqual(payload["minimum_execution_modes"], 1)
        self.assertEqual(payload["mixture_measured_on"], "training_transitions")
        self.assertEqual(payload["train_on"], "train_core")
        self.assertEqual(payload["views"], ["D_geom", "D_temporal", "D_dyn", "D_task"])
        canonical = json.loads(V2_PROFILE.read_text(encoding="utf-8"))
        self.assertNotEqual(canonical["config"]["dataset"]["schema_id"], "icgs_episode_v2")


class StructuredCatalogTests(unittest.TestCase):
    def test_structured_steps_match_catalog_text_for_every_primary_program(self):
        primary = [spec for spec in program_catalog().values() if spec.split in {"train", "development", "test"}]
        self.assertEqual(len(primary), 36)
        for spec in primary:
            steps = structured_steps(spec.program_id)
            self.assertEqual(tuple(step.text for step in steps), spec.steps)
            self.assertGreaterEqual(len(steps), 2)
            for index, step in enumerate(steps, 1):
                self.assertEqual(step.step_id, f"s{index}")
                self.assertTrue(step.primitive)
                self.assertTrue(step.success_predicate)
                self.assertNotEqual(step.kind, "recovery")

    def test_family_fixes_do_not_claim_restore_when_catalog_has_none(self):
        self.assertEqual(PROGRAM_FAMILIES["T13"], "park-retrieve")
        self.assertEqual(PROGRAM_FAMILIES["T15"], "park-retrieve")
        self.assertEqual(PROGRAM_FAMILIES["T20"], "park-retrieve")
        self.assertEqual(PROGRAM_FAMILIES["T14"], "park-restore")
        self.assertNotEqual(structured_steps("T13")[-1].primitive, "restore")
        self.assertEqual(structured_steps("T14")[-1].primitive, "restore")

    def test_required_train_families_are_present(self):
        train_families = {PROGRAM_FAMILIES[pid] for pid in program_catalog() if get_program(pid).split == "train"}
        self.assertTrue(set(V3_PROTOCOL.required_train_families) <= train_families)


class CompilerParityTests(unittest.TestCase):
    def test_compiled_events_equal_structured_steps(self):
        compiled = compile_v3_catalog()
        self.assertEqual(len(compiled), 36)
        for pid, spec in compiled.items():
            events = events_from_steps(structured_steps(pid))
            self.assertEqual(spec.events, events)
            self.assertEqual(len(spec.events), len(get_program(pid).steps))
            self.assertEqual(
                [event["primitive"] for event in spec.events],
                [step.primitive for step in structured_steps(pid)],
            )
            self.assertTrue(spec.compiler_routine_id.startswith("stepparse-v3/"))

    def test_mismatch_fixes(self):
        compiled = compile_v3_catalog()
        self.assertEqual(compiled["T03"].events[0]["primitive"], "push")
        self.assertEqual(compiled["T03"].routine[0]["type"], "push")
        self.assertEqual(compiled["T04"].events[0]["articulation"], "open")
        self.assertEqual(compiled["T04"].events[1]["articulation"], "closed")
        self.assertEqual(compiled["T04"].routine[0]["type"], "open_articulation")
        self.assertEqual(compiled["T04"].routine[1]["type"], "close_articulation")
        for pid in ("T06", "T08", "T19"):
            self.assertEqual(compiled[pid].events[-1]["primitive"], "close")
            self.assertEqual(compiled[pid].events[-1]["success_predicate"], "drawer_closed")
            self.assertIn("drawer_handle", compiled[pid].objects)
            self.assertEqual(compiled[pid].routine[-1]["type"], "close_articulation")
        self.assertEqual(compiled["T10"].events[0]["primitive"], "push_through_aperture")
        self.assertEqual(compiled["T10"].routine[0]["type"], "push")
        self.assertNotEqual(compiled["T10"].routine[0]["type"], "transport_through_aperture")
        for pid in ("T12", "G3"):
            self.assertEqual(compiled[pid].events[0]["articulation"], "open")
            self.assertEqual(compiled[pid].events[-1]["articulation"], "closed")
            self.assertEqual(compiled[pid].events[-1]["success_predicate"], "gate_closed")
        self.assertEqual(compiled["T17"].events[-1]["kind"], "nominal")
        self.assertEqual(compiled["T17"].events[-1]["target_role"], "target_a")
        self.assertEqual(compiled["G4"].events[2]["primitive"], "regrasp")
        self.assertEqual(compiled["G4"].events[2]["kind"], "nominal")
        self.assertEqual(len(compiled["G4"].events), 5)
        self.assertIsNotNone(compiled["T18"].future_dependency)
        self.assertIsNotNone(compiled["P4"].future_dependency)
        self.assertNotIn("touch_retreat", [step["type"] for step in compiled["T14"].routine])

    def test_deferred_and_non_train_programs_are_not_training_eligible(self):
        eligible = training_eligible_ids(compile_v3_catalog())
        self.assertTrue(set(eligible) <= set(V3_PROTOCOL.physics_passed_program_ids))
        self.assertEqual(len(eligible), 20)
        self.assertTrue(set(eligible) == {f"T{i:02d}" for i in range(1, 21)})
        self.assertNotIn("E01", eligible)
        self.assertNotIn("V01", eligible)
        self.assertNotIn("P1", eligible)
        self.assertNotIn("G1", eligible)


class DiversityTests(unittest.TestCase):
    def test_episode_seeds_are_independent_per_index(self):
        seeds = [episode_seed("T01", 7, index) for index in range(20)]
        self.assertEqual(len(set(seeds)), 20)
        self.assertEqual(episode_seed("T01", 7, 0), episode_seed("T01", 7, 0))
        self.assertNotEqual(episode_seed("T01", 7, 0), episode_seed("T02", 7, 0))
        self.assertNotEqual(
            episode_seed("T01", 7, 0, dataset_version="icgs-primary-v3"),
            episode_seed("T01", 7, 0, dataset_version="other"),
        )

    def test_stratified_sampling_covers_position_yaw_scale_bins(self):
        bounds = {
            "translation_m": {"x": (-0.012, 0.012), "y": (-0.012, 0.012), "z": (0.0, 0.0)},
            "yaw_deg": (-30.0, 30.0),
            "scale": (0.8, 1.2),
            "camera_profile_id": "rlbench-wrist-depth-v1",
        }
        samples = [sample_randomization("T01", 11, index, bounds) for index in range(36)]
        self.assertEqual({item["stratum"]["position"] for item in samples}, {0, 1, 2, 3})
        self.assertEqual({item["stratum"]["yaw"] for item in samples}, {0, 1, 2})
        self.assertEqual({item["stratum"]["scale"] for item in samples}, {0, 1, 2})
        item = samples[0]
        self.assertIn("scene_seed", item)
        self.assertIn("object_translation_m", item)
        self.assertIn("object_yaw_deg", item)
        self.assertIn("object_scale_applied", item)
        self.assertIn("asset_instance_id", item)
        self.assertIn("inter_object_gap_m", item)
        self.assertIn("approach_waypoint", item)
        self.assertIn("release_waypoint", item)
        self.assertIn("lighting_applied", item)
        self.assertIn("camera_viewpoint_applied", item)
        self.assertIn(item["train_subset"], {"train_core", "train_val"})

    def test_duplicate_signature_and_seed_is_rejected(self):
        sample = {
            "asset_family_id": "icgs-train-basic-manipulation-v1",
            "scene_seed": 184920,
            "object_translation_m": {"x": 0.027, "y": -0.018, "z": 0.0},
            "object_yaw_deg": 43.2,
            "scale": 1.08,
            "camera_profile_id": "rlbench-wrist-depth-v1",
        }
        signature = scene_signature(sample)
        seen = {signature: {184920}}
        self.assertTrue(is_duplicate_episode(sample, seen))
        same_pose = dict(sample)
        same_pose["scene_seed"] = 184921
        self.assertTrue(is_duplicate_episode(same_pose, seen))
        sample = dict(sample)
        sample["scene_seed"] = 184921
        sample["object_translation_m"] = {"x": 0.12, "y": 0.12, "z": 0.0}
        self.assertFalse(is_duplicate_episode(sample, seen, previous=[{
            "asset_family_id": "icgs-train-basic-manipulation-v1",
            "camera_profile_id": "rlbench-wrist-depth-v1",
            "object_translation_m": {"x": 0.0, "y": 0.0, "z": 0.0},
        }]))


class PerturbationTests(unittest.TestCase):
    def test_five_intervention_kinds(self):
        routine = [{"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03}]
        objects = {"target_a": {"pos": [0.25, 0.15, 0.775]}, "object_a": {"pos": [0.25, -0.10, 0.775]}}
        offset = apply_perturbation("action_pose_offset", {"dx_m": 0.012, "dy_m": -0.008}, objects, routine)
        self.assertAlmostEqual(offset["objects"]["target_a"]["pos"][0], 0.262)
        timing = apply_perturbation("gripper_timing", {"delta_intervals": -1}, objects, routine)
        self.assertEqual(timing["routine"][0]["grip_timing_delta_intervals"], -1)
        displaced = apply_perturbation("object_displacement", {"dx_m": 0.02, "dy_m": 0.0}, objects, routine)
        self.assertTrue(displaced["intervention"]["external_intervention"])
        self.assertTrue(displaced["intervention"]["episode_has_external_intervention"])
        self.assertEqual(displaced["intervention"]["application_scope"], "initial_scene")
        self.assertIsNone(displaced["intervention"]["application_t"])
        blocker = apply_perturbation("blocker_insertion", {"pos": [0.3, 0.0, 0.775]}, objects, routine)
        self.assertIn("inserted_blocker", blocker["objects"])
        pause = apply_perturbation("pause_hold", {"intervals": 4}, objects, routine)
        self.assertEqual(pause["routine"][0]["type"], "pause_hold")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            apply_perturbation("target_offset", {}, objects, routine)

    def test_blocker_quota_redistributes_when_not_applicable(self):
        quota = perturbation_quota("T01")
        self.assertEqual(quota["blocker_insertion"], "not_applicable")
        self.assertEqual(sum(value for value in quota.values() if isinstance(value, int)), 80)
        self.assertIsInstance(perturbation_quota("T03")["blocker_insertion"], int)

    def test_simulator_crash_is_not_a_physical_failure(self):
        self.assertEqual(classify_attempt_outcome(simulator_crash=True, predicates_ok=False), SIMULATOR_CRASH)
        self.assertEqual(classify_attempt_outcome(simulator_crash=False, predicates_ok=False), VALID_FAILURE)
        self.assertEqual(classify_attempt_outcome(simulator_crash=False, predicates_ok=True), SUCCESS)


class CameraAndSchemaTests(unittest.TestCase):
    def test_camera_profile_declares_units_and_frames(self):
        profile = get_camera_profile("rlbench-wrist-depth-v1")
        self.assertEqual(profile["depth_unit"], "meter")
        self.assertEqual(profile["pointcloud_frame"], "world")
        self.assertTrue(profile["rgb_depth_mask_synchronized"])
        self.assertEqual(profile["action_space_id"], "ee_pose_xyzw_grip_v1")
        self.assertEqual(profile["orientation_convention"], "xyzw")
        self.assertEqual(profile["gripper_unit"], "open_1_closed_0")
        self.assertIn("camera_intrinsics_id", profile)
        self.assertIn("intrinsics", profile)
        self.assertIn("distortion", profile)
        self.assertIn("extrinsics", profile)
        self.assertIn("rlbench-wrist-depth-v1", CAMERA_PROFILES)

    def test_episode_v2_requires_t_plus_one_observations_and_dt(self):
        record = _episode_v2_record(n_actions=3)
        validate_episode_v2(record)
        record["online_observations"].pop()
        with self.assertRaisesRegex(ValueError, "T\\+1|observations"):
            validate_episode_v2(record)

    def test_episode_v2_rejects_v1_schema_id(self):
        record = _episode_v2_record(n_actions=2)
        record["schema_version"] = "icgs_episode_v1"
        with self.assertRaisesRegex(ValueError, "icgs_episode_v2"):
            validate_episode_v2(record)


class ViewTests(unittest.TestCase):
    def test_only_phase1_views_are_built(self):
        self.assertEqual(SUPPORTED_V3_VIEWS, ("D_geom", "D_temporal", "D_dyn", "D_task"))
        records = [_episode_v2_record(n_actions=4, episode_id="ep_a")]
        geom = build_v3_view(records, "D_geom")
        temporal = build_v3_view(records, "D_temporal")
        dyn = build_v3_view(records, "D_dyn")
        task = build_v3_view(records, "D_task")
        self.assertEqual(len(geom), 5)
        self.assertEqual(len(temporal), 4)
        self.assertEqual(temporal[0]["dt"], 0.1)
        self.assertEqual(dyn[1]["history"], [0])
        self.assertEqual(dyn[1]["episode_kind"], "nominal")
        self.assertFalse(dyn[1]["external_intervention"])
        self.assertIsNone(dyn[1]["intervention_id"])
        self.assertEqual(len(task), 5)
        for name in ("D_value", "D_recovery", "D_pair"):
            with self.assertRaisesRegex(ValueError, "unsupported"):
                build_v3_view(records, name)

    def test_dyn_marks_external_intervention_only_at_timestep_scope(self):
        record = _episode_v2_record(n_actions=4, episode_id="ep_pert")
        record["provenance"]["episode_kind"] = "perturbed"
        record["provenance"]["intervention_id"] = "object_displacement_v1"
        record["provenance"]["intervention_params"] = {"dx_m": 0.02}
        record["provenance"]["application_scope"] = "initial_scene"
        record["provenance"]["application_t"] = None
        record["provenance"]["external_intervention"] = True
        record["intervention"] = {
            "intervention_id": "object_displacement_v1",
            "application_scope": "initial_scene",
            "application_t": None,
            "external_intervention": True,
        }
        dyn = build_v3_view([record], "D_dyn", mix=False)
        self.assertTrue(all(not item["transition_has_external_intervention"] for item in dyn))
        self.assertTrue(all(item["episode_has_external_intervention"] for item in dyn))
        self.assertTrue(all(item["initial_scene_intervention_id"] == "object_displacement_v1" for item in dyn))
        self.assertTrue(all(not item["external_intervention"] for item in dyn))
        record["intervention"]["application_scope"] = "timestep"
        record["intervention"]["application_t"] = 1
        dyn = build_v3_view([record], "D_dyn", mix=False)
        self.assertFalse(dyn[0]["transition_has_external_intervention"])
        self.assertTrue(dyn[1]["transition_has_external_intervention"])
        self.assertIsNone(dyn[1]["initial_scene_intervention_id"])
        self.assertEqual(dyn[1]["intervention_id"], "object_displacement_v1")

    def test_training_views_exclude_eval_and_train_val(self):
        eval_record = _episode_v2_record(n_actions=4, episode_id="ep_eval")
        eval_record["provenance"]["split"] = "dev"
        eval_record["provenance"]["episode_kind"] = "perturbed"
        self.assertEqual(build_v3_view([eval_record], "D_geom"), [])
        self.assertEqual(len(build_v3_view([eval_record], "D_geom", role="evaluation")), 5)
        val_record = _episode_v2_record(n_actions=4, episode_id="ep_val")
        val_record["provenance"]["subset"] = "train_val"
        self.assertEqual(build_v3_view([val_record], "D_geom"), [])
        self.assertEqual(len(build_v3_view([val_record], "D_geom", role="validation")), 5)

    def test_temporal_dyn_mixture_targets_70_30(self):
        from icgs.data.collection.v3.sampler import mix_training_transitions

        samples = [{"episode_id": "n", "t": i, "episode_kind": "nominal"} for i in range(70)]
        samples.extend({"episode_id": "p", "t": i, "episode_kind": "perturbed"} for i in range(30))
        mixed = mix_training_transitions(samples, view="D_dyn")
        pert = sum(1 for item in mixed if item["episode_kind"] == "perturbed")
        self.assertEqual(pert / len(mixed), 0.3)
        geom = mix_training_transitions(samples, view="D_geom")
        self.assertEqual(len(geom), 100)


class ReportAndLayoutTests(unittest.TestCase):
    def test_parity_report_lists_pilot_programs(self):
        compiled = compile_v3_catalog()
        report = program_parity_report(compiled)
        self.assertTrue(report["all_primary_match"])
        for pid in PILOT_IDS:
            self.assertTrue(report["programs"][pid]["match"])
        compiled = compile_v3_catalog()
        record = _episode_v2_record(n_actions=2, episode_id="ep_sem")
        record["events"] = list(compiled["T01"].events)
        observed = observed_semantics_report({"T01": compiled["T01"]}, [record])
        self.assertTrue(observed["programs"]["T01"]["match"])
        self.assertTrue(observed["all_observed_match"])

    def test_v3_manifest_preserves_v2_and_adds_phase1_fields(self):
        manifest = build_v3_manifest(V2_MANIFEST)
        self.assertEqual(manifest["composition_protocol_id"], "icgs-composition-primary-v2")
        self.assertEqual(manifest["manifest_version"], 3)
        self.assertEqual(len(manifest["catalog"]), 36)
        t01 = next(row for row in manifest["catalog"] if row["program_id"] == "T01")
        self.assertEqual(t01["structured_steps"][0]["primitive"], "grasp")
        self.assertIn("training_eligible", t01)
        v2 = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(v2["protocol_id"], "icgs-composition-primary-v1")

    def test_phase1_layout_has_views_not_branches(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = initialize_v3_dataset_layout(directory)
            self.assertTrue((root / "views" / "D_geom.jsonl").is_file())
            self.assertTrue((root / "views" / "D_task.jsonl").is_file())
            self.assertFalse((root / "anchors").exists())
            self.assertFalse((root / "branches").exists())
            self.assertFalse((root / "views" / "D_recovery.jsonl").exists())


def _episode_v2_record(*, n_actions: int, episode_id: str = "ep_test") -> dict:
    import numpy as np

    observations = []
    for index in range(n_actions + 1):
        observations.append({
            "points": np.ones((4, 3), dtype=np.float32) * (index + 1),
            "point_valid": np.ones(4, dtype=bool),
            "T_w_e": np.eye(4, dtype=np.float64),
            "grip": 0,
        })
    transitions = []
    for index in range(n_actions):
        transitions.append({
            "command": {"T_w_e": np.eye(4), "grip": 0, "duration_s": 0.1},
            "achieved_duration_s": 0.1,
            "physics_substeps": 1,
            "before_boundary": index,
            "after_boundary": index + 1,
        })
    return {
        "schema_version": "icgs_episode_v2",
        "provenance": {
            "episode_id": episode_id,
            "program_id": "T01",
            "split": "train",
            "scene_seed": 184920,
            "asset_instance_id": "t01-asset-184920",
            "execution_mode": "scripted_waypoint_v1",
            "calibration_id": "rlbench-wrist-depth-v1",
            "observation_origin": "measured",
            "raw_commands_id": "raw-1",
            "materialized_commands_id": "mat-1",
            "episode_kind": "nominal",
            "outcome": "success",
            "subset": "train_core",
            "source_lineage_id": "t01-seed-root-v1",
            "asset_family_id": "icgs-train-basic-manipulation-v1",
            "dataset_version": "icgs-primary-v3",
            "camera_profile_id": "rlbench-wrist-depth-v1",
            "controller_version": "rlbench-timed-ik-v2",
            "program_manifest_version": 3,
        },
        "online_observations": observations,
        "transitions": transitions,
        "dt": [0.1] * n_actions,
        "robot_states": [{} for _ in range(n_actions + 1)],
        "object_states": [{} for _ in range(n_actions + 1)],
    }


if __name__ == "__main__":
    unittest.main()
