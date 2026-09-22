"""Phase-1 collection batch plans: seeds, quota, provenance."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from icgs.data.collection.generation.attempt_prep import plan_full_generation, prepare_attempt
from icgs.data.collection.generation.batch import (
    AttemptPlan,
    AttemptPlanner,
    apply_layout_randomization,
    bounds_from_row,
    plan_pilot_batch,
    plan_program_attempts,
    provenance_from_plan,
    sample_intervention,
)
from icgs.data.collection.generation.quota import (
    QuotaCounts,
    next_episode_kind,
    quota_for_program,
    quota_met,
    record_outcome,
    valid_attempt_mixture,
)
from icgs.data.collection.generation.report import generation_parity_gate, split_disjointness_report
from icgs.data.collection.generation.compiler import compile_generation_catalog
from icgs.data.collection.generation.episode_record import assemble_episode, classify_generation_outcome, quarantine_sidecar
from icgs.data.collection.generation.manifest import build_manifest
from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL
from icgs.data.schemas.episode_records import validate_episode


APPROVED_MANIFEST = Path("artifacts/composition/approved_composition_manifest.json")


class BatchPlanTests(unittest.TestCase):
    def setUp(self):
        self.manifest = build_manifest(APPROVED_MANIFEST)
        self.rows = {row["program_id"]: row for row in self.manifest["catalog"]}

    def test_pilot_batch_covers_family_programs_with_nominal_and_perturbed(self):
        plans = plan_pilot_batch(self.rows)
        self.assertEqual(len(plans), 11 * 4)
        by_program = {}
        for plan in plans:
            by_program.setdefault(plan.program_id, []).append(plan)
        self.assertEqual(set(by_program), set(GENERATION_PROTOCOL.pilot_program_ids))
        for program_id, group in by_program.items():
            kinds = [item.episode_kind for item in group]
            self.assertEqual(kinds.count("nominal"), 2)
            self.assertEqual(kinds.count("perturbed"), 2)
            nominal = [item for item in group if item.episode_kind == "nominal"]
            perturbed = [item for item in group if item.episode_kind == "perturbed"]
            self.assertEqual(len({item.scene_seed for item in nominal}), 2)
            self.assertTrue(all(item.intervention["source_episode_id"] in {n.episode_id for n in nominal} for item in perturbed))

    def test_blocker_not_applied_to_t01(self):
        row = self.rows["T01"]
        plans = plan_program_attempts(
            "T01",
            n_nominal=0,
            n_perturbed=8,
            bounds=bounds_from_row(row),
            asset_family_id=row["asset_family_id"],
        )
        kinds = {plan.intervention["kind"] for plan in plans}
        self.assertNotIn("blocker_insertion", kinds)
        self.assertIn("action_pose_offset", kinds)
        self.assertIn("pause_hold", kinds)

    def test_blocker_applied_to_t03(self):
        row = self.rows["T03"]
        plans = plan_program_attempts(
            "T03",
            n_nominal=0,
            n_perturbed=10,
            bounds=bounds_from_row(row),
            asset_family_id=row["asset_family_id"],
        )
        kinds = {plan.intervention["kind"] for plan in plans}
        self.assertIn("blocker_insertion", kinds)

    def test_provenance_validates_as_episode_v2(self):
        import numpy as np

        plans = plan_program_attempts(
            "T01",
            n_nominal=1,
            n_perturbed=1,
            bounds=bounds_from_row(self.rows["T01"]),
            asset_family_id=self.rows["T01"]["asset_family_id"],
        )
        for plan in plans:
            provenance = provenance_from_plan(plan, binding=self.rows["T01"])
            provenance["outcome"] = "success"
            observations = [{
                "points": np.ones((3, 3), dtype=np.float32),
                "point_valid": np.ones(3, dtype=bool),
                "T_w_e": np.eye(4),
                "grip": 0,
            } for _ in range(3)]
            record = {
                "schema_version": "icgs_episode_v2",
                "provenance": provenance,
                "online_observations": observations,
                "transitions": [
                    {"command": {"T_w_e": np.eye(4), "grip": 0, "duration_s": 0.1}, "achieved_duration_s": 0.1},
                    {"command": {"T_w_e": np.eye(4), "grip": 0, "duration_s": 0.1}, "achieved_duration_s": 0.1},
                ],
                "dt": [0.1, 0.1],
            }
            validate_episode(record)

    def test_layout_randomization_shifts_positions(self):
        objects = {"object_a": {"pos": [0.25, 0.0, 0.775], "size": [0.04, 0.04, 0.04]}}
        shifted = apply_layout_randomization(objects, {"object_translation_m": {"x": 0.01, "y": -0.02, "z": 0.0}})
        self.assertAlmostEqual(shifted["object_a"]["pos"][0], 0.26)
        self.assertAlmostEqual(shifted["object_a"]["pos"][1], -0.02)

    def test_layout_randomization_changes_inter_object_gap_and_scale(self):
        objects = {
            "object_a": {"pos": [0.25, -0.10, 0.775], "size": [0.04, 0.04, 0.04]},
            "target_a": {"pos": [0.25, 0.15, 0.775], "size": [0.02, 0.02, 0.001]},
            "pad": {"pos": [0.25, 0.15, 0.755], "size": [0.08, 0.08, 0.025]},
        }
        shifted = apply_layout_randomization(objects, {
            "object_translation_m": {"x": 0.0, "y": 0.0, "z": 0.0},
            "inter_object_gap_m": {"dx_m": 0.01, "dy_m": 0.0},
            "object_yaw_deg": 12.0,
            "object_scale_applied": 1.1,
        })
        self.assertAlmostEqual(shifted["object_a"]["pos"][0], 0.26)
        self.assertAlmostEqual(shifted["target_a"]["pos"][0], 0.25)
        self.assertAlmostEqual(shifted["object_a"]["size"][0], 0.044)
        self.assertAlmostEqual(shifted["object_a"]["yaw_deg"], 12.0)

    def test_assemble_episode_from_attempt_plan(self):
        import numpy as np

        plans = plan_program_attempts(
            "T01",
            n_nominal=1,
            n_perturbed=1,
            bounds=bounds_from_row(self.rows["T01"]),
            asset_family_id=self.rows["T01"]["asset_family_id"],
        )
        observations = [{
            "points": np.ones((3, 3), dtype=np.float32),
            "point_valid": np.ones(3, dtype=bool),
            "T_w_e": np.eye(4),
            "grip": 0,
        } for _ in range(3)]
        transitions = [
            {"command": {"T_w_e": np.eye(4), "grip": 0, "duration_s": 0.1}, "achieved_duration_s": 0.1},
            {"command": {"T_w_e": np.eye(4), "grip": 0, "duration_s": 0.1}, "achieved_duration_s": 0.1},
        ]
        record = assemble_episode(
            plan=plans[0],
            binding=self.rows["T01"],
            observations=observations,
            transitions=transitions,
            result_class="success",
        )
        self.assertEqual(record["schema_version"], "icgs_episode_v2")
        self.assertEqual(record["provenance"]["episode_kind"], "nominal")
        self.assertEqual(record["provenance"]["outcome"], "success")
        self.assertIsNone(record["provenance"]["failure_type"])
        self.assertIn(record["provenance"]["subset"], {"train_core", "train_val"})
        self.assertEqual(record["provenance"]["execution_source"], "scripted_waypoint_v1")
        self.assertIsNone(record["provenance"]["policy_id"])
        self.assertIsNone(record["provenance"]["recovery_id"])
        self.assertIsNone(record["provenance"]["snapshot_fidelity"])
        self.assertIn("sim_time", record["snapshot"])
        self.assertFalse(record["provenance"]["episode_has_external_intervention"])
        self.assertTrue(record["provenance"]["terminated"])
        self.assertFalse(record["provenance"]["truncated"])
        self.assertIsNone(record["provenance"]["initial_scene_intervention_id"])
        for key in (
            "dataset_version",
            "program_id",
            "program_semantics_version",
            "split",
            "subset",
            "scene_signature",
            "scene_seed",
            "asset_family_id",
            "asset_instance_id",
            "episode_kind",
            "execution_mode",
            "controller_version",
            "simulator_version",
            "calibration_id",
            "action_space_id",
            "outcome",
            "valid_observation_until",
            "terminated",
            "truncated",
            "policy_version",
            "candidate_index",
            "trial_seed",
            "recovery_id",
            "recovery_budget",
            "recovery_outcome",
            "snapshot_fidelity",
            "restore_validation_result",
        ):
            self.assertIn(key, record["provenance"], key)
        self.assertEqual(len(record["dt"]), 2)
        self.assertEqual(len(record["robot_states"]), 3)
        crash = classify_generation_outcome(simulator_crash=True, predicates_ok=False)
        fail = classify_generation_outcome(simulator_crash=False, predicates_ok=False)
        self.assertEqual(crash, "simulator_crash")
        self.assertEqual(fail, "valid_failure")
        sidecar = quarantine_sidecar(episode_id="x", program_id="T02", outcome=crash, error="boom")
        self.assertEqual(sidecar["outcome"], "simulator_crash")
        self.assertIsNone(sidecar["episode_id"])
        self.assertTrue(sidecar["attempt_id"])

    def test_full_generation_includes_train_perturbations_not_eval(self):
        subset = {pid: self.rows[pid] for pid in ("T01", "T03", "V01", "P1")}
        plans = plan_full_generation(subset)
        t01 = [p for p in plans if p.program_id == "T01"]
        self.assertEqual(len([p for p in t01 if p.episode_kind == "nominal"]), 200)
        self.assertEqual(len([p for p in t01 if p.episode_kind == "perturbed"]), 80)
        kinds = {p.intervention["kind"] for p in t01 if p.intervention}
        self.assertNotIn("blocker_insertion", kinds)
        t03 = [p for p in plans if p.program_id == "T03" and p.episode_kind == "perturbed"]
        self.assertTrue(any(p.intervention and p.intervention["kind"] == "blocker_insertion" for p in t03))
        v01 = [p for p in plans if p.program_id == "V01"]
        self.assertEqual(len([p for p in v01 if p.episode_kind == "nominal"]), 100)
        self.assertEqual(len([p for p in v01 if p.episode_kind == "perturbed"]), 20)
        self.assertTrue(all(p.intervention["held_out"] for p in v01 if p.intervention))
        p1 = [p for p in plans if p.program_id == "P1"]
        self.assertEqual(len([p for p in p1 if p.episode_kind == "nominal"]), 100)
        self.assertEqual(len([p for p in p1 if p.episode_kind == "perturbed"]), 20)
        perturbed = next(p for p in t01 if p.intervention)
        self.assertEqual(perturbed.intervention["application_scope"], "event")
        self.assertFalse(perturbed.intervention["held_out"])
        self.assertEqual(perturbed.intervention["magnitude_bucket"], "inner")
        self.assertIsNotNone(perturbed.intervention["source_episode_id"])
        self.assertIn(perturbed.intervention["source_episode_id"], {p.episode_id for p in t01 if p.episode_kind == "nominal"})
        self.assertEqual(perturbed.scene_seed, next(p.scene_seed for p in t01 if p.episode_id == perturbed.intervention["source_episode_id"]))
        self.assertIn("external_intervention", perturbed.intervention)
        self.assertTrue(all(p.randomization["stratum"]["position"] < 3 for p in t01))
        self.assertTrue(all(p.randomization["stratum"]["position"] == 3 for p in v01))
        self.assertTrue(all(p.intervention["magnitude_bucket"] == "outer" for p in v01 if p.intervention))
        records = []
        for plan in plans:
            records.append({
                "provenance": provenance_from_plan(plan, binding=self.rows[plan.program_id]),
                "randomization": plan.randomization,
            })
        leakage = split_disjointness_report(records)
        self.assertTrue(leakage["disjoint"], leakage)
        self.assertEqual(leakage["source_episode_leaks"], [])

    def test_inner_and_outer_magnitude_buckets_do_not_overlap(self):
        bound = GENERATION_PROTOCOL.target_offset_m
        for index in range(40):
            inner = sample_intervention("T01", 20260920, index, "action_pose_offset", held_out=False)
            outer = sample_intervention("V01", 20260920, index, "action_pose_offset", held_out=True)
            inner_r = abs(inner["intervention_params"]["dx_m"]) / bound
            outer_r = abs(outer["intervention_params"]["dx_m"]) / bound
            self.assertLess(inner_r, GENERATION_PROTOCOL.perturbation_inner_fraction)
            self.assertGreaterEqual(outer_r, GENERATION_PROTOCOL.perturbation_inner_fraction)
            self.assertLessEqual(outer_r, 1.0)

    def test_streamed_retries_reuse_pool_uniqueness_index(self):
        subset = {"T01": self.rows["T01"]}
        pool, planners = plan_full_generation(subset, return_planners=True)
        planner = planners["T01"]
        signatures = {item.randomization["scene_signature"] for item in pool}
        extra = [planner.next_plan("nominal") for _ in range(5)]
        extra_sigs = {item.randomization["scene_signature"] for item in extra}
        self.assertTrue(signatures.isdisjoint(extra_sigs))
        fresh = AttemptPlanner(
            "T01",
            bounds=bounds_from_row(self.rows["T01"]),
            asset_family_id=self.rows["T01"]["asset_family_id"],
            n_perturbed=80,
        )
        fresh.ingest_plans(pool)
        resumed = fresh.next_plan("nominal")
        self.assertNotIn(resumed.randomization["scene_signature"], signatures)

    def test_parity_gate_allows_authorized_catalog(self):
        compiled = compile_generation_catalog()
        gate = generation_parity_gate(compiled)
        self.assertTrue(gate["ok"])
        self.assertEqual(gate["blocked"], [])

    def test_quota_runs_nominal_until_successes_then_perturbed_attempts(self):
        quota = quota_for_program("T01")
        self.assertEqual(quota.nominal_success_target, 200)
        self.assertEqual(quota.perturbed_attempt_target, 80)
        counts = QuotaCounts()
        self.assertEqual(next_episode_kind(quota, counts), "nominal")
        for _ in range(200):
            record_outcome(counts, "nominal", "success")
        self.assertTrue(counts.nominal_successes >= quota.nominal_success_target)
        self.assertEqual(next_episode_kind(quota, counts), "perturbed")
        for _ in range(50):
            record_outcome(counts, "perturbed", "valid_failure")
        for _ in range(30):
            record_outcome(counts, "perturbed", "success")
        self.assertTrue(quota_met(quota, counts))
        self.assertIsNone(next_episode_kind(quota, counts))
        mixture = valid_attempt_mixture(counts)
        self.assertEqual(mixture["measured_on"], "collection_valid_attempts")
        self.assertEqual(mixture["training_mixture_measured_on"], "training_transitions")
        self.assertEqual(mixture["nominal"] + mixture["perturbed"], 280)

    def test_quota_does_not_count_crash_as_perturbed_attempt(self):
        quota = quota_for_program("T01")
        counts = QuotaCounts()
        for _ in range(200):
            record_outcome(counts, "nominal", "success")
        record_outcome(counts, "perturbed", "simulator_crash")
        self.assertEqual(counts.perturbed_valid_attempts, 0)
        self.assertEqual(next_episode_kind(quota, counts), "perturbed")
        self.assertFalse(quota_met(quota, counts))

    def test_attempt_planner_streams_past_initial_pool(self):
        planner = AttemptPlanner(
            "T01",
            bounds=bounds_from_row(self.rows["T01"]),
            asset_family_id=self.rows["T01"]["asset_family_id"],
            n_perturbed=4,
        )
        first = [planner.next_plan("nominal") for _ in range(3)]
        extra = planner.next_plan("nominal")
        self.assertEqual(len({item.scene_seed for item in first + [extra]}), 4)
        perturbed = planner.next_plan("perturbed")
        self.assertEqual(perturbed.episode_kind, "perturbed")
        self.assertIsNotNone(perturbed.intervention)

    def test_eval_quota_includes_held_out_perturbation(self):
        quota = quota_for_program("V01")
        self.assertEqual(quota.nominal_success_target, 100)
        self.assertEqual(quota.perturbed_attempt_target, 20)
        counts = QuotaCounts()
        for _ in range(100):
            record_outcome(counts, "nominal", "success")
        self.assertEqual(next_episode_kind(quota, counts), "perturbed")
        for _ in range(20):
            record_outcome(counts, "perturbed", "valid_failure")
        self.assertTrue(quota_met(quota, counts))

    def test_prepare_attempt_applies_seed_and_pause_hold(self):
        compiled = compile_generation_catalog()
        spec = compiled["T01"]
        plan = plan_program_attempts(
            "T01",
            n_nominal=0,
            n_perturbed=8,
            bounds=bounds_from_row(self.rows["T01"]),
            asset_family_id=self.rows["T01"]["asset_family_id"],
        )
        pause = next(p for p in plan if p.intervention and p.intervention["kind"] == "pause_hold")
        prepared = prepare_attempt(pause, spec.objects, spec.routine)
        self.assertEqual(prepared["episode_kind"], "perturbed")
        self.assertEqual(prepared["routine"][0]["type"], "pause_hold")
        offset = next(p for p in plan if p.intervention and p.intervention["kind"] == "action_pose_offset")
        prepared_off = prepare_attempt(offset, spec.objects, spec.routine)
        dx = offset.intervention["intervention_params"]["dx_m"]
        base_x = spec.objects["target_a"][0][0] if not isinstance(spec.objects["target_a"], dict) else spec.objects["target_a"]["pos"][0]
        rand_x = offset.randomization["object_translation_m"]["x"]
        self.assertAlmostEqual(prepared_off["objects"]["target_a"]["pos"][0], base_x + rand_x + dx, places=4)

    def test_generation_manifest_records_perturbation_bounds_and_position_bins(self):
        targets = self.manifest["generation_targets"]
        self.assertEqual(targets["eval_perturbed_attempts_per_program"], 20)
        bounds = targets["perturbation_bounds"]
        self.assertEqual(bounds["action_pose_offset"]["translation_m"], 0.02)
        self.assertEqual(bounds["action_pose_offset"]["yaw_deg"], 10.0)
        self.assertEqual(bounds["object_displacement"]["shift_m"], 0.03)
        self.assertEqual(bounds["gripper_timing"]["intervals"], 1)
        self.assertEqual(bounds["pause_hold"]["intervals"], [2, 10])
        self.assertEqual(bounds["magnitude_buckets"]["inner"]["max_exclusive"], 0.7)
        self.assertEqual(bounds["magnitude_buckets"]["outer"]["min_inclusive"], 0.7)
        self.assertEqual(targets["position_bins"]["train"], [0, 1, 2])
        self.assertEqual(targets["position_bins"]["eval"], [3])

    def test_generation_manifest_uses_wrist_depth_camera(self):
        self.assertEqual(self.rows["T01"]["randomization"]["camera_profile_id"], "rlbench-wrist-depth-v1")
        manifest = json.loads(APPROVED_MANIFEST.read_text())
        t01 = next(row for row in manifest["catalog"] if row["program_id"] == "T01")
        self.assertEqual(t01["randomization"]["camera_profile_id"], "rlbench-wrist-depth-v1")
