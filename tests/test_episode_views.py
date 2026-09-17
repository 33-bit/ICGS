import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from icgs.configuration.method import MethodConfig
from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
from icgs.contracts.records import Observation
from icgs.data.archives import write_episode_archive
from icgs.data.datasets.episodes import (
    adapter_a0,
    adapter_a1,
    build_view,
    validate_dataset_manifest,
)


class EpisodeViewsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.config = MethodConfig(
            dataset={"shard_intervals": 4},
            stages={"A1": {"supervised_intervals": 4, "burnin_intervals": 1, "rollout_curriculum": (1, 2)}},
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_episode(
        self,
        episode_id: str,
        *,
        num_transitions: int,
        split: str = "train",
        lineage_id: str = "root-01",
        asset_family_id: str = "family-01",
        program_id: str = "T01",
    ) -> Path:
        timed_obs = []
        online_obs = []
        for i in range(num_transitions + 1):
            pts = np.ones((5, 3), dtype=np.float64) * (i + 1)
            pose = np.eye(4, dtype=np.float64)
            timed_obs.append(
                TimedObservation(Observation(pts, pose, 0.0), i, 1.0 + 0.1 * i, 2.0 + 0.1 * i, "sensor-01")
            )
            online_obs.append({
                "points": pts.astype(np.float32),
                "point_valid": np.ones(5, dtype=bool),
                "T_w_e": pose.copy(),
                "grip": 0,
            })

        transitions = []
        for i in range(num_transitions):
            transitions.append(
                ExecutedTransition(
                    timed_obs[i],
                    timed_obs[i + 1],
                    TimedCommand(timed_obs[i + 1].observation.T_w_e, 0, 0.1),
                    0.1,
                    10,
                    "ok",
                )
            )

        record = {
            "schema_version": "icgs_episode_v1",
            "provenance": {
                "episode_id": episode_id,
                "program_id": program_id,
                "source_lineage_id": lineage_id,
                "asset_family_id": asset_family_id,
                "split": split,
                "calibration_id": "calib-01",
                "observation_origin": "measured",
                "raw_commands_id": "raw-01",
                "materialized_commands_id": "mat-01",
            },
            "online_observations": online_obs,
            "transitions": transitions,
        }
        return write_episode_archive(self.root, record, config=self.config)

    def _make_dataset_manifest(
        self,
        manifest_path: Path,
        *,
        episodes: list[dict],
        lineage: list[dict],
        asset_families: list[dict],
        dataset_track: str | None = None,
    ) -> Path:
        data = {
            "manifest_version": 1,
            "episodes": episodes,
            "lineage": lineage,
            "asset_families": asset_families,
            "provenance": {
                "generator_seed": 12345,
                "config_digest": "sha256-test",
                "code_revision": "git-test",
            },
        }
        if dataset_track is not None:
            data["dataset_track"] = dataset_track
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(data, indent=2))
        return manifest_path

    def test_exploratory_track_is_explicit_and_not_consumed_by_training_views(self):
        manifest = {
            "manifest_version": 1,
            "dataset_track": "exploratory",
            "episodes": [],
            "lineage": [],
            "asset_families": [],
        }
        self.assertEqual(validate_dataset_manifest(manifest)["dataset_track"], "exploratory")
        path = self.root / "exploratory.json"
        self._make_dataset_manifest(path, episodes=[], lineage=[], asset_families=[], dataset_track="exploratory")
        with self.assertRaisesRegex(ValueError, "exploratory|training views"):
            build_view(path, "geom", split="train", config=self.config)

    def test_unknown_dataset_track_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset_track"):
            validate_dataset_manifest({"manifest_version": 1, "dataset_track": "surprise", "episodes": [], "lineage": [], "asset_families": []})

    def test_lineage_cycle_and_dangling_parent_rejected(self):
        # Dangling parent
        lineage_dangling = [
            {"lineage_id": "node-1", "parent_ids": ["missing-root"], "split": "train"},
        ]
        manifest_dangling = {
            "manifest_version": 1,
            "episodes": [],
            "lineage": lineage_dangling,
            "asset_families": [{"asset_family_id": "fam-1", "split": "train"}],
        }
        with self.assertRaisesRegex(ValueError, "dangling parent|unknown parent"):
            validate_dataset_manifest(manifest_dangling)

        # Cycle
        lineage_cycle = [
            {"lineage_id": "node-A", "parent_ids": ["node-B"], "split": "train"},
            {"lineage_id": "node-B", "parent_ids": ["node-A"], "split": "train"},
        ]
        manifest_cycle = {
            "manifest_version": 1,
            "episodes": [],
            "lineage": lineage_cycle,
            "asset_families": [{"asset_family_id": "fam-1", "split": "train"}],
        }
        with self.assertRaisesRegex(ValueError, "cycle detected"):
            validate_dataset_manifest(manifest_cycle)

    def test_cross_split_lineage_and_asset_leakage_rejected(self):
        # Parent train, child dev -> leakage!
        lineage_leak = [
            {"lineage_id": "root-1", "parent_ids": [], "split": "train"},
            {"lineage_id": "child-1", "parent_ids": ["root-1"], "split": "dev"},
        ]
        manifest_leak = {
            "manifest_version": 1,
            "episodes": [],
            "lineage": lineage_leak,
            "asset_families": [{"asset_family_id": "fam-1", "split": "train"}],
        }
        with self.assertRaisesRegex(ValueError, "(?i)cross-split|split disagreement"):
            validate_dataset_manifest(manifest_leak)

        # Duplicate asset family in multiple splits -> leakage!
        manifest_asset_leak = {
            "manifest_version": 1,
            "episodes": [],
            "lineage": [{"lineage_id": "root-1", "parent_ids": [], "split": "train"}],
            "asset_families": [
                {"asset_family_id": "fam-shared", "split": "train"},
                {"asset_family_id": "fam-shared", "split": "test"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "(?i)cross-split|duplicate"):
            validate_dataset_manifest(manifest_asset_leak)

    def test_program_split_mismatch_rejected(self):
        # T01 is a train program in PROGRAM_CATALOG. If assigned to dev in an episode, must reject!
        ep_path = self._create_episode("ep-mismatch", num_transitions=2, split="dev", program_id="T01")
        ep_sha = hashlib.sha256(ep_path.read_bytes()).hexdigest()
        manifest_data = {
            "manifest_version": 1,
            "episodes": [
                {
                    "episode_id": "ep-mismatch",
                    "manifest_path": str(ep_path.relative_to(self.root)),
                    "sha256": ep_sha,
                }
            ],
            "lineage": [{"lineage_id": "root-01", "parent_ids": [], "split": "dev"}],
            "asset_families": [{"asset_family_id": "family-01", "split": "dev"}],
        }
        with self.assertRaisesRegex(ValueError, "(?i)program.*catalog split.*!="):
            validate_dataset_manifest(manifest_data, dataset_root=self.root)

    def test_geom_and_dyn_views_and_short_failure_exclusion(self):
        # ep1 has 5 transitions (6 boundaries, spans 2 shards since shard_intervals=4)
        ep1_path = self._create_episode("ep-long", num_transitions=5, split="train")
        ep1_sha = hashlib.sha256(ep1_path.read_bytes()).hexdigest()

        # ep2 has 2 transitions (3 boundaries, shorter than supervised_intervals=3)
        ep2_path = self._create_episode("ep-short", num_transitions=2, split="train")
        ep2_sha = hashlib.sha256(ep2_path.read_bytes()).hexdigest()

        manifest_file = self.root / "dataset_manifest.json"
        self._make_dataset_manifest(
            manifest_file,
            episodes=[
                {"episode_id": "ep-long", "manifest_path": str(ep1_path.relative_to(self.root)), "sha256": ep1_sha},
                {"episode_id": "ep-short", "manifest_path": str(ep2_path.relative_to(self.root)), "sha256": ep2_sha},
            ],
            lineage=[{"lineage_id": "root-01", "parent_ids": [], "split": "train"}],
            asset_families=[{"asset_family_id": "family-01", "split": "train"}],
        )

        # 1. GEOM view
        geom_view = build_view(manifest_file, "geom", split="train", config=self.config)
        # ep1 has 6 unique boundaries, ep2 has 3 unique boundaries -> total 9 samples
        self.assertEqual(len(geom_view), 9)
        sample0 = geom_view[0]
        self.assertIn("points", sample0)
        self.assertIn("valid", sample0)
        self.assertEqual(sample0["points"].shape, (5, 3))
        self.assertEqual(sample0["valid"].shape, (5,))

        # Test A0 adapter
        a0_batch = adapter_a0(sample0)
        self.assertIn("points", a0_batch)
        self.assertIn("valid", a0_batch)

        # 2. DYN view
        dyn_view = build_view(manifest_file, "dyn", split="train", config=self.config)
        # ep1 has 5 transitions and supervised_intervals=4 -> starting boundaries: 0, 1 (2 windows)
        # ep2 has 2 transitions (< 4) -> 0 windows from ep2!
        self.assertEqual(len(dyn_view), 2)
        self.assertEqual(dyn_view.excluded_windows_count, 1)  # short failure retained in geom, counted as excluded in dyn

        sample_dyn = dyn_view[0]
        self.assertIn("transitions", sample_dyn)
        self.assertEqual(sample_dyn["boundary"], 0)
        self.assertEqual(sample_dyn["supervised_intervals"], 4)

        # Test A1 adapter
        a1_batch = adapter_a1(sample_dyn)
        self.assertIn("transitions", a1_batch)
        self.assertIn("boundary", a1_batch)
        self.assertIn("supervised_intervals", a1_batch)
        self.assertIn("burnin", a1_batch)
        self.assertGreaterEqual(len(a1_batch["transitions"]), a1_batch["boundary"] + a1_batch["supervised_intervals"])

    def test_consumer_boundary_satisfaction(self):
        from icgs.training.method import _step_a0_reconstruction

        ep_path = self._create_episode("ep-seam", num_transitions=4, split="train")
        ep_sha = hashlib.sha256(ep_path.read_bytes()).hexdigest()
        manifest_file = self.root / "dataset_manifest.json"
        self._make_dataset_manifest(
            manifest_file,
            episodes=[{"episode_id": "ep-seam", "manifest_path": str(ep_path.relative_to(self.root)), "sha256": ep_sha}],
            lineage=[{"lineage_id": "root-01", "parent_ids": [], "split": "train"}],
            asset_families=[{"asset_family_id": "family-01", "split": "train"}],
        )

        geom_view = build_view(manifest_file, "geom", split="train", config=self.config)
        a0_batch = adapter_a0(geom_view[0])

        # Test double for geometry encoder/decoder to test consumer handoff
        class DummyOutput:
            def __init__(self, pts, valid):
                self.points_w = pts
                self.point_valid = valid

        class DummyEncoder(torch.nn.Module):
            def forward(self, pts, val):
                return (pts, val)

        class DummyDecoder(torch.nn.Module):
            def forward(self, enc):
                return DummyOutput(enc[0], enc[1])

        class DummyGeom(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = DummyEncoder()
                self.decoder = DummyDecoder()

        loss = _step_a0_reconstruction({"geometry": DummyGeom()}, a0_batch, self.config)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss).item())

    def test_dyn_view_causal_history_target_window_separation_and_no_future_leakage(self):
        # Episode with 10 transitions, supervised_intervals = 4
        ep_path = self._create_episode("ep-leakage-test", num_transitions=10, split="train")
        ep_sha = hashlib.sha256(ep_path.read_bytes()).hexdigest()
        manifest_file = self.root / "dataset_manifest.json"
        self._make_dataset_manifest(
            manifest_file,
            episodes=[{"episode_id": "ep-leakage-test", "manifest_path": str(ep_path.relative_to(self.root)), "sha256": ep_sha}],
            lineage=[{"lineage_id": "root-01", "parent_ids": [], "split": "train"}],
            asset_families=[{"asset_family_id": "family-01", "split": "train"}],
        )

        dyn_view = build_view(manifest_file, "dyn", split="train", config=self.config)
        # 10 transitions, supervised_intervals=4 -> start_b ranges from 0 to 6 (7 samples)
        self.assertEqual(len(dyn_view), 7)

        for sample in dyn_view:
            b = sample["boundary"]
            h = sample["supervised_intervals"]
            history = sample["history_transitions"]
            target = sample["target_transitions"]
            transitions = sample["transitions"]

            # Causal history must have length exactly b
            self.assertEqual(len(history), b)
            # Target window must have length exactly h
            self.assertEqual(len(target), h)
            # Permitted transitions must NEVER exceed boundary + supervised_intervals
            self.assertEqual(len(transitions), b + h)
            self.assertLess(len(transitions), 10 if b + h < 10 else 11)

            # Adapter must also honor separation
            batch = adapter_a1(sample)
            self.assertEqual(len(batch["history_transitions"]), b)
            self.assertEqual(len(batch["target_transitions"]), h)
            self.assertEqual(len(batch["transitions"]), b + h)

    def test_dataset_manifest_path_and_sha_strict_validation(self):
        # Valid episode
        ep_path = self._create_episode("ep-strict-valid", num_transitions=2, split="train")
        ep_sha = hashlib.sha256(ep_path.read_bytes()).hexdigest()
        rel_path = str(ep_path.relative_to(self.root))

        base_manifest = {
            "manifest_version": 1,
            "lineage": [{"lineage_id": "root-01", "parent_ids": [], "split": "train"}],
            "asset_families": [{"asset_family_id": "family-01", "split": "train"}],
        }

        # 1. Missing manifest_path
        m1 = dict(base_manifest, episodes=[{"episode_id": "ep-strict-valid", "sha256": ep_sha}])
        with self.assertRaisesRegex(ValueError, "missing required 'manifest_path'"):
            validate_dataset_manifest(m1, dataset_root=self.root)

        # 2. Absolute path rejected
        m2 = dict(base_manifest, episodes=[{"episode_id": "ep-strict-valid", "manifest_path": str(ep_path), "sha256": ep_sha}])
        with self.assertRaisesRegex(ValueError, "must be relative"):
            validate_dataset_manifest(m2, dataset_root=self.root)

        # 3. Path traversal rejected
        m3 = dict(base_manifest, episodes=[{"episode_id": "ep-strict-valid", "manifest_path": f"../{rel_path}", "sha256": ep_sha}])
        with self.assertRaisesRegex(ValueError, "path traversal"):
            validate_dataset_manifest(m3, dataset_root=self.root)

        # 4. Non-existent file rejected
        m4 = dict(base_manifest, episodes=[{"episode_id": "ep-strict-valid", "manifest_path": "nonexistent/manifest.json", "sha256": ep_sha}])
        with self.assertRaises(FileNotFoundError):
            validate_dataset_manifest(m4, dataset_root=self.root)

        # 5. Invalid sha256 checksum rejected
        m5 = dict(base_manifest, episodes=[{"episode_id": "ep-strict-valid", "manifest_path": rel_path, "sha256": "wrong_length"}])
        with self.assertRaisesRegex(ValueError, "valid 64-char sha256"):
            validate_dataset_manifest(m5, dataset_root=self.root)

        # 6. Mismatched sha256 checksum rejected
        bad_sha = "0" * 64
        m6 = dict(base_manifest, episodes=[{"episode_id": "ep-strict-valid", "manifest_path": rel_path, "sha256": bad_sha}])
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            validate_dataset_manifest(m6, dataset_root=self.root)

    def test_unsupported_view_rejected(self):
        manifest_file = self.root / "dataset_manifest.json"
        self._make_dataset_manifest(
            manifest_file,
            episodes=[],
            lineage=[],
            asset_families=[],
        )
        with self.assertRaisesRegex(ValueError, "Unsupported view"):
            build_view(manifest_file, "task_outcomes", split="train", config=self.config)

    def test_actual_p11_record_provenance_validation_at_consumer_boundary(self):
        """Issue 4: Raw geom/dyn records AND adapter_a0/a1 must satisfy actual training._validate_record_provenance."""
        from icgs.training.method import _validate_record_provenance

        ep_path = self._create_episode("ep-p11-seam", num_transitions=6, split="train")
        ep_sha = hashlib.sha256(ep_path.read_bytes()).hexdigest()
        manifest_file = self.root / "dataset_manifest.json"
        self._make_dataset_manifest(
            manifest_file,
            episodes=[{"episode_id": "ep-p11-seam", "manifest_path": str(ep_path.relative_to(self.root)), "sha256": ep_sha}],
            lineage=[{"lineage_id": "root-01", "parent_ids": [], "split": "train"}],
            asset_families=[{"asset_family_id": "family-01", "split": "train"}],
        )

        geom_view = build_view(manifest_file, "geom", split="train", config=self.config)
        dyn_view = build_view(manifest_file, "dyn", split="train", config=self.config)

        # 1. Raw geom record must satisfy _validate_record_provenance
        raw_geom = geom_view[0]
        ep_id, lin_id = _validate_record_provenance(raw_geom, is_eval=False)
        self.assertEqual(ep_id, "ep-p11-seam")
        self.assertEqual(lin_id, "root-01")

        # 2. adapter_a0 output must satisfy _validate_record_provenance
        a0_batch = adapter_a0(raw_geom)
        ep_id_a0, lin_id_a0 = _validate_record_provenance(a0_batch, is_eval=False)
        self.assertEqual(ep_id_a0, "ep-p11-seam")
        self.assertEqual(lin_id_a0, "root-01")

        # 3. Raw dyn record must satisfy _validate_record_provenance
        raw_dyn = dyn_view[0]
        ep_id_dyn, lin_id_dyn = _validate_record_provenance(raw_dyn, is_eval=False)
        self.assertEqual(ep_id_dyn, "ep-p11-seam")
        self.assertEqual(lin_id_dyn, "root-01")

        # 4. adapter_a1 output must satisfy _validate_record_provenance
        a1_batch = adapter_a1(raw_dyn)
        ep_id_a1, lin_id_a1 = _validate_record_provenance(a1_batch, is_eval=False)
        self.assertEqual(ep_id_a1, "ep-p11-seam")
        self.assertEqual(lin_id_a1, "root-01")
