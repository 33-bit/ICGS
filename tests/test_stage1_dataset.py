"""RoboHiMan dataset layout, split enforcement, derived products and Stage-1 views (no simulator)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from icgs.data.stage1.dataset import Dataset, make_episode_id, remove_scratch_dataset
from icgs.data.stage1.split import SplitViolation
from icgs.data.stage1.store import manifest_sha256, read_episode
from icgs.data.stage1.views import LineageViolation, ReferencePolicyRequired, build_views
from stage1_fixtures import LEGEND, SPLIT_MANIFEST, make_episode

TRAIN_SEED = 100_000_001
TEST_QUERY_SEED = 300_000_001
TEST_CONTEXT_SEED = 390_000_001


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class DatasetCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "datasets" / "robohiman"
        self.dataset = Dataset.create(self.root, SPLIT_MANIFEST)

    def tearDown(self):
        remove_scratch_dataset(self.tmp)

    def episode(self, split, index, seed, *, task="open_drawer", run8="a81f2c3d", env_seed=None, **kwargs):
        env_seed = env_seed if env_seed is not None else {"train": 42, "dev": 4242, "test": 244}[split]
        strategy = kwargs.pop("strategy", (0, "no_variations"))
        provenance = self.dataset.split_provenance(split=split, task=task, strategy=strategy[0], variation=0,
                                                   seed=seed, factor_env_seed=env_seed)
        return make_episode(make_episode_id(task, run8, index), provenance=provenance, seed=seed, task=task,
                            env_seed=env_seed, strategy=strategy, **kwargs)

    def commit(self, split, index, seed, **kwargs):
        manifest, arrays = self.episode(split, index, seed, **kwargs)
        return self.dataset.commit_episode(split, manifest, arrays)


class LayoutTests(DatasetCase):
    def test_layout_and_metadata(self):
        expected = {"manifest.json", "splits.json", "preprocessing", "reports", "train", "dev", "test"}
        self.assertEqual({p.name for p in self.root.iterdir()}, expected)
        for split in ("train", "dev", "test"):
            self.assertEqual({p.name for p in (self.root / split).iterdir()},
                             {"episodes", "attempts", "derived", "views"})
        self.assertEqual({p.name for p in (self.root / "reports").iterdir()},
                         {"audit.json", "leakage.json", "reproducibility.json", "generation_log.jsonl"})
        self.assertEqual((self.root / "splits.json").read_bytes(), SPLIT_MANIFEST.read_bytes())
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual(manifest["split_manifest_sha256"], hashlib.sha256(SPLIT_MANIFEST.read_bytes()).hexdigest())
        self.assertEqual(manifest["gradient_splits"], ["train"])
        for part in self.root.relative_to(self.tmp).parts:
            self.assertNotRegex(part, r"icgs|stage1|v\d")
        self.assertEqual(Dataset.create(self.root, SPLIT_MANIFEST).manifest, manifest, "re-create is idempotent")

    def test_tampered_split_copy_is_rejected(self):
        path = self.root / "splits.json"
        path.chmod(0o644)
        path.write_text(path.read_text().replace("199999999", "299999999"))
        with self.assertRaises(SplitViolation):
            Dataset.open(self.root)


class SplitEnforcementTests(DatasetCase):
    def test_commit_lands_in_its_split_and_is_logged(self):
        path = self.commit("train", 0, TRAIN_SEED)
        self.assertEqual(path, self.root / "train" / "episodes" / "ep-open_drawer-a81f2c3d-0000")
        self.assertEqual([p.name for p in (self.root / "train" / "episodes").iterdir()], [path.name])
        log = _rows(self.root / "reports" / "generation_log.jsonl")
        self.assertEqual((log[-1]["event"], log[-1]["split"], log[-1]["seed"]),
                         ("episode_committed", "train", TRAIN_SEED))

    def test_disagreeing_episodes_are_refused(self):
        def case(label, split="train", mutate=None, destination=None, **kwargs):
            manifest, arrays = self.episode(kwargs.pop("declared", "train"), 7, kwargs.pop("seed", TRAIN_SEED),
                                            **kwargs)
            if mutate:
                mutate(manifest, arrays)
            with self.subTest(label), self.assertRaises(SplitViolation):
                self.dataset.commit_episode(split, manifest, arrays, destination=destination)

        case("declared train, written to dev", split="dev")
        case("wrong destination", destination=self.root / "dev")
        case("no split provenance", mutate=lambda m, a: m["lineage"].pop("split"))
        case("stale split hash", mutate=lambda m, a: m["lineage"]["split"].update(split_manifest_sha256="0" * 64))
        case("forged gradient flag", split="dev", declared="dev", seed=200_000_001,
             mutate=lambda m, a: m["lineage"]["split"].update(gradient_eligible=True))
        case("forged tracks", mutate=lambda m, a: m["lineage"]["split"].update(tracks=["TEST-config"]))
        case("withheld perturbation strategy outside TEST",
             mutate=lambda m, a: m["source"]["collection_strategy"].update(name="distractor"))
        case("seed outside the split range",
             mutate=lambda m, a: m["lineage"]["split"].update(episode_seed=250_000_000))
        case("reset RNG from another seed",
             mutate=lambda m, a: a.update(rng_state_mt19937=np.zeros_like(a["rng_state_mt19937"])))
        case("factor seed of another split", mutate=lambda m, a: m["environment"].update(env_seed=244))
        case("strategy not in split", mutate=lambda m, a: m["source"]["collection_strategy"].update(index=1))
        case("parent outside the split",
             mutate=lambda m, a: m["lineage"].update(parent_episode_id="ep-open_drawer-00000000-0000"))
        case("mask handle without legend entry",
             mutate=lambda m, a: a["cam_front_mask_handles"].__setitem__((0, 0, 0), 999))
        with self.assertRaises(SplitViolation):  # held-out TEST task never gets TRAIN provenance
            self.dataset.split_provenance(split="train", task="put_in_and_close", strategy=0, variation=0,
                                          seed=TRAIN_SEED, factor_env_seed=42)
        self.assertEqual(list((self.root / "train" / "episodes").iterdir()), [])
        self.assertEqual(list((self.root / "dev" / "episodes").iterdir()), [])

    def test_episode_ids_are_unique_across_splits(self):
        self.commit("train", 0, TRAIN_SEED)
        with self.assertRaisesRegex(SplitViolation, "already exists in train"):
            self.commit("dev", 0, 200_000_001)
        self.commit("dev", 0, 200_000_001, run8="0b1c2d3e")
        self.assertEqual(self.dataset.episode_index(), {"ep-open_drawer-a81f2c3d-0000": "train",
                                                        "ep-open_drawer-0b1c2d3e-0000": "dev"})

    def test_test_roles_follow_the_context_seed_range(self):
        self.assertEqual(self.dataset.split_provenance(split="test", task="open_drawer", strategy=0, variation=0,
                                                       seed=TEST_CONTEXT_SEED, factor_env_seed=244)["test_role"],
                         "context")
        manifest, arrays = self.episode("test", 0, TEST_QUERY_SEED)
        manifest["lineage"]["split"]["test_role"] = "context"
        with self.assertRaises(SplitViolation):
            self.dataset.commit_episode("test", manifest, arrays)

    def test_attempts_are_not_episodes(self):
        path = self.dataset.record_attempt("train", "att-open_drawer-a81f2c3d-0003", {"outcome": "simulator_error"})
        self.assertEqual(path.parent, self.root / "train" / "attempts")
        self.assertEqual(self.dataset.episode_index(), {})
        with self.assertRaises(ValueError):
            self.dataset.record_attempt("train", "open_drawer-3", {})


class DerivedAndPreprocessingTests(DatasetCase):
    def metadata(self, path):
        manifest = json.loads((path / "manifest.json").read_text())
        return {"is_derived": True, "derivation": "test fk", "derivation_version": 1,
                "source_episode_id": manifest["episode_id"], "source_manifest_sha256": manifest_sha256(path),
                "source_arrays_sha256": manifest["arrays"]["sha256"],
                "conventions": {"frame": "world", "orientation": "quaternion xyzw"},
                "reconstruction_error": {"position_max_m": 1e-4}}

    def test_derived_products_never_touch_raw_episodes(self):
        path = self.commit("train", 0, TRAIN_SEED)
        before = {p.name: p.read_bytes() for p in path.iterdir()}
        episode = path.name
        meta = self.metadata(path)
        for key in ("is_derived", "reconstruction_error", "source_arrays_sha256"):
            broken = {k: v for k, v in meta.items() if k != key}
            with self.subTest(key), self.assertRaises(ValueError):
                self.dataset.write_derived("train", episode, "canonical_ee_actions_v1", {"x": np.zeros(3)}, broken)
        with self.assertRaises(ValueError):
            self.dataset.write_derived("train", episode, "canonical_ee_actions_v1", {"x": np.zeros(3)},
                                       dict(meta, source_arrays_sha256="0" * 64))
        with self.assertRaises(ValueError):
            self.dataset.write_derived("train", episode, "canonical_commands", {"x": np.zeros(3)}, meta)
        out = self.dataset.write_derived("train", episode, "canonical_ee_actions_v1", {"x": np.zeros(3)}, meta)
        self.assertEqual(out, self.root / "train" / "derived" / episode)
        record = json.loads((out / "canonical_ee_actions_v1.json").read_text())
        self.assertEqual(record["arrays_sha256"],
                         hashlib.sha256((out / "canonical_ee_actions_v1.npz").read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            self.dataset.write_derived("train", episode, "canonical_ee_actions_v1", {"x": np.ones(3)}, meta)
        with self.assertRaises(SplitViolation):
            self.dataset.write_derived("dev", episode, "canonical_ee_actions_v1", {"x": np.ones(3)}, meta)
        self.assertEqual({p.name: p.read_bytes() for p in path.iterdir()}, before)

    def test_preprocessing_is_fitted_on_train_only(self):
        self.commit("train", 0, TRAIN_SEED)
        self.commit("dev", 0, 200_000_001, run8="0b1c2d3e")
        with self.assertRaises(SplitViolation):
            self.dataset.write_preprocessing("joint_stats_v1", {"mean": [0.0]}, method="mean",
                                             source_episode_ids=["ep-open_drawer-a81f2c3d-0000",
                                                                 "ep-open_drawer-0b1c2d3e-0000"])
        path = self.dataset.write_preprocessing("joint_stats_v1", {"mean": [0.0]}, method="mean",
                                                source_episode_ids=["ep-open_drawer-a81f2c3d-0000"])
        self.assertEqual(json.loads(path.read_text())["fitted_on"], "train")
        index = json.loads((self.root / "preprocessing" / "index.json").read_text())
        self.assertEqual(index["files"], ["joint_stats_v1.json"])


class ViewTests(DatasetCase):
    def test_d_dyn_separates_command_from_achieved_state(self):
        path = self.commit("train", 0, TRAIN_SEED)
        build_views(self.root / "train", view_names=("D_dyn",))
        rows = _rows(self.root / "train" / "views" / "D_dyn.jsonl")
        _, arrays = read_episode(path)
        self.assertEqual(len(rows), 12)
        for row in rows:
            s = row["command_row"]
            self.assertEqual((row["transition"], row["state_boundary"], row["achieved_boundary"]),
                             ("physics_step", s, s + 1))
            self.assertAlmostEqual(row["dt_s"], 0.05)
        commanded = arrays["cmd_arm_joint_target"]
        achieved_next = arrays["step_joint_positions"][1:]
        self.assertGreater(np.abs(commanded - achieved_next).min(), 0, "command is not the achieved state")
        self.assertEqual((rows[2]["frame_before"], rows[2]["frame_after"]), (1, None))

    def test_d_geom_masks_reconstruct_roles(self):
        path = self.commit("train", 0, TRAIN_SEED)
        self.commit("train", 1, TRAIN_SEED + 1, masks=False)
        build_views(self.root / "train", view_names=("D_geom",))
        rows = _rows(self.root / "train" / "views" / "D_geom.jsonl")
        primary = [r for r in rows if r["primary"]]
        self.assertEqual({r["episode_id"] for r in primary}, {path.name})
        self.assertTrue(all(r["non_primary_reason"] for r in rows if not r["primary"]))
        manifest, arrays = read_episode(path)
        row = primary[3]
        self.assertEqual(int(arrays["frame_step"][row["frame"]]), row["boundary"])
        handles = arrays[row["arrays"]["mask"].replace("<cam>", "front")][row["frame"]]
        legend = manifest["cameras"]["mask_legend"]
        roles = np.vectorize(lambda h: legend[str(h)]["role"])(handles)
        self.assertEqual(int((roles == "task_fixture").sum()), 4)
        self.assertEqual(int((roles == "robot").sum()), 1)
        foreground = roles != "background"
        np.testing.assert_array_equal(foreground, handles != 0)
        self.assertEqual(legend, LEGEND)

    def test_d_task_pairs_distinct_lineage_and_records_provenance(self):
        a = self.commit("train", 0, TRAIN_SEED)                                   # success
        b = self.commit("train", 1, TRAIN_SEED + 1, status="failure")             # failure query
        branch = self.commit("train", 2, TRAIN_SEED + 2, parent=a.name)            # branch of a
        summary = build_views(self.root / "train", view_names=("D_task",))
        rows = _rows(self.root / "train" / "views" / "D_task.jsonl")
        self.assertTrue(rows)
        # a and its branch share a lineage root, so neither may serve as the other's context.
        self.assertEqual({r["query_episode"] for r in rows}, {b.name})
        self.assertTrue(all(set(r["context_episodes"]) <= {a.name, branch.name} for r in rows))
        self.assertEqual(set(summary["leakage"]["unpaired_queries"]), {a.name, branch.name})
        failure_rows = [r for r in rows if r["query_episode"] == b.name]
        self.assertEqual(failure_rows[-1]["alignment_target"], 1, "grasped but never opened")
        self.assertFalse(failure_rows[-1]["alignment_is_null"])
        self.assertEqual([t["event_id"] for t in failure_rows[0]["context_tokens"]],
                         ["handle_grasped", "drawer_opened"])
        for key in ("rho", "nu", "epsilon"):
            self.assertEqual(len(failure_rows[0][key]), len(failure_rows[0][f"{key}_valid"]))
        manifest = json.loads((self.root / "train" / "views" / "manifest.json").read_text())
        for episode in (a, b, branch):
            source = manifest["sources"][episode.name]
            self.assertEqual(source["manifest_sha256"], manifest_sha256(episode))
            self.assertEqual(source["arrays_sha256"], json.loads((episode / "manifest.json").read_text())
                             ["arrays"]["sha256"])
        payload = (self.root / "train" / "views" / "D_task.jsonl").read_bytes()
        self.assertEqual(manifest["views"]["D_task"]["sha256"], hashlib.sha256(payload).hexdigest())
        with self.assertRaises(ReferencePolicyRequired):
            build_views(self.root / "train", view_names=("D_value",))

    def test_test_queries_and_contexts_never_mix(self):
        q1 = self.commit("test", 0, TEST_QUERY_SEED)
        q2 = self.commit("test", 1, TEST_QUERY_SEED + 1, status="failure")
        c1 = self.commit("test", 2, TEST_CONTEXT_SEED)
        c2 = self.commit("test", 3, TEST_CONTEXT_SEED + 1)
        summary = build_views(self.root / "test", view_names=("D_task",), contexts_per_query=2)
        rows = _rows(self.root / "test" / "views" / "D_task.jsonl")
        self.assertEqual({r["query_episode"] for r in rows}, {q1.name, q2.name})
        self.assertEqual({c for r in rows for c in r["context_episodes"]}, {c1.name, c2.name})
        self.assertEqual(summary["leakage"]["pairs"], 4)
        self.assertEqual(summary["sources"][c1.name]["test_role"], "context")
        again = build_views(self.root / "test", view_names=("D_task",), contexts_per_query=2)
        self.assertEqual(again["views"]["D_task"]["sha256"], summary["views"]["D_task"]["sha256"],
                         "pairing is deterministic")

    def test_misplaced_episode_is_a_lineage_violation(self):
        path = self.commit("dev", 0, 200_000_001)
        target = self.root / "train" / "episodes" / path.name
        path.rename(target)
        with self.assertRaises(LineageViolation):
            build_views(self.root / "train")


if __name__ == "__main__":
    unittest.main()
