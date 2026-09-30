"""Frozen Stage-1 split: leakage rules, lock enforcement and episode assignment."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from icgs.data.stage1.split import (
    SplitViolation,
    assert_allowed,
    load_locked_manifest,
    manifest_sha256,
    validate_split_manifest,
)


def _rule(strategies=(0, 2)):
    return {"strategies": list(strategies), "variations": [0, 1, 2]}


def _manifest():
    return {
        "split_id": "test-split-v1", "status": "frozen", "excluded": [{"task": "custom", "reason": "legacy"}],
        "lineage_rules": ["no leakage"],
        "splits": {
            "train": {"tasks": {"a": _rule(), "c1": _rule()}, "numpy_seed_range": [100, 199], "factor_env_seed": 42},
            "dev": {"tasks": {"a": _rule(), "c2": _rule()}, "numpy_seed_range": [200, 299], "factor_env_seed": 43},
            "test": {"tasks": {"a": _rule((0, 2, 9)), "c3": _rule()}, "numpy_seed_range": [300, 399],
                     "factor_env_seed": 244},
        },
        "tracks": [
            {"track_id": "dev-comp", "split": "dev", "task_novelty": "held_out_task", "tasks": ["c2"],
             "claim": "x", "not_claimed": "y"},
            {"track_id": "dev-config", "split": "dev", "task_novelty": "seen_task", "tasks": ["a"],
             "claim": "x", "not_claimed": "y"},
            {"track_id": "test-comp", "split": "test", "task_novelty": "held_out_task", "tasks": ["c3"],
             "claim": "x", "not_claimed": "y"},
            {"track_id": "test-config", "split": "test", "task_novelty": "seen_task", "tasks": ["a"],
             "claim": "x", "not_claimed": "y"},
        ],
    }


class SplitTests(unittest.TestCase):
    def test_valid_manifest_and_assignment(self):
        manifest = _manifest()
        validate_split_manifest(manifest)
        info = assert_allowed(manifest, split="test", task="c3", strategy=2, variation=1, seed=350,
                              factor_env_seed=244)
        self.assertFalse(info["gradient_eligible"])
        self.assertEqual(info["tracks"], ["test-comp"])
        self.assertTrue(assert_allowed(manifest, split="train", task="a", strategy=0, variation=0, seed=150,
                                       factor_env_seed=42)["gradient_eligible"])

    def test_leakage_is_rejected(self):
        cases = []
        m = _manifest(); m["splits"]["train"]["tasks"]["c3"] = _rule(); cases.append(("held-out task in TRAIN", m))
        m = _manifest(); m["splits"]["dev"]["tasks"]["c3"] = _rule()
        m["tracks"][0]["tasks"].append("c3"); cases.append(("held-out task in DEV and TEST", m))
        m = _manifest(); m["splits"]["dev"]["numpy_seed_range"] = [150, 250]; cases.append(("seed overlap", m))
        m = _manifest(); m["splits"]["dev"]["factor_env_seed"] = 42; cases.append(("shared factor seed", m))
        m = _manifest(); m["splits"]["test"]["tasks"]["mystery"] = _rule(); cases.append(("undeclared eval task", m))
        m = _manifest(); del m["tracks"][2]["not_claimed"]; cases.append(("claim scope missing", m))
        m = _manifest(); m["excluded"].append({"task": "a", "reason": "?"}); cases.append(("excluded but assigned", m))
        m = _manifest(); m["status"] = "draft"; cases.append(("not frozen", m))
        for label, manifest in cases:
            with self.subTest(label), self.assertRaises(SplitViolation):
                validate_split_manifest(manifest)

    def test_collector_assignment_refusals(self):
        manifest = _manifest()
        for kwargs in (dict(split="train", task="c3", strategy=0, variation=0, seed=150, factor_env_seed=42),
                       dict(split="train", task="a", strategy=9, variation=0, seed=150, factor_env_seed=42),
                       dict(split="train", task="a", strategy=0, variation=5, seed=150, factor_env_seed=42),
                       dict(split="train", task="a", strategy=0, variation=0, seed=250, factor_env_seed=42),
                       dict(split="train", task="a", strategy=0, variation=0, seed=150, factor_env_seed=244)):
            with self.subTest(kwargs), self.assertRaises(SplitViolation):
                assert_allowed(manifest, **kwargs)

    def test_lock_detects_any_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(json.dumps(_manifest(), sort_keys=True))
            path.with_suffix(".lock").write_text(json.dumps({"split_id": "test-split-v1",
                                                             "manifest_sha256": manifest_sha256(path)}))
            self.assertEqual(load_locked_manifest(path)["split_id"], "test-split-v1")
            changed = copy.deepcopy(_manifest()); changed["splits"]["train"]["numpy_seed_range"] = [100, 198]
            path.write_text(json.dumps(changed, sort_keys=True))
            with self.assertRaises(SplitViolation):
                load_locked_manifest(path)


if __name__ == "__main__":
    unittest.main()


class FrozenRepositorySplitTests(unittest.TestCase):
    """The committed split-v1 must stay byte-identical to its lock."""

    def test_committed_split_matches_lock_and_rules(self):
        root = Path(__file__).resolve().parents[1] / "artifacts/robohiman"
        manifest = load_locked_manifest(root / "icgs_robohiman_stage1_split_v1.json")
        self.assertEqual(manifest["split_id"], "icgs-robohiman-stage1-split-v1")
        held_out = {t for track in manifest["tracks"] if track["task_novelty"] == "held_out_task"
                    for t in track["tasks"]}
        self.assertFalse(held_out & set(manifest["splits"]["train"]["tasks"]))
        self.assertEqual(len(manifest["splits"]["train"]["tasks"]), 14)
        info = assert_allowed(manifest, split="test", task="put_in_and_close", strategy=0, variation=1,
                              seed=300_000_001, factor_env_seed=244)
        self.assertIn("TEST-dependency-stress", info["tracks"])
        with self.assertRaises(SplitViolation):
            assert_allowed(manifest, split="train", task="put_in_and_close", strategy=0, variation=0,
                           seed=100_000_001, factor_env_seed=42)
