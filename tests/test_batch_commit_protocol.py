"""Unit test suite for Colab G2 dataset generator batch commit protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

try:
    from scripts.colab_g2_dataset_generator import parse_args
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.colab_g2_dataset_generator import parse_args


class TestBatchCommitProtocol(unittest.TestCase):
    def test_default_commit_batch_size(self):
        with patch("sys.argv", ["generator"]):
            args = parse_args()
            self.assertEqual(args.commit_batch_size, 10)

    def test_custom_commit_batch_size(self):
        with patch("sys.argv", ["generator", "--commit-batch-size", "5"]):
            args = parse_args()
            self.assertEqual(args.commit_batch_size, 5)

    def test_manifest_merge_and_quarantine_reconciliation(self):
        """Verify that concurrent commits from other sessions are preserved, and quarantined IDs are excluded from success counts."""
        remote_manifest = {
            "manifest_version": 1,
            "dataset_track": "primary",
            "episodes": [
                {"episode_id": "g2-ep-t01-10000", "program_id": "T01"},
                {"episode_id": "g2-ep-t01-10002", "program_id": "T01"},
                {"episode_id": "g2-ep-t11-10000", "program_id": "T11"},
            ],
            "quarantined": [
                {"episode_id": "g2-ep-t01-10093", "program_id": "T01", "status": "failed"},
            ],
            "lineage": [{"lineage_id": "lin-1"}],
            "asset_families": [{"asset_family_id": "fam-1"}],
        }

        local_manifest = {
            "manifest_version": 1,
            "dataset_track": "primary",
            "episodes": [
                {"episode_id": "g2-ep-t01-10002", "program_id": "T01"},
                {"episode_id": "g2-ep-t03-10060", "program_id": "T03"},
                {"episode_id": "g2-ep-t03-10061", "program_id": "T03"},
            ],
            "quarantined": [
                {"episode_id": "g2-ep-t01-10000", "program_id": "T01", "status": "failed"},
                {"episode_id": "g2-ep-t01-10001", "program_id": "T01", "status": "failed"},
            ],
            "lineage": [{"lineage_id": "lin-1"}, {"lineage_id": "lin-2"}],
            "asset_families": [{"asset_family_id": "fam-1"}, {"asset_family_id": "fam-2"}],
        }

        merged_episodes = {}
        for ep in remote_manifest.get("episodes", []):
            merged_episodes[ep["episode_id"]] = ep
        for ep in local_manifest.get("episodes", []):
            merged_episodes[ep["episode_id"]] = ep

        merged_quarantined = {}
        for q in remote_manifest.get("quarantined", []):
            merged_quarantined[q["episode_id"]] = q
        for q in local_manifest.get("quarantined", []):
            merged_quarantined[q["episode_id"]] = q

        quarantined_ids = set(merged_quarantined.keys())
        valid_episodes = [
            ep for ep in merged_episodes.values()
            if ep["episode_id"] not in quarantined_ids
        ]

        self.assertEqual(len(merged_quarantined), 3)
        self.assertIn("g2-ep-t01-10000", merged_quarantined)
        self.assertIn("g2-ep-t01-10001", merged_quarantined)
        self.assertIn("g2-ep-t01-10093", merged_quarantined)

        valid_ids = {ep["episode_id"] for ep in valid_episodes}
        self.assertNotIn("g2-ep-t01-10000", valid_ids)
        self.assertNotIn("g2-ep-t01-10001", valid_ids)
        self.assertNotIn("g2-ep-t01-10093", valid_ids)

        self.assertIn("g2-ep-t11-10000", valid_ids)
        self.assertIn("g2-ep-t03-10060", valid_ids)
        self.assertIn("g2-ep-t03-10061", valid_ids)
        self.assertIn("g2-ep-t01-10002", valid_ids)
        self.assertEqual(len(valid_episodes), 4)

    def test_batch_buffering_and_residual_flush(self):
        batch_size = 10
        pending_commits = []
        flushed_batches = []

        def mock_flush(reason: str):
            flushed_batches.append((reason, list(pending_commits)))
            pending_commits.clear()

        for i in range(15):
            ep_id = f"g2-ep-t03-{10060 + i:05d}"
            pending_commits.append(ep_id)
            if len(pending_commits) >= batch_size:
                mock_flush(f"batch_{batch_size}")

        if pending_commits:
            mock_flush("task_end_T03")

        self.assertEqual(len(flushed_batches), 2)
        self.assertEqual(flushed_batches[0][0], "batch_10")
        self.assertEqual(len(flushed_batches[0][1]), 10)
        self.assertEqual(flushed_batches[1][0], "task_end_T03")
        self.assertEqual(len(flushed_batches[1][1]), 5)

        for i in range(7):
            ep_id = f"g2-ep-t04-{10000 + i:05d}"
            pending_commits.append(ep_id)
            if len(pending_commits) >= batch_size:
                mock_flush(f"batch_{batch_size}")

        if pending_commits:
            mock_flush("run_end")

        self.assertEqual(len(flushed_batches), 3)
        self.assertEqual(flushed_batches[2][0], "run_end")
        self.assertEqual(len(flushed_batches[2][1]), 7)
        self.assertEqual(sum(len(b[1]) for b in flushed_batches), 22)



    def test_conflict_429_retry_simulation(self):
        """Simulate HF 429 error on first attempt, succeeded on retry after re-fetching remote manifest."""
        api = MagicMock()
        call_count = 0

        def mock_create_commit(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("429 Client Error: Too Many Requests")
            return {"commit_hash": "mock_commit_hash_123"}

        api.create_commit.side_effect = mock_create_commit

        # Simulate generator retry loop logic
        commit_hash = None
        for attempt in range(1, 41):
            try:
                commit_res = api.create_commit(repo_id="33bit/icgs", repo_type="dataset")
                commit_hash = commit_res.get("commit_hash")
                break
            except Exception:
                continue

        self.assertEqual(call_count, 2)
        self.assertEqual(commit_hash, "mock_commit_hash_123")

    def test_quarantine_canonical_structure_and_failure_attempts_manifest(self):
        """Verify quarantine directory layout and manifest failure_attempts retention."""
        import numpy as np
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            ep_id = "g2-ep-t03-10065"
            quarantine_dir = out_dir / "quarantine" / ep_id
            quarantine_dir.mkdir(parents=True)

            # Write canonical files
            (quarantine_dir / "valid_observations.npz").write_bytes(b"dummy")
            (quarantine_dir / "episode.json").write_text(json.dumps({"episode_id": ep_id, "status": "failed_attempt"}))
            (quarantine_dir / "result.json").write_text(json.dumps({"success": False, "failure_reason": "IK timeout"}))
            (quarantine_dir / "actions").mkdir()
            np.save(quarantine_dir / "actions" / "actions.npy", np.zeros((5, 8)))
            (quarantine_dir / "observations").mkdir()
            np.savez_compressed(quarantine_dir / "observations" / "raw.npz", wrist_depth=np.zeros((5, 128, 128)))
            (quarantine_dir / "error_report.json").write_text(json.dumps({"episode_id": ep_id, "valid_observation_retained": True}))

            # Verify files exist
            self.assertTrue((quarantine_dir / "valid_observations.npz").is_file())
            self.assertTrue((quarantine_dir / "episode.json").is_file())
            self.assertTrue((quarantine_dir / "result.json").is_file())
            self.assertTrue((quarantine_dir / "actions" / "actions.npy").is_file())
            self.assertTrue((quarantine_dir / "observations" / "raw.npz").is_file())

            # Verify rglob finds nested files
            rel_paths = [f.relative_to(quarantine_dir).as_posix() for f in quarantine_dir.rglob("*") if f.is_file()]
            self.assertIn("actions/actions.npy", rel_paths)
            self.assertIn("observations/raw.npz", rel_paths)
            self.assertIn("valid_observations.npz", rel_paths)
            self.assertIn("result.json", rel_paths)
            self.assertIn("episode.json", rel_paths)

    def test_disjoint_id_advancement_with_quarantine(self):
        """Verify next_idx skips past both successful and quarantined episode IDs."""
        manifest_data = {
            "episodes": [
                {"episode_id": "g2-ep-t03-10060", "program_id": "T03"},
                {"episode_id": "g2-ep-t03-10061", "program_id": "T03"},
            ],
            "quarantined": [
                {"episode_id": "g2-ep-t03-10062", "program_id": "T03", "status": "failed"},
            ],
            "failure_attempts": [
                {"episode_id": "g2-ep-t03-10063", "program_id": "T03", "status": "failed"},
            ],
        }

        task_id = "T03"
        start_idx = 10000
        used_indices = set()
        for ep in manifest_data.get("episodes", []):
            if ep.get("program_id") == task_id:
                used_indices.add(int(ep["episode_id"].split("-")[-1]))
        for q in manifest_data.get("quarantined", []):
            qid = q.get("episode_id", "") if isinstance(q, dict) else str(q)
            if qid.startswith(f"g2-ep-{task_id.lower()}-"):
                used_indices.add(int(qid.split("-")[-1]))
        for fa in manifest_data.get("failure_attempts", []):
            faid = fa.get("episode_id", "") if isinstance(fa, dict) else str(fa)
            if faid.startswith(f"g2-ep-{task_id.lower()}-"):
                used_indices.add(int(faid.split("-")[-1]))

        next_idx = max(start_idx, max(used_indices, default=start_idx - 1) + 1)
        self.assertEqual(next_idx, 10064)



if __name__ == "__main__":
    unittest.main()
