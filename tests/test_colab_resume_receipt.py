from __future__ import annotations

import unittest

try:
    from scripts.colab_resume_receipt import build_resume_receipt
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.colab_resume_receipt import build_resume_receipt


class TestColabResumeReceipt(unittest.TestCase):
    def test_counts_are_derived_from_merged_manifest(self):
        approved = {
            "generation_targets": {
                "train_successes_per_program": 200,
                "development_successes_by_program": {"V01": 63},
                "test_contexts_per_composition": 100,
            },
            "catalog": [
                {"program_id": "T01", "split": "train"},
                {"program_id": "V01", "split": "development"},
                {"program_id": "P1", "split": "test"},
            ],
        }
        manifest = {
            "episodes": [
                {"episode_id": "e1", "program_id": "T01"},
                {"episode_id": "e2", "program_id": "T01"},
                {"episode_id": "e3", "program_id": "V01"},
            ],
            "quarantined": [{"episode_id": "q1"}],
            "failure_attempts": [{"episode_id": "q1"}, {"episode_id": "q2"}],
        }

        receipt = build_resume_receipt(
            manifest,
            approved,
            approved_manifest_digest="digest",
            hf_repo="33bit/icgs",
            hf_subfolder="primary_v2",
            recent_commits=["old"],
            timestamp="2026-01-01T00:00:00+00:00",
        )

        self.assertEqual(receipt["total_target"], 363)
        self.assertEqual(receipt["total_generated"], 3)
        self.assertEqual(receipt["target_shortfall"], 360)
        self.assertEqual(receipt["total_quarantined"], 1)
        self.assertEqual(receipt["total_failure_attempts"], 2)
        self.assertEqual(receipt["task_progress"]["V01"]["generated"], 1)
        self.assertEqual(receipt["recent_commits"], ["old"])


if __name__ == "__main__":
    unittest.main()
