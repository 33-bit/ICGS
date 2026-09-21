from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    from scripts.colab_failure_staging import (
        find_ready_failures,
        plan_failure_sidecar_copies,
        select_failure_publish_root,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.colab_failure_staging import (
        find_ready_failures,
        plan_failure_sidecar_copies,
        select_failure_publish_root,
    )


class TestColabFailureStaging(unittest.TestCase):
    def test_discovers_error_reports_and_skips_committed_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = root / "quarantine" / "g2-ep-t03-10000"
            failed.mkdir(parents=True)
            (failed / "error_report.json").write_text(
                json.dumps({"episode_id": failed.name, "status": "failed"}),
                encoding="utf-8",
            )
            found = find_ready_failures([root], set())
            self.assertEqual([item[0] for item in found], [failed.name])
            self.assertEqual(found[0][2]["status"], "failed")
            self.assertEqual(find_ready_failures([root], {failed.name}), [])

    def test_plans_missing_remote_failure_sidecar_copies(self):
        quarantine_files = {
            "primary_v2/quarantine/g2-ep-t01-10034/error_report.json",
            "primary_v2/quarantine/g2-ep-t01-10034/episode.json",
            "primary_v2/quarantine/g2-ep-t11-00110/episode.json",
        }
        existing_failure_files = {
            "primary_v2/failure_attempts/g2-ep-t11-00110/episode.json",
        }

        copies = plan_failure_sidecar_copies(quarantine_files, existing_failure_files)

        self.assertEqual(
            copies,
            [
                (
                    "primary_v2/quarantine/g2-ep-t01-10034/episode.json",
                    "primary_v2/failure_attempts/g2-ep-t01-10034/episode.json",
                ),
                (
                    "primary_v2/quarantine/g2-ep-t01-10034/error_report.json",
                    "primary_v2/failure_attempts/g2-ep-t01-10034/error_report.json",
                ),
            ],
        )

    def test_falls_back_to_quarantine_when_v2_sidecar_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quarantine = root / "quarantine" / "episode"
            sidecar = root / "failure_attempts_v2" / "episode"
            quarantine.mkdir(parents=True)
            (quarantine / "error_report.json").write_text("{}", encoding="utf-8")

            self.assertEqual(select_failure_publish_root(quarantine, sidecar), quarantine)

            sidecar.mkdir(parents=True)
            (sidecar / "layout_manifest.json").write_text("{}", encoding="utf-8")
            self.assertEqual(select_failure_publish_root(quarantine, sidecar), sidecar)


if __name__ == "__main__":
    unittest.main()
