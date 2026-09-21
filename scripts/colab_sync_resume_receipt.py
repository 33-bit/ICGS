"""Synchronize primary_v2/resume_receipt.json with the current HF manifest.

This is a metadata-only repair: it never changes dataset_manifest.json, episode
files, quarantine files, or the legacy primary track.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from colab_resume_receipt import (
    build_resume_receipt,
    load_approved_manifest,
    write_resume_receipt,
)


HF_REPO = "33bit/icgs"
HF_SUBFOLDER = "primary_v2"
TOKEN_FILE = Path("/content/.icgs_hf_token")
APPROVED_MANIFEST = Path("/content/ICGS/artifacts/composition/approved_composition_manifest.json")
RECEIPT_TMP = Path("/content/primary_v2_resume_receipt_sync.json")


def main() -> None:
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    api = HfApi(token=token)
    approved, digest = load_approved_manifest(APPROVED_MANIFEST)

    for attempt in range(1, 11):
        try:
            manifest_path = hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=f"{HF_SUBFOLDER}/dataset_manifest.json",
                token=token,
                force_download=True,
            )
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

            old_receipt = {}
            try:
                old_path = hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    filename=f"{HF_SUBFOLDER}/resume_receipt.json",
                    token=token,
                    force_download=True,
                )
                old_receipt = json.loads(Path(old_path).read_text(encoding="utf-8"))
            except Exception:
                pass

            receipt = build_resume_receipt(
                manifest,
                approved,
                approved_manifest_digest=digest,
                hf_repo=HF_REPO,
                hf_subfolder=HF_SUBFOLDER,
                recent_commits=old_receipt.get("recent_commits", []),
            )
            write_resume_receipt(RECEIPT_TMP, receipt)
            commit = api.create_commit(
                repo_id=HF_REPO,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(
                        path_in_repo=f"{HF_SUBFOLDER}/resume_receipt.json",
                        path_or_fileobj=str(RECEIPT_TMP),
                    )
                ],
                commit_message=(
                    "Synchronize primary_v2 resume receipt with merged manifest "
                    f"({receipt['total_generated']} successes)"
                ),
            )
            oid = getattr(commit, "oid", str(commit))
            print(json.dumps({"commit": oid, **receipt}, indent=2))
            return
        except Exception as exc:
            if attempt == 10:
                raise
            print(f"receipt sync attempt {attempt} failed: {exc}")
            time.sleep(min(60, 3 * attempt))


if __name__ == "__main__":
    main()
