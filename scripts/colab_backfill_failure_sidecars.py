"""Backfill missing primary_v2 failure-attempt sidecars from quarantine files.

This is intentionally additive: it never edits ``primary`` and does not rewrite
the success manifest or receipt.  It only copies missing files inside
``primary_v2`` on the same Hub revision.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationCopy, HfApi

try:
    from colab_failure_staging import plan_failure_sidecar_copies
except ImportError:
    # ``colab exec -f`` executes this file in a notebook kernel whose
    # ``/content/ICGS`` checkout may lag the local workspace.  Keep the
    # read-only mapping self-contained so the backfill remains reproducible.
    def plan_failure_sidecar_copies(
        quarantine_files,
        existing_failure_files,
        *,
        quarantine_prefix="primary_v2/quarantine",
        failure_prefix="primary_v2/failure_attempts",
    ):
        q_prefix = quarantine_prefix.rstrip("/") + "/"
        f_prefix = failure_prefix.rstrip("/") + "/"
        existing = {str(path).lstrip("/") for path in existing_failure_files}
        planned = []
        for raw_path in sorted({str(path).lstrip("/") for path in quarantine_files}):
            if not raw_path.startswith(q_prefix):
                continue
            relative = raw_path[len(q_prefix) :]
            if not relative or relative.endswith("/"):
                continue
            destination = f_prefix + relative
            if destination not in existing:
                planned.append((raw_path, destination))
        return planned


REPO = "33bit/icgs"
SUBFOLDER = "primary_v2"
TOKEN_FILE = Path("/content/.icgs_hf_token")


def _file_paths(api: HfApi, path: str) -> set[str]:
    result: set[str] = set()
    for item in api.list_repo_tree(repo_id=REPO, repo_type="dataset", path_in_repo=path, recursive=True):
        item_path = getattr(item, "path", None)
        if item_path and item.__class__.__name__ == "RepoFile":
            result.add(str(item_path))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="only print the planned copies")
    args, _unknown = parser.parse_known_args()

    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    api = HfApi(token=token)
    quarantine_files = _file_paths(api, f"{SUBFOLDER}/quarantine")
    failure_files = _file_paths(api, f"{SUBFOLDER}/failure_attempts")
    copies = plan_failure_sidecar_copies(quarantine_files, failure_files)
    print(f"quarantine_files={len(quarantine_files)} failure_files={len(failure_files)} missing_copies={len(copies)}")
    for source, destination in copies[:10]:
        print(f"COPY {source} -> {destination}")
    if len(copies) > 10:
        print(f"... {len(copies) - 10} more copies")

    if args.dry_run or not copies:
        return 0

    operations = [
        CommitOperationCopy(src_path_in_repo=source, path_in_repo=destination)
        for source, destination in copies
    ]
    result = api.create_commit(
        repo_id=REPO,
        repo_type="dataset",
        operations=operations,
        commit_message=(
            f"Backfill {len(copies)} missing primary_v2 failure-attempt sidecar files"
        ),
    )
    print(f"commit={getattr(result, 'oid', result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
