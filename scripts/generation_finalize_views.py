#!/usr/bin/env python3
"""Finalize archive-backed generation views at a pinned HF revision.

The source prefix and output prefix are explicit. The script downloads only the
dataset manifest and compact episode manifests from the pinned source revision;
it never downloads numeric archive chunks. Final snapshots are committed under
``<output-prefix>/<source-revision>/seed-<mixture-seed>/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from icgs.data.datasets.generation_view_index import (
    DEFAULT_ROLE_SPECS,
    finalize_generation_views,
    validate_source_revision,
)


def _prefix(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or path.as_posix() != value.rstrip("/")
    ):
        raise ValueError(f"{field} must be a normalized repo-relative prefix")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError(f"{field} must be a normalized repo-relative prefix")
    return value.rstrip("/")


def _download_pinned_manifest_tree(
    *,
    repo_id: str,
    source_prefix: str,
    source_revision: str,
    root: Path,
) -> Path:
    manifest_repo_path = f"{source_prefix}/dataset_manifest.json"
    downloaded_manifest = Path(hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=manifest_repo_path,
        revision=source_revision,
    ))
    source_root = root / source_prefix
    source_root.mkdir(parents=True, exist_ok=True)
    manifest_path = source_root / "dataset_manifest.json"
    shutil.copyfile(downloaded_manifest, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("episodes"), list):
        raise ValueError("pinned HF dataset manifest is malformed")

    downloaded_paths: set[str] = set()
    for row in manifest["episodes"]:
        if not isinstance(row, dict) or not isinstance(row.get("archive_manifest"), str):
            raise ValueError("pinned HF dataset episode row is missing archive_manifest")
        relative = PurePosixPath(row["archive_manifest"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in row["archive_manifest"]:
            raise ValueError("pinned HF archive manifest path is unsafe")
        relative_name = relative.as_posix()
        if relative_name in downloaded_paths:
            continue
        downloaded_paths.add(relative_name)
        downloaded_archive = Path(hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"{source_prefix}/{relative_name}",
            revision=source_revision,
        ))
        destination = source_root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded_archive, destination)
    return manifest_path


def _ensure_disjoint_prefixes(source_prefix: str, output_prefix: str) -> None:
    if (
        source_prefix == output_prefix
        or source_prefix.startswith(output_prefix + "/")
        or output_prefix.startswith(source_prefix + "/")
    ):
        raise ValueError("output-prefix must be separate from the immutable source prefix")


def _remote_snapshot_matches(
    api: HfApi,
    *,
    repo_id: str,
    revision: str,
    file_paths: list[tuple[str, Path]],
) -> bool:
    exists = [api.file_exists(
        repo_id=repo_id,
        filename=remote_path,
        repo_type="dataset",
        revision=revision,
    ) for remote_path, _local_path in file_paths]
    if not any(exists):
        return False
    if not all(exists):
        raise ValueError("partial immutable final view snapshot already exists at output-prefix")
    for (remote_path, local_path), present in zip(file_paths, exists):
        if not present:
            return False
        remote_path_local = Path(hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=remote_path,
            revision=revision,
        ))
        if remote_path_local.read_bytes() != local_path.read_bytes():
            raise ValueError(f"immutable final view conflict at {remote_path}")
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repository ID")
    parser.add_argument("--source-prefix", required=True, help="Prefix containing dataset_manifest.json")
    parser.add_argument("--source-revision", required=True, help="Pinned 40/64-character HF commit OID")
    parser.add_argument("--output-prefix", required=True, help="New, separate prefix for immutable final views")
    parser.add_argument("--mixture-seed", required=True, type=int, help="Recorded deterministic 70/30 selection seed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_revision = validate_source_revision(args.source_revision)
    if args.mixture_seed < 0:
        raise ValueError("mixture-seed must be nonnegative")
    source_prefix = _prefix(args.source_prefix, "source-prefix")
    output_prefix = _prefix(args.output_prefix, "output-prefix")
    _ensure_disjoint_prefixes(source_prefix, output_prefix)

    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="icgs-finalize-views-") as temporary:
        scratch = Path(temporary)
        dataset_manifest_path = _download_pinned_manifest_tree(
            repo_id=args.repo_id,
            source_prefix=source_prefix,
            source_revision=source_revision,
            root=scratch / "source",
        )
        output_dir = scratch / "final"
        receipt = finalize_generation_views(
            dataset_manifest_path,
            source_revision=source_revision,
            role_specs=DEFAULT_ROLE_SPECS,
            mixture_seed=args.mixture_seed,
            output_dir=output_dir,
        )
        remote_prefix = f"{output_prefix}/{source_revision}/seed-{args.mixture_seed}"
        files = sorted(
            (path for path in output_dir.rglob("*.json") if path.is_file()),
            key=lambda path: path.relative_to(output_dir).as_posix(),
        )
        file_pairs = [
            (f"{remote_prefix}/{path.relative_to(output_dir).as_posix()}", path)
            for path in files
        ]
        repo_head = api.repo_info(repo_id=args.repo_id, repo_type="dataset").sha
        if _remote_snapshot_matches(
            api,
            repo_id=args.repo_id,
            revision=repo_head,
            file_paths=file_pairs,
        ):
            print(json.dumps(receipt.as_dict(), sort_keys=True, indent=2))
            return 0
        operations = [
            CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=str(local_path))
            for remote_path, local_path in file_pairs
        ]
        api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=(
                f"Finalize ICGS generation views at {source_revision} "
                f"(mixture seed {args.mixture_seed})"
            ),
            parent_commit=repo_head,
        )
        print(json.dumps(receipt.as_dict(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"generation view finalization failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
