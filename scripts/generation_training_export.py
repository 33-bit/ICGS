#!/usr/bin/env python3
"""Publish a compact, pinned training export under the exact ``training`` prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile

from icgs.data.datasets.generation_view_index import validate_source_revision
from icgs.data.datasets.training_export import (
    TRAINING_PREFIX,
    export_training_views,
)


def _prefix(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value or path.as_posix() != value.rstrip("/"):
        raise ValueError(f"{field} must be a normalized repo-relative prefix")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError(f"{field} must be a normalized repo-relative prefix")
    return value.rstrip("/")


def _download_bytes(*, repo_id: str, filename: str, revision: str, cache_root: Path) -> bytes:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        revision=revision,
        cache_dir=str(cache_root),
    )
    path = Path(downloaded)
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"HF download did not produce a regular file: {filename}")
    return resolved.read_bytes()


def _write_download(root: Path, relative: str, content: bytes) -> None:
    path = root.joinpath(*PurePosixPath(relative).parts)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"downloaded view path traverses a symlink: {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"downloaded view conflict: {relative}")
    path.write_bytes(content)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repository ID")
    parser.add_argument("--source-prefix", required=True, help="Immutable archive prefix containing dataset_manifest.json")
    parser.add_argument("--source-revision", required=True, help="Pinned archive commit OID")
    parser.add_argument("--views-prefix", required=True, help="Prefix containing the twelve FINAL role/view JSON files")
    parser.add_argument("--views-revision", required=True, help="Pinned final-view commit OID")
    parser.add_argument("--output-prefix", default=TRAINING_PREFIX, help="Must remain exactly 'training'")
    parser.add_argument("--cache-dir", help="Owned temporary/cache directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output_prefix != TRAINING_PREFIX:
        raise ValueError(f"output-prefix must be exactly {TRAINING_PREFIX!r}")
    source_prefix = _prefix(args.source_prefix, "source-prefix")
    views_prefix = _prefix(args.views_prefix, "views-prefix")
    source_revision = validate_source_revision(args.source_revision)
    views_revision = validate_source_revision(args.views_revision)

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="icgs-training-export-", dir=args.cache_dir) as temporary:
        scratch = Path(temporary)
        hf_cache = scratch / "hf-cache"
        view_root = scratch / "final-views"
        for role in ("train", "validation", "evaluation"):
            for view in ("D_geom", "D_temporal", "D_dyn", "D_task"):
                relative = f"{role}/{view}.json"
                content = _download_bytes(
                    repo_id=args.repo_id,
                    filename=f"{views_prefix}/{relative}",
                    revision=views_revision,
                    cache_root=hf_cache,
                )
                _write_download(view_root, relative, content)

        # The source dataset manifest is metadata-only for export, but checking
        # its exact downloaded bytes proves that snapshots bind to the requested
        # archive revision before any target commit is attempted.
        source_bytes = _download_bytes(
            repo_id=args.repo_id,
            filename=f"{source_prefix}/dataset_manifest.json",
            revision=source_revision,
            cache_root=hf_cache,
        )
        source_manifest_path = scratch / "source" / "dataset_manifest.json"
        source_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        source_manifest_path.write_bytes(source_bytes)
        expected_manifest_sha = None
        for role in ("train", "validation", "evaluation"):
            for view in ("D_geom", "D_temporal", "D_dyn", "D_task"):
                payload = json.loads((view_root / role / f"{view}.json").read_text(encoding="utf-8"))
                if payload.get("source_revision") != source_revision or payload.get("source_prefix") != source_prefix:
                    raise ValueError(f"{role}/{view} is not bound to the requested source")
                expected_manifest_sha = expected_manifest_sha or payload.get("source_manifest_sha256")
                if payload.get("source_manifest_sha256") != expected_manifest_sha:
                    raise ValueError("final view snapshots disagree on source manifest hash")
        if expected_manifest_sha is None:
            raise ValueError("final view snapshots contain no source manifest hash")
        import hashlib

        if hashlib.sha256(source_bytes).hexdigest() != expected_manifest_sha:
            raise ValueError("source dataset manifest hash does not match final views")

        local_export = scratch / "training"
        receipt = export_training_views(
            view_root,
            repo_id=args.repo_id,
            source_revision=source_revision,
            views_revision=views_revision,
            source_prefix=source_prefix,
            output_dir=local_export,
            source_manifest=source_manifest_path,
            target_prefix=TRAINING_PREFIX,
        )
        files = sorted(path for path in local_export.rglob("*") if path.is_file())
        pairs = [
            (f"{TRAINING_PREFIX}/{path.relative_to(local_export).as_posix()}", path)
            for path in files
        ]
        head = api.repo_info(repo_id=args.repo_id, repo_type="dataset").sha
        expected_remote = {remote for remote, _ in pairs}
        try:
            remote_inventory = set(api.list_repo_files(repo_id=args.repo_id, repo_type="dataset", revision=head))
        except AttributeError:
            remote_inventory = {
                remote for remote, _ in pairs
                if api.file_exists(repo_id=args.repo_id, filename=remote, repo_type="dataset", revision=head)
            }
        existing_remote = {path for path in remote_inventory if path == TRAINING_PREFIX or path.startswith(f"{TRAINING_PREFIX}/")}
        if existing_remote:
            if existing_remote != expected_remote:
                raise ValueError("partial training prefix already exists")
            for remote, local in pairs:
                remote_bytes = _download_bytes(
                    repo_id=args.repo_id,
                    filename=remote,
                    revision=head,
                    cache_root=hf_cache,
                )
                if remote_bytes != local.read_bytes():
                    raise ValueError(f"immutable training export conflict: {remote}")
            output = receipt.as_dict()
            output.pop("output_directory", None)
            output["target_revision"] = head
            print(json.dumps(output, sort_keys=True, indent=2))
            return 0

        commit = api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
                for remote, local in pairs
            ],
            commit_message=f"Publish ICGS training export from {source_revision}",
            parent_commit=head,
        )
        target_revision = validate_source_revision(getattr(commit, "oid", getattr(commit, "commit_hash", "")))
        for remote, local in pairs:
            remote_bytes = _download_bytes(
                repo_id=args.repo_id,
                filename=remote,
                revision=target_revision,
                cache_root=hf_cache,
            )
            if remote_bytes != local.read_bytes():
                raise ValueError(f"training export verification mismatch: {remote}")
        output = receipt.as_dict()
        output.pop("output_directory", None)
        output["target_revision"] = target_revision
        print(json.dumps(output, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"training export failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
