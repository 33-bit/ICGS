"""Drive-first, Hugging Face second uploader for exploratory RLBench data.

Enforces verified SHA-256 checksums, idempotent resumption, strict prefix isolation
under 'exploratory/...', and strict rejection of credentials, tokens, or source snapshots.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
from typing import Any, Sequence

logger = logging.getLogger(__name__)

FORBIDDEN_FILE_PATTERNS: tuple[str, ...] = (
    ".env*",
    "*.token*",
    "*.key",
    "*.secret*",
    "*.pem",
    "*.py",
    "*.pyc",
    "__pycache__*",
    ".git*",
    "credentials*",
)

ALLOWED_EXPLORATORY_EXTENSIONS: tuple[str, ...] = (
    ".json",
    ".npz",
    ".npy",
    ".md",
    ".txt",
)


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a local file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


try:
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError
except ImportError:
    EntryNotFoundError = None  # type: ignore[assignment, misc]
    RepositoryNotFoundError = None  # type: ignore[assignment, misc]
    HfHubHTTPError = None  # type: ignore[assignment, misc]


def _is_hf_not_found(exc: BaseException) -> bool:
    """Return True if and only if the exception represents a documented 404/not-found.

    Auth (401/403), network timeouts, connection errors, and 500 server errors return False
    so they propagate and fail closed.
    """
    if isinstance(exc, FileNotFoundError):
        return True
    if EntryNotFoundError is not None and isinstance(exc, EntryNotFoundError):
        return True
    if RepositoryNotFoundError is not None and isinstance(exc, RepositoryNotFoundError):
        return True
    if HfHubHTTPError is not None and isinstance(exc, HfHubHTTPError):
        resp = getattr(exc, "response", None)
        if resp is not None and getattr(resp, "status_code", None) == 404:
            return True
    exc_name = type(exc).__name__
    if exc_name in ("EntryNotFoundError", "RepositoryNotFoundError", "LocalEntryNotFoundError"):
        return True
    return False


def is_forbidden_file(path: Path) -> bool:
    """Return True if filename matches any credential, token, or source pattern."""
    name = path.name.lower()
    for pattern in FORBIDDEN_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


class ExploratoryUploader:
    """Manages two-tier replication of local exploratory collection artifacts.

    Tier 1 (Synchronous Drive-First): Mirrors verified episode archives to mounted Google Drive.
    Tier 2 (HF-Second): Replicates verified archives to Hugging Face dataset under 'exploratory/...'.
    """

    def __init__(
        self,
        local_root: str | Path,
        *,
        drive_root: str | Path | None = None,
        hf_repo_id: str | None = None,
        hf_subfolder: str = "exploratory",
        hf_token: str | None = None,
        hf_client: Any = None,
    ) -> None:
        self.local_root = Path(local_root).resolve()
        self.drive_root = Path(drive_root).resolve() if drive_root is not None else None
        self.hf_repo_id = str(hf_repo_id).strip() if hf_repo_id else None
        self.hf_subfolder = str(hf_subfolder).strip().strip("/") or "exploratory"
        self.hf_token = hf_token
        self._hf_client = hf_client
        if self.hf_subfolder != "exploratory" and not self.hf_subfolder.startswith("exploratory/"):
            raise ValueError("exploratory publication must remain under the isolated 'exploratory/' prefix")

    def _require_publication_targets(self) -> None:
        if self.drive_root is None:
            raise RuntimeError("Drive-first publication requires an explicit mounted Drive root")
        mounted_root = Path("/content/drive/MyDrive")
        if mounted_root.is_dir():
            try:
                self.drive_root.relative_to(mounted_root.resolve())
            except ValueError as exc:
                raise RuntimeError("Drive root must be a descendant of /content/drive/MyDrive") from exc
        elif Path("/content").is_dir():
            raise RuntimeError("Google Drive is not mounted under /content/drive/MyDrive")
        if not self.hf_repo_id:
            raise RuntimeError("HF-second publication requires an explicit dataset repository")

    def verify_publication_preflight(self, *, probe_drive: bool = True, verify_hf: bool = True) -> dict[str, Any]:
        """Explicit preflight target verification before any simulator construction or demo generation.

        Checks:
        1. Drive target containment, directory existence, no symlink escape, and safe isolated write/read/delete probe.
        2. HF authentication and repository accessibility capability check without any public dummy commit.
           Note: Read-only check confirms authenticated connectivity and repository presence;
           write permissions are validated upon commit.
        """
        self._require_publication_targets()
        logger.info("Executing publication preflight (Drive probe=%s, HF verification=%s)", probe_drive, verify_hf)
        report: dict[str, Any] = {}

        if probe_drive:
            if self.drive_root is None:
                raise RuntimeError("Drive root target is required for publication preflight")
            drive_root = Path(self.drive_root).resolve()
            if not drive_root.is_dir():
                raise FileNotFoundError(f"Drive root does not exist or is not a directory: {drive_root}")
            if drive_root.is_symlink():
                raise ValueError(f"Drive root cannot be a symlink: {drive_root}")

            # Safe isolated write/read/delete probe
            probe_dir = drive_root / ".icgs_preflight_probe"
            probe_id = secrets.token_hex(8)
            probe_file = probe_dir / f"probe_{probe_id}.tmp"
            probe_data = f"icgs_probe_{probe_id}".encode("utf-8")

            try:
                probe_dir.mkdir(parents=True, exist_ok=True)
                probe_file.write_bytes(probe_data)
                read_data = probe_file.read_bytes()
                if read_data != probe_data:
                    raise IOError("Drive probe readback mismatch")
            except Exception as exc:
                raise IOError(f"Drive target write/read probe failed: {exc}") from exc
            finally:
                try:
                    if probe_file.exists():
                        probe_file.unlink()
                except OSError:
                    pass
                try:
                    if probe_dir.exists():
                        probe_dir.rmdir()
                except OSError:
                    pass

            report["drive_target"] = "PASS"
            report["drive_root"] = str(drive_root)

        if verify_hf:
            if not self.hf_repo_id:
                raise RuntimeError("HF repository ID is required for publication preflight")

            api = self._hf_client
            if api is not None:
                if callable(getattr(api, "get_token_permission", None)):
                    perm = api.get_token_permission(self.hf_token)
                    if perm == "read":
                        raise PermissionError(
                            f"Hugging Face token has 'read' permission only; write permission is required to publish to '{self.hf_repo_id}'"
                        )
                    report["hf_token_permission"] = perm
                if callable(getattr(api, "whoami", None)):
                    try:
                        who = api.whoami(token=self.hf_token)
                        if isinstance(who, dict) and "name" in who:
                            report["hf_authenticated_user"] = who["name"]
                    except Exception:
                        pass
                if callable(getattr(api, "auth_check", None)):
                    api.auth_check(self.hf_repo_id, repo_type="dataset", token=self.hf_token)
                elif callable(getattr(api, "verify_preflight", None)):
                    api.verify_preflight(self.hf_repo_id)
                elif callable(getattr(api, "repo_info", None)):
                    api.repo_info(repo_id=self.hf_repo_id, repo_type="dataset")
            else:
                if not self.hf_token:
                    raise RuntimeError("HF token is missing; authenticated publication preflight cannot proceed")
                try:
                    from huggingface_hub import HfApi  # type: ignore
                    client = HfApi(token=self.hf_token)
                    if hasattr(client, "get_token_permission"):
                        perm = client.get_token_permission(token=self.hf_token)
                        if perm == "read":
                            raise PermissionError(
                                f"Hugging Face token has 'read' permission only; write permission is required to publish to '{self.hf_repo_id}'"
                            )
                        report["hf_token_permission"] = perm
                    if hasattr(client, "whoami"):
                        try:
                            who = client.whoami(token=self.hf_token)
                            if isinstance(who, dict) and "name" in who:
                                report["hf_authenticated_user"] = who["name"]
                        except Exception:
                            pass
                    if hasattr(client, "auth_check"):
                        client.auth_check(self.hf_repo_id, repo_type="dataset", token=self.hf_token)
                    else:
                        client.repo_info(repo_id=self.hf_repo_id, repo_type="dataset")
                except Exception as exc:
                    if isinstance(exc, PermissionError):
                        raise
                    if _is_hf_not_found(exc):
                        raise FileNotFoundError(f"Hugging Face dataset repository not found: '{self.hf_repo_id}'") from exc
                    raise RuntimeError(f"Hugging Face publication preflight failed: {exc}") from exc

            report["hf_target"] = "PASS"
            report["hf_repo_id"] = self.hf_repo_id
            report["hf_write_note"] = (
                "read/auth accessibility and token capability verified without dummy public commit; "
                "write permission cannot be fully guaranteed until commit creation"
            )

        return report

    def _sync_file_to_drive(self, local_file: Path, relative_path: Path, *, is_immutable: bool = True) -> bool:
        """Sync a single file to Google Drive with SHA-256 integrity verification."""
        if self.drive_root is None:
            return False

        if is_forbidden_file(local_file):
            raise ValueError(f"Refusing to mirror forbidden file {local_file.name} to Drive")

        target_file = self.drive_root / relative_path
        local_sha = compute_file_sha256(local_file)

        # Idempotent skip: check if target already exists with identical checksum
        if target_file.is_file():
            target_sha = compute_file_sha256(target_file)
            if local_sha == target_sha:
                return True
            if is_immutable:
                raise ValueError(f"immutable episode artifact differs on Drive: {relative_path}")

        target_file.parent.mkdir(parents=True, exist_ok=True)
        # Copy to temporary file first then replace
        tmp_target = target_file.parent / f".{target_file.name}.tmp"
        shutil.copy2(local_file, tmp_target)
        tmp_sha = compute_file_sha256(tmp_target)
        if tmp_sha != local_sha:
            if tmp_target.exists():
                tmp_target.unlink()
            raise IOError(f"Checksum mismatch copying {local_file} to Drive: {tmp_sha} != {local_sha}")

        os.replace(tmp_target, target_file)
        return True

    def _upload_file_to_hf(self, local_file: Path, relative_path: Path) -> bool:
        """Reject unsafe one-file publication; callers must use one verified commit."""
        del local_file, relative_path
        raise RuntimeError("single-file HF upload is forbidden; publish a complete episode via one verified commit")

    def _read_hf_bytes_optional(self, path_in_repo: str, revision: str = "main") -> bytes | None:
        """Read bytes from HF Hub, returning None ONLY if documented not-found.

        Auth, network timeout, connection, and server errors propagate.
        """
        try:
            return self._read_hf_bytes(path_in_repo, revision)
        except Exception as exc:
            if _is_hf_not_found(exc):
                return None
            raise

    def _get_head_commit(self, revision: str = "main") -> str | None:
        """Resolve current head commit SHA for target repository and revision."""
        api = self._hf_client
        if api is not None:
            if callable(getattr(api, "get_head_commit", None)):
                return api.get_head_commit(revision)
            if callable(getattr(api, "repo_info", None)):
                info = api.repo_info(repo_id=self.hf_repo_id, repo_type="dataset", revision=revision)
                return getattr(info, "sha", None)
            if hasattr(api, "head"):
                return getattr(api, "head")
            if hasattr(api, "revisions") and api.revisions:
                return api.revisions[-1]
            return None

        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.hf_token)
            info = api.repo_info(repo_id=self.hf_repo_id, repo_type="dataset", revision=revision)
            return getattr(info, "sha", None)
        except Exception as exc:
            if _is_hf_not_found(exc):
                return None
            raise

    def _hf_commit(
        self,
        files: Sequence[tuple[Path, Path]],
        *,
        is_immutable: bool = True,
        parent_commit: str | None = None,
    ) -> str:
        """Submit all episode artifacts as one commit and return its immutable revision."""
        self._require_publication_targets()

        resolved_parent = parent_commit if parent_commit is not None else self._get_head_commit("main")
        read_rev = resolved_parent if resolved_parent is not None else "main"

        # Immutable conflict check: verify existing remote artifacts do not conflict
        for local_file, rel in files:
            remote_path = f"{self.hf_subfolder}/{rel.as_posix()}"
            remote_bytes = self._read_hf_bytes_optional(remote_path, read_rev)
            if remote_bytes is not None:
                local_bytes = local_file.read_bytes()
                if is_immutable and remote_bytes != local_bytes:
                    raise ValueError(f"immutable episode artifact differs on HF: {rel.as_posix()}")

        api = self._hf_client
        operations: list[Any] = []
        if api is None:
            try:
                from huggingface_hub import CommitOperationAdd, HfApi  # type: ignore
                api = HfApi(token=self.hf_token)
                operations = [
                    CommitOperationAdd(
                        path_in_repo=f"{self.hf_subfolder}/{rel.as_posix()}",
                        path_or_fileobj=str(local),
                    )
                    for local, rel in files
                ]
            except ImportError as exc:
                raise RuntimeError("huggingface_hub is required for HF publication") from exc
        else:
            operations = [
                {
                    "path_in_repo": f"{self.hf_subfolder}/{rel.as_posix()}",
                    "path_or_fileobj": str(local),
                }
                for local, rel in files
            ]
        episode_label = files[0][1].parts[1] if len(files[0][1].parts) > 1 else "manifest"
        commit_kwargs: dict[str, Any] = {
            "repo_id": self.hf_repo_id,
            "repo_type": "dataset",
            "operations": operations,
            "commit_message": f"Publish exploratory artifact {episode_label}",
        }
        if resolved_parent is not None:
            commit_kwargs["parent_commit"] = resolved_parent

        result = api.create_commit(**commit_kwargs)
        revision = result.get("commit_hash") if isinstance(result, dict) else getattr(result, "oid", None)
        if not isinstance(revision, str) or not revision:
            raise RuntimeError("HF publication did not return an immutable commit revision")
        return revision

    def _read_hf_bytes(self, path_in_repo: str, revision: str) -> bytes:
        api = self._hf_client
        if api is not None and callable(getattr(api, "readback", None)):
            res = api.readback(path_in_repo, revision)
            if res is None:
                if EntryNotFoundError is not None:
                    raise EntryNotFoundError(f"File {path_in_repo} not found in revision {revision}")
                raise FileNotFoundError(f"File {path_in_repo} not found in revision {revision}")
            return bytes(res)
        if api is None:
            from huggingface_hub import hf_hub_download  # type: ignore
            downloaded = hf_hub_download(
                repo_id=self.hf_repo_id,
                repo_type="dataset",
                filename=path_in_repo,
                revision=revision,
                token=self.hf_token,
            )
            return Path(downloaded).read_bytes()
        raise RuntimeError("HF client must provide readback(path, revision) for verified publication")

    def sync_episode(self, episode_id: str) -> dict[str, Any]:
        """Sync all artifacts of one episode to Drive first, then Hugging Face."""
        self._require_publication_targets()
        if (
            not isinstance(episode_id, str)
            or not episode_id.strip()
            or episode_id in (".", "..")
            or "/" in episode_id
            or "\\" in episode_id
        ):
            raise ValueError(f"invalid immutable episode id: {episode_id!r}")
        episodes_root = (self.local_root / "episodes").resolve()
        ep_dir = (episodes_root / episode_id).resolve()
        try:
            ep_dir.relative_to(episodes_root)
        except ValueError as exc:
            raise ValueError("episode path escapes local root") from exc
        if not ep_dir.is_dir():
            raise FileNotFoundError(f"Episode directory not found: {ep_dir}")
        if ep_dir.is_symlink():
            raise ValueError(f"Episode directory is a symlink: {ep_dir}")

        # Reject forbidden files immediately across the episode directory tree
        for root, _, filenames in os.walk(ep_dir):
            for file in filenames:
                file_path = Path(root) / file
                if is_forbidden_file(file_path):
                    raise ValueError(f"Refusing to mirror forbidden file {file_path.name}")

        manifest_file = ep_dir / "manifest.json"
        if manifest_file.is_symlink():
            raise ValueError(f"Episode manifest is a symlink: {manifest_file}")
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Episode manifest not found: {manifest_file}")
        if manifest_file.resolve().parent != ep_dir.resolve():
            raise ValueError(f"Episode manifest escapes episode directory: {manifest_file}")

        try:
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Episode manifest is not valid JSON: {manifest_file}") from exc

        # Validate complete archive before upload
        from icgs.data.archives import read_episode_archive
        read_episode_archive(manifest_file)

        shards = manifest_data.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"Episode {episode_id} manifest contains no valid shards")

        # Derive exact allowlist of filenames from validated archive manifest
        allowed_filenames: set[str] = {"manifest.json"}
        for s_info in shards:
            if not isinstance(s_info, dict):
                raise ValueError(f"malformed shard record in episode manifest: {s_info}")
            shard_name = s_info.get("filename")
            if not isinstance(shard_name, str):
                raise ValueError(f"malformed shard filename in episode manifest: {s_info}")
            if "/" in shard_name or "\\" in shard_name or ".." in shard_name:
                raise ValueError(f"unsafe path components in shard filename: {shard_name!r}")
            if not re.match(r"^shard_\d{4}\.npz$", shard_name):
                raise ValueError(f"unauthorized shard filename pattern: {shard_name!r}")
            shard_path = ep_dir / shard_name
            if shard_path.is_symlink():
                raise ValueError(f"shard is a symlink: {shard_name}")
            if not shard_path.is_file():
                raise FileNotFoundError(f"referenced shard not found: {shard_path}")
            if shard_path.resolve().parent != ep_dir.resolve():
                raise ValueError(f"shard escapes episode directory: {shard_name}")
            allowed_filenames.add(shard_name)

        meta = manifest_data.get("metadata")
        if isinstance(meta, dict):
            for aux_key in ("video_front", "telemetry"):
                aux_val = meta.get(aux_key)
                if isinstance(aux_val, str) and aux_val:
                    if "/" in aux_val or "\\" in aux_val or ".." in aux_val:
                        raise ValueError(f"unsafe path in metadata {aux_key}: {aux_val!r}")
                    aux_path = ep_dir / aux_val
                    if aux_path.is_symlink():
                        raise ValueError(f"auxiliary file is a symlink: {aux_val}")
                    if not aux_path.is_file():
                        raise FileNotFoundError(f"referenced auxiliary file not found: {aux_path}")
                    if aux_path.resolve().parent != ep_dir.resolve():
                        raise ValueError(f"auxiliary file escapes episode directory: {aux_val}")
                    allowed_filenames.add(aux_val)

        # Audit entire directory tree BEFORE ANY WRITE:
        # Strictly reject extra nested files, unsafe path components, and all symlink escapes
        for root, dirs, filenames in os.walk(ep_dir, followlinks=False):
            root_path = Path(root)
            if root_path.is_symlink():
                raise ValueError(f"symlink directory escape detected: {root_path}")
            if dirs:
                raise ValueError(f"nested directories are forbidden in episode archive: {dirs}")
            for file in filenames:
                file_path = root_path / file
                if file_path.is_symlink():
                    raise ValueError(f"symlink file escape detected: {file}")
                if file not in allowed_filenames:
                    raise ValueError(f"unauthorized file in episode archive: {file}")

        # Deterministic ordered file list restricted strictly to validated manifest and shards
        files: list[tuple[Path, Path]] = [
            (ep_dir / "manifest.json", Path(f"episodes/{episode_id}/manifest.json")),
        ]
        for shard_name in sorted(allowed_filenames - {"manifest.json"}):
            files.append((ep_dir / shard_name, Path(f"episodes/{episode_id}/{shard_name}")))

        synced_files = [rel.as_posix() for _, rel in files]

        # Drive is the authoritative first copy. Verify every byte before HF commit.
        for file_path, rel in files:
            self._sync_file_to_drive(file_path, rel)
            drive_copy = self.drive_root / rel
            if compute_file_sha256(file_path) != compute_file_sha256(drive_copy):
                raise IOError(f"Drive verification failed for immutable episode {episode_id}: {rel}")
        drive_status = True

        # Resolve head commit to enforce CAS concurrency guard
        head_sha = self._get_head_commit("main")
        revision = self._hf_commit(files, is_immutable=True, parent_commit=head_sha)
        for file_path, rel in files:
            remote_path = f"{self.hf_subfolder}/{rel.as_posix()}"
            expected = file_path.read_bytes()
            if self._read_hf_bytes(remote_path, revision) != expected:
                raise IOError(f"HF readback verification failed for immutable episode {episode_id}: {rel}")
        hf_status = True

        return {
            "episode_id": episode_id,
            "synced_files": synced_files,
            "drive_synced": drive_status,
            "hf_synced": hf_status,
            "drive_verified": drive_status,
            "hf_verified": hf_status,
            "hf_commit": revision,
        }

    def sync_manifest(self, manifest_path: str | Path | None = None) -> dict[str, Any]:
        """Sync dataset_manifest.json to Drive first, then Hugging Face.

        Requires all referenced episodes verified complete on storage targets first,
        and enforces safe atomic/version-checked monotonic index updates.
        """
        self._require_publication_targets()
        if manifest_path is None:
            manifest_file = self.local_root / "dataset_manifest.json"
        else:
            manifest_file = Path(manifest_path).resolve()
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_file}")
        try:
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError("dataset manifest is not valid JSON") from exc
        if not isinstance(manifest_data, dict) or manifest_data.get("dataset_track") != "exploratory":
            raise ValueError("refusing to publish an index without dataset_track='exploratory'")

        episodes = manifest_data.get("episodes", [])
        if not isinstance(episodes, list):
            raise ValueError("dataset manifest 'episodes' must be a list")

        # Resolve head commit before pre-verification to bind all reads and enforce CAS
        head_sha = self._get_head_commit("main")
        read_rev = head_sha if head_sha is not None else "main"

        # 1. Require all referenced episodes verified complete first
        from icgs.data.archives import read_episode_archive

        for ep in episodes:
            if not isinstance(ep, dict) or "episode_id" not in ep:
                raise ValueError(f"malformed episode entry in manifest: {ep}")
            ep_id = str(ep["episode_id"])
            ep_manifest_local = self.local_root / "episodes" / ep_id / "manifest.json"
            if not ep_manifest_local.is_file():
                raise FileNotFoundError(f"cannot publish index: referenced episode manifest missing: {ep_manifest_local}")

            # Verify local archive integrity
            read_episode_archive(ep_manifest_local)

            # Verify Drive mirror has complete verified episode manifest
            if self.drive_root is not None:
                drive_ep_manifest = self.drive_root / "episodes" / ep_id / "manifest.json"
                if not drive_ep_manifest.is_file():
                    raise ValueError(f"cannot publish index: referenced episode {ep_id} manifest missing on Drive")
                if compute_file_sha256(drive_ep_manifest) != compute_file_sha256(ep_manifest_local):
                    raise ValueError(f"cannot publish index: referenced episode {ep_id} manifest differs on Drive")

            # Verify HF mirror has verified episode manifest bound to resolved head
            remote_ep_path = f"{self.hf_subfolder}/episodes/{ep_id}/manifest.json"
            remote_ep_bytes = self._read_hf_bytes(remote_ep_path, read_rev)
            if remote_ep_bytes != ep_manifest_local.read_bytes():
                raise ValueError(f"cannot publish index: referenced episode {ep_id} manifest differs on HF")

        # 2. Remote index consistency checks (safe monotonic version-checked updates)
        rel = Path("dataset_manifest.json")
        new_ep_map = {e["episode_id"]: e.get("sha256") for e in episodes if isinstance(e, dict)}

        if self.drive_root is not None:
            drive_manifest_file = self.drive_root / rel
            if drive_manifest_file.is_file():
                try:
                    old_drive_data = json.loads(drive_manifest_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise ValueError("existing Drive manifest is not valid JSON") from exc
                old_ep_map = {e["episode_id"]: e.get("sha256") for e in old_drive_data.get("episodes", []) if isinstance(e, dict)}
                for old_id, old_sha in old_ep_map.items():
                    if old_id not in new_ep_map:
                        raise ValueError(f"index update conflicts: drops previously published episode {old_id} from Drive")
                    if old_sha != new_ep_map[old_id]:
                        raise ValueError(f"index update conflicts: sha256 mismatch for episode {old_id} on Drive")

        remote_manifest_path = f"{self.hf_subfolder}/{rel.as_posix()}"
        old_hf_bytes = self._read_hf_bytes_optional(remote_manifest_path, read_rev)
        if old_hf_bytes is not None:
            try:
                old_hf_data = json.loads(old_hf_bytes.decode("utf-8"))
            except Exception as exc:
                raise ValueError("existing HF manifest is not valid JSON") from exc
            old_hf_ep_map = {e["episode_id"]: e.get("sha256") for e in old_hf_data.get("episodes", []) if isinstance(e, dict)}
            for old_id, old_sha in old_hf_ep_map.items():
                if old_id not in new_ep_map:
                    raise ValueError(f"index update conflicts: drops previously published episode {old_id} from HF")
                if old_sha != new_ep_map[old_id]:
                    raise ValueError(f"index update conflicts: sha256 mismatch for episode {old_id} on HF")

        # 3. Publish to Drive (mutable index update with SHA-256 verification)
        drive_ok = None
        if self.drive_root is not None:
            drive_ok = self._sync_file_to_drive(manifest_file, rel, is_immutable=False)

        # 4. Publish to HF (mutable index update) with CAS parent_commit
        revision = self._hf_commit([(manifest_file, rel)], is_immutable=False, parent_commit=head_sha)

        # 5. Readback verify from HF at exact revision
        if self._read_hf_bytes(remote_manifest_path, revision) != manifest_file.read_bytes():
            raise IOError("HF manifest readback verification failed")
        hf_ok = True

        return {
            "manifest_file": str(manifest_file),
            "drive_synced": drive_ok,
            "hf_synced": hf_ok,
            "drive_verified": drive_ok is True,
            "hf_verified": hf_ok is True,
            "hf_commit": revision,
        }

    def sync_all(self) -> dict[str, Any]:
        """Sync all episodes in dataset manifest and the manifest itself."""
        manifest_file = self.local_root / "dataset_manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        ep_results = []
        for ep_entry in data.get("episodes", []):
            ep_id = ep_entry["episode_id"]
            ep_results.append(self.sync_episode(ep_id))

        manifest_result = self.sync_manifest()

        return {
            "episodes": ep_results,
            "manifest": manifest_result,
        }


__all__ = [
    "ALLOWED_EXPLORATORY_EXTENSIONS",
    "FORBIDDEN_FILE_PATTERNS",
    "ExploratoryUploader",
    "compute_file_sha256",
    "is_forbidden_file",
]
