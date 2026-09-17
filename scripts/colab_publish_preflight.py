"""Push allowlisted closed setup evidence from Drive; never publish credentials.

This is a finite preflight publication, not a continuous training-data uploader.
"""

import hashlib
import json
from pathlib import Path


def main():
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    receipt = json.loads(Path('/content/icgs_storage_receipt.json').read_text())
    folder = Path(receipt['drive_file']).parent
    prefix = receipt['hf_path'].rsplit('/', 1)[0]
    token = Path('/content/.icgs_hf_token').read_text().strip()
    names = (
        'environment-setup.json', 'environment-setup.log',
        'simulator-readiness.json', 'simulator-readiness.npz',
        'simulator-readiness.log', 'collection-readiness.json',
    )
    files = [folder / name for name in names if (folder / name).is_file()]
    history = folder / 'attempt-history'
    if history.is_dir():
        files.extend(sorted(history.glob('*-simulator-readiness.log')))
        files.extend(sorted(history.glob('*-simulator-readiness.json')))
    if not files:
        raise RuntimeError('No closed allowlisted preflight artifacts found')
    for path in files:
        if path.is_symlink() or path.stat().st_size > 20 * 1024 * 1024:
            raise RuntimeError('Unexpected preflight artifact type/size')
        if token.encode() in path.read_bytes():
            raise RuntimeError('Credential detected in artifact; publication refused')
    api = HfApi(token=token)
    info = api.create_commit(
        repo_id='33bit/icgs', repo_type='dataset',
        operations=[CommitOperationAdd(path_in_repo=f'{prefix}/{p.relative_to(folder)}', path_or_fileobj=str(p)) for p in files],
        commit_message='Record bounded simulator/storage readiness; zero training episodes',
    )
    hashes = {}
    for path in files:
        downloaded = hf_hub_download(
            repo_id='33bit/icgs', repo_type='dataset', filename=f'{prefix}/{path.relative_to(folder)}',
            revision=info.oid, token=token,
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Remote artifact checksum differs from Drive')
        hashes[str(path.relative_to(folder))] = digest
    result = {'commit': info.oid, 'prefix': prefix, 'sha256': hashes, 'training_episodes': 0}
    (folder / 'publication-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
