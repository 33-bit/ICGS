"""Verify Drive/HF persistence for the named initial-data job; no training data.

Run remotely with colab exec after mounting Drive and securely transferring only
the HF token to /content/.icgs_hf_token (never to Drive or a public artifact).
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main():
    from huggingface_hub import HfApi, hf_hub_download

    drive = Path('/content/drive/MyDrive')
    if not drive.is_dir():
        raise RuntimeError('Drive must be mounted before creating any job artifact')
    token_path = Path('/content/.icgs_hf_token')
    token_path.chmod(0o600)
    token = token_path.read_text().strip()
    if not token:
        raise RuntimeError('HF credential missing')
    run_id = 'setup-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = drive / 'ICGS-data-20260916' / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        'schema_version': 1,
        'run_id': run_id,
        'kind': 'storage_preflight_not_training_data',
        'training_episodes': 0,
        'drive_relative_path': str(run_dir.relative_to(drive)),
        'hf_repo': '33bit/icgs',
        'session': 'icgs-initial-data-20260916',
        'physical_collection_started': False,
        'publication_rule': 'Drive first, verified HF commit second, index last',
    }
    data = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode()
    source = run_dir / 'storage-preflight.json'
    with source.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if source.read_bytes() != data:
        raise RuntimeError('Drive write/read verification failed')
    api = HfApi(token=token)
    if api.whoami().get('name') != '33bit':
        raise RuntimeError('HF account does not match intended owner')
    remote_path = f'preflight/{run_id}/storage-preflight.json'
    commit = api.upload_file(
        repo_id='33bit/icgs', repo_type='dataset', path_or_fileobj=str(source),
        path_in_repo=remote_path, commit_message=f'Verify initial-data storage: {run_id}',
    )
    downloaded = hf_hub_download(
        repo_id='33bit/icgs', repo_type='dataset', filename=remote_path,
        revision=commit.oid, token=token, force_download=True,
    )
    if Path(downloaded).read_bytes() != data:
        raise RuntimeError('HF committed bytes differ from Drive')
    receipt = {
        'drive_file': str(source), 'hf_path': remote_path,
        'commit': commit.oid, 'sha256': hashlib.sha256(data).hexdigest(),
        'drive_readback': 'PASS', 'hf_readback': 'PASS', 'training_episodes': 0,
    }
    (run_dir / 'storage-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    Path('/content/icgs_storage_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
