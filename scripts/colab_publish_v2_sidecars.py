"""Publish compact v2 sidecars for already-published canonical episodes."""
from pathlib import Path
import json
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

HF_REPO='33bit/icgs'
SUBFOLDER='primary_v2'
TOKEN=Path('/content/.icgs_hf_token').read_text().strip()
api=HfApi(token=TOKEN)

def keep(path: Path) -> bool:
    # The compact NPZ is canonical for v2. Skip the old per-frame files left in
    # staging so one batch does not exceed the Hub API request quota.
    if path.suffix in ('.png', '.npy') and path.parent.name in ('rgb','depth','masks','pointcloud'):
        return path.name == 'frames.npz'
    return path.is_file()

def publish(root: str):
    root=Path(root)
    if not root.is_dir(): return
    manifest_path=Path('/content/dataset_g2_suite/dataset_manifest.json') if 'suite' in root.parts else Path('/content/dataset_g2_v2_b/dataset_manifest.json')
    ids=[]
    if manifest_path.exists():
        ids=[x['episode_id'] for x in json.loads(manifest_path.read_text()).get('episodes',[])]
    if not ids:
        try:
            remote=hf_hub_download(repo_id=HF_REPO,repo_type='dataset',filename=f'{SUBFOLDER}/dataset_manifest.json',token=TOKEN,force_download=True)
            ids=[x['episode_id'] for x in json.loads(Path(remote).read_text()).get('episodes',[])]
        except Exception:
            ids=[]
    ops=[]
    for eid in ids:
        ep=root/eid
        if not ep.is_dir(): continue
        for f in ep.rglob('*'):
            if keep(f):
                ops.append(CommitOperationAdd(path_in_repo=f'{SUBFOLDER}/episodes/{eid}/{f.relative_to(ep)}',path_or_fileobj=str(f)))
    for base in ('metadata','programs'):
        d=root.parent/base if root.name=='episodes_v2' else None
        if d and d.is_dir():
            for f in d.rglob('*'):
                if f.is_file(): ops.append(CommitOperationAdd(path_in_repo=f'{SUBFOLDER}/{base}/{f.relative_to(d)}',path_or_fileobj=str(f)))
    if not ops:
        print('NO_SIDECARS'); return
    print('OPERATIONS',len(ops),'EPISODES',len(ids))
    result=api.create_commit(repo_id=HF_REPO,repo_type='dataset',operations=ops,commit_message=f'Publish compact v2 training sidecars ({len(ids)} episodes)')
    print('COMMIT',getattr(result,'oid',result))

publish('/content/dataset_g2_suite/episodes_v2')
