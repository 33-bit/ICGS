"""Compact per-frame v2 sidecars into one NPZ per modality before HF upload."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
try:
    from PIL import Image
except Exception:
    Image = None


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def repack(root):
    root=Path(root)
    for ep in sorted(p for p in root.glob('*') if p.is_dir()):
        obs=ep/'observations'
        for modality in ('depth','masks','pointcloud'):
            d=obs/modality
            if not d.is_dir() or (d/'frames.npz').exists(): continue
            files=sorted(d.glob('*.npy'))
            if not files: continue
            arrays=[np.load(p,allow_pickle=False) for p in files]
            if modality=='pointcloud':
                offsets=[0]
                for a in arrays: offsets.append(offsets[-1]+len(a))
                points=np.concatenate(arrays,axis=0)
                np.savez_compressed(d/'frames.npz',points=points,offsets=np.asarray(offsets,dtype=np.int64))
            else:
                np.savez_compressed(d/'frames.npz',frames=np.stack(arrays,axis=0))
        rgb=obs/'rgb'
        if Image is not None and rgb.is_dir() and not (rgb/'frames.npz').exists():
            pngs=sorted(rgb.glob('*.png'))
            if pngs:
                np.savez_compressed(rgb/'frames.npz',frames=np.stack([np.asarray(Image.open(p)) for p in pngs],axis=0))
        # RGB PNGs are optional; preserve canonical video and avoid decoding
        # them here. The schema marks RGB as available via video_front in the
        # canonical archive when PNG compaction is unavailable.
        inv={}
        for f in sorted(p for p in ep.rglob('*') if p.is_file() and not p.name.endswith('.tmp')):
            inv[str(f.relative_to(ep))]={'sha256':sha(f),'bytes':f.stat().st_size}
        manifest=ep/'layout_manifest.json'
        if manifest.exists():
            data=json.loads(manifest.read_text())
            data['files']=inv
            manifest.write_text(json.dumps(data,indent=2)+'\n')
        print(ep.name, 'compacted')

for root in ('/content/dataset_g2_suite/episodes_v2','/content/dataset_g2_v2_b/episodes_v2'):
    if Path(root).exists(): repack(root)
