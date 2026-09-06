"""Hash-bound published profile. No binary/reference checkout is needed at runtime."""
from dataclasses import replace
from importlib.resources import files
import hashlib
import json
from pathlib import Path

PUBLISHED_SHA256='119fa871091c7082b98d8a795dd80eca38295c4b7ab454e1f88549194bd4a4a5'


def published_config(*, device='cuda', num_demos=2, diffusion_steps=4):
    from icgs.configuration.defaults import from_legacy
    payload=json.loads(files('icgs.artifacts').joinpath('profiles/vv19-119fa871.json').read_text())
    config=from_legacy(payload['legacy_config'])
    return replace(config, name='instant_policy_published_vv19_119fa871',
                   runtime=replace(config.runtime,device=device,compile_models=False,live_voxel_size=.01,cache_context=False),
                   graph=replace(config.graph,num_demos=num_demos),
                   sampling=replace(config.sampling,steps=diffusion_steps)).validate()


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def load_published_policy(checkpoint, *, device='cuda', num_demos=2, diffusion_steps=4):
    """Strictly load the verified published model; unsupported artifacts fail."""
    import torch
    from icgs.composition import build_policy
    from icgs.artifacts.checkpoints import load_state_dict_compatible
    path=Path(checkpoint).expanduser().resolve()
    if path.is_dir(): path=path/'model.pt'
    config=published_config(device=device,num_demos=num_demos,diffusion_steps=diffusion_steps)
    digest=sha256_file(path)
    if digest!=PUBLISHED_SHA256: raise ValueError(f'unknown published checkpoint hash: {digest}')
    artifact=torch.load(path,map_location='cpu',weights_only=True)
    policy=build_policy(config)
    policy.checkpoint_report=load_state_dict_compatible(policy.network,artifact['state_dict'],strict=True)
    policy.eval()
    policy.network.requires_grad_(False)
    policy.artifact_sha256=digest
    return policy
