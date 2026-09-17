"""Persist safe setup/handoff artifacts, report honest collection readiness.

Does not install ICGS, instantiate injected test doubles, or claim training data.
Source bundle is private Drive-only. Simulator distribution stays off public HF.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


receipt = json.loads(Path('/content/icgs_storage_receipt.json').read_text())
output = Path(receipt['drive_file']).parent
report = {
    'kind': 'collection_readiness_not_training_data',
    'status': 'BLOCKED_IMPLEMENTATION',
    'drive_write_read': 'PASS',
    'hugging_face_commit_readback': 'PASS',
    'python_environment': '/content/icgs-data-env/bin/python',
    'training_episodes': 0,
    'continuous_collection_running': False,
    'continuous_episode_upload_running': False,
    'remaining_implementation': [
        'Concrete P01 low-level controller over PyRep with measured timestamps',
        'Custom scenes and scripted experts for T06/T08/T09/T11/T13/T14',
        'Concrete asset-family/ancestry split bindings and seed demonstrations',
        'Camera/crop/gravity and predicate observability calibration',
        'Per-episode validated Drive-first HF-second publication integrated with collector',
    ],
    'no_silent_substitution': 'Upstream RLBench demo tasks are not approved ICGS program equivalents',
    'storage_run_directory': str(output),
    'hf_preflight_prefix': receipt['hf_path'].rsplit('/', 1)[0],
}
source = Path('/content/icgs-source.tar.gz')
if source.exists():
    destination = output / 'source-snapshot.tar.gz'
    shutil.copy2(source, destination)
    with destination.open('rb') as stream:
        report['source_snapshot_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    report['source_snapshot_scope'] = 'src, tests, pyproject only; no .env or credentials; Drive only'
sim_archive = Path('/content/icgs-simulator/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz')
if sim_archive.is_file():
    cache = output.parent / 'private-setup-cache'
    cache.mkdir(exist_ok=True)
    copied = cache / sim_archive.name
    if not copied.exists():
        shutil.copy2(sim_archive, copied)
    with copied.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != '512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8':
        raise RuntimeError('Drive simulator cache checksum differs from downloaded archive')
    report['private_simulator_cache_sha256'] = digest
    report['private_simulator_cache_bytes'] = copied.stat().st_size
    # Upstream sources are cached privately for exact-revision setup reproduction.
    cached_sources = output / 'upstream-sources.tar.gz'
    subprocess.run([
        'tar', '--exclude=.git', '--exclude=__pycache__', '-czf', str(cached_sources),
        '-C', '/content/icgs-simulator', 'PyRep', 'RLBench',
    ], timeout=180, check=True)
report_path = output / 'collection-readiness.json'
report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
Path('/content/icgs_collection_readiness.json').write_text(report_path.read_text())
print(json.dumps(report, indent=2))
