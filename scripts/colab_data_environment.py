"""Bounded, explicit Colab simulator provisioning; no episode collection.

Requires colab_data_storage.py receipt. Installs upstream dependencies only in
the isolated VM. Logs and a software manifest are saved on mounted Drive.
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

UPSTREAM_REVISIONS = {
    "PyRep": "8f420be8064b1970aae18a9cfbc978dfb15747ef",
    "RLBench": "02720bba4c73fe02eb75df946b8791b806028a9d",
}
PINNED_COPPELIASIM_ARCHIVE_SHA256 = "512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8"


def pinned_source_commands(name: str, source: Path) -> list[list[str]]:
    """Return the exact detached-source setup commands for a pinned upstream."""
    revision = UPSTREAM_REVISIONS[name]
    repository = f"https://github.com/stepjam/{name}.git"
    return [
        ["git", "clone", "--no-checkout", repository, str(source)],
        ["git", "-C", str(source), "fetch", "--depth", "1", "origin", revision],
        ["git", "-C", str(source), "checkout", "--detach", revision],
    ]


def editable_install_command(python: str, source: Path) -> list[str]:
    """Install the verified source tree so imported module origins remain inspectable."""
    return ["uv", "pip", "install", "--python", python, "--editable", str(source)]


def main():
    receipt = json.loads(Path('/content/icgs_storage_receipt.json').read_text())
    output = Path(receipt['drive_file']).parent
    if not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Drive not mounted')
    root = Path('/content/icgs-simulator')
    root.mkdir(exist_ok=True)
    log_path = output / 'environment-setup.log'
    env = os.environ.copy()
    env.update({'DEBIAN_FRONTEND': 'noninteractive', 'GIT_TERMINAL_PROMPT': '0'})
    deadline = time.monotonic() + 1500
    results = []

    def run(label, args, timeout=300, cwd=None):
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError('Setup exceeded bounded 25-minute allowance')
        print(label, flush=True)
        with log_path.open('a') as log:
            log.write('\nSTEP: ' + label + '\n')
            log.flush()
            completed = subprocess.run(
                args, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                timeout=min(timeout, remaining), check=False,
            )
        results.append({'step': label, 'returncode': completed.returncode})
        if completed.returncode:
            # Public installer output contains no credential-bearing commands.
            print(log_path.read_text()[-6000:], flush=True)
            raise RuntimeError(f'{label} failed; log retained on Drive')

    report = {'kind': 'simulator_provisioning_not_training_data', 'results': results}
    try:
        run('create-supported-python', ['uv', 'venv', '--python', '3.11', '/content/icgs-data-env'])
        python = '/content/icgs-data-env/bin/python'
        run('system-package-index', ['apt-get', 'update', '-qq'])
        run('headless-render-dependencies', [
            'apt-get', 'install', '-y', '-qq', 'libgl1', 'libglu1-mesa',
            'libxcb-xinerama0', 'libxkbcommon-x11-0', 'libxcb-cursor0', 'libegl1',
            'xauth', 'xvfb', 'libxrender1', 'libxi6', 'libxrandr2',
        ])
        run('python-data-dependencies', [
            'uv', 'pip', 'install', '--python', python,
            'numpy==1.26.4', 'scipy==1.14.1', 'cffi', 'setuptools', 'wheel',
            'pillow', 'pyquaternion', 'natsort', 'transforms3d', 'gymnasium==1.2.3',
            'huggingface-hub==0.26.2',
        ])
        archive = root / 'CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz'
        url = 'https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz'
        cache_archive = Path('/content/drive/MyDrive/ICGS-data-20260916/private-setup-cache/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz')
        if cache_archive.is_file():
            run('copy-cached-coppeliasim-4.1', ['cp', str(cache_archive), str(archive)])
        else:
            run('download-upstream-coppeliasim-4.1', [
                'curl', '--fail', '--location', '--retry', '2', '--max-time', '420',
                '--output', str(archive), url,
            ], timeout=450)
        with archive.open('rb') as stream:
            archive_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if archive_digest != PINNED_COPPELIASIM_ARCHIVE_SHA256:
            raise RuntimeError('Downloaded CoppeliaSim archive SHA256 differs from the pinned 4.1 distribution')
        sim = root / 'CoppeliaSim'
        sim.mkdir(exist_ok=True)
        run('extract-coppeliasim', ['tar', '-xJf', str(archive), '-C', str(sim), '--strip-components=1'])
        env['COPPELIASIM_ROOT'] = str(sim)
        env['LD_LIBRARY_PATH'] = str(sim) + ':' + env.get('LD_LIBRARY_PATH', '')
        env['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(sim)
        refs = {}
        for name, expected_revision in UPSTREAM_REVISIONS.items():
            source = root / name
            clone, fetch, checkout = pinned_source_commands(name, source)
            run('clone-' + name, clone)
            run('fetch-pinned-' + name, fetch)
            run('checkout-pinned-' + name, checkout)
            revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
            if revision != expected_revision:
                raise RuntimeError(f'{name} checkout mismatch: {revision!r} != {expected_revision!r}')
            refs[name] = revision
            run('install-editable-' + name, editable_install_command(python, source), timeout=420)
        report.update({
            'status': 'installed_not_physically_validated', 'python': python,
            'upstream_revisions': refs, 'coppeliasim_download': url,
            'upstream_installation_mode': 'editable detached source trees',
            'coppeliasim_sha256': archive_digest,
            'simulator_root': str(sim), 'training_episodes': 0,
        })
        report['packages'] = subprocess.check_output(['uv', 'pip', 'freeze', '--python', python], text=True).splitlines()
        print(json.dumps({k: v for k, v in report.items() if k != 'packages'}, indent=2), flush=True)
    except Exception as exc:
        report.update({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc), 'training_episodes': 0})
        raise
    finally:
        (output / 'environment-setup.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
