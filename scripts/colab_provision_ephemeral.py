"""Provision a disposable pinned RLBench environment without Drive.

This is an explicit user-authorized fallback when Colab Drive mount is
unavailable.  It writes only to the ephemeral VM and never handles credentials.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess


ROOT = Path('/content/icgs-ephemeral')
PYTHON = Path('/content/icgs-data-env/bin/python')
COPPELIASIM_SHA256 = '512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8'
UPSTREAM = {
    'PyRep': '8f420be8064b1970aae18a9cfbc978dfb15747ef',
    'RLBench': '02720bba4c73fe02eb75df946b8791b806028a9d',
}


def run(args: list[str], *, timeout: int = 900, env: dict[str, str] | None = None) -> None:
    print('$', ' '.join(args), flush=True)
    subprocess.run(args, check=True, timeout=timeout, env=env)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({'DEBIAN_FRONTEND': 'noninteractive', 'GIT_TERMINAL_PROMPT': '0'})
    if not PYTHON.exists():
        run(['uv', 'venv', '--python', '3.11', str(PYTHON.parent.parent)], timeout=180)
    run(['apt-get', 'update', '-qq'], timeout=300, env=env)
    run(['apt-get', 'install', '-y', '-qq', 'libgl1', 'libglu1-mesa', 'libxcb-xinerama0',
         'libxkbcommon-x11-0', 'libxcb-cursor0', 'libegl1', 'xauth', 'xvfb', 'libxrender1',
         'libxi6', 'libxrandr2'], timeout=600, env=env)
    run(['uv', 'pip', 'install', '--python', str(PYTHON), 'numpy==1.26.4', 'scipy==1.14.1',
         'cffi', 'setuptools', 'wheel', 'pillow', 'pyquaternion', 'natsort', 'transforms3d',
         'gymnasium==1.2.3', 'huggingface-hub==0.26.2'], timeout=900)

    archive = ROOT / 'CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz'
    run(['curl', '--fail', '--location', '--retry', '2', '--max-time', '900',
         '--output', str(archive),
         'https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz'], timeout=1000)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != COPPELIASIM_SHA256:
        raise RuntimeError(f'CoppeliaSim checksum mismatch: {digest}')
    sim = ROOT / 'CoppeliaSim'
    sim.mkdir(exist_ok=True)
    run(['tar', '-xJf', str(archive), '-C', str(sim), '--strip-components=1'], timeout=300)
    env.update({'COPPELIASIM_ROOT': str(sim), 'LD_LIBRARY_PATH': str(sim), 'QT_QPA_PLATFORM_PLUGIN_PATH': str(sim)})

    for name, revision in UPSTREAM.items():
        source = ROOT / name
        run(['git', 'clone', '--no-checkout', f'https://github.com/stepjam/{name}.git', str(source)], timeout=600)
        run(['git', '-C', str(source), 'fetch', '--depth', '1', 'origin', revision], timeout=600)
        run(['git', '-C', str(source), 'checkout', '--detach', revision], timeout=120)
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise RuntimeError(f'{name} revision mismatch: {actual}')
        run(['uv', 'pip', 'install', '--python', str(PYTHON), '--editable', str(source)], timeout=600, env=env)
    if Path('/content/ICGS').is_dir():
        run(['uv', 'pip', 'install', '--python', str(PYTHON), '--no-deps', '--editable', '/content/ICGS'], timeout=600, env=env)
    print('PROVISIONED', ROOT, PYTHON, sim, flush=True)


if __name__ == '__main__':
    main()
