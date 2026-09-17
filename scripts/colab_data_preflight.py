"""Read-only Colab hardware/storage/dependency check; never collects data.

Execute using `colab exec -s SESSION -f scripts/colab_data_preflight.py`.
No credentials or environment variable values are printed.
"""

import importlib.util
import json
import platform
import shutil
import subprocess
from pathlib import Path


def main():
    drive = Path('/content/drive/MyDrive')
    usage = shutil.disk_usage('/content')
    packages = ('rlbench', 'pyrep', 'numpy', 'torch', 'huggingface_hub')
    report = {
        'kind': 'environment_preflight_not_training_data',
        'platform': platform.platform(),
        'python': platform.python_version(),
        'drive_mounted': drive.is_dir(),
        'disk_free_bytes': usage.free,
        'packages': {name: importlib.util.find_spec(name) is not None for name in packages},
        'executables': {name: bool(shutil.which(name)) for name in ('nvidia-smi', 'Xvfb', 'git', 'uv')},
    }
    if shutil.which('nvidia-smi'):
        proc = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=15, check=False,
        )
        report['gpu'] = proc.stdout.strip() if proc.returncode == 0 else 'query_failed'
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
