"""Bounded repair of RLBench's missing declared import; preserve setup evidence."""

import json
import subprocess
from pathlib import Path


receipt = json.loads(Path('/content/icgs_storage_receipt.json').read_text())
output = Path(receipt['drive_file']).parent
with (output / 'environment-setup.log').open('a') as log:
    log.write('\nREPAIR: missing gymnasium import in installed RLBench\n')
    log.flush()
    result = subprocess.run([
        'uv', 'pip', 'install', '--python', '/content/icgs-data-env/bin/python',
        'gymnasium==1.2.3',
    ], stdout=log, stderr=subprocess.STDOUT, timeout=120)
if result.returncode:
    print((output / 'environment-setup.log').read_text()[-3000:])
    raise RuntimeError('Gymnasium dependency installation failed')
report_path = output / 'environment-setup.json'
report = json.loads(report_path.read_text())
report['repairs'] = ['Installed gymnasium==1.2.3 after ModuleNotFoundError']
report['packages'] = subprocess.check_output([
    'uv', 'pip', 'freeze', '--python', '/content/icgs-data-env/bin/python',
], text=True).splitlines()
report_path.write_text(json.dumps(report, indent=2) + '\n')
print('PASS: gymnasium dependency repair, evidence retained on Drive')
