"""Single-package migration gates; no false-green empty source root."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SrcPackageTests(unittest.TestCase):
    def test_actual_src_contains_runtime_and_no_legacy_imports(self):
        root = ROOT / 'src/icgs'
        self.assertTrue((root/'__init__.py').is_file(), 'missing canonical src/icgs')
        files = list(root.rglob('*.py'))
        self.assertGreater(len(files), 20)
        for path in files:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or '']
                else:
                    continue
                for name in names:
                    self.assertFalse(name == 'ip' or name.startswith('ip.') or name == 'instant_policy', str(path))

    def test_old_runtime_is_removed_without_deleting_checkpoint(self):
        self.assertFalse(list((ROOT/'ip').rglob('*.py')), 'legacy Python runtime still exists')
        self.assertTrue((ROOT/'pyproject.toml').is_file())


if __name__ == '__main__':
    unittest.main()
