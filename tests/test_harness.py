"""Behavioral tests for the static harness validator; fixtures never touch ip/."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_fast.py"


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if VALIDATOR.exists():
            spec = importlib.util.spec_from_file_location("validate_fast", VALIDATOR)
            cls.guard = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.guard)

    def setUp(self):
        self.assertTrue(VALIDATOR.is_file(), "Missing scripts/validate_fast.py")
        self.temp = tempfile.TemporaryDirectory(prefix="ip-harness-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.put("ip/__init__.py", "")
        self.put("AGENTS.md", "# Agents\n")

    def put(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_syntax_rejects_invalid_file_without_importing_valid_code(self):
        self.put("ip/good.py", "raise RuntimeError('must not execute')\n")
        self.assertEqual(self.guard.check_syntax(self.root), [])
        self.put("ip/bad.py", "def broken(:\n")
        errors = self.guard.check_syntax(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn("ip/bad.py", errors[0])
        self.assertIn("SYNTAX", errors[0])

    def test_links_accept_relative_images_anchors_and_reference_links(self):
        self.put("docs/a.md", "# Hello World\n\n## Hello World\n")
        self.put("media/a.gif", "fixture")
        self.put("README.md", '[guide](docs/a.md#hello-world)\n'
                 '[again](docs/a.md#hello-world-1)\n![image](media/a.gif)\n'
                 '[ref][guide]\n\n[guide]: docs/a.md#hello-world\n'
                 '[remote](https://example.invalid/a)\n'
                 '```markdown\n[example](missing.md)\n```\n')
        self.assertEqual(self.guard.check_links(self.root), [])

    def test_syntax_rejects_context_invalid_python(self):
        self.put("ip/bad.py", "return 1\n")
        self.assertEqual(len(self.guard.check_syntax(self.root)), 1)

    def test_links_reject_undefined_reference_label(self):
        self.put("README.md", "[guide][missing-label]\n")
        errors = self.guard.check_links(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn("undefined reference", errors[0])

    def test_links_reject_missing_file_anchor_and_reference_target(self):
        self.put("docs/a.md", "# Present\n")
        self.put("README.md", '[a](absent.md)\n[b](docs/a.md#absent)\n'
                 '[c][target]\n\n[target]: absent-ref.md\n')
        errors = self.guard.check_links(self.root)
        self.assertEqual(len(errors), 3)
        self.assertTrue(all("LINK" in error for error in errors))

    def test_links_reject_escape_and_accept_encoded_spaces(self):
        self.put("docs/with space.md", "# Space\n")
        self.put("README.md", '[ok](docs/with%20space.md#space)\n'
                 '[escape](../outside.md)\n')
        errors = self.guard.check_links(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn("outside repository", errors[0])

    def test_boundary_allows_runtime_dependencies_and_harness_to_runtime(self):
        self.put("ip/core.py", 'from ip.utils import common_utils\n'
                 'from lightning.pytorch.loggers import WandbLogger\n'
                 'checkpoint = "./checkpoints/model.pt"\n'
                 'message = "Read docs/ for guidance"\n')
        self.put("scripts/tool.py", "from ip import core\n")
        self.assertEqual(self.guard.check_boundary(self.root), [])

    def test_boundary_rejects_direct_nested_and_relative_harness_imports(self):
        self.put("ip/core.py", 'import docs.helper\n'
                 'def f():\n    from scripts.validate_fast import main\n'
                 'from ..tests import test_harness\n')
        errors = self.guard.check_boundary(self.root)
        self.assertEqual(len(errors), 3)
        self.assertTrue(all("HARNESS_BOUNDARY" in e for e in errors))
        self.assertTrue(all("0001-harness-boundary.md" in e for e in errors))

    def test_boundary_rejects_literal_dynamic_import_and_resource_path(self):
        self.put("ip/core.py", 'import importlib\n'
                 'importlib.import_module("scripts.validate_fast")\n'
                 'open("./docs/ARCHITECTURE.md")\n'
                 'p = ".agents/skills/onboard-repository/SKILL.md"\n')
        errors = self.guard.check_boundary(self.root)
        self.assertEqual(len(errors), 3)

    def test_boundary_rejects_path_parts_and_agent_entrypoint(self):
        self.put("ip/core.py", 'from pathlib import Path\n'
                 'Path("docs") / "ARCHITECTURE.md"\n'
                 'open("AGENTS.md")\n')
        self.assertEqual(len(self.guard.check_boundary(self.root)), 2)

    def test_empty_root_is_not_a_successful_validation(self):
        with tempfile.TemporaryDirectory(prefix="ip-empty-") as directory:
            self.assertTrue(self.guard.check_syntax(Path(directory)))
            self.assertTrue(self.guard.check_boundary(Path(directory)))

    def test_cli_reports_fail_and_nonzero_for_invalid_tree(self):
        self.put("ip/bad.py", "def broken(:\n")
        result = subprocess.run(
            [sys.executable, "-B", str(VALIDATOR), "--root", str(self.root)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL", result.stdout)
        self.assertIn("ip/bad.py", result.stdout)

    def test_cli_reports_pass_and_zero_for_valid_fixture(self):
        self.put("ip/good.py", "raise RuntimeError('never import runtime')\n")
        result = subprocess.run(
            [sys.executable, "-B", str(VALIDATOR), "--root", str(self.root)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS: Python syntax", result.stdout)
        self.assertIn("NOT RUN: harness self-tests", result.stdout)

    def test_cli_reports_boundary_failure_for_valid_python(self):
        self.put("ip/bad.py", "import scripts.validate_fast\n")
        result = subprocess.run(
            [sys.executable, "-B", str(VALIDATOR), "--root", str(self.root)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("PASS: Python syntax", result.stdout)
        self.assertIn("FAIL: Static harness boundary", result.stdout)
        self.assertIn("HARNESS_BOUNDARY ip/bad.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
