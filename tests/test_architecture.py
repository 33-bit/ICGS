"""Accepted local dependency closure; does not inspect third-party package internals."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CORE = ('ip.types', 'ip.geometry', 'ip.actions', 'ip.policy', 'ip.algorithms',
        'ip.configs.structured', 'ip.configs.original', 'ip.models.denoiser',
        'ip.models.scene_encoder', 'ip.models.graph_rep', 'ip.models.graph_transformer',
        'ip.models.backbone', 'ip.models.embeddings')
FORBIDDEN = ('rlbench', 'pyrep', 'wandb', 'lightning', 'pytorch_lightning', 'argparse',
             'ip.training', 'ip.evaluation', 'ip.train', 'ip.eval', 'ip.deployment',
             'ip.prepare_data', 'ip.checkpoints', 'ip.composition', 'ip.models.model',
             'ip.models.diffusion', 'ip.models.occupancy_net', 'ip.environments')


def matches(module, prefixes):
    return any(module == prefix or module.startswith(prefix + '.') for prefix in prefixes)


def imports(source, module, package=False):
    result = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parent = module.split('.') if package else module.split('.')[:-1]
            if node.level:
                parent = parent[:len(parent) - node.level + 1]
                base = '.'.join(parent + ([node.module] if node.module else []))
            else:
                base = node.module or ''
            if base:
                result.add(base)
                result.update(base + '.' + alias.name for alias in node.names if alias.name != '*')
    return result


def violations(graph, roots, forbidden):
    result = []
    for root in roots:
        queue, seen = [(root, [root])], set()
        while queue:
            module, chain = queue.pop()
            if module in seen:
                continue
            seen.add(module)
            if matches(module, forbidden):
                result.append(' -> '.join(chain))
                continue
            for dependency in graph.get(module, ()):
                queue.append((dependency, chain + [dependency]))
                # Importing a submodule executes local parent package initializers.
                parts = dependency.split('.')
                for length in range(1, len(parts)):
                    parent = '.'.join(parts[:length])
                    if parent in graph:
                        queue.append((parent, chain + [parent]))
    return result


class ArchitectureTests(unittest.TestCase):
    def test_real_core_dependency_closure(self):
        graph = {}
        for path in (ROOT / 'ip').rglob('*.py'):
            parts = list(path.relative_to(ROOT).with_suffix('').parts)
            package = parts[-1] == '__init__'
            if package:
                parts.pop()
            name = '.'.join(parts)
            graph[name] = imports(path.read_text(), name, package)
        roots = [name for name in graph if matches(name, CORE)]
        self.assertGreater(len(roots), 10, 'missing core modules is not a passing scan')
        self.assertEqual(violations(graph, roots, FORBIDDEN), [], 'ADR 0002: move outer dependencies out of reusable core')
        pure = [name for name in roots if name not in ('ip.policy',) and not name.startswith('ip.algorithms')]
        self.assertEqual(violations(graph, pure, ('open3d',)), [])

    def test_guard_detects_indirect_and_initializer_violations(self):
        graph = {'ip.core': {'ip.shared.util'}, 'ip.shared.util': set(), 'ip.shared': {'wandb'}}
        self.assertTrue(violations(graph, ['ip.core'], ('wandb',)))
        graph['ip.shared'] = set()
        self.assertEqual(violations(graph, ['ip.core'], ('wandb',)), [])
        graph['ip.shared.util'] = {'rlbench.tasks'}
        self.assertTrue(violations(graph, ['ip.core'], ('rlbench',)))

    def test_relative_and_nested_import_resolution(self):
        targets = imports('from ..geometry import transform_pcd\ndef f():\n import rlbench.tasks\n', 'ip.models.leaf')
        self.assertIn('ip.geometry', targets)
        self.assertIn('rlbench.tasks', targets)


if __name__ == '__main__':
    unittest.main()
