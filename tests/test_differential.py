"""Opt-in exact-method comparison against verified original source, not a second runtime."""
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

BASE = os.environ.get('IP_LEGACY_SOURCE_ROOT')


def legacy_definition(relative, name, namespace, owner=None):
    root = Path(__file__).resolve().parents[1]
    hashes = {line.split('  ', 1)[1]: line.split('  ', 1)[0]
              for line in (root / 'docs/baselines/upstream-files.sha256').read_text().splitlines()}
    path = Path(BASE) / relative
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != hashes[relative]:
        raise AssertionError(f'legacy source hash mismatch: {relative}')
    tree = ast.parse(source)
    members = tree.body
    if owner:
        members = next(node.body for node in members if isinstance(node, ast.ClassDef) and node.name == owner)
    definition = next(node for node in members if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@unittest.skipUnless(BASE, 'SKIPPED: set IP_LEGACY_SOURCE_ROOT to verified original source for differential tests')
class DifferentialTests(unittest.TestCase):
    def test_all_canonical_config_values_match_original_source(self):
        import runpy
        from icgs.configuration.defaults import instant_policy_original, to_legacy
        # Verify the independent source bytes before executing its torch/NumPy config.
        root = Path(__file__).resolve().parents[1]
        expected_hash = next(line.split()[0] for line in
                             (root / 'docs/baselines/upstream-files.sha256').read_text().splitlines()
                             if line.endswith('  ip/configs/base_config.py'))
        source = Path(BASE) / 'ip/configs/base_config.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected_hash)
        old = runpy.run_path(str(source))['config']
        actual = to_legacy(instant_policy_original())
        self.assertEqual(set(actual), set(old))
        for key, value in old.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            else:
                self.assertEqual(actual[key], value, key)

    def test_graph_structure_and_seeded_parameters(self):
        from torch_geometric.data import HeteroData
        from icgs.models.layers.embeddings import PositionalEncoder, SinusoidalPosEmb
        from icgs.models.graphs.ip_graph import GraphRep
        from icgs.configuration.defaults import instant_policy_original, to_legacy
        from dataclasses import replace
        c = instant_policy_original()
        c = replace(c, runtime=replace(c.runtime, batch_size=1, device='cpu'))
        old_cls = legacy_definition('ip/models/graph_rep.py', 'GraphRep',
                    dict(torch=torch, nn=torch.nn, HeteroData=HeteroData,
                         PositionalEncoder=PositionalEncoder, SinusoidalPosEmb=SinusoidalPosEmb))
        torch.manual_seed(12)
        old = old_cls(to_legacy(c)); old.initialise_graph()
        torch.manual_seed(12)
        new = GraphRep(c.graph, 1, 'cpu'); new.initialise_graph()
        self.assertEqual(old.state_dict().keys(), new.state_dict().keys())
        for key, value in old.state_dict().items():
            torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(old.graph.edge_types, new.graph.edge_types)
        for edge in old.graph.edge_types:
            torch.testing.assert_close(new.graph[edge].edge_index, old.graph[edge].edge_index, rtol=0, atol=0)

    def test_denoiser_forward_and_labels_match_original_methods(self):
        from test_composition import CompositionTests
        from torch_geometric.utils import to_dense_batch
        helper = CompositionTests(); helper.setUp()
        network = helper.build(helper.c, helper.factories)
        old_forward = legacy_definition('ip/models/model.py', 'forward',
                                       dict(torch=torch, to_dense_batch=to_dense_batch), owner='AGI')
        data = helper.sample()
        before = old_forward(network, data.clone())
        after = network(data.clone())
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        old_labels = legacy_definition('ip/models/model.py', 'get_labels', dict(torch=torch), owner='AGI')
        noisy = data.actions.clone(); noisy[...,0,3] = .01
        args = (data.actions, noisy, data.actions_grip.unsqueeze(-1), data.actions_grip.unsqueeze(-1))
        torch.testing.assert_close(network.get_labels(*args), old_labels(network, *args), rtol=0, atol=0)

    def test_sampler_matches_original_loop_with_controlled_schedule(self):
        from test_policy import PolicyTests, LinearSchedule
        from icgs.configuration.defaults import to_legacy
        from icgs.geometry.transforms import actions_to_transforms, transforms_to_actions, get_rigid_transforms
        helper = PolicyTests(); helper.setUp(); policy = helper.policy(custom=False)
        old_step = legacy_definition('ip/models/diffusion.py', 'test_step',
                    dict(torch=torch, actions_to_transforms=actions_to_transforms,
                         transforms_to_actions=transforms_to_actions, get_rigid_transforms=get_rigid_transforms),
                    owner='GraphDiffusion')
        config = to_legacy(helper.c); config['num_diffusion_iters_test'] = 2
        old = SimpleNamespace(config=config, model=policy.network, device='cpu',
                              normalizer=policy.network.codec, noise_scheduler=LinearSchedule(None))
        torch.manual_seed(7)
        original_actions, original_grips = old_step(old, helper.sample(), 0)
        torch.manual_seed(7)
        actual = policy.sampler.sample(policy.network, helper.sample())
        torch.testing.assert_close(actual.transforms, original_actions, rtol=0, atol=0)
        torch.testing.assert_close(actual.grips, original_grips, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
