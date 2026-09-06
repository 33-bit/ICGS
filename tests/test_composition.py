from dataclasses import replace
import importlib.util
import unittest
import torch


class TinyScene(torch.nn.Module):
    def __init__(self, config, value=1.):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))
        self.embd_dim = config.embd_dim

    def forward(self, x, pos, batch):
        # Two nodes per input cloud: an actual injectable representation, not a fake dependency.
        ids = torch.cat([torch.where(batch == b)[0][:2] for b in batch.unique(sorted=True)])
        return self.weight * torch.ones(len(ids), self.embd_dim), pos[ids], batch[ids]


class TinyStage(torch.nn.Module):
    def forward(self, x, edge_index, edge_attr):
        return {key: value[:, :128] for key, value in x.items()}


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('icgs.composition'), 'Missing composition layer')
        from icgs.configuration.defaults import instant_policy_original
        from icgs.composition import ComponentFactories, build_network
        self.build = build_network
        c = instant_policy_original()
        self.c = replace(c, scene=replace(c.scene, kind='tiny', embd_dim=128, pretrained=False),
                         graph=replace(c.graph, num_demos=1, traj_horizon=2, num_scene_nodes=2, embd_dim=128, pred_horizon=2),
                         backbone=replace(c.backbone, kind='tiny', hidden_dim=128),
                         action=replace(c.action, pred_horizon=2), runtime=replace(c.runtime, device='cpu', batch_size=1),
                         evaluation=replace(c.evaluation,execution_horizon=2))
        self.factories = ComponentFactories(scene={'tiny': lambda cfg: TinyScene(cfg)},
                                          backbone={'tiny': lambda cfg, inputs, edges: (TinyStage(), TinyStage(), TinyStage())})

    def sample(self):
        from torch_geometric.data import Data
        return Data(pos_demos=torch.arange(24, dtype=torch.float32).view(8,3)/100,
                    batch_demos=torch.tensor([0]*4+[1]*4),
                    pos_obs=torch.arange(12,dtype=torch.float32).view(4,3)/100,
                    batch_pos_obs=torch.zeros(4,dtype=torch.long),
                    graps_demos=torch.ones(1,1,2,1), demo_T_w_es=torch.eye(4).repeat(1,1,2,1,1),
                    current_grip=torch.ones(1), actions=torch.eye(4).repeat(1,2,1,1),
                    actions_grip=torch.ones(1,2), T_w_e=torch.eye(4).unsqueeze(0), diff_time=torch.ones(1,1))

    def test_real_graph_forward_backward_with_replaced_scene_and_stages(self):
        model = self.build(self.c, self.factories)
        output = model(self.sample())
        self.assertEqual(output.shape, (1,2,6,7))
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertIsNotNone(model.prediction_head.lins[0].weight.grad)
        self.assertIn('scene_encoder.weight', model.state_dict())
        self.assertFalse(any(k.startswith('backbone.') for k in model.state_dict()))

    def test_factory_instances_and_config_are_not_shared(self):
        a, b = self.build(self.c, self.factories), self.build(self.c, self.factories)
        self.assertIsNot(a.scene_encoder, b.scene_encoder)
        a.reinit_graphs(2, num_demos=1)
        self.assertEqual(b.batch_size, 1)
        self.assertEqual(self.c.runtime.batch_size, 1)
        with self.assertRaises(TypeError):
            self.factories.scene['new'] = lambda cfg: None

    def test_unknown_component_fails_before_weight_loading(self):
        with self.assertRaisesRegex(ValueError, 'scene.*unknown'):
            self.build(replace(self.c, scene=replace(self.c.scene, kind='unknown')), self.factories)

    def test_unknown_codec_or_sampler_is_resolved_before_artifact_io(self):
        from icgs.composition import build_policy
        c = replace(self.c, scene=replace(self.c.scene, pretrained=True, checkpoint='/absent/encoder.pt'))
        with self.assertRaisesRegex(ValueError, 'codec.*unknown'):
            self.build(replace(c, action=replace(c.action, kind='unknown')), self.factories)
        with self.assertRaisesRegex(ValueError, 'sampler.*unknown'):
            build_policy(replace(c, sampling=replace(c.sampling, kind='unknown')), self.factories)

    def test_legacy_graph_constructor_still_accepts_original_dictionary(self):
        from icgs.configuration.defaults import to_legacy
        from icgs.models.graphs.ip_graph import GraphRep
        graph = GraphRep(to_legacy(self.c))
        graph.initialise_graph()
        self.assertEqual(graph.batch_size, 1)
        self.assertEqual(graph.num_scenes_nodes, 2)

    def test_graph_builder_replacement_changes_graph_geometry_through_factory(self):
        from icgs.models.graphs.ip_graph import GraphRep
        def half_geometry(cfg, batch_size, device):
            graph = GraphRep(cfg, batch_size, device)
            graph.gripper_node_pos = graph.gripper_node_pos * .5
            return graph
        cfg = replace(self.c, graph=replace(self.c.graph, kind='half_geometry'))
        model = self.build(cfg, replace(self.factories, graph={'half_geometry': half_geometry}))
        model(self.sample())
        torch.testing.assert_close(model.graph.graph['gripper'].pos[1], torch.tensor([0.,0.,-.03]))
        torch.testing.assert_close(model.codec.gripper_node_pos, model.graph.gripper_node_pos)


if __name__ == '__main__':
    unittest.main()
