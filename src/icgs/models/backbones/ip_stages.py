"""Original three-stage graph processing, separate from semantic graph building."""
from icgs.models.backbones.graph_transformer import GraphTransformer


def original_stages(config, input_channels, edge_dim):
    local = GraphTransformer(in_channels=input_channels,
                                          hidden_channels=config.hidden_dim,
                                          heads=config.hidden_dim // 64,
                                          num_layers=config.num_layers,
                                          metadata=(['scene', 'gripper'],
                                                    [
                                                        ('scene', 'rel', 'scene'),
                                                        ('scene', 'rel', 'gripper'),
                                                        ('gripper', 'rel', 'gripper'),
                                                    ]),
                                          edge_dim=edge_dim,
                                          dropout=0.0,
                                          norm='layer')

    conditioning = GraphTransformer(in_channels=config.hidden_dim,
                                         hidden_channels=config.hidden_dim,
                                         heads=config.hidden_dim // 64,
                                         num_layers=config.num_layers,
                                         metadata=(['gripper', 'scene'],
                                                   [
                                                       ('gripper', 'cond', 'gripper'),
                                                       ('gripper', 'demo', 'gripper'),
                                                       ('scene', 'rel_demo', 'gripper'),
                                                       ('scene', 'rel_demo', 'scene'),
                                                   ]),
                                         edge_dim=edge_dim,
                                         dropout=0.0,
                                         norm='layer')

    action = GraphTransformer(in_channels=config.hidden_dim,
                                           hidden_channels=config.hidden_dim,
                                           heads=config.hidden_dim // 64,
                                           num_layers=config.num_layers,
                                           metadata=(['gripper', 'scene'],
                                                     [
                                                         ('gripper', 'time_action', 'gripper'),
                                                         ('gripper', 'rel_cond', 'gripper'),
                                                         ('scene', 'rel_action', 'gripper'),
                                                         ('scene', 'rel_action', 'scene'),
                                                     ]),
                                           edge_dim=edge_dim,
                                           dropout=0.0,
                                           norm='layer')
    return local, conditioning, action


def run_stages(local, conditioning, action, graph):
    x = local(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)
    x = conditioning(x, graph.edge_index_dict, graph.edge_attr_dict)
    return action(x, graph.edge_index_dict, graph.edge_attr_dict)
