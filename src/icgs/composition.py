"""Explicit construction and artifact boundary shared by all entry points."""
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Callable

from icgs.configuration.defaults import from_legacy


@dataclass(frozen=True)
class ComponentFactories:
    scene: Mapping[str, Callable] = field(default_factory=dict)
    graph: Mapping[str, Callable] = field(default_factory=dict)
    backbone: Mapping[str, Callable] = field(default_factory=dict)
    codec: Mapping[str, Callable] = field(default_factory=dict)
    sampler: Mapping[str, Callable] = field(default_factory=dict)
    objective: Mapping[str, Callable] = field(default_factory=dict)
    scheduler: Mapping[str, Callable] = field(default_factory=dict)

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def select(factories, family, kind, original):
    if kind in getattr(factories, family):
        return getattr(factories, family)[kind]
    if kind == 'original':
        return original
    raise ValueError(f"{family} component {kind!r} is unknown; supply an explicit factory")


def build_scene_encoder(config, factory=None, device='cpu'):
    """Construct/load the selected encoder. No component performs artifact IO."""
    import torch
    if factory is None:
        if config.kind != 'original':
            raise ValueError(f"scene component {config.kind!r} is unknown; supply an explicit factory")
        from icgs.models.encoders.ip_scene import SceneEncoder
        factory = lambda cfg: SceneEncoder(cfg.num_freqs, cfg.embd_dim)
    encoder = factory(config).to(device)
    if config.pretrained:
        # Trusted original state-dict artifact; preserve original no-map-location load.
        encoder.load_state_dict(torch.load(config.checkpoint, weights_only=False))
        if config.freeze:
            for parameter in encoder.parameters():
                parameter.requires_grad = False
    return encoder


def build_components(config, factories=None):
    from torch_geometric.nn import MLP
    from icgs.models.encoders.ip_scene import SceneEncoder
    from icgs.models.graphs.ip_graph import GraphRep
    from icgs.models.backbones.ip_stages import original_stages
    config = from_legacy(config)
    f = factories or ComponentFactories()
    # Resolve identifiers before artifact IO; construct in the original RNG order.
    scene_factory = select(f, 'scene', config.scene.kind, lambda c: SceneEncoder(c.num_freqs, c.embd_dim))
    graph_factory = select(f, 'graph', config.graph.kind, GraphRep)
    backbone_factory = select(f, 'backbone', config.backbone.kind, original_stages)
    scene = build_scene_encoder(config.scene, scene_factory, config.runtime.device)
    graph = graph_factory(config.graph, config.runtime.batch_size, config.runtime.device)
    graph.initialise_graph()
    inputs = config.graph.embd_dim + (graph.edge_dim // 2 if config.graph.pos_in_nodes else 0)
    local, conditioning, action = backbone_factory(config.backbone, inputs, graph.edge_dim)
    local, conditioning, action = (m.to(config.runtime.device) for m in (local, conditioning, action))
    # Keep head construction order and original registered names.
    heads = [MLP([config.backbone.hidden_dim, config.graph.embd_dim, n], act='GELU',
                 plain_last=True, norm='layer_norm') for n in (3, 3, 1)]
    return dict(scene_encoder=scene, graph=graph, local_encoder=local,
                cond_encoder=conditioning, action_encoder=action,
                prediction_head=heads[0], prediction_head_rot=heads[1], prediction_head_g=heads[2])


def build_network(config, factories=None):
    from icgs.algorithms.diffusion.codec import OriginalActionCodec
    from icgs.models.denoisers.ip_graph import GraphDenoiser
    config = from_legacy(config)
    f = factories or ComponentFactories()
    codec_factory = select(f, 'codec', config.action.kind, OriginalActionCodec)
    parts = build_components(config, f)
    codec = codec_factory(
        config.action, parts['graph'].gripper_node_pos, config.runtime.device)
    model = GraphDenoiser(config.graph, config.backbone, config.runtime, parts, codec)
    if config.runtime.compile_models:
        model.compile_models()
    return model.to(config.runtime.device)


def build_policy(config, factories=None):
    from icgs.algorithms.diffusion.process import OriginalDiffusionSampler, OriginalDiffusionObjective, original_scheduler
    from icgs.policies.instant_policy import InstantPolicy
    config = from_legacy(config)
    f = factories or ComponentFactories()
    scheduler_factory = select(f, 'scheduler', config.diffusion.scheduler_kind, original_scheduler)
    sampler_factory = select(f, 'sampler', config.sampling.kind, OriginalDiffusionSampler)
    objective_factory = select(f, 'objective', config.diffusion.kind, OriginalDiffusionObjective)
    model = build_network(config, f)
    schedule = scheduler_factory(config.diffusion)
    sampler = sampler_factory(
        config.sampling, config.diffusion, model.codec, schedule)
    objective = objective_factory(
        config.diffusion, model.codec, schedule)
    return InstantPolicy(model, sampler, objective, config.graph, config.runtime)


def build_training_module(config, factories=None):
    from icgs.training.modules.diffusion import GraphDiffusion
    config = from_legacy(config)
    return GraphDiffusion(config, policy=build_policy(config, factories))


def read_experiment_config(directory):
    """Read a versioned config if saved, otherwise trusted legacy config.pkl."""
    import json
    import pickle
    from pathlib import Path
    from icgs.configuration.defaults import from_resolved
    directory = Path(directory)
    resolved = directory / 'resolved_config.json'
    if resolved.exists():
        return from_resolved(json.loads(resolved.read_text()))
    with (directory / 'config.pkl').open('rb') as stream:
        return from_legacy(pickle.load(stream))  # Trusted artifacts only.


def load_policy(directory='./checkpoints', model_name='model.pt', *, mode='eval',
                num_demos=2, compile_models=False, factories=None, overrides=None):
    """Shared trusted-artifact inference construction for eval and deployment."""
    from dataclasses import replace
    from pathlib import Path
    from icgs.configuration.defaults import profile
    from icgs.artifacts.checkpoints import load_checkpoint_state
    config = profile(read_experiment_config(directory), mode, num_demos=num_demos)
    if overrides is not None:
        config = overrides(config).validate()
    # Old entry points load uncompiled weights first, then optionally compile.
    config = replace(config, runtime=replace(config.runtime, compile_models=False))
    policy = build_policy(config, factories)
    policy.checkpoint_report = load_checkpoint_state(Path(directory) / model_name, policy.network,
                                                     strict=mode != 'deploy', map_location=config.runtime.device)
    policy.network.reinit_graphs(config.runtime.batch_size, num_demos=config.graph.num_demos)
    policy.eval()
    if compile_models:
        policy.network.compile_models()
    return policy
