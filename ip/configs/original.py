"""Canonical original composition and explicit conversion of historical config."""
from dataclasses import asdict, replace
from .structured import (
    ExperimentConfig, SceneConfig, GraphConfig, BackboneConfig, ActionConfig,
    DiffusionConfig, SamplingConfig, RuntimeConfig, TrainingConfig,
)


def instant_policy_original():
    return ExperimentConfig().validate()


def to_legacy(config):
    """Fresh tensors/dictionary for legacy artifacts; never shared runtime state."""
    import torch
    config.validate()
    result = asdict(config.training)
    result.update(
        scene_encoder_path=config.scene.checkpoint, pre_trained_encoder=config.scene.pretrained,
        freeze_encoder=config.scene.freeze, compile_models=config.runtime.compile_models,
        local_num_freq=config.graph.local_num_freq, local_nn_dim=config.graph.embd_dim,
        hidden_dim=config.backbone.hidden_dim, num_demos=config.graph.num_demos,
        traj_horizon=config.graph.traj_horizon, device=config.runtime.device,
        batch_size=config.runtime.batch_size, num_scenes_nodes=config.graph.num_scene_nodes,
        pre_horizon=config.graph.pred_horizon, pos_in_nodes=config.graph.pos_in_nodes,
        num_layers=config.backbone.num_layers, num_diffusion_iters_train=config.diffusion.train_steps,
        num_diffusion_iters_test=config.sampling.steps,
        min_actions=torch.tensor(config.action.minimum, dtype=torch.float32),
        max_actions=torch.tensor(config.action.maximum, dtype=torch.float32),
    )
    return result


def from_legacy(values):
    """Require every original key; unknown checkpoint config is not defaulted away."""
    if isinstance(values, ExperimentConfig):
        return values.validate()
    expected = set(to_legacy(ExperimentConfig()))
    missing, unknown = expected - set(values), set(values) - expected
    if missing or unknown:
        raise ValueError(f"legacy config missing={sorted(missing)} unknown={sorted(unknown)}")
    v = values
    train_keys = TrainingConfig.__dataclass_fields__
    result = ExperimentConfig(
        scene=SceneConfig(embd_dim=v["local_nn_dim"], pretrained=v["pre_trained_encoder"],
                          freeze=v["freeze_encoder"], checkpoint=v["scene_encoder_path"]),
        graph=GraphConfig(num_demos=v["num_demos"], traj_horizon=v["traj_horizon"],
                          num_scene_nodes=v["num_scenes_nodes"], local_num_freq=v["local_num_freq"],
                          embd_dim=v["local_nn_dim"], pred_horizon=v["pre_horizon"], pos_in_nodes=v["pos_in_nodes"]),
        backbone=BackboneConfig(hidden_dim=v["hidden_dim"], num_layers=v["num_layers"]),
        action=ActionConfig(pred_horizon=v["pre_horizon"],
                            minimum=tuple(float(x) for x in v["min_actions"]),
                            maximum=tuple(float(x) for x in v["max_actions"])),
        diffusion=DiffusionConfig(train_steps=v["num_diffusion_iters_train"]),
        sampling=SamplingConfig(steps=v["num_diffusion_iters_test"]),
        runtime=RuntimeConfig(device=v["device"], batch_size=v["batch_size"], compile_models=v["compile_models"]),
        training=TrainingConfig(**{k: v[k] for k in train_keys}),
    )
    return result.validate()


def profile(config, mode, *, num_demos=None, batch_size=None, record=None, save_dir=None, compile_models=False):
    """Apply only documented entry differences; explicit experiments may replace sections afterwards."""
    config = from_legacy(config)
    if mode not in {"train", "fine_tune", "eval", "deploy"}:
        raise ValueError(f"unknown entry profile: {mode}")
    if mode in {"eval", "deploy"}:
        config = replace(config,
                         graph=replace(config.graph, num_demos=config.graph.num_demos if num_demos is None else num_demos),
                         runtime=replace(config.runtime, batch_size=1, compile_models=compile_models),
                         sampling=replace(config.sampling, steps=4))
        if mode == "deploy":
            config = replace(config, backbone=replace(config.backbone, num_layers=2))
    if mode == "fine_tune":
        config = replace(config, runtime=replace(config.runtime,
                         batch_size=16 if batch_size is None else batch_size, compile_models=compile_models))
    if mode in {"train", "fine_tune"} and record is not None:
        config = replace(config, training=replace(config.training, record=record, save_dir=save_dir))
    return config.validate()


def resolved_config(config):
    """JSON-safe versioned metadata, separate from legacy config.pkl."""
    return {"schema_version": 1, "config": asdict(config.validate())}


def from_resolved(payload):
    """Decode only the versioned dataclass schema, without dynamic imports."""
    from .structured import EvaluationConfig
    if set(payload) != {'schema_version', 'config'} or payload['schema_version'] != 1:
        raise ValueError('unsupported resolved config schema')
    values = dict(payload['config'])
    sections = dict(scene=SceneConfig, graph=GraphConfig, backbone=BackboneConfig,
                    action=ActionConfig, diffusion=DiffusionConfig, sampling=SamplingConfig,
                    runtime=RuntimeConfig, training=TrainingConfig, evaluation=EvaluationConfig)
    if set(values) != {'name', *sections}:
        raise ValueError('resolved config requires exact named sections')
    for name, cls in sections.items():
        section = dict(values[name])
        if set(section) != set(cls.__dataclass_fields__):
            raise ValueError(f'incomplete or unknown resolved section: {name}')
        if name == 'action':
            section['minimum'], section['maximum'] = tuple(section['minimum']), tuple(section['maximum'])
        values[name] = cls(**section)
    return ExperimentConfig(**values).validate()
