"""Immutable experiment sections. No imports of tensor or experiment frameworks."""
from dataclasses import dataclass, field
import math
import re


@dataclass(frozen=True)
class SceneConfig:
    kind: str = "original"
    num_freqs: int = 10
    embd_dim: int = 512
    pretrained: bool = True
    freeze: bool = True
    checkpoint: str = "./checkpoints/scene_encoder.pt"


@dataclass(frozen=True)
class GraphConfig:
    kind: str = "original"
    num_demos: int = 2
    traj_horizon: int = 10
    num_scene_nodes: int = 16
    local_num_freq: int = 10
    embd_dim: int = 512
    pred_horizon: int = 8
    pos_in_nodes: bool = True


@dataclass(frozen=True)
class BackboneConfig:
    kind: str = "original"
    hidden_dim: int = 1024
    num_layers: int = 2


@dataclass(frozen=True)
class ActionConfig:
    kind: str = "original"
    pred_horizon: int = 8
    minimum: tuple[float, ...] = (-0.01,) * 3 + (-math.pi / 60,) * 3
    maximum: tuple[float, ...] = (0.01,) * 3 + (math.pi / 60,) * 3


@dataclass(frozen=True)
class DiffusionConfig:
    kind: str = "original"
    scheduler_kind: str = "original"
    train_steps: int = 100


@dataclass(frozen=True)
class SamplingConfig:
    kind: str = "original"
    steps: int = 8


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda"
    batch_size: int = 16
    compile_models: bool = False
    live_voxel_size: float | None = None
    cache_context: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    record: bool = False
    save_dir: str | None = None
    save_every: int = 100000
    randomise_num_demos: bool = False
    num_demos_test: int = 2
    batch_size_val: int = 1
    lr: float = 1e-5
    weight_decay: float = 1e-2
    use_lr_scheduler: bool = False
    num_warmup_steps: int = 1000
    num_iters: int = 50000000001
    test_every: int = 50000
    randomize_g_prob: float = 0.1


@dataclass(frozen=True)
class EvaluationConfig:
    task_name: str = "plate_out"
    num_rollouts: int = 5
    execution_horizon: int = 8
    max_execution_steps: int = 30
    restrict_rot: bool = True
    headless: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "instant_policy_original"
    scene: SceneConfig = field(default_factory=SceneConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self):
        if not isinstance(self.runtime.device,str) or not re.fullmatch(r'(cpu|mps|cuda(?::[0-9]+)?)',self.runtime.device):
            raise ValueError('device must be cpu, mps, cuda, or cuda:index')
        for owner,names in ((self.runtime,('compile_models','cache_context')),
                            (self.scene,('pretrained','freeze')),(self.graph,('pos_in_nodes',)),
                            (self.evaluation,('restrict_rot','headless')),
                            (self.training,('record','randomise_num_demos','use_lr_scheduler'))):
            for name in names:
                if type(getattr(owner,name)) is not bool:raise ValueError(f'{name} must be boolean')
        voxel=self.runtime.live_voxel_size
        if voxel is not None and (isinstance(voxel,bool) or not isinstance(voxel,(int,float)) or not math.isfinite(voxel) or voxel<=0):
            raise ValueError('live_voxel_size must be positive finite or null')
        positive = {
            "num_demos": self.graph.num_demos, "traj_horizon": self.graph.traj_horizon,
            "scene_nodes": self.graph.num_scene_nodes, "embd_dim": self.graph.embd_dim,
            "pred_horizon": self.graph.pred_horizon, "batch_size": self.runtime.batch_size,
            "train_steps": self.diffusion.train_steps, "sampling_steps": self.sampling.steps,
            "hidden_dim": self.backbone.hidden_dim, "num_layers": self.backbone.num_layers,
            "num_rollouts": self.evaluation.num_rollouts,
            "execution_horizon": self.evaluation.execution_horizon,
            "max_execution_steps": self.evaluation.max_execution_steps,
            "save_every": self.training.save_every, "num_iters": self.training.num_iters,
        }
        for key, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer, got {value!r}")
        if self.scene.embd_dim != self.graph.embd_dim:
            raise ValueError("scene/graph embedding widths must agree")
        if self.action.pred_horizon != self.graph.pred_horizon:
            raise ValueError("action/graph prediction horizons must agree")
        if self.evaluation.execution_horizon>self.action.pred_horizon:
            raise ValueError('execution horizon exceeds prediction horizon')
        if self.graph.kind == 'original' and self.graph.embd_dim <= 64:
            raise ValueError("original gripper embedding requires width > 64")
        if self.backbone.kind == "original" and self.backbone.hidden_dim % 64:
            raise ValueError("original backbone hidden_dim must be divisible by 64")
        if self.sampling.steps > self.diffusion.train_steps:
            raise ValueError("sampling steps exceed training steps")
        if not isinstance(self.action.minimum, tuple) or not isinstance(self.action.maximum, tuple):
            raise ValueError("action limits must be immutable tuples")
        if len(self.action.minimum) != 6 or len(self.action.maximum) != 6:
            raise ValueError("action limits must have six components")
        for low, high in zip(self.action.minimum, self.action.maximum):
            if not math.isfinite(low) or not math.isfinite(high) or not low < 0 < high:
                raise ValueError("action limits must be finite and straddle zero")
        if not 0 <= self.training.randomize_g_prob <= 1:
            raise ValueError("grip randomization probability must be in [0,1]")
        return self
