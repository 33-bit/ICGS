"""Phase/freeze matrix, stage boundaries, curricula and stage optimizers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch
from torch import nn

from icgs.configuration.method import MethodConfig

_STAGE_TRAINABLE: dict[str, frozenset[str]] = {
    "A0": frozenset({"geometry"}),
    "A1": frozenset({"geometry", "physical_memory", "dynamics"}),
    "B": frozenset({"events", "task"}),
    "C": frozenset(),
    "D1": frozenset({"dynamics"}),
    "D2": frozenset({"evaluation", "terminal"}),
    "E": frozenset({"temperatures"}),
    "Test": frozenset(),
}

_STAGE_ALIASES: dict[str, str] = {
    "a0": "A0",
    "a1": "A1",
    "b": "B",
    "c": "C",
    "d1": "D1",
    "d2": "D2",
    "e": "E",
    "test": "Test",
}


def _canonical_stage(stage: str) -> str:
    if not isinstance(stage, str):
        raise TypeError(f"stage must be a string, got {type(stage).__name__}")
    normalized = stage.strip()
    if normalized in _STAGE_TRAINABLE:
        return normalized
    lowered = normalized.lower()
    if lowered in _STAGE_ALIASES:
        return _STAGE_ALIASES[lowered]
    raise ValueError(f"unknown training stage: {stage!r}")


def trainable_components(stage: str) -> frozenset[str]:
    """Return the set of trainable component role names for a given phase."""
    canonical = _canonical_stage(stage)
    return _STAGE_TRAINABLE[canonical]


def rollout_horizon(
    stage: str,
    update: int,
    max_updates: int,
    *,
    curriculum: Sequence[int] | None = None,
    config: MethodConfig | None = None,
) -> int:
    """Return the unroll horizon K for a stage and update counter."""
    canonical = _canonical_stage(stage)
    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError("update must be a nonnegative integer")
    if isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates < 1:
        raise ValueError("max_updates must be a positive integer")

    if update >= max_updates:
        raise ValueError(
            f"update {update} is out of bounds (must be less than max_updates {max_updates})"
        )

    if curriculum is None:
        if config is not None and hasattr(config, "stages"):
            stage_cfg = getattr(config.stages, canonical, None)
            if stage_cfg is not None and hasattr(stage_cfg, "rollout_curriculum"):
                curriculum = getattr(stage_cfg, "rollout_curriculum")
        if curriculum is None:
            if canonical == "A1":
                curriculum = (1, 2, 4)
            elif canonical == "D1":
                curriculum = (2, 4, 8, 16)
            else:
                raise ValueError(f"stage {stage!r} does not define a rollout curriculum")

    if not isinstance(curriculum, (tuple, list)) or not curriculum:
        raise ValueError("curriculum must be a nonempty sequence of integer horizons")
    for k in curriculum:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("curriculum entries must be positive integers")

    step_count = len(curriculum)
    idx = (update * step_count) // max_updates
    if idx >= step_count:
        idx = step_count - 1
    return int(curriculum[idx])


_STAGE_REQUIRED_ROLES: dict[str, frozenset[str]] = {
    "A0": frozenset({"geometry"}),
    "A1": frozenset({"geometry", "physical_memory", "dynamics"}),
    "B": frozenset({"events", "task"}),
    "C": frozenset(),
    "D1": frozenset({"geometry", "physical_memory", "dynamics"}),
    "D2": frozenset({"evaluation", "terminal"}),
    "E": frozenset(),
    "Test": frozenset(),
}


def get_geometry_encoder_decoder(geometry: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Extract encoder and decoder submodules from a single geometry owner."""
    if not isinstance(geometry, nn.Module):
        raise TypeError(f"geometry must be an nn.Module, got {type(geometry).__name__}")
    if isinstance(geometry, Mapping) and "encoder" in geometry and "decoder" in geometry:
        encoder = geometry["encoder"]
        decoder = geometry["decoder"]
    elif hasattr(geometry, "encoder") and hasattr(geometry, "decoder"):
        encoder = getattr(geometry, "encoder")
        decoder = getattr(geometry, "decoder")
    else:
        raise KeyError("geometry module must contain both 'encoder' and 'decoder' submodules")
    if not isinstance(encoder, nn.Module) or not isinstance(decoder, nn.Module):
        raise TypeError("both encoder and decoder in geometry must be instances of nn.Module")
    return encoder, decoder


def apply_freeze_boundary(
    stage: str,
    components: Mapping[str, nn.Module],
) -> frozenset[str]:
    """Enforce parameter requires_grad strictly per the stage matrix.

    Rejects non-modules, shared/aliased parameters/modules crossing trainable and
    frozen boundaries, and sets module train/eval state (frozen modules set to eval).
    Requires all declared stage roles to be present before mutating.
    """
    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping of role names to modules")

    for name, module in components.items():
        if not isinstance(module, nn.Module):
            raise TypeError(f"component {name!r} must be an nn.Module, got {type(module).__name__}")

    canonical = _canonical_stage(stage)
    required = _STAGE_REQUIRED_ROLES[canonical]
    missing = required - set(components.keys())
    if missing:
        raise KeyError(f"stage {stage!r} missing required component role(s): {sorted(missing)}")

    if canonical in ("A0", "A1", "D1") and "geometry" in components:
        geo = components["geometry"]
        has_encoder = (isinstance(geo, Mapping) and "encoder" in geo) or hasattr(geo, "encoder")
        has_decoder = (isinstance(geo, Mapping) and "decoder" in geo) or hasattr(geo, "decoder")
        if not (has_encoder and has_decoder):
            raise KeyError(f"geometry component for {stage!r} must contain both 'encoder' and 'decoder'")

    trainable = trainable_components(stage)

    # Detect shared modules or parameters between trainable and frozen roles
    module_roles: dict[int, set[str]] = {}
    param_roles: dict[int, set[str]] = {}
    for name, module in components.items():
        module_roles.setdefault(id(module), set()).add(name)
        for param in module.parameters():
            param_roles.setdefault(id(param), set()).add(name)

    for m_id, roles in module_roles.items():
        has_trainable = any(r in trainable for r in roles)
        has_frozen = any(r not in trainable for r in roles)
        if has_trainable and has_frozen:
            raise ValueError(
                f"Conflicting trainable and frozen role assignment for shared module: {sorted(roles)}"
            )

    for p_id, roles in param_roles.items():
        has_trainable = any(r in trainable for r in roles)
        has_frozen = any(r not in trainable for r in roles)
        if has_trainable and has_frozen:
            raise ValueError(
                f"Conflicting trainable and frozen role assignment for shared parameter: {sorted(roles)}"
            )

    # Enforce requires_grad and train/eval mode
    for name, module in components.items():
        is_trainable = name in trainable
        module.requires_grad_(is_trainable)
        if is_trainable:
            module.train()
        else:
            module.eval()

    return trainable



def build_stage_optimizer(
    stage: str,
    components: Mapping[str, nn.Module],
    cfg: MethodConfig,
) -> torch.optim.Optimizer | None:
    """Build the AdamW optimizer for permitted trainable parameters only."""
    trainable = trainable_components(stage)
    if not trainable or stage in ("C", "c", "Test", "test"):
        return None

    trainable_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name in sorted(components.keys()):
        module = components[name]
        if name in trainable and isinstance(module, nn.Module):
            for param in module.parameters():
                if param.requires_grad and id(param) not in seen:
                    seen.add(id(param))
                    trainable_params.append(param)

    if not trainable_params:
        return None

    opt_cfg = cfg.optimizer
    betas = tuple(opt_cfg.betas)
    return torch.optim.AdamW(
        trainable_params,
        lr=float(opt_cfg.learning_rate),
        betas=(float(betas[0]), float(betas[1])),
        weight_decay=float(opt_cfg.weight_decay),
    )


def build_stage_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: MethodConfig,
    max_updates: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup followed by cosine decay down to minimum_learning_rate."""
    if isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates < 1:
        raise ValueError("max_updates must be a positive integer")

    opt_cfg = cfg.optimizer
    warmup = max(0, int(opt_cfg.warmup_updates))
    lr_base = float(opt_cfg.learning_rate)
    lr_min = float(opt_cfg.minimum_learning_rate)
    min_ratio = lr_min / lr_base if lr_base > 0 else 0.0

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup:
            return min_ratio + (1.0 - min_ratio) * (float(current_step) / float(max(1, warmup)))
        progress = float(current_step - warmup) / float(max(1, max_updates - warmup))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


__all__ = [
    "apply_freeze_boundary",
    "build_stage_optimizer",
    "build_stage_scheduler",
    "rollout_horizon",
    "trainable_components",
]
