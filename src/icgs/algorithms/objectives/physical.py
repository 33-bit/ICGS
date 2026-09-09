"""Executed-transition physical dynamics losses and bootstrap support."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from icgs.configuration.method import MethodConfig
from icgs.contracts.method import ExecutedTransition, PhysicalPrediction, TimedObservation
from icgs.contracts.records import Observation
from icgs.state.physical import PhysicalState


def _require_config(config: MethodConfig) -> MethodConfig:
    if not isinstance(config, MethodConfig):
        raise TypeError("config must be a resolved MethodConfig")
    return config


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def bootstrap_mask(
    episode_ids: Sequence[str],
    seed: int,
    *,
    config: MethodConfig,
) -> Tensor:
    """Return a deterministic Bernoulli mask with at least one head per episode."""

    _require_config(config)
    _validate_seed(seed)
    if isinstance(episode_ids, (str, bytes)):
        raise TypeError("episode_ids must be a sequence of identifiers")
    episode_ids = tuple(episode_ids)
    if not episode_ids:
        raise ValueError("episode_ids must be nonempty")
    if any(not isinstance(identifier, str) or not identifier for identifier in episode_ids):
        raise ValueError("episode_ids must contain nonempty strings")

    probability = config.dynamics.bootstrap_probability
    rows = []
    for identifier in episode_ids:
        digest = hashlib.sha256(f"{seed}\0{identifier}".encode("utf-8")).digest()
        generator_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(generator_seed)
        row = torch.rand(3, generator=generator) < probability
        if not bool(row.any().item()):
            row[int.from_bytes(digest[8:16], "little") % 3] = True
        rows.append(row)
    return torch.stack(rows, dim=0)


def _target_observation(
    target: ExecutedTransition | TimedObservation | Observation,
) -> Observation:
    if isinstance(target, ExecutedTransition):
        return target.after.observation
    if isinstance(target, TimedObservation):
        return target.observation
    if isinstance(target, Observation):
        return target
    raise TypeError(
        "target must be an ExecutedTransition, TimedObservation, or Observation"
    )


def _detached_tensor(
    value: object,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if torch.is_tensor(value):
        return value.to(dtype=dtype, device=device).detach()
    return torch.tensor(value, dtype=dtype, device=device)


def _target_tensors(
    target: ExecutedTransition | TimedObservation | Observation,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    observation = _target_observation(target)
    points = _detached_tensor(observation.points, dtype=dtype, device=device)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("target observation points must have shape [N,3]")
    if not bool(torch.isfinite(points).all().item()):
        raise ValueError("target observation points must be finite")
    points = points.unsqueeze(0)
    point_valid = torch.ones(points.shape[:2], dtype=torch.bool, device=device)

    pose = _detached_tensor(observation.T_w_e, dtype=dtype, device=device)
    if pose.shape != (4, 4):
        raise ValueError("target observation pose must have shape [4,4]")
    if not bool(torch.isfinite(pose).all().item()):
        raise ValueError("target observation pose must be finite")
    pose = pose.unsqueeze(0)

    grip = _detached_tensor(observation.grip, dtype=dtype, device=device).reshape(1, 1)
    if not bool(torch.isfinite(grip).all().item()) or not bool(((grip == 0) | (grip == 1)).all().item()):
        raise ValueError("target observation grip must be 0 or 1")
    return points, point_valid, pose, grip


def _prediction_tensors(
    prediction: PhysicalPrediction,
) -> tuple[PhysicalState, Tensor, Tensor, Tensor, Tensor]:
    if not isinstance(prediction, PhysicalPrediction):
        raise TypeError("prediction must be a PhysicalPrediction")
    state = prediction.next_state
    if not isinstance(state, PhysicalState):
        raise TypeError("prediction.next_state must be a PhysicalState")
    points = state.cached_world_cloud
    point_valid = state.cached_world_cloud_valid
    if points is None or point_valid is None:
        raise ValueError("prediction has no cached geometry")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("prediction cloud must have shape [B,N,3]")
    if point_valid.shape != points.shape[:2] or point_valid.dtype != torch.bool:
        raise ValueError("prediction cloud validity must have shape [B,N] and boolean dtype")
    if not bool(torch.isfinite(points).all().item()):
        raise ValueError("prediction cloud must be finite")

    logits = (
        prediction.grip_logits.to(dtype=state.T_w_e.dtype, device=state.T_w_e.device)
        if torch.is_tensor(prediction.grip_logits)
        else torch.tensor(
            prediction.grip_logits,
            dtype=state.T_w_e.dtype,
            device=state.T_w_e.device,
        )
    )
    if logits.ndim == 1:
        logits = logits[:, None]
    if logits.ndim != 2 or logits.shape[1:] != (1,):
        raise ValueError("prediction grip_logits must have shape [B,1]")
    if logits.shape[0] != points.shape[0] or not bool(torch.isfinite(logits).all().item()):
        raise ValueError("prediction grip_logits must match the prediction batch and be finite")
    return state, points, point_valid, state.T_w_e, logits


def _sample_valid(valid: Tensor, batch: int, *, device: torch.device) -> Tensor:
    mask = torch.as_tensor(valid, device=device)
    if mask.shape != (batch,) or mask.dtype != torch.bool:
        raise ValueError(f"valid must have shape [{batch}] and boolean dtype")
    return mask


def _rotation_angle_squared(prediction: Tensor, target: Tensor, margin: float) -> Tensor:
    relative = prediction[..., :3, :3].transpose(-1, -2) @ target[..., :3, :3]
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0)
    clipped = cosine.clamp(-1.0 + margin, 1.0 - margin)
    return torch.acos(clipped).square()


def _physical_loss(
    prediction: PhysicalPrediction,
    target: ExecutedTransition | TimedObservation | Observation,
    valid: Tensor,
    *,
    config: MethodConfig,
) -> Tensor:
    state, predicted_points, predicted_valid, predicted_pose, predicted_logits = _prediction_tensors(prediction)
    batch = predicted_points.shape[0]
    sample_valid = _sample_valid(valid, batch, device=predicted_points.device)
    target_points, target_valid, target_pose, target_grip = _target_tensors(
        target,
        device=predicted_points.device,
        dtype=predicted_points.dtype,
    )
    if batch != 1:
        raise ValueError("executed physical targets support one prediction sample per record")
    if target_points.shape[0] != batch:
        raise ValueError("prediction and target batch sizes must match")

    losses = []
    cloud_scale = config.losses.cloud_scale_m ** 2
    translation_scale = config.losses.translation_scale_m ** 2
    rotation_scale = math.radians(config.losses.rotation_scale_deg) ** 2
    for index in range(batch):
        if not bool(sample_valid[index].item()):
            continue
        predicted = predicted_points[index][predicted_valid[index]]
        expected = target_points[index][target_valid[index]]
        if predicted.numel() == 0 or expected.numel() == 0:
            raise ValueError("valid physical samples require predicted and target geometry")
        distances = (predicted[:, None, :] - expected[None, :, :]).square().sum(dim=-1)
        directed_prediction = distances.min(dim=1).values.mean()
        directed_target = distances.min(dim=0).values.mean()
        chamfer = directed_prediction + directed_target
        translation = (
            predicted_pose[index, :3, 3] - target_pose[index, :3, 3]
        ).square().mean()
        rotation = _rotation_angle_squared(
            predicted_pose[index:index + 1],
            target_pose[index:index + 1],
            config.numerics.rotation_training_clip_margin,
        )[0]
        grip = F.binary_cross_entropy_with_logits(
            predicted_logits[index:index + 1],
            target_grip[index:index + 1],
        )
        losses.append(
            chamfer / cloud_scale
            + translation / translation_scale
            + rotation / rotation_scale
            + config.losses.grip_weight * grip
        )
    if losses:
        return torch.stack(losses).mean()
    return state.X.sum() * 0.0


def physical_loss(
    prediction: PhysicalPrediction,
    target: ExecutedTransition | TimedObservation | Observation,
    valid: Tensor,
    *,
    config: MethodConfig,
) -> Tensor:
    """Compute masked normalized physical loss against an executed target."""

    _require_config(config)
    return _physical_loss(prediction, target, valid, config=config)


def rollout_loss(
    predictions: Sequence[Sequence[Sequence[PhysicalPrediction]]],
    targets: Sequence[Sequence[ExecutedTransition | TimedObservation | Observation]],
    bootstrap: Tensor,
    *,
    valid: Tensor | None = None,
    reconstruction_loss: Tensor | None = None,
    encoder_decoder_trainable: bool = True,
    config: MethodConfig,
) -> Tensor:
    """Average recursive executed-transition losses over selected heads.

    ``predictions`` is nested as ``[episode][head][rollout_step]`` and contains
    concrete :class:`PhysicalPrediction` records.  ``targets`` is
    ``[episode][rollout_step]`` of native executed-transition records.  When
    supplied, ``valid`` must be a boolean ``[episode, rollout_step]`` mask;
    false entries are padding and are excluded from both loss sum and count.
    """

    _require_config(config)
    if not isinstance(predictions, Sequence) or not isinstance(targets, Sequence):
        raise TypeError("predictions and targets must be nested sequences")
    episode_count = len(predictions)
    if episode_count == 0 or len(targets) != episode_count:
        raise ValueError("predictions and targets must contain the same nonempty episode count")

    bootstrap = torch.as_tensor(bootstrap)
    if bootstrap.shape != (episode_count, 3) or bootstrap.dtype != torch.bool:
        raise ValueError(
            f"bootstrap must have shape [{episode_count},3] and boolean dtype"
        )
    if not bool(bootstrap.any(dim=1).all().item()):
        raise ValueError("bootstrap must select at least one head per episode")

    rollout_steps = len(targets[0]) if isinstance(targets[0], Sequence) else -1
    if rollout_steps <= 0:
        raise ValueError("each episode requires nonempty target steps")
    if any(not isinstance(episode_targets, Sequence) or len(episode_targets) != rollout_steps
           for episode_targets in targets):
        raise ValueError("all episodes must share one rollout_steps length")

    if valid is None:
        supervision = torch.ones((episode_count, rollout_steps), dtype=torch.bool)
    else:
        supervision = torch.as_tensor(valid)
        if supervision.shape != (episode_count, rollout_steps) or supervision.dtype != torch.bool:
            raise ValueError(
                f"valid must have shape [{episode_count},{rollout_steps}] and boolean dtype"
            )

    loss_terms = []
    supervised_steps = 0
    for episode_index, episode_predictions in enumerate(predictions):
        if not isinstance(episode_predictions, Sequence) or len(episode_predictions) != 3:
            raise ValueError("each episode requires three fixed head slots")
        for head_index in range(3):
            if not bool(bootstrap[episode_index, head_index].item()):
                continue
            head_predictions = episode_predictions[head_index]
            if not isinstance(head_predictions, Sequence) or len(head_predictions) != rollout_steps:
                raise ValueError("each selected head must predict every target step")
            for step_index, prediction in enumerate(head_predictions):
                if not isinstance(prediction, PhysicalPrediction):
                    raise TypeError("selected rollout entries must be PhysicalPrediction records")
                if prediction.head_id != head_index:
                    raise ValueError(
                        f"prediction head_id {prediction.head_id} does not match head slot {head_index}"
                    )
                if not bool(supervision[episode_index, step_index].item()):
                    continue
                state = prediction.next_state
                if not isinstance(state, PhysicalState):
                    raise TypeError("prediction.next_state must be a PhysicalState")
                step_valid = torch.ones(
                    state.T_w_e.shape[0],
                    dtype=torch.bool,
                    device=state.T_w_e.device,
                )
                loss_terms.append(
                    _physical_loss(
                        prediction,
                        targets[episode_index][step_index],
                        step_valid,
                        config=config,
                    )
                )
                supervised_steps += 1

    if supervised_steps == 0:
        raise ValueError("rollout has no supervised valid steps")
    total = torch.stack(loss_terms).sum() / supervised_steps
    if encoder_decoder_trainable:
        if reconstruction_loss is None:
            raise ValueError("reconstruction_loss is required while encoder_decoder_trainable")
        reconstruction = torch.as_tensor(
            reconstruction_loss,
            dtype=total.dtype,
            device=total.device,
        )
        if reconstruction.ndim != 0 or not bool(torch.isfinite(reconstruction).item()):
            raise ValueError("reconstruction_loss must be a finite scalar")
        total = total + config.losses.reconstruction_weight * reconstruction
    return total


__all__ = ["bootstrap_mask", "physical_loss", "rollout_loss"]
