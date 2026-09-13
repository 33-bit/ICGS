"""Independently masked supervision for raw task-tracker logits."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from icgs.configuration.method import MethodConfig
from icgs.models.memories.task import TaskEncoding


def _validate_encoding(encoding: TaskEncoding) -> tuple[int, int]:
    if not isinstance(encoding, TaskEncoding):
        raise TypeError("encoding must be a TaskEncoding")
    if not torch.is_tensor(encoding.r) or encoding.r.ndim != 2:
        raise ValueError("encoding.r must have shape [B,W]")
    batch, width = encoding.r.shape
    if batch <= 0 or width <= 0 or not encoding.r.is_floating_point():
        raise ValueError("encoding.r must have nonempty floating shape [B,W]")
    if not torch.is_tensor(encoding.event_logits) or encoding.event_logits.ndim != 3:
        raise ValueError("encoding.event_logits must have shape [B,L,3]")
    if encoding.event_logits.shape[0] != batch or encoding.event_logits.shape[2] != 3:
        raise ValueError("encoding.event_logits must have shape [B,L,3]")
    event_count = encoding.event_logits.shape[1]
    if event_count <= 0:
        raise ValueError("encoding.event_logits must have nonempty L")
    if (
        not torch.is_tensor(encoding.alignment_logits)
        or encoding.alignment_logits.shape != (batch, event_count + 1)
    ):
        raise ValueError("encoding.alignment_logits must have shape [B,L+1]")
    if (
        not torch.is_tensor(encoding.event_valid)
        or encoding.event_valid.shape != (batch, event_count)
    ):
        raise ValueError("encoding.event_valid must have shape [B,L]")
    if encoding.event_valid.dtype != torch.bool:
        raise TypeError("encoding.event_valid must use torch.bool dtype")
    for name, value in (
        ("r", encoding.r),
        ("alignment_logits", encoding.alignment_logits),
        ("event_logits", encoding.event_logits),
    ):
        if not value.is_floating_point():
            raise TypeError(f"encoding.{name} must use a floating dtype")
        if value.dtype != encoding.r.dtype or value.device != encoding.r.device:
            raise ValueError(f"encoding.{name} must share encoding.r dtype and device")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"encoding.{name} must contain only finite values")
    if encoding.event_valid.device != encoding.r.device:
        raise ValueError("encoding.event_valid must share the encoding device")

    invalid = ~encoding.event_valid
    sentinel = torch.finfo(encoding.alignment_logits.dtype).min
    invalid_alignment = encoding.alignment_logits[:, :-1].masked_select(invalid)
    if bool((invalid_alignment != sentinel).any().item()):
        raise ValueError(
            "invalid encoding alignment logits must equal the finite mask sentinel"
        )
    invalid_events = invalid[..., None].expand_as(encoding.event_logits)
    if bool((encoding.event_logits.masked_select(invalid_events) != 0).any().item()):
        raise ValueError("invalid encoding event logits must be exactly zero")
    return batch, event_count


def _target_tensor(
    value: Tensor,
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if not torch.is_tensor(value) or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype} dtype")
    if value.device != device:
        raise ValueError(f"{name} must share the encoding device")
    return value


def _validate_alignment_target(
    target: Tensor,
    valid: Tensor,
    event_valid: Tensor,
) -> None:
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("alignment_target must contain only finite values")
    if bool((target < 0).any().item()):
        raise ValueError("alignment_target must be nonnegative")
    if bool((target[~valid] != 0).any().item()):
        raise ValueError("masked alignment_target rows must use the zero placeholder")
    if bool((target[:, :-1].masked_select(~event_valid) != 0).any().item()):
        raise ValueError("alignment_target must assign zero mass to invalid events")
    selected = target[valid]
    if selected.numel() == 0:
        return
    tolerance = 8.0 * float(torch.finfo(target.dtype).eps)
    if not bool(
        torch.allclose(
            selected.sum(dim=-1),
            torch.ones(
                selected.shape[0], dtype=target.dtype, device=target.device
            ),
            atol=tolerance,
            rtol=tolerance,
        )
    ):
        raise ValueError("valid alignment_target rows must sum to one")


def _validate_auxiliary_target(
    target: Tensor,
    valid: Tensor,
    event_valid: Tensor,
    name: str,
) -> None:
    if bool((valid & ~event_valid).any().item()):
        raise ValueError(f"{name}_valid must be contained by encoding.event_valid")
    if bool((target & ~valid).any().item()):
        raise ValueError(f"masked {name}_target entries must use the false placeholder")


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(dtype=values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1)


def task_loss(
    encoding: TaskEncoding,
    *,
    alignment_target: Tensor,
    alignment_valid: Tensor,
    rho_target: Tensor,
    rho_valid: Tensor,
    nu_target: Tensor,
    nu_valid: Tensor,
    eligibility_target: Tensor,
    eligibility_valid: Tensor,
    config: MethodConfig,
) -> Tensor:
    """Return the four independently averaged, configured task losses."""

    if not isinstance(config, MethodConfig):
        raise TypeError("config must be an explicit resolved MethodConfig")
    batch, event_count = _validate_encoding(encoding)
    device = encoding.event_logits.device

    if not torch.is_tensor(alignment_target) or alignment_target.shape != (
        batch,
        event_count + 1,
    ):
        raise ValueError("alignment_target must have shape [B,L+1]")
    if not alignment_target.is_floating_point():
        raise TypeError("alignment_target must use a floating dtype")
    if alignment_target.device != device:
        raise ValueError("alignment_target must share the encoding device")
    alignment_valid = _target_tensor(
        alignment_valid,
        "alignment_valid",
        (batch,),
        device=device,
        dtype=torch.bool,
    )
    _validate_alignment_target(
        alignment_target,
        alignment_valid,
        encoding.event_valid,
    )

    auxiliary = []
    for name, target, valid in (
        ("rho", rho_target, rho_valid),
        ("nu", nu_target, nu_valid),
        ("eligibility", eligibility_target, eligibility_valid),
    ):
        target = _target_tensor(
            target,
            f"{name}_target",
            (batch, event_count),
            device=device,
            dtype=torch.bool,
        )
        valid = _target_tensor(
            valid,
            f"{name}_valid",
            (batch, event_count),
            device=device,
            dtype=torch.bool,
        )
        _validate_auxiliary_target(target, valid, encoding.event_valid, name)
        auxiliary.append((target, valid))

    alignment_values = -(
        alignment_target.to(dtype=encoding.alignment_logits.dtype)
        * F.log_softmax(encoding.alignment_logits, dim=-1)
    ).sum(dim=-1)
    alignment = _masked_mean(alignment_values, alignment_valid)

    event_terms = []
    for channel, (target, valid) in enumerate(auxiliary):
        values = F.binary_cross_entropy_with_logits(
            encoding.event_logits[..., channel],
            target.to(dtype=encoding.event_logits.dtype),
            reduction="none",
        )
        event_terms.append(_masked_mean(values, valid))

    return (
        config.losses.alignment_weight * alignment
        + config.losses.occurrence_weight * event_terms[0]
        + config.losses.relation_weight * event_terms[1]
        + config.losses.eligibility_weight * event_terms[2]
    )


__all__ = ["task_loss"]
