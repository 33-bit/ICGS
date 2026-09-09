"""Closed, branch-local physical dynamics rollouts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor

from icgs.configuration.method import MethodConfig
from icgs.contracts.method import PhysicalPrediction, TimedCommand
from icgs.geometry.se3 import se3_exp
from icgs.models.decoders.physical import DecodedCloud
from icgs.models.encoders.physical import EncodedCloud
from icgs.models.dynamics.physical import validate_head_id
from icgs.models.memories.physical import action_descriptor, proprioception
from icgs.state.physical import PhysicalState, validate_next_boundary


def _require_finite(value: Tensor, name: str) -> None:
    if not torch.is_tensor(value) or not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"physical rollout produced nonfinite {name}")


def _dynamics_outputs(
    value: Any,
    batch: int,
    *,
    anchor_count: int,
    width: int,
    point_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError("dynamics must return (geometry_delta, pose_grip)")
    geometry_delta, pose_grip = value
    geometry_output_dim = width + point_dim
    pose_output_dim = 2 * point_dim + 1
    if not torch.is_tensor(geometry_delta) or geometry_delta.shape != (batch, anchor_count, geometry_output_dim):
        raise ValueError(
            f"geometry_delta must have shape [B,{anchor_count},{geometry_output_dim}]"
        )
    if not torch.is_tensor(pose_grip) or pose_grip.shape != (batch, pose_output_dim):
        raise ValueError(f"pose_grip must have shape [B,{pose_output_dim}]")
    if geometry_delta.device != device or pose_grip.device != device:
        raise ValueError("dynamics outputs must share the physical state device")
    if geometry_delta.dtype != dtype or pose_grip.dtype != dtype:
        raise ValueError("dynamics outputs must share the physical state dtype")
    _require_finite(geometry_delta, "geometry residuals")
    _require_finite(pose_grip, "pose/grip residuals")
    return geometry_delta, pose_grip


class PhysicalRollout:
    """Own explicit model collaborators for one physical rollout capability.

    The object is intentionally bound by outer composition.  No collaborator is
    placed in :class:`PhysicalState`, and this module keeps no process-global
    model or encoder/decoder binding.
    """

    def __init__(
        self,
        dynamics: Callable[..., Any],
        encoder: Callable[..., EncodedCloud],
        decoder: Callable[..., DecodedCloud],
        physical_memory: Any,
        *,
        config: MethodConfig,
    ) -> None:
        if not isinstance(config, MethodConfig):
            raise TypeError("config must be a resolved MethodConfig")
        if not callable(dynamics):
            raise TypeError("dynamics must be callable")
        if not callable(encoder):
            raise TypeError("encoder must be callable")
        if not callable(decoder):
            raise TypeError("decoder must be callable")
        if not callable(physical_memory) or not callable(getattr(physical_memory, "physical_tokens", None)):
            raise TypeError("physical_memory must be callable and expose physical_tokens")
        self.dynamics = dynamics
        self.encoder = encoder
        self.decoder = decoder
        self.physical_memory = physical_memory
        self.config = config
        dimensions = config.derived_dimensions()
        self.anchor_count = config.geometry.num_anchors
        self.width = config.geometry.width
        self.point_dim = config.geometry.point_dim
        self.physical_token_count = dimensions["physical_tokens"]
        self.descriptor_dim = config.memory.descriptor_dim

    def predict_step(
        self,
        state: PhysicalState,
        command: TimedCommand,
        *,
        head_id: int,
    ) -> PhysicalPrediction:
        """Predict one interval from a branch-local physical state."""

        if not isinstance(state, PhysicalState):
            raise TypeError("state must be a PhysicalState")
        if not isinstance(command, TimedCommand):
            raise TypeError("command must be a TimedCommand")
        validate_head_id(head_id)

        branch = state.branch_copy()
        tokens, token_valid = self.physical_memory.physical_tokens(branch)
        u_before = action_descriptor(branch.T_w_e, command, config=self.config)
        dynamics_value = self.dynamics(tokens, token_valid, u_before, head_id)
        geometry_delta, pose_grip = _dynamics_outputs(
            dynamics_value,
            branch.X.shape[0],
            anchor_count=self.anchor_count,
            width=self.width,
            point_dim=self.point_dim,
            device=branch.X.device,
            dtype=branch.X.dtype,
        )

        geometry = self.config.geometry
        feature_slice = slice(0, self.width)
        point_slice = slice(self.width, self.width + self.point_dim)
        X_tilde = branch.X + geometry_delta[..., feature_slice]
        x_tilde = branch.x + geometry.ell0_m * geometry_delta[..., point_slice]
        residual_encoded = EncodedCloud(X=X_tilde, x=x_tilde, anchor_valid=branch.valid)

        decoded = self.decoder(residual_encoded)
        if not isinstance(decoded, DecodedCloud):
            raise TypeError("decoder must return a DecodedCloud")
        _require_finite(decoded.points_w, "decoded cloud")
        if decoded.point_valid.dtype != torch.bool:
            raise ValueError("decoded point_valid must be boolean")
        encoded_next = self.encoder(decoded.points_w, decoded.point_valid)
        if not isinstance(encoded_next, EncodedCloud):
            raise TypeError("encoder must return an EncodedCloud")

        rotation_start = self.point_dim
        grip_start = 2 * self.point_dim
        unscaled_twist = torch.cat(
            (
                pose_grip[:, :self.point_dim] * geometry.ell0_m,
                pose_grip[:, rotation_start:grip_start],
            ),
            dim=-1,
        )
        T_next = branch.T_w_e @ se3_exp(unscaled_twist)
        grip_next = (pose_grip[:, grip_start:grip_start + 1] >= 0).to(dtype=branch.grip.dtype)
        gravity_start = 3 * self.point_dim + 1
        gravity = branch.p[:, gravity_start:gravity_start + self.point_dim]
        p_next = proprioception(T_next, grip_next, gravity, config=self.config)
        memory_next = self.physical_memory(
            encoded_next.X,
            encoded_next.anchor_valid,
            p_next,
            u_before,
            branch.memory,
        )

        next_boundary = branch.boundary + 1
        validate_next_boundary(branch.boundary, next_boundary)
        for name, value in (
            ("encoded state", encoded_next.X),
            ("encoded anchors", encoded_next.x),
            ("next pose", T_next),
            ("next proprioception", p_next),
            ("next memory", memory_next),
        ):
            _require_finite(value, name)
        next_state = PhysicalState(
            X=encoded_next.X,
            x=encoded_next.x,
            valid=encoded_next.anchor_valid,
            p=p_next,
            memory=memory_next,
            T_w_e=T_next,
            grip=grip_next,
            cached_world_cloud=decoded.points_w,
            cached_world_cloud_valid=decoded.point_valid,
            boundary=next_boundary,
            encoder_lineage=branch.encoder_lineage,
            memory_lineage=branch.memory_lineage,
            origin="imagined",
        )
        return PhysicalPrediction(
            next_state=next_state,
            grip_logits=pose_grip[:, grip_start:grip_start + 1],
            head_id=head_id,
        )

    def rollout(
        self,
        state: PhysicalState,
        commands: Iterable[TimedCommand],
        *,
        head_id: int,
    ) -> tuple[PhysicalPrediction, ...]:
        """Recursively decode, re-encode and update one fixed-head branch."""

        validate_head_id(head_id)
        commands = tuple(commands)
        if not commands:
            raise ValueError("physical rollout requires at least one command")
        current = state
        predictions = []
        for command in commands:
            prediction = self.predict_step(current, command, head_id=head_id)
            if prediction.head_id != head_id:
                raise ValueError("physical rollout head changed within a branch")
            predictions.append(prediction)
            current = prediction.next_state
        return tuple(predictions)


def bind_predict_step(
    dynamics: Callable[..., Any],
    encoder: Callable[..., EncodedCloud],
    decoder: Callable[..., DecodedCloud],
    physical_memory: Any,
    *,
    config: MethodConfig,
) -> Callable[..., PhysicalPrediction]:
    """Return a bound three-argument ``predict_step`` capability.

    A bound method supplies the normative ``(state, command, *, head_id)``
    signature while keeping model ownership in the composition boundary.
    """

    return PhysicalRollout(
        dynamics,
        encoder,
        decoder,
        physical_memory,
        config=config,
    ).predict_step


__all__ = ["PhysicalRollout", "bind_predict_step"]
