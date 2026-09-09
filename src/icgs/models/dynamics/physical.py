"""Physical-only action-conditioned dynamics for the added method path."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from icgs.configuration.method import MethodConfig


def validate_head_id(head_id: int) -> None:
    """Require one of the three fixed bootstrap heads."""

    if isinstance(head_id, bool) or not isinstance(head_id, int) or head_id not in (0, 1, 2):
        raise ValueError("head_id must be one of the three fixed heads")


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"P07 supports only the central GELU activation, got {name!r}")


class _SelfAttentionBlock(nn.Module):
    """Masked pre-LN self-attention with zeroed invalid query rows."""

    def __init__(self, *, width: int, heads: int, ffn_width: int,
                 dropout: float, layer_norm_eps: float, activation: str):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width, eps=layer_norm_eps)
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout_attention = nn.Dropout(dropout)
        self.norm_feedforward = nn.LayerNorm(width, eps=layer_norm_eps)
        self.feedforward = nn.Sequential(
            nn.Linear(width, ffn_width),
            _activation(activation),
            nn.Linear(ffn_width, width),
        )
        self.dropout_feedforward = nn.Dropout(dropout)

    def forward(self, values: Tensor, valid: Tensor) -> Tensor:
        row_mask = valid[..., None]
        values = torch.where(row_mask, values, torch.zeros_like(values))
        normalized = self.norm_attention(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=False,
        )
        values = values + self.dropout_attention(attended)
        values = torch.where(row_mask, values, torch.zeros_like(values))

        feedforward = self.feedforward(self.norm_feedforward(values))
        values = values + self.dropout_feedforward(feedforward)
        return torch.where(row_mask, values, torch.zeros_like(values))


class _ResidualHead(nn.Module):
    """One independent two-layer residual head."""

    def __init__(self, *, width: int, output_dim: int, residual_rows: int,
                 residual_init_std: float, activation: str):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(width, width),
            _activation(activation),
        )
        self.output = nn.Linear(width, output_dim)
        with torch.no_grad():
            self.output.weight[:residual_rows].normal_(std=residual_init_std)
            self.output.bias[:residual_rows].zero_()

    def forward(self, values: Tensor) -> Tensor:
        return self.output(self.hidden(values))


class PhysicalDynamics(nn.Module):
    """Four-block physical trunk with three fixed, independent output heads.

    ``forward`` consumes only P04 physical tokens, their validity mask, and the
    configured absolute-command descriptor.  It returns geometry
    residuals ``[B,128,259]`` followed by achieved pose/grip outputs ``[B,7]``.
    """

    def __init__(self, config: MethodConfig):
        super().__init__()
        if not isinstance(config, MethodConfig):
            raise TypeError("config must be a resolved MethodConfig")

        dimensions = config.derived_dimensions()
        self.config = config
        geometry = config.geometry
        neural = config.neural
        dynamics = config.dynamics
        width = geometry.width
        if dynamics.width != width:
            raise ValueError("dynamics width must match the physical token width")
        if dynamics.heads != 3:
            raise ValueError("P07 requires exactly three fixed dynamics heads")

        self.width = width
        self.anchor_count = geometry.num_anchors
        self.physical_token_count = dimensions["physical_tokens"]
        self.dynamics_token_count = dimensions["dynamics_tokens"]
        self.descriptor_dim = config.memory.descriptor_dim
        self.geometry_output_dim = width + geometry.point_dim
        self.pose_output_dim = 2 * geometry.point_dim + 1
        self.residual_init_std = dynamics.residual_init_std
        self.action_mlp = nn.Sequential(
            nn.Linear(self.descriptor_dim, width),
            _activation(neural.activation),
            nn.Linear(width, width),
        )
        self.action_type = nn.Parameter(torch.zeros(1, width))
        self.trunk = nn.ModuleList(
            _SelfAttentionBlock(
                width=width,
                heads=neural.attention_heads,
                ffn_width=neural.ffn_width,
                dropout=neural.dropout,
                layer_norm_eps=neural.layer_norm_eps,
                activation=neural.activation,
            )
            for _ in range(dynamics.transformer_layers)
        )
        self.geometry_heads = nn.ModuleList(
            _ResidualHead(
                width=width,
                output_dim=self.geometry_output_dim,
                residual_rows=self.geometry_output_dim,
                residual_init_std=dynamics.residual_init_std,
                activation=neural.activation,
            )
            for _ in range(dynamics.heads)
        )
        self.pose_heads = nn.ModuleList(
            _ResidualHead(
                width=width,
                output_dim=self.pose_output_dim,
                residual_rows=self.pose_output_dim - 1,
                residual_init_std=dynamics.residual_init_std,
                activation=neural.activation,
            )
            for _ in range(dynamics.heads)
        )

    def forward(self, S: Tensor, valid: Tensor, u: Tensor, head_id: int) -> tuple[Tensor, Tensor]:
        """Predict physical geometry and achieved pose/grip residuals."""

        validate_head_id(head_id)
        if not torch.is_tensor(S) or S.ndim != 3 or S.shape[1:] != (self.physical_token_count, self.width):
            raise ValueError(f"S must have shape [B,{self.physical_token_count},{self.width}]")
        if not torch.is_tensor(valid) or valid.shape != S.shape[:2] or valid.dtype != torch.bool:
            raise ValueError(f"valid must have shape [B,{self.physical_token_count}] and boolean dtype")
        if not torch.is_tensor(u) or u.shape != (S.shape[0], self.descriptor_dim):
            raise ValueError(f"u must have shape [B,{self.descriptor_dim}]")
        if not S.is_floating_point() or not u.is_floating_point():
            raise TypeError("S and u must use floating dtypes")
        if S.device != u.device or S.dtype != u.dtype:
            raise ValueError("S and u must share device and dtype")
        if not bool(torch.isfinite(S).all().item()) or not bool(torch.isfinite(u).all().item()):
            raise ValueError("S and u must contain only finite values")
        if not bool(valid[:, :self.anchor_count].any(dim=1).all().item()):
            raise ValueError("valid must include at least one valid geometry anchor per sample")

        physical_mask = valid[..., None]
        physical_tokens = torch.where(physical_mask, S, torch.zeros_like(S))
        action_token = self.action_mlp(u).unsqueeze(1) + self.action_type.unsqueeze(0)
        tokens = torch.cat((physical_tokens, action_token), dim=1)
        token_valid = torch.cat(
            (valid, torch.ones(S.shape[0], 1, dtype=torch.bool, device=S.device)),
            dim=1,
        )
        for block in self.trunk:
            tokens = block(tokens, token_valid)

        geometry_delta = self.geometry_heads[head_id](tokens[:, :self.anchor_count])
        geometry_delta = torch.where(
            valid[:, :self.anchor_count, None], geometry_delta, torch.zeros_like(geometry_delta)
        )
        pose_grip = self.pose_heads[head_id](tokens[:, self.anchor_count])
        if not bool(torch.isfinite(geometry_delta).all().item()) or not bool(torch.isfinite(pose_grip).all().item()):
            raise FloatingPointError("physical dynamics produced nonfinite residuals")
        return geometry_delta, pose_grip


__all__ = ["PhysicalDynamics", "validate_head_id"]
