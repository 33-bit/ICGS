"""Width-256 physical point-cloud encoder for the added method path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from icgs.configuration.method import MethodConfig
from icgs.data.preprocessing.physical import (
    PhysicalPreprocessConfig,
    PreparedPhysicalCloud,
    prepare_physical_cloud,
)
from icgs.models.layers.method import GeometryBlock


@dataclass(frozen=True)
class EncodedCloud:
    """Encoded anchors and their validity mask in world metric coordinates."""

    X: Tensor
    x: Tensor
    anchor_valid: Tensor


def _config(
    config: PhysicalPreprocessConfig | Mapping[str, Any] | MethodConfig | None,
    method_config: MethodConfig,
) -> PhysicalPreprocessConfig:
    if config is None:
        return PhysicalPreprocessConfig.from_method_config(method_config)
    if isinstance(config, MethodConfig):
        return PhysicalPreprocessConfig.from_method_config(config)
    if isinstance(config, PhysicalPreprocessConfig):
        return config
    if isinstance(config, Mapping):
        return PhysicalPreprocessConfig(method_config=method_config, **dict(config))
    raise TypeError("config must be PhysicalPreprocessConfig, mapping, or None")


class PhysicalEncoder(nn.Module):
    """Encode deterministic FPS anchors with masked local neighborhoods."""

    def __init__(
        self,
        config: PhysicalPreprocessConfig | Mapping[str, Any] | MethodConfig | None = None,
        *,
        method_config: MethodConfig | None = None,
    ):
        super().__init__()
        if method_config is not None and not isinstance(method_config, MethodConfig):
            raise TypeError("method_config must be a MethodConfig or None")
        if isinstance(config, MethodConfig):
            if method_config is not None:
                raise ValueError("a MethodConfig cannot be supplied twice")
            method_config = config
            config = None
        resolved = method_config if method_config is not None else MethodConfig()
        self.config = _config(config, resolved)
        geometry = resolved.geometry
        neural = resolved.neural
        self.width = geometry.width
        local_hidden_dims = geometry.local_hidden_dims
        local_layers: list[nn.Module] = []
        input_width = 2 * geometry.point_dim
        for hidden_width in local_hidden_dims:
            local_layers.extend((nn.Linear(input_width, hidden_width), nn.GELU()))
            input_width = hidden_width
        local_layers.append(nn.Linear(input_width, self.width))
        self.local = nn.Sequential(*local_layers)
        self.geometry_blocks = nn.ModuleList(
            GeometryBlock(
                width=self.width,
                heads=neural.attention_heads,
                ffn_width=neural.ffn_width,
                geometry_config=geometry,
                neural_config=neural,
            )
            for _ in range(geometry.transformer_layers)
        )

    def _encode_prepared(self, prepared: PreparedPhysicalCloud) -> EncodedCloud:
        neighbors = prepared.neighbors
        anchors = prepared.anchors
        anchor_coordinates = anchors / self.config.ell0
        neighbor_coordinates = neighbors / self.config.ell0
        relative = neighbor_coordinates - anchor_coordinates[:, :, None, :]
        local_input = torch.cat((relative, anchor_coordinates[:, :, None, :].expand_as(relative)), dim=-1)
        local_features = self.local(local_input)
        neighbor_mask = prepared.neighbor_valid[..., None]
        masked_features = torch.where(
            neighbor_mask,
            local_features,
            torch.full_like(local_features, -torch.inf),
        )
        X = masked_features.amax(dim=2)
        has_neighbor = prepared.neighbor_valid.any(dim=2, keepdim=True)
        X = torch.where(has_neighbor, X, torch.zeros_like(X))
        for block in self.geometry_blocks:
            X = block(X, anchor_coordinates, prepared.anchor_valid)
        X = torch.where(prepared.anchor_valid[..., None], X, torch.zeros_like(X))
        return EncodedCloud(X=X, x=anchors, anchor_valid=prepared.anchor_valid)

    def forward(self, points_w: Tensor, point_valid: Tensor) -> EncodedCloud:
        prepared = prepare_physical_cloud(points_w, point_valid, self.config)
        return self._encode_prepared(prepared)


def encode_cloud(
    points_w: Tensor,
    point_valid: Tensor,
    *,
    encoder: PhysicalEncoder | None = None,
) -> EncodedCloud:
    """Call an explicitly supplied encoder, or a fresh default encoder."""

    return (encoder if encoder is not None else PhysicalEncoder())(points_w, point_valid)


__all__ = ["EncodedCloud", "PhysicalEncoder", "encode_cloud"]
