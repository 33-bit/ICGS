"""Width-256 physical point-cloud encoder for the added method path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

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


def _config(config: PhysicalPreprocessConfig | Mapping[str, Any] | None) -> PhysicalPreprocessConfig:
    if config is None:
        return PhysicalPreprocessConfig()
    if isinstance(config, PhysicalPreprocessConfig):
        return config
    if isinstance(config, Mapping):
        return PhysicalPreprocessConfig(**dict(config))
    raise TypeError("config must be PhysicalPreprocessConfig, mapping, or None")


class PhysicalEncoder(nn.Module):
    """Encode deterministic FPS anchors with masked local neighborhoods."""

    def __init__(self, config: PhysicalPreprocessConfig | Mapping[str, Any] | None = None):
        super().__init__()
        self.config = _config(config)
        self.width = 256
        self.local = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
        )
        self.geometry_blocks = nn.ModuleList((GeometryBlock(256, 8, 1024), GeometryBlock(256, 8, 1024)))

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
