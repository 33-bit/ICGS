"""Patch decoder for world-frame physical geometry."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from icgs.configuration.method import DecoderConfig, GeometryConfig, MethodConfig, NeuralConfig
from icgs.models.encoders.physical import EncodedCloud


class DecodedCloud:
    """Fixed patch storage together with the mask of emitted world points."""

    def __init__(self, points_w: Tensor, point_valid: Tensor):
        self.points_w = points_w
        self.point_valid = point_valid

    @property
    def points(self) -> Tensor:
        return self.points_w

    @property
    def valid(self) -> Tensor:
        return self.point_valid


class PhysicalDecoder(nn.Module):
    """Decode sixteen fixed grid patches from every valid anchor."""

    def __init__(
        self,
        width: int | None = None,
        *,
        method_config: MethodConfig | None = None,
        geometry_config: GeometryConfig | None = None,
        neural_config: NeuralConfig | None = None,
        decoder_config: DecoderConfig | None = None,
    ):
        super().__init__()
        if method_config is not None:
            if not isinstance(method_config, MethodConfig):
                raise TypeError("method_config must be a MethodConfig or None")
            if any(section is not None for section in (geometry_config, neural_config, decoder_config)):
                raise ValueError("method_config cannot be combined with typed sections")
            geometry_config = method_config.geometry
            neural_config = method_config.neural
            decoder_config = method_config.decoder
        if any(section is None for section in (geometry_config, neural_config, decoder_config)):
            resolved = MethodConfig()
            geometry_config = resolved.geometry if geometry_config is None else geometry_config
            neural_config = resolved.neural if neural_config is None else neural_config
            decoder_config = resolved.decoder if decoder_config is None else decoder_config
        if not isinstance(geometry_config, GeometryConfig):
            raise TypeError("geometry_config must be a GeometryConfig or None")
        if not isinstance(neural_config, NeuralConfig):
            raise TypeError("neural_config must be a NeuralConfig or None")
        if not isinstance(decoder_config, DecoderConfig):
            raise TypeError("decoder_config must be a DecoderConfig or None")
        width = geometry_config.width if width is None else width
        if width != 256:
            raise ValueError("the physical decoder width is fixed at 256")
        if decoder_config.grid_side != 4:
            raise ValueError("the physical decoder grid is fixed at 4x4")
        if tuple(decoder_config.hidden_dims) != (256, 128):
            raise ValueError("the physical decoder hidden widths are fixed at [256, 128]")
        self.norm = nn.LayerNorm(width, eps=neural_config.layer_norm_eps)
        hidden_dims = decoder_config.hidden_dims
        patch_layers: list[nn.Module] = []
        input_width = width + 2
        for hidden_width in hidden_dims:
            patch_layers.extend((nn.Linear(input_width, hidden_width), nn.GELU()))
            input_width = hidden_width
        patch_layers.append(nn.Linear(input_width, 3))
        self.patch = nn.Sequential(*patch_layers)
        values = torch.tensor(
            tuple(
                (decoder_config.grid_min * (decoder_config.grid_side - 1 - index)
                 + decoder_config.grid_max * index) / (decoder_config.grid_side - 1)
                for index in range(decoder_config.grid_side)
            ),
            dtype=torch.float32,
        )
        self.register_buffer("grid", torch.stack((
            values.repeat_interleave(4),
            values.repeat(4),
        ), dim=-1), persistent=False)
        self.patch_radius_m = decoder_config.patch_radius_m

    def forward(self, encoded: EncodedCloud) -> DecodedCloud:
        if not isinstance(encoded, EncodedCloud):
            raise TypeError("encoded must be an EncodedCloud")
        if encoded.X.ndim != 3 or encoded.X.shape[0] == 0 or encoded.X.shape[1:] != (128, 256):
            raise ValueError("encoded.X must have shape [B,128,256] with a nonempty batch")
        if encoded.x.shape != (encoded.X.shape[0], 128, 3) or encoded.anchor_valid.shape != (encoded.X.shape[0], 128):
            raise ValueError("encoded anchor shapes must be [B,128,3] and [B,128]")
        if encoded.anchor_valid.dtype != torch.bool:
            raise ValueError("anchor_valid must be boolean")
        anchor_valid = encoded.anchor_valid.to(device=encoded.X.device)
        if not bool(anchor_valid.any(dim=1).all().item()):
            raise ValueError("decoder requires at least one valid anchor per sample")
        valid_X = anchor_valid[..., None].expand_as(encoded.X)
        valid_x = anchor_valid[..., None].expand_as(encoded.x)
        if not bool(torch.isfinite(encoded.X.masked_select(valid_X)).all().item()):
            raise ValueError("valid encoded X must be finite")
        if not bool(torch.isfinite(encoded.x.masked_select(valid_x)).all().item()):
            raise ValueError("valid encoded x must be finite")
        rows = anchor_valid[..., None]
        features = torch.where(rows, encoded.X, torch.zeros_like(encoded.X))
        normalized = self.norm(features)
        grid = self.grid.to(dtype=encoded.X.dtype, device=encoded.X.device)
        grid = grid[None, None, :, :].expand(encoded.X.shape[0], encoded.X.shape[1], -1, -1)
        decoder_input = torch.cat((normalized[:, :, None, :].expand(-1, -1, 16, -1), grid), dim=-1)
        patches = encoded.x[:, :, None, :] + self.patch_radius_m * torch.tanh(self.patch(decoder_input))
        point_valid = anchor_valid[:, :, None].expand(-1, -1, 16)
        patches = torch.where(point_valid[..., None], patches, torch.zeros_like(patches))
        return DecodedCloud(
            points_w=patches.reshape(encoded.X.shape[0], -1, 3),
            point_valid=point_valid.reshape(encoded.X.shape[0], -1),
        )


def decode_cloud(encoded: EncodedCloud, *, decoder: PhysicalDecoder | None = None) -> DecodedCloud:
    """Call an explicitly supplied decoder, or a fresh default decoder."""

    return (decoder if decoder is not None else PhysicalDecoder())(encoded)


__all__ = ["DecodedCloud", "PhysicalDecoder", "decode_cloud"]
