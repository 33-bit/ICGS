"""Small masked neural blocks owned by the added ICGS method components."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def masked_mean(values: Tensor, valid: Tensor, dim: int) -> Tensor:
    """Average only valid entries and reject an empty reduction."""

    if not torch.is_tensor(values) or not torch.is_tensor(valid):
        raise TypeError("values and valid must be tensors")
    if not values.is_floating_point():
        raise TypeError("values must use a floating dtype")
    if valid.ndim > values.ndim or valid.shape[:valid.ndim] != values.shape[:valid.ndim]:
        raise ValueError("valid must match the leading value dimensions")
    dim = dim if dim >= 0 else values.ndim + dim
    if dim < 0 or dim >= values.ndim:
        raise ValueError("dim is outside values")
    mask = valid.to(dtype=torch.bool, device=values.device)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    try:
        mask = mask.expand_as(values)
    except RuntimeError as exc:
        raise ValueError("valid cannot broadcast to values") from exc
    count = mask.sum(dim=dim)
    if bool((count == 0).any().item()):
        raise ValueError("masked_mean cannot reduce an empty set")
    masked_values = torch.where(mask, values, torch.zeros_like(values))
    return masked_values.sum(dim=dim) / count.to(dtype=values.dtype)


class GeometryBlock(nn.Module):
    """Pre-LN self-attention with learned relative geometry bias."""

    def __init__(self, width: int = 256, heads: int = 8, ffn_width: int = 1024):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.norm_attention = nn.LayerNorm(width, eps=1e-5)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.geometry_bias = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, heads),
        )
        self.norm_feedforward = nn.LayerNorm(width, eps=1e-5)
        self.feedforward = nn.Sequential(
            nn.Linear(width, ffn_width),
            nn.GELU(),
            nn.Linear(ffn_width, width),
        )

    def forward(self, values: Tensor, coordinates: Tensor, valid: Tensor) -> Tensor:
        if values.ndim != 3 or coordinates.shape != values.shape[:2] + (3,):
            raise ValueError("values and coordinates must have shapes [B,N,C] and [B,N,3]")
        if valid.shape != values.shape[:2]:
            raise ValueError("valid must have shape [B,N]")
        valid = valid.to(dtype=torch.bool, device=values.device)
        if bool((~valid).all(dim=1).any().item()):
            raise ValueError("geometry attention requires at least one valid key per sample")
        row_mask = valid[..., None]
        values = torch.where(row_mask, values, torch.zeros_like(values))
        coordinates = torch.where(row_mask, coordinates, torch.zeros_like(coordinates))

        normalized = self.norm_attention(values)
        qkv = self.qkv(normalized).reshape(values.shape[0], values.shape[1], 3, self.heads, self.head_width)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_width ** 0.5)
        relative = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        bias = self.geometry_bias(relative).permute(0, 3, 1, 2)
        scores = scores + bias
        scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, v).transpose(1, 2).reshape_as(values)
        values = values + self.output(attended)
        values = torch.where(row_mask, values, torch.zeros_like(values))

        feedforward_input = self.norm_feedforward(values)
        values = values + self.feedforward(feedforward_input)
        return torch.where(row_mask, values, torch.zeros_like(values))


__all__ = ["GeometryBlock", "masked_mean"]
