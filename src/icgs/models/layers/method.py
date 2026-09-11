"""Small masked neural blocks owned by the added ICGS method components."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from icgs.configuration.method import GeometryConfig, MethodConfig, NeuralConfig


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


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"method attention supports only GELU, got {name!r}")


class MaskedCrossAttentionBlock(nn.Module):
    """Generic masked pre-LN query-to-key/value attention."""

    def __init__(self, *, width: int, neural_config: NeuralConfig):
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(neural_config, NeuralConfig):
            raise TypeError("neural_config must be an explicit NeuralConfig")
        heads = neural_config.attention_heads
        if heads <= 0 or width % heads:
            raise ValueError("neural attention heads must divide width")
        if not neural_config.pre_norm:
            raise ValueError("method attention requires pre_norm=True")

        self.width = width
        self.norm_query = nn.LayerNorm(width, eps=neural_config.layer_norm_eps)
        self.norm_keys = nn.LayerNorm(width, eps=neural_config.layer_norm_eps)
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=neural_config.dropout,
            batch_first=True,
        )
        self.dropout_attention = nn.Dropout(neural_config.dropout)
        self.norm_feedforward = nn.LayerNorm(width, eps=neural_config.layer_norm_eps)
        self.feedforward = nn.Sequential(
            nn.Linear(width, neural_config.ffn_width),
            _activation(neural_config.activation),
            nn.Linear(neural_config.ffn_width, width),
        )
        self.dropout_feedforward = nn.Dropout(neural_config.dropout)

    def forward(
        self,
        query: Tensor,
        query_valid: Tensor,
        keys: Tensor,
        key_valid: Tensor,
    ) -> Tensor:
        if (
            not torch.is_tensor(query)
            or query.ndim != 3
            or query.shape[-1] != self.width
            or query.shape[1] <= 0
        ):
            raise ValueError(f"query must have nonempty shape [B,Q,{self.width}]")
        if (
            not torch.is_tensor(keys)
            or keys.ndim != 3
            or keys.shape[0] != query.shape[0]
            or keys.shape[-1] != self.width
            or keys.shape[1] <= 0
        ):
            raise ValueError(f"keys must have nonempty shape [B,K,{self.width}]")
        if not query.is_floating_point() or not keys.is_floating_point():
            raise TypeError("query and keys must use floating dtypes")
        if query.dtype != keys.dtype or query.device != keys.device:
            raise ValueError("query and keys must share dtype and device")
        if not torch.is_tensor(query_valid) or query_valid.shape != query.shape[:2]:
            raise ValueError("query_valid must have shape [B,Q]")
        if not torch.is_tensor(key_valid) or key_valid.shape != keys.shape[:2]:
            raise ValueError("key_valid must have shape [B,K]")
        if query_valid.dtype != torch.bool or key_valid.dtype != torch.bool:
            raise TypeError("query_valid and key_valid must use torch.bool dtype")
        if query_valid.device != query.device or key_valid.device != keys.device:
            raise ValueError("attention masks must share their tensor device")
        if bool((~key_valid).all(dim=1).any().item()):
            raise ValueError("masked cross-attention requires at least one valid key")

        expanded_query_valid = query_valid[..., None].expand_as(query)
        expanded_key_valid = key_valid[..., None].expand_as(keys)
        if not bool(torch.isfinite(query.masked_select(expanded_query_valid)).all().item()):
            raise ValueError("valid query values must contain only finite values")
        if not bool(torch.isfinite(keys.masked_select(expanded_key_valid)).all().item()):
            raise ValueError("valid key values must contain only finite values")

        query_mask = query_valid[..., None]
        key_mask = key_valid[..., None]
        query = torch.where(query_mask, query, torch.zeros_like(query))
        keys = torch.where(key_mask, keys, torch.zeros_like(keys))
        normalized_query = self.norm_query(query)
        normalized_keys = self.norm_keys(keys)
        attended, _ = self.attention(
            normalized_query,
            normalized_keys,
            normalized_keys,
            key_padding_mask=~key_valid,
            need_weights=False,
        )
        query = query + self.dropout_attention(attended)
        query = torch.where(query_mask, query, torch.zeros_like(query))

        feedforward = self.feedforward(self.norm_feedforward(query))
        query = query + self.dropout_feedforward(feedforward)
        return torch.where(query_mask, query, torch.zeros_like(query))


class MaskedSelfAttentionBlock(nn.Module):
    """Generic masked pre-LN attention without spatial geometry bias."""

    def __init__(self, *, width: int, neural_config: NeuralConfig):
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(neural_config, NeuralConfig):
            raise TypeError("neural_config must be an explicit NeuralConfig")
        heads = neural_config.attention_heads
        if heads <= 0 or width % heads:
            raise ValueError("neural attention heads must divide width")
        if not neural_config.pre_norm:
            raise ValueError("method attention requires pre_norm=True")

        self.width = width
        self.norm_attention = nn.LayerNorm(width, eps=neural_config.layer_norm_eps)
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=neural_config.dropout,
            batch_first=True,
        )
        self.dropout_attention = nn.Dropout(neural_config.dropout)
        self.norm_feedforward = nn.LayerNorm(width, eps=neural_config.layer_norm_eps)
        self.feedforward = nn.Sequential(
            nn.Linear(width, neural_config.ffn_width),
            _activation(neural_config.activation),
            nn.Linear(neural_config.ffn_width, width),
        )
        self.dropout_feedforward = nn.Dropout(neural_config.dropout)

    def forward(self, values: Tensor, valid: Tensor) -> Tensor:
        if not torch.is_tensor(values) or values.ndim != 3 or values.shape[-1] != self.width:
            raise ValueError(f"values must have shape [B,L,{self.width}]")
        if not values.is_floating_point():
            raise TypeError("values must use a floating dtype")
        if not torch.is_tensor(valid) or valid.shape != values.shape[:2]:
            raise ValueError("valid must have shape [B,L]")
        if valid.dtype != torch.bool:
            raise TypeError("valid must use torch.bool dtype")
        if valid.device != values.device:
            raise ValueError("valid must share the values device")
        if bool((~valid).all(dim=1).any().item()):
            raise ValueError("masked attention requires at least one valid key per sample")

        expanded_valid = valid[..., None].expand_as(values)
        if not bool(torch.isfinite(values.masked_select(expanded_valid)).all().item()):
            raise ValueError("valid attention values must contain only finite values")

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


class GeometryBlock(nn.Module):
    """Pre-LN self-attention with learned relative geometry bias."""

    def __init__(
        self,
        width: int | None = None,
        heads: int | None = None,
        ffn_width: int | None = None,
        *,
        method_config: MethodConfig | None = None,
        geometry_config: GeometryConfig | None = None,
        neural_config: NeuralConfig | None = None,
    ):
        super().__init__()
        if method_config is not None:
            if not isinstance(method_config, MethodConfig):
                raise TypeError("method_config must be a MethodConfig or None")
            if geometry_config is not None or neural_config is not None:
                raise ValueError("method_config cannot be combined with typed sections")
            geometry_config = method_config.geometry
            neural_config = method_config.neural
        if geometry_config is None or neural_config is None:
            resolved = MethodConfig()
            geometry_config = resolved.geometry if geometry_config is None else geometry_config
            neural_config = resolved.neural if neural_config is None else neural_config
        if not isinstance(geometry_config, GeometryConfig):
            raise TypeError("geometry_config must be a GeometryConfig or None")
        if not isinstance(neural_config, NeuralConfig):
            raise TypeError("neural_config must be a NeuralConfig or None")
        width = geometry_config.width if width is None else width
        heads = neural_config.attention_heads if heads is None else heads
        ffn_width = neural_config.ffn_width if ffn_width is None else ffn_width
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.norm_attention = nn.LayerNorm(width, eps=neural_config.layer_norm_eps)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        bias_hidden_dim = neural_config.geometry_bias_hidden_dim
        self.geometry_bias = nn.Sequential(
            nn.Linear(3, bias_hidden_dim),
            nn.GELU(),
            nn.Linear(bias_hidden_dim, heads),
        )
        self.norm_feedforward = nn.LayerNorm(width, eps=neural_config.layer_norm_eps)
        self.feedforward = nn.Sequential(
            nn.Linear(width, ffn_width),
            nn.GELU(),
            nn.Linear(ffn_width, width),
        )
        self.dropout = nn.Dropout(neural_config.dropout)

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
        values = values + self.dropout(self.output(attended))
        values = torch.where(row_mask, values, torch.zeros_like(values))

        feedforward_input = self.norm_feedforward(values)
        values = values + self.dropout(self.feedforward(feedforward_input))
        return torch.where(row_mask, values, torch.zeros_like(values))


__all__ = [
    "GeometryBlock",
    "MaskedCrossAttentionBlock",
    "MaskedSelfAttentionBlock",
    "masked_mean",
]
