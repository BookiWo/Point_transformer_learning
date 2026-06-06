from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _batch_gather(data: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.dim() == 2:
        return data.gather(1, index.unsqueeze(-1).expand(-1, -1, data.shape[-1]))
    if index.dim() == 3:
        expanded = data.unsqueeze(1).expand(-1, index.shape[1], -1, -1)
        return torch.gather(expanded, 2, index.unsqueeze(-1).expand(-1, -1, -1, data.shape[-1]))
    raise ValueError(f"Unsupported index rank: {index.dim()}")


def _batched_knn(query_points: torch.Tensor, reference_points: torch.Tensor, k: int, exclude_self: bool = False) -> torch.Tensor:
    with torch.no_grad():
        if reference_points.shape[1] == 0:
            raise ValueError("reference_points must contain at least one point")

        distances = torch.cdist(query_points, reference_points)
        if exclude_self and query_points.shape == reference_points.shape and query_points.data_ptr() == reference_points.data_ptr():
            eye = torch.eye(reference_points.shape[1], device=reference_points.device, dtype=torch.bool).unsqueeze(0)
            distances = distances.masked_fill(eye, float("inf"))

        k = min(k, reference_points.shape[1])
        return distances.topk(k=k, dim=-1, largest=False).indices


class PointTransformerBlock(nn.Module):
    """Local neighborhood point transformer block for batched point clouds."""

    def __init__(self, dim: int, num_heads: int, k: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.k = k

        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(features)
        query = self.q_proj(normalized).view(features.shape[0], features.shape[1], self.num_heads, self.head_dim)

        neighbor_idx = _batched_knn(points, points, self.k, exclude_self=True)
        neighbor_features = _batch_gather(normalized, neighbor_idx)
        neighbor_points = _batch_gather(points, neighbor_idx)

        key = self.k_proj(neighbor_features).view(features.shape[0], features.shape[1], neighbor_idx.shape[-1], self.num_heads, self.head_dim)
        value = self.v_proj(neighbor_features).view(features.shape[0], features.shape[1], neighbor_idx.shape[-1], self.num_heads, self.head_dim)

        rel_pos = points.unsqueeze(2) - neighbor_points
        rel_pos = self.pos_mlp(rel_pos).view(features.shape[0], features.shape[1], neighbor_idx.shape[-1], self.num_heads, self.head_dim)

        attn_logits = ((query.unsqueeze(2) - key + rel_pos) * self.scale).sum(dim=-1)
        attn = F.softmax(attn_logits, dim=2)
        attn = self.attn_dropout(attn)

        out = ((value + rel_pos) * attn.unsqueeze(-1)).sum(dim=2).reshape(features.shape[0], features.shape[1], self.dim)
        features = features + self.out_proj(out)
        features = features + self.ffn(self.norm2(features))
        return features


SimplePointTransformerBlock = PointTransformerBlock
