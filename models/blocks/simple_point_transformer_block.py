from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _batch_gather(data: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Memory-efficient gather via fancy indexing (avoids [B,N,N,C] intermediates)."""
    B = data.shape[0]
    device = data.device
    if index.dim() == 2:
        batch_idx = torch.arange(B, device=device).view(B, 1)
        return data[batch_idx, index]
    if index.dim() == 3:
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, index.shape[1], index.shape[2])
        return data[batch_idx, index]
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
    """Point Transformer block with Vector Attention (PTv1 paper).

    Vector Attention: each feature channel gets an independent attention weight
    (softmax over neighbours, per-channel).  This is the original PTv1 formulation
    — more expressive but O(C²) parameters and prone to overfitting on small datasets.

    For a more efficient variant see ``PointTransformerV2Block`` which uses Grouped
    Vector Attention (GVA) to share weights across channel groups.
    """

    def __init__(self, dim: int, num_heads: int, k: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # PTv1: scale by head_dim because we sum over head_dim in dot product
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
        B, N, C = features.shape
        K = self.k

        normalized = self.norm1(features)
        q = self.q_proj(normalized).view(B, N, self.num_heads, self.head_dim)

        neighbor_idx = _batched_knn(points, points, K, exclude_self=True)
        neighbor_features = _batch_gather(normalized, neighbor_idx)
        neighbor_points = _batch_gather(points, neighbor_idx)

        k = self.k_proj(neighbor_features).view(B, N, K, self.num_heads, self.head_dim)
        v = self.v_proj(neighbor_features).view(B, N, K, self.num_heads, self.head_dim)

        rel_pos = points.unsqueeze(2) - neighbor_points  # [B, N, K, 3]
        rel_pos = self.pos_mlp(rel_pos).view(B, N, K, self.num_heads, self.head_dim)

        # ---- Vector Attention (per-channel weights, NO sum over head_dim) ----
        # q.unsqueeze(2):    [B, N, 1, H, D]
        # k:                 [B, N, K, H, D]
        # rel_pos:           [B, N, K, H, D]
        # attn_logits:       [B, N, K, H, D]  ← per-channel logits
        attn_logits = (q.unsqueeze(2) - k + rel_pos) * self.scale
        attn = F.softmax(attn_logits, dim=2)  # softmax over neighbours, per-channel
        attn = self.attn_dropout(attn)         # [B, N, K, H, D]

        # Weighted sum: each channel uses its own attention weight
        out = ((v + rel_pos) * attn).sum(dim=2).reshape(B, N, C)

        features = features + self.out_proj(out)
        features = features + self.ffn(self.norm2(features))
        return features


SimplePointTransformerBlock = PointTransformerBlock
