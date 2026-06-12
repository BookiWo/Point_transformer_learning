"""
Point Transformer V2 Block — Grouped Vector Attention + Position Encoding Multiplier.

Key improvements over V1 (simple_point_transformer_block.py):
  1. Grouped Vector Attention (GVA): split head_dim into groups, group-wise shared
     attention weights → reduces O(C²) params to O(C²/g), prevents overfitting.
  2. Position Encoding Multiplier: multiplicative pos modulation on top of additive
     bias → PE becomes "tuner" not just "background."
  3. Bias-free Q/K/V projections (following PTv2 best practice).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.grid_pooling import _batch_gather, _batched_knn


class PointTransformerV2Block(nn.Module):
    """PTv2 local neighbourhood transformer block with GVA and PE Multiplier.

    Args:
        dim: feature dimension.
        num_heads: number of attention heads.
        num_groups: groups per head for GVA. 1 = scalar attn (like MHA),
                    ``head_dim`` = full vector attn (V1).
                    Recommended: 2–4 for best accuracy/efficiency trade-off.
        k: number of neighbours.
        dropout: attention + FFN dropout.
        pe_multiplier: enable multiplicative position encoding (default True).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int = 1,
        k: int = 16,
        dropout: float = 0.1,
        pe_multiplier: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        head_dim = dim // num_heads
        if head_dim % num_groups != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by num_groups ({num_groups})"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = head_dim
        self.group_dim = head_dim // num_groups
        self.scale = self.group_dim ** -0.5  # PTv2 scales by group_dim
        self.k = k
        self.pe_multiplier = pe_multiplier

        # --- bias=False following PTv2 best practice ---
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # Position encoding: when pe_multiplier=True output 2*dim (mul + bias),
        # otherwise dim (additive bias only, V1-compatible).
        pos_out = dim * 2 if pe_multiplier else dim
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, pos_out),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

        # FFN with expansion ratio 4
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

        # ---- pre-norm + projections ----
        normalized = self.norm1(features)
        q = self.q_proj(normalized)

        # kNN neighbour lookup
        neighbor_idx = _batched_knn(points, points, self.k, exclude_self=True)
        K = neighbor_idx.shape[-1]
        neighbor_features = _batch_gather(normalized, neighbor_idx)
        neighbor_points = _batch_gather(points, neighbor_idx)

        k = self.k_proj(neighbor_features)
        v = self.v_proj(neighbor_features)

        # ---- position encoding ----
        rel_pos = points.unsqueeze(2) - neighbor_points  # [B, N, K, 3]
        pos_enc = self.pos_mlp(rel_pos)  # [B, N, K, dim] or [B, N, K, 2*dim]

        if self.pe_multiplier:
            pos_mul, pos_bias = pos_enc.chunk(2, dim=-1)
            # reshape to [B, N, K, H, G, D//G]
            pos_mul = pos_mul.view(B, N, K, self.num_heads, self.num_groups, self.group_dim)
            pos_bias = pos_bias.view(B, N, K, self.num_heads, self.num_groups, self.group_dim)
        else:
            pos_bias = pos_enc.view(B, N, K, self.num_heads, self.num_groups, self.group_dim)
            pos_mul = None

        # reshape Q: [B, N, H, G, Dg]
        q = q.view(B, N, self.num_heads, self.num_groups, self.group_dim)
        # reshape K, V: [B, N, K, H, G, Dg]
        k = k.view(B, N, K, self.num_heads, self.num_groups, self.group_dim)
        v = v.view(B, N, K, self.num_heads, self.num_groups, self.group_dim)

        # ---- GVA attention ----
        # q: [B, N, 1, H, G, Dg], k: [B, N, K, H, G, Dg]
        if pos_mul is not None:
            # PE multiplier: pos_mul modulates the Q-K similarity
            attn_logits = ((pos_mul * q.unsqueeze(2) - k + pos_bias) * self.scale).sum(dim=-1)
        else:
            attn_logits = ((q.unsqueeze(2) - k + pos_bias) * self.scale).sum(dim=-1)

        # attn_logits: [B, N, K, H, G]
        attn = F.softmax(attn_logits, dim=2)
        attn = self.attn_dropout(attn)  # [B, N, K, H, G]

        # Weighted sum within each group, then over neighbours
        # v: [B, N, K, H, G, Dg], attn: [B, N, K, H, G, 1]
        if pos_mul is not None:
            weighted = (v + pos_bias) * attn.unsqueeze(-1)
        else:
            weighted = v * attn.unsqueeze(-1)

        out = weighted.sum(dim=2)  # sum over neighbours → [B, N, H, G, Dg]
        out = out.reshape(B, N, C)

        features = features + self.out_proj(out)
        features = features + self.ffn(self.norm2(features))
        return features
