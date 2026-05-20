from __future__ import annotations

import torch
import torch.nn as nn

from models.blocks.simple_point_transformer_block import SimplePointTransformerBlock


class PointTransformerBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [SimplePointTransformerBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = self.stem(points)
        for block in self.blocks:
            x = block(x, points)
        return x
