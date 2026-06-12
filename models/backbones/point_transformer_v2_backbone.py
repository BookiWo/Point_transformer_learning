"""
Point Transformer V2 Backbone — GVA blocks + Grid Pooling.

Differences from V1 backbone:
  1. PointTransformerV2Block with GVA + PE Multiplier (per-block).
  2. GridPoolingDown (grid partition) replaces TransitionDown (FPS + kNN).
  3. GridPoolingUp remains kNN-interpolation (up-sampling is pooling-agnostic).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.blocks.point_transformer_v2_block import PointTransformerV2Block
from models.blocks.grid_pooling import GridPoolingDown, GridPoolingUp


class PointTransformerV2Backbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_groups: int = 2,
        dropout: float = 0.1,
        pe_multiplier: bool = True,
        grid_cell_size: float = 0.04,
    ) -> None:
        super().__init__()
        stage_depth = max(1, num_layers // 4)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        block_kwargs = dict(
            num_heads=num_heads, num_groups=num_groups,
            dropout=dropout, pe_multiplier=pe_multiplier,
        )

        # --- Encoder ---
        self.stem_blocks = nn.ModuleList([
            PointTransformerV2Block(hidden_dim, **block_kwargs)
            for _ in range(stage_depth)
        ])
        self.down1 = GridPoolingDown(hidden_dim, hidden_dim * 2, cell_size=grid_cell_size)
        self.stage2_blocks = nn.ModuleList([
            PointTransformerV2Block(hidden_dim * 2, **block_kwargs)
            for _ in range(stage_depth)
        ])
        self.down2 = GridPoolingDown(hidden_dim * 2, hidden_dim * 4, cell_size=grid_cell_size * 2)
        self.stage3_blocks = nn.ModuleList([
            PointTransformerV2Block(hidden_dim * 4, **block_kwargs)
            for _ in range(stage_depth)
        ])
        self.down3 = GridPoolingDown(hidden_dim * 4, hidden_dim * 8, cell_size=grid_cell_size * 4)
        self.bottleneck_blocks = nn.ModuleList([
            PointTransformerV2Block(hidden_dim * 8, **block_kwargs)
            for _ in range(stage_depth)
        ])

        # --- Decoder ---
        self.up3 = GridPoolingUp(hidden_dim * 8, hidden_dim * 4, hidden_dim * 4)
        self.decoder3_blocks = nn.ModuleList([
            PointTransformerV2Block(hidden_dim * 4, **block_kwargs)
            for _ in range(stage_depth)
        ])
        self.up2 = GridPoolingUp(hidden_dim * 4, hidden_dim * 2, hidden_dim * 2)
        self.decoder2_blocks = nn.ModuleList([
            PointTransformerV2Block(hidden_dim * 2, **block_kwargs)
            for _ in range(stage_depth)
        ])
        self.up1 = GridPoolingUp(hidden_dim * 2, hidden_dim, hidden_dim)
        self.decoder1_blocks = nn.ModuleList([
            PointTransformerV2Block(hidden_dim, **block_kwargs)
            for _ in range(stage_depth)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.input_proj(points)

        # Stem
        for block in self.stem_blocks:
            features = block(features, points)
        skip1_points, skip1_features = points, features

        # Stage 2
        points2, features2 = self.down1(points, features)
        for block in self.stage2_blocks:
            features2 = block(features2, points2)
        skip2_points, skip2_features = points2, features2

        # Stage 3
        points3, features3 = self.down2(points2, features2)
        for block in self.stage3_blocks:
            features3 = block(features3, points3)
        skip3_points, skip3_features = points3, features3

        # Bottleneck
        points4, features4 = self.down3(points3, features3)
        for block in self.bottleneck_blocks:
            features4 = block(features4, points4)

        # Decoder
        points3_up, features3_up = self.up3(points4, features4, skip3_points, skip3_features)
        for block in self.decoder3_blocks:
            features3_up = block(features3_up, points3_up)

        points2_up, features2_up = self.up2(points3_up, features3_up, skip2_points, skip2_features)
        for block in self.decoder2_blocks:
            features2_up = block(features2_up, points2_up)

        points1_up, features1_up = self.up1(points2_up, features2_up, skip1_points, skip1_features)
        for block in self.decoder1_blocks:
            features1_up = block(features1_up, points1_up)

        return self.output_norm(features1_up)
