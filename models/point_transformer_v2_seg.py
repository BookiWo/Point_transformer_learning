"""
Point Transformer V2 for Part Segmentation — drop-in replacement for V1.

Usage:
    from models.point_transformer_v2_seg import PointTransformerV2Seg
    model = PointTransformerV2Seg(input_dim=3, hidden_dim=128, num_layers=4,
                                   num_heads=4, num_parts=50, num_groups=2)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.backbones.point_transformer_v2_backbone import PointTransformerV2Backbone
from models.heads.segmentation_head import SegmentationHead


class PointTransformerV2Seg(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_parts: int,
        num_groups: int = 2,
        dropout: float = 0.1,
        pe_multiplier: bool = True,
        grid_cell_size: float = 0.04,
    ) -> None:
        super().__init__()
        self.backbone = PointTransformerV2Backbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_groups=num_groups,
            dropout=dropout,
            pe_multiplier=pe_multiplier,
            grid_cell_size=grid_cell_size,
        )
        self.head = SegmentationHead(hidden_dim, num_parts, dropout=dropout)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.backbone(points)
        return self.head(features)
