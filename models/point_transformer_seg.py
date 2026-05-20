from __future__ import annotations

import torch
import torch.nn as nn

from models.backbones.point_transformer_backbone import PointTransformerBackbone
from models.heads.segmentation_head import SegmentationHead


class PointTransformerSeg(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_parts: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = PointTransformerBackbone(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.head = SegmentationHead(hidden_dim, num_parts, dropout=dropout)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.backbone(points)
        logits = self.head(features)
        return logits
