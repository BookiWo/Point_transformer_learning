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
        num_shape_classes: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = PointTransformerBackbone(
            input_dim, hidden_dim, num_layers, num_heads, dropout,
            num_shape_classes=num_shape_classes,
        )
        self.head = SegmentationHead(hidden_dim, num_parts, dropout=dropout)

    def forward(self, points: torch.Tensor, feat: torch.Tensor | None = None,
                cls_token: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        features = self.backbone(points, feat=feat, cls_token=cls_token)
        logits = self.head(features)
        return logits
