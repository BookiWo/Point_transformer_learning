"""
PTX segmentation wrapper — drop-in compatible with V1/V2/V3 training pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.ptx_model import PointTransformerX
from models.heads.segmentation_head import SegmentationHead


class PointTransformerXSeg(nn.Module):
    def __init__(self, in_channels=6, num_classes=50, grid_size=0.05,
                 **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.backbone = PointTransformerX(in_channels=in_channels, **kwargs)
        dec_channels = kwargs.get("dec_channels", (64, 64, 128, 256))
        self.head = SegmentationHead(dec_channels[0], num_classes)

    def forward(self, coord, feat, cls_token=None):
        B, N = coord.shape[:2]
        coord_flat = coord.reshape(-1, 3)
        feat_flat = feat.reshape(-1, feat.shape[-1])
        batch = torch.arange(B, device=coord.device).repeat_interleave(N)
        offset = torch.arange(1, B + 1, device=coord.device) * N

        data = {
            "coord": coord_flat,
            "feat": feat_flat,
            "batch": batch,
            "offset": offset,
            "grid_size": torch.tensor(self.grid_size, device=coord.device),
        }
        features = self.backbone(data)
        features = features.reshape(B, N, -1)
        return self.head(features)
