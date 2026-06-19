"""
Point Transformer V3 for Part Segmentation.

Usage:
    from models.point_transformer_v3_seg import PointTransformerV3Seg
    model = PointTransformerV3Seg(in_channels=3, num_parts=50, enc_channels=(32, 64, 128, 256, 512))
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.ptv3_model import PointTransformerV3
from models.heads.segmentation_head import SegmentationHead


class PointTransformerV3Seg(nn.Module):
    """PTv3 segmentation model for PartNet.

    Config keys (matching config YAML):
        in_channels: input dim (3 for xyz, 6 for xyz+rgb)
        enc_channels: encoder channels per stage (e.g. [32, 64, 128, 256, 512])
        enc_depths: blocks per encoder stage (e.g. [2, 2, 2, 6, 2])
        enc_num_head: heads per encoder stage (e.g. [2, 4, 8, 16, 32])
        enc_patch_size: patch size per stage
        dec_channels: decoder channels
        dec_depths: decoder depths
        dec_num_head: decoder heads
        dec_patch_size: decoder patch size
        stride: pooling stride per stage (e.g. [2, 2, 2, 2])
        mlp_ratio: MLP expansion ratio
        drop_path: max stochastic depth rate
        enable_rpe: relative position encoding
        grid_size: voxel size for serialization (e.g. 0.05)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_parts: int = 50,
        # Encoder
        enc_channels=(32, 64, 128, 256, 512),
        enc_depths=(2, 2, 2, 6, 2),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # Decoder
        dec_channels=(64, 64, 128, 256),
        dec_depths=(2, 2, 2, 2),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        # Shared
        stride=(2, 2, 2, 2),
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.1,
        drop_path=0.1,
        enable_rpe=False,
        grid_size=0.05,
        order=("z", "z-trans"),
        shuffle_orders=True,
        **kwargs,  # absorb unused config keys
    ) -> None:
        super().__init__()

        self.grid_size = grid_size
        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            enable_rpe=enable_rpe,
            enable_flash=False,
            upcast_attention=False,
            upcast_softmax=False,
            shuffle_orders=shuffle_orders,
        )
        self.head = SegmentationHead(dec_channels[0], num_parts, dropout=proj_drop)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            points: (B, N, 3) point coordinates
        Returns:
            logits: (B, N, num_parts)
        """
        B, N, C = points.shape
        device = points.device

        # Flatten to (B*N, C) for PTv3 Point format
        coord = points.reshape(-1, C)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        offset = torch.arange(1, B + 1, device=device) * N

        data_dict = {
            "coord": coord,
            "feat": coord,  # use xyz as initial features
            "batch": batch,
            "offset": offset,
            "grid_size": torch.tensor(self.grid_size, device=device),
        }

        features = self.backbone(data_dict)  # returns Point; .feat is (B*N, dec_channels[0])
        features = features.feat.reshape(B, N, -1)
        return self.head(features)
