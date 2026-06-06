from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.blocks.simple_point_transformer_block import PointTransformerBlock


def _batch_gather(data: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.dim() == 2:
        return data.gather(1, index.unsqueeze(-1).expand(-1, -1, data.shape[-1]))
    if index.dim() == 3:
        expanded = data.unsqueeze(1).expand(-1, index.shape[1], -1, -1)
        return torch.gather(expanded, 2, index.unsqueeze(-1).expand(-1, -1, -1, data.shape[-1]))
    raise ValueError(f"Unsupported index rank: {index.dim()}")


def _batched_knn(query_points: torch.Tensor, reference_points: torch.Tensor, k: int) -> torch.Tensor:
    with torch.no_grad():
        if reference_points.shape[1] == 0:
            raise ValueError("reference_points must contain at least one point")
        distances = torch.cdist(query_points, reference_points)
        k = min(k, reference_points.shape[1])
        return distances.topk(k=k, dim=-1, largest=False).indices


def _farthest_point_sample(points: torch.Tensor, ratio: float) -> torch.Tensor:
    with torch.no_grad():
        batch_size, num_points, _ = points.shape
        num_sample = max(1, int(math.ceil(num_points * ratio)))
        centroids = torch.zeros(batch_size, num_sample, dtype=torch.long, device=points.device)
        distances = torch.full((batch_size, num_points), float("inf"), device=points.device)
        # Deterministic start: point furthest from centroid (fallback to 0 on tiny clouds)
        centroid = points.mean(dim=1, keepdim=True)
        dist_to_centroid = ((points - centroid) ** 2).sum(dim=-1)
        farthest = dist_to_centroid.max(dim=-1).indices
        batch_idx = torch.arange(batch_size, device=points.device)

        for i in range(num_sample):
            centroids[:, i] = farthest
            centroid = points[batch_idx, farthest].unsqueeze(1)
            dist = ((points - centroid) ** 2).sum(dim=-1)
            distances = torch.minimum(distances, dist)
            farthest = distances.max(dim=-1).indices

        return centroids


class TransitionDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ratio: float = 0.5, k: int = 16) -> None:
        super().__init__()
        self.ratio = ratio
        self.k = k
        self.proj = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sample_idx = _farthest_point_sample(points, self.ratio)
        sampled_points = _batch_gather(points, sample_idx)
        grouped_idx = _batched_knn(sampled_points, points, self.k)

        grouped_points = _batch_gather(points, grouped_idx)
        grouped_features = _batch_gather(features, grouped_idx)
        relative_points = grouped_points - sampled_points.unsqueeze(2)
        fused = torch.cat([relative_points, grouped_features], dim=-1)
        fused = self.proj(fused).max(dim=2).values
        fused = self.norm(fused)
        return sampled_points, fused


class TransitionUp(nn.Module):
    def __init__(self, coarse_channels: int, skip_channels: int, out_channels: int, k: int = 3) -> None:
        super().__init__()
        self.k = k
        self.coarse_proj = nn.Linear(coarse_channels, out_channels, bias=False)
        self.skip_proj = nn.Linear(skip_channels, out_channels, bias=False)
        self.fuse = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self,
        coarse_points: torch.Tensor,
        coarse_features: torch.Tensor,
        skip_points: torch.Tensor,
        skip_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            distances = torch.cdist(skip_points, coarse_points)
            k = min(self.k, coarse_points.shape[1])
            knn_dist, knn_idx = distances.topk(k=k, dim=-1, largest=False)
            weights = 1.0 / (knn_dist + 1e-8)
            weights = weights / weights.sum(dim=-1, keepdim=True)

        interpolated = _batch_gather(coarse_features, knn_idx)
        interpolated = (interpolated * weights.unsqueeze(-1)).sum(dim=2)
        fused = self.coarse_proj(interpolated) + self.skip_proj(skip_features)
        fused = self.fuse(fused)
        return skip_points, fused


class PointTransformerBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        stage_depth = max(1, num_layers // 4)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.stem_blocks = nn.ModuleList(
            [PointTransformerBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(stage_depth)]
        )
        self.down1 = TransitionDown(hidden_dim, hidden_dim * 2)
        self.stage2_blocks = nn.ModuleList(
            [PointTransformerBlock(hidden_dim * 2, num_heads, dropout=dropout) for _ in range(stage_depth)]
        )
        self.down2 = TransitionDown(hidden_dim * 2, hidden_dim * 4)
        self.stage3_blocks = nn.ModuleList(
            [PointTransformerBlock(hidden_dim * 4, num_heads, dropout=dropout) for _ in range(stage_depth)]
        )
        self.down3 = TransitionDown(hidden_dim * 4, hidden_dim * 8)
        self.bottleneck_blocks = nn.ModuleList(
            [PointTransformerBlock(hidden_dim * 8, num_heads, dropout=dropout) for _ in range(stage_depth)]
        )

        self.up3 = TransitionUp(hidden_dim * 8, hidden_dim * 4, hidden_dim * 4)
        self.decoder3_blocks = nn.ModuleList(
            [PointTransformerBlock(hidden_dim * 4, num_heads, dropout=dropout) for _ in range(stage_depth)]
        )
        self.up2 = TransitionUp(hidden_dim * 4, hidden_dim * 2, hidden_dim * 2)
        self.decoder2_blocks = nn.ModuleList(
            [PointTransformerBlock(hidden_dim * 2, num_heads, dropout=dropout) for _ in range(stage_depth)]
        )
        self.up1 = TransitionUp(hidden_dim * 2, hidden_dim, hidden_dim)
        self.decoder1_blocks = nn.ModuleList(
            [PointTransformerBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(stage_depth)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.input_proj(points)

        for block in self.stem_blocks:
            features = block(features, points)
        skip1_points, skip1_features = points, features

        points2, features2 = self.down1(points, features)
        for block in self.stage2_blocks:
            features2 = block(features2, points2)
        skip2_points, skip2_features = points2, features2

        points3, features3 = self.down2(points2, features2)
        for block in self.stage3_blocks:
            features3 = block(features3, points3)
        skip3_points, skip3_features = points3, features3

        points4, features4 = self.down3(points3, features3)
        for block in self.bottleneck_blocks:
            features4 = block(features4, points4)

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
