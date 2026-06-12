"""
Partition-based Pooling (Grid Pooling) — PTv2's efficient replacement for FPS+kNN.

Instead of Farthest Point Sampling + kNN query (V1), we quantise coordinates onto a
uniform grid and pool all points within each occupied cell via scatter operations.
This yields non-overlapping partitions with better spatial alignment and O(N) memory.

Reference: Point Transformer V2, NeurIPS 2022, Section 3.3
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _batch_gather(data: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Memory-efficient gather for 2-D and 3-D index tensors.

    Uses fancy indexing instead of expand+gather to avoid huge intermediate
    tensors (e.g. [B, P, N, C]) during the backward pass.
    """
    B = data.shape[0]
    device = data.device

    if index.dim() == 2:
        batch_idx = torch.arange(B, device=device).view(B, 1)
        return data[batch_idx, index]

    if index.dim() == 3:
        # index: [B, Nq, K] — gather from data: [B, N, C] → [B, Nq, K, C]
        batch_idx = (
            torch.arange(B, device=device)
            .view(B, 1, 1)
            .expand(-1, index.shape[1], index.shape[2])
        )
        return data[batch_idx, index]

    raise ValueError(f"Unsupported index rank: {index.dim()}")


def _batched_knn(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: int,
    exclude_self: bool = False,
) -> torch.Tensor:
    """kNN with optional self-exclusion."""
    with torch.no_grad():
        if reference_points.shape[1] == 0:
            raise ValueError("reference_points must contain at least one point")

        distances = torch.cdist(query_points, reference_points)
        if (
            exclude_self
            and query_points.shape == reference_points.shape
            and query_points.data_ptr() == reference_points.data_ptr()
        ):
            eye = torch.eye(
                reference_points.shape[1], device=reference_points.device, dtype=torch.bool
            ).unsqueeze(0)
            distances = distances.masked_fill(eye, float("inf"))

        k = min(k, reference_points.shape[1])
        return distances.topk(k=k, dim=-1, largest=False).indices


# ---------------------------------------------------------------------------
# Grid sub-sampling — scatter-based, O(N) per batch
# ---------------------------------------------------------------------------


def _grid_subsample(
    points: torch.Tensor, cell_size: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform grid sub-sampling — vectorised scatter-based implementation.

    Args:
        points: [B, N, 3] point coordinates.
        cell_size: grid cell edge length in world units.

    Returns:
        pooled_points: [B, N_out, 3] — centroid of each occupied cell.
        pooled_idx:   [B, N_out] — indices mapping pooled → original points
                       (first point in each cell).
        inv_idx:      [B, N] — reverse map: original point → cell index.
    """
    with torch.no_grad():
        B, N, _ = points.shape
        device = points.device

        grid_coord = (points / cell_size).floor().long()
        offset = 100_000
        grid_coord_shifted = grid_coord + offset
        max_val = grid_coord_shifted.max().item() + 1
        hash_ = (
            grid_coord_shifted[..., 0] * (max_val * max_val)
            + grid_coord_shifted[..., 1] * max_val
            + grid_coord_shifted[..., 2]
        )

        pooled_points_list: list[torch.Tensor] = []
        pooled_idx_list: list[torch.Tensor] = []
        inv_idx_list: list[torch.Tensor] = []

        for b in range(B):
            h = hash_[b]
            unique_hash, inverse = h.unique(return_inverse=True)
            num_cells = unique_hash.shape[0]

            # Vectorised centroid
            inv_exp = inverse.unsqueeze(-1).expand(-1, 3)
            centroids = points.new_zeros(num_cells, 3)
            centroids = centroids.scatter_reduce(0, inv_exp, points[b], reduce="sum")
            counts = points.new_zeros(num_cells)
            counts = counts.scatter_reduce(0, inverse, points.new_ones(N), reduce="sum")
            centroids = centroids / counts.unsqueeze(-1).clamp(min=1)

            # Vectorised representative
            point_idx = torch.arange(N, device=device)
            INT_MAX = torch.iinfo(torch.long).max
            repr_idx = torch.full((num_cells,), INT_MAX, dtype=torch.long, device=device)
            repr_idx = repr_idx.scatter_reduce(
                0, inverse, point_idx, reduce="amin", include_self=False
            )

            pooled_points_list.append(centroids)
            pooled_idx_list.append(repr_idx)
            inv_idx_list.append(inverse)

        max_cells = max(p.shape[0] for p in pooled_points_list)
        pooled_batch = points.new_zeros(B, max_cells, 3)
        pooled_idx = torch.zeros(B, max_cells, dtype=torch.long, device=device)
        inv_idx = torch.full((B, N), -1, dtype=torch.long, device=device)

        for b in range(B):
            n = pooled_points_list[b].shape[0]
            pooled_batch[b, :n] = pooled_points_list[b]
            pooled_idx[b, :n] = pooled_idx_list[b]
            inv_idx[b] = inv_idx_list[b]

        return pooled_batch, pooled_idx, inv_idx


# ---------------------------------------------------------------------------
# Grid Pooling modules (PTv2-style — no kNN, scatter-based pooling)
# ---------------------------------------------------------------------------


class GridPoolingDown(nn.Module):
    """Grid-based transition down — scatter max-pool within each cell.

    Unlike V1's TransitionDown (FPS + kNN → huge intermediate tensors),
    this directly pools features per grid cell, keeping memory O(N).
    """

    def __init__(
        self, in_channels: int, out_channels: int, cell_size: float = 0.05
    ) -> None:
        super().__init__()
        self.cell_size = cell_size
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(
        self, points: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Grid-sub-sample points and max-pool features per cell."""
        B, N, C = features.shape
        pooled_points, pooled_idx, inv_idx = _grid_subsample(points, self.cell_size)
        P = pooled_points.shape[1]  # number of occupied cells

        # Scatter max-pool features within each grid cell
        pooled_feat_list: list[torch.Tensor] = []
        for b in range(B):
            inv = inv_idx[b]  # [N] → cell index 0..P-1
            valid = inv >= 0
            inv_valid = inv[valid]  # [N_valid]
            feat_valid = features[b][valid]  # [N_valid, C]

            idx_exp = inv_valid.unsqueeze(-1).expand(-1, C)
            cell_feat = features.new_zeros(P, C)
            cell_feat = cell_feat.scatter_reduce(0, idx_exp, feat_valid, reduce="amax")
            pooled_feat_list.append(cell_feat)

        pooled_features = torch.stack(pooled_feat_list, dim=0)  # [B, P, C]
        fused = self.norm(self.proj(pooled_features))
        return pooled_points, fused


class GridPoolingUp(nn.Module):
    """Grid-compatible transition up via kNN interpolation.

    Uses the memory-efficient _batch_gather to avoid large backward intermediates.
    """

    def __init__(
        self, coarse_channels: int, skip_channels: int, out_channels: int, k: int = 3
    ) -> None:
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
        return skip_points, self.fuse(fused)
