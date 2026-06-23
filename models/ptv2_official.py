"""
Official Point Transformer V2 implementation, ported from:
https://github.com/Gofinge/PointTransformerV2

Key components matched exactly:
  - GroupedVectorAttention: MLP weight encoding (NOT dot-product)
  - Block: pre-activation with PointBatchNorm, DropPath, single-Linear FFN
  - GridPool: voxel_grid + segment_csr (torch_geometric + torch_scatter)
  - UnpoolWithSkip: "map" backend unpooling
  - PointTransformerV2: 4-stage encoder-decoder

Our replacements for pointops C++ extension:
  - pointops.knn_query → _flat_knn (PyTorch-native)
  - pointops.grouping → _flat_grouping (PyTorch-native)
  - pointops.interpolation → _flat_interp (PyTorch-native)

Usage:
  from models.ptv2_official import build_ptv2_seg
  model = build_ptv2_seg(in_channels=6, num_classes=50, num_shape_classes=16)
"""

from __future__ import annotations

import math
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch_scatter import segment_csr

def _voxel_grid(pos: torch.Tensor, size: float, batch: torch.Tensor, start: int = 0):
    """PyTorch-native voxel grid clustering (replaces torch_geometric.nn.pool.voxel_grid).

    Args:
        pos: (N, 3) point coordinates
        size: grid cell size
        batch: (N,) batch indices
        start: offset for batch indices

    Returns:
        cluster: (N,) per-point cluster/voxel ID
    """
    # Compute voxel indices per dimension
    voxel_idx = (pos / size).floor().long()
    # Unique hash: combine x, y, z, batch into a single integer
    max_per_dim = voxel_idx.max(dim=0).values + 1
    # Simple hash: batch * max_x * max_y * max_z + z * max_x * max_y + y * max_x + x
    # But for simplicity, use Python's hash via string key approach
    # More efficient: use torch.unique with stacked indices
    stacked = torch.cat([
        (batch - start).unsqueeze(1).long(),
        voxel_idx
    ], dim=1)  # (N, 4): [batch, x, y, z]
    # Use hashing via multiplication
    B = stacked[:, 0].max().item() + 1
    hash_val = stacked[:, 0] * (max_per_dim[0].item() * max_per_dim[1].item() * max_per_dim[2].item()) \
             + stacked[:, 3] * (max_per_dim[0].item() * max_per_dim[1].item()) \
             + stacked[:, 2] * max_per_dim[0].item() \
             + stacked[:, 1]
    _, cluster = torch.unique(hash_val, return_inverse=True)
    return cluster


# =============================================================================
# PyTorch-native replacements for pointops
# =============================================================================

def _flat_knn(neighbours: int, coord: torch.Tensor, offset: torch.Tensor):
    """PyTorch KNN on flat tensors with offset-based batching.

    Args:
        neighbours: number of neighbours (k)
        coord: (total_points, 3)
        offset: (B,) cumulative point counts

    Returns:
        reference_index: (total_points, neighbours) — flat indices of neighbours
        reference_xyz: unused, returned as None for compat
    """
    total = coord.shape[0]
    reference_index = torch.zeros(total, neighbours, dtype=torch.long, device=coord.device)

    for b in range(len(offset)):
        start = 0 if b == 0 else offset[b - 1].item()
        end = offset[b].item()
        if end - start <= neighbours:
            # fewer points than k → repeat last
            idx = torch.arange(start, end, device=coord.device)
            reference_index[start:end, :] = idx.unsqueeze(0).expand(end - start, neighbours).contiguous()
            continue

        pts = coord[start:end]  # (n_b, 3)
        dist = torch.cdist(pts, pts)  # (n_b, n_b)
        _, knn_idx = dist.topk(k=neighbours, dim=-1, largest=False)
        reference_index[start:end] = knn_idx + start  # global indexing

    return reference_index, None


def _flat_grouping(reference_index: torch.Tensor, features: torch.Tensor,
                   coords: torch.Tensor, with_xyz: bool = True):
    """PyTorch grouping: gather neighbour features by reference_index.

    Args:
        reference_index: (total_points, neighbours)
        features: (total_points, C)
        coords: (total_points, 3)
        with_xyz: if True, concat relative position

    Returns:
        grouped: (total_points, neighbours, out_C)
    """
    total, K = reference_index.shape
    # Fancy-index gather: features[reference_index] → (total, K, C)
    neighbour_feat = features[reference_index]  # (total, K, C)

    if with_xyz:
        neighbour_coord = coords[reference_index]  # (total, K, 3)
        rel_coord = neighbour_coord - coords.unsqueeze(1)  # (total, K, 3)
        return torch.cat([rel_coord, neighbour_feat], dim=-1)

    return neighbour_feat


def _flat_interp(source_coord: torch.Tensor, target_coord: torch.Tensor,
                 source_feat: torch.Tensor, source_offset: torch.Tensor,
                 target_offset: torch.Tensor, k: int = 3):
    """PyTorch interpolation: KNN-weighted average from source to target points.

    Args:
        source_coord: (M, 3)
        target_coord: (N, 3)
        source_feat: (M, C)
        source_offset: (B,)
        target_offset: (B,)

    Returns:
        interp_feat: (N, C)
    """
    total_target = target_coord.shape[0]
    C = source_feat.shape[1]
    interp_feat = torch.zeros(total_target, C, device=source_feat.device, dtype=source_feat.dtype)

    for b in range(len(source_offset)):
        s_start = 0 if b == 0 else source_offset[b - 1].item()
        s_end = source_offset[b].item()
        t_start = 0 if b == 0 else target_offset[b - 1].item()
        t_end = target_offset[b].item()

        sc = source_coord[s_start:s_end]
        tc = target_coord[t_start:t_end]
        sf = source_feat[s_start:s_end]
        n_source = sc.shape[0]

        k_actual = min(k, n_source)
        dist = torch.cdist(tc, sc)  # (N_t, N_s)
        knn_dist, knn_idx = dist.topk(k=k_actual, dim=-1, largest=False)
        weights = 1.0 / (knn_dist + 1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        sf_gathered = sf[knn_idx]  # (N_t, k, C)
        interp_feat[t_start:t_end] = (sf_gathered * weights.unsqueeze(-1)).sum(dim=-2)

    return interp_feat


# =============================================================================
# PointBatchNorm — official normalization for point cloud data
# =============================================================================

class PointBatchNorm(nn.Module):
    """BatchNorm1d that handles both (B*N, C) and (B*N, L, C) tensors."""

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


# =============================================================================
# DropPath (from timm, simplified)
# =============================================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# =============================================================================
# GroupedVectorAttention — official: MLP weight encoding, NOT dot-product
# =============================================================================

class GroupedVectorAttention(nn.Module):
    """Official PTv2 GVA with MLP-based weight encoding.

    Key: attention weights are produced by a learned `weight_encoding(relation_qk)`
    MLP, NOT dot-product between Q and K."""

    def __init__(self,
                 embed_channels,
                 groups,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True):
        super().__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups)
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        query, key, value = self.linear_q(feat), self.linear_k(feat), self.linear_v(feat)
        key = _flat_grouping(reference_index, key, coord, with_xyz=True)
        value = _flat_grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]  # (N, K, 3), (N, K, C)

        relation_qk = key - query.unsqueeze(1)

        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)  # (N, K, groups)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)

        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat


# =============================================================================
# Block — official: pre-activation, PointBatchNorm, DropPath, single-Linear FFN
# =============================================================================

class Block(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False):
        super().__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = self.attn(feat, coord, reference_index)
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]


class BlockSequence(nn.Module):
    def __init__(self, depth, embed_channels, groups, neighbours=16,
                 qkv_bias=True, pe_multiplier=False, pe_bias=True,
                 attn_drop_rate=0., drop_path_rate=0., enable_checkpoint=False):
        super().__init__()
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels, groups=groups,
                qkv_bias=qkv_bias, pe_multiplier=pe_multiplier, pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint)
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset = points
        reference_index, _ = _flat_knn(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points


# =============================================================================
# GridPool + UnpoolWithSkip — official implementation
# =============================================================================

def offset2batch(offset):
    return torch.arange(len(offset), device=offset.device, dtype=torch.long).repeat_interleave(
        torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)))


def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class GridPool(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super().__init__()
        self.grid_size = grid_size
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = segment_csr(coord, torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                            reduce="min") if start is None else start
        cluster = _voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster


class UnpoolWithSkip(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bias=True, skip=True, backend="map"):
        super().__init__()
        self.skip = skip
        self.backend = backend
        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels, bias=bias),
                                  PointBatchNorm(out_channels), nn.ReLU(inplace=True))
        self.proj_skip = nn.Sequential(nn.Linear(skip_channels, out_channels, bias=bias),
                                       PointBatchNorm(out_channels), nn.ReLU(inplace=True))

    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            feat = _flat_interp(coord, skip_coord, self.proj(feat), offset, skip_offset)
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset]


# =============================================================================
# GVAPatchEmbed — initial convolution-like embedding
# =============================================================================

class GVAPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_channels, groups, depth=1, neighbours=8,
                 qkv_bias=True, pe_multiplier=False, pe_bias=True,
                 attn_drop_rate=0., drop_path_rate=0., enable_checkpoint=False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True))
        self.blocks = BlockSequence(
            depth=depth, embed_channels=embed_channels, groups=groups,
            neighbours=neighbours, qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier, pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint)

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])


# =============================================================================
# Encoder + Decoder stages
# =============================================================================

class Encoder(nn.Module):
    def __init__(self, depth, in_channels, embed_channels, groups, grid_size=None,
                 neighbours=16, qkv_bias=True, pe_multiplier=False, pe_bias=True,
                 attn_drop_rate=0., drop_path_rate=0., enable_checkpoint=False):
        super().__init__()
        self.down = GridPool(in_channels=in_channels, out_channels=embed_channels, grid_size=grid_size)
        self.blocks = BlockSequence(
            depth=depth, embed_channels=embed_channels, groups=groups,
            neighbours=neighbours, qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier, pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint)

    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, embed_channels, groups, depth,
                 neighbours=16, qkv_bias=True, pe_multiplier=False, pe_bias=True,
                 attn_drop_rate=0., drop_path_rate=0., enable_checkpoint=False,
                 unpool_backend="map"):
        super().__init__()
        self.up = UnpoolWithSkip(in_channels=in_channels, out_channels=embed_channels,
                                 skip_channels=skip_channels, backend=unpool_backend)
        self.blocks = BlockSequence(
            depth=depth, embed_channels=embed_channels, groups=groups,
            neighbours=neighbours, qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier, pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint)

    def forward(self, points, skip_points, cluster=None):
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


# =============================================================================
# PointTransformerV2 — full model (official architecture)
# =============================================================================

class PointTransformerV2(nn.Module):
    def __init__(self,
                 in_channels=6,
                 num_classes=50,
                 patch_embed_depth=1,
                 patch_embed_channels=48,
                 patch_embed_groups=6,
                 patch_embed_neighbours=8,
                 enc_depths=(2, 2, 6, 2),
                 enc_channels=(96, 192, 384, 512),
                 enc_groups=(12, 24, 48, 64),
                 enc_neighbours=(16, 16, 16, 16),
                 dec_depths=(1, 1, 1, 1),
                 dec_channels=(48, 96, 192, 384),
                 dec_groups=(6, 12, 24, 48),
                 dec_neighbours=(16, 16, 16, 16),
                 grid_sizes=(0.06, 0.12, 0.24, 0.48),
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0,
                 enable_checkpoint=False,
                 unpool_backend="map"):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)

        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels, embed_channels=patch_embed_channels,
            groups=patch_embed_groups, depth=patch_embed_depth,
            neighbours=patch_embed_neighbours, qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier, pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate, enable_checkpoint=enable_checkpoint)

        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        dec_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]

        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i], in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1], groups=enc_groups[i],
                grid_size=grid_sizes[i], neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias, pe_multiplier=pe_multiplier, pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint)
            self.enc_stages.append(enc)
        # Create decoders in REVERSE order to match reversed skips in forward
        for i in range(self.num_stages):
            dec = Decoder(
                in_channels=dec_channels[self.num_stages - i],  # reversed: 512→384→192→96
                skip_channels=enc_channels[self.num_stages - 1 - i],  # reversed: 384→192→96→48
                embed_channels=dec_channels[self.num_stages - 1 - i],  # reversed: 384→192→96→48
                groups=dec_groups[self.num_stages - 1 - i], depth=dec_depths[self.num_stages - 1 - i],
                neighbours=dec_neighbours[self.num_stages - 1 - i], qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier, pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[sum(dec_depths[:self.num_stages - 1 - i]):sum(dec_depths[:self.num_stages - i])],
                enable_checkpoint=enable_checkpoint, unpool_backend=unpool_backend)
            self.dec_stages.append(dec)

        self.classifier = nn.Sequential(
            nn.Linear(dec_channels[0], dec_channels[0]),
            nn.BatchNorm1d(dec_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(dec_channels[0], num_classes),
        ) if num_classes > 0 else nn.Identity()

    def forward(self, points):
        """points: [coord, feat, offset] — flat tensors with offsets"""
        points = self.patch_embed(points)
        skips = []
        clusters_list = []
        for i, enc in enumerate(self.enc_stages):
            skips.append(points)  # save before downsampling
            points, cluster = enc(points)
            clusters_list.append(cluster)
        # Reverse for decoder (LIFO)
        skips = skips[::-1]
        clusters_list = clusters_list[::-1]
        for i, dec in enumerate(self.dec_stages):
            points = dec(points, skips[i], clusters_list[i])
        coord, feat, offset = points
        return self.classifier(feat)


# =============================================================================
# Wrapper for (B,N,C) format → flat tensors → (B,N,num_classes)
# =============================================================================

class PointTransformerV2Seg(nn.Module):
    """Official PTv2 segmentation model for ShapeNet Part / PartNet."""

    def __init__(self, in_channels=6, num_classes=50, num_shape_classes=0,
                 patch_embed_channels=48, patch_embed_groups=6,
                 enc_depths=(2, 2, 6, 2), enc_channels=(96, 192, 384, 512),
                 enc_groups=(12, 24, 48, 64), enc_neighbours=(16, 16, 16, 16),
                 dec_depths=(1, 1, 1, 1), dec_channels=(48, 96, 192, 384),
                 dec_groups=(6, 12, 24, 48), dec_neighbours=(16, 16, 16, 16),
                 grid_sizes=(0.06, 0.12, 0.24, 0.48),
                 attn_qkv_bias=True, pe_multiplier=False, pe_bias=True,
                 attn_drop_rate=0., drop_path_rate=0., enable_checkpoint=False,
                 **kwargs):
        super().__init__()
        self.backbone = PointTransformerV2(
            in_channels=in_channels, num_classes=num_classes,
            patch_embed_channels=patch_embed_channels,
            patch_embed_groups=patch_embed_groups,
            enc_depths=enc_depths, enc_channels=enc_channels,
            enc_groups=enc_groups, enc_neighbours=enc_neighbours,
            dec_depths=dec_depths, dec_channels=dec_channels,
            dec_groups=dec_groups, dec_neighbours=dec_neighbours,
            grid_sizes=grid_sizes, attn_qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier, pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint)

    def forward(self, coord, feat):
        """coord: (B, N, 3), feat: (B, N, C) → logits: (B, N, num_classes)"""
        B, N = coord.shape[:2]
        # Flatten to (B*N, ...)
        coord_flat = coord.reshape(-1, 3)
        feat_flat = feat.reshape(-1, feat.shape[-1])
        offset = torch.arange(1, B + 1, device=coord.device, dtype=torch.long) * N

        result = self.backbone([coord_flat, feat_flat, offset])  # (B*N, num_classes)
        return result.reshape(B, N, -1)
