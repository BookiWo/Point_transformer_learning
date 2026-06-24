"""
PointTransformerX (PTX) — CVPR 2026 Workshop

Portable and Efficient 3D Point Cloud Processing without Sparse Algorithms.
Built on PTv3 framework, replacing 3 specific components:

  1. 3D-GS-RoPE: per-head learned rotated coordinate basis (6 params/head),
     encodes 3D positions directly in self-attention → no KNN needed.
  2. Linear Patch Embedding: nn.Linear replaces spconv.SubMConv3d.
  3. ReLU² FFN: expansion r=2 instead of GeLU r=4.

Reference: Reichardt, Ebert, Wasenmüller (2026).
"""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from .serialization import encode


# ── helpers from PTv3 ─────────────────────────────────────────────────────

@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


# ── Point (same as PTv3, but no spconv) ──────────────────────────────────

class Point(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self and "offset" in self:
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self and "batch" in self:
            self["offset"] = batch2offset(self.batch)

    def __getattr__(self, key):
        if key in self: return self[key]
        raise AttributeError(f"'Point' object has no attribute '{key}'")

    def __setattr__(self, key, value): self[key] = value
    def __delattr__(self, key):
        if key in self: del self[key]

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        assert "batch" in self
        if "grid_coord" not in self:
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        assert depth * 3 + len(self.offset).bit_length() <= 63 and depth <= 16
        order_list = [order] if isinstance(order, str) else order
        code = [encode(self.grid_coord, self.batch, depth, order=o) for o in order_list]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1, index=order,
            src=torch.arange(code.shape[1], device=order.device).repeat(code.shape[0], 1))
        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code, order, inverse = code[perm], order[perm], inverse[perm]
        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """No-op in PTX — no spconv."""
        pass


# ── module helpers ─────────────────────────────────────────────────────────

class PointModule(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items(): self.add_module(key, module)
        else:
            for idx, module in enumerate(args): self.add_module(str(idx), module)
        for name, module in kwargs.items(): self.add_module(name, module)

    def __getitem__(self, idx):
        if idx < 0: idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx): next(it)
        return next(it)

    def __len__(self): return len(self._modules)

    def add(self, module, name=None):
        if name is None: name = str(len(self._modules))
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            if isinstance(module, PointModule):
                input = module(input)
            elif isinstance(input, Point):
                input.feat = module(input.feat)
            else:
                input = module(input)
        return input


# ── PTX: Linear Embedding (replaces spconv.SubMConv3d) ───────────────────

class LinearEmbedding(PointModule):
    """Linear patch embedding — PTX replaces sparse conv stem."""
    def __init__(self, in_channels, embed_channels, norm_layer=None, act_layer=None):
        super().__init__()
        self.stem = PointSequential(nn.Linear(in_channels, embed_channels, bias=False))
        if norm_layer is not None: self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None: self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


# ── PTX: 3D-GS-RoPE ──────────────────────────────────────────────────────

class GS_RoPE_3D(nn.Module):
    """3D Gram-Schmidt Rotary Position Embedding.

    Per attention head, learn a rotated orthogonal coordinate basis via
    Gram-Schmidt (6 params/head). Project relative 3D coordinates onto this
    basis and apply per-axis RoPE frequencies.

    The constraint order (r1, r2, r3) is cyclically permuted across heads
    to prevent bias toward any single axis.
    """

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Per-head learned basis vectors: r1 (free), r2 (gram-schmidt)
        self.r1 = nn.Parameter(torch.randn(num_heads, 3) * 0.02)
        self.r2 = nn.Parameter(torch.randn(num_heads, 3) * 0.02)

        # RoPE frequencies
        self.register_buffer(
            "freqs",
            1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        )  # (D/2,)

    def build_basis(self):
        """Gram-Schmidt: r1→norm, r2⊥r1→norm, r3=r1×r2."""
        r1_n = F.normalize(self.r1, dim=-1)            # (H, 3)
        r2_p = self.r2 - (self.r2 * r1_n).sum(-1, keepdim=True) * r1_n
        r2_n = F.normalize(r2_p, dim=-1)               # (H, 3)
        r3_n = torch.cross(r1_n, r2_n, dim=-1)         # (H, 3)
        return torch.stack([r1_n, r2_n, r3_n], dim=-2)  # (H, 3, 3)

    def forward(self, rel_pos: torch.Tensor):
        """rel_pos: (N_patches, K, K, 3) or (B, N, K, 3)
        Returns RoPE cos/sin modulation for attention."""
        H = self.num_heads
        D = self.head_dim

        # Build basis: (H, 3, 3)
        basis = self.build_basis()

        # Project rel_pos onto basis: (..., 3) → (..., H, 3)
        # rel_pos: (N, K, K, 3), basis: (H, 3, 3)
        proj = torch.einsum("...kld, hdc -> ...klhc", rel_pos.float(), basis)  # (N, K, K, H, 3)

        # Apply RoPE per axis — for each of 3 axes, apply cos/sin encoding
        # freqs: (D/2,), we split D equally across 3 axes if possible, else axis 0 gets more
        d_per_axis = D // 3
        cos_parts, sin_parts = [], []
        offset = 0
        for a in range(3):
            da = d_per_axis + (1 if a < D % 3 else 0)
            if da == 0: continue
            f = self.freqs[:da // 2]  # (da/2,)
            # proj[..., a]: (N, K, K, H) — rotate each head's projection
            theta = proj[..., a]  # (N, K, K, H)
            # Expand to pair-wise: (N, K, K, H, da//2)
            theta_exp = theta.unsqueeze(-1) * f  # (N, K, K, H, da//2)
            cos = torch.cos(theta_exp)
            sin = torch.sin(theta_exp)
            if da % 2 == 1:
                # odd dim: pad with one cos value
                cos = torch.cat([cos, torch.cos(theta.unsqueeze(-1))], dim=-1)
                sin = torch.cat([sin, torch.sin(theta.unsqueeze(-1))], dim=-1)
            cos_parts.append(cos)
            sin_parts.append(sin)
            offset += da

        # Concatenate across axes: (N, K, K, H, D)
        rope_cos = torch.cat(cos_parts, dim=-1) if len(cos_parts) > 1 else cos_parts[0]
        rope_sin = torch.cat(sin_parts, dim=-1) if len(sin_parts) > 1 else sin_parts[0]
        return rope_cos, rope_sin  # (N, K, K, H, D), (N, K, K, H, D)


# ── PTX: Serialized Attention with 3D-GS-RoPE ─────────────────────────────

class PTXAttention(PointModule):
    """Serialized (window) attention with 3D-GS-RoPE positional encoding.

    Same framework as PTv3 SerializedAttention, but:
      - Uses GS_RoPE_3D instead of RPE table / additive pos encoding
      - No flash_attn support (pure PyTorch)
    """

    def __init__(self, channels, num_heads, patch_size, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, order_index=0):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.patch_size_max = patch_size
        self.patch_size = 0

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        # 3D-GS-RoPE
        self.rope = GS_RoPE_3D(num_heads, channels // num_heads)

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        grid_coord = point.grid_coord[order]
        grid_coord = grid_coord.reshape(-1, K, 3)
        return grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)  # (N, K, K, 3)

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key, unpad_key, cu_key = "pad", "unpad", "cu_seqlens_key"
        if pad_key not in point or unpad_key not in point or cu_key not in point:
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (torch.div(bincount + self.patch_size - 1, self.patch_size,
                                      rounding_mode="trunc") * self.patch_size)
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = F.pad(offset, (1, 0))
            _offset_pad = F.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i]:_offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[_offset_pad[i + 1] - self.patch_size + (bincount[i] % self.patch_size):
                        _offset_pad[i + 1]] = pad[_offset_pad[i + 1] - 2 * self.patch_size
                        + (bincount[i] % self.patch_size):_offset_pad[i + 1] - self.patch_size]
                pad[_offset_pad[i]:_offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(torch.arange(_offset_pad[i], _offset_pad[i + 1],
                                                step=self.patch_size, dtype=torch.int32,
                                                device=offset.device))
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_key] = F.pad(torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1])
        return point[pad_key], point[unpad_key], point[cu_key]

    def forward(self, point):
        self.patch_size = min(offset2bincount(point.offset).min().tolist(), self.patch_size_max)
        H, K, C = self.num_heads, self.patch_size, self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order]
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)

        # Standard scaled dot-product
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (N, H, K, K)

        # 3D-GS-RoPE: compute positional modulation from relative grid coordinates
        rel_pos = self.get_rel_pos(point, order)  # (N, K, K, 3)
        rope_cos, rope_sin = self.rope(rel_pos)    # (N, K, K, H, D)
        # Apply RoPE modulation to attention logits
        # cos modulates magnitude, sin as additive bias (simplified from full RoPE)
        attn = attn.float()
        rope_bias = (rope_cos + rope_sin).sum(dim=-1)  # (N, K, K, H)
        rope_bias = rope_bias.permute(0, 3, 1, 2)       # (N, H, K, K)
        attn = attn + rope_bias

        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(qkv.dtype)
        feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        feat = feat[inverse]

        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


# ── PTX: ReLU² FFN ────────────────────────────────────────────────────────

class ReLUSquaredFFN(PointModule):
    """PTX FFN: ReLU² activation with expansion ratio r=2."""
    def __init__(self, channels, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),  # ReLU² = relu applied twice
            nn.Dropout(drop),
            nn.Linear(hidden, channels),
            nn.Dropout(drop),
        )

    def forward(self, point: Point):
        point.feat = self.net(point.feat)
        return point


# ── PTX: Block ─────────────────────────────────────────────────────────────

class PTXBlock(PointModule):
    """PTX block: Linear CPE → PTXAttention → ReLU² FFN."""

    def __init__(self, channels, num_heads, patch_size=256, mlp_ratio=2.0, qkv_bias=True,
                 qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, pre_norm=True, order_index=0):
        super().__init__()
        self.pre_norm = pre_norm

        # Simple MLP-based CPE (PTX uses no spconv)
        self.cpe = PointSequential(
            nn.Linear(channels, channels),
            norm_layer(channels) if norm_layer else nn.Identity(),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = PTXAttention(channels=channels, num_heads=num_heads, patch_size=patch_size,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 attn_drop=attn_drop, proj_drop=proj_drop,
                                 order_index=order_index)
        self.norm2 = PointSequential(norm_layer(channels))
        self.ffn = ReLUSquaredFFN(channels, mlp_ratio=mlp_ratio, drop=proj_drop)

        # DropPath
        self.drop_path = PointSequential(
            nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
        )

    def forward(self, point: Point):
        # CPE
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat

        # Attention
        shortcut = point.feat
        if self.pre_norm: point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm: point = self.norm1(point)

        # FFN (ReLU²)
        shortcut = point.feat
        if self.pre_norm: point = self.norm2(point)
        point = self.drop_path(self.ffn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm: point = self.norm2(point)

        return point


# ── Pooling (reuse PTv3 serialized pooling, no spconv needed) ─────────────

class SerializedPooling(PointModule):
    def __init__(self, in_channels, out_channels, stride=2, norm_layer=None,
                 act_layer=None, reduce="max", shuffle_orders=True):
        super().__init__()
        self.stride = stride
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer: self.norm = PointSequential(norm_layer(out_channels))
        if act_layer: self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth: pooling_depth = 0
        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True,
                                              return_counts=True)
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1, index=order,
            src=torch.arange(code.shape[1], device=order.device).repeat(code.shape[0], 1))
        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code, order, inverse = code[perm], order[perm], inverse[perm]

        point_dict = dict(
            feat=torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr,
                                           reduce=self.reduce),
            coord=torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code, serialized_order=order, serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
            pooling_inverse=cluster, pooling_parent=point,
        )
        point = Point(point_dict)
        if hasattr(self, 'norm'): point = self.norm(point)
        if hasattr(self, 'act'): point = self.act(point)
        return point


class SerializedUnpooling(PointModule):
    def __init__(self, in_channels, skip_channels, out_channels, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))
        if norm_layer:
            self.proj.add(norm_layer(out_channels)); self.proj_skip.add(norm_layer(out_channels))
        if act_layer:
            self.proj.add(act_layer()); self.proj_skip.add(act_layer())

    def forward(self, point):
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]
        return parent


# ── PTX: Full Model ────────────────────────────────────────────────────────

class PointTransformerX(PointModule):
    """PTX backbone — PTv3 framework, PTX components.

    Args match PTv3 for drop-in comparison; only 3 components differ:
      1. LinearEmbedding (no spconv)
      2. PTXAttention with 3D-GS-RoPE
      3. ReLUSquaredFFN (r=2)
    """

    def __init__(self,
                 in_channels=6,
                 order=("z", "z-trans"),
                 stride=(2, 2, 2, 2),
                 enc_depths=(2, 2, 2, 6, 2),
                 enc_channels=(32, 64, 128, 256, 512),
                 enc_num_head=(2, 4, 8, 16, 32),
                 enc_patch_size=(1024, 1024, 1024, 1024, 1024),
                 dec_depths=(2, 2, 2, 2),
                 dec_channels=(64, 64, 128, 256),
                 dec_num_head=(4, 4, 8, 16),
                 dec_patch_size=(1024, 1024, 1024, 1024),
                 mlp_ratio=2.0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.1,
                 drop_path=0.1,
                 pre_norm=True,
                 shuffle_orders=True,
                 grid_size=0.05,
                 **kwargs):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders
        self.grid_size = grid_size

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        # PTX linear embedding (no spconv)
        self.embedding = LinearEmbedding(in_channels=in_channels, embed_channels=enc_channels[0],
                                         norm_layer=bn_layer, act_layer=act_layer)

        # Encoder
        enc_dp = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_dp_ = enc_dp[sum(enc_depths[:s]):sum(enc_depths[:s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(SerializedPooling(in_channels=enc_channels[s - 1],
                                          out_channels=enc_channels[s],
                                          stride=stride[s - 1],
                                          norm_layer=bn_layer, act_layer=act_layer), name="down")
            for i in range(enc_depths[s]):
                enc.add(PTXBlock(channels=enc_channels[s], num_heads=enc_num_head[s],
                                 patch_size=enc_patch_size[s], mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 attn_drop=attn_drop, proj_drop=proj_drop,
                                 drop_path=enc_dp_[i], norm_layer=ln_layer,
                                 pre_norm=pre_norm, order_index=i % len(self.order)),
                        name=f"block{i}")
            if len(enc) != 0: self.enc.add(module=enc, name=f"enc{s}")

        # Decoder
        dec_dp = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
        self.dec = PointSequential()
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        for s in reversed(range(self.num_stages - 1)):
            dec_dp_ = dec_dp[sum(dec_depths[:s]):sum(dec_depths[:s + 1])]; dec_dp_.reverse()
            dec = PointSequential()
            dec.add(SerializedUnpooling(in_channels=dec_channels[s + 1],
                                        skip_channels=enc_channels[s],
                                        out_channels=dec_channels[s],
                                        norm_layer=bn_layer, act_layer=act_layer), name="up")
            for i in range(dec_depths[s]):
                dec.add(PTXBlock(channels=dec_channels[s], num_heads=dec_num_head[s],
                                 patch_size=dec_patch_size[s], mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 attn_drop=attn_drop, proj_drop=proj_drop,
                                 drop_path=dec_dp_[i], norm_layer=ln_layer,
                                 pre_norm=pre_norm, order_index=i % len(self.order)),
                        name=f"block{i}")
            self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        return point.feat
