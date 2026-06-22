"""
ShapeNet Part dataset loader — official format (coord + normal, global part labels).

Each .npz contains:
    coord:      (N, 3) float32 — centered coordinates
    feat:       (N, 6) float32 — [coord, normal]
    seg_labels: (N,) int64 — global part ID 0-49
    category_id: int — category index 0-15
    part_ids:   (K,) int64 — valid part IDs for this category
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNetPartCleanDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        num_points: int = 2048,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.num_points = num_points
        self.augment = augment

        sample_dir = self.root / "samples" / split
        self.files = sorted(sample_dir.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files in {sample_dir}")

        # Load metadata
        meta_path = self.root / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
            self.num_classes = self.meta["num_classes"]       # 50
            self.num_categories = self.meta["num_categories"]  # 16
            self.categories = self.meta["categories"]
            self.category2part = self.meta["category2part"]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self.files[idx], allow_pickle=True)
        coord = data["coord"].astype(np.float32)        # (N, 3)
        feat = data["feat"].astype(np.float32)          # (N, 6): coord+normal
        labels = data["seg_labels"].astype(np.int64)     # (N,)  global 0-49
        cat_idx = int(data["category_id"].item())        # 0-15
        cat_name = str(data["category_name"])            # e.g. "Chair"

        if self.augment:
            coord, feat = self._augment(coord, feat)

        # Sample to num_points
        n = coord.shape[0]
        if n > self.num_points:
            idx_sample = np.random.choice(n, self.num_points, replace=False)
        else:
            idx_sample = np.random.choice(n, self.num_points, replace=True)
        coord = coord[idx_sample]
        feat = feat[idx_sample]
        labels = labels[idx_sample]

        return {
            "coord": torch.from_numpy(coord),      # (2048, 3)
            "feat": torch.from_numpy(feat),        # (2048, 6)
            "labels": torch.from_numpy(labels),     # (2048,)
            "category_idx": cat_idx,                # int
            "category_name": cat_name,
            "sample_id": self.files[idx].stem,
        }

    def _augment(self, coord: np.ndarray, feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # feature = [coord, normal] — rotate coord, normal stays consistent
        theta = np.random.uniform(0, 2 * np.pi)
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ], dtype=np.float32)
        coord = (coord @ rot.T).astype(np.float32)
        # Also rotate the coord part of feat (first 3 channels)
        feat_coord = feat[:, :3] @ rot.T
        feat_normal = feat[:, 3:6] @ rot.T
        feat = np.concatenate([feat_coord, feat_normal], axis=1).astype(np.float32)

        # Random scaling
        scale = np.float32(np.random.uniform(0.9, 1.1))
        coord *= scale
        feat[:, :3] *= scale  # scale coord part, keep normal unchanged

        # Random jitter
        coord += np.random.normal(0, 0.005, coord.shape).astype(np.float32)
        feat[:, :3] = feat[:, :3] + np.random.normal(0, 0.005, coord.shape).astype(np.float32)

        return coord, feat
