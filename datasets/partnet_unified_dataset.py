"""
Unified PartNet dataset — all categories mixed with global label remapping.

Each category has its own independent part label space (0..K_i-1).
For joint training we remap to a global label space:
    category_i: 0..K_i-1 → offset_i .. offset_i + K_i - 1
    where offset_i = sum(K_j for j < i)

This allows training ONE model on ALL 50 categories simultaneously.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PartNetUnifiedDataset(Dataset):
    """Loads ALL PartNet categories, remaps labels to global space.

    Each sample returns:
        points:    (num_points, 3) float32
        labels:    (num_points,) int64 — GLOBAL label indices
        category:  str — category name (for per-category eval)
    """

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

        # Load pre-computed global label mapping (or build from train split)
        cache_path = self.root / "global_labels.json"
        if cache_path.exists():
            with open(cache_path) as f:
                mapping = json.load(f)
            self.categories = mapping["categories"]
            self.cat_num_parts = mapping["cat_num_parts"]
            self.cat_offset = mapping["cat_offset"]
            self.global_num_parts = mapping["global_num_parts"]
        else:
            # Build cache from TRAIN split (scan all files for accurate max_label)
            train_dir = "train"
            self.categories = sorted(
                d.name for d in self.root.iterdir()
                if d.is_dir() and (d / "samples" / train_dir).exists()
            )
            self.cat_num_parts = {}
            self.cat_offset = {}
            offset = 0
            for cat in self.categories:
                sd = self.root / cat / "samples" / train_dir
                files = sorted(sd.glob("*.npz"))
                max_label = 0
                for f in files:  # scan ALL files for correct max_label
                    data = np.load(f, allow_pickle=True)
                    max_label = max(max_label, int(data["seg_labels"].max()))
                K = max_label + 1
                self.cat_num_parts[cat] = K
                self.cat_offset[cat] = offset
                offset += K
            self.global_num_parts = offset
            # Save cache so future runs are instant
            mapping = {
                "categories": self.categories,
                "cat_num_parts": self.cat_num_parts,
                "cat_offset": self.cat_offset,
                "global_num_parts": self.global_num_parts,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(mapping, f, indent=2)
            print(f"[Unified] Built label cache: {len(self.categories)} categories, "
                  f"{self.global_num_parts} total parts → {cache_path}")

        print(f"[Unified] {len(self.categories)} categories, "
              f"{self.global_num_parts} total parts")

        # Category name → index mapping (for cls_token)
        self.category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        self.num_categories = len(self.categories)

        # Collect all sample files
        self.samples: list[tuple[Path, str, int, int, int]] = []  # (path, cat, K, offset, cat_idx)
        for cat in self.categories:
            sd = self.root / cat / "samples" / split
            K = self.cat_num_parts[cat]
            offset = self.cat_offset[cat]
            cat_idx = self.category_to_idx[cat]
            for f in sorted(sd.glob("*.npz")):
                self.samples.append((f, cat, K, offset, cat_idx))

        print(f"[Unified] {len(self.samples)} samples ({split})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, cat, K, offset, cat_idx = self.samples[idx]
        data = np.load(path, allow_pickle=True)
        points = data["points"].astype(np.float32)
        labels = data["seg_labels"].astype(np.int64)

        # Remap to global label space
        labels = labels + offset

        if self.augment:
            points = self._augment(points)

        # Pad / sample to num_points
        n = points.shape[0]
        if n > self.num_points:
            idx_sample = np.random.choice(n, self.num_points, replace=False)
        else:
            idx_sample = np.random.choice(n, self.num_points, replace=True)
        points = points[idx_sample]
        labels = labels[idx_sample]

        return {
            "points": torch.from_numpy(points),
            "labels": torch.from_numpy(labels),
            "sample_id": f"{cat}_{path.stem}",
            "category": cat,
            "category_idx": cat_idx,
            "cat_num_parts": K,
            "cat_offset": offset,
        }

    def _augment(self, points: np.ndarray) -> np.ndarray:
        theta = np.random.uniform(0, 2 * np.pi)
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ], dtype=np.float32)
        points = (points @ rot.T).astype(np.float32)
        points *= np.float32(np.random.uniform(0.8, 1.2))
        points += np.random.normal(0, 0.01, points.shape).astype(np.float32)
        return points
