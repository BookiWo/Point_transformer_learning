"""
Per-category PartNet dataset for train/eval on unified or per-category models.

Each category directory contains samples/{train,val,test}/*.npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PartNetFullDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        category: str | None = None,
        num_points: int = 2048,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.num_points = num_points
        self.augment = augment

        # List all sample files
        if category:
            sample_dir = self.root / category / "samples" / split
        else:
            # Collect from all categories
            sample_dir = None

        self.samples: list[Path] = []
        if sample_dir and sample_dir.exists():
            self.samples = sorted(sample_dir.glob("*.npz"))
        elif not category:
            for cat_dir in sorted(self.root.iterdir()):
                sd = cat_dir / "samples" / split
                if sd.exists():
                    self.samples.extend(sorted(sd.glob("*.npz")))

        # Determine num_parts from first sample
        self.num_parts = 50  # default
        if len(self.samples) > 0:
            data = np.load(self.samples[0], allow_pickle=True)
            if "num_parts" in data:
                self.num_parts = int(data["num_parts"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self.samples[idx], allow_pickle=True)
        points = data["points"].astype(np.float32)
        labels = data["seg_labels"].astype(np.int64)
        num_parts = int(data.get("num_parts", self.num_parts))

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
            "sample_id": str(self.samples[idx].stem),
            "num_parts": num_parts,
        }

    def _augment(self, points: np.ndarray) -> np.ndarray:
        # Random rotation around Z
        theta = np.random.uniform(0, 2 * np.pi)
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ], dtype=np.float32)
        points = (points @ rot.T).astype(np.float32)
        # Random scaling
        points *= np.float32(np.random.uniform(0.8, 1.2))
        # Random jitter
        points += np.random.normal(0, 0.01, points.shape).astype(np.float32)
        return points
