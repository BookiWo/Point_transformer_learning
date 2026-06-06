from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class PartNetDataset(Dataset):
    """Load processed PartNet samples from npz files."""

    def __init__(
        self,
        processed_root: str | Path,
        split_file: str | Path,
        num_points: int | None = None,
        augment: bool = False,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.split_file = Path(split_file)
        self.num_points = num_points
        self.augment = augment

        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        with self.split_file.open("r", encoding="utf-8") as f:
            rel_paths = [line.strip() for line in f.readlines() if line.strip()]

        self.sample_paths: List[Path] = [(self.processed_root / rel).resolve() for rel in rel_paths]
        if not self.sample_paths:
            raise FileNotFoundError(f"No samples found in split file: {self.split_file}")
        # Skip per-file existence check on WSL (slow DrvFs); trust the split file.

    def __len__(self) -> int:
        return len(self.sample_paths)

    def _resample(self, points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.num_points is None or points.shape[0] == self.num_points:
            return points, labels

        n = points.shape[0]
        replace = n < self.num_points
        idx = np.random.choice(n, size=self.num_points, replace=replace)
        return points[idx], labels[idx]

    def _augment(self, points: np.ndarray) -> np.ndarray:
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        jitter = np.random.normal(0.0, 0.005, size=points.shape).astype(np.float32)
        return (points @ rot.T).astype(np.float32) + jitter

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        path = self.sample_paths[idx]
        data = np.load(path, allow_pickle=True)

        points = np.asarray(data["points"], dtype=np.float32)
        if "seg_labels" in data:
            labels = np.asarray(data["seg_labels"], dtype=np.int64)
        elif "labels" in data:
            labels = np.asarray(data["labels"], dtype=np.int64)
        else:
            labels = np.full((points.shape[0],), -1, dtype=np.int64)

        points, labels = self._resample(points, labels)
        if self.augment:
            points = self._augment(points)

        if "category_id" in data:
            category_id = int(np.asarray(data["category_id"]).reshape(-1)[0])
        else:
            category_id = -1

        sample_id = str(data["sample_id"]) if "sample_id" in data else path.stem

        return {
            "points": torch.from_numpy(points),
            "labels": torch.from_numpy(labels),
            "category": torch.tensor(category_id, dtype=torch.long),
            "sample_id": sample_id,
        }
