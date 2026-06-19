"""
Preprocess full PartNet HDF5 dataset → per-category .npz for training.

Usage:
    python tools/preprocess_partnet_full.py --input datasets/raw/PartNet_full/sem_seg_h5/sem_seg_h5 \
                                            --output datasets/processed/partnet_full \
                                            --num-points 2048
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import h5py


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-categories", type=int, default=0)
    return p.parse_args()


def normalize(points: np.ndarray):
    """Unit-sphere normalize."""
    center = points.mean(axis=0)
    shifted = points - center
    scale = np.max(np.linalg.norm(shifted, axis=1))
    if scale < 1e-12:
        return shifted, center, 1.0
    return shifted / scale, center, scale


def remap_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Map labels to contiguous 0..K-1, return (remapped, K)."""
    unique = np.unique(labels)
    unique = unique[unique >= 0]  # filter -1/padding
    mapping = {old: new for new, old in enumerate(unique)}
    remapped = np.full_like(labels, -1)
    for old, new in mapping.items():
        remapped[labels == old] = new
    return remapped, len(unique)


def main():
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    rng = np.random.default_rng(args.seed)

    categories = sorted(
        d for d in input_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if args.max_categories > 0:
        categories = categories[: args.max_categories]

    all_stats = {}
    total_samples = 0

    for cat_dir in categories:
        cat_name = cat_dir.name
        print(f"\n{'='*60}\n  {cat_name}\n{'='*60}")

        # Collect HDF5 files per split
        for split in ["train", "val", "test"]:
            list_file = cat_dir / f"{split}_files.txt"
            if not list_file.exists():
                print(f"  [{split}] no split file, skipping")
                continue

            with open(list_file) as f:
                h5_rel = [line.strip().lstrip("./") for line in f if line.strip()]

            out_dir = output_root / cat_name / "samples" / split
            out_dir.mkdir(parents=True, exist_ok=True)
            label_map_path = output_root / cat_name / "meta" / "label_map.json"

            samples_written = 0
            for rel in h5_rel:
                h5_path = (cat_dir / rel).resolve()
                if not h5_path.exists():
                    print(f"  MISSING: {h5_path}")
                    continue

                with h5py.File(h5_path, "r") as hf:
                    data = hf["data"][:]       # (N, 10000, 3)
                    labels = hf["label_seg"][:]  # (N, 10000)
                    data_num = hf["data_num"][:]  # (N,)

                for i in range(data.shape[0]):
                    n_valid = int(data_num[i])
                    pts = data[i, :n_valid].astype(np.float64)
                    lbl = labels[i, :n_valid].astype(np.int64)

                    # Remove padding (-1) points
                    valid_mask = lbl >= 0
                    pts = pts[valid_mask]
                    lbl = lbl[valid_mask]

                    if pts.shape[0] < 10:
                        continue

                    # Sample num_points
                    if pts.shape[0] >= args.num_points:
                        idx = rng.choice(pts.shape[0], args.num_points, replace=False)
                    else:
                        idx = rng.choice(pts.shape[0], args.num_points, replace=True)
                    pts = pts[idx]
                    lbl = lbl[idx]

                    # Normalize
                    pts, center, scale = normalize(pts)

                    # Remap labels
                    lbl, num_parts = remap_labels(lbl)

                    # Save
                    sample_id = f"{h5_path.stem}_{i:06d}"
                    out_path = out_dir / f"{sample_id}.npz"
                    np.savez_compressed(
                        out_path,
                        points=pts.astype(np.float32),
                        seg_labels=lbl.astype(np.int64),
                        category=cat_name,
                        num_parts=np.array([num_parts], dtype=np.int64),
                    )
                    samples_written += 1

            print(f"  [{split}] {samples_written} samples ({len(h5_rel)} h5 files)")
            all_stats.setdefault(cat_name, {})[split] = samples_written
            total_samples += samples_written

        # Save label metadata
        meta_dir = output_root / cat_name / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "stats.json", "w") as f:
            json.dump(all_stats.get(cat_name, {}), f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_samples:,} samples across {len(categories)} categories")
    print(f"  Output: {output_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
