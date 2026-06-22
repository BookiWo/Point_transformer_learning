"""
Preprocess official ShapeNet Part dataset for V1/V2/V3 training.

Reads the original .txt files (x y z nx ny nz label), centers coordinates,
samples to num_points, and outputs .npz files compatible with
our training pipeline.

Output format per .npz:
    coord:      (num_points, 3)  float32 — 3D coordinates (for KNN / pooling)
    feat:       (num_points, 6)  float32 — [coord, normal] (for feature backbone)
    seg_labels: (num_points,)    int64   — global part label 0-49
    category_id: int             — category index 0-15

Usage:
    python tools/preprocess_shapenet_part.py \
        --input datasets/raw/shapeNet/shapenetcore_partanno_segmentation_benchmark_v0_normal \
        --output datasets/processed/shapenet_part_clean \
        --num-points 2048
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CATEGORY2PART = {
    "Airplane":   [0, 1, 2, 3],
    "Bag":        [4, 5],
    "Cap":        [6, 7],
    "Car":        [8, 9, 10, 11],
    "Chair":      [12, 13, 14, 15],
    "Earphone":   [16, 17, 18],
    "Guitar":     [19, 20, 21],
    "Knife":      [22, 23],
    "Lamp":       [24, 25, 26, 27],
    "Laptop":     [28, 29],
    "Motorbike":  [30, 31, 32, 33, 34, 35],
    "Mug":        [36, 37],
    "Pistol":     [38, 39, 40],
    "Rocket":     [41, 42, 43],
    "Skateboard": [44, 45, 46],
    "Table":      [47, 48, 49],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to extracted _normal dataset")
    p.add_argument("--output", required=True, help="Output directory for processed .npz")
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_txt(path: Path) -> np.ndarray:
    """Load a ShapeNet Part .txt file (x y z nx ny nz label)."""
    data = np.loadtxt(path, dtype=np.float32)
    return data


def main():
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    rng = np.random.default_rng(args.seed)

    # Category mapping: offset_dir → category_name
    synset2cat = {}
    mapping_file = input_root / "synsetoffset2category.txt"
    with open(mapping_file) as f:
        for line in f:
            cat_name, offset = line.strip().split()
            synset2cat[offset] = cat_name

    categories = sorted(synset2cat.keys())

    # Build split file lists
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_file = input_root / "train_test_split" / f"shuffled_{split_name}_file_list.json"
        with open(split_file) as f:
            file_list = json.load(f)
        # Each entry like "shape_data/02691156/1021a...txt"
        # Map to local path: "02691156/1021a...txt"
        resolved = []
        for entry in file_list:
            parts = entry.replace("shape_data/", "").split("/")
            offset_dir = parts[0]
            filename = parts[1]
            resolved.append((offset_dir, filename))
        splits[split_name] = resolved
        print(f"{split_name}: {len(resolved)} samples")

    # Category index mapping
    cat_order = sorted(synset2cat.keys())
    offset2idx = {od: i for i, od in enumerate(cat_order)}

    total = 0
    for split_name, file_list in splits.items():
        out_dir = output_root / "samples" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        split_count = 0

        for offset_dir, filename in file_list:
            txt_path = input_root / offset_dir / (filename + ".txt")
            if not txt_path.exists():
                print(f"  MISSING: {txt_path}")
                continue

            cat_name = synset2cat[offset_dir]
            cat_idx = offset2idx[offset_dir]
            part_ids = CATEGORY2PART.get(cat_name, [])

            data = load_txt(txt_path)  # (N, 7): x y z nx ny nz label
            coord = data[:, :3].astype(np.float32)
            normal = data[:, 3:6].astype(np.float32)
            labels = data[:, 6].astype(np.int64)

            # Center only (no unit-sphere scaling — matches official NormalizeCoord)
            center = coord.mean(axis=0)
            coord = coord - center

            # Features: 6 channels (coord + normal)
            feat = np.concatenate([coord, normal], axis=1).astype(np.float32)

            # Sample to num_points
            n = coord.shape[0]
            if n >= args.num_points:
                idx = rng.choice(n, args.num_points, replace=False)
            else:
                idx = rng.choice(n, args.num_points, replace=True)
            coord = coord[idx]
            feat = feat[idx]
            labels = labels[idx]

            # Save
            np.savez_compressed(
                out_dir / f"{cat_name}_{filename}.npz",
                coord=coord,
                feat=feat,
                seg_labels=labels,
                category_id=np.array([cat_idx], dtype=np.int64),
                category_name=cat_name,
                part_ids=np.array(part_ids, dtype=np.int64),
            )
            split_count += 1

        print(f"  [{split_name}] {split_count}/{len(file_list)} samples")
        total += split_count

    # Save metadata
    meta = {
        "num_classes": 50,
        "num_categories": 16,
        "categories": [synset2cat[od] for od in categories],
        "category2part": CATEGORY2PART,
    }
    with open(output_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTotal: {total} samples → {output_root}")
    print(f"Classes: 50, Input dim: 6 (xyz + normal)")


if __name__ == "__main__":
    main()
