from pathlib import Path

import numpy as np

from datasets.partnet_dataset import PartNetDataset


def test_dataset_loads_npz(tmp_path: Path):
    processed_root = tmp_path / "processed"
    split_dir = tmp_path / "splits"
    sample_dir = processed_root / "samples" / "train"
    sample_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)

    sample_path = sample_dir / "sample_000.npz"
    np.savez_compressed(
        sample_path,
        points=np.random.randn(32, 3).astype(np.float32),
        seg_labels=np.random.randint(0, 4, size=(32,), dtype=np.int64),
        category_id=np.array([2], dtype=np.int64),
        sample_id="sample_000",
    )

    split_file = split_dir / "train.txt"
    split_file.write_text("samples/train/sample_000.npz\n", encoding="utf-8")

    ds = PartNetDataset(processed_root=processed_root, split_file=split_file, num_points=16, augment=False)
    item = ds[0]

    assert tuple(item["points"].shape) == (16, 3)
    assert tuple(item["labels"].shape) == (16,)
    assert int(item["category"].item()) == 2
