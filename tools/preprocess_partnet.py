"""PartNet preprocessing entry script.

This script converts raw PartNet data (H5 or PLY) into normalized,
optionally resampled sample files for training and evaluation.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
from functools import lru_cache

import numpy as np
import open3d as o3d
import yaml
from tqdm import tqdm

try:
    import h5py  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    h5py = None

@lru_cache(maxsize=1)
def _load_label_hook():
    """Dynamically load optional label payload builder once.

    Keeping this hook optional allows preprocessing to run even when
    task-specific label conversion logic is absent.
    """
    hook_path = Path(__file__).resolve().with_name("preprocess_partnet_label_hooks.py")
    if not hook_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("preprocess_partnet_label_hooks", hook_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    hook = getattr(module, "build_label_payload", None)
    if callable(hook):
        return hook
    return None


def build_label_payload(file_path, config):
    hook = _load_label_hook()
    if callable(hook):
        return hook(file_path, config)
    return {}


def get_config(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_config():
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "Partnet_preprocess.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f), repo_root


def resolve_paths(config, repo_root):
    # Support both new dataset.* keys and legacy top-level keys.
    input_root = get_config(config, "dataset", "input_root", default=get_config(config, "input_root"))
    output_root = get_config(config, "dataset", "output_root", default=get_config(config, "output_root"))

    if not input_root or not output_root:
        raise KeyError("Missing input_root/output_root in config (supports both dataset.* and top-level keys).")

    dataset_path = Path(input_root)
    output_path = Path(output_root)

    if not dataset_path.is_absolute():
        dataset_path = (repo_root / dataset_path).resolve()
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()

    output_path.mkdir(parents=True, exist_ok=True)
    return dataset_path, output_path


def _resolve_split_dir(config, repo_root):
    split_dir = get_config(config, "dataset", "split_dir", default=get_config(config, "split_dir"))
    if not split_dir:
        return None
    split_path = Path(split_dir)
    if not split_path.is_absolute():
        split_path = (repo_root / split_path).resolve()
    return split_path


def normalize_points(points, method):
    if points.shape[0] == 0:
        return points, np.zeros(3, dtype=np.float64), 1.0

    if method == "unit_sphere":
        center = np.mean(points, axis=0)
        shifted = points - center
        scale = float(np.max(np.linalg.norm(shifted, axis=1)))
    elif method == "bounding_box":
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        center = (min_bound + max_bound) / 2.0
        shifted = points - center
        scale = float(np.max(max_bound - min_bound) / 2.0)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    # Avoid NaN/Inf on degenerate clouds with near-zero scale.
    if scale <= 1e-12:
        return shifted, center, 1.0
    return shifted / scale, center, scale


def maybe_sample(points, labels, colors, normals, cfg_sampling, rng):
    if not isinstance(cfg_sampling, dict):
        return points, labels, colors, normals

    num_points = cfg_sampling.get("num_points")
    if num_points is None:
        return points, labels, colors, normals

    num_points = int(num_points)
    if num_points <= 0 or points.shape[0] == 0:
        return points, labels, colors, normals

    # If points are insufficient, optionally upsample with replacement.
    allow_repeat = bool(cfg_sampling.get("allow_repeat_when_few_points", True))

    original_n = points.shape[0]
    if original_n >= num_points:
        idx = rng.choice(original_n, size=num_points, replace=False)
    elif allow_repeat:
        idx = rng.choice(original_n, size=num_points, replace=True)
    else:
        idx = np.arange(original_n)

    points = points[idx]
    if colors is not None and colors.shape[0] == original_n:
        colors = colors[idx]
    if normals is not None and normals.shape[0] == original_n:
        normals = normals[idx]
    if labels is not None and labels.shape[0] == original_n:
        labels = labels[idx]
    return points, labels, colors, normals


def save_sample(base_out_path, sample_format, points, seg_labels, category_id, colors, normals, extra_payload):
    sample_format = (sample_format or "ply").lower()
    base_out_path.parent.mkdir(parents=True, exist_ok=True)

    if sample_format == "ply":
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if colors is not None and colors.shape[0] == points.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if normals is not None and normals.shape[0] == points.shape[0]:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.io.write_point_cloud(str(base_out_path.with_suffix(".ply")), pcd)
        return

    if sample_format == "npz":
        # Keep a stable minimal schema consumed by PartNetDataset.
        payload = {
            "points": points.astype(np.float32),
            "seg_labels": seg_labels.astype(np.int64) if seg_labels is not None else np.zeros((points.shape[0],), dtype=np.int64),
            "category_id": np.array([int(category_id)], dtype=np.int64),
        }
        if colors is not None and colors.shape[0] == points.shape[0]:
            payload["colors"] = colors.astype(np.float32)
        if normals is not None and normals.shape[0] == points.shape[0]:
            payload["normals"] = normals.astype(np.float32)
        payload.update(extra_payload)
        np.savez_compressed(str(base_out_path.with_suffix(".npz")), **payload)
        return

    raise ValueError(f"Unsupported output.sample_format: {sample_format}")


def _iter_h5_samples(dataset_path, split_dir):
    if h5py is None:
        raise ImportError("h5py is required to preprocess HDF5 PartNet data.")

    split_files = {
        "train": "train_hdf5_file_list.txt",
        "val": "val_hdf5_file_list.txt",
        "test": "test_hdf5_file_list.txt",
    }
    # Iterate by official split lists so split assignment stays deterministic.
    for split_name, list_name in split_files.items():
        list_path = (split_dir / list_name).resolve()
        if not list_path.exists():
            continue
        with list_path.open("r", encoding="utf-8") as f:
            rel_h5_paths = [line.strip() for line in f.readlines() if line.strip()]

        for rel_h5 in rel_h5_paths:
            h5_path = (dataset_path / rel_h5).resolve()
            with h5py.File(h5_path, "r") as h5f:
                points_arr = h5f["data"][:]
                labels_arr = h5f["pid"][:]
                cat_arr = h5f["label"][:]

                for idx in range(points_arr.shape[0]):
                    sample_id = f"{split_name}_{h5_path.stem}_{idx:06d}"
                    yield {
                        "source": str(h5_path),
                        "split": split_name,
                        "sample_id": sample_id,
                        "points": np.asarray(points_arr[idx], dtype=np.float64),
                        "seg_labels": np.asarray(labels_arr[idx], dtype=np.int64),
                        "category_id": int(np.asarray(cat_arr[idx]).reshape(-1)[0]),
                    }


def _iter_ply_samples(dataset_path):
    for ply_path in sorted(dataset_path.rglob("*.ply")):
        rel_path = ply_path.relative_to(dataset_path)
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points, dtype=np.float64)
        if points.shape[0] == 0:
            continue
        colors = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None
        yield {
            "source": str(ply_path),
            "split": "unspecified",
            "sample_id": rel_path.as_posix().replace("/", "__").replace(".ply", ""),
            "points": points,
            "colors": colors,
            "normals": normals,
            "seg_labels": None,
            "category_id": -1,
            "rel_path": rel_path,
        }


def _save_split_files(split_samples, split_root):
    split_root.mkdir(parents=True, exist_ok=True)
    for split, ids in split_samples.items():
        out_file = split_root / f"{split}.txt"
        with out_file.open("w", encoding="utf-8") as f:
            for item in ids:
                f.write(f"{item}\n")


def main():
    config, repo_root = load_config()
    dataset_path, output_path = resolve_paths(config, repo_root)
    split_dir = _resolve_split_dir(config, repo_root)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Input root does not exist: {dataset_path}")

    seed = int(get_config(config, "runtime", "seed", default=42))
    rng = np.random.default_rng(seed)

    file_paths = sorted(dataset_path.rglob("*.ply"))
    max_samples = get_config(config, "runtime", "max_samples")
    if max_samples is not None:
        file_paths = file_paths[: int(max_samples)]

    overwrite = bool(get_config(config, "runtime", "overwrite", default=False))
    sample_format = get_config(config, "output", "sample_format", default="ply")
    normalize_enabled = bool(get_config(config, "normalize", "enabled", default=True))
    normalize_method = get_config(config, "normalize", "method", default="unit_sphere")
    save_center_scale = bool(get_config(config, "normalize", "save_center_scale", default=False))
    sampling_cfg = get_config(config, "sampling", default={})
    source_mode = str(get_config(config, "dataset", "source_format", default="auto")).lower()

    if source_mode not in {"auto", "h5", "ply"}:
        raise ValueError("dataset.source_format must be one of: auto/h5/ply")

    split_registry = {"train": [], "val": [], "test": [], "unspecified": []}
    stats = {
        "num_samples": 0,
        "num_points_total": 0,
        "split_counts": {"train": 0, "val": 0, "test": 0, "unspecified": 0},
        "category_counts": {},
    }

    # In auto mode, prefer H5 if split index files exist; otherwise fallback to PLY scan.
    use_h5 = source_mode == "h5" or (source_mode == "auto" and split_dir is not None and any((split_dir / p).exists() for p in ["train_hdf5_file_list.txt", "val_hdf5_file_list.txt", "test_hdf5_file_list.txt"]))

    if use_h5:
        sample_iter = _iter_h5_samples(dataset_path, split_dir)
        progress_desc = "Preprocessing H5 PartNet"
    else:
        sample_iter = _iter_ply_samples(dataset_path)
        progress_desc = "Preprocessing PLY PartNet"

    processed = 0
    for sample in tqdm(sample_iter, desc=progress_desc):
        if max_samples is not None and processed >= int(max_samples):
            break

        points = sample["points"]
        seg_labels = sample.get("seg_labels")
        colors = sample.get("colors")
        normals = sample.get("normals")
        category_id = int(sample.get("category_id", -1))
        split = sample.get("split", "unspecified")
        sample_id = sample.get("sample_id", f"sample_{processed:08d}")

        points, seg_labels, colors, normals = maybe_sample(points, seg_labels, colors, normals, sampling_cfg, rng)

        center = np.zeros(3, dtype=np.float64)
        scale = 1.0
        if normalize_enabled:
            points, center, scale = normalize_points(points, normalize_method)

        # Output path layout: samples/{split}/{sample_id}.{ext}
        out_base = output_path / "samples" / split / sample_id
        target_path = out_base.with_suffix(f".{sample_format}")
        if target_path.exists() and not overwrite:
            continue

        extra_payload = {
            "source_path": sample["source"],
            "sample_id": sample_id,
            **build_label_payload(sample["source"], config),
        }
        if save_center_scale:
            extra_payload["norm_center"] = center.astype(np.float32)
            extra_payload["norm_scale"] = np.array([scale], dtype=np.float32)

        save_sample(out_base, sample_format, points, seg_labels, category_id, colors, normals, extra_payload)

        split_registry.setdefault(split, []).append(f"samples/{split}/{sample_id}.{sample_format}")
        stats["num_samples"] += 1
        stats["num_points_total"] += int(points.shape[0])
        stats["split_counts"][split] = stats["split_counts"].get(split, 0) + 1
        stats["category_counts"][str(category_id)] = stats["category_counts"].get(str(category_id), 0) + 1
        processed += 1

    # Persist generated split index files for training/evaluation loaders.
    split_out_root = repo_root / "datasets" / "splits"
    _save_split_files(split_registry, split_out_root)

    meta_root = output_path / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    avg_points = 0.0
    if stats["num_samples"] > 0:
        avg_points = stats["num_points_total"] / stats["num_samples"]
    stats["avg_points"] = avg_points

    with (meta_root / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    with (meta_root / "preprocess_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()


