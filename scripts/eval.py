from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.partnet_dataset import PartNetDataset  # type: ignore[import-not-found]
from models.point_transformer_seg import PointTransformerSeg  # type: ignore[import-not-found]
from models.point_transformer_v2_seg import PointTransformerV2Seg  # type: ignore[import-not-found]
from utils.metrics.segmentation_metrics import compute_segmentation_metrics  # type: ignore[import-not-found]
from utils.visualization.pointcloud_viz import label_to_color, save_xyzrgb_ply  # type: ignore[import-not-found]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Point Transformer baseline")
    parser.add_argument("--config", type=str, default="configs/partnet_pt_baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--save-viz", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcfg = cfg["dataset"]

    split_key = f"{args.split}_split"
    split_file = Path(dcfg["split_dir"]).resolve() / dcfg[split_key]
    ds = PartNetDataset(
        processed_root=Path(dcfg["processed_root"]).resolve(),
        split_file=split_file,
        num_points=dcfg.get("num_points"),
        augment=False,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    mcfg = cfg["model"]
    if "num_groups" in mcfg or "pe_multiplier" in mcfg or "grid_cell_size" in mcfg:
        model = PointTransformerV2Seg(
            input_dim=int(mcfg.get("input_dim", 3)),
            hidden_dim=int(mcfg.get("hidden_dim", 128)),
            num_layers=int(mcfg.get("num_layers", 4)),
            num_heads=int(mcfg.get("num_heads", 4)),
            num_parts=int(cfg["dataset"]["num_parts"]),
            num_groups=int(mcfg.get("num_groups", 2)),
            dropout=float(mcfg.get("dropout", 0.1)),
            pe_multiplier=bool(mcfg.get("pe_multiplier", True)),
            grid_cell_size=float(mcfg.get("grid_cell_size", 0.04)),
        ).to(device)
    else:
        model = PointTransformerSeg(
            input_dim=int(mcfg.get("input_dim", 3)),
            hidden_dim=int(mcfg.get("hidden_dim", 128)),
            num_layers=int(mcfg.get("num_layers", 4)),
            num_heads=int(mcfg.get("num_heads", 4)),
            num_parts=int(cfg["dataset"]["num_parts"]),
            dropout=float(mcfg.get("dropout", 0.1)),
        ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    total = {"overall_acc": 0.0, "miou": 0.0}
    num_batches = 0

    pred_root = Path("outputs/preds")
    viz_root = Path("outputs/viz")
    pred_root.mkdir(parents=True, exist_ok=True)
    if args.save_viz:
        viz_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            labels = batch["labels"].to(device)
            logits = model(points)
            metrics = compute_segmentation_metrics(
                logits,
                labels,
                num_parts=int(cfg["dataset"]["num_parts"]),
                ignore_index=int(cfg["loss"].get("ignore_index", -1)),
            )
            total["overall_acc"] += metrics["overall_acc"]
            total["miou"] += metrics["miou"]
            num_batches += 1

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            points_np = points.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(preds.shape[0]):
                sample_id = str(batch["sample_id"][i])
                np.savez_compressed(
                    pred_root / f"{sample_id}.npz",
                    points=points_np[i],
                    pred_labels=preds[i],
                    gt_labels=labels_np[i],
                )

                if args.save_viz:
                    colors = label_to_color(preds[i])
                    save_xyzrgb_ply(viz_root / f"{sample_id}_pred.ply", points_np[i], colors)

    if num_batches > 0:
        total = {k: v / num_batches for k, v in total.items()}

    print(f"Eval split={args.split} overall_acc={total['overall_acc']:.4f} miou={total['miou']:.4f}")


if __name__ == "__main__":
    main()
