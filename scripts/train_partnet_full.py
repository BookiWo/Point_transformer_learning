"""
Per-category PartNet training script.

Supports:
  - Single category training:  --category Chair-3
  - All categories training:   --category all
  - Resume from checkpoint:    --resume path/to/best.pth

Usage:
    python scripts/train_partnet_full.py --config configs/partnet_pt_v2_full.yaml
    python scripts/train_partnet_full.py --config ... --category Chair-3 --max-train 100
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.partnet_full_dataset import PartNetFullDataset
from losses.segmentation_loss import SegmentationLoss
from models.point_transformer_v2_seg import PointTransformerV2Seg
from utils.metrics.segmentation_metrics import compute_segmentation_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/partnet_pt_v2_full.yaml")
    p.add_argument("--category", default="all")
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--resume", type=str, default="")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gather_categories(root: Path) -> list[str]:
    """Return sorted list of category dirs containing samples/."""
    cats = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "samples" / "train").exists():
            cats.append(d.name)
    return cats


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.get("training", {}).get("seed", 42)))

    data_root = Path(cfg["dataset"]["processed_root"])
    categories = [args.category] if args.category != "all" else gather_categories(data_root)
    if not categories:
        raise RuntimeError(f"No categories found in {data_root}")

    epochs = args.epochs if args.epochs > 0 else cfg["training"].get("epochs", 30)
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    grad_accum = int(tcfg.get("grad_accum", 1))

    all_results: dict[str, float] = {}

    for cat_idx, cat in enumerate(categories):
        print(f"\n{'='*60}")
        print(f"  [{cat_idx+1}/{len(categories)}] Category: {cat}")
        print(f"{'='*60}\n")

        # ---- Build datasets ----
        train_ds = PartNetFullDataset(data_root, "train", cat, num_points=cfg["dataset"].get("num_points", 2048), augment=True)
        val_ds = PartNetFullDataset(data_root, "val", cat, num_points=cfg["dataset"].get("num_points", 2048), augment=False)

        if args.max_train > 0:
            train_ds = Subset(train_ds, list(range(min(args.max_train, len(train_ds)))))
        if args.max_val > 0:
            val_ds = Subset(val_ds, list(range(min(args.max_val, len(val_ds)))))

        # Scan ALL training samples for max label (different samples may use
        # different annotation levels → variable num_parts per sample).
        max_label = 0
        raw_train = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
        for i in range(len(raw_train)):
            lbl = raw_train[i]["labels"]
            if hasattr(lbl, 'max'):
                max_label = max(max_label, int(lbl.max()))
            else:
                max_label = max(max_label, int(lbl.max()))
        num_parts = max_label + 1
        print(f"Samples: train={len(train_ds)}, val={len(val_ds)}, parts={num_parts} (max_label={max_label})")

        train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                                  num_workers=tcfg.get("num_workers", 0), pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                                num_workers=tcfg.get("num_workers", 0), pin_memory=False)

        # ---- Model ----
        model = PointTransformerV2Seg(
            input_dim=mcfg.get("input_dim", 3),
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            num_heads=mcfg["num_heads"],
            num_parts=num_parts,
            num_groups=mcfg.get("num_groups", 2),
            dropout=mcfg.get("dropout", 0.1),
            pe_multiplier=mcfg.get("pe_multiplier", True),
            grid_cell_size=mcfg.get("grid_cell_size", 0.05),
        ).to(device)
        print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

        criterion = SegmentationLoss(ignore_index=int(cfg.get("loss", {}).get("ignore_index", -1)))
        opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 1e-4))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        start_epoch = 0
        best_miou = -1.0

        # Resume
        if args.resume:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
            start_epoch = ckpt.get("epoch", 0)
            print(f"Resumed from epoch {start_epoch}")

        # ---- Output dirs ----
        exp_dir = Path(cfg["experiment"]["output_dir"]) / cat
        ckpt_dir = exp_dir / "checkpoints"
        log_path = exp_dir / "train_log.txt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        exp_dir.mkdir(parents=True, exist_ok=True)

        # ---- Training loop ----
        global_step = 0
        with log_path.open("w") as log_f:
            for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
                t0 = time.time()
                model.train()
                opt.zero_grad(set_to_none=True)
                train_losses = []

                for step, batch in enumerate(train_loader, start=1):
                    pts = batch["points"].to(device)
                    lbl = batch["labels"].to(device)
                    loss = criterion(model(pts), lbl) / grad_accum
                    loss.backward()
                    train_losses.append(loss.item())

                    if step % grad_accum == 0:
                        opt.step()
                        opt.zero_grad(set_to_none=True)

                    if step % 50 == 0:
                        print(f"  [E{epoch:03d}] step={step:04d} loss={loss.item():.4f}", flush=True)
                    global_step += 1

                scheduler.step()

                # Validation
                model.eval()
                val_acc, val_miou, n_val = 0.0, 0.0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        pts = batch["points"].to(device)
                        lbl = batch["labels"].to(device)
                        m = compute_segmentation_metrics(model(pts), lbl, num_parts, -1)
                        val_acc += m["overall_acc"]
                        val_miou += m["miou"]
                        n_val += 1

                if n_val > 0:
                    val_acc /= n_val
                    val_miou /= n_val

                train_loss = np.mean(train_losses) if train_losses else 0
                elapsed = time.time() - t0
                summary = (
                    f"[Epoch {epoch:03d}] loss={train_loss:.4f} "
                    f"val_acc={val_acc:.4f} val_miou={val_miou:.4f} | {elapsed:.0f}s"
                )
                print(f"  {summary}", flush=True)
                log_f.write(summary + "\n")

                # Save
                state = {"epoch": epoch, "model": model.state_dict(), "val_miou": val_miou}
                if epoch % max(1, tcfg.get("save_every", 5)) == 0:
                    torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pth")
                if val_miou > best_miou:
                    best_miou = val_miou
                    torch.save(state, ckpt_dir / "best.pth")

        all_results[cat] = best_miou
        print(f"\n  >> {cat}: best mIoU = {best_miou*100:.2f}%")

        # Free GPU memory before next category
        del model, opt, scheduler
        torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*60}")
    print(f"  ALL CATEGORIES COMPLETE")
    print(f"{'='*60}")
    if all_results:
        avg = sum(all_results.values()) / len(all_results)
        for cat, miou in sorted(all_results.items()):
            print(f"  {cat:<25} {miou*100:.2f}%")
        print(f"  {'─'*30}")
        print(f"  {'AVERAGE':<25} {avg*100:.2f}%")


if __name__ == "__main__":
    main()
