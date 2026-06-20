"""
Unified PartNet training — ONE model trained on ALL categories simultaneously.

Labels are remapped to a global label space so the model learns all parts
from a shared feature backbone. Per-category mIoU is computed for evaluation.

Usage:
    python scripts/train_partnet_unified.py --config configs/partnet_pt_v2_unified.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.partnet_unified_dataset import PartNetUnifiedDataset
from losses.segmentation_loss import SegmentationLoss
from models.point_transformer_v2_seg import PointTransformerV2Seg
from models.point_transformer_v3_seg import PointTransformerV3Seg
from utils.metrics.segmentation_metrics import compute_segmentation_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/partnet_pt_v2_unified.yaml")
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


def compute_per_category_miou(logits, labels, batch_categories, dataset):
    """Compute mIoU per category from a batch of mixed samples."""
    cat_metrics = {}
    for i, cat in enumerate(batch_categories):
        if cat not in cat_metrics:
            cat_metrics[cat] = {"acc_sum": 0.0, "miou_sum": 0.0, "count": 0}

        offset = dataset.cat_offset[cat]
        K = dataset.cat_num_parts[cat]

        # Extract per-sample logits and labels, remap back to local
        sample_logits = logits[i:i+1, :, offset:offset+K]  # (1, N, K)
        sample_labels = labels[i:i+1] - offset  # back to local

        # Mask out -1 (padding)
        valid = sample_labels >= 0
        if valid.sum() == 0:
            continue

        # Use only valid parts for metrics
        metrics = compute_segmentation_metrics(
            sample_logits, sample_labels, num_parts=K, ignore_index=-1
        )
        cat_metrics[cat]["acc_sum"] += metrics["overall_acc"]
        cat_metrics[cat]["miou_sum"] += metrics["miou"]
        cat_metrics[cat]["count"] += 1

    return cat_metrics


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.get("training", {}).get("seed", 42)))

    data_root = Path(cfg["dataset"]["processed_root"])
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    epochs = args.epochs if args.epochs > 0 else tcfg.get("epochs", 50)
    grad_accum = int(tcfg.get("grad_accum", 1))

    # ---- Unified datasets (ALL categories mixed) ----
    train_ds = PartNetUnifiedDataset(data_root, "train",
                                     num_points=cfg["dataset"].get("num_points", 2048),
                                     augment=True)
    val_ds = PartNetUnifiedDataset(data_root, "val",
                                   num_points=cfg["dataset"].get("num_points", 2048),
                                   augment=False)

    if args.max_train > 0:
        train_ds = Subset(train_ds, list(range(min(args.max_train, len(train_ds)))))
    if args.max_val > 0:
        val_ds = Subset(val_ds, list(range(min(args.max_val, len(val_ds)))))

    # Unwrap Subset to get the real dataset (for global_num_parts, category offset, etc.)
    ds_ref = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
    global_num_parts = ds_ref.global_num_parts

    print(f"\nSamples: train={len(train_ds):,}, val={len(val_ds):,}")
    print(f"Global parts: {global_num_parts}")

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                              num_workers=tcfg.get("num_workers", 4), pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                            num_workers=tcfg.get("num_workers", 2), pin_memory=True)

    # ---- Model ----
    if mcfg.get("model_type") == "ptv3":
        print("[Setup] Using Point Transformer V3 backbone")
        model = PointTransformerV3Seg(
            in_channels=int(mcfg.get("in_channels", 3)),
            num_parts=global_num_parts,
            enc_channels=list(mcfg.get("enc_channels", [32, 64, 128, 256, 512])),
            enc_depths=list(mcfg.get("enc_depths", [2, 2, 2, 2, 2])),
            enc_num_head=list(mcfg.get("enc_num_head", [2, 4, 8, 16, 32])),
            enc_patch_size=list(mcfg.get("enc_patch_size", [1024]*5)),
            dec_depths=list(mcfg.get("dec_depths", [2, 2, 2, 2])),
            dec_channels=list(mcfg.get("dec_channels", [64, 64, 128, 256])),
            dec_num_head=list(mcfg.get("dec_num_head", [4, 4, 8, 16])),
            dec_patch_size=list(mcfg.get("dec_patch_size", [1024]*4)),
            stride=list(mcfg.get("stride", [2, 2, 2, 2])),
            mlp_ratio=float(mcfg.get("mlp_ratio", 2.0)),
            attn_drop=float(mcfg.get("attn_drop", 0.0)),
            proj_drop=float(mcfg.get("proj_drop", 0.1)),
            drop_path=float(mcfg.get("drop_path", 0.1)),
            enable_rpe=bool(mcfg.get("enable_rpe", False)),
            grid_size=float(mcfg.get("grid_size", 0.05)),
            order=mcfg.get("order", ("z", "z-trans")),
            shuffle_orders=bool(mcfg.get("shuffle_orders", True)),
        ).to(device)
    else:
        print("[Setup] Using Point Transformer V2 backbone")
        model = PointTransformerV2Seg(
            input_dim=mcfg.get("input_dim", 3),
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            num_heads=mcfg["num_heads"],
            num_parts=global_num_parts,
            num_groups=mcfg.get("num_groups", 2),
            dropout=mcfg.get("dropout", 0.1),
            pe_multiplier=mcfg.get("pe_multiplier", True),
            grid_cell_size=mcfg.get("grid_cell_size", 0.05),
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    criterion = SegmentationLoss(ignore_index=int(cfg.get("loss", {}).get("ignore_index", -1)))
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=min(epochs, 40), T_mult=1, eta_min=tcfg["lr"] * 0.01
    )
    start_epoch = 0
    best_miou = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        start_epoch = ckpt.get("epoch", 0)
        best_miou = ckpt.get("val_miou", -1.0)
        print(f"Resumed from epoch {start_epoch}")

    # ---- Output dirs ----
    exp_dir = Path(cfg["experiment"]["output_dir"])
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

                if step % max(1, tcfg.get("log_interval", 100)) == 0:
                    print(f"  [E{epoch:03d}] step={step:05d} loss={loss.item():.4f}", flush=True)
                global_step += 1

            # Flush remaining gradients
            if len(train_losses) % grad_accum != 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

            scheduler.step()

            # ---- Validation (global + per-category) ----
            model.eval()
            global_acc, global_miou, n_val = 0.0, 0.0, 0
            all_cat_metrics = {}

            with torch.no_grad():
                for batch in val_loader:
                    pts = batch["points"].to(device)
                    lbl = batch["labels"].to(device)
                    logits = model(pts)

                    # Global metrics
                    m = compute_segmentation_metrics(logits, lbl, global_num_parts, -1)
                    global_acc += m["overall_acc"]
                    global_miou += m["miou"]
                    n_val += 1

                    # Per-category metrics
                    cat_metrics = compute_per_category_miou(
                        logits, lbl, batch["category"], train_ds.dataset
                        if hasattr(train_ds, 'dataset') else train_ds
                    )
                    for cat, metrics in cat_metrics.items():
                        if cat not in all_cat_metrics:
                            all_cat_metrics[cat] = {"acc_sum": 0.0, "miou_sum": 0.0, "count": 0}
                        all_cat_metrics[cat]["acc_sum"] += metrics["acc_sum"]
                        all_cat_metrics[cat]["miou_sum"] += metrics["miou_sum"]
                        all_cat_metrics[cat]["count"] += metrics["count"]

            if n_val > 0:
                global_acc /= n_val
                global_miou /= n_val

            train_loss = np.mean(train_losses) if train_losses else 0
            elapsed = time.time() - t0

            # Per-category summary
            cat_summary = ""
            if all_cat_metrics:
                cat_mious = {
                    cat: m["miou_sum"] / max(m["count"], 1)
                    for cat, m in all_cat_metrics.items()
                }
                avg_cat_miou = np.mean(list(cat_mious.values()))
                cat_summary = f" | avg_cat_miou={avg_cat_miou:.4f}"

            summary = (
                f"[Epoch {epoch:03d}] loss={train_loss:.4f} "
                f"val_acc={global_acc:.4f} val_miou={global_miou:.4f}{cat_summary} "
                f"| {elapsed:.0f}s lr={scheduler.get_last_lr()[0]:.2e}"
            )
            print(f"  {summary}", flush=True)
            log_f.write(summary + "\n")

            # Save
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "val_miou": global_miou,
                "cat_mious": {
                    cat: m["miou_sum"] / max(m["count"], 1)
                    for cat, m in all_cat_metrics.items()
                } if all_cat_metrics else {},
            }
            if epoch % max(1, tcfg.get("save_every", 10)) == 0:
                torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pth")
            if global_miou > best_miou:
                best_miou = global_miou
                torch.save(state, ckpt_dir / "best.pth")

    # Save category mapping for evaluation
    mapping = {
        "categories": train_ds.dataset.categories if hasattr(train_ds, 'dataset') else [],
        "cat_num_parts": train_ds.dataset.cat_num_parts if hasattr(train_ds, 'dataset') else {},
        "cat_offset": train_ds.dataset.cat_offset if hasattr(train_ds, 'dataset') else {},
        "global_num_parts": global_num_parts,
    }
    with (exp_dir / "label_mapping.json").open("w") as f:
        json.dump(mapping, f, indent=2)

    # Final per-category report
    print(f"\n{'='*60}")
    print("  Per-category results:")
    print(f"{'='*60}")
    for cat in sorted(all_cat_metrics.keys()):
        m = all_cat_metrics[cat]
        miou = m["miou_sum"] / max(m["count"], 1)
        acc = m["acc_sum"] / max(m["count"], 1)
        print(f"  {cat:<25} acc={acc:.4f}  miou={miou:.4f}")


if __name__ == "__main__":
    main()
