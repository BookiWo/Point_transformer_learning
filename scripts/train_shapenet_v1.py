"""
V1 baseline training on ShapeNet Part (official benchmark).

6-channel input (coord + normal), 50 global part classes, 16 categories.
Evaluates mIoU per official protocol using category2part mapping.
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
from torch.utils.data import DataLoader
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.shapenet_part_clean_dataset import ShapeNetPartCleanDataset
from losses.segmentation_loss import SegmentationLoss
from models.point_transformer_seg import PointTransformerSeg
from models.ptv2_official import PointTransformerV2Seg
from models.point_transformer_v3_seg import PointTransformerV3Seg
from models.ptx_seg import PointTransformerXSeg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/shapenet_v1_baseline.yaml")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--resume", type=str, default="")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def per_category_miou(logits, labels, cat_indices, dataset):
    """Official ShapeNet Part eval: per-category mIoU using category2part."""
    cat_metrics = {}
    for i in range(logits.shape[0]):
        cat_name = dataset.categories[cat_indices[i]]
        part_ids = dataset.category2part[cat_name]  # e.g. [12,13,14,15]

        # Only evaluate parts belonging to this category
        sample_logits = logits[i, :, part_ids]  # (N, K)
        pred = sample_logits.argmax(dim=-1)      # (N,)  index into part_ids
        gt = labels[i]                           # (N,)  global label 0-49

        parts_iou = []
        for k, pid in enumerate(part_ids):
            inter = ((pred == k) & (gt == pid)).sum().float()
            union = ((pred == k) | (gt == pid)).sum().float()
            if union > 0:
                parts_iou.append((inter / union).item())

        if parts_iou:
            if cat_name not in cat_metrics:
                cat_metrics[cat_name] = []
            cat_metrics[cat_name].append(np.mean(parts_iou))

    # Average per category
    cat_mious = {cat: np.mean(ious) for cat, ious in cat_metrics.items()}
    avg_miou = np.mean(list(cat_mious.values())) if cat_mious else 0.0
    return avg_miou, cat_mious


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.get("training", {}).get("seed", 42)))

    mcfg = cfg["model"]
    tcfg = cfg["training"]
    dcfg = cfg["dataset"]
    epochs = args.epochs if args.epochs > 0 else tcfg.get("epochs", 200)

    # Datasets
    train_ds = ShapeNetPartCleanDataset(dcfg["processed_root"], "train",
                                        num_points=dcfg.get("num_points", 2048),
                                        augment=True)
    val_ds = ShapeNetPartCleanDataset(dcfg["processed_root"], "val",
                                      num_points=dcfg.get("num_points", 2048),
                                      augment=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Classes: {train_ds.num_classes}, Categories: {train_ds.num_categories}")

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                              num_workers=tcfg.get("num_workers", 4), pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                            num_workers=tcfg.get("num_workers", 2), pin_memory=True)

    # Model — V1 or V2 official
    model_type = mcfg.get("model_type", "ptv1")
    if model_type == "ptv2_official":
        print("[Setup] Using Official PTv2 (Gofinge)")
        model = PointTransformerV2Seg(
            in_channels=int(mcfg["in_channels"]),
            num_classes=int(mcfg["num_classes"]),
            num_shape_classes=int(mcfg.get("num_shape_classes", 0)),
            patch_embed_channels=int(mcfg.get("patch_embed_channels", 48)),
            patch_embed_groups=int(mcfg.get("patch_embed_groups", 6)),
            enc_depths=list(mcfg.get("enc_depths", [2, 2, 6, 2])),
            enc_channels=list(mcfg.get("enc_channels", [96, 192, 384, 512])),
            enc_groups=list(mcfg.get("enc_groups", [12, 24, 48, 64])),
            enc_neighbours=list(mcfg.get("enc_neighbours", [16, 16, 16, 16])),
            dec_depths=list(mcfg.get("dec_depths", [1, 1, 1, 1])),
            dec_channels=list(mcfg.get("dec_channels", [48, 96, 192, 384])),
            dec_groups=list(mcfg.get("dec_groups", [6, 12, 24, 48])),
            dec_neighbours=list(mcfg.get("dec_neighbours", [16, 16, 16, 16])),
            grid_sizes=list(mcfg.get("grid_sizes", [0.06, 0.12, 0.24, 0.48])),
            pe_multiplier=bool(mcfg.get("pe_multiplier", False)),
            pe_bias=bool(mcfg.get("pe_bias", True)),
            drop_path_rate=float(mcfg.get("drop_path_rate", 0.1)),
        ).to(device)
    elif model_type == "ptv3":
        print("[Setup] Using Point Transformer V3 (official)")
        model = PointTransformerV3Seg(
            in_channels=int(mcfg["in_channels"]),
            num_parts=int(mcfg["num_parts"]),
            enc_channels=list(mcfg.get("enc_channels", [32, 64, 128, 256, 512])),
            enc_depths=list(mcfg.get("enc_depths", [2, 2, 2, 6, 2])),
            enc_num_head=list(mcfg.get("enc_num_head", [2, 4, 8, 16, 32])),
            enc_patch_size=list(mcfg.get("enc_patch_size", [1024]*5)),
            dec_depths=list(mcfg.get("dec_depths", [2, 2, 2, 2])),
            dec_channels=list(mcfg.get("dec_channels", [64, 64, 128, 256])),
            dec_num_head=list(mcfg.get("dec_num_head", [4, 4, 8, 16])),
            dec_patch_size=list(mcfg.get("dec_patch_size", [1024]*4)),
            stride=list(mcfg.get("stride", [2, 2, 2, 2])),
            mlp_ratio=float(mcfg.get("mlp_ratio", 4.0)),
            attn_drop=float(mcfg.get("attn_drop", 0.0)),
            proj_drop=float(mcfg.get("proj_drop", 0.1)),
            drop_path=float(mcfg.get("drop_path", 0.3)),
            enable_rpe=bool(mcfg.get("enable_rpe", False)),
            grid_size=float(mcfg.get("grid_size", 0.05)),
            order=mcfg.get("order", ("z", "z-trans")),
            shuffle_orders=bool(mcfg.get("shuffle_orders", True)),
        ).to(device)
    elif model_type == "ptx":
        print("[Setup] Using Point Transformer X (PTX)")
        model = PointTransformerXSeg(
            in_channels=int(mcfg["in_channels"]),
            num_classes=int(mcfg["num_parts"]),
            grid_size=float(mcfg.get("grid_size", 0.05)),
            enc_channels=list(mcfg.get("enc_channels", [32, 64, 128, 256, 512])),
            enc_depths=list(mcfg.get("enc_depths", [2, 2, 2, 6, 2])),
            enc_num_head=list(mcfg.get("enc_num_head", [2, 4, 8, 16, 32])),
            enc_patch_size=list(mcfg.get("enc_patch_size", [1024]*5)),
            dec_depths=list(mcfg.get("dec_depths", [2, 2, 2, 2])),
            dec_channels=list(mcfg.get("dec_channels", [64, 64, 128, 256])),
            dec_num_head=list(mcfg.get("dec_num_head", [4, 4, 8, 16])),
            dec_patch_size=list(mcfg.get("dec_patch_size", [1024]*4)),
            stride=list(mcfg.get("stride", [2, 2, 2, 2])),
            mlp_ratio=float(mcfg.get("mlp_ratio", 2.0)),
            qkv_bias=bool(mcfg.get("qkv_bias", True)),
            attn_drop=float(mcfg.get("attn_drop", 0.0)),
            proj_drop=float(mcfg.get("proj_drop", 0.1)),
            drop_path=float(mcfg.get("drop_path", 0.1)),
            pre_norm=bool(mcfg.get("pre_norm", True)),
            order=mcfg.get("order", ("z", "z-trans")),
            shuffle_orders=bool(mcfg.get("shuffle_orders", True)),
        ).to(device)
    else:
        model = PointTransformerSeg(
            input_dim=int(mcfg["input_dim"]),
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            num_heads=mcfg["num_heads"],
            num_parts=mcfg["num_parts"],
            dropout=float(mcfg.get("dropout", 0.1)),
            num_shape_classes=int(mcfg.get("num_shape_classes", 0)),
        ).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    criterion = SegmentationLoss(ignore_index=int(cfg.get("loss", {}).get("ignore_index", -1)))
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=40, T_mult=1, eta_min=tcfg["lr"] * 0.01
    )
    start_epoch, best_miou = 0, -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        start_epoch = ckpt.get("epoch", 0)
        best_miou = ckpt.get("miou", -1.0)
        print(f"Resumed from epoch {start_epoch}, best mIoU={best_miou:.4f}")

    exp_dir = Path(cfg["experiment"]["output_dir"])
    ckpt_dir = exp_dir / "checkpoints"
    log_path = exp_dir / "train_log.txt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log_f:
        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            t0 = time.time()
            model.train()
            opt.zero_grad(set_to_none=True)
            train_losses = []

            for step, batch in enumerate(train_loader, start=1):
                coord = batch["coord"].to(device)
                feat = batch["feat"].to(device)
                lbl = batch["labels"].to(device)

                cls_token = torch.tensor(
                    [train_ds.categories.index(c) for c in batch["category_name"]],
                    device=device, dtype=torch.long)
                loss = criterion(model(coord, feat=feat, cls_token=cls_token), lbl)
                loss.backward()
                train_losses.append(loss.item())
                opt.step()
                opt.zero_grad(set_to_none=True)

                if step % tcfg.get("log_interval", 100) == 0:
                    print(f"  [E{epoch:03d}] step={step:04d} loss={loss.item():.4f}", flush=True)

            scheduler.step()

            # Validation
            model.eval()
            val_losses, all_preds, all_labels, all_cats = [], [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    coord = batch["coord"].to(device)
                    feat = batch["feat"].to(device)
                    lbl = batch["labels"].to(device)
                    cat_idx = [train_ds.categories.index(c) for c in batch["category_name"]]

                    cls_token = torch.tensor(
                        [train_ds.categories.index(c) for c in batch["category_name"]],
                        device=device, dtype=torch.long)
                    logits = model(coord, feat=feat, cls_token=cls_token)
                    loss = criterion(logits, lbl)
                    val_losses.append(loss.item())
                    all_preds.append(logits)
                    all_labels.append(lbl)
                    all_cats.extend(cat_idx)

            train_loss = np.mean(train_losses) if train_losses else 0
            val_loss = np.mean(val_losses) if val_losses else 0

            # Per-category mIoU
            cat_miou, cat_mious = per_category_miou(
                torch.cat(all_preds), torch.cat(all_labels), all_cats, val_ds
            )

            elapsed = time.time() - t0
            summary = (f"[Epoch {epoch:03d}] loss={train_loss:.4f} val_loss={val_loss:.4f} "
                       f"cat_mIoU={cat_miou:.4f} | {elapsed:.0f}s lr={scheduler.get_last_lr()[0]:.2e}")
            print(f"  {summary}", flush=True)
            log_f.write(summary + "\n")

            # Per-category detail every 20 epochs
            if epoch % 20 == 0 and cat_mious:
                for cat, miou in sorted(cat_mious.items()):
                    print(f"    {cat}: {miou:.4f}", flush=True)

            state = {"epoch": epoch, "model": model.state_dict(), "miou": cat_miou}
            if epoch % max(1, tcfg.get("save_every", 10)) == 0:
                torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pth")
            if cat_miou > best_miou:
                best_miou = cat_miou
                torch.save(state, ckpt_dir / "best.pth")

    print(f"\nBest cat_mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
