"""Quick training script with real-time stderr output and metrics JSON export."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.partnet_dataset import PartNetDataset
from losses.segmentation_loss import SegmentationLoss
from models.point_transformer_seg import PointTransformerSeg
from utils.metrics.segmentation_metrics import compute_segmentation_metrics


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> None:
    import yaml

    config_path = REPO_ROOT / "configs" / "partnet_pt_baseline.yaml"
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["dataset"]
    processed_root = Path(dcfg["processed_root"]).resolve()
    split_dir = Path(dcfg["split_dir"]).resolve()

    # ── load full datasets then subset for speed ──────────────────────────
    log("Loading datasets...")
    full_train = PartNetDataset(
        processed_root=processed_root,
        split_file=split_dir / dcfg["train_split"],
        num_points=512,  # fewer points → faster
        augment=True,
    )
    full_val = PartNetDataset(
        processed_root=processed_root,
        split_file=split_dir / dcfg["val_split"],
        num_points=512,
        augment=False,
    )

    # Use subsets for quick experimentation
    n_train = min(2000, len(full_train))
    n_val = min(400, len(full_val))
    indices_train = list(range(n_train))
    indices_val = list(range(n_val))

    train_ds = Subset(full_train, indices_train)
    val_ds = Subset(full_val, indices_val)
    log(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # ── model (smaller for speed) ─────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    model = PointTransformerSeg(
        input_dim=3,
        hidden_dim=64,  # smaller hidden dim
        num_layers=4,
        num_heads=4,
        num_parts=int(dcfg["num_parts"]),
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {n_params:,}")

    criterion = SegmentationLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    epochs = 30

    # ── output dirs ───────────────────────────────────────────────────────
    exp_dir = REPO_ROOT / "experiments" / "quick_train"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output: {exp_dir}")

    # ── metrics history ───────────────────────────────────────────────────
    history: list[dict] = []
    best_miou = -1.0
    global_step = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- train ---
        model.train()
        train_losses = []
        for batch in train_loader:
            points = batch["points"].to(device)
            labels = batch["labels"].to(device)
            logits = model(points)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            global_step += 1

        # --- val ---
        model.eval()
        val_acc = 0.0
        val_miou = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                points = batch["points"].to(device)
                labels = batch["labels"].to(device)
                logits = model(points)
                metrics = compute_segmentation_metrics(
                    logits, labels, num_parts=int(dcfg["num_parts"]), ignore_index=-1
                )
                val_acc += metrics["overall_acc"]
                val_miou += metrics["miou"]
                n_val_batches += 1

        if n_val_batches > 0:
            val_acc /= n_val_batches
            val_miou /= n_val_batches

        scheduler.step()
        train_loss = sum(train_losses) / max(1, len(train_losses))
        elapsed = time.time() - t0

        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_miou": round(val_miou, 4),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            "time_s": round(elapsed, 1),
        }
        history.append(epoch_info)

        log(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"acc={val_acc:.4f} | miou={val_miou:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.0f}s"
        )

        # save best
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_miou": val_miou},
                exp_dir / "best.pth",
            )

        # save periodic
        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_miou": val_miou},
                exp_dir / f"epoch_{epoch:03d}.pth",
            )

    # ── save metrics history ──────────────────────────────────────────────
    with (exp_dir / "metrics.json").open("w") as f:
        json.dump(history, f, indent=2)

    log(f"Done! Best mIoU: {best_miou:.4f}, metrics saved to {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
