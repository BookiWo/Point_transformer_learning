from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.partnet_dataset import PartNetDataset  # type: ignore[import-not-found]
from losses.segmentation_loss import SegmentationLoss  # type: ignore[import-not-found]
from models.point_transformer_seg import PointTransformerSeg  # type: ignore[import-not-found]
from models.point_transformer_v2_seg import PointTransformerV2Seg  # type: ignore[import-not-found]
from utils.metrics.segmentation_metrics import compute_segmentation_metrics  # type: ignore[import-not-found]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Point Transformer baseline on processed PartNet data")
    parser.add_argument("--config", type=str, default="configs/partnet_pt_baseline.yaml")
    parser.add_argument("--max-train", type=int, default=0, help="Limit training samples (0 = all)")
    parser.add_argument("--max-val", type=int, default=0, help="Limit validation samples (0 = all)")
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs (0 = use config)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_cfg: str) -> torch.device:
    if device_cfg != "auto":
        return torch.device(device_cfg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(cfg: dict, max_train: int = 0, max_val: int = 0) -> tuple[DataLoader, DataLoader]:
    dcfg = cfg["dataset"]
    tcfg = cfg["training"]

    processed_root = Path(dcfg["processed_root"]).resolve()
    split_dir = Path(dcfg["split_dir"]).resolve()

    train_ds = PartNetDataset(
        processed_root=processed_root,
        split_file=split_dir / dcfg["train_split"],
        num_points=dcfg.get("num_points"),
        augment=True,
    )
    val_ds = PartNetDataset(
        processed_root=processed_root,
        split_file=split_dir / dcfg["val_split"],
        num_points=dcfg.get("num_points"),
        augment=False,
    )

    if max_train > 0 and max_train < len(train_ds):
        train_ds = Subset(train_ds, list(range(max_train)))
        print(f"[Subset] Training samples: {max_train}", flush=True)
    if max_val > 0 and max_val < len(val_ds):
        val_ds = Subset(val_ds, list(range(max_val)))
        print(f"[Subset] Validation samples: {max_val}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(tcfg.get("num_workers", 4)),
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=int(tcfg.get("num_workers", 4)),
        pin_memory=False,
    )
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["training"].get("seed", 42)))
    device = choose_device(str(cfg["training"].get("device", "auto")))

    print(f"[Setup] Device: {device}", flush=True)
    print(f"[Setup] Points: {cfg['dataset']['num_points']}, Parts: {cfg['dataset']['num_parts']}", flush=True)
    print(f"[Setup] Hidden dim: {cfg['model']['hidden_dim']}, Layers: {cfg['model']['num_layers']}, Heads: {cfg['model']['num_heads']}", flush=True)

    train_loader, val_loader = build_dataloaders(
        cfg, max_train=args.max_train, max_val=args.max_val
    )

    # Auto-select V2 when config includes V2-specific keys
    mcfg = cfg["model"]
    if "num_groups" in mcfg or "pe_multiplier" in mcfg or "grid_cell_size" in mcfg:
        print("[Setup] Using Point Transformer V2 backbone", flush=True)
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
        print("[Setup] Using Point Transformer V1 backbone", flush=True)
        model = PointTransformerSeg(
            input_dim=int(mcfg.get("input_dim", 3)),
            hidden_dim=int(mcfg.get("hidden_dim", 128)),
            num_layers=int(mcfg.get("num_layers", 4)),
            num_heads=int(mcfg.get("num_heads", 4)),
            num_parts=int(cfg["dataset"]["num_parts"]),
            dropout=float(mcfg.get("dropout", 0.1)),
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Setup] Model parameters: {n_params:,}", flush=True)

    criterion = SegmentationLoss(ignore_index=int(cfg["loss"].get("ignore_index", -1)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["training"].get("epochs", 30)),
    )

    exp_dir = Path(cfg["experiment"].get("output_dir", "experiments/exp_pt_partnet_baseline"))
    ckpt_dir = exp_dir / "checkpoints"
    log_path = exp_dir / "train_log.txt"
    tb_dir = Path(cfg.get("logging", {}).get("tensorboard_dir", exp_dir / "tensorboard"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    best_miou = -1.0
    epochs = args.epochs if args.epochs > 0 else int(cfg["training"].get("epochs", 30))
    epoch_times: list[float] = []
    log_interval = int(cfg["training"].get("log_interval", 10))
    save_every = int(cfg["training"].get("save_every", 5))
    use_tensorboard = bool(cfg.get("logging", {}).get("tensorboard", True))

    writer = SummaryWriter(log_dir=str(tb_dir)) if use_tensorboard else None
    global_step = 0

    try:
        with log_path.open("w", encoding="utf-8") as log_f:
            for epoch in range(1, epochs + 1):
                t_epoch_start = time.time()
                model.train()
                train_losses = []
                for step, batch in enumerate(train_loader, start=1):
                    points = batch["points"].to(device)
                    labels = batch["labels"].to(device)

                    logits = model(points)
                    loss = criterion(logits, labels)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    train_losses.append(float(loss.item()))
                    if writer is not None:
                        writer.add_scalar("train/step_loss", float(loss.item()), global_step)
                        writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)
                    global_step += 1

                    if step % log_interval == 0:
                        msg = f"[Epoch {epoch:03d}] step={step:04d} loss={loss.item():.4f}"
                        print(msg, flush=True)
                        log_f.write(msg + "\n")
                        log_f.flush()

                model.eval()
                val_metrics = {"overall_acc": 0.0, "miou": 0.0}
                n_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        points = batch["points"].to(device)
                        labels = batch["labels"].to(device)
                        logits = model(points)
                        metrics = compute_segmentation_metrics(
                            logits,
                            labels,
                            num_parts=int(cfg["dataset"]["num_parts"]),
                            ignore_index=int(cfg["loss"].get("ignore_index", -1)),
                        )
                        val_metrics["overall_acc"] += metrics["overall_acc"]
                        val_metrics["miou"] += metrics["miou"]
                        n_batches += 1

                if n_batches > 0:
                    val_metrics = {k: v / n_batches for k, v in val_metrics.items()}

                scheduler.step()
                train_loss = sum(train_losses) / max(1, len(train_losses))
                elapsed = time.time() - t_epoch_start
                epoch_times.append(elapsed)
                avg_time = sum(epoch_times) / len(epoch_times)
                eta = avg_time * (epochs - epoch)
                summary = (
                    f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
                    f"val_acc={val_metrics['overall_acc']:.4f} val_miou={val_metrics['miou']:.4f} "
                    f"| {elapsed:.0f}s/epoch | ETA {eta/3600:.1f}h"
                )
                print(summary, flush=True)
                print(summary, file=sys.stderr)
                log_f.write(summary + "\n")
                log_f.flush()

                if writer is not None:
                    writer.add_scalar("train/epoch_loss", train_loss, epoch)
                    writer.add_scalar("val/overall_acc", val_metrics["overall_acc"], epoch)
                    writer.add_scalar("val/miou", val_metrics["miou"], epoch)

                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "val_miou": val_metrics["miou"],
                }

                if epoch % save_every == 0:
                    torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pth")

                if val_metrics["miou"] > best_miou:
                    best_miou = val_metrics["miou"]
                    torch.save(state, ckpt_dir / "best.pth")
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
