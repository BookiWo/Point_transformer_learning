"""Generate training metrics visualization plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.use("Agg")  # non-interactive backend

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_metrics(json_path: Path) -> list[dict]:
    with json_path.open() as f:
        return json.load(f)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def plot_dashboard(history: list[dict], out_dir: Path) -> None:
    set_style()
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_miou = [h["val_miou"] for h in history]
    lr = [h["lr"] for h in history]

    # ── Dashboard: 2x2 grid ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, "o-", color="#d62728", markersize=4, linewidth=1.5, label="Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))

    # --- mIoU ---
    ax = axes[0, 1]
    ax.plot(epochs, val_miou, "s-", color="#2ca02c", markersize=4, linewidth=1.5, label="Val mIoU")
    best_epoch = np.argmax(val_miou)
    best_val = val_miou[best_epoch]
    ax.annotate(
        f"Best: {best_val:.4f}",
        xy=(epochs[best_epoch], best_val),
        xytext=(epochs[best_epoch] + 1, best_val + 0.01),
        arrowprops=dict(arrowstyle="->", color="#2ca02c"),
        fontsize=9,
        color="#2ca02c",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.set_title(f"Validation mIoU (Best: {best_val:.4f} @ Epoch {epochs[best_epoch]})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))

    # --- Accuracy ---
    ax = axes[1, 0]
    ax.plot(epochs, val_acc, "D-", color="#1f77b4", markersize=4, linewidth=1.5, label="Val Acc")
    best_acc_epoch = np.argmax(val_acc)
    best_acc = val_acc[best_acc_epoch]
    ax.annotate(
        f"Best: {best_acc:.4f}",
        xy=(epochs[best_acc_epoch], best_acc),
        xytext=(epochs[best_acc_epoch] + 1, best_acc + 0.01),
        arrowprops=dict(arrowstyle="->", color="#1f77b4"),
        fontsize=9,
        color="#1f77b4",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Validation Accuracy (Best: {best_acc:.4f} @ Epoch {epochs[best_acc_epoch]})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))

    # --- LR ---
    ax = axes[1, 1]
    ax.plot(epochs, lr, "-", color="#9467bd", linewidth=1.5, label="Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_title("Learning Rate Schedule (Cosine Annealing)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))

    fig.suptitle(
        "Point Transformer — PartNet Segmentation (Quick Train)\n"
        f"hidden_dim=64, num_points=512, 2000 samples, 30 epochs",
        fontsize=10,
        color="#666666",
    )
    plt.tight_layout()
    dashboard_path = out_dir / "training_dashboard.png"
    fig.savefig(dashboard_path)
    plt.close(fig)
    print(f"Saved dashboard to {dashboard_path}", file=sys.stderr)


def plot_combined(history: list[dict], out_dir: Path) -> None:
    """Single combined plot with dual y-axes."""
    set_style()
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_miou = [h["val_miou"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_loss = "#d62728"
    color_miou = "#2ca02c"
    color_acc = "#1f77b4"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color_loss)
    l1 = ax1.plot(epochs, train_loss, "o-", color=color_loss, markersize=4, linewidth=1.5, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_ylim(0, max(train_loss) * 1.15)
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Metric Value", color="#333333")
    l2 = ax2.plot(epochs, val_miou, "s-", color=color_miou, markersize=4, linewidth=1.5, label="Val mIoU")
    l3 = ax2.plot(epochs, val_acc, "D-", color=color_acc, markersize=4, linewidth=1.5, label="Val Acc")
    ax2.set_ylim(0, max(max(val_miou), max(val_acc)) * 1.2)

    # Combined legend
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=10)

    best_epoch = np.argmax(val_miou)
    ax2.axvline(x=epochs[best_epoch], color=color_miou, linestyle="--", alpha=0.4, linewidth=1)
    ax1.text(
        epochs[best_epoch] + 0.3,
        train_loss[best_epoch],
        f"Best mIoU\n{val_miou[best_epoch]:.4f}",
        fontsize=9,
        color=color_miou,
        verticalalignment="center",
    )

    ax1.set_title(
        f"Point Transformer — PartNet Segmentation Training\n"
        f"Best mIoU={val_miou[best_epoch]:.4f}, Best Acc={max(val_acc):.4f}",
        fontweight="bold",
    )
    ax1.set_xlim(0.5, len(epochs) + 0.5)

    plt.tight_layout()
    combined_path = out_dir / "training_combined.png"
    fig.savefig(combined_path)
    plt.close(fig)
    print(f"Saved combined plot to {combined_path}", file=sys.stderr)


def main() -> None:
    metrics_path = REPO_ROOT / "experiments" / "quick_train" / "metrics.json"
    out_dir = metrics_path.parent

    if not metrics_path.exists():
        print(f"ERROR: metrics.json not found at {metrics_path}", file=sys.stderr)
        sys.exit(1)

    history = load_metrics(metrics_path)
    print(f"Loaded {len(history)} epochs of metrics", file=sys.stderr)

    plot_dashboard(history, out_dir)
    plot_combined(history, out_dir)

    # Print summary
    best = max(history, key=lambda h: h["val_miou"])
    latest = history[-1]
    print(
        f"\nTraining Summary:\n"
        f"  Best mIoU:   {best['val_miou']:.4f}  (epoch {best['epoch']})\n"
        f"  Final mIoU:  {latest['val_miou']:.4f}  (epoch {latest['epoch']})\n"
        f"  Final Acc:   {latest['val_acc']:.4f}\n"
        f"  Final Loss:  {latest['train_loss']:.4f}\n"
        f"  Epochs:      {len(history)}\n"
        f"  Total time:  {sum(h['time_s'] for h in history):.0f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
