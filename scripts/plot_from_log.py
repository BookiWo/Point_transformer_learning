"""Parse train_log.txt and generate visualization plots."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def parse_log(log_path: str) -> list[dict]:
    """Extract epoch-level metrics from train_log.txt."""
    history = []
    with open(log_path) as f:
        for line in f:
            # Match epoch summary: [Epoch XXX] train_loss=... val_acc=... val_miou=... | XXs/epoch
            m = re.match(
                r"\[Epoch (\d+)\]\s+train_loss=([\d.]+)\s+val_acc=([\d.]+)\s+val_miou=([\d.]+)\s*\|\s*([\d.]+)s",
                line,
            )
            if m:
                history.append({
                    "epoch": int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "val_acc": float(m.group(3)),
                    "val_miou": float(m.group(4)),
                    "time_s": float(m.group(5)),
                })
    return history


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


def plot_dashboard(history: list[dict], out_path: Path, title_note: str) -> None:
    set_style()
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_miou = [h["val_miou"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ── Loss ──
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, "o-", color="#d62728", markersize=3, linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.grid(True, alpha=0.3); ax.set_xlim(1, len(epochs))

    # ── mIoU ──
    ax = axes[0, 1]
    ax.plot(epochs, val_miou, "s-", color="#2ca02c", markersize=3, linewidth=1.5)
    i_best = np.argmax(val_miou)
    ax.annotate(f"Best: {val_miou[i_best]:.4f}", xy=(epochs[i_best], val_miou[i_best]),
                xytext=(epochs[i_best] + 1.5, val_miou[i_best] - 0.02),
                arrowprops=dict(arrowstyle="->", color="#2ca02c"), fontsize=9, color="#2ca02c")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mIoU")
    ax.set_title(f"Validation mIoU  (Best: {val_miou[i_best]:.4f} @ Epoch {epochs[i_best]})")
    ax.grid(True, alpha=0.3); ax.set_xlim(1, len(epochs))

    # ── Accuracy ──
    ax = axes[1, 0]
    ax.plot(epochs, val_acc, "D-", color="#1f77b4", markersize=3, linewidth=1.5)
    i_best_a = np.argmax(val_acc)
    ax.annotate(f"Best: {val_acc[i_best_a]:.4f}", xy=(epochs[i_best_a], val_acc[i_best_a]),
                xytext=(epochs[i_best_a] + 1.5, val_acc[i_best_a] - 0.03),
                arrowprops=dict(arrowstyle="->", color="#1f77b4"), fontsize=9, color="#1f77b4")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Validation Accuracy  (Best: {val_acc[i_best_a]:.4f} @ Epoch {epochs[i_best_a]})")
    ax.grid(True, alpha=0.3); ax.set_xlim(1, len(epochs))

    # ── Time ──
    ax = axes[1, 1]
    times_min = [h["time_s"] / 60 for h in history]
    ax.bar(epochs, times_min, color="#9467bd", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Minutes")
    ax.set_title("Epoch Duration"); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Point Transformer — PartNet 50-class Segmentation\n{title_note}", fontsize=10, color="#555")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Dashboard → {out_path}")


def plot_combined(history: list[dict], out_path: Path, title_note: str) -> None:
    set_style()
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_miou = [h["val_miou"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, ax1 = plt.subplots(figsize=(13, 7))

    c_loss, c_miou, c_acc = "#d62728", "#2ca02c", "#1f77b4"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=c_loss)
    l1 = ax1.plot(epochs, train_loss, "o-", color=c_loss, markersize=4, linewidth=1.5, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=c_loss)
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Metric Value", color="#333")
    l2 = ax2.plot(epochs, val_miou, "s-", color=c_miou, markersize=4, linewidth=1.5, label="Val mIoU")
    l3 = ax2.plot(epochs, val_acc, "D-", color=c_acc, markersize=4, linewidth=1.5, label="Val Acc")

    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=10)

    i_best = np.argmax(val_miou)
    ax2.axvline(x=epochs[i_best], color=c_miou, linestyle="--", alpha=0.4)
    ax1.text(epochs[i_best] + 0.5, train_loss[i_best],
             f"Best mIoU\n{val_miou[i_best]:.4f}", fontsize=9, color=c_miou, va="center")

    ax1.set_title(
        f"Point Transformer — PartNet 50-class Segmentation\n"
        f"Best mIoU = {val_miou[i_best]:.4f}  |  Best Acc = {max(val_acc):.4f}  |  {title_note}",
        fontweight="bold",
    )
    ax1.set_xlim(0.5, len(epochs) + 0.5)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Combined → {out_path}")


def plot_smoothed(history: list[dict], out_path: Path, title_note: str) -> None:
    """Triple y-axis plot with exponential moving average smoothing."""
    set_style()
    epochs = np.array([h["epoch"] for h in history])
    train_loss = np.array([h["train_loss"] for h in history])
    val_miou = np.array([h["val_miou"] for h in history])
    val_acc = np.array([h["val_acc"] for h in history])

    def ema(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        s = np.zeros_like(data)
        s[0] = data[0]
        for i in range(1, len(data)):
            s[i] = alpha * data[i] + (1 - alpha) * s[i - 1]
        return s

    fig, ax1 = plt.subplots(figsize=(14, 7))
    c_loss, c_miou, c_acc = "#d62728", "#2ca02c", "#1f77b4"

    # Raw loss (faint) + smoothed (bold)
    ax1.plot(epochs, train_loss, "-", color=c_loss, alpha=0.2, linewidth=1)
    ax1.plot(epochs, ema(train_loss, 0.3), "-", color=c_loss, linewidth=2.5, label="Train Loss (smoothed)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color=c_loss, fontsize=13)
    ax1.tick_params(axis="y", labelcolor=c_loss)
    ax1.grid(True, alpha=0.15)

    # mIoU
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_miou, "s-", color=c_miou, markersize=5, linewidth=2, label="Val mIoU")
    ax2.set_ylabel("mIoU", color=c_miou, fontsize=13)
    ax2.tick_params(axis="y", labelcolor=c_miou)

    # Acc
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.08))
    ax3.plot(epochs, val_acc, "D-", color=c_acc, markersize=4, linewidth=1.5, label="Val Acc")
    ax3.set_ylabel("Accuracy", color=c_acc, fontsize=13)
    ax3.tick_params(axis="y", labelcolor=c_acc)

    # Unified legend
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color=c_loss, linewidth=2.5, label="Train Loss (EMA)"),
        Line2D([0], [0], color=c_miou, marker="s", linewidth=2, label="Val mIoU"),
        Line2D([0], [0], color=c_acc, marker="D", linewidth=1.5, label="Val Acc"),
    ]
    ax1.legend(handles=legend_lines, loc="upper center", fontsize=10, ncol=3,
               bbox_to_anchor=(0.5, -0.12))

    i_best = np.argmax(val_miou)
    ax2.axvline(x=epochs[i_best], color=c_miou, linestyle="--", alpha=0.3, linewidth=1.5)
    ax1.set_title(
        f"Point Transformer — PartNet 50-class Segmentation\n"
        f"Best mIoU = {val_miou[i_best]:.4f} (Epoch {epochs[i_best]})  |  "
        f"Final Loss = {train_loss[-1]:.4f}  |  {title_note}",
        fontweight="bold", fontsize=13,
    )
    ax1.set_xlim(0.5, len(epochs) + 0.5)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Smoothed → {out_path}")


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "/home/laplace37/autodl_results/train_log.txt"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / "outputs" / "viz"

    if not Path(log_path).exists():
        print(f"ERROR: {log_path} not found", file=sys.stderr)
        sys.exit(1)

    history = parse_log(log_path)
    if not history:
        print("ERROR: No epoch summaries found in log", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(history)} epochs")

    # Extract config info from log
    title = "RTX 5090 32GB | 1024 pts | hidden_dim=192 | layers=8 | heads=8 | batch=8 | full 12k"

    # Best results
    best = max(history, key=lambda h: h["val_miou"])
    latest = history[-1]
    total_h = sum(h["time_s"] for h in history) / 3600
    print(f"\nTraining Summary:")
    print(f"  Epochs:      {len(history)}")
    print(f"  Best  mIoU:  {best['val_miou']:.4f}  (epoch {best['epoch']})")
    print(f"  Best  Acc:   {max(h['val_acc'] for h in history):.4f}")
    print(f"  Final mIoU:  {latest['val_miou']:.4f}")
    print(f"  Final Loss:  {latest['train_loss']:.4f}")
    print(f"  Total time:  {total_h:.1f}h")

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_dashboard(history, out_dir / "training_dashboard.png", title)
    plot_combined(history, out_dir / "training_combined.png", title)
    plot_smoothed(history, out_dir / "training_smoothed.png", title)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
