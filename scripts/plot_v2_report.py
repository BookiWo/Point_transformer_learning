#!/usr/bin/env python3
"""Generate training comparison plots for the V2 experiment report."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Parse helpers ──────────────────────────────────────────────────────────

def parse_log(path: str) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (epoch_losses, epoch_accs, epoch_mious, epoch_times) from a training log."""
    losses, accs, mious, times = [], [], [], []
    pat = re.compile(
        r"train_loss=([\d.]+)\s+val_acc=([\d.]+)\s+val_miou=([\d.]+)\s+\|\s+([\d.]+)s/epoch"
    )
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                losses.append(float(m.group(1)))
                accs.append(float(m.group(2)))
                mious.append(float(m.group(3)))
                times.append(float(m.group(4)))
    return losses, accs, mious, times


# ── Data ────────────────────────────────────────────────────────────────────

DATA = {
    "V1 256/8": "/home/laplace37/autodl_results/train_log.txt",
    "V2 128/4": "/tmp/results/backup/train.log",
    "V2 256/8": "/mnt/d/sysu.courses/大二/拔尖计划/isee_robotic/point_transformer_partnet/experiments/exp_pt_partnet_v2_full/train_log.txt",
}

COLORS = {
    "V1 256/8": "#999999",
    "V2 128/4": "#2ca02c",
    "V2 256/8": "#1f77b4",
}

LINESTYLES = {
    "V1 256/8": ":",
    "V2 128/4": "-",
    "V2 256/8": "-",
}

results = {}
for name, path in DATA.items():
    results[name] = parse_log(path)

# ── Plot ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Point Transformer V2 — PartNet Segmentation Training Report", fontsize=15, fontweight="bold", y=0.98)

# ── 1. Loss ─────────────────────────────────────────────────────────────────
ax = axes[0, 0]
for name, (loss, acc, miou, _times) in results.items():
    ax.plot(range(1, len(loss) + 1), loss, color=COLORS[name], ls=LINESTYLES[name], lw=2, label=name)
ax.set_ylabel("Train Loss")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(max(v[0][:5]) if v[0] else 1 for v in results.values()) * 1.05)

# ── 2. Val Accuracy ─────────────────────────────────────────────────────────
ax = axes[0, 1]
for name, (loss, acc, miou, _times) in results.items():
    ax.plot(range(1, len(acc) + 1), [a * 100 for a in acc], color=COLORS[name], ls=LINESTYLES[name], lw=2, label=name)
ax.set_ylabel("Val Accuracy (%)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── 3. Val mIoU ─────────────────────────────────────────────────────────────
ax = axes[0, 2]
for name, (loss, acc, miou, _times) in results.items():
    ax.plot(range(1, len(miou) + 1), [m * 100 for m in miou], color=COLORS[name], ls=LINESTYLES[name], lw=2, label=name)
ax.set_ylabel("Val mIoU (%)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── 4. Combined view ────────────────────────────────────────────────────────
ax = axes[1, 0]
for name, (loss, acc, miou, _times) in results.items():
    ax.plot(range(1, len(miou) + 1), [m * 100 for m in miou], color=COLORS[name], ls=LINESTYLES[name], lw=2.5, label=f"{name}: mIoU")
ax.set_xlabel("Epoch")
ax.set_ylabel("Val mIoU (%)")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
# Annotate final values
for name, (loss, acc, miou, _times) in results.items():
    if miou:
        ax.annotate(f"{miou[-1]*100:.1f}%", xy=(len(miou), miou[-1]*100),
                     xytext=(5, 5), textcoords="offset points", fontsize=9, color=COLORS[name], fontweight="bold")

# ── 5. Epoch time ───────────────────────────────────────────────────────────
ax = axes[1, 1]
for name, (loss, acc, miou, times) in results.items():
    ax.plot(range(1, len(times) + 1), times, color=COLORS[name], ls=LINESTYLES[name], lw=2, label=name)
ax.set_xlabel("Epoch")
ax.set_ylabel("Seconds / Epoch")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── 6. Summary table ────────────────────────────────────────────────────────
ax = axes[1, 2]
ax.axis("off")

headers = ["", "V1 256/8", "V2 128/4", "V2 256/8"]
rows = [
    ("Params", "~110M", "26.1M", "208.6M"),
    ("hidden_dim", "256", "128", "256"),
    ("num_layers", "8", "4", "8"),
    ("PE Mul.", "N/A (V1)", "True", "False"),
    ("Grid Pool", "No (FPS)", "Yes", "Yes"),
    ("Batch", "24", "8", "4+GA×6=24"),
    ("Train Loss", f"{results['V1 256/8'][0][-1]:.4f}", f"{results['V2 128/4'][0][-1]:.4f}", f"{results['V2 256/8'][0][-1]:.4f}"),
    ("Val Acc", f"{results['V1 256/8'][1][-1]*100:.1f}%", f"{results['V2 128/4'][1][-1]*100:.1f}%", f"{results['V2 256/8'][1][-1]*100:.1f}%"),
    ("Val mIoU", f"{results['V1 256/8'][2][-1]*100:.1f}%", f"{results['V2 128/4'][2][-1]*100:.1f}%", f"{results['V2 256/8'][2][-1]*100:.1f}%"),
    ("Epoch Time", f"{results['V1 256/8'][3][-1]:.0f}s", f"{results['V2 128/4'][3][-1]:.0f}s", f"{results['V2 256/8'][3][-1]:.0f}s"),
]

for j, h in enumerate(headers):
    ax.text(0.15 * j, 0.95, h, transform=ax.transAxes, fontweight="bold", fontsize=9, ha="center")
for i, (label, *vals) in enumerate(rows):
    y = 0.85 - i * 0.08
    ax.text(0.0, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="center")
    for j, v in enumerate(vals):
        ax.text(0.15 * (j + 1), y, v, transform=ax.transAxes, fontsize=8, ha="center", va="center",
                color=COLORS.get(headers[j + 1], "black"))

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = Path(__file__).resolve().parent.parent / "outputs/viz/v2_report/training_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
