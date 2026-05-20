from __future__ import annotations

import torch


def compute_segmentation_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_parts: int,
    ignore_index: int = -1,
) -> dict[str, float]:
    preds = torch.argmax(logits, dim=-1)

    valid_mask = labels != ignore_index
    if not torch.any(valid_mask):
        return {"overall_acc": 0.0, "miou": 0.0}

    valid_labels = labels[valid_mask]
    valid_preds = preds[valid_mask]
    overall_acc = float((valid_preds == valid_labels).float().mean().item())

    ious = []
    for cls in range(num_parts):
        pred_mask = valid_preds == cls
        label_mask = valid_labels == cls
        union = torch.logical_or(pred_mask, label_mask).sum().item()
        if union == 0:
            continue
        inter = torch.logical_and(pred_mask, label_mask).sum().item()
        ious.append(inter / union)

    miou = float(sum(ious) / len(ious)) if ious else 0.0
    return {"overall_acc": overall_acc, "miou": miou}
