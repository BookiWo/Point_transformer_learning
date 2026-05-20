from __future__ import annotations

import torch
import torch.nn as nn


class SegmentationLoss(nn.Module):
    def __init__(self, ignore_index: int = -1) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, N, C], labels: [B, N]
        b, n, c = logits.shape
        logits = logits.reshape(b * n, c)
        labels = labels.reshape(b * n)
        return self.loss_fn(logits, labels)
