from __future__ import annotations

import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_parts: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_parts),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)
