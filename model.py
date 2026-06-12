from __future__ import annotations

import torch
from torch import nn


class VanillaSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.attention(x, x, x, need_weights=False)
        return output
