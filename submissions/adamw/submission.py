from __future__ import annotations

import torch


class Submission(torch.optim.AdamW):
    def __init__(self, params):
        super().__init__(params, lr=1.0e-2, betas=(0.9, 0.99), weight_decay=0.0)
