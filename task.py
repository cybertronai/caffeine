from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    embed_dim: int = 128
    sequence_length: int = 64
    train_samples: int = 8192
    eval_samples: int = 2048
    batch_size: int = 512
    max_steps: int = 400
    eval_every: int = 100
    target_mse: float = 4.0e-7

    teacher_seed: int = 1729
    student_seed: int = 314159
    train_seed: int = 271828
    eval_seed: int = 161803
    batch_seed: int = 141421


CONFIG = TaskConfig()
