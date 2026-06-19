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

    teacher_seed: int = 1729
    student_seed: int = 314159
    train_seed: int = 271828
    eval_seed: int = 161803
    batch_seed: int = 141421


CONFIG = TaskConfig()

RANDOM_TEACHER_TRACK = "random_teacher"
SINGLE_AR_TRACK = "single_ar"
MQAR_TRACK = "mqar"
TRACKS = (RANDOM_TEACHER_TRACK, SINGLE_AR_TRACK, MQAR_TRACK)
DEFAULT_TRACK = RANDOM_TEACHER_TRACK
