from __future__ import annotations

from dataclasses import dataclass

import torch

from model import VanillaSelfAttention
from task import TaskConfig


@dataclass(frozen=True)
class TensorDataset:
    inputs: torch.Tensor
    targets: torch.Tensor


def _randn(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator)


def build_model(seed: int, config: TaskConfig) -> VanillaSelfAttention:
    torch.manual_seed(seed)
    return VanillaSelfAttention(config.embed_dim)


def initialize_teacher(model: VanillaSelfAttention, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for parameter in model.parameters():
        parameter.data.uniform_(-1.0, 1.0, generator=generator)


def build_teacher(config: TaskConfig) -> VanillaSelfAttention:
    model = VanillaSelfAttention(config.embed_dim)
    initialize_teacher(model, config.teacher_seed)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_student(config: TaskConfig) -> VanillaSelfAttention:
    return build_model(config.student_seed, config)


def build_dataset(
    teacher: VanillaSelfAttention,
    *,
    samples: int,
    seed: int,
    config: TaskConfig,
) -> TensorDataset:
    inputs = _randn((samples, config.sequence_length, config.embed_dim), seed)
    with torch.no_grad():
        targets = teacher(inputs).detach().clone()
    return TensorDataset(inputs=inputs, targets=targets)


def build_train_dataset(teacher: VanillaSelfAttention, config: TaskConfig) -> TensorDataset:
    return build_dataset(
        teacher,
        samples=config.train_samples,
        seed=config.train_seed,
        config=config,
    )


def build_eval_dataset(teacher: VanillaSelfAttention, config: TaskConfig) -> TensorDataset:
    return build_dataset(
        teacher,
        samples=config.eval_samples,
        seed=config.eval_seed,
        config=config,
    )


def build_batch_indices(config: TaskConfig) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(config.batch_seed)
    return torch.randint(
        low=0,
        high=config.train_samples,
        size=(config.max_steps, config.batch_size),
        generator=generator,
    )
