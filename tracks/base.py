from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class TensorDataset:
    inputs: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True)
class RunConfig:
    train_samples: int
    eval_samples: int
    batch_size: int
    max_steps: int
    eval_every: int
    train_seed: int = 271828
    eval_seed: int = 161803
    batch_seed: int = 141421


class BenchmarkModel(Protocol):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def train(self, mode: bool = True) -> Any: ...

    def eval(self) -> Any: ...

    def to(self, device: torch.device) -> Any: ...

    def parameters(self, recurse: bool = True) -> Any: ...


class DatasetBuilder(Protocol):
    def build(self, *, samples: int, seed: int) -> TensorDataset: ...


class BenchmarkTrack(Protocol):
    name: str

    @property
    def run_config(self) -> RunConfig: ...

    @property
    def max_steps(self) -> int: ...

    @property
    def eval_every(self) -> int: ...

    def build_student(self) -> BenchmarkModel: ...

    def build_train_dataset(self) -> TensorDataset: ...

    def build_eval_dataset(self) -> TensorDataset: ...

    def build_batch_indices(self) -> torch.Tensor: ...

    def loss(self, model: BenchmarkModel, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...

    def evaluate(self, model: BenchmarkModel, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]: ...

    def metric_passed(self, metrics: dict[str, float]) -> bool: ...

    def metric_score(self, metrics: dict[str, float]) -> float: ...

    def update_best(self, best: dict[str, float], metrics: dict[str, float]) -> dict[str, float]: ...

    def target_metrics(self) -> dict[str, float]: ...


def seeded_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def build_batch_indices(config: RunConfig) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=config.train_samples,
        size=(config.max_steps, config.batch_size),
        generator=seeded_generator(config.batch_seed),
    )
