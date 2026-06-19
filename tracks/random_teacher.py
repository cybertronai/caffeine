from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model import VanillaSelfAttention
from task import RANDOM_TEACHER_TRACK
from tracks.base import BenchmarkModel, RunConfig, TensorDataset, build_batch_indices, seeded_generator


DEFAULT_RANDOM_TEACHER_RUN = RunConfig(
    train_samples=8192,
    eval_samples=2048,
    batch_size=512,
    max_steps=400,
    eval_every=100,
)


@dataclass(frozen=True)
class RandomTeacherConfig:
    embed_dim: int = 128
    sequence_length: int = 64
    target_mse: float = 4.0e-7
    teacher_seed: int = 1729
    student_seed: int = 314159
    run: RunConfig = DEFAULT_RANDOM_TEACHER_RUN


def build_random_teacher(config: RandomTeacherConfig) -> VanillaSelfAttention:
    model = VanillaSelfAttention(config.embed_dim)
    generator = seeded_generator(config.teacher_seed)
    for parameter in model.parameters():
        parameter.data.uniform_(-1.0, 1.0, generator=generator)
        parameter.requires_grad_(False)
    model.eval()
    return model


@dataclass(frozen=True)
class RandomTeacherDatasetBuilder:
    config: RandomTeacherConfig
    teacher: VanillaSelfAttention

    def build(self, *, samples: int, seed: int) -> TensorDataset:
        inputs = torch.randn(
            (samples, self.config.sequence_length, self.config.embed_dim),
            generator=seeded_generator(seed),
        )
        with torch.no_grad():
            targets = self.teacher(inputs).detach().clone()
        return TensorDataset(inputs=inputs, targets=targets)


@dataclass(frozen=True)
class RandomTeacherTrack:
    config: RandomTeacherConfig = RandomTeacherConfig()
    name: str = RANDOM_TEACHER_TRACK

    @property
    def run_config(self) -> RunConfig:
        return self.config.run

    @property
    def max_steps(self) -> int:
        return self.run_config.max_steps

    @property
    def eval_every(self) -> int:
        return self.run_config.eval_every

    def build_student(self) -> VanillaSelfAttention:
        torch.manual_seed(self.config.student_seed)
        return VanillaSelfAttention(self.config.embed_dim)

    def build_train_dataset(self) -> TensorDataset:
        return self._dataset_builder().build(samples=self.run_config.train_samples, seed=self.run_config.train_seed)

    def build_eval_dataset(self) -> TensorDataset:
        return self._dataset_builder().build(samples=self.run_config.eval_samples, seed=self.run_config.eval_seed)

    def _dataset_builder(self) -> RandomTeacherDatasetBuilder:
        return RandomTeacherDatasetBuilder(self.config, build_random_teacher(self.config))

    def build_batch_indices(self) -> torch.Tensor:
        return build_batch_indices(self.run_config)

    def loss(self, model: BenchmarkModel, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(model(inputs), targets)

    @torch.no_grad()
    def evaluate(self, model: BenchmarkModel, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        model.eval()
        mse = float(F.mse_loss(model(inputs), targets).item())
        return {"loss": mse, "mse": mse, "accuracy": float("nan")}

    def metric_passed(self, metrics: dict[str, float]) -> bool:
        return metrics["mse"] <= self.config.target_mse

    def metric_score(self, metrics: dict[str, float]) -> float:
        return metrics["mse"]

    def update_best(self, best: dict[str, float], metrics: dict[str, float]) -> dict[str, float]:
        return {**best, "mse": min(best["mse"], metrics["mse"])}

    def target_metrics(self) -> dict[str, float]:
        return {"target_mse": self.config.target_mse, "target_accuracy": float("nan")}
