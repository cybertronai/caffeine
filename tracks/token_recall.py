from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from task import MQAR_TRACK, SINGLE_AR_TRACK
from tracks.base import BenchmarkModel, RunConfig, TensorDataset, build_batch_indices, seeded_generator


DEFAULT_TOKEN_RECALL_RUN = RunConfig(
    train_samples=65536,
    eval_samples=4096,
    batch_size=256,
    max_steps=400,
    eval_every=50,
)


@dataclass(frozen=True)
class TokenRecallConfig:
    num_queries: int
    embed_dim: int = 64
    num_keys: int = 64
    num_values: int = 64
    num_pairs: int = 8
    target_accuracy: float = 0.99
    student_seed: int = 314159
    run: RunConfig = DEFAULT_TOKEN_RECALL_RUN

    @property
    def vocab_size(self) -> int:
        return self.num_keys * self.num_values + self.num_keys


class TokenAttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_values: int,
        num_pairs: int,
        num_queries: int,
    ):
        super().__init__()
        self.num_pairs = num_pairs
        self.num_queries = num_queries
        self.attention_scale = embed_dim**-0.5
        self.q = nn.Embedding(vocab_size, embed_dim)
        self.k = nn.Embedding(vocab_size, embed_dim)
        self.v = nn.Embedding(vocab_size, embed_dim)
        self.readout = nn.Linear(embed_dim, num_values)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        context_tokens = tokens[:, : self.num_pairs]
        query_tokens = tokens[:, self.num_pairs : self.num_pairs + self.num_queries]
        q = self.q(query_tokens)
        k = self.k(context_tokens)
        v = self.v(context_tokens)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        logits = self.readout(context)
        if self.num_queries == 1:
            return logits[:, 0, :]
        return logits


@dataclass(frozen=True)
class TokenRecallDatasetBuilder:
    config: TokenRecallConfig

    def build(self, *, samples: int, seed: int) -> TensorDataset:
        generator = seeded_generator(seed)
        keys = torch.empty((samples, self.config.num_pairs), dtype=torch.long)
        for row in range(samples):
            keys[row] = torch.randperm(self.config.num_keys, generator=generator)[: self.config.num_pairs]
        values = torch.randint(self.config.num_values, (samples, self.config.num_pairs), generator=generator)
        pair_tokens = keys * self.config.num_values + values
        rows = torch.arange(samples).unsqueeze(1)

        query_slots = torch.randint(self.config.num_pairs, (samples, self.config.num_queries), generator=generator)
        query_keys = keys[rows, query_slots]
        query_tokens = self._query_tokens(query_keys)
        targets = values[rows, query_slots]
        inputs = torch.cat([pair_tokens, query_tokens], dim=1)
        if self.config.num_queries == 1:
            targets = targets[:, 0]
        return TensorDataset(inputs=inputs, targets=targets)

    def _query_tokens(self, query_keys: torch.Tensor) -> torch.Tensor:
        return self.config.num_keys * self.config.num_values + query_keys


@dataclass(frozen=True)
class TokenRecallTrack:
    config: TokenRecallConfig
    name: str

    @property
    def run_config(self) -> RunConfig:
        return self.config.run

    @property
    def max_steps(self) -> int:
        return self.run_config.max_steps

    @property
    def eval_every(self) -> int:
        return self.run_config.eval_every

    def build_student(self) -> TokenAttentionClassifier:
        torch.manual_seed(self.config.student_seed)
        return TokenAttentionClassifier(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            num_values=self.config.num_values,
            num_pairs=self.config.num_pairs,
            num_queries=self.config.num_queries,
        )

    def build_train_dataset(self) -> TensorDataset:
        return self._dataset_builder().build(samples=self.run_config.train_samples, seed=self.run_config.train_seed)

    def build_eval_dataset(self) -> TensorDataset:
        return self._dataset_builder().build(samples=self.run_config.eval_samples, seed=self.run_config.eval_seed)

    def _dataset_builder(self) -> TokenRecallDatasetBuilder:
        return TokenRecallDatasetBuilder(self.config)

    def build_batch_indices(self) -> torch.Tensor:
        return build_batch_indices(self.run_config)

    def loss(self, model: BenchmarkModel, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return classification_loss(model(inputs), targets)

    @torch.no_grad()
    def evaluate(self, model: BenchmarkModel, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        model.eval()
        outputs = model(inputs)
        loss = classification_loss(outputs, targets)
        accuracy = classification_accuracy(outputs, targets)
        return {"loss": float(loss.item()), "mse": float("nan"), "accuracy": float(accuracy.item())}

    def metric_passed(self, metrics: dict[str, float]) -> bool:
        return metrics["accuracy"] >= self.config.target_accuracy

    def metric_score(self, metrics: dict[str, float]) -> float:
        return metrics["accuracy"]

    def update_best(self, best: dict[str, float], metrics: dict[str, float]) -> dict[str, float]:
        return {**best, "accuracy": max(best["accuracy"], metrics["accuracy"])}

    def target_metrics(self) -> dict[str, float]:
        return {"target_mse": float("nan"), "target_accuracy": self.config.target_accuracy}


def classification_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if outputs.ndim == 2:
        return F.cross_entropy(outputs, targets)
    return F.cross_entropy(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))


def classification_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred = outputs.argmax(dim=-1)
    return (pred == targets).float().mean()


def single_query_associative_recall_track() -> TokenRecallTrack:
    return TokenRecallTrack(config=TokenRecallConfig(num_queries=1), name=SINGLE_AR_TRACK)


def multi_query_associative_recall_track() -> TokenRecallTrack:
    return TokenRecallTrack(config=TokenRecallConfig(num_queries=8), name=MQAR_TRACK)
