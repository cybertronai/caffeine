# caffeine: attention optimization problem

**Motivation:** most gradient optimizers are agnostic to values being optimized.
Muon has demonstrated that matrix-specific optimizer can outperform a fully agnostic one.
The goal of this task is to see whether we can create an even better optimizer by finding the best optimizer specifically for attention models.

Task: design an optimizer that trains a fixed vanilla self-attention model to match a deterministic teacher from input/output samples.


## fixed values

1. dataset and train/val/eval splits
2. model architecture

## submission contract

1. A submission is a Python file defining `Submission`, a subclass of `torch.optim.Optimizer`.
2. The harness instantiates it strictly as `Submission(model.parameters())`.
3. The harness calls `optimizer.step()` without a closure; optimizers must update parameters using the `.grad` fields populated by the harness.

## scoring

Official v0 scoring is wall-clock time on GitHub Actions `macos-15` arm64. The harness uses MPS when available and falls back to CPU. It stops at the first fixed evaluation checkpoint where parameter MSE against the deterministic teacher is at or below `target_weight_mse`; held-out output MSE is reported as a sanity metric, and submissions that miss the weight target within `max_steps` fail.

Run locally:

```bash
uv sync
uv run python test_benchmark.py
uv run python run_eval.py --submission submissions/adamw/submission.py --results-json result.json
```

## dataset

The harness deterministically initializes a teacher self-attention model, a student self-attention model, train inputs, eval inputs, and a fixed stochastic batch order from public seeds in `task.py`. Train and eval targets are teacher outputs on those inputs.

## model architecture

Model: single-head vanilla self-attention via `torch.nn.MultiheadAttention`.
