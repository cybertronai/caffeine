# caffeine: attention optimization problem

**Motivation:** most gradient optimizers are agnostic to values being optimized.
Muon has demonstrated that matrix-specific optimizer can outperform a fully agnostic one.
The goal of this task is to see whether we can create an even better optimizer by finding the best optimizer specifically for attention models.

Task: design an optimizer that trains a fixed vanilla self-attention model to match a deterministic teacher from input/output samples.


## fixed values

1. dataset and train/val/eval splits
2. model architecture
3. training step budget

## submission contract

1. A submission is a Python file defining `Submission`, a subclass of `torch.optim.Optimizer`.
2. The harness instantiates it strictly as `Submission(model.parameters())`.
3. The harness calls `optimizer.step()` without a closure; optimizers must update parameters using the `.grad` fields populated by the harness.

## scoring

Official v0 scoring runs every submission for the fixed `max_steps` budget in `task.py`, then ranks by held-out final eval MSE. Training-only wall time is reported as a secondary metric. Official leaderboard entries should be measured with the GitHub Actions `benchmark` workflow on `macos-15` arm64.

The leaderboard is ranked by:

1. lower `final_eval_mse`
2. lower `training_wall_time_s` as a tiebreaker

## leaderboard

Current fixed-step results, ranked by final eval MSE:

| Rank | Submission | Author | Final Eval MSE | Training Wall Time (s) |
|---:|---|---|---:|---:|
| 1 | `adaptive_cm` | [@SethTS](https://github.com/SethTS) | `2.29481` | `8.989` |
| 2 | `adamw` | [@cybertronai](https://github.com/cybertronai) | `58.3003` | `3.980` |
| 3 | `comp_muon` | [@SethTS](https://github.com/SethTS) | `1266.47` | `9.557` |

Use the manual `seed leaderboard` workflow to evaluate all existing submissions independently when seeding this table. Future entries should come from the individual submission's `benchmark` workflow run.

Run locally:

```bash
uv sync
uv run python test_benchmark.py
uv run python run_eval.py --submission submissions/adamw/submission.py --results-json result.json
uv run python run_leaderboard.py --output-dir artifacts
```

## dataset

The harness deterministically initializes a teacher self-attention model, a student self-attention model, train inputs, eval inputs, and a fixed stochastic batch order from public seeds in `task.py`. Train and eval targets are teacher outputs on those inputs. The default task uses `8192` train samples, `2048` eval samples, batch size `512`, and `400` training steps.

## model architecture

Model: single-head vanilla self-attention via `torch.nn.MultiheadAttention`.
