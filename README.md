# caffeine: attention optimization problem

**Motivation:** most gradient optimizers are agnostic to values being optimized.
Muon has demonstrated that matrix-specific optimizer can outperform a fully agnostic one.
The goal of this task is to see whether we can create an even better optimizer by finding the best optimizer specifically for attention models.

Task: design an optimizer that trains a fixed vanilla self-attention model on a toy task, each track defines a unique task.

## tracks

| Track | Objective | Target |
| --- | --- | --- |
| `random_teacher` | Train a model to match the outputs of another randomly initialized hidden model (i.e. teacher). | final eval MSE |
| `single_ar` | Opaque single-query associative recall: `8` pair tokens + `1` query token -> value class. | eval accuracy >= `0.99` |
| `mqar` | Opaque multi-query associative recall: `8` pair tokens + `8` query tokens -> value classes. | eval accuracy >= `0.99` |

`mqar` is inspired by the multi-query associative recall task from
[Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927).

The benchmark supports multiple tracks through `run_eval.py --track`:

```bash
uv run python run_eval.py --track single_ar --submission submissions/adamw/submission.py
uv run python run_eval.py --track mqar --submission submissions/adamw/submission.py
```

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

| Rank | Submission | Author | Final Eval MSE | Training Wall Time (s) |
|---:|---|---|---:|---:|
| 1 | `adaptive_cm` | [@SethTS](https://github.com/SethTS) | `2.34818` | `30.000` |
| 2 | `adamw` | [@ab-10](https://github.com/ab-10) | `58.3003` | `24.359` |
| 3 | `comp_muon` | [@SethTS](https://github.com/SethTS) | `1266.47` | `31.601` |

_Evaluated on macos-15-arm64 GH Runner._

## run locally

```bash
uv sync
uv run python test_benchmark.py
uv run python run_eval.py --submission submissions/adamw/submission.py --results-json result.json
```
