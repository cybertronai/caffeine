# caffeine: attention optimization problem

**Motivation:** most gradient optimizers are agnostic to values being optimized.
Muon has demonstrated that matrix-specific optimizer can outperform a fully agnostic one.
The goal of this task is to see whether we can create an even better optimizer by finding the best optimizer specifically for attention models.

Task: design an optimizer that trains a fixed vanilla self-attention model on a toy task, each track defines a unique task.

## tracks

| Track | Objective | Target |
| --- | --- | --- |
| `random_teacher` | Train a model to match the outputs of another randomly initialized hidden model (i.e. teacher). | final eval MSE |
| `single_ar` | Each example is a short list of key/value binding tokens followed by one query token. The model must use the query to find the matching binding and predict its value. | eval accuracy >= `0.99` |
| `mqar` | Each example has the same kind of key/value binding list, followed by several query tokens. The model must answer all lookups in the sequence, reusing the same learned recall mechanism. | eval accuracy >= `0.99` |

### `random_teacher`

This is the original caffeine task. The harness creates two single-head
self-attention models:

1. a frozen teacher with deterministic random weights
2. a student with deterministic random initialization

Random matrix sequences are passed through the teacher to create regression
targets. The submitted optimizer trains the student to match those teacher
outputs. This track is useful as a pure optimizer stress test because the target
function is fixed and generated entirely by another attention layer.

### `single_ar`

This is a small next-token associative-recall task. Each example contains eight
context tokens followed by one query token:

```text
[pair_1, pair_2, ..., pair_8, query_key] -> value
```

For example, if the hidden bindings are `A -> red`, `B -> blue`, and
`C -> green`, then a query for `B` should produce `blue`. In the actual
benchmark these symbols are opaque token IDs, not readable strings.

Internally, each `pair_i` represents a key/value binding, and `query_key` asks
for the value associated with one of those keys. Submissions are not given that
internal key/value decomposition. They only see gradients through the model's
token embeddings and readout weights.

The model must learn to attend from the query token to the matching context
token and classify the associated value.

### `mqar`

This is the multi-query version of associative recall. Each example contains the
same eight context pair tokens, but now has eight query tokens:

```text
[pair_1, ..., pair_8, query_1, ..., query_8] -> [value_1, ..., value_8]
```

For example, with hidden bindings `A -> red`, `B -> blue`, and `C -> green`,
the queries `[C, A, B]` should produce `[green, red, blue]`. The task tests
whether one attention layer can reuse the same lookup rule several times in a
single example.

The same learned binding mechanism has to answer several lookups in one
sequence. This track is inspired by the multi-query associative recall task from
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
