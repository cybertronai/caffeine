# CM + Gauge Balance + Tail Schedule (`cm_tail_ema`)

`cm_tail_ema` combines the `adaptive_cm` update with a one-time,
function-preserving attention gauge balance and a tuned training tail. Its
official `random_teacher` MSE at step 400 is **0.0109174000** on macOS 15
arm64/MPS.

## Optimizer

The base update is partner-whitened matrix sign for the Q, K, V, and O weight
blocks, with λ=1.0 Gram regularization and heavy-ball momentum. Biases use
Adam. A shifted lognormal learning-rate schedule peaks at step 40.

The tail adds five mechanisms:

1. An LR floor of 0.002 keeps the final updates from vanishing.
2. At step 270, Q/K and V/O are refactored into balanced SVD gauges.
3. Q/K updates ramp from 1× to 1.75× between steps 300 and 400.
4. Gram regularization ramps from 1.0 to 0.01 over the final 100 steps.
5. An EMA starts at step 325 and is copied into the model at step 400.

The numerical settings came from exploratory deterministic CPU and MPS
sweeps. The quality claim below uses only the frozen, audited official runs.

## Function-preserving gauge balance

For PyTorch's weight layout, attention logits depend on the Q/K product
`W_Q.T @ W_K`. The optimizer replaces its factors with

```text
W_Q_new = sqrt(S) @ U.T
W_K_new = sqrt(S) @ Vh
```

where `U @ S @ Vh = W_Q.T @ W_K`. It also solves for a new query bias that
preserves `b_q @ W_K`. The key bias is set to zero because its contribution is
a row-wise softmax constant.

The value/output path depends on `W_V.T @ W_O.T` and is balanced in the same
way. The value-bias contribution is absorbed into the output bias before the
value bias is zeroed. The regression test requires output MSE below `1e-11`
across the refactor.

All matrix momentum, bias moments, and Adam bias-correction counters are reset
after the coordinate change. An earlier PR revision reset the moment tensors
but accidentally retained the Adam counters; its result is superseded by the
audited result below.

## Audited official result

Both rows used Python 3.12.10, PyTorch 2.12.0, macOS 15.7.7 arm64/MPS, 8,192
training samples, 2,048 evaluation samples, 400 batches of 512, and identical
SHA-256 hashes for every input, target, and batch-index tensor.

| Package | Final eval MSE | Training time | Official run |
|---|---:|---:|---|
| Pre-gauge tail package (`qk_tail=1.25`) | 0.0247666091 | 36.72 s | [baseline audit](https://github.com/npow/caffeine/actions/runs/33454278926) |
| Gauge-balanced package (`qk_tail=0.75`) | **0.0109174000** | 35.73 s | [current artifact](https://github.com/npow/caffeine/actions/runs/33454022031) |

This is a **55.92% MSE reduction** for the complete package change. It should
not be interpreted as an isolated estimate of the gauge operation: the final
package also retunes the Q/K tail and correctly restarts all transformed
optimizer state.

The corrected optimizer independently reproduced `0.010917400009930134` in
the automatic PR run and a [manual workflow run](https://github.com/npow/caffeine/actions/runs/33453755531).
The strict JSON artifact from the automatic run is committed as
`result.macos-arm64.json`, including shapes, denominators, and full tensor
hashes.

The artifact's `status` is `fail` because the benchmark's absolute target is
`4e-7`; leaderboard ranking is by lower final MSE, so this remains a valid
quality improvement without being a target pass.

Teacher-generated target hashes can vary across numerical platforms. Results
should therefore be compared only when the audit hashes match; the two
official rows above do.

## Reproduce

```bash
uv run python run_eval.py --submission submissions/cm_tail_ema/submission.py \
    --results-json submissions/cm_tail_ema/result.json
```
