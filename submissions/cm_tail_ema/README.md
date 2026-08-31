# CM + Gauge Balance + Tail Schedule (`cm_tail_ema`)

**Submission: `cm_tail_ema`** — the adaptive-CM core, a one-time
function-preserving attention gauge balance, and a retuned training tail.
Official `random_teacher` MSE@400: **0.0112375** on macOS 15 arm64/MPS.

---

## Base

Same update rule as `adaptive_cm`: partner-whitened matrix-sign (Newton–Schulz)
updates for the Q,K,V,O factors with λ=1.0 Gram regularization, heavy-ball
momentum (0.95), AdamW for biases, and a shifted lognormal LR schedule peaking
at step 40.

## What changed and why

1. **LR floor (0.002).** The stock lognormal tail decays to ~1e-4 by step 400.
   A per-step line-search probe on a held-out proxy showed the optimal LR in the
   last 100 steps is 2–20× larger than the scheduled value, so the tail is
   step-size-limited. Holding `lr = max(lognorm, 0.002)` from ~step 330 onward
   recovers most of that headroom.

2. **Exact gauge balance (step 270).** Attention depends on Q/K and V/O
   through the products `W_Q.T @ W_K` and `W_V.T @ W_O.T`. At step 270 those
   products are refactored into balanced SVD factors. The query/value/output
   biases are transformed at the same time, so the model function is preserved
   to floating-point precision (measured output MSE `2.26e-14`). Momentum is
   restarted because its coordinates belong to the old gauge. This lowers the
   official MPS result from `0.0247666` to the `0.0126` range before retuning.

3. **Late Q,K boost (`qk_tail=0.75` from step 300).** The value/output blocks
   converge early (an exact ALS re-solve of W_V, W_O gains only ~2%); the
   residual error is almost entirely in the attention logits, i.e. in
   M = W_Qᵀ W_K. Ramping the Q,K step size up to 1.75× over the last quarter
   of training keeps the routing converging after V,O are done.

4. **λ decay (`lam_end=0.01`).** The Gram-root Tikhonov λ is ramped 1.0 → 0.01
   over the last 100 steps. Late in training the weights are well-conditioned
   (σ_min no longer collapsing), so the whitening cap `1/√λ` is the binding
   constraint, not stability. The response is smooth and monotone —
   λ_end ∈ {1.0, 0.3, 0.1, 0.05, 0.03, 0.01} gives
   {0.0206, 0.0186, 0.0177, 0.0175, 0.0174, 0.0163} — not a chaotic artifact.

5. **Polyak EMA finish (start 325, decay 0.96).** Parameters are averaged over
   the last 75 steps and the EMA is copied into the model at step 400. The
   final-iterate MSE of this family is jagged in the knobs (routing flips in
   the last 50 steps); EMA over the constant-LR tail damps that oscillation.

## Sweep context

The tail package was selected from ~140 deterministic CPU configurations. The
gauge change was then screened locally and refined with official arm64/MPS
workflow matrices. Three materially different refinements independently beat
the 50% threshold:

| Variant | Official final MSE | Reduction vs pre-gauge MPS |
|---|---:|---:|
| gauge + `qk_tail=0.75` | **0.0112375** | **54.63%** |
| gauge + `qk_tail=1.00` | 0.0119419 | 51.78% |
| gauge + matrix-state reset only | 0.0121643 | 50.88% |

The comparison point is this submission before gauge balancing, measured by
the same workflow at `0.0247666`. The 50% cutoff is `0.0123833`.

Notable dead ends: in-optimizer Newton/HVP steps (finite-difference HVP
calibration has 70–100% relative error from batch noise), exact M-space polar
updates via square inverses (diverges), SGLD-style noise injection, momentum
ramps in either direction, cyclic/restart schedules, gradient accumulation,
final logit upscaling.

## Official verification

The packaged file was rerun independently by the repository's `benchmark`
workflow, including the complete test suite and arm64 assertion. It reproduced
the winning matrix result exactly at `0.011237496510148048`; the artifact is
committed as `result.macos-arm64.json`.

## Reproduce

```bash
uv run python run_eval.py --submission submissions/cm_tail_ema/submission.py \
    --results-json submissions/cm_tail_ema/result.json
```
