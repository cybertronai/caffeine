# CM + Tail-Schedule Package (cm_tail_ema)

**Submission: `cm_tail_ema`** — adaptive_cm core with a retuned lognormal
schedule and three late-training mechanisms.
Local `random_teacher` MSE@400: **0.0163** (vs adaptive_cm 0.0368, AdamW 58.3
on the same local harness; Linux x86_64, torch 2.13.0+cpu).

---

## Base

Same update rule as `adaptive_cm`: partner-whitened matrix-sign (Newton–Schulz)
updates for the Q,K,V,O factors with λ=1.0 Gram regularization, heavy-ball
momentum (0.95), AdamW for biases, and a shifted lognormal LR schedule peaking
at step 40. Changes are confined to schedule shape and the last ~100 steps.

## What changed and why

1. **LR floor (0.002).** The stock lognormal tail decays to ~1e-4 by step 400.
   A per-step line-search probe on a held-out proxy showed the optimal LR in the
   last 100 steps is 2–20× larger than the scheduled value, so the tail is
   step-size-limited. Holding `lr = max(lognorm, 0.002)` from ~step 330 onward
   recovers most of that headroom.

2. **Late Q,K boost (`qk_tail=1.25` from step 300).** The value/output blocks
   converge early (an exact ALS re-solve of W_V, W_O gains only ~2%); the
   residual error is almost entirely in the attention logits, i.e. in
   M = W_Qᵀ W_K. Ramping the Q,K step size up to 2.25× over the last quarter
   of training keeps the routing converging after V,O are done.

3. **λ decay (`lam_end=0.01`).** The Gram-root Tikhonov λ is ramped 1.0 → 0.01
   over the last 100 steps. Late in training the weights are well-conditioned
   (σ_min no longer collapsing), so the whitening cap `1/√λ` is the binding
   constraint, not stability. The response is smooth and monotone —
   λ_end ∈ {1.0, 0.3, 0.1, 0.05, 0.03, 0.01} gives
   {0.0206, 0.0186, 0.0177, 0.0175, 0.0174, 0.0163} — not a chaotic artifact.

4. **Polyak EMA finish (start 325, decay 0.96).** Parameters are averaged over
   the last 75 steps and the EMA is copied into the model at step 400. The
   final-iterate MSE of this family is jagged in the knobs (routing flips in
   the last 50 steps); EMA over the constant-LR tail damps that oscillation.

## Sweep context

~140 deterministic configurations were evaluated with an in-process replica of
the harness (bit-identical final MSE to `run_eval.py` on this machine).
Full log: `exp/results.jsonl` (not part of the submission).

Notable dead ends: in-optimizer Newton/HVP steps (finite-difference HVP
calibration has 70–100% relative error from batch noise), exact M-space polar
updates via square inverses (diverges), SGLD-style noise injection, momentum
ramps in either direction, cyclic/restart schedules, gradient accumulation,
final logit upscaling.

## Caveats

The number above was measured locally on CPU. Official scoring runs the
`benchmark` workflow on `macos-15` (MPS); different floating-point numerics
reroll the chaotic tail, so the exact value will shift. The four mechanisms
are individually stable across perturbations (family members land in
0.016–0.021), so the submission should remain well ahead of the current
leaderboard entry (0.198) there.

## Reproduce

```bash
uv run python run_eval.py --submission submissions/cm_tail_ema/submission.py \
    --results-json submissions/cm_tail_ema/result.local.json
```
