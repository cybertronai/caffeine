# Direct Lion + Curvature + Learned Slow Subnetworks

This submission is a legitimate optimizer-only entry for the Caffeine attention
benchmark. It only uses parameter values and `.grad` tensors available inside
`Submission.step()`.

Local direct run on MPS:

```text
best_eval_mse  = 0.1984204649925232
final_eval_mse = 0.1984204649925232
max_steps      = 400
target_mse     = 4e-07
```

The optimizer combines several mechanisms found during search:

- Fast sign/Lion-style entry into a useful basin.
- Stored-gradient finite-difference curvature from `g_t - g_{t-1}`.
- Small Shampoo-style factored matrix preconditioning for attention matrices.
- Rectangular matrix secant-gradient correction for the packed attention input
  projection.
- Learned coherent/residual slow subnetworks:
  - maintain `sign_ema = EMA(sign(g))`;
  - use sign persistence as a soft per-parameter coherence score;
  - keep separate slow gradient memories for coherent and residual coordinates;
  - recombine those memories before adding the slow correction.
- Coherence-weighted late averaging:
  - parameters with stronger sign-persistent trajectories are nudged more toward
    the slow parameter EMA late in training;
  - the averaging strength is adaptive from the mean coherence level.

The main empirical jump came from sign-aligned slow memory and learned
coherent/residual subnetworks. Richer arbitrary buckets, explicit time-window
membership, and current/slow-gradient dot-product groups were tested locally but
were noisier than the simple two-way coherent/residual split.
