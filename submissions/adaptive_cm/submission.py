"""Compositional Muon with calibrated Tikhonov regularisation for the caffeine benchmark.

== λ is the Tikhonov regularisation constant ==

The CM inverse Gram root is:

    C_K^{-1} = (W_K^T W_K + λI)^{-1/2}

This is standard Tikhonov regularisation of the Gram matrix W_K^T W_K. In the
usual Tikhonov setting, λ is chosen to be small — it just prevents numerical
blow-up near zero eigenvalues, and smaller λ means a more faithful inversion.

Here the optimal λ is 1.0, which is large. The reason is structural, not
numerical: the GL(d) gauge symmetry of attention forces σ_min → 0.

== The spectral collapse problem ==

The attention product M = W_Q W_K^T has a GL(d) gauge symmetry: any invertible A
leaves M unchanged under (W_Q, W_K) → (W_Q A^T, W_K A^{-1}). This means σ_min(W_K)
drifts toward zero universally within 10–25 optimisation steps, regardless of the
optimiser. The matrix W_K^T W_K is structurally (not merely numerically) singular.

The Tikhonov amplification ceiling is therefore:

    C_inv_max = 1/√(σ_min² + λ)  →  1/√λ  as  σ_min → 0  (always, structurally)

With standard λ=0.01, the ceiling is 1/√0.01 = 10: every update step amplifies
the degenerate directions 10×. With λ=1.0, the ceiling is 1.0: degenerate directions
receive no net amplification. The gauge orbit makes the difference between these
two choices catastrophic rather than minor.

== Why λ = 0.01 fails (the feedback loop) ==

With the default λ=0.01, C_inv_max ≈ 10 at all times. Updates in degenerate
directions are amplified 10×, driving W_K to grow. This increases lam_nat =
tr(W_K^T W_K)/d (the mean squared singular value), which in turn makes the
spectral structure worse. On the standard benchmark, lam_nat grows by a factor
of 10^14 under λ=0.01 compared to a stable ~160 under λ=1.0. The feedback loop:

    σ_min → 0  →  C_inv_max = 10  →  ΔW_K large in degenerate directions
             →  lam_nat grows  →  λ/lam_nat shrinks  →  worse calibration

== Why λ = 1.0 fixes it ==

Setting λ=1.0 caps C_inv_max ≤ 1/√λ = 1.0 regardless of how σ_min evolves.
Degenerate directions receive no more amplification than unit-norm directions —
no runaway. In the standard Tikhonov picture: λ=1.0 is large enough that the
regularised inverse is effectively bounded even when the true Gram is rank-deficient,
which is the persistent condition here due to the gauge orbit.

λ sweep result (lr=1.5, standard benchmark):

    λ=0.01 → MSE 1983  (10× amplification, runaway spectrum)
    λ=1.0  → MSE 2.27  (1× amplification, stable)   ← optimal
    λ=10.0 → MSE 6.6   (whitening too suppressed)

== The natural-λ interpretation ==

lam_nat = tr(W_K^T W_K)/d is the mean squared singular value. With λ=1.0:

    Small-teacher regime (lam_nat ≈ 1):  λ ≈ lam_nat → C_inv_max = 1.0
    Large-teacher regime (lam_nat ≈ 300): λ << lam_nat → bulk whitening is natural
                                           (sv >> 1, so (sv² + 1)^{-1/2} ≈ 1/sv)

In both regimes, λ=1.0 provides a stable floor without over-suppressing the
whitening of the well-conditioned directions. This is the primary recommendation.

== Adaptive mode ==

For settings where the right λ is genuinely unknown (e.g. custom architectures
where lam_nat < 1 at init), set adaptive=True. Then λ = max(eps × lam_nat, lam_min)
tracks the spectrum automatically. This is most useful when lam_nat ≈ 1, where
adaptive ε=1.0 is equivalent to fixed λ=1.0.

Note: adaptive mode is WORSE on the standard benchmark (lam_nat → 300 → λ → 300 →
C_inv_max → 0.058, too conservative). Use fixed λ=1.0 for best results here.
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _inv_gram_root(W_math: torch.Tensor, lam: float) -> torch.Tensor:
    """(W^T W + λI)^{-1/2} via CPU eigh (MPS-safe, fast for d=128)."""
    gram = (W_math.T @ W_math).float().cpu()
    n = gram.shape[-1]
    eye = torch.eye(n, dtype=torch.float32)
    vals, vecs = torch.linalg.eigh(gram + lam * eye)
    C_inv = (vecs * vals.clamp(min=1e-12).rsqrt().unsqueeze(-2)) @ vecs.mT
    return C_inv.to(device=W_math.device, dtype=W_math.dtype)


def _inv_gram_root_adaptive(
    W_math: torch.Tensor, eps: float, lam_min: float
) -> tuple[torch.Tensor, float, float]:
    """(W^T W + λI)^{-1/2} with λ = max(eps × lam_nat, lam_min).

    Returns (C_inv, lam_nat, lam_used).
    """
    gram = (W_math.T @ W_math).float().cpu()
    n = gram.shape[-1]
    lam_nat = gram.trace().item() / n
    lam = max(eps * lam_nat, lam_min)
    eye = torch.eye(n, dtype=torch.float32)
    vals, vecs = torch.linalg.eigh(gram + lam * eye)
    C_inv = (vecs * vals.clamp(min=1e-12).rsqrt().unsqueeze(-2)) @ vecs.mT
    return C_inv.to(device=W_math.device, dtype=W_math.dtype), lam_nat, lam


def _msign(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Polar factor of G via degree-5 Newton-Schulz."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ A @ X)
    return X.T if G.shape[0] > G.shape[1] else X


def _restore_norm(delta: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Scale delta to ||ref||_F (comp-muon per_mat_renorm)."""
    return delta * (ref.norm() / delta.norm().clamp(min=1e-12))


# ---------------------------------------------------------------------------
# CM update steps
# ---------------------------------------------------------------------------

def _cm_qk_delta(
    G_Q: torch.Tensor, G_K: torch.Tensor,
    W_Q: torch.Tensor, W_K: torch.Tensor,
    lam: float, ns_steps: int,
    renorm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Half-split CM update for QK pair. Returns (ΔW_Q, ΔW_K) in PyTorch convention."""
    W_Q_m, W_K_m = W_Q.T.float(), W_K.T.float()
    G_Q_m, G_K_m = G_Q.T.float(), G_K.T.float()

    C_K_inv = _inv_gram_root(W_K_m, lam)
    C_Q_inv = _inv_gram_root(W_Q_m, lam)

    M_Q = _msign(G_Q_m @ C_K_inv, ns_steps)
    M_K = _msign(G_K_m @ C_Q_inv, ns_steps)
    dQ_m = M_Q @ C_K_inv
    dK_m = M_K @ C_Q_inv

    if renorm:
        dQ_m = _restore_norm(dQ_m, M_Q)
        dK_m = _restore_norm(dK_m, M_K)

    return dQ_m.T.to(W_Q.dtype), dK_m.T.to(W_K.dtype)


def _cm_ov_delta(
    G_V: torch.Tensor, G_O: torch.Tensor,
    W_V: torch.Tensor, W_O: torch.Tensor,
    lam: float, ns_steps: int,
    renorm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Half-split CM update for OV pair. Returns (ΔW_V, ΔW_O) in PyTorch convention."""
    W_V_m, W_O_m = W_V.T.float(), W_O.T.float()
    G_V_m, G_O_m = G_V.T.float(), G_O.T.float()

    C_V_inv = _inv_gram_root(W_V_m, lam)
    gram_O = (W_O_m @ W_O_m.T).float().cpu()
    n = gram_O.shape[-1]
    eye = torch.eye(n, dtype=torch.float32)
    vals, vecs = torch.linalg.eigh(gram_O + lam * eye)
    C_O_inv = ((vecs * vals.clamp(min=1e-12).rsqrt().unsqueeze(-2)) @ vecs.mT
               ).to(device=W_V.device, dtype=W_V.dtype)

    M_V = _msign(G_V_m @ C_O_inv, ns_steps)
    M_O = _msign(C_V_inv @ G_O_m, ns_steps)
    dV_m = M_V @ C_O_inv
    dO_m = C_V_inv @ M_O

    if renorm:
        dV_m = _restore_norm(dV_m, M_V)
        dO_m = _restore_norm(dO_m, M_O)

    return dV_m.T.to(W_V.dtype), dO_m.T.to(W_O.dtype)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class Submission(torch.optim.Optimizer):
    """Compositional Muon with λ=1.0 for the caffeine benchmark.

    The key hyperparameter is lambda_reg=1.0 (see module docstring). The lr=1.5
    default is tuned for the standard benchmark (uniform(-1,1) teacher, d=128).

    Identifies nn.MultiheadAttention circuits from parameter shapes:
      in_proj_weight [3d, d] → W_Q [:d], W_K [d:2d], W_V [2d:]
      out_proj.weight [d, d] → W_O
    Applies CM half-split to QK and OV pairs; AdamW for biases.

    Args:
        params:       model.parameters()
        lr:           base learning rate (cosine-annealed to 0 over max_steps)
        lambda_reg:   Tikhonov regularisation for Gram inverse (1.0 recommended)
        adaptive:     if True, use λ = max(eps × lam_nat, lam_min) per step
        eps:          adaptive λ scale factor (only used when adaptive=True)
        lam_min:      minimum λ in adaptive mode
        momentum:     Nesterov momentum coefficient
        ns_steps:     Newton-Schulz iterations for polar approximation
        max_steps:    total training steps (controls cosine LR schedule)
    """

    def __init__(
        self,
        params,
        lr: float = 1.5,
        lambda_reg: float = 1.0,
        adaptive: bool = False,
        eps: float = 1.0,
        lam_min: float = 0.1,
        momentum: float = 0.95,
        ns_steps: int = 5,
        max_steps: int = 400,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
    ):
        params = list(params)
        defaults = dict(
            lr=lr, lambda_reg=lambda_reg, adaptive=adaptive, eps=eps, lam_min=lam_min,
            momentum=momentum, ns_steps=ns_steps, max_steps=max_steps,
            adamw_lr=adamw_lr, adamw_betas=adamw_betas, adamw_eps=adamw_eps,
        )
        super().__init__([{"params": params}], defaults)
        self._step = 0

    def _get_lam(self, W_math: torch.Tensor, group: dict) -> float:
        if group["adaptive"]:
            _, _, lam = _inv_gram_root_adaptive(W_math, group["eps"], group["lam_min"])
            return lam
        return group["lambda_reg"]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1

        for group in self.param_groups:
            base_lr = group["lr"]
            max_steps = group["max_steps"]
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * self._step / max_steps))
            mu = group["momentum"]
            ns = group["ns_steps"]
            alr = group["adamw_lr"]
            b1, b2 = group["adamw_betas"]
            eps_adam = group["adamw_eps"]

            in_proj = None
            out_w = None
            bias_params = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim == 2 and p.shape[0] == 3 * p.shape[1]:
                    in_proj = p
                elif p.ndim == 2 and p.shape[0] == p.shape[1]:
                    out_w = p
                else:
                    bias_params.append(p)

            if in_proj is not None and out_w is not None:
                d = in_proj.shape[1]
                W_Q = in_proj.data[:d];    W_K = in_proj.data[d:2*d]
                W_V = in_proj.data[2*d:];  W_O = out_w.data
                G_Q = in_proj.grad[:d];    G_K = in_proj.grad[d:2*d]
                G_V = in_proj.grad[2*d:];  G_O = out_w.grad

                s = self.state[in_proj]
                if not s:
                    s["bQ"] = torch.zeros_like(W_Q)
                    s["bK"] = torch.zeros_like(W_K)
                    s["bV"] = torch.zeros_like(W_V)
                s_o = self.state[out_w]
                if not s_o:
                    s_o["bO"] = torch.zeros_like(W_O)

                def nes(G, buf):
                    buf.mul_(mu).add_(G)
                    return G.add(buf, alpha=mu)

                g_Q = nes(G_Q, s["bQ"])
                g_K = nes(G_K, s["bK"])
                g_V = nes(G_V, s["bV"])
                g_O = nes(G_O, s_o["bO"])

                lam_qk = self._get_lam(W_K.T.float(), group)
                lam_ov = self._get_lam(W_V.T.float(), group)

                dQ, dK = _cm_qk_delta(g_Q, g_K, W_Q, W_K, lam_qk, ns)
                dV, dO = _cm_ov_delta(g_V, g_O, W_V, W_O, lam_ov, ns)

                in_proj.data[:d].add_(dQ, alpha=-lr / 2)
                in_proj.data[d:2*d].add_(dK, alpha=-lr / 2)
                in_proj.data[2*d:].add_(dV, alpha=-lr / 2)
                out_w.data.add_(dO, alpha=-lr / 2)

            for p in bias_params:
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(g)
                    st["v"] = torch.zeros_like(g)
                st["step"] += 1
                t = st["step"]
                m, v = st["m"], st["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                p.addcdiv_(m / (1 - b1**t), (v / (1 - b2**t)).sqrt().add_(eps_adam), value=-alr)

        return loss
