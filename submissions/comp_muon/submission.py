"""Compositional Muon for the caffeine attention optimization problem.

Standard Muon treats W_Q and W_K as independent matrices and orthogonalizes
each gradient separately.  The loss only sees W_Q and W_K through the composed
product M = W_Q W_K^T, so the natural update should control the operator norm of
ΔM, not of ΔW_Q or ΔW_K individually.

This creates a gauge redundancy: rotating (W_Q R, W_K R^T) for any orthogonal R
leaves M unchanged, so gradients have a "vertical" component that moves along the
gauge orbit without changing the loss.  Standard Muon wastes update budget on
this orbit.  Compositional Muon removes it by whitening each factor's gradient
with the partner's inverse Gram root before the spectral sign:

    ΔW_Q = -(η/2) msign(G_Q C_K⁻¹) C_K⁻¹,   C_K = (W_K^T W_K + λI)^{1/2}
    ΔW_K = -(η/2) msign(G_K C_Q⁻¹) C_Q⁻¹

Same structure for the OV circuit (W_O W_V).

All math in "math convention" (W: [d_model, d_head]) internally; PyTorch stores
weights transposed ([d_head, d_model]).  nn.MultiheadAttention packs Q, K, V into
in_proj_weight [3d, d]; we slice it and update each slice in place.

Reference: https://github.com/tilde-research/comp-muon-release
"""
from __future__ import annotations

from typing import Any

import torch


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _inv_gram_root(W_math: torch.Tensor, lam: float) -> torch.Tensor:
    """(W^T W + λI)^{-1/2} where W is in math convention [d_model, d_head].

    Computed via eigh on CPU to avoid MPS limitations; matrices are small
    (128×128) so this is fast regardless.
    """
    gram = (W_math.T @ W_math).float().cpu()
    n = gram.shape[-1]
    eye = torch.eye(n, dtype=torch.float32)
    vals, vecs = torch.linalg.eigh(gram + lam * eye)
    C_inv = (vecs * vals.clamp(min=1e-12).rsqrt().unsqueeze(-2)) @ vecs.mT
    return C_inv.to(device=W_math.device, dtype=W_math.dtype)


def _msign(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Polar factor of G (spectral sign) via degree-5 Newton-Schulz."""
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
    """Rescale delta to the Frobenius norm of ref (the unwhitened polar factor).

    This is comp-muon's per_mat_renorm: the whitening only redistributes
    geometry, not the overall step magnitude.  Without this, effective LR
    shrinks as weights grow — harmful when student and teacher are at very
    different scales.
    """
    scale = ref.norm() / delta.norm().clamp(min=1e-12)
    return delta * scale


def _cm_qk_delta(
    G_Q: torch.Tensor, G_K: torch.Tensor,
    W_Q: torch.Tensor, W_K: torch.Tensor,
    lam: float, ns_steps: int,
    renorm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Half-split CM update for one QK pair.

    All tensors in PyTorch convention [d_head, d_model]; internally converts
    to math convention for the whitening.
    Returns (ΔW_Q, ΔW_K) in PyTorch convention.
    """
    W_Q_m = W_Q.T.float()
    W_K_m = W_K.T.float()
    G_Q_m = G_Q.T.float()
    G_K_m = G_K.T.float()

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
    """Half-split CM update for one OV pair.

    nn.MultiheadAttention stores V as in_proj_weight[2d:], shape [d, d_model].
    out_proj.weight is W_O with shape [d_model, d].  Math convention for OV is
    M = W_O W_V, both transposed relative to PyTorch.
    Returns (ΔW_V, ΔW_O) in PyTorch convention.
    """
    W_V_m = W_V.T.float()   # [d_model, d_v]
    W_O_m = W_O.T.float()   # [d_v, d_model]
    G_V_m = G_V.T.float()
    G_O_m = G_O.T.float()

    # For OV: C_V acts on d_v columns of W_V; C_O acts on d_v rows of W_O.
    C_V_inv = _inv_gram_root(W_V_m, lam)       # (W_V^T W_V + λI)^{-1/2}: [d_v, d_v]
    gram_O = (W_O_m @ W_O_m.T).float().cpu()  # W_O W_O^T acts on the left: [d_v, d_v]
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
    """Compositional Muon for nn.MultiheadAttention weight matrices.

    Identifies circuits from parameter shapes:
      in_proj_weight [3d, d] → W_Q [:d], W_K [d:2d], W_V [2d:]  (packed QKV)
      out_proj.weight [d, d] → W_O
    Applies CM half-split to (W_Q, W_K) and (W_V, W_O) pairs; AdamW for biases.
    """

    def __init__(
        self,
        params,
        lr: float = 0.10,
        momentum: float = 0.95,
        ns_steps: int = 5,
        lambda_reg: float = 1e-2,
        max_steps: int = 400,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
    ):
        params = list(params)
        defaults = dict(
            lr=lr, momentum=momentum, ns_steps=ns_steps, lambda_reg=lambda_reg,
            max_steps=max_steps,
            adamw_lr=adamw_lr, adamw_betas=adamw_betas, adamw_eps=adamw_eps,
        )
        super().__init__([{"params": params}], defaults)
        self._step = 0

    @torch.no_grad()
    def step(self, closure=None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        import math
        self._step += 1

        for group in self.param_groups:
            base_lr = group["lr"]
            max_steps = group["max_steps"]
            # cosine decay: lr * 0.5 * (1 + cos(pi * t / T))
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * self._step / max_steps))
            mu = group["momentum"]
            ns = group["ns_steps"]
            lam = group["lambda_reg"]
            alr = group["adamw_lr"]
            b1, b2 = group["adamw_betas"]
            eps = group["adamw_eps"]

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

            # --- Compositional Muon on QK and OV circuits ---
            if in_proj is not None and out_w is not None:
                d = in_proj.shape[1]
                W_Q = in_proj.data[:d]
                W_K = in_proj.data[d:2 * d]
                W_V = in_proj.data[2 * d:]
                W_O = out_w.data

                G_Q = in_proj.grad[:d]
                G_K = in_proj.grad[d:2 * d]
                G_V = in_proj.grad[2 * d:]
                G_O = out_w.grad

                s = self.state[in_proj]
                if not s:
                    s["bQ"] = torch.zeros_like(W_Q)
                    s["bK"] = torch.zeros_like(W_K)
                    s["bV"] = torch.zeros_like(W_V)
                s_o = self.state[out_w]
                if not s_o:
                    s_o["bO"] = torch.zeros_like(W_O)

                # Nesterov momentum
                def nes(G, buf):
                    buf.mul_(mu).add_(G)
                    return G.add(buf, alpha=mu)

                g_Q = nes(G_Q, s["bQ"])
                g_K = nes(G_K, s["bK"])
                g_V = nes(G_V, s["bV"])
                g_O = nes(G_O, s_o["bO"])

                dQ, dK = _cm_qk_delta(g_Q, g_K, W_Q, W_K, lam, ns)
                dV, dO = _cm_ov_delta(g_V, g_O, W_V, W_O, lam, ns)

                in_proj.data[:d].add_(dQ, alpha=-lr / 2)
                in_proj.data[d:2 * d].add_(dK, alpha=-lr / 2)
                in_proj.data[2 * d:].add_(dV, alpha=-lr / 2)
                out_w.data.add_(dO, alpha=-lr / 2)

            # --- AdamW for biases ---
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
                p.addcdiv_(m / (1 - b1 ** t), (v / (1 - b2 ** t)).sqrt().add_(eps), value=-alr)

        return loss
