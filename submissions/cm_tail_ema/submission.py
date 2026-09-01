"""CM optimizer with a function-preserving gauge balance and tuned tail.

Same core as adaptive_cm (whitened matrix-sign updates for Q,K,V,O with
Gram-root preconditioning, AdamW for biases), plus five tail mechanisms:

- lr_floor: hold a small constant LR once the lognormal tail decays below it.
- balance_step: canonicalize the Q/K and V/O gauges once at step 270.
- qk_tail: ramp up the Q,K step size over the last quarter of training.
- lam_end: ramp the Gram-root regularization down over the last 100 steps.
- Polyak EMA over the final stretch, copied into the params at max_steps.
"""
from __future__ import annotations

import math
from typing import Any

import torch


def _inv_gram_root(W_math: torch.Tensor, lam: float) -> torch.Tensor:
    gram = (W_math.T @ W_math).float().cpu()
    n = gram.shape[-1]
    eye = torch.eye(n, dtype=torch.float32)
    vals, vecs = torch.linalg.eigh(gram + lam * eye)
    C_inv = (vecs * vals.clamp(min=1e-12).rsqrt().unsqueeze(-2)) @ vecs.mT
    return C_inv.to(device=W_math.device, dtype=W_math.dtype)


def _msign(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
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
    return delta * (ref.norm() / delta.norm().clamp(min=1e-12))


def _cm_qk_delta(G_Q, G_K, W_Q, W_K, lam, ns_steps):
    W_Q_m, W_K_m = W_Q.T.float(), W_K.T.float()
    G_Q_m, G_K_m = G_Q.T.float(), G_K.T.float()
    C_K_inv = _inv_gram_root(W_K_m, lam)
    C_Q_inv = _inv_gram_root(W_Q_m, lam)
    M_Q = _msign(G_Q_m @ C_K_inv, ns_steps)
    M_K = _msign(G_K_m @ C_Q_inv, ns_steps)
    dQ_m = M_Q @ C_K_inv
    dK_m = M_K @ C_Q_inv
    return dQ_m.T.to(W_Q.dtype), dK_m.T.to(W_K.dtype)


def _cm_ov_delta(G_V, G_O, W_V, W_O, lam, ns_steps):
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
    dV_m = _restore_norm(M_V @ C_O_inv, M_V)
    dO_m = _restore_norm(C_V_inv @ M_O, M_O)
    return dV_m.T.to(W_V.dtype), dO_m.T.to(W_O.dtype)


class Submission(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 3.05,
        lambda_reg: float = 1.0,
        momentum: float = 0.95,
        ns_steps: int = 5,
        max_steps: int = 400,
        peak_step: float = 40,
        lognorm_sigma: float = 0.63,
        lognorm_shift: float = 1.0,
        sched_T: int = 430,
        lr_floor: float = 0.002,
        qk_tail: float = 0.75,
        qk_tail_start: int = 300,
        balance_step: int = 270,
        ema_start: int = 325,
        ema_decay: float = 0.96,
        lam_end: float = 0.01,
        adamw_lr: float = 3e-3,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
    ):
        params = list(params)
        defaults = dict(
            lr=lr, lambda_reg=lambda_reg, momentum=momentum, ns_steps=ns_steps,
            max_steps=max_steps, peak_step=peak_step, lognorm_sigma=lognorm_sigma,
            lognorm_shift=lognorm_shift, sched_T=sched_T, lr_floor=lr_floor,
            qk_tail=qk_tail, qk_tail_start=qk_tail_start,
            balance_step=balance_step,
            ema_start=ema_start, ema_decay=ema_decay, lam_end=lam_end,
            adamw_lr=adamw_lr, adamw_betas=adamw_betas, adamw_eps=adamw_eps,
        )
        super().__init__([{"params": params}], defaults)
        self._step = 0
        c = lognorm_shift
        mu = math.log(peak_step + c) + lognorm_sigma**2
        _sq2pi = math.sqrt(2 * math.pi)

        def _logn(x):
            return math.exp(-(math.log(x) - mu)**2 / (2 * lognorm_sigma**2)) / (
                x * lognorm_sigma * _sq2pi)

        self._lognorm_offset = _logn(sched_T + c)
        self._lognorm_peak_raw = _logn(peak_step + c) - self._lognorm_offset
        self._lognorm_fn = _logn
        self._ema: dict[int, torch.Tensor] = {}

    def _sched_lr(self, group):
        t = self._step
        c = group["lognorm_shift"]
        raw = max(self._lognorm_fn(t + c) - self._lognorm_offset, 0.0)
        return max(group["lr"] * raw / self._lognorm_peak_raw, group["lr_floor"])

    @torch.no_grad()
    def step(self, closure=None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step += 1

        for group in self.param_groups:
            lr = self._sched_lr(group)
            mu, ns = group["momentum"], group["ns_steps"]
            alr = group["adamw_lr"]
            b1, b2 = group["adamw_betas"]
            eps_adam = group["adamw_eps"]
            lam = group["lambda_reg"]
            le = group["lam_end"]
            if le:
                r100 = min(max((self._step - (group["max_steps"] - 100)) / 100, 0.0), 1.0)
                lam = lam + (le - lam) * r100

            in_proj = out_w = None
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

                g_Q = nes(G_Q, s["bQ"]); g_K = nes(G_K, s["bK"])
                g_V = nes(G_V, s["bV"]); g_O = nes(G_O, s_o["bO"])

                dQ, dK = _cm_qk_delta(g_Q, g_K, W_Q, W_K, lam, ns)
                dV, dO = _cm_ov_delta(g_V, g_O, W_V, W_O, lam, ns)

                t0 = group["qk_tail_start"]
                ramp = max(0.0, (self._step - t0) / max(group["max_steps"] - t0, 1))
                qk_eff = 1.0 + group["qk_tail"] * ramp
                half_qk = lr * qk_eff / 2
                half_ov = lr / 2
                in_proj.data[:d].add_(dQ, alpha=-half_qk)
                in_proj.data[d:2*d].add_(dK, alpha=-half_qk)
                in_proj.data[2*d:].add_(dV, alpha=-half_ov)
                out_w.data.add_(dO, alpha=-half_ov)

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

            # Polyak averaging over ALL params
            es = group["ema_start"]
            if self._step >= es:
                decay = group["ema_decay"]
                for p in group["params"]:
                    key = id(p)
                    if key not in self._ema:
                        self._ema[key] = p.detach().clone()
                    else:
                        self._ema[key].mul_(decay).add_(p.detach(), alpha=1 - decay)
            if self._step == group["max_steps"]:
                for p in group["params"]:
                    key = id(p)
                    if key in self._ema:
                        p.data.copy_(self._ema[key])

            if self._step == group["balance_step"]:
                self._balance_attention_gauges(group)

        return loss

    @torch.no_grad()
    def _balance_attention_gauges(self, group) -> None:
        """Put both matrix products in balanced SVD gauges.

        Attention depends on Q/K and V/O through their matrix products, not
        the individual factors.  This refactorization leaves the model's
        function unchanged while restoring well-conditioned coordinates for
        the remaining optimizer steps.
        """
        in_proj = out_w = None
        for p in group["params"]:
            if p.ndim == 2 and p.shape[0] == 3 * p.shape[1]:
                in_proj = p
            elif p.ndim == 2 and p.shape[0] == p.shape[1]:
                out_w = p
        if in_proj is None or out_w is None:
            return

        d = out_w.shape[0]
        in_bias = next(
            (p for p in group["params"] if p.ndim == 1 and p.numel() == 3 * d),
            None,
        )
        out_bias = next(
            (p for p in group["params"] if p.ndim == 1 and p.numel() == d),
            None,
        )
        if in_bias is None or out_bias is None:
            return

        WQ = in_proj.data[:d].float().cpu()
        WK = in_proj.data[d:2 * d].float().cpu()
        WV = in_proj.data[2 * d:].float().cpu()
        WO = out_w.data.float().cpu()
        bq = in_bias.data[:d].float().cpu()
        bv = in_bias.data[2 * d:].float().cpu()
        bo = out_bias.data.float().cpu()

        # logits = x (WQ.T @ WK) x.T + 1 (bq @ WK) x.T, modulo
        # row-wise softmax constants.
        U, singular, Vh = torch.linalg.svd(WQ.T @ WK, full_matrices=False)
        root = singular.clamp_min(1e-12).sqrt()
        WQ_new = root[:, None] * U.T
        WK_new = root[:, None] * Vh
        key_linear = bq @ WK
        bq_new = torch.linalg.solve(WK_new.T, key_linear)

        # output = x (WV.T @ WO.T) + bv @ WO.T + bo.
        U, singular, Vh = torch.linalg.svd(WV.T @ WO.T, full_matrices=False)
        root = singular.clamp_min(1e-12).sqrt()
        WV_new = root[:, None] * U.T
        WO_new = Vh.T * root[None, :]
        bo_new = bv @ WO.T + bo

        in_proj.data[:d].copy_(WQ_new.to(in_proj))
        in_proj.data[d:2 * d].copy_(WK_new.to(in_proj))
        in_proj.data[2 * d:].copy_(WV_new.to(in_proj))
        out_w.data.copy_(WO_new.to(out_w))
        in_bias.data[:d].copy_(bq_new.to(in_bias))
        in_bias.data[d:2 * d].zero_()  # key bias is a softmax-row constant
        in_bias.data[2 * d:].zero_()
        out_bias.data.copy_(bo_new.to(out_bias))

        # All stored optimizer state belongs to the old coordinates. Reset
        # both tensor moments and Adam's bias-correction counter.
        for state in self.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()
            if "step" in state:
                state["step"] = 0
