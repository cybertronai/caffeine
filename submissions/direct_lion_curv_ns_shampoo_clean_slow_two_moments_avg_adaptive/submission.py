from __future__ import annotations

import torch


def zeropower(g: torch.Tensor) -> torch.Tensor:
    x = g
    transposed = False
    if x.ndim != 2:
        return x
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    x = x / x.norm().clamp_min(1.0e-8)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(3):
        xx = x @ x.T
        x = a * x + b * (xx @ x) + c * (xx @ xx @ x)
    return x.T if transposed else x


def invsqrt_ns(mat: torch.Tensor, steps: int = 4) -> torch.Tensor:
    dim = mat.shape[0]
    eye = torch.eye(dim, device=mat.device, dtype=mat.dtype)
    mat = mat + 1.0e-4 * eye
    scale = mat.norm().clamp_min(1.0e-8)
    y = mat / scale
    z = eye.clone()
    for _ in range(steps):
        t = 0.5 * (3.0 * eye - z @ y)
        y = y @ t
        z = t @ z
    return z / scale.sqrt()


class Submission(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {"lr": 2.9e-3, "eps": 1.0e-8})

    @torch.no_grad()
    def step(self, closure=None):
        sync_mps = False
        for group in self.param_groups:
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)
                    state["prev_g"] = torch.zeros_like(param)
                    state["prev_u"] = torch.zeros_like(param)
                    state["avg"] = param.detach().clone()
                    state["avg_fast"] = param.detach().clone()
                    state["gmean"] = torch.zeros_like(param)
                    state["gvar"] = torch.zeros_like(param)
                    state["sign_ema"] = torch.zeros_like(param)
                    state["gmean_coh"] = torch.zeros_like(param)
                    state["gmean_res"] = torch.zeros_like(param)
                    state["prev_p"] = param.detach().clone()
                    state["hsec"] = torch.ones_like(param)
                    state["scale"] = torch.tensor(1.0, device=param.device)
                    state["reset_done"] = False
                state["step"] += 1
                step = state["step"]
                shape_scale = 2.3 if param.ndim == 2 and param.shape[0] == param.shape[1] else 1.0
                prev_g = state["prev_g"]
                gmean = state["gmean"]
                gvar = state["gvar"]
                sign_ema = state["sign_ema"]
                gmean_coh = state["gmean_coh"]
                gmean_res = state["gmean_res"]
                scale = state["scale"]
                prev_u = state["prev_u"]
                prev_p = state["prev_p"]
                hsec = state["hsec"]
                align = torch.sum(grad * prev_g) / (grad.norm().mul(prev_g.norm()).add(1.0e-8))
                dp = (param - prev_p).abs()
                signed_dp = param - prev_p
                signed_dg = grad - prev_g
                bb = signed_dp.square().sum() / (signed_dp.mul(signed_dg).abs().sum().add(1.0e-8))
                bb_ref = state.get("bb_ref")
                if bb_ref is None:
                    state["bb_ref"] = bb.detach().clone()
                    bb_ref = state["bb_ref"]
                bb_ref.mul_(0.98).add_(bb.detach(), alpha=0.02)
                bb_scale = torch.clamp(bb / bb_ref.clamp_min(1.0e-8), 0.85, 1.15)
                h_now = (grad - prev_g).abs() / dp.clamp_min(1.0e-5)
                h_now = (h_now / h_now.mean().clamp_min(1.0e-8)).clamp(0.25, 4.0)
                hsec.mul_(0.95).add_(h_now, alpha=0.05)
                curv = (grad - prev_g).norm() / grad.norm().clamp_min(1.0e-8)
                elem_curv = (grad - prev_g).abs() / grad.abs().clamp_min(1.0e-6)
                elem_trust = torch.clamp(1.35 / (1.0 + 0.35 * elem_curv), 0.55, 1.25)
                dg_sign = (grad - prev_g).sign()
                damp = torch.clamp(1.35 / (1.0 + 0.35 * curv), 0.55, 1.25)
                scale.mul_(torch.where(align > 0.15, 1.010, torch.where(align < -0.08, 0.90, 0.999)))
                scale.clamp_(0.80 if step >= 340 else 0.55, 1.25 if step >= 340 else 1.75)
                sign_ema.mul_(0.96).add_(grad.sign(), alpha=0.04)
                prev_g.copy_(grad)
                m = state["m"]
                v = state["v"]
                if step in (210, 300, 365):
                    v.zero_()
                raw = m.mul(0.9).add(grad * damp, alpha=0.1)
                v.mul_(0.98).addcmul_(grad, grad, value=0.02)
                adam = (raw / v.sqrt().add(group["eps"])).clamp(-2.0, 2.0)
                if step < 240:
                    update = raw.sign() + 0.05 * adam
                    lr = 2.9e-3
                elif step < 360:
                    mag = (raw.abs() / raw.abs().median().clamp_min(1.0e-8)).sqrt().clamp(0.70, 1.45)
                    phase = min(max((step - 240) / 70.0, 0.0), 1.0)
                    hot_update = raw.sign() + 0.05 * adam
                    tail_update = 0.30 * raw.sign() * mag + 0.50 * adam
                    update = (1.0 - phase) * hot_update + phase * tail_update
                    lr = (1.0 - phase) * 2.9e-3 + phase * 1.10e-3
                else:
                    update = 0.25 * raw.sign() + 0.25 * adam
                    lr = 6.0e-4
                u_align = torch.sum(update * prev_u) / (update.norm().mul(prev_u.norm()).add(1.0e-8))
                if u_align > 0.20:
                    update = update + 0.18 * prev_u
                if step >= 300:
                    update = update * elem_trust
                    dg_alpha = 0.02 if step >= 370 else 0.04
                    htrust = (hsec.mean().clamp_min(1.0e-8) / hsec).sqrt().clamp(0.70, 1.35)
                    update = update - dg_alpha * dg_sign * (2.0 - elem_trust).clamp(0.75, 1.45) * htrust
                    secant_grad = (grad * htrust).clamp(-2.0, 2.0)
                    secant_grad = secant_grad / secant_grad.square().mean().sqrt().clamp_min(1.0e-8)
                    if param.ndim == 2 and param.shape[0] != param.shape[1]:
                        update = update + 0.035 * secant_grad
                    slow = gmean / gmean.square().mean().sqrt().clamp_min(1.0e-8)
                    coh = sign_ema.abs()
                    coh = coh / coh.mean().clamp_min(1.0e-8)
                    slow_coh = gmean_coh / gmean_coh.square().mean().sqrt().clamp_min(1.0e-8)
                    slow_res = gmean_res / gmean_res.square().mean().sqrt().clamp_min(1.0e-8)
                    slow = 0.75 * slow + 0.18 * slow_coh + 0.07 * slow_res
                    update = update + 0.015 * (slow * coh.clamp(0.50, 1.75)).clamp(-2.0, 2.0)
                    update = update + 0.08 * prev_u
                    if param.ndim == 2:
                        left = state.get("left_cov")
                        right = state.get("right_cov")
                        if left is None:
                            state["left_cov"] = torch.eye(param.shape[0], device=param.device, dtype=param.dtype)
                            state["right_cov"] = torch.eye(param.shape[1], device=param.device, dtype=param.dtype)
                            left = state["left_cov"]
                            right = state["right_cov"]
                        left.mul_(0.95).add_(grad @ grad.T, alpha=0.05)
                        right.mul_(0.95).add_(grad.T @ grad, alpha=0.05)
                        pre = invsqrt_ns(left) @ update @ invsqrt_ns(right)
                        pre = pre / pre.square().mean().sqrt().clamp_min(1.0e-8)
                        pre = pre.clamp(-2.0, 2.0)
                        blend = 0.20 if param.shape[0] != param.shape[1] else 0.15
                        update = (1.0 - blend) * update + blend * pre
                elif step >= 210 and param.ndim == 2:
                    left = state.get("left_cov")
                    right = state.get("right_cov")
                    if left is None:
                        state["left_cov"] = torch.eye(param.shape[0], device=param.device, dtype=param.dtype)
                        state["right_cov"] = torch.eye(param.shape[1], device=param.device, dtype=param.dtype)
                        left = state["left_cov"]
                        right = state["right_cov"]
                    left.mul_(0.95).add_(grad @ grad.T, alpha=0.05)
                    right.mul_(0.95).add_(grad.T @ grad, alpha=0.05)
                    pre = invsqrt_ns(left) @ update @ invsqrt_ns(right)
                    pre = pre / pre.square().mean().sqrt().clamp_min(1.0e-8)
                    pre = pre.clamp(-2.0, 2.0)
                    blend = 0.15 if param.shape[0] != param.shape[1] else 0.10
                    update = (1.0 - blend) * update + blend * pre
                step_scale = lr * shape_scale * float(scale.item())
                step_scale *= float(damp.item())
                if step >= 370:
                    step_scale *= float(bb_scale.item())
                if step <= 150:
                    step_scale *= 1.15
                if step >= 210 and param.ndim == 2 and param.shape[0] != param.shape[1]:
                    polar = zeropower(raw)
                    polar = polar / polar.square().mean().sqrt().clamp_min(1.0e-8)
                    update = update + 0.06 * polar.clamp(-2.0, 2.0)
                param.add_(update, alpha=-step_scale)
                if step >= 300:
                    agree = torch.where(update * grad > 0, torch.ones_like(update), torch.zeros_like(update))
                    snr = gmean.abs() / gvar.sqrt().add(1.0e-6)
                    snr_scale = (snr / snr.mean().clamp_min(1.0e-8)).clamp(0.45, 2.10)
                    extra = update * agree / agree.mean().clamp_min(0.20)
                    extra = extra * snr_scale
                    gdir = grad / grad.square().mean().sqrt().clamp_min(1.0e-8)
                    extra = extra + 0.05 * gdir * agree
                    param.add_(extra, alpha=-0.45 * step_scale)
                prev_u.copy_(update)
                if step >= 220:
                    state["avg"].mul_(0.96).add_(param, alpha=0.04)
                if step >= 320:
                    state["avg_fast"].mul_(0.90).add_(param, alpha=0.10)
                if step >= 350:
                    param.lerp_(state["avg"], 0.03)
                    coh_avg = sign_ema.abs()
                    avg_alpha = float((0.002 + 0.006 * coh_avg.mean().clamp(0.0, 1.0)).item())
                    coh_avg = coh_avg / coh_avg.mean().clamp_min(1.0e-8)
                    param.add_((state["avg"] - param) * coh_avg.clamp(0.50, 1.75), alpha=avg_alpha)
                if step == 400:
                    param.lerp_(state["avg_fast"], 0.60)
                    param.lerp_(state["avg"], 0.50)
                    param.add_(adam, alpha=-2.0e-4 * shape_scale)
                    sync_mps = sync_mps or param.device.type == "mps"
                if step >= 390:
                    sync_mps = sync_mps or param.device.type == "mps"
                m.mul_(0.99).add_(grad, alpha=0.01)
                diff = grad - gmean
                gmean.mul_(0.98).add_(grad, alpha=0.02)
                gvar.mul_(0.98).addcmul_(diff, diff, value=0.02)
                coh_raw = sign_ema.abs()
                coh_raw = (coh_raw / coh_raw.mean().clamp_min(1.0e-8)).clamp(0.25, 1.75)
                coh_raw = ((coh_raw - 0.25) / 1.50).clamp(0.0, 1.0)
                gmean_coh.mul_(0.98).add_(grad * coh_raw, alpha=0.02)
                gmean_res.mul_(0.98).add_(grad * (1.0 - coh_raw), alpha=0.02)
                prev_p.copy_(param)
        if sync_mps:
            torch.mps.synchronize()
        return None
