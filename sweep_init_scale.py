"""Sweep teacher init scale to measure optimizer sensitivity.

Varies the teacher's Uniform[-alpha, alpha] initialization from near-standard
(alpha=0.09) to the current aggressive setting (alpha=1.0) and compares
AdamW, Muon, and Compositional Muon.

Usage:
    uv run python sweep_init_scale.py
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from data import build_student, build_train_dataset, build_eval_dataset, build_batch_indices
from model import VanillaSelfAttention
from task import CONFIG


def build_teacher_alpha(alpha: float) -> VanillaSelfAttention:
    model = VanillaSelfAttention(CONFIG.embed_dim)
    gen = torch.Generator(device="cpu").manual_seed(CONFIG.teacher_seed)
    for p in model.parameters():
        p.data.uniform_(-alpha, alpha, generator=gen)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_submission(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(make_opt, X_tr, Y_tr, X_ev, Y_ev, batch_idx, cosine_base_lr=None):
    model = build_student(CONFIG).to(X_ev.device)
    opt = make_opt(model.parameters())
    best = float("inf")
    pass_step = None
    for step in range(1, CONFIG.max_steps + 1):
        if cosine_base_lr is not None:
            lr = cosine_base_lr * 0.5 * (1 + math.cos(math.pi * step / CONFIG.max_steps))
            for g in opt.param_groups:
                g["lr"] = lr
        idx = batch_idx[step - 1]
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(X_tr[idx]), Y_tr[idx]).backward()
        opt.step()
        if step % CONFIG.eval_every == 0:
            with torch.no_grad():
                mse = F.mse_loss(model(X_ev), Y_ev).item()
            best = min(best, mse)
            if mse <= CONFIG.target_mse and pass_step is None:
                pass_step = step
    return best, pass_step


def main() -> None:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}\n")

    muon = _load_submission("submissions/muon/submission.py")
    cm   = _load_submission("submissions/comp_muon/submission.py")

    batch_idx = build_batch_indices(CONFIG).to(device)

    alphas = [0.09, 0.15, 0.25, 0.40, 0.60, 1.00]

    optimizers = [
        ("AdamW",     lambda p: torch.optim.AdamW(p, lr=1e-2, betas=(0.9, 0.99)), None),
        ("Muon",      lambda p: muon.Submission(p, lr=0.05),                       None),
        ("CM cosine", lambda p: cm.Submission(p, lr=0.10),                         0.10),
    ]

    # Header
    col = 13
    print(f"{'alpha':>6}  {'init_mse':>9}  {'ratio':>6}  ", end="")
    for name, _, _ in optimizers:
        print(f"  {name+' best':>{col}}  {'pass':>5}", end="")
    print()
    print("-" * (6 + 2 + 9 + 2 + 6 + 2 + len(optimizers) * (col + 9)))

    for alpha in alphas:
        teacher = build_teacher_alpha(alpha)
        train_data = build_train_dataset(teacher, CONFIG)
        eval_data  = build_eval_dataset(teacher, CONFIG)
        X_tr = train_data.inputs.to(device)
        Y_tr = train_data.targets.to(device)
        X_ev = eval_data.inputs.to(device)
        Y_ev = eval_data.targets.to(device)

        with torch.no_grad():
            t_std = teacher(eval_data.inputs[:4]).std().item()
            s0 = build_student(CONFIG)
            s_std = s0(eval_data.inputs[:4]).std().item()
            init_mse = F.mse_loss(build_student(CONFIG).to(device)(X_ev), Y_ev).item()

        ratio = t_std / s_std
        print(f"{alpha:>6.2f}  {init_mse:>9.2e}  {ratio:>5.0f}x  ", end="")

        for name, make_opt, cosine_lr in optimizers:
            best, ps = run(make_opt, X_tr, Y_tr, X_ev, Y_ev, batch_idx, cosine_lr)
            mark = "✓" if ps else " "
            ps_str = str(ps) if ps else "—"
            print(f"  {best:>{col}.2e}{mark} {ps_str:>5}", end="")
        print()


if __name__ == "__main__":
    main()
