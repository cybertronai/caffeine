"""Diagnostic script for the random_teacher track calibration issue.

Runs AdamW to plateau, then checks whether the plateau is a local minimum
or an active optimization failure, and shows why the aggressive teacher
initialization makes the problem hard.

Usage:
    uv run python diagnose_random_teacher.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from data import build_student, build_teacher, build_train_dataset, build_eval_dataset, build_batch_indices
from task import CONFIG


def main() -> None:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}\n")

    teacher = build_teacher(CONFIG)
    train_data = build_train_dataset(teacher, CONFIG)
    eval_data = build_eval_dataset(teacher, CONFIG)
    batch_idx = build_batch_indices(CONFIG).to(device)
    X_tr = train_data.inputs.to(device)
    Y_tr = train_data.targets.to(device)
    X_ev = eval_data.inputs.to(device)
    Y_ev = eval_data.targets.to(device)

    # -------------------------------------------------------------------------
    # 1. Teacher vs student output scale
    # -------------------------------------------------------------------------
    student0 = build_student(CONFIG).to(device)
    # teacher lives on CPU (build_teacher doesn't move it); eval data is on device
    X_ev_cpu = X_ev.cpu()
    with torch.no_grad():
        teacher_out_std = teacher(X_ev_cpu[:4]).std().item()
        student_out_std = student0(X_ev[:4]).cpu().std().item()

    print("=== Output scale ===")
    print(f"  teacher output std : {teacher_out_std:.4f}")
    print(f"  student output std : {student_out_std:.4f}  (ratio {teacher_out_std / max(student_out_std, 1e-9):.0f}x)")
    print()

    # -------------------------------------------------------------------------
    # 2. Attention logit statistics (what softmax sees)
    # -------------------------------------------------------------------------
    X_sample = X_ev_cpu[:1]
    t_attn = teacher.attention          # on CPU
    s_attn = student0.attention.cpu()

    dim = CONFIG.embed_dim
    wq_t = t_attn.in_proj_weight[:dim]
    wk_t = t_attn.in_proj_weight[dim:2 * dim]
    wq_s = s_attn.in_proj_weight[:dim]
    wk_s = s_attn.in_proj_weight[dim:2 * dim]

    with torch.no_grad():
        Q_t = X_sample @ wq_t.T
        K_t = X_sample @ wk_t.T
        logits_t = (Q_t @ K_t.transpose(-2, -1)) / (dim ** 0.5)
        sm_t = F.softmax(logits_t[0], dim=-1)

        Q_s = X_sample @ wq_s.T
        K_s = X_sample @ wk_s.T
        logits_s = (Q_s @ K_s.transpose(-2, -1)) / (dim ** 0.5)
        sm_s = F.softmax(logits_s[0], dim=-1)

    ent_t = -(sm_t * sm_t.clamp(min=1e-9).log()).sum(-1).mean().item()
    ent_s = -(sm_s * sm_s.clamp(min=1e-9).log()).sum(-1).mean().item()
    seq_len = CONFIG.sequence_length
    uniform_ent = torch.log(torch.tensor(float(seq_len))).item()

    print("=== Attention softmax regime ===")
    print(f"  {'':30s}  {'logit std':>10}  {'entropy':>10}  {'top-1 avg':>10}")
    print(f"  {'teacher (Uniform[-1,1] init)':30s}  {logits_t.std().item():10.2f}  {ent_t:10.3f}  {sm_t.max(-1).values.mean().item():10.4f}")
    print(f"  {'student (default PyTorch init)':30s}  {logits_s.std().item():10.2f}  {ent_s:10.3f}  {sm_s.max(-1).values.mean().item():10.4f}")
    print(f"  (uniform entropy over {seq_len} positions would be {uniform_ent:.3f})")
    print()

    # -------------------------------------------------------------------------
    # 3. Train AdamW to plateau, record MSE trajectory
    # -------------------------------------------------------------------------
    model = build_student(CONFIG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.99), weight_decay=0.0)

    print("=== AdamW MSE trajectory ===")
    with torch.no_grad():
        init_mse = F.mse_loss(model(X_ev), Y_ev).item()
    print(f"  step   0  eval_mse={init_mse:.4e}")

    for step in range(1, CONFIG.max_steps + 1):
        model.train()
        idx = batch_idx[step - 1]
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(X_tr[idx]), Y_tr[idx]).backward()
        opt.step()
        if step % CONFIG.eval_every == 0:
            with torch.no_grad():
                mse = F.mse_loss(model(X_ev), Y_ev).item()
            marker = " *** PASS" if mse <= CONFIG.target_mse else ""
            print(f"  step {step:3d}  eval_mse={mse:.4e}{marker}")

    print(f"  target          = {CONFIG.target_mse:.4e}")
    print()

    # -------------------------------------------------------------------------
    # 4. Gradient norm at plateau — is it a local minimum?
    # -------------------------------------------------------------------------
    opt.zero_grad(set_to_none=True)
    full_train_loss = F.mse_loss(model(X_tr), Y_tr)
    full_train_loss.backward()

    total_gnorm = sum(
        p.grad.norm().item() ** 2
        for p in model.parameters()
        if p.grad is not None
    ) ** 0.5

    print("=== Gradient at plateau (full training set) ===")
    print(f"  train MSE  : {full_train_loss.item():.4e}")
    print(f"  total gnorm: {total_gnorm:.4e}  (near zero = local min; large = active failure)")
    print()
    print("  per-parameter breakdown:")
    for name, p in model.attention.named_parameters():
        if p.grad is not None:
            print(f"    {name:30s}  grad_norm={p.grad.norm().item():.4e}  param_norm={p.data.norm().item():.4e}")


if __name__ == "__main__":
    main()
