"""Caffeine benchmark with Pythia-14m teacher initialization.

Tests whether the λ=1.0 Tikhonov fix (and adaptive λ) generalizes to teachers
initialized from real pre-trained weights rather than Uniform[-1,1].

Pythia-14m: hidden_size=128 (matches benchmark d), 4 heads, 6 layers.
lam_nat(W_K) ranges 0.13–5.12 per layer (vs ~43 for uniform(-1,1) teacher).

Sections:
  1. All 6 Pythia layers as teacher: quick sweep of CM λ=0.01 vs λ=1.0 vs AdamW
  2. Best layer(s): full spectral flow + adaptive λ comparison

Usage:
    uv run python run_pythia_teacher.py
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from data import build_batch_indices, build_eval_dataset, build_student, build_train_dataset
from model import VanillaSelfAttention
from task import CONFIG


# ---------------------------------------------------------------------------
# Pythia teacher builder
# ---------------------------------------------------------------------------

def _load_pythia():
    from transformers import GPTNeoXForCausalLM
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-14m",
        cache_dir="/tmp/pythia-cache",
    ).float()
    model.eval()
    return model


def build_pythia_teacher(layer_idx: int, pythia=None) -> VanillaSelfAttention:
    """VanillaSelfAttention(d=128, num_heads=1) initialized from Pythia-14m layer."""
    if pythia is None:
        pythia = _load_pythia()
    h = pythia.config.hidden_size  # 128

    attn = pythia.gpt_neox.layers[layer_idx].attention
    W_qkv = attn.query_key_value.weight.data.float().clone()  # [384, 128]
    b_qkv = attn.query_key_value.bias.data.float().clone()    # [384]
    W_o   = attn.dense.weight.data.float().clone()            # [128, 128]
    b_o   = attn.dense.bias.data.float().clone()              # [128]

    teacher = VanillaSelfAttention(h, num_heads=1)
    teacher.attention.in_proj_weight.data.copy_(W_qkv)
    teacher.attention.in_proj_bias.data.copy_(b_qkv)
    teacher.attention.out_proj.weight.data.copy_(W_o)
    teacher.attention.out_proj.bias.data.copy_(b_o)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _pythia_spectral_summary(pythia) -> None:
    h = pythia.config.hidden_size
    print(f"\nPythia-14m weight statistics (d={h}):")
    print(f"{'layer':>5}  {'lam_nat(K)':>10}  {'sv_min(K)':>9}  {'sv_max(K)':>9}  {'lam_nat(Q)':>10}  {'lam_nat(O)':>10}")
    print("-" * 65)
    for li in range(pythia.config.num_hidden_layers):
        attn = pythia.gpt_neox.layers[li].attention
        W = attn.query_key_value.weight.data.float()
        W_Q, W_K = W[:h], W[h:2*h]
        W_O = attn.dense.weight.data.float()
        sv_k = torch.linalg.svdvals(W_K)
        lk = (W_K.T @ W_K).trace().item() / h
        lq = (W_Q.T @ W_Q).trace().item() / h
        lo = (W_O.T @ W_O).trace().item() / h
        print(f"{li:>5}  {lk:>10.4f}  {sv_k[-1].item():>9.4f}  {sv_k[0].item():>9.4f}  {lq:>10.4f}  {lo:>10.4f}")


# ---------------------------------------------------------------------------
# Shared helpers (mirrors run_adaptive_cm_caffeine.py)
# ---------------------------------------------------------------------------

def _load(path: str):
    spec = importlib.util.spec_from_file_location("sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gram_stats(W_py: torch.Tensor, lam: float) -> dict:
    sv = torch.linalg.svdvals(W_py.float().cpu())
    return {
        "lam_nat":   sv.pow(2).mean().item(),
        "sv_min":    sv[-1].item(),
        "C_inv_max": (sv[-1].item() ** 2 + lam) ** -0.5,
    }


def build_datasets(teacher, device):
    """Build train/eval datasets from a CPU teacher, return on target device."""
    cpu_teacher = teacher.cpu()
    X_tr = build_train_dataset(cpu_teacher, CONFIG).inputs.to(device)
    Y_tr = build_train_dataset(cpu_teacher, CONFIG).targets.to(device)
    X_ev = build_eval_dataset(cpu_teacher, CONFIG).inputs.to(device)
    Y_ev = build_eval_dataset(cpu_teacher, CONFIG).targets.to(device)
    batch_idx = build_batch_indices(CONFIG).to(device)
    return X_tr, Y_tr, X_ev, Y_ev, batch_idx


def run_training(student, opt_factory, datasets, device, checkpoints, lam_for_stats):
    X_tr, Y_tr, X_ev, Y_ev, batch_idx = datasets

    d = CONFIG.embed_dim
    s = student
    opt = opt_factory(s.parameters())
    log = []

    for step in range(1, CONFIG.max_steps + 1):
        s.train()
        idx = batch_idx[step - 1]
        opt.zero_grad(set_to_none=True)
        F.mse_loss(s(X_tr[idx]), Y_tr[idx]).backward()
        opt.step()

        if step in checkpoints:
            s.eval()
            with torch.no_grad():
                mse = F.mse_loss(s(X_ev), Y_ev).item()
            in_proj = next(p for p in s.parameters()
                           if p.ndim == 2 and p.shape[0] == 3 * p.shape[1])
            W_K = in_proj.data[d:2*d].detach()
            stats = _gram_stats(W_K, lam_for_stats)
            log.append((step, mse, stats))

    return log


def print_log(name, log, target=CONFIG.target_mse):
    hdr = (f"{'step':>5}  {'MSE':>10}  "
           f"{'lam_nat':>9}  {'sv_min':>7}  {'C_inv_max':>9}")
    bar = "=" * len(hdr)
    print(f"\n{bar}\n  {name}\n{bar}")
    print(hdr)
    print("-" * len(hdr))
    for step, mse, s in log:
        flag = "  <-- PASS" if mse < target else ""
        print(f"{step:>5}  {mse:>10.3e}  "
              f"{s['lam_nat']:>9.3f}  {s['sv_min']:>7.4f}  {s['C_inv_max']:>9.4f}{flag}")


# ---------------------------------------------------------------------------
# Section 1: Quick sweep — all 6 Pythia layers as teacher
# ---------------------------------------------------------------------------

def section_layer_sweep(device, pythia, cm_mod, adapt_mod):
    print("\n" + "=" * 70)
    print("SECTION 1: All 6 Pythia layers as teacher (quick sweep, step 400 only)")
    print("=" * 70)

    checkpoints = {400}
    configs = [
        ("AdamW",        lambda p: torch.optim.AdamW(p, lr=3e-4),          0.01),
        ("CM  λ=0.01",   lambda p: cm_mod.Submission(p, lr=1.5, lambda_reg=0.01), 0.01),
        ("CM  λ=1.0",    lambda p: cm_mod.Submission(p, lr=1.5, lambda_reg=1.0),  1.0),
        ("Adaptive ε=1", lambda p: adapt_mod.Submission(p, lr=1.5, adaptive=True, eps=1.0, lam_min=0.1), 1.0),
    ]

    print(f"\n{'layer':>5}  {'lam_nat(K)':>10}  ", end="")
    for name, _, _ in configs:
        print(f"  {name:>14}", end="")
    print()
    print("-" * (20 + 16 * len(configs)))

    results = {}
    for li in range(pythia.config.num_hidden_layers):
        teacher = build_pythia_teacher(li, pythia).to(device)

        # lam_nat of teacher's W_K
        in_p = next(p for p in teacher.parameters()
                    if p.ndim == 2 and p.shape[0] == 3 * CONFIG.embed_dim)
        W_K_t = in_p.data[CONFIG.embed_dim:2*CONFIG.embed_dim].float()
        lam_nat_t = (W_K_t.T @ W_K_t).trace().item() / CONFIG.embed_dim

        datasets = build_datasets(teacher, device)
        row = [li, lam_nat_t]
        print(f"{li:>5}  {lam_nat_t:>10.4f}  ", end="", flush=True)

        for name, make_opt, lam in configs:
            torch.manual_seed(CONFIG.student_seed)
            s = build_student(CONFIG).to(device)
            log = run_training(s, make_opt, datasets, device, checkpoints, lam)
            mse = log[-1][1]
            row.append(mse)
            flag = "*" if mse < CONFIG.target_mse else " "
            print(f"  {mse:>13.3e}{flag}", end="", flush=True)

        print()
        results[li] = row

    return results


# ---------------------------------------------------------------------------
# Section 2: Full spectral flow for the most interesting layer
# ---------------------------------------------------------------------------

def section_spectral_flow(device, pythia, cm_mod, adapt_mod, layer_idx: int):
    print("\n" + "=" * 70)
    print(f"SECTION 2: Spectral flow — Pythia layer {layer_idx} teacher")
    print("=" * 70)

    teacher = build_pythia_teacher(layer_idx, pythia)
    datasets = build_datasets(teacher, device)
    checkpoints = {1, 25, 100, 200, 300, 400}

    configs = [
        ("AdamW",
         lambda p: torch.optim.AdamW(p, lr=3e-4), 0.01),
        ("CM  λ=0.01  lr=1.5",
         lambda p: cm_mod.Submission(p, lr=1.5, lambda_reg=0.01), 0.01),
        ("CM  λ=1.0   lr=1.5",
         lambda p: cm_mod.Submission(p, lr=1.5, lambda_reg=1.0), 1.0),
        ("CM adaptive ε=1.0  lr=1.5",
         lambda p: adapt_mod.Submission(p, lr=1.5, adaptive=True, eps=1.0, lam_min=0.1), 1.0),
    ]

    for name, make_opt, lam in configs:
        torch.manual_seed(CONFIG.student_seed)
        s = build_student(CONFIG).to(device)
        log = run_training(s, make_opt, datasets, device, checkpoints, lam)
        print_log(name, log)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}  d={CONFIG.embed_dim}  max_steps={CONFIG.max_steps}")

    print("Loading Pythia-14m...")
    pythia = _load_pythia()
    _pythia_spectral_summary(pythia)

    cm_mod   = _load("submissions/comp_muon/submission.py")
    adapt_mod = _load("submissions/adaptive_cm/submission.py")

    t0 = time.monotonic()
    results = section_layer_sweep(device, pythia, cm_mod, adapt_mod)

    # Pick most interesting layer for full flow: highest lam_nat (layer 5)
    # and a mid-range layer (layer 3) for comparison
    for li in [5, 3]:
        section_spectral_flow(device, pythia, cm_mod, adapt_mod, li)

    print(f"\nTotal wall time: {time.monotonic()-t0:.0f}s")
    print("\n--- Key questions ---")
    print("  1. Does CM λ=0.01 still cause catastrophic feedback with Pythia weights?")
    print("  2. Does λ=1.0 remain near-optimal, or does smaller lam_nat favor smaller λ?")
    print("  3. Does adaptive ε=1.0 track the Pythia weight scale better than fixed λ=1.0?")
    print("  4. Which Pythia layer is hardest to learn — highest lam_nat (layer 5)?")


if __name__ == "__main__":
    main()
