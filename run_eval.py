from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data import build_batch_indices, build_eval_dataset, build_student, build_teacher, build_train_dataset
from task import CONFIG, TaskConfig


def import_submission(path: Path) -> type[torch.optim.Optimizer]:
    spec = importlib.util.spec_from_file_location("caffeine_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import submission: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    submission = getattr(module, "Submission", None)
    if not isinstance(submission, type):
        raise TypeError("submission must define a class named Submission")
    if not issubclass(submission, torch.optim.Optimizer):
        raise TypeError("Submission must subclass torch.optim.Optimizer")
    return submission


@torch.no_grad()
def evaluate(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    model.eval()
    return float(F.mse_loss(model(inputs), targets).item())


@torch.no_grad()
def parameter_mse(model: torch.nn.Module, teacher: torch.nn.Module) -> float:
    total = 0.0
    count = 0
    for parameter, teacher_parameter in zip(model.parameters(), teacher.parameters(), strict=True):
        diff = parameter.detach() - teacher_parameter.detach()
        total += float(diff.square().sum().item())
        count += diff.numel()
    return total / count


def run_benchmark(submission_path: Path, config: TaskConfig) -> dict[str, Any]:
    optimizer_cls = import_submission(submission_path)

    torch.set_num_threads(1)
    teacher = build_teacher(config)
    train_data = build_train_dataset(teacher, config)
    eval_data = build_eval_dataset(teacher, config)
    batch_indices = build_batch_indices(config)
    model = build_student(config)
    optimizer = optimizer_cls(model.parameters())

    initial_eval_mse = evaluate(model, eval_data.inputs, eval_data.targets)
    initial_weight_mse = parameter_mse(model, teacher)
    best_eval_mse = initial_eval_mse
    best_weight_mse = initial_weight_mse
    passed = initial_weight_mse <= config.target_weight_mse
    pass_step = 0 if passed else None
    pass_duration_s = 0.0 if passed else None

    start = time.perf_counter()
    last_loss = float("nan")
    for step in range(1, config.max_steps + 1):
        model.train()
        indices = batch_indices[step - 1]
        inputs = train_data.inputs[indices]
        targets = train_data.targets[indices]

        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

        if step % config.eval_every == 0 or step == config.max_steps:
            eval_mse = evaluate(model, eval_data.inputs, eval_data.targets)
            weight_mse = parameter_mse(model, teacher)
            best_eval_mse = min(best_eval_mse, eval_mse)
            best_weight_mse = min(best_weight_mse, weight_mse)
            if weight_mse <= config.target_weight_mse:
                passed = True
                pass_step = step
                pass_duration_s = time.perf_counter() - start
                break

    total_duration_s = time.perf_counter() - start
    final_eval_mse = evaluate(model, eval_data.inputs, eval_data.targets)
    final_weight_mse = parameter_mse(model, teacher)
    status = "pass" if passed else "fail"

    return {
        "status": status,
        "submission": submission_path.parent.name if submission_path.name == "submission.py" else submission_path.stem,
        "submission_path": str(submission_path),
        "duration_s": pass_duration_s if pass_duration_s is not None else total_duration_s,
        "total_duration_s": total_duration_s,
        "steps": pass_step if pass_step is not None else config.max_steps,
        "max_steps": config.max_steps,
        "eval_every": config.eval_every,
        "initial_eval_mse": initial_eval_mse,
        "initial_weight_mse": initial_weight_mse,
        "final_eval_mse": final_eval_mse,
        "final_weight_mse": final_weight_mse,
        "best_eval_mse": best_eval_mse,
        "best_weight_mse": best_weight_mse,
        "target_mse": config.target_mse,
        "target_weight_mse": config.target_weight_mse,
        "last_train_loss": last_loss,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "date_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the caffeine optimizer benchmark.")
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("submissions/adamw/submission.py"),
        help="Python file defining Submission(torch.optim.Optimizer).",
    )
    parser.add_argument("--results-json", type=Path, default=None)
    parser.add_argument("--require-arm64", action="store_true")
    args = parser.parse_args()

    if args.require_arm64 and platform.machine() != "arm64":
        raise SystemExit(f"official benchmark requires arm64, got {platform.machine()!r}")

    result = run_benchmark(args.submission, CONFIG)
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(text)

    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
