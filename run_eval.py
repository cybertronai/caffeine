from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from task import CONFIG, DEFAULT_TRACK, TRACKS, TaskConfig
from tracks import BenchmarkTrack, track_for_name
from tracks.base import RunConfig
from tracks.random_teacher import RandomTeacherConfig, RandomTeacherTrack


def import_submission(path: Path) -> type[Any]:
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


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def random_teacher_track_from_config(config: TaskConfig) -> RandomTeacherTrack:
    return RandomTeacherTrack(
        RandomTeacherConfig(
            embed_dim=config.embed_dim,
            sequence_length=config.sequence_length,
            teacher_seed=config.teacher_seed,
            student_seed=config.student_seed,
            run=RunConfig(
                train_samples=config.train_samples,
                eval_samples=config.eval_samples,
                batch_size=config.batch_size,
                max_steps=config.max_steps,
                eval_every=config.max_steps,
                train_seed=config.train_seed,
                eval_seed=config.eval_seed,
                batch_seed=config.batch_seed,
            ),
        )
    )


def coerce_track(track_or_config: BenchmarkTrack | TaskConfig) -> BenchmarkTrack:
    if isinstance(track_or_config, TaskConfig):
        return random_teacher_track_from_config(track_or_config)
    return track_or_config


def finite_or_nan(metrics: dict[str, float], key: str) -> float:
    value = metrics.get(key, float("nan"))
    return value if math.isfinite(value) else float("nan")


def run_benchmark(submission_path: Path, track_or_config: BenchmarkTrack | TaskConfig) -> dict[str, Any]:
    optimizer_cls = import_submission(submission_path)
    track = coerce_track(track_or_config)
    run_config = track.run_config

    torch.set_num_threads(1)
    device = select_device()
    train_data = track.build_train_dataset()
    eval_data = track.build_eval_dataset()
    batch_indices = track.build_batch_indices().to(device)
    train_inputs = train_data.inputs.to(device)
    train_targets = train_data.targets.to(device)
    eval_inputs = eval_data.inputs.to(device)
    eval_targets = eval_data.targets.to(device)
    model = track.build_student().to(device)
    optimizer = optimizer_cls(model.parameters())

    initial_metrics = track.evaluate(model, eval_inputs, eval_targets)
    best_metrics = dict(initial_metrics)
    passed = track.metric_passed(initial_metrics)
    pass_step = 0 if passed else None

    synchronize_device(device)
    start = time.perf_counter()
    last_loss = float("nan")
    for step in range(1, track.max_steps + 1):
        model.train()
        indices = batch_indices[step - 1]
        inputs = train_inputs[indices]
        targets = train_targets[indices]

        optimizer.zero_grad(set_to_none=True)
        loss = track.loss(model, inputs, targets)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

        if step % track.eval_every == 0 or step == track.max_steps:
            eval_metrics = track.evaluate(model, eval_inputs, eval_targets)
            best_metrics = track.update_best(best_metrics, eval_metrics)
            if pass_step is None and track.metric_passed(eval_metrics):
                passed = True
                pass_step = step

    synchronize_device(device)
    training_wall_time_s = time.perf_counter() - start
    final_metrics = track.evaluate(model, eval_inputs, eval_targets)

    return {
        "submission": submission_path.parent.name if submission_path.name == "submission.py" else submission_path.stem,
        "submission_path": str(submission_path),
        "track": track.name,
        "status": "pass" if passed else "fail",
        "training_wall_time_s": training_wall_time_s,
        "duration_s": training_wall_time_s,
        "steps": pass_step if pass_step is not None else track.max_steps,
        "max_steps": track.max_steps,
        "eval_every": track.eval_every,
        "train_samples": run_config.train_samples,
        "eval_samples": run_config.eval_samples,
        "batch_size": run_config.batch_size,
        "initial_eval_mse": finite_or_nan(initial_metrics, "mse"),
        "final_eval_mse": finite_or_nan(final_metrics, "mse"),
        "best_eval_mse": finite_or_nan(best_metrics, "mse"),
        "initial_eval_accuracy": finite_or_nan(initial_metrics, "accuracy"),
        "final_eval_accuracy": finite_or_nan(final_metrics, "accuracy"),
        "best_eval_accuracy": finite_or_nan(best_metrics, "accuracy"),
        **track.target_metrics(),
        "last_train_loss": last_loss,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
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
    parser.add_argument(
        "--track",
        choices=TRACKS,
        default=DEFAULT_TRACK,
        help="Benchmark track to run.",
    )
    parser.add_argument("--require-arm64", action="store_true")
    args = parser.parse_args()

    if args.require_arm64 and platform.machine() != "arm64":
        raise SystemExit(f"official benchmark requires arm64, got {platform.machine()!r}")

    result = run_benchmark(args.submission, track_for_name(args.track))
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(text)


if __name__ == "__main__":
    main()
