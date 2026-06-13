from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

from run_eval import run_benchmark
from task import CONFIG


def discover_submissions(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/submission.py") if path.is_file())


def format_mse(value: float) -> str:
    return f"{value:.6g}"


def format_seconds(value: float) -> str:
    return f"{value:.3f}"


def build_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# caffeine leaderboard",
        "",
        f"Ranked by final eval MSE after {CONFIG.max_steps} fixed training steps. Wall time is training only.",
        "",
        "| Rank | Submission | Final Eval MSE | Training Wall Time (s) | Last Train Loss |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, result in enumerate(results, start=1):
        lines.append(
            "| "
            f"{rank} | "
            f"{result['submission']} | "
            f"{format_mse(result['final_eval_mse'])} | "
            f"{format_seconds(result['training_wall_time_s'])} | "
            f"{format_mse(result['last_train_loss'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ranked_by": ["final_eval_mse", "training_wall_time_s"],
        "max_steps": CONFIG.max_steps,
        "train_samples": CONFIG.train_samples,
        "eval_samples": CONFIG.eval_samples,
        "batch_size": CONFIG.batch_size,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "results": results,
    }
    (output_dir / "leaderboard.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "leaderboard.md").write_text(build_markdown(results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all caffeine submissions and rank the leaderboard.")
    parser.add_argument("--submissions-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--require-arm64", action="store_true")
    args = parser.parse_args()

    if args.require_arm64 and platform.machine() != "arm64":
        raise SystemExit(f"official benchmark requires arm64, got {platform.machine()!r}")

    submission_paths = discover_submissions(args.submissions_dir)
    if not submission_paths:
        raise SystemExit(f"no submissions found under {args.submissions_dir}")

    results = [run_benchmark(path, CONFIG) for path in submission_paths]
    results.sort(key=lambda result: (result["final_eval_mse"], result["training_wall_time_s"]))
    write_outputs(results, args.output_dir)
    print(build_markdown(results), end="")


if __name__ == "__main__":
    main()
