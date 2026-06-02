from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from data import build_batch_indices, build_eval_dataset, build_student, build_teacher, build_train_dataset
from run_eval import import_submission, run_benchmark, select_device
from task import CONFIG, TaskConfig


def test_data_generation_is_deterministic() -> None:
    teacher1 = build_teacher(CONFIG)
    teacher2 = build_teacher(CONFIG)
    train1 = build_train_dataset(teacher1, CONFIG)
    train2 = build_train_dataset(teacher2, CONFIG)
    eval1 = build_eval_dataset(teacher1, CONFIG)
    eval2 = build_eval_dataset(teacher2, CONFIG)

    assert torch.equal(train1.inputs, train2.inputs)
    assert torch.equal(train1.targets, train2.targets)
    assert torch.equal(eval1.inputs, eval2.inputs)
    assert torch.equal(eval1.targets, eval2.targets)
    assert torch.equal(build_batch_indices(CONFIG), build_batch_indices(CONFIG))


def test_student_initialization_is_deterministic() -> None:
    params1 = dict(build_student(CONFIG).named_parameters())
    params2 = dict(build_student(CONFIG).named_parameters())
    assert params1.keys() == params2.keys()
    for name in params1:
        assert torch.equal(params1[name], params2[name]), name


def test_submission_must_be_optimizer_subclass() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_submission.py"
        path.write_text("class Submission:\n    pass\n")
        try:
            import_submission(path)
        except TypeError as exc:
            assert "subclass" in str(exc)
        else:
            raise AssertionError("bad submission should fail")


def test_select_device_prefers_mps_when_available() -> None:
    with patch.object(torch.backends.mps, "is_available", return_value=True):
        assert select_device().type == "mps"


def test_select_device_falls_back_to_cpu() -> None:
    with patch.object(torch.backends.mps, "is_available", return_value=False):
        assert select_device().type == "cpu"


def test_baseline_runs_on_tiny_config() -> None:
    config = TaskConfig(
        embed_dim=8,
        sequence_length=4,
        train_samples=128,
        eval_samples=64,
        batch_size=32,
        max_steps=2,
        eval_every=1,
        target_mse=0.0,
        target_relative_weight_error=0.0,
    )
    result = run_benchmark(Path("submissions/adamw/submission.py"), config)
    assert result["status"] == "fail"
    assert result["steps"] == 2
    assert result["initial_eval_mse"] >= 0.0
    assert result["initial_relative_weight_error"] >= 0.0
    assert result["final_eval_mse"] >= 0.0
    assert result["final_relative_weight_error"] >= 0.0
    assert result["device"] in {"cpu", "mps"}


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
