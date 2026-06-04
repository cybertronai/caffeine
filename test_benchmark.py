from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from model import VanillaSelfAttention
from run_eval import import_submission, run_benchmark, select_device
from run_leaderboard import build_markdown, discover_submissions
from task import DEFAULT_TRACK, MQAR_TRACK, SINGLE_AR_TRACK
from tracks import track_for_name
from tracks.base import RunConfig
from tracks.random_teacher import RandomTeacherConfig, RandomTeacherTrack, build_random_teacher
from tracks.token_recall import TokenRecallConfig, TokenRecallTrack


def test_data_generation_is_deterministic() -> None:
    track = track_for_name(DEFAULT_TRACK)
    train1 = track.build_train_dataset()
    train2 = track.build_train_dataset()
    eval1 = track.build_eval_dataset()
    eval2 = track.build_eval_dataset()

    assert torch.equal(train1.inputs, train2.inputs)
    assert torch.equal(train1.targets, train2.targets)
    assert torch.equal(eval1.inputs, eval2.inputs)
    assert torch.equal(eval1.targets, eval2.targets)
    assert torch.equal(track.build_batch_indices(), track.build_batch_indices())


def test_student_initialization_is_deterministic() -> None:
    track = track_for_name(DEFAULT_TRACK)
    params1 = dict(track.build_student().named_parameters())
    params2 = dict(track.build_student().named_parameters())
    assert params1.keys() == params2.keys()
    for name in params1:
        assert torch.equal(params1[name], params2[name]), name


def test_teacher_uses_aggressive_uniform_initialization() -> None:
    track = track_for_name(DEFAULT_TRACK)
    assert isinstance(track, RandomTeacherTrack)
    teacher = build_random_teacher(track.config)
    student = track.build_student()
    assert isinstance(teacher, VanillaSelfAttention)
    teacher_params = torch.cat([parameter.flatten() for parameter in teacher.parameters()])
    student_params = torch.cat([parameter.flatten() for parameter in student.parameters()])

    assert teacher_params.min() >= -1.0
    assert teacher_params.max() <= 1.0
    assert teacher_params.std() > student_params.std() * 5.0


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
    track = RandomTeacherTrack(
        RandomTeacherConfig(
            embed_dim=8,
            sequence_length=4,
            target_mse=0.0,
            run=RunConfig(
                train_samples=128,
                eval_samples=64,
                batch_size=32,
                max_steps=2,
                eval_every=1,
            ),
        )
    )
    result = run_benchmark(Path("submissions/adamw/submission.py"), track)
    assert result["status"] == "fail"
    assert result["steps"] == 2
    assert result["max_steps"] == 2
    assert result["train_samples"] == 128
    assert result["eval_samples"] == 64
    assert result["batch_size"] == 32
    assert result["initial_eval_mse"] >= 0.0
    assert result["final_eval_mse"] >= 0.0
    assert result["training_wall_time_s"] >= 0.0
    assert result["device"] in {"cpu", "mps"}


def test_leaderboard_discovers_submissions() -> None:
    paths = discover_submissions(Path("submissions"))
    assert Path("submissions/adamw/submission.py") in paths


def test_leaderboard_markdown_uses_final_eval_mse() -> None:
    markdown = build_markdown([
        {
            "submission": "example",
            "final_eval_mse": 1.25,
            "training_wall_time_s": 0.5,
            "last_train_loss": 2.5,
        }
    ])
    assert "Final Eval MSE" in markdown
    assert "Training Wall Time" in markdown
    assert "example" in markdown


def test_single_ar_track_is_deterministic_and_runs() -> None:
    assert_token_track_is_deterministic_and_runs(SINGLE_AR_TRACK)


def test_mqar_track_is_deterministic_and_runs() -> None:
    assert_token_track_is_deterministic_and_runs(MQAR_TRACK)


def tiny_token_track(track_name: str):
    track = track_for_name(track_name)
    assert isinstance(track, TokenRecallTrack)
    config = TokenRecallConfig(
        num_queries=track.config.num_queries,
        target_accuracy=1.0,
        run=RunConfig(
            train_samples=128,
            eval_samples=64,
            batch_size=32,
            max_steps=2,
            eval_every=1,
        ),
    )
    return TokenRecallTrack(config=config, name=track.name)


def assert_token_track_is_deterministic_and_runs(track_name: str) -> None:
    track = tiny_token_track(track_name)
    train1 = track.build_train_dataset()
    train2 = track.build_train_dataset()
    assert torch.equal(train1.inputs, train2.inputs)
    assert torch.equal(train1.targets, train2.targets)

    result = run_benchmark(Path("submissions/adamw/submission.py"), track)
    assert result["track"] == track_name
    assert "final_eval_accuracy" in result


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
