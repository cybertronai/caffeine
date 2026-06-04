from __future__ import annotations

from collections.abc import Callable

from task import MQAR_TRACK, RANDOM_TEACHER_TRACK, SINGLE_AR_TRACK, TRACKS
from tracks.base import BenchmarkTrack
from tracks.random_teacher import RandomTeacherTrack
from tracks.token_recall import multi_query_associative_recall_track, single_query_associative_recall_track


TrackFactory = Callable[[], BenchmarkTrack]


TRACK_FACTORIES: dict[str, TrackFactory] = {
    RANDOM_TEACHER_TRACK: RandomTeacherTrack,
    SINGLE_AR_TRACK: single_query_associative_recall_track,
    MQAR_TRACK: multi_query_associative_recall_track,
}


def track_for_name(name: str) -> BenchmarkTrack:
    try:
        return TRACK_FACTORIES[name]()
    except KeyError as exc:
        raise ValueError(f"unknown track: {name}") from exc


__all__ = [
    "BenchmarkTrack",
    "TRACKS",
    "track_for_name",
]
