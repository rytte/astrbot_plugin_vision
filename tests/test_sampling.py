import math

import pytest
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.sampling import (
    Candidate,
    InspectArgumentError,
    inspect_timestamps,
    maximum_gap,
    overview_candidates,
)


def test_defaults_match_design():
    settings = Settings.from_mapping({})
    assert settings.overview_interval == 4
    assert settings.overview_max_frames == 200
    assert settings.overview_max_edge == 640
    assert settings.overview_model_enabled is False
    assert settings.overview_provider_id == ""


@pytest.mark.parametrize(
    "value, expected", [("  ", ""), (" vision-model ", "vision-model")]
)
def test_overview_provider_setting_is_trimmed(value, expected):
    assert (
        Settings.from_mapping({"overview_provider_id": value}).overview_provider_id
        == expected
    )


@pytest.mark.parametrize("duration", [0.05, 3, 10, 600, 1200, 7200])
def test_overview_preserves_whole_timeline_within_budget(duration):
    scenes = [
        Candidate(duration * index / 500, "scene_change", index / 500)
        for index in range(500)
    ]
    last = max(0, duration - 0.04)
    candidates, interval = overview_candidates(duration, last, 4, 200, scenes)
    times = [candidate.timestamp for candidate in candidates]
    assert len(candidates) <= 200
    assert times == sorted(set(times))
    assert times[0] == 0
    assert times[-1] == pytest.approx(last)
    assert maximum_gap(times, duration) <= interval + 0.04 + 1e-6
    if duration == 1200:
        assert interval > 4


def test_scene_additions_are_limited_per_window():
    scenes = [
        Candidate(1 + index / 100, "scene_change", index / 100) for index in range(100)
    ]
    candidates, _ = overview_candidates(20, 19.9, 4, 30, scenes)
    keys = [candidate for candidate in candidates if candidate.reason == "scene_change"]
    assert len(keys) == 1
    assert keys[0].timestamp == pytest.approx(1.99)


def test_scene_matching_interval_is_tagged_as_key_even_at_budget():
    candidates, _ = overview_candidates(
        12, 11.9, 4, 4, [Candidate(3.966667, "scene_change", 0.9)]
    )
    assert len(candidates) == 4
    assert candidates[1].reason == "scene_change"


@pytest.mark.parametrize(
    "values",
    [
        {"overview_interval": float("nan")},
        {"overview_max_frames": 1},
        {"overview_max_frames": 2.5},
        {"overview_max_frames": True},
        {"inspect_max_calls": 0},
        {"scene_threshold": 1},
        {"typo": 1},
        {"max_edge_limit": 128},
        {"overview_model_enabled": "true"},
        {"overview_model_enabled": 0},
        {"overview_model_enabled": None},
        {"overview_provider_id": None},
        {"overview_provider_id": 123},
        {"overview_provider_id": True},
        {"overview_provider_id": []},
    ],
)
def test_invalid_settings_rejected(values):
    with pytest.raises(ValueError):
        Settings.from_mapping(values)


def test_inspect_range_uses_short_window_and_endpoints():
    points, request = inspect_timestamps(100, 99.9, "range", [10, 14], 8)
    assert len(points) == 8
    assert points[0] == 10
    assert points[-1] == 14
    assert request == {"mode": "range", "times_sec": [10, 14]}


@pytest.mark.parametrize(
    "mode,times",
    [
        (None, None),
        ("auto", [0]),
        ("range", [10, 5]),
        ("range", [-1, 5]),
        ("range", [0, math.inf]),
        ("range", [0]),
        ("range", [0, 1, 2]),
        ("range", [0, True]),
        ("timestamps", []),
        ("timestamps", [True]),
        ("timestamps", [float("nan")]),
        ("timestamps", [101]),
        ("timestamps", list(range(9))),
        ("timestamps", "0,1"),
    ],
)
def test_invalid_inspect_requests_rejected(mode, times):
    with pytest.raises(InspectArgumentError):
        inspect_timestamps(100, 99.9, mode, times, 8)


def test_point_mode_deduplicates_and_sorts():
    points, request = inspect_timestamps(100, 99.9, "timestamps", [9, 3, 9], 8)
    assert points == [3, 9]
    assert request == {"mode": "timestamps", "times_sec": [9, 3, 9]}


@pytest.mark.parametrize("mode,times", [("range", [0, 0]), ("timestamps", [0])])
def test_zero_is_a_valid_explicit_timestamp(mode, times):
    points, _ = inspect_timestamps(100, 99.9, mode, times, 8)
    assert points == [0]
