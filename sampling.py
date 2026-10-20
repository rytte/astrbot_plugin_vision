from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    timestamp: float
    reason: str
    score: float = 0.0


def maximum_gap(timestamps: list[float], duration: float) -> float:
    ordered = sorted({0.0, duration, *timestamps})
    return max((right - left for left, right in zip(ordered, ordered[1:])), default=0.0)


def overview_candidates(
    duration: float,
    last_timestamp: float,
    interval: float,
    budget: int,
    scenes: list[Candidate],
) -> tuple[list[Candidate], float]:
    last_timestamp = max(0.0, min(duration, last_timestamp))
    segment_count = max(1, math.ceil(last_timestamp / interval))
    if segment_count + 1 > budget:
        segment_count = budget - 1
        effective_interval = last_timestamp / segment_count
        times = [last_timestamp * index / segment_count for index in range(budget)]
    else:
        effective_interval = interval
        times = [index * interval for index in range(segment_count)]
        times.append(last_timestamp)
    times = sorted(set(round(timestamp, 6) for timestamp in times))
    candidates = [
        Candidate(timestamp, "boundary" if index in (0, len(times) - 1) else "interval")
        for index, timestamp in enumerate(times)
    ]
    occupied = set()
    for candidate in sorted(scenes, key=lambda item: item.score, reverse=True):
        if not 0 <= candidate.timestamp <= last_timestamp:
            continue
        matching = next(
            (
                index
                for index, item in enumerate(candidates)
                if abs(candidate.timestamp - item.timestamp) < 0.04
            ),
            None,
        )
        if matching is not None:
            existing = candidates[matching]
            if existing.reason != "boundary":
                candidates[matching] = Candidate(
                    existing.timestamp, "scene_change", candidate.score
                )
            continue
        if len(candidates) >= budget:
            continue
        bucket = int(candidate.timestamp / max(effective_interval, 1.0))
        if bucket in occupied:
            continue
        occupied.add(bucket)
        candidates.append(candidate)
    return sorted(candidates, key=lambda item: item.timestamp), effective_interval


def inspect_timestamps(
    duration: float,
    last_timestamp: float,
    start_sec: float | None,
    end_sec: float | None,
    timestamps_sec: list[float] | None,
    max_frames: int,
) -> tuple[list[float], dict]:
    if timestamps_sec is not None:
        if start_sec is not None or end_sec is not None:
            raise ValueError("时间范围与 timestamps_sec 不能同时提供")
        if not isinstance(timestamps_sec, list) or not timestamps_sec:
            raise ValueError("timestamps_sec 必须是非空数字列表")
        if len(timestamps_sec) > max_frames:
            raise ValueError(f"指定时间点数量不能超过本次帧数上限 {max_frames}")
        points = timestamps_sec
        request = {"timestamps_sec": timestamps_sec}
    else:
        if start_sec is None or end_sec is None:
            raise ValueError("请提供 start_sec、end_sec，或 timestamps_sec")
        points = [start_sec, end_sec]
        request = {"start_sec": start_sec, "end_sec": end_sec}
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 <= value <= duration
        for value in points
    ):
        raise ValueError(f"时间必须是 0～{duration:.3f} 秒之间的有限数字")
    if timestamps_sec is not None:
        return sorted(set(float(value) for value in points)), request
    if start_sec > end_sec:
        raise ValueError("start_sec 不能大于 end_sec")
    if start_sec == end_sec or max_frames == 1:
        return [float(start_sec)], request
    endpoint = max(float(start_sec), min(float(end_sec), last_timestamp))
    return [
        float(start_sec) + (endpoint - start_sec) * index / (max_frames - 1)
        for index in range(max_frames)
    ], request
