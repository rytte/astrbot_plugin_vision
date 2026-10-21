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
    *,
    trace: list[dict] | None = None,
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
        decision = {"timestamp": candidate.timestamp, "score": candidate.score}

        def record(action: str, **details) -> None:
            if trace is not None:
                trace.append({**decision, "action": action, **details})

        if not 0 <= candidate.timestamp <= last_timestamp:
            record("outside", explanation="超出可抽帧时间范围")
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
            record(
                "matched",
                target_sec=existing.timestamp,
                explanation="与已有候选相距小于 0.04 秒；保留时间点，边界帧保持边界标记，其余标记为场景变化",
            )
            continue
        if len(candidates) >= budget:
            record("budget", explanation="帧数预算已用满")
            continue
        bucket = int(candidate.timestamp / max(effective_interval, 1.0))
        if bucket in occupied:
            record("window", explanation="该时间窗口已有更高分的场景候选")
            continue
        occupied.add(bucket)
        candidates.append(candidate)
        record("added", explanation="加入场景候选")
    return sorted(candidates, key=lambda item: item.timestamp), effective_interval


class InspectArgumentError(ValueError):
    """回看参数不合法，尚未执行抽帧。"""


def inspect_timestamps(
    duration: float,
    last_timestamp: float,
    mode: str,
    times_sec: list[float],
    max_frames: int,
) -> tuple[list[float], dict]:
    if mode not in ("range", "timestamps"):
        raise InspectArgumentError(
            'mode 必须是 "range" 或 "timestamps"；'
            "range 的 times_sec 为 [起始秒, 结束秒]，"
            "timestamps 的 times_sec 为非空时间点列表"
        )
    if not isinstance(times_sec, list) or not times_sec:
        raise InspectArgumentError("times_sec 必须是非空数字列表")
    if mode == "range" and len(times_sec) != 2:
        raise InspectArgumentError("range 模式的 times_sec 必须恰好包含起始秒和结束秒")
    if mode == "timestamps" and len(times_sec) > max_frames:
        raise InspectArgumentError(f"指定时间点数量不能超过本次帧数上限 {max_frames}")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 <= value <= duration
        for value in times_sec
    ):
        raise InspectArgumentError(f"时间必须是 0～{duration:.3f} 秒之间的有限数字")
    request = {"mode": mode, "times_sec": times_sec}
    if mode == "timestamps":
        return sorted(set(float(value) for value in times_sec)), request
    start_sec, end_sec = times_sec
    if start_sec > end_sec:
        raise InspectArgumentError("range 模式的起始秒不能大于结束秒")
    if start_sec == end_sec or max_frames == 1:
        return [float(start_sec)], request
    endpoint = max(float(start_sec), min(float(end_sec), last_timestamp))
    return [
        float(start_sec) + (endpoint - start_sec) * index / (max_frames - 1)
        for index in range(max_frames)
    ], request
