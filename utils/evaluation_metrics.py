"""Paper-aligned evaluation metrics for SimWorld-RealTime rollouts."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Mapping, Sequence


NOMINAL_WALKING_SPEED_CM_S = 200.0


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def percentile(values: Iterable[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    ordered = sorted(_number(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * min(100.0, max(0.0, percentile_value)) / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_metrics(decision_trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize model-response latency and simulated inference exposure."""
    response_latencies = [
        _number(item.get("response_time_seconds"))
        for item in decision_trace
        if item.get("raw_model_response") is not None
        and not str(item.get("feedback") or "").lower().startswith("parse error")
        and item.get("response_time_seconds") is not None
    ]
    modeled_latencies = [
        _number(item.get("modeled_thinking_latency_seconds"))
        for item in decision_trace
        if item.get("modeled_thinking_latency_seconds") is not None
    ]
    return {
        "valid_response_count": len(response_latencies),
        "mean_response_seconds": (
            statistics.fmean(response_latencies) if response_latencies else 0.0
        ),
        "median_response_seconds": percentile(response_latencies, 50),
        "p90_response_seconds": percentile(response_latencies, 90),
        "p95_response_seconds": percentile(response_latencies, 95),
        "cumulative_response_seconds": sum(response_latencies),
        "cumulative_simulated_inference_exposure_seconds": sum(modeled_latencies),
        "response_samples_seconds": response_latencies,
    }


def conflict_vehicle_metrics(
    event_groups: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Compute impact probability conditional on a valid conflict launch."""
    events = [event for group in event_groups for event in group]
    valid_launches = [
        event
        for event in events
        if event.get("disposition") == "launched"
        or (
            event.get("disposition") is None
            and event.get("vehicle_id") is not None
        )
    ]
    impacts = sum(bool(event.get("collision_triggered")) for event in valid_launches)
    return {
        "trigger_count": len(events),
        "valid_launch_count": len(valid_launches),
        "impact_count": impacts,
        "impact_given_valid_launch": (
            impacts / len(valid_launches) if valid_launches else None
        ),
    }


def episode_metrics(
    *,
    success: bool,
    shortest_path_length_cm: float,
    traveled_path_length_cm: float,
    sim_time_seconds: float,
    collision_count: int,
    passive_collision_count: int,
    fall_count: int,
    oil_count: int,
    water_count: int,
    red_light_violation_count: int,
    illegal_crossing_violation_count: int,
    decision_trace: Sequence[Mapping[str, Any]],
    conflict_event_groups: Sequence[Sequence[Mapping[str, Any]]],
    total_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    parse_error_count: int,
    invalid_decision_count: int,
    timeout_count: int = 0,
    collision_type_counts: Mapping[str, Any] | None = None,
    passive_collision_type_counts: Mapping[str, Any] | None = None,
    vehicle_collision_count: int = 0,
) -> dict[str, Any]:
    """Build the benchmark's per-episode paper metrics."""
    shortest = max(0.0, _number(shortest_path_length_cm))
    traveled = max(0.0, _number(traveled_path_length_cm))
    sim_time = max(0.0, _number(sim_time_seconds))
    success_indicator = int(bool(success))
    spl = (
        success_indicator * shortest / max(shortest, traveled)
        if shortest > 0.0
        else 0.0
    )
    optimal_time = shortest / NOMINAL_WALKING_SPEED_CM_S
    ste = (
        success_indicator * optimal_time / sim_time
        if sim_time > 0.0
        else 0.0
    )
    collision_count = max(0, int(collision_count or 0))
    passive_collision_count = max(0, int(passive_collision_count or 0))
    active_collision_count = max(0, collision_count - passive_collision_count)
    supplied_type_counts = collision_type_counts or {}
    supplied_passive_type_counts = passive_collision_type_counts or {}
    collision_counts_by_type = {
        "human": max(0, int(supplied_type_counts.get("human", 0) or 0)),
        "object": max(0, int(supplied_type_counts.get("object", 0) or 0)),
        "building": max(0, int(supplied_type_counts.get("building", 0) or 0)),
        "vehicle": max(0, int(vehicle_collision_count or 0)),
    }
    passive_counts_by_type = {
        "human": max(
            0, int(supplied_passive_type_counts.get("human", 0) or 0)
        ),
        "object": max(
            0, int(supplied_passive_type_counts.get("object", 0) or 0)
        ),
        "building": max(
            0, int(supplied_passive_type_counts.get("building", 0) or 0)
        ),
    }
    passive_counts_by_type["vehicle"] = max(
        0,
        passive_collision_count - sum(passive_counts_by_type.values()),
    )
    active_counts_by_type = {
        key: max(0, collision_counts_by_type[key] - passive_counts_by_type[key])
        for key in collision_counts_by_type
    }
    unsafe_event_count = sum(
        max(0, int(value or 0))
        for value in (
            collision_count,
            fall_count,
            oil_count,
            water_count,
            red_light_violation_count,
            illegal_crossing_violation_count,
        )
    )
    safe_episode = unsafe_event_count == 0
    traveled_m = traveled / 100.0
    completion_tokens = max(0, int(completion_tokens or 0))
    total_tokens = max(0, int(total_tokens or 0))
    reasoning_tokens = max(0, int(reasoning_tokens or 0))
    return {
        "schema_version": "paper_metrics_v1",
        "success_indicator": success_indicator,
        "shortest_path_length_cm": shortest,
        "traveled_path_length_cm": traveled,
        "spl": spl,
        "optimal_time_seconds": optimal_time,
        "sim_time_seconds": sim_time,
        "simulation_time_efficiency": ste,
        "collision_episode_indicator": int(collision_count > 0),
        "collision_count": collision_count,
        "active_collision_count": active_collision_count,
        "passive_collision_count": passive_collision_count,
        "collision_type_counts": collision_counts_by_type,
        "active_collision_type_counts": active_counts_by_type,
        "passive_collision_type_counts": passive_counts_by_type,
        "collisions_per_100m": (
            100.0 * collision_count / traveled_m if traveled_m > 0.0 else 0.0
        ),
        "unsafe_event_count": unsafe_event_count,
        "safe_episode_indicator": int(safe_episode),
        "safe_success_indicator": int(bool(success) and safe_episode),
        "conflict_vehicle": conflict_vehicle_metrics(conflict_event_groups),
        "latency": latency_metrics(decision_trace),
        "tokens": {
            "prompt": max(0, total_tokens - completion_tokens),
            "completion": completion_tokens,
            "reasoning": reasoning_tokens,
            "total": total_tokens,
        },
        "errors": {
            "parse": max(0, int(parse_error_count or 0)),
            "invalid_decision": max(0, int(invalid_decision_count or 0)),
            "timeout": max(0, int(timeout_count or 0)),
        },
    }


def aggregate_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate episode ``benchmark_metrics`` dictionaries for a suite."""
    metrics = [
        item.get("benchmark_metrics", item.get("evaluation_metrics", item))
        for item in episodes
    ]
    count = len(metrics)
    if not count:
        return {
            "schema_version": "paper_metrics_v1",
            "episode_count": 0,
            "success_rate": 0.0,
            "spl": 0.0,
            "simulation_time_efficiency": 0.0,
            "collision_episode_rate": 0.0,
            "collision_count": 0,
            "mean_collision_count": 0.0,
            "active_collision_count": 0,
            "mean_active_collision_count": 0.0,
            "passive_collision_count": 0,
            "mean_passive_collision_count": 0.0,
            "collision_type_counts": {
                "human": 0,
                "object": 0,
                "building": 0,
                "vehicle": 0,
            },
            "mean_collision_type_counts": {
                "human": 0.0,
                "object": 0.0,
                "building": 0.0,
                "vehicle": 0.0,
            },
            "active_collision_type_counts": {
                "human": 0,
                "object": 0,
                "building": 0,
                "vehicle": 0,
            },
            "passive_collision_type_counts": {
                "human": 0,
                "object": 0,
                "building": 0,
                "vehicle": 0,
            },
            "collisions_per_100m": 0.0,
            "safe_success_rate": 0.0,
        }

    total_collisions = sum(int(item.get("collision_count", 0) or 0) for item in metrics)
    collision_types = ("human", "object", "building", "vehicle")
    collision_type_totals = {
        key: sum(
            int((item.get("collision_type_counts") or {}).get(key, 0) or 0)
            for item in metrics
        )
        for key in collision_types
    }
    active_collisions = sum(
        int(item.get("active_collision_count", 0) or 0) for item in metrics
    )
    passive_collisions = sum(
        int(item.get("passive_collision_count", 0) or 0) for item in metrics
    )
    active_collision_type_totals = {
        key: sum(
            int((item.get("active_collision_type_counts") or {}).get(key, 0) or 0)
            for item in metrics
        )
        for key in collision_types
    }
    passive_collision_type_totals = {
        key: sum(
            int((item.get("passive_collision_type_counts") or {}).get(key, 0) or 0)
            for item in metrics
        )
        for key in collision_types
    }
    total_traveled_m = sum(
        _number(item.get("traveled_path_length_cm")) / 100.0 for item in metrics
    )
    response_samples = []
    total_response_exposure = 0.0
    total_simulated_exposure = 0.0
    valid_response_count = 0
    valid_launches = 0
    impacts = 0
    token_totals = {"prompt": 0, "completion": 0, "reasoning": 0, "total": 0}
    error_totals = {"parse": 0, "invalid_decision": 0, "timeout": 0}
    for item in metrics:
        latency = item.get("latency") or {}
        sample_count = int(latency.get("valid_response_count", 0) or 0)
        valid_response_count += sample_count
        total_response_exposure += _number(latency.get("cumulative_response_seconds"))
        total_simulated_exposure += _number(
            latency.get("cumulative_simulated_inference_exposure_seconds")
        )
        samples = latency.get("response_samples_seconds")
        if isinstance(samples, list):
            response_samples.extend(_number(value) for value in samples)
        else:
            response_samples.extend(
                [_number(latency.get("mean_response_seconds"))] * sample_count
            )
        conflict = item.get("conflict_vehicle") or {}
        valid_launches += int(conflict.get("valid_launch_count", 0) or 0)
        impacts += int(conflict.get("impact_count", 0) or 0)
        for key in token_totals:
            token_totals[key] += int((item.get("tokens") or {}).get(key, 0) or 0)
        for key in error_totals:
            error_totals[key] += int((item.get("errors") or {}).get(key, 0) or 0)

    mean = lambda key: sum(_number(item.get(key)) for item in metrics) / count
    return {
        "schema_version": "paper_metrics_v1",
        "episode_count": count,
        "success_rate": mean("success_indicator"),
        "spl": mean("spl"),
        "simulation_time_efficiency": mean("simulation_time_efficiency"),
        "collision_episode_rate": mean("collision_episode_indicator"),
        "collision_count": total_collisions,
        "mean_collision_count": total_collisions / count,
        "collision_type_counts": collision_type_totals,
        "mean_collision_type_counts": {
            key: value / count for key, value in collision_type_totals.items()
        },
        "active_collision_count": active_collisions,
        "mean_active_collision_count": active_collisions / count,
        "passive_collision_count": passive_collisions,
        "mean_passive_collision_count": passive_collisions / count,
        "active_collision_type_counts": active_collision_type_totals,
        "passive_collision_type_counts": passive_collision_type_totals,
        "collisions_per_100m": (
            100.0 * total_collisions / total_traveled_m
            if total_traveled_m > 0.0
            else 0.0
        ),
        "safe_success_rate": mean("safe_success_indicator"),
        "conflict_vehicle_impact_given_valid_launch": (
            impacts / valid_launches if valid_launches else None
        ),
        "conflict_vehicle_valid_launch_count": valid_launches,
        "conflict_vehicle_impact_count": impacts,
        "latency": {
            "valid_response_count": valid_response_count,
            "mean_response_seconds": (
                total_response_exposure / valid_response_count
                if valid_response_count
                else 0.0
            ),
            "median_response_seconds": percentile(response_samples, 50),
            "p90_response_seconds": percentile(response_samples, 90),
            "p95_response_seconds": percentile(response_samples, 95),
            "cumulative_response_seconds": total_response_exposure,
            "cumulative_simulated_inference_exposure_seconds": total_simulated_exposure,
        },
        "tokens": token_totals,
        "errors": error_totals,
    }
