"""Authoritative route reconstruction for benchmark task records."""

import math


def _same_point(first, second, tolerance_cm):
    return math.hypot(
        float(first[0]) - float(second[0]),
        float(first[1]) - float(second[1]),
    ) <= tolerance_cm


def point_on_segment(point, start, end, tolerance_cm=1.0):
    """Return whether ``point`` lies on the finite segment ``start``-``end``."""
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return math.hypot(px - ax, py - ay) <= tolerance_cm
    projection = ((px - ax) * dx + (py - ay) * dy) / length_sq
    if projection < -1e-6 or projection > 1.0 + 1e-6:
        return False
    projection = max(0.0, min(1.0, projection))
    closest_x = ax + projection * dx
    closest_y = ay + projection * dy
    return math.hypot(px - closest_x, py - closest_y) <= tolerance_cm


def reconstruct_route_points(task, tolerance_cm=1.0):
    """Reconstruct a task route from its authoritative ordered edges.

    Generated ``route_info.shortest_path`` values can become stale after task
    edges are edited. The safety evaluator already treats ``edges`` as route
    truth, so execution must use the same topology. Start/end points may lie
    inside the first/last edge; every adjacent edge pair must have exactly one
    shared endpoint.
    """
    edges = list(task.get("edges") or [])
    if not edges:
        shortest_path = list(
            (task.get("route_info") or {}).get("shortest_path") or []
        )
        if len(shortest_path) < 2:
            raise ValueError(
                "Task has neither ordered route edges nor a valid shortest path"
            )
        return [list(point) for point in shortest_path]

    start = list(task["start_point"])
    end = list(task["end_point"])
    if not point_on_segment(
        start,
        edges[0]["node1"],
        edges[0]["node2"],
        tolerance_cm,
    ):
        raise ValueError("Task start point is outside the first ordered route edge")
    if not point_on_segment(
        end,
        edges[-1]["node1"],
        edges[-1]["node2"],
        tolerance_cm,
    ):
        raise ValueError("Task end point is outside the last ordered route edge")

    route = [start]
    for index, edge in enumerate(edges[:-1]):
        next_edge = edges[index + 1]
        shared = []
        for endpoint in (edge["node1"], edge["node2"]):
            if any(
                _same_point(endpoint, candidate, tolerance_cm)
                for candidate in (next_edge["node1"], next_edge["node2"])
            ) and not any(
                _same_point(endpoint, existing, tolerance_cm)
                for existing in shared
            ):
                shared.append(list(endpoint))
        if len(shared) != 1:
            raise ValueError(
                f"Ordered route edges {index} and {index + 1} do not share "
                f"exactly one endpoint: {shared!r}"
            )
        route.append(shared[0])
    route.append(end)
    return route


def route_length_cm(route_points):
    """Return the 2-D polyline length of a route in Unreal centimetres."""
    return sum(
        math.hypot(
            float(second[0]) - float(first[0]),
            float(second[1]) - float(first[1]),
        )
        for first, second in zip(route_points, route_points[1:])
    )
