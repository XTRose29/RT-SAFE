#!/usr/bin/env python3
"""Print route-crosswalk signal coverage without starting Unreal Engine."""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMWORLD_ROOT = REPO_ROOT / "SimWorld"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SIMWORLD_ROOT))

Config = None
RTTrafficController = None
cal_waypoints = None
extend_control_point = None
get_bezier_points = None


def _load_offline_dependencies():
    """Load graph-only dependencies without polluting test import state."""
    global Config
    global RTTrafficController
    global cal_waypoints
    global extend_control_point
    global get_bezier_points

    if Config is not None:
        return

    # Loading ``simworld.__init__`` pulls optional asset-retrieval models that
    # are irrelevant to this geometry-only preflight.  Register the package
    # path only when the CLI actually runs; importing the pure route helper in
    # a test must not replace the production UnrealCV module in sys.modules.
    if "simworld" not in sys.modules:
        simworld_package = ModuleType("simworld")
        simworld_package.__path__ = [str(SIMWORLD_ROOT / "simworld")]
        sys.modules["simworld"] = simworld_package

    if "simworld.communicator.unrealcv" not in sys.modules:
        unrealcv_module = ModuleType("simworld.communicator.unrealcv")

        class OfflineUnrealCV:
            pass

        unrealcv_module.UnrealCV = OfflineUnrealCV
        sys.modules["simworld.communicator.unrealcv"] = unrealcv_module

    from simworld.config import Config as ConfigClass
    from base.rt_traffic_controller import RTTrafficController as ControllerClass
    from simworld.utils.traffic_utils import (
        cal_waypoints as calculate_waypoints,
        extend_control_point as extend_point,
        get_bezier_points as bezier_points,
    )

    Config = ConfigClass
    RTTrafficController = ControllerClass
    cal_waypoints = calculate_waypoints
    extend_control_point = extend_point
    get_bezier_points = bezier_points


def _point_on_segment(point, start, end, tolerance_cm=1.0):
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
    qx, qy = ax + projection * dx, ay + projection * dy
    return math.hypot(px - qx, py - qy) <= tolerance_cm


def reconstruct_route_points_from_task_edges(task, tolerance_cm=1.0):
    """Return the authoritative centerline implied by ordered task edges."""
    edges = list(task.get("edges") or [])
    if not edges:
        return [
            list(point)
            for point in (task.get("route_info") or {}).get("shortest_path", [])
        ]
    start = list(task["start_point"])
    end = list(task["end_point"])
    if not _point_on_segment(
        start, edges[0]["node1"], edges[0]["node2"], tolerance_cm
    ):
        raise ValueError("task start point is outside the first ordered edge")
    if not _point_on_segment(
        end, edges[-1]["node1"], edges[-1]["node2"], tolerance_cm
    ):
        raise ValueError("task end point is outside the last ordered edge")

    route = [start]
    for index, edge in enumerate(edges[:-1]):
        next_edge = edges[index + 1]
        shared = []
        for endpoint in (edge["node1"], edge["node2"]):
            if any(
                math.hypot(
                    float(endpoint[0]) - float(candidate[0]),
                    float(endpoint[1]) - float(candidate[1]),
                )
                <= tolerance_cm
                for candidate in (next_edge["node1"], next_edge["node2"])
            ):
                shared.append(list(endpoint))
        if len(shared) != 1:
            raise ValueError(
                f"ordered edges {index} and {index + 1} do not share exactly "
                f"one endpoint: {shared!r}"
            )
        route.append(shared[0])
    route.append(end)
    return route


def segment_intersects(start_a, end_a, start_b, end_b) -> bool:
    """Return whether two finite, non-collinear 2-D segments intersect."""
    route_delta = end_a - start_a
    crosswalk_delta = end_b - start_b
    denominator = route_delta.cross(crosswalk_delta)
    if abs(denominator) <= 1e-6:
        return False
    offset = start_b - start_a
    route_projection = offset.cross(crosswalk_delta) / denominator
    crosswalk_projection = offset.cross(route_delta) / denominator
    tolerance = 1e-4
    return (
        -tolerance <= route_projection <= 1.0 + tolerance
        and -tolerance <= crosswalk_projection <= 1.0 + tolerance
    )


def vehicle_route_points(controller, incoming_lane, outgoing_lane, intersection):
    """Reproduce the staged vehicle centerline without starting Unreal."""
    dot = max(-1.0, min(1.0, incoming_lane.direction.dot(outgoing_lane.direction)))
    angle = math.degrees(math.acos(dot))
    if angle <= 5.0:
        transition = cal_waypoints(
            incoming_lane.end,
            outgoing_lane.start,
            controller.config["traffic.gap_between_waypoints"],
        )
    elif abs(angle - 180.0) <= 1e-6:
        transition = [outgoing_lane.start]
    else:
        transition = get_bezier_points(
            incoming_lane.end,
            outgoing_lane.start,
            extend_control_point(
                incoming_lane.end,
                outgoing_lane.start,
                intersection.center,
                0.15,
            ),
            controller.config["traffic.steering_point_num"],
        )
    return [
        incoming_lane.end,
        *transition,
        outgoing_lane.start + outgoing_lane.direction * 5000.0,
    ]


def route_intersects(route_points, crosswalk) -> bool:
    return any(
        segment_intersects(start, end, crosswalk.start, crosswalk.end)
        for start, end in zip(route_points, route_points[1:])
    )


def incoming_lane_coverage(controller, crosswalk):
    """List real legal lane movements crossing a route crosswalk."""
    intersection = next(
        item for item in controller.intersections
        if crosswalk in item.crosswalks
    )
    incoming_lanes = [
        lane for lane in intersection.lane_mapping
        if any(light.lane_id == lane.id for light in intersection.traffic_lights)
    ]
    coverage = []
    for incoming_lane in incoming_lanes:
        for outgoing_lane in intersection.lane_mapping[incoming_lane]:
            route_points = vehicle_route_points(
                controller,
                incoming_lane,
                outgoing_lane,
                intersection,
            )
            if route_intersects(route_points, crosswalk):
                coverage.append(
                    {
                        "incoming_lane": incoming_lane.id,
                        "outgoing_lane": outgoing_lane.id,
                    }
                )
    return sorted(
        coverage,
        key=lambda item: (item["incoming_lane"], item["outgoing_lane"]),
    )


def main() -> None:
    _load_offline_dependencies()
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser()
    parser.add_argument("task_file")
    parser.add_argument(
        "--task-specs",
        help="comma-separated one-based task_number:task_id pairs",
    )
    parser.add_argument(
        "--require-signal-coverage",
        action="store_true",
        help="exit nonzero unless every selected crosswalk is signalized and lane-covered",
    )
    parser.add_argument("--output", type=Path, help="optional JSONL report path")
    args = parser.parse_args()
    task_file = args.task_file
    selected_specs = {}
    if args.task_specs:
        for raw_spec in args.task_specs.split(","):
            try:
                task_number, task_id = (
                    int(value) for value in raw_spec.split(":", 1)
                )
            except (TypeError, ValueError) as exc:
                parser.error(
                    f"invalid task spec {raw_spec!r}; expected task_number:task_id"
                )
            selected_specs[task_number] = task_id
    config = Config(str(REPO_ROOT / "config.yaml"))
    with open(task_file, encoding="utf-8") as handle:
        tasks = json.load(handle)["tasks"]

    failed = []
    records = []
    for number, task in enumerate(tasks, 1):
        if selected_specs and number not in selected_specs:
            continue
        controller = RTTrafficController(
            config,
            task["map_path"],
            seed=1,
            num_vehicles=3,
            num_pedestrians=2,
        )
        ordered_route = reconstruct_route_points_from_task_edges(task)
        generated_route = (task.get("route_info") or {}).get("shortest_path", [])
        route_crosswalks = controller.get_route_crosswalks(ordered_route)
        groups = controller.get_crosswalk_signal_groups(route_crosswalks)
        details = []
        for crosswalk in route_crosswalks:
            signals = groups.get(crosswalk.id, [])
            intersections = [
                intersection.id
                for intersection in controller.intersections
                if crosswalk in intersection.crosswalks
            ]
            details.append(
                {
                    "crosswalk": crosswalk.id,
                    "intersections": intersections,
                    "signal_count": len(signals),
                    "vehicle_signals": sum(
                        getattr(signal, "type", None) != "pedestrian"
                        for signal in signals
                    ),
                    "pedestrian_signals": sum(
                        getattr(signal, "type", None) == "pedestrian"
                        for signal in signals
                    ),
                    "incoming_lane_coverage": incoming_lane_coverage(
                        controller,
                        crosswalk,
                    ),
                }
            )
        declared_crosswalk_hops = int(task.get("crosswalk_hops", 0) or 0)
        coverage_ok = (
            len(details) == declared_crosswalk_hops
            and all(
                item["signal_count"] > 0
                and item["pedestrian_signals"] > 0
                and item["incoming_lane_coverage"]
                for item in details
            )
        )
        record = {
            "task_number": number,
            "task_id": task.get("task_id"),
            "declared_crosswalk_hops": declared_crosswalk_hops,
            "matched_route_crosswalks": len(details),
            "unmatched_declared_crosswalk_hops": max(
                0, declared_crosswalk_hops - len(details)
            ),
            "traffic_signal_applicable": any(
                item["pedestrian_signals"] > 0 for item in details
            ),
            "ordered_route": ordered_route,
            "stale_generated_route_replaced": ordered_route != generated_route,
            "route_crosswalks": details,
            "coverage_ok": coverage_ok,
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True))
        expected_task_id = selected_specs.get(number)
        if expected_task_id is not None and task.get("task_id") != expected_task_id:
            failed.append(
                f"task number {number} has task_id={task.get('task_id')}, "
                f"expected {expected_task_id}"
            )
        if args.require_signal_coverage and not coverage_ok:
            failed.append(
                f"task number {number} / task_id={task.get('task_id')} "
                "does not have complete signal/lane route-crosswalk coverage"
            )

    missing_numbers = sorted(set(selected_specs) - set(range(1, len(tasks) + 1)))
    if missing_numbers:
        failed.append(f"task numbers not present: {missing_numbers}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
    if failed:
        for message in failed:
            print(f"COVERAGE_ERROR: {message}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
