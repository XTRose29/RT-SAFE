#!/usr/bin/env python3
"""Offline safe-path planner for a recorded SimWorld task.

This script reconstructs the task/environment setup from JSON artifacts only.
It does not connect to Unreal Engine and does not call any VLM.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DIFFICULTY_DYNAMIC_ACTIVATION_RATIO = {
    "default": 1.0,
    "medium": 0.8,
    "easy": 0.6,
}

PED_NUM_PER_EDGE = 20
SPEED_OPTIONS = [100.0, 150.0, 200.0]


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 4), "y": round(self.y, 4)}


@dataclass
class MovingAgent:
    id: str
    kind: str
    position: Point
    speed: float
    waypoints: List[Point]
    cyclic: bool = True

    def route(self) -> List[Point]:
        return [self.position] + self.waypoints

    def position_at(self, time_s: float) -> Point:
        route = self.route()
        if len(route) == 1 or self.speed <= 0:
            return route[0]

        segments = []
        total_duration = 0.0
        for start, end in zip(route, route[1:]):
            distance = start.distance(end)
            if distance <= 1e-6:
                continue
            duration = distance / self.speed
            segments.append((start, end, distance, duration))
            total_duration += duration

        if not segments:
            return route[-1]

        t = max(0.0, time_s)
        if self.cyclic:
            t = t % total_duration
        elif t >= total_duration:
            return route[-1]

        elapsed = 0.0
        for start, end, distance, duration in segments:
            if t <= elapsed + duration:
                ratio = (t - elapsed) / duration
                return Point(
                    start.x + (end.x - start.x) * ratio,
                    start.y + (end.y - start.y) * ratio,
                )
            elapsed += duration
        return segments[-1][1]

    def as_dict(self) -> Dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "position": self.position.as_dict(),
            "speed_cm_s": self.speed,
            "num_waypoints": len(self.waypoints),
            "waypoints": [p.as_dict() for p in self.waypoints],
        }


@dataclass(frozen=True)
class TrafficLightSchedule:
    crosswalk_edges: List[Tuple[Point, Point]]
    crosswalk_width: float
    vehicle_phase_count: int
    vehicle_green_duration: float
    yellow_duration: float
    pedestrian_green_duration: float
    safety_margin: float

    @property
    def first_pedestrian_green_start(self) -> float:
        return self.vehicle_phase_count * (self.vehicle_green_duration + self.yellow_duration)

    @property
    def cycle_duration(self) -> float:
        return self.first_pedestrian_green_start + self.pedestrian_green_duration

    def segment_uses_crosswalk(self, start: "Point", end: "Point") -> bool:
        for crosswalk_start, crosswalk_end in self.crosswalk_edges:
            if (
                segment_to_segment_distance(start, end, crosswalk_start, crosswalk_end) <= self.crosswalk_width
                or point_to_segment_distance(start, crosswalk_start, crosswalk_end) <= self.crosswalk_width
                or point_to_segment_distance(end, crosswalk_start, crosswalk_end) <= self.crosswalk_width
            ):
                return True
        return False

    def pedestrian_walk_interval(self, time_s: float) -> Tuple[float, float]:
        if self.cycle_duration <= 0:
            return (float("inf"), float("inf"))
        cycle_index = math.floor(max(0.0, time_s) / self.cycle_duration)
        start = cycle_index * self.cycle_duration + self.first_pedestrian_green_start
        end = start + self.pedestrian_green_duration
        if time_s > end - self.safety_margin:
            start += self.cycle_duration
            end += self.cycle_duration
        return start + self.safety_margin, end - self.safety_margin

    def allows_crossing(self, start_time: float, duration: float) -> Tuple[bool, Dict]:
        green_start, green_end = self.pedestrian_walk_interval(start_time)
        end_time = start_time + duration
        allowed = start_time >= green_start and end_time <= green_end
        return allowed, {
            "traffic_window_start_s": round(green_start, 4),
            "traffic_window_end_s": round(green_end, 4),
            "traffic_checked": True,
        }

    def as_dict(self) -> Dict:
        return {
            "mode": "modeled_fixed_cycle",
            "vehicle_phase_count": self.vehicle_phase_count,
            "vehicle_green_duration_s": self.vehicle_green_duration,
            "yellow_duration_s": self.yellow_duration,
            "pedestrian_green_duration_s": self.pedestrian_green_duration,
            "safety_margin_s": self.safety_margin,
            "first_pedestrian_green_start_s": round(self.first_pedestrian_green_start, 4),
            "cycle_duration_s": round(self.cycle_duration, 4),
            "crosswalk_edges": [
                {"start": start.as_dict(), "end": end.as_dict()} for start, end in self.crosswalk_edges
            ],
        }


def point_from_xy(xy: Sequence[float]) -> Point:
    return Point(float(xy[0]), float(xy[1]))


def interpolate_distance(start: Point, end: Point, distance: float) -> Point:
    segment_length = start.distance(end)
    if segment_length <= 1e-6:
        return end
    ratio = max(0.0, min(1.0, distance / segment_length))
    return Point(
        start.x + (end.x - start.x) * ratio,
        start.y + (end.y - start.y) * ratio,
    )


def point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return point.distance(start)
    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest = Point(start.x + dx * t, start.y + dy * t)
    return point.distance(closest)


def segment_to_segment_distance(a_start: Point, a_end: Point, b_start: Point, b_end: Point) -> float:
    if segments_intersect(a_start, a_end, b_start, b_end):
        return 0.0
    return min(
        point_to_segment_distance(a_start, b_start, b_end),
        point_to_segment_distance(a_end, b_start, b_end),
        point_to_segment_distance(b_start, a_start, a_end),
        point_to_segment_distance(b_end, a_start, a_end),
    )


def segments_intersect(a_start: Point, a_end: Point, b_start: Point, b_end: Point) -> bool:
    def orient(p: Point, q: Point, r: Point) -> float:
        return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)

    def on_segment(p: Point, q: Point, r: Point) -> bool:
        eps = 1e-9
        return (
            min(p.x, r.x) - eps <= q.x <= max(p.x, r.x) + eps
            and min(p.y, r.y) - eps <= q.y <= max(p.y, r.y) + eps
        )

    o1 = orient(a_start, a_end, b_start)
    o2 = orient(a_start, a_end, b_end)
    o3 = orient(b_start, b_end, a_start)
    o4 = orient(b_start, b_end, a_end)
    eps = 1e-9
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) <= eps and on_segment(a_start, b_start, a_end):
        return True
    if abs(o2) <= eps and on_segment(a_start, b_end, a_end):
        return True
    if abs(o3) <= eps and on_segment(b_start, a_start, b_end):
        return True
    if abs(o4) <= eps and on_segment(b_start, a_end, b_end):
        return True
    return False


def segment_intersects_aabb(start: Point, end: Point, bounds: Dict[str, float], margin: float = 0.0) -> bool:
    min_x = min(bounds["min_x"], bounds["max_x"]) - margin
    max_x = max(bounds["min_x"], bounds["max_x"]) + margin
    min_y = min(bounds["min_y"], bounds["max_y"]) - margin
    max_y = max(bounds["min_y"], bounds["max_y"]) + margin

    if min_x <= start.x <= max_x and min_y <= start.y <= max_y:
        return True
    if min_x <= end.x <= max_x and min_y <= end.y <= max_y:
        return True

    dx = end.x - start.x
    dy = end.y - start.y
    p = [-dx, dx, -dy, dy]
    q = [start.x - min_x, max_x - start.x, start.y - min_y, max_y - start.y]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) <= 1e-12:
            if qi < 0:
                return False
            continue
        ratio = qi / pi
        if pi < 0:
            u1 = max(u1, ratio)
        else:
            u2 = min(u2, ratio)
        if u1 > u2:
            return False
    return True


def segment_to_aabb_distance(start: Point, end: Point, bounds: Dict[str, float]) -> float:
    if segment_intersects_aabb(start, end, bounds):
        return 0.0
    min_x = min(bounds["min_x"], bounds["max_x"])
    max_x = max(bounds["min_x"], bounds["max_x"])
    min_y = min(bounds["min_y"], bounds["max_y"])
    max_y = max(bounds["min_y"], bounds["max_y"])
    corners = [
        Point(min_x, min_y),
        Point(max_x, min_y),
        Point(max_x, max_y),
        Point(min_x, max_y),
    ]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    return min(
        [point_to_segment_distance(corner, start, end) for corner in corners]
        + [segment_to_segment_distance(start, end, edge_start, edge_end) for edge_start, edge_end in edges]
    )


def relative_segment_distance(
    agent_start: Point,
    agent_end: Point,
    other_start: Point,
    other_end: Point,
) -> float:
    rel_start = Point(agent_start.x - other_start.x, agent_start.y - other_start.y)
    rel_end = Point(agent_end.x - other_end.x, agent_end.y - other_end.y)
    return point_to_segment_distance(Point(0.0, 0.0), rel_start, rel_end)


def polyline_length(route: Sequence[Point]) -> float:
    return sum(route[i].distance(route[i + 1]) for i in range(len(route) - 1))


def distance_to_route(point: Point, route: Sequence[Point]) -> float:
    return min(point_to_segment_distance(point, route[i], route[i + 1]) for i in range(len(route) - 1))


def route_progress(point: Point, route: Sequence[Point]) -> float:
    best_distance = float("inf")
    best_progress = 0.0
    cumulative = 0.0
    for start, end in zip(route, route[1:]):
        dx = end.x - start.x
        dy = end.y - start.y
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            continue
        t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq
        clamped = max(0.0, min(1.0, t))
        projected = Point(start.x + dx * clamped, start.y + dy * clamped)
        distance = point.distance(projected)
        if distance < best_distance:
            best_distance = distance
            best_progress = cumulative + start.distance(projected)
        cumulative += math.sqrt(length_sq)
    return best_progress


@dataclass(frozen=True)
class PedestrianEdge:
    start: Point
    end: Point
    kind: str
    road_side_sign: Optional[int] = None


class PedestrianRegion:
    """Task-edge-aware pedestrian search region.

    Sidewalks allow detours only on the pedestrian side of the curb. Crosswalks
    allow crossing the road, but only inside a narrower band around the marked
    crosswalk edge.
    """

    def __init__(
        self,
        raw_edges: Sequence[Dict],
        route: Sequence[Point],
        sidewalk_width: float,
        crosswalk_width: float,
        road_side_tolerance: float,
    ) -> None:
        self.route = list(route)
        self.sidewalk_width = sidewalk_width
        self.crosswalk_width = crosswalk_width
        self.road_side_tolerance = road_side_tolerance
        self.edges = self._build_edges(raw_edges)

    def _build_edges(self, raw_edges: Sequence[Dict]) -> List[PedestrianEdge]:
        raw_segments = [
            (point_from_xy(edge["node1"]), point_from_xy(edge["node2"]), edge.get("type", "sidewalk"))
            for edge in raw_edges
            if "node1" in edge and "node2" in edge
        ]
        built: List[PedestrianEdge] = []
        for start, end, kind in raw_segments:
            road_side_sign: Optional[int] = None
            if kind == "sidewalk":
                road_side_sign = self._infer_road_side_sign(start, end, raw_segments)
            built.append(PedestrianEdge(start=start, end=end, kind=kind, road_side_sign=road_side_sign))
        return built

    def _infer_road_side_sign(
        self,
        start: Point,
        end: Point,
        raw_segments: Sequence[Tuple[Point, Point, str]],
    ) -> Optional[int]:
        dx = end.x - start.x
        dy = end.y - start.y
        vertical = abs(dy) >= abs(dx)
        axis_value = start.x if vertical else start.y

        for cross_start, cross_end, kind in raw_segments:
            if kind != "crosswalk":
                continue
            for sidewalk_endpoint in (start, end):
                for crosswalk_endpoint, other_endpoint in (
                    (cross_start, cross_end),
                    (cross_end, cross_start),
                ):
                    if sidewalk_endpoint.distance(crosswalk_endpoint) > 1e-6:
                        continue
                    if vertical:
                        delta = other_endpoint.x - axis_value
                    else:
                        delta = other_endpoint.y - axis_value
                    if abs(delta) > 1e-6:
                        return 1 if delta > 0 else -1
        return None

    def contains(self, point: Point) -> bool:
        return any(self._edge_contains(point, edge) for edge in self.edges)

    def segment_contains(self, start: Point, end: Point, sample_spacing: float) -> bool:
        distance = start.distance(end)
        samples = max(2, int(math.ceil(distance / max(1.0, sample_spacing))) + 1)
        for idx in range(samples):
            ratio = idx / (samples - 1)
            point = Point(start.x + (end.x - start.x) * ratio, start.y + (end.y - start.y) * ratio)
            if not self.contains(point):
                return False
        return True

    def _edge_contains(self, point: Point, edge: PedestrianEdge) -> bool:
        width = self.crosswalk_width if edge.kind == "crosswalk" else self.sidewalk_width
        if point_to_segment_distance(point, edge.start, edge.end) > width:
            return False

        if edge.kind != "sidewalk" or edge.road_side_sign is None:
            return True

        dx = edge.end.x - edge.start.x
        dy = edge.end.y - edge.start.y
        if abs(dy) >= abs(dx):
            offset = point.x - edge.start.x
        else:
            offset = point.y - edge.start.y
        return offset * edge.road_side_sign <= self.road_side_tolerance


def load_static_obstacles(map_dir: Path) -> List[Dict]:
    static_obstacles: List[Dict] = []

    obstacles_path = map_dir / "obstacles.json"
    if obstacles_path.exists():
        with obstacles_path.open("r", encoding="utf-8") as f:
            obstacles_data = json.load(f)
        for node in obstacles_data.get("nodes", []):
            location = node.get("properties", {}).get("location", {})
            if "x" not in location or "y" not in location:
                continue
            static_obstacles.append(
                {
                    "id": node.get("id", "unknown"),
                    "kind": "static",
                    "type": node.get("instance_name", "unknown"),
                    "point": Point(float(location["x"]), float(location["y"])),
                }
            )

    buildings_path = map_dir / "buildings.json"
    if buildings_path.exists():
        with buildings_path.open("r", encoding="utf-8") as f:
            buildings_data = json.load(f)
        for idx, building in enumerate(buildings_data.get("buildings", [])):
            bounds = building.get("bounds", {})
            if not all(key in bounds for key in ("x", "y", "width", "height")):
                continue
            static_obstacles.append(
                {
                    "id": f"building_{idx}",
                    "kind": "building",
                    "type": building.get("type", "building"),
                    "bounds": {
                        "min_x": bounds["x"] * 100.0,
                        "min_y": bounds["y"] * 100.0,
                        "max_x": (bounds["x"] + bounds["width"]) * 100.0,
                        "max_y": (bounds["y"] + bounds["height"]) * 100.0,
                    },
                }
            )
    return static_obstacles


def convert_task_to_scenario(task: Dict) -> Dict:
    route = task.get("route_info", {}).get("shortest_path", [])
    required_time = 1000.0
    if len(route) >= 2:
        total = polyline_length([point_from_xy(p) for p in route])
        crosswalk_count = sum(1 for edge in task.get("edges", []) if edge.get("type") == "crosswalk")
        required_time = total / 100.0 * 1.2 + crosswalk_count * 60.0
    return {
        "task_id": task.get("task_id"),
        "task": {
            "start_point": task["start_point"],
            "end_point": task["end_point"],
            "edges": task.get("edges", []),
            "total_hops": task.get("total_hops", 0),
            "crosswalk_hops": task.get("crosswalk_hops", 0),
            "required_time": required_time,
        },
        "route_info": task.get("route_info", {"shortest_path": []}),
        "pedestrian_routes": task.get("pedestrian_routes", []),
        "irregular_routes": task.get("irregular_routes", []),
        "falling_objects": task.get("falling_objects", []),
        "map_path": task.get("map_path"),
    }


def try_place_on_route(
    route_nodes: List[Point],
    occupied_positions: List[Point],
    agent_start_pos: Point,
    rng: random.Random,
    max_attempts_per_segment: int = 30,
) -> Tuple[Optional[Point], Optional[int]]:
    indices = list(range(len(route_nodes) - 1))
    rng.shuffle(indices)
    for idx in indices:
        start = route_nodes[idx]
        end = route_nodes[idx + 1]
        for _ in range(max_attempts_per_segment):
            ratio = rng.random()
            candidate = Point(
                start.x + (end.x - start.x) * ratio,
                start.y + (end.y - start.y) * ratio,
            )
            safe_from_endpoints = candidate.distance(start) >= 500 and candidate.distance(end) >= 500
            safe_from_others = all(candidate.distance(other) >= 200 for other in occupied_positions)
            safe_from_agent = candidate.distance(agent_start_pos) >= 200
            if safe_from_endpoints and safe_from_others and safe_from_agent:
                return candidate, idx
    return None, None


def build_regular_pedestrians(
    scenario: Dict,
    num_agents: int,
    difficulty: str,
    rng: random.Random,
) -> List[MovingAgent]:
    pedestrian_routes = scenario.get("pedestrian_routes", [])
    if not pedestrian_routes:
        return []

    agent_start_pos = point_from_xy(scenario["task"]["start_point"])
    activation_ratio = DIFFICULTY_DYNAMIC_ACTIVATION_RATIO[difficulty]
    num_pedestrians = int(round(num_agents * activation_ratio))
    pedestrians: List[MovingAgent] = []

    for i in range(num_pedestrians):
        route_data = rng.choice(pedestrian_routes)
        route_points = route_data["route"]
        is_loop = bool(route_data.get("is_loop", False))
        if len(route_points) < 2:
            continue

        random_offset_x = rng.uniform(-50, 50)
        random_offset_y = rng.uniform(-50, 50)
        route_nodes = [Point(p[0] + random_offset_x, p[1] + random_offset_y) for p in route_points]
        if rng.random() < 0.5:
            route_nodes = list(reversed(route_nodes))

        occupied_positions = [p.position for p in pedestrians]
        position, chosen_idx = try_place_on_route(route_nodes, occupied_positions, agent_start_pos, rng)

        if position is None:
            other_routes = list(pedestrian_routes)
            if route_data in other_routes:
                other_routes.remove(route_data)
            rng.shuffle(other_routes)
            for alt_route in other_routes:
                alt_points = alt_route["route"]
                if len(alt_points) < 2:
                    continue
                alt_nodes = [Point(p[0] + random_offset_x, p[1] + random_offset_y) for p in alt_points]
                if rng.random() < 0.5:
                    alt_nodes = list(reversed(alt_nodes))
                position, chosen_idx = try_place_on_route(alt_nodes, occupied_positions, agent_start_pos, rng)
                if position is not None:
                    route_nodes = alt_nodes
                    is_loop = bool(alt_route.get("is_loop", False))
                    break

        if position is None:
            lengths = [route_nodes[j].distance(route_nodes[j + 1]) for j in range(len(route_nodes) - 1)]
            if not lengths:
                continue
            chosen_idx = int(max(range(len(lengths)), key=lambda k: lengths[k]))
            start = route_nodes[chosen_idx]
            end = route_nodes[chosen_idx + 1]
            position = Point((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)

        assert chosen_idx is not None
        if is_loop:
            waypoints = route_nodes[chosen_idx + 1 :] + route_nodes[: chosen_idx + 1]
        else:
            forward = route_nodes[chosen_idx + 1 : -1] + list(reversed(route_nodes[chosen_idx + 1 :]))
            reverse = list(reversed(route_nodes[: chosen_idx + 1])) + route_nodes[1 : chosen_idx + 1]
            waypoints = forward + reverse

        interpolated: List[Point] = []
        for j, waypoint in enumerate(waypoints):
            interpolated.append(waypoint)
            if j >= len(waypoints) - 1:
                continue
            distance = waypoint.distance(waypoints[j + 1])
            if distance > 2500:
                num_segments = int(distance / 2500)
                for k in range(1, num_segments + 1):
                    ratio = k / (num_segments + 1)
                    interpolated.append(
                        Point(
                            waypoint.x + (waypoints[j + 1].x - waypoint.x) * ratio,
                            waypoint.y + (waypoints[j + 1].y - waypoint.y) * ratio,
                        )
                    )
        speed = rng.choice(SPEED_OPTIONS)
        pedestrians.append(MovingAgent(f"RT_PEDESTRIAN_{i}", "pedestrian", position, speed, interpolated))

    return pedestrians


def build_irregular_pedestrians(
    scenario: Dict,
    difficulty: str,
    rng: random.Random,
    existing_pedestrians: List[MovingAgent],
    static_obstacles: List[Dict],
) -> List[MovingAgent]:
    irregular_routes = scenario.get("irregular_routes", [])
    if not irregular_routes:
        return []

    activation_ratio = DIFFICULTY_DYNAMIC_ACTIVATION_RATIO[difficulty]
    target_count = int(round(len(irregular_routes) * activation_ratio))
    target_count = max(0, min(len(irregular_routes), target_count))
    sampled_routes = rng.sample(irregular_routes, target_count) if target_count > 0 else []

    agent_start_pos = point_from_xy(scenario["task"]["start_point"])
    static_points = [ob["point"] for ob in static_obstacles if ob.get("point")]
    irregulars: List[MovingAgent] = []

    for route_data in sampled_routes:
        route_points = route_data.get("route", [])
        if len(route_points) < 2:
            continue
        route_nodes = [point_from_xy(p) for p in route_points]
        occupied = [p.position for p in existing_pedestrians] + [p.position for p in irregulars]
        chosen_idx = None
        for idx, candidate in enumerate(route_nodes):
            safe_from_others = all(candidate.distance(other) >= 200 for other in occupied)
            safe_from_agent = candidate.distance(agent_start_pos) >= 200
            safe_from_obstacles = all(candidate.distance(obs) >= 100 for obs in static_points)
            if safe_from_others and safe_from_agent and safe_from_obstacles:
                chosen_idx = idx
                break
        if chosen_idx is None:
            continue

        position = route_nodes[chosen_idx]
        part1 = route_nodes[chosen_idx + 1 :]
        part2 = list(reversed(route_nodes[:-1]))
        part3 = route_nodes[1 : chosen_idx + 1] if chosen_idx > 0 else []
        waypoints = part1 + part2 + part3
        if not waypoints:
            continue
        agent_id = f"RT_SCOOTER_{len(irregulars)}"
        irregulars.append(MovingAgent(agent_id, "irregular_robot_dog", position, 100.0, waypoints))

    return irregulars


def build_traffic_light_schedule(
    task_edges: Sequence[Dict],
    crosswalk_width: float,
    vehicle_phase_count: int,
    vehicle_green_duration: float,
    yellow_duration: float,
    pedestrian_green_duration: float,
    safety_margin: float,
) -> Optional[TrafficLightSchedule]:
    crosswalk_edges = [
        (point_from_xy(edge["node1"]), point_from_xy(edge["node2"]))
        for edge in task_edges
        if edge.get("type") == "crosswalk" and "node1" in edge and "node2" in edge
    ]
    if not crosswalk_edges:
        return None
    return TrafficLightSchedule(
        crosswalk_edges=crosswalk_edges,
        crosswalk_width=crosswalk_width,
        vehicle_phase_count=vehicle_phase_count,
        vehicle_green_duration=vehicle_green_duration,
        yellow_duration=yellow_duration,
        pedestrian_green_duration=pedestrian_green_duration,
        safety_margin=safety_margin,
    )


class OfflineSafePathPlanner:
    def __init__(
        self,
        route: List[Point],
        pedestrian_region: PedestrianRegion,
        static_obstacles: List[Dict],
        moving_agents: List[MovingAgent],
        agent_speed: float,
        grid_resolution: float,
        corridor_width: float,
        static_clearance: float,
        dynamic_clearance: float,
        building_margin: float,
        time_step: float,
        max_time: float,
        traffic_light_schedule: Optional[TrafficLightSchedule] = None,
    ) -> None:
        self.route = route
        self.pedestrian_region = pedestrian_region
        self.static_obstacles = static_obstacles
        self.moving_agents = moving_agents
        self.agent_speed = agent_speed
        self.grid_resolution = grid_resolution
        self.corridor_width = corridor_width
        self.static_clearance = static_clearance
        self.dynamic_clearance = dynamic_clearance
        self.building_margin = building_margin
        self.time_step = time_step
        self.max_ticks = int(math.ceil(max_time / time_step))
        self.traffic_light_schedule = traffic_light_schedule
        self.total_route_length = polyline_length(route)
        self.nodes: Dict[Tuple[int, int], Point] = {}

    def build_grid(self, start: Point, goal: Point) -> None:
        margin = self.corridor_width + self.grid_resolution * 2
        min_x = min(p.x for p in self.route) - margin
        max_x = max(p.x for p in self.route) + margin
        min_y = min(p.y for p in self.route) - margin
        max_y = max(p.y for p in self.route) + margin

        ix_min = math.floor(min_x / self.grid_resolution)
        ix_max = math.ceil(max_x / self.grid_resolution)
        iy_min = math.floor(min_y / self.grid_resolution)
        iy_max = math.ceil(max_y / self.grid_resolution)

        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                point = Point(ix * self.grid_resolution, iy * self.grid_resolution)
                if distance_to_route(point, self.route) <= self.corridor_width and self.point_is_valid(point):
                    self.nodes[(ix, iy)] = point

        self.nodes[self.key_for_point(start)] = start
        self.nodes[self.key_for_point(goal)] = goal

    def key_for_point(self, point: Point) -> Tuple[int, int]:
        return (int(round(point.x / self.grid_resolution)), int(round(point.y / self.grid_resolution)))

    def point_is_valid(self, point: Point) -> bool:
        if not self.pedestrian_region.contains(point):
            return False
        return self.static_segment_clearance(point, point)[0] >= self.static_clearance

    def static_segment_clearance(self, start: Point, end: Point) -> Tuple[float, Optional[Dict]]:
        min_distance = float("inf")
        closest: Optional[Dict] = None
        for obstacle in self.static_obstacles:
            if obstacle.get("bounds"):
                distance = segment_to_aabb_distance(start, end, obstacle["bounds"])
                if segment_intersects_aabb(start, end, obstacle["bounds"], self.building_margin):
                    distance = 0.0
            else:
                distance = point_to_segment_distance(obstacle["point"], start, end)
            if distance < min_distance:
                min_distance = distance
                closest = obstacle
        return min_distance, closest

    def dynamic_segment_clearance(self, start: Point, end: Point, start_time: float, duration: float) -> Tuple[float, Optional[MovingAgent]]:
        min_distance = float("inf")
        closest = None
        end_time = start_time + duration
        for agent in self.moving_agents:
            other_start = agent.position_at(start_time)
            other_end = agent.position_at(end_time)
            distance = relative_segment_distance(start, end, other_start, other_end)
            if distance < min_distance:
                min_distance = distance
                closest = agent
        return min_distance, closest

    def segment_is_valid(self, start: Point, end: Point, start_tick: int, duration_ticks: int) -> Tuple[bool, Dict]:
        start_time = start_tick * self.time_step
        duration = duration_ticks * self.time_step
        if not self.pedestrian_region.segment_contains(
            start,
            end,
            sample_spacing=self.grid_resolution / 2.0,
        ):
            return False, {"reason": "outside_pedestrian_region"}

        static_clearance, static_obstacle = self.static_segment_clearance(start, end)
        if static_clearance < self.static_clearance:
            return False, {
                "reason": "static_clearance",
                "clearance_cm": round(static_clearance, 4),
                "obstacle_id": static_obstacle.get("id") if static_obstacle else None,
                "obstacle_kind": static_obstacle.get("kind") if static_obstacle else None,
            }

        dynamic_clearance = float("inf")
        dynamic_agent = None
        if self.dynamic_clearance > 0.0 and self.moving_agents:
            dynamic_clearance, dynamic_agent = self.dynamic_segment_clearance(start, end, start_time, duration)
            if dynamic_clearance < self.dynamic_clearance:
                return False, {
                    "reason": "dynamic_clearance",
                    "clearance_cm": round(dynamic_clearance, 4),
                    "obstacle_id": dynamic_agent.id if dynamic_agent else None,
                    "obstacle_kind": dynamic_agent.kind if dynamic_agent else None,
                }

        traffic_details: Dict = {"traffic_checked": False}
        if self.traffic_light_schedule and self.traffic_light_schedule.segment_uses_crosswalk(start, end):
            allowed, traffic_details = self.traffic_light_schedule.allows_crossing(start_time, duration)
            if not allowed:
                return False, {
                    "reason": "traffic_light",
                    **traffic_details,
                }

        return True, {
            "static_clearance_cm": round(static_clearance, 4) if static_clearance != float("inf") else "inf",
            "dynamic_clearance_cm": round(dynamic_clearance, 4) if dynamic_clearance != float("inf") else "inf",
            "closest_static_id": static_obstacle.get("id") if static_obstacle else None,
            "closest_dynamic_id": dynamic_agent.id if dynamic_agent else None,
            **traffic_details,
        }

    def neighbors(self, key: Tuple[int, int]) -> Iterable[Tuple[Tuple[int, int], int]]:
        ix, iy = key
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                next_key = (ix + dx, iy + dy)
                if next_key not in self.nodes:
                    continue
                distance = self.nodes[key].distance(self.nodes[next_key])
                if distance > self.grid_resolution * math.sqrt(2.0) + 1e-6:
                    continue
                duration = max(1, int(math.ceil((distance / self.agent_speed) / self.time_step)))
                yield next_key, duration
        if (self.dynamic_clearance > 0.0 and self.moving_agents) or self.traffic_light_schedule:
            yield key, int(round(1.0 / self.time_step))

    def heuristic(self, key: Tuple[int, int], goal: Point) -> float:
        point = self.nodes[key]
        euclidean = point.distance(goal) / self.agent_speed
        progress = route_progress(point, self.route)
        route_remaining = max(0.0, self.total_route_length - progress) / self.agent_speed
        return min(euclidean, route_remaining)

    def plan(self, start: Point, goal: Point) -> Tuple[Optional[List[Dict]], Dict]:
        self.build_grid(start, goal)
        start_key = self.key_for_point(start)
        goal_key = self.key_for_point(goal)
        time_dependent = (self.dynamic_clearance > 0.0 and bool(self.moving_agents)) or bool(self.traffic_light_schedule)
        use_node_dominance = self.traffic_light_schedule is None

        queue: List[Tuple[float, float, int, Tuple[int, int], int]] = []
        counter = 0
        heapq.heappush(queue, (self.heuristic(start_key, goal), 0.0, counter, start_key, 0))
        best_cost: Dict[Tuple[Tuple[int, int], int], float] = {(start_key, 0): 0.0}
        best_node_cost: Dict[Tuple[int, int], float] = {start_key: 0.0}
        parent: Dict[Tuple[Tuple[int, int], int], Tuple[Tuple[Tuple[int, int], int], int, Dict]] = {}
        rejected: Dict[str, int] = {}
        expanded = 0
        best_goal_state = None

        while queue:
            _, cost, _, key, tick = heapq.heappop(queue)
            state = (key, tick)
            if cost > best_cost.get(state, float("inf")) + 1e-9:
                continue
            if not time_dependent and cost > best_node_cost.get(key, float("inf")) + 1e-9:
                continue
            expanded += 1
            current = self.nodes[key]
            if current.distance(goal) <= self.grid_resolution:
                best_goal_state = state
                break
            if tick >= self.max_ticks:
                continue

            for next_key, duration_ticks in self.neighbors(key):
                next_tick = tick + duration_ticks
                if next_tick > self.max_ticks:
                    continue
                valid, details = self.segment_is_valid(current, self.nodes[next_key], tick, duration_ticks)
                if not valid:
                    rejected[details["reason"]] = rejected.get(details["reason"], 0) + 1
                    continue
                next_cost = cost + duration_ticks * self.time_step
                next_state = (next_key, next_tick)
                if next_key != key and next_cost + 1e-9 >= best_node_cost.get(next_key, float("inf")):
                    continue
                if next_cost + 1e-9 < best_cost.get(next_state, float("inf")):
                    counter += 1
                    best_cost[next_state] = next_cost
                    if use_node_dominance and next_key != key:
                        best_node_cost[next_key] = next_cost
                    parent[next_state] = (state, duration_ticks, details)
                    heapq.heappush(
                        queue,
                        (next_cost + self.heuristic(next_key, goal), next_cost, counter, next_key, next_tick),
                    )

        diagnostics = {
            "grid_nodes": len(self.nodes),
            "expanded_states": expanded,
            "rejected_transitions": rejected,
            "max_time_s": self.max_ticks * self.time_step,
        }
        if best_goal_state is None:
            return None, diagnostics

        states = []
        state = best_goal_state
        while state in parent:
            prev_state, duration_ticks, details = parent[state]
            states.append((prev_state, state, duration_ticks, details))
            state = prev_state
        states.reverse()

        trajectory = []
        for step_idx, (prev_state, next_state, duration_ticks, details) in enumerate(states):
            prev_key, prev_tick = prev_state
            next_key, next_tick = next_state
            start_point = self.nodes[prev_key]
            end_point = self.nodes[next_key]
            duration = duration_ticks * self.time_step
            action_type = "wait" if prev_key == next_key else "move"
            trajectory.append(
                {
                    "step": step_idx,
                    "action_type": action_type,
                    "start": start_point.as_dict(),
                    "end": end_point.as_dict(),
                    "start_time_s": round(prev_tick * self.time_step, 4),
                    "duration_s": round(duration, 4),
                    "end_time_s": round(next_tick * self.time_step, 4),
                    "distance_cm": round(start_point.distance(end_point), 4),
                    "static_clearance_cm": details.get("static_clearance_cm"),
                    "dynamic_clearance_cm": details.get("dynamic_clearance_cm"),
                    "closest_static_id": details.get("closest_static_id"),
                    "closest_dynamic_id": details.get("closest_dynamic_id"),
                    "traffic_checked": details.get("traffic_checked", False),
                    "traffic_window_start_s": details.get("traffic_window_start_s"),
                    "traffic_window_end_s": details.get("traffic_window_end_s"),
                }
            )
        return trajectory, diagnostics


def _trajectory_points(result: Dict) -> List[Point]:
    trajectory = result.get("trajectory") or []
    points: List[Point] = []
    for idx, step in enumerate(trajectory):
        if idx == 0:
            points.append(Point(float(step["start"]["x"]), float(step["start"]["y"])))
        points.append(Point(float(step["end"]["x"]), float(step["end"]["y"])))
    return points


def write_trajectory_map(output_dir: Path, result: Dict, static_obstacles: List[Dict]) -> Optional[Path]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    route_points = [Point(float(p["x"]), float(p["y"])) for p in result["route"]["shortest_path"]]
    trajectory_points = _trajectory_points(result)
    edge_points = []
    for edge in result["route"].get("task_edges", []):
        edge_points.extend([point_from_xy(edge["node1"]), point_from_xy(edge["node2"])])

    all_points = route_points + trajectory_points + edge_points
    if not all_points:
        return None

    view_margin = 900.0
    min_x = min(p.x for p in all_points) - view_margin
    max_x = max(p.x for p in all_points) + view_margin
    min_y = min(p.y for p in all_points) - view_margin
    max_y = max(p.y for p in all_points) + view_margin

    width = 1400
    height = 1400
    padding = 80
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    scale = min((width - padding * 2) / span_x, (height - padding * 2) / span_y)

    def to_px(point: Point) -> Tuple[int, int]:
        return (
            int(round(padding + (point.x - min_x) * scale)),
            int(round(height - padding - (point.y - min_y) * scale)),
        )

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    view_bounds = {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}
    for obstacle in static_obstacles:
        if obstacle.get("bounds"):
            bounds = obstacle["bounds"]
            if (
                bounds["max_x"] < view_bounds["min_x"]
                or bounds["min_x"] > view_bounds["max_x"]
                or bounds["max_y"] < view_bounds["min_y"]
                or bounds["min_y"] > view_bounds["max_y"]
            ):
                continue
            top_left = to_px(Point(max(bounds["min_x"], min_x), min(bounds["max_y"], max_y)))
            bottom_right = to_px(Point(min(bounds["max_x"], max_x), max(bounds["min_y"], min_y)))
            draw.rectangle([top_left, bottom_right], fill=(232, 232, 232), outline=(180, 180, 180))
        elif obstacle.get("point"):
            point = obstacle["point"]
            if min_x <= point.x <= max_x and min_y <= point.y <= max_y:
                px, py = to_px(point)
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(115, 115, 115))

    for edge in result["route"].get("task_edges", []):
        start = point_from_xy(edge["node1"])
        end = point_from_xy(edge["node2"])
        color = (42, 112, 219) if edge.get("type") == "sidewalk" else (232, 145, 30)
        draw.line([to_px(start), to_px(end)], fill=color, width=8)

    if len(route_points) >= 2:
        draw.line([to_px(point) for point in route_points], fill=(30, 30, 30), width=2)

    if len(trajectory_points) >= 2:
        draw.line([to_px(point) for point in trajectory_points], fill=(215, 30, 30), width=5)
        for point in trajectory_points:
            px, py = to_px(point)
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(215, 30, 30))

    start = Point(float(result["route"]["start"]["x"]), float(result["route"]["start"]["y"]))
    goal = Point(float(result["route"]["goal"]["x"]), float(result["route"]["goal"]["y"]))
    for point, color, label in ((start, (25, 150, 70), "START"), (goal, (150, 40, 165), "GOAL")):
        px, py = to_px(point)
        draw.ellipse((px - 12, py - 12, px + 12, py + 12), fill=color, outline=(0, 0, 0), width=2)
        draw.text((px + 16, py - 12), label, fill=(0, 0, 0), font=small_font)

    draw.rectangle((20, 20, 520, 160), fill=(255, 255, 255), outline=(180, 180, 180))
    draw.text((38, 34), "Offline safe trajectory", fill=(0, 0, 0), font=font)
    legend = [
        ((215, 30, 30), "planned trajectory"),
        ((42, 112, 219), "sidewalk edge"),
        ((232, 145, 30), "crosswalk edge"),
        ((115, 115, 115), "static obstacle"),
    ]
    y = 70
    for color, text in legend:
        draw.line((42, y + 9, 88, y + 9), fill=color, width=6)
        draw.text((100, y), text, fill=(0, 0, 0), font=small_font)
        y += 22

    map_path = output_dir / "offline_safe_trajectory_map.png"
    image.save(map_path)
    return map_path


def write_outputs(output_dir: Path, result: Dict, static_obstacles: List[Dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "offline_safe_trajectory.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    csv_path = output_dir / "offline_safe_trajectory.csv"
    trajectory = result.get("trajectory") or []
    if trajectory:
        fieldnames = list(trajectory[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trajectory)

    map_path = write_trajectory_map(output_dir, result, static_obstacles)
    if map_path is not None:
        result.setdefault("artifacts", {})["map_png"] = str(map_path.name)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


def apply_traffic_light_waits(
    trajectory: List[Dict],
    planner: OfflineSafePathPlanner,
    traffic_light_schedule: TrafficLightSchedule,
    dynamic_clearance: float,
    time_step: float,
    max_wait_s: float,
) -> Tuple[Optional[List[Dict]], Dict]:
    crosswalk_indices = [
        idx
        for idx, step in enumerate(trajectory)
        if traffic_light_schedule.segment_uses_crosswalk(
            Point(float(step["start"]["x"]), float(step["start"]["y"])),
            Point(float(step["end"]["x"]), float(step["end"]["y"])),
        )
    ]
    if not crosswalk_indices:
        return trajectory, {
            "traffic_wait_inserted_s": 0.0,
            "traffic_wait_insert_index": None,
            "checked_crosswalk_steps": [],
            "num_violations": 0,
            "violations": [],
        }

    first_crosswalk_idx = crosswalk_indices[0]
    wait_point = Point(
        float(trajectory[first_crosswalk_idx]["start"]["x"]),
        float(trajectory[first_crosswalk_idx]["start"]["y"]),
    )
    wait_start_time = float(trajectory[first_crosswalk_idx]["start_time_s"])
    max_wait_ticks = int(math.ceil(max_wait_s / time_step))

    best_wait: Optional[float] = None
    best_min_dynamic = float("inf")
    for wait_ticks in range(max_wait_ticks + 1):
        wait_s = wait_ticks * time_step
        traffic_violations = []
        min_dynamic = float("inf")

        if wait_s > 0.0 and dynamic_clearance > 0.0 and planner.moving_agents:
            wait_clearance, _ = planner.dynamic_segment_clearance(wait_point, wait_point, wait_start_time, wait_s)
            min_dynamic = min(min_dynamic, wait_clearance)
            if wait_clearance < dynamic_clearance:
                continue

        dynamic_ok = True
        for idx, step in enumerate(trajectory):
            start = Point(float(step["start"]["x"]), float(step["start"]["y"]))
            end = Point(float(step["end"]["x"]), float(step["end"]["y"]))
            start_time = float(step["start_time_s"]) + (wait_s if idx >= first_crosswalk_idx else 0.0)
            duration = float(step["duration_s"])

            if traffic_light_schedule.segment_uses_crosswalk(start, end):
                allowed, window = traffic_light_schedule.allows_crossing(start_time, duration)
                if not allowed:
                    traffic_violations.append({"step": idx, **window})

            if dynamic_clearance > 0.0 and planner.moving_agents:
                dyn_clearance, _ = planner.dynamic_segment_clearance(start, end, start_time, duration)
                min_dynamic = min(min_dynamic, dyn_clearance)
                if dyn_clearance < dynamic_clearance:
                    dynamic_ok = False
                    break

        if dynamic_ok and not traffic_violations:
            best_wait = wait_s
            best_min_dynamic = min_dynamic
            break

    if best_wait is None:
        return None, {
            "traffic_wait_insert_index": first_crosswalk_idx,
            "reason": "no_wait_satisfies_traffic_and_dynamic_clearance",
            "max_wait_s": max_wait_s,
        }

    updated: List[Dict] = []

    def append_step(
        source_step: Dict,
        start: Point,
        end: Point,
        start_time: float,
        duration: float,
        action_type: str,
    ) -> None:
        static_clearance, static_obstacle = planner.static_segment_clearance(start, end)
        dynamic_clearance_value = float("inf")
        dynamic_agent = None
        if dynamic_clearance > 0.0 and planner.moving_agents and duration > 0.0:
            dynamic_clearance_value, dynamic_agent = planner.dynamic_segment_clearance(start, end, start_time, duration)

        traffic_checked = traffic_light_schedule.segment_uses_crosswalk(start, end)
        traffic_window_start = None
        traffic_window_end = None
        traffic_compliant = None
        if traffic_checked:
            traffic_compliant, window = traffic_light_schedule.allows_crossing(start_time, duration)
            traffic_window_start = window["traffic_window_start_s"]
            traffic_window_end = window["traffic_window_end_s"]

        updated.append(
            {
                "step": len(updated),
                "action_type": action_type,
                "start": start.as_dict(),
                "end": end.as_dict(),
                "start_time_s": round(start_time, 4),
                "duration_s": round(duration, 4),
                "end_time_s": round(start_time + duration, 4),
                "distance_cm": round(start.distance(end), 4),
                "static_clearance_cm": round(static_clearance, 4) if static_clearance != float("inf") else "inf",
                "dynamic_clearance_cm": round(dynamic_clearance_value, 4)
                if dynamic_clearance_value != float("inf")
                else "inf",
                "closest_static_id": static_obstacle.get("id") if static_obstacle else None,
                "closest_dynamic_id": dynamic_agent.id if dynamic_agent else None,
                "traffic_checked": traffic_checked,
                "traffic_compliant": traffic_compliant,
                "traffic_window_start_s": traffic_window_start,
                "traffic_window_end_s": traffic_window_end,
                "source_step": source_step.get("step"),
            }
        )

    for idx, step in enumerate(trajectory):
        if idx == first_crosswalk_idx and best_wait > 0.0:
            append_step(
                {"step": "traffic_wait"},
                wait_point,
                wait_point,
                wait_start_time,
                best_wait,
                "wait",
            )

        start = Point(float(step["start"]["x"]), float(step["start"]["y"]))
        end = Point(float(step["end"]["x"]), float(step["end"]["y"]))
        shifted_start_time = float(step["start_time_s"]) + (best_wait if idx >= first_crosswalk_idx else 0.0)
        append_step(step, start, end, shifted_start_time, float(step["duration_s"]), step["action_type"])

    checked_steps = [step["step"] for step in updated if step.get("traffic_checked")]
    violations = [
        step
        for step in updated
        if step.get("traffic_checked") and step.get("traffic_compliant") is False
    ]
    return updated, {
        "traffic_wait_inserted_s": round(best_wait, 4),
        "traffic_wait_insert_index": first_crosswalk_idx,
        "checked_crosswalk_steps": checked_steps,
        "num_checked_crosswalk_steps": len(checked_steps),
        "num_violations": len(violations),
        "violations": violations,
        "min_dynamic_clearance_after_wait_cm": round(best_min_dynamic, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", default="results/20260624_194515/task_1.json")
    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--difficulty", default=None, choices=sorted(DIFFICULTY_DYNAMIC_ACTIVATION_RATIO))
    parser.add_argument("--agent-speed", type=float, default=200.0)
    parser.add_argument("--grid-resolution", type=float, default=100.0)
    parser.add_argument("--corridor-width", type=float, default=650.0)
    parser.add_argument("--crosswalk-width", type=float, default=300.0)
    parser.add_argument("--road-side-tolerance", type=float, default=50.0)
    parser.add_argument("--static-clearance", type=float, default=100.0)
    parser.add_argument("--dynamic-clearance", type=float, default=200.0)
    parser.add_argument("--building-margin", type=float, default=20.0)
    parser.add_argument("--time-step", type=float, default=0.5)
    parser.add_argument("--max-time", type=float, default=180.0)
    parser.add_argument("--follow-traffic-lights", action="store_true")
    parser.add_argument("--vehicle-phase-count", type=int, default=3)
    parser.add_argument("--vehicle-green-duration", type=float, default=10.0)
    parser.add_argument("--yellow-duration", type=float, default=2.0)
    parser.add_argument("--pedestrian-green-duration", type=float, default=25.0)
    parser.add_argument("--traffic-safety-margin", type=float, default=1.0)
    parser.add_argument("--max-traffic-wait", type=float, default=180.0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source_result_path = repo_root / args.source_result
    with source_result_path.open("r", encoding="utf-8") as f:
        source_result = json.load(f)

    task_index = args.task_index if args.task_index is not None else int(source_result.get("task_id", 1))
    seed = args.seed if args.seed is not None else int(source_result.get("seed", 0))
    difficulty = args.difficulty or source_result.get("difficulty", "easy")

    task_file_path = repo_root / args.task_file
    with task_file_path.open("r", encoding="utf-8") as f:
        task_file = json.load(f)
    task = task_file["tasks"][task_index]
    scenario = convert_task_to_scenario(task)
    route = [point_from_xy(p) for p in scenario["route_info"]["shortest_path"]]
    start = point_from_xy(scenario["task"]["start_point"])
    goal = point_from_xy(scenario["task"]["end_point"])
    pedestrian_region = PedestrianRegion(
        raw_edges=scenario["task"].get("edges", []),
        route=route,
        sidewalk_width=args.corridor_width,
        crosswalk_width=args.crosswalk_width,
        road_side_tolerance=args.road_side_tolerance,
    )

    map_path = repo_root / scenario["map_path"]
    static_obstacles = load_static_obstacles(map_path.parent)

    rng = random.Random(seed + task_index)
    total_hops = int(scenario["task"].get("total_hops", 0))
    num_regular_candidates = PED_NUM_PER_EDGE * total_hops
    regular_pedestrians = build_regular_pedestrians(scenario, num_regular_candidates, difficulty, rng)
    irregular_pedestrians = build_irregular_pedestrians(
        scenario,
        difficulty,
        rng,
        regular_pedestrians,
        static_obstacles,
    )
    moving_agents = regular_pedestrians + irregular_pedestrians
    traffic_light_schedule = None
    if args.follow_traffic_lights:
        traffic_light_schedule = build_traffic_light_schedule(
            task_edges=scenario["task"].get("edges", []),
            crosswalk_width=args.crosswalk_width,
            vehicle_phase_count=args.vehicle_phase_count,
            vehicle_green_duration=args.vehicle_green_duration,
            yellow_duration=args.yellow_duration,
            pedestrian_green_duration=args.pedestrian_green_duration,
            safety_margin=args.traffic_safety_margin,
        )

    planner = OfflineSafePathPlanner(
        route=route,
        pedestrian_region=pedestrian_region,
        static_obstacles=static_obstacles,
        moving_agents=moving_agents,
        agent_speed=args.agent_speed,
        grid_resolution=args.grid_resolution,
        corridor_width=args.corridor_width,
        static_clearance=args.static_clearance,
        dynamic_clearance=args.dynamic_clearance,
        building_margin=args.building_margin,
        time_step=args.time_step,
        max_time=args.max_time,
        traffic_light_schedule=None,
    )
    trajectory, diagnostics = planner.plan(start, goal)
    success = trajectory is not None
    traffic_postprocess = None
    if success and trajectory and traffic_light_schedule:
        trajectory, traffic_postprocess = apply_traffic_light_waits(
            trajectory,
            planner,
            traffic_light_schedule,
            args.dynamic_clearance,
            args.time_step,
            args.max_traffic_wait,
        )
        success = trajectory is not None
        diagnostics["traffic_light_postprocess"] = traffic_postprocess

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "results" / f"offline_safe_path_{timestamp}"
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    total_distance = sum(step["distance_cm"] for step in trajectory) if trajectory else 0.0
    total_duration = trajectory[-1]["end_time_s"] if trajectory else None
    traffic_compliance = None
    if traffic_light_schedule and trajectory:
        checked_steps = []
        violations = []
        for step in trajectory:
            start_point = Point(float(step["start"]["x"]), float(step["start"]["y"]))
            end_point = Point(float(step["end"]["x"]), float(step["end"]["y"]))
            if not traffic_light_schedule.segment_uses_crosswalk(start_point, end_point):
                continue
            allowed, window = traffic_light_schedule.allows_crossing(
                float(step["start_time_s"]),
                float(step["duration_s"]),
            )
            checked_steps.append(step["step"])
            if not allowed:
                violations.append({"step": step["step"], **window})
        traffic_compliance = {
            "checked_crosswalk_steps": checked_steps,
            "num_checked_crosswalk_steps": len(checked_steps),
            "num_violations": len(violations),
            "violations": violations,
        }
    result = {
        "created_at": datetime.now().isoformat(),
        "source_result": str(source_result_path.relative_to(repo_root)),
        "task_file": str(task_file_path.relative_to(repo_root)),
        "task_index": task_index,
        "task_id": task.get("task_id"),
        "seed": seed,
        "difficulty": difficulty,
        "planner": {
            "type": "offline_time_expanded_astar",
            "agent_speed_cm_s": args.agent_speed,
            "grid_resolution_cm": args.grid_resolution,
            "sidewalk_width_cm": args.corridor_width,
            "crosswalk_width_cm": args.crosswalk_width,
            "road_side_tolerance_cm": args.road_side_tolerance,
            "static_clearance_cm": args.static_clearance,
            "dynamic_clearance_cm": args.dynamic_clearance,
            "building_margin_cm": args.building_margin,
            "time_step_s": args.time_step,
            "max_time_s": args.max_time,
            "follow_traffic_lights": bool(args.follow_traffic_lights),
            "traffic_light_schedule": traffic_light_schedule.as_dict() if traffic_light_schedule else None,
        },
        "environment": {
            "map_path": scenario["map_path"],
            "num_static_obstacles": sum(1 for ob in static_obstacles if ob.get("kind") == "static"),
            "num_buildings": sum(1 for ob in static_obstacles if ob.get("kind") == "building"),
            "num_regular_pedestrians": len(regular_pedestrians),
            "num_irregular_pedestrians": len(irregular_pedestrians),
            "num_moving_agents": len(moving_agents),
            "moving_agents": [agent.as_dict() for agent in moving_agents],
        },
        "route": {
            "start": start.as_dict(),
            "goal": goal.as_dict(),
            "shortest_path": [p.as_dict() for p in route],
            "task_edges": scenario["task"].get("edges", []),
            "shortest_path_length_cm": round(polyline_length(route), 4),
            "required_time_s": round(float(scenario["task"].get("required_time", 0.0)), 4),
        },
        "success": success,
        "summary": {
            "num_steps": len(trajectory) if trajectory else 0,
            "total_distance_cm": round(total_distance, 4),
            "total_duration_s": round(total_duration, 4) if total_duration is not None else None,
            "min_static_clearance_cm": min(
                (step["static_clearance_cm"] for step in trajectory if isinstance(step.get("static_clearance_cm"), (int, float))),
                default=None,
            )
            if trajectory
            else None,
            "min_dynamic_clearance_cm": min(
                (step["dynamic_clearance_cm"] for step in trajectory if isinstance(step.get("dynamic_clearance_cm"), (int, float))),
                default=None,
            )
            if trajectory
            else None,
            "traffic_light_compliance": traffic_compliance,
        },
        "diagnostics": diagnostics,
        "trajectory": trajectory or [],
    }
    write_outputs(output_dir, result, static_obstacles)

    print(json.dumps({
        "output_dir": str(output_dir),
        "success": success,
        "num_steps": result["summary"]["num_steps"],
        "total_duration_s": result["summary"]["total_duration_s"],
        "total_distance_cm": result["summary"]["total_distance_cm"],
        "diagnostics": diagnostics,
    }, indent=2))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
