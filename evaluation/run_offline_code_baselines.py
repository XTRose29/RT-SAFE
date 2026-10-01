#!/usr/bin/env python3
"""Run rule-based code baselines fully offline.

This runner does not launch UE and does not call a VLM. It reconstructs the
task route, static obstacles, regular pedestrians, and irregular moving agents
from the task/map JSON files, then simulates the agent action-by-action and
checks geometric collisions as time advances.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.offline_safe_path_planner import (  # noqa: E402
    DIFFICULTY_DYNAMIC_ACTIVATION_RATIO,
    PED_NUM_PER_EDGE,
    MovingAgent,
    Point,
    build_irregular_pedestrians,
    build_regular_pedestrians,
    build_traffic_light_schedule,
    convert_task_to_scenario,
    load_static_obstacles,
    point_from_xy,
    point_to_segment_distance,
    relative_segment_distance,
    segment_intersects_aabb,
    segment_to_aabb_distance,
)


TURN_ACTIONS = ("L30", "L60", "L90", "R30", "R60", "R90")
MOVE_RADII_AND_ANGLES = ((100.0, (0.0,)), (200.0, (0.0, 45.0, -45.0)), (400.0, (0.0, 45.0, -45.0)))
MAX_MOVE_RADIUS_CM = max(radius for radius, _ in MOVE_RADII_AND_ANGLES)


@dataclass(frozen=True)
class Action:
    action_type: str
    action_param: str
    end_position: Point
    end_yaw_deg: float
    distance_cm: float

    @property
    def action_id(self) -> str:
        if self.action_type == "wait":
            return f"WAIT_{self.action_param}"
        return self.action_param


@dataclass
class AgentState:
    position: Point
    yaw_deg: float
    subgoal_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=243)
    parser.add_argument("--difficulty", default="easy", choices=sorted(DIFFICULTY_DYNAMIC_ACTIVATION_RATIO))
    parser.add_argument("--baselines", nargs="+", default=["greedy", "safety"], choices=["greedy", "safety"])
    parser.add_argument("--env-modes", nargs="+", default=["realtime", "static"], choices=["realtime", "static"])
    parser.add_argument("--fixed-action-seconds", type=float, default=1.0)
    parser.add_argument("--internal-latency-seconds", type=float, default=4.0)
    parser.add_argument("--goal-threshold-cm", type=float, default=250.0)
    parser.add_argument("--collision-radius-cm", type=float, default=100.0)
    parser.add_argument("--static-collision-radius-cm", type=float, default=80.0)
    parser.add_argument("--building-margin-cm", type=float, default=0.0)
    parser.add_argument("--safety-lookahead-depth", type=int, default=4)
    parser.add_argument("--safety-beam-width", type=int, default=0)
    parser.add_argument(
        "--preplan-initial-latency",
        action="store_true",
        help="Begin the realtime clock before t=0 so the first action is already planned at task start.",
    )
    parser.add_argument(
        "--agent-adaptive-green-light",
        action="store_true",
        help="Treat traffic lights as green for the agent when its selected action reaches a crosswalk.",
    )
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--suite-name", default=None)
    return parser.parse_args()


def normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def bearing(start: Point, end: Point) -> float:
    return math.degrees(math.atan2(end.y - start.y, end.x - start.x))


def move_point(start: Point, yaw_deg: float, distance: float) -> Point:
    yaw = math.radians(yaw_deg)
    return Point(start.x + distance * math.cos(yaw), start.y + distance * math.sin(yaw))


def waypoint_actions(state: AgentState) -> List[Action]:
    actions: List[Action] = []
    for radius, angles in MOVE_RADII_AND_ANGLES:
        for rel_angle in angles:
            # Policy angles are positive to the visual left; UE yaw increases
            # to the visual right (left-handed world).
            target_yaw = normalize_angle(state.yaw_deg - rel_angle)
            end = move_point(state.position, target_yaw, radius)
            actions.append(Action("move_to", str(len(actions) + 1), end, target_yaw, radius))
    return actions


def turn_action(state: AgentState, param: str) -> Action:
    direction = param[0]
    angle = float(param[1:])
    yaw_delta = -angle if direction == "L" else angle
    return Action("turn_around", param, state.position, normalize_angle(state.yaw_deg + yaw_delta), 0.0)


def wait_action(state: AgentState) -> Action:
    return Action("wait", "1", state.position, state.yaw_deg, 0.0)


def all_actions(state: AgentState) -> List[Action]:
    return waypoint_actions(state) + [turn_action(state, param) for param in TURN_ACTIONS] + [wait_action(state)]


def waypoint_completion_action(state: AgentState, route: Sequence[Point], goal_threshold: float) -> Optional[Action]:
    if state.subgoal_index >= len(route):
        return None
    target = route[state.subgoal_index]
    distance = state.position.distance(target)
    if distance <= goal_threshold or distance > MAX_MOVE_RADIUS_CM:
        return None
    return Action("move_to", "target", target, bearing(state.position, target), distance)


def relative_angle_to(state: AgentState, target: Point) -> float:
    """Policy-convention angle to ``target``: positive is visual left."""
    return normalize_angle(state.yaw_deg - bearing(state.position, target))


def greedy_action(state: AgentState, route: Sequence[Point]) -> Tuple[Action, Dict[str, Any]]:
    target = route[state.subgoal_index]
    rel_angle = relative_angle_to(state, target)
    if abs(rel_angle) > 20.0:
        abs_angle = abs(rel_angle)
        if abs_angle > 75.0:
            angle = 90
        elif abs_angle > 45.0:
            angle = 60
        else:
            angle = 30
        direction = "L" if rel_angle > 0.0 else "R"
        action = turn_action(state, f"{direction}{angle}")
        return action, {
            "selection_rule": "turn_to_active_subgoal",
            "relative_angle_deg": round(rel_angle, 4),
            "active_subgoal_index": state.subgoal_index,
        }

    current_distance = state.position.distance(target)
    scored = []
    for action in waypoint_actions(state):
        progress = current_distance - action.end_position.distance(target)
        step_dx = action.end_position.x - state.position.x
        step_dy = action.end_position.y - state.position.y
        target_dx = target.x - state.position.x
        target_dy = target.y - state.position.y
        step_len = math.hypot(step_dx, step_dy)
        target_len = math.hypot(target_dx, target_dy)
        alignment = 0.0
        if step_len > 1e-6 and target_len > 1e-6:
            alignment = (step_dx * target_dx + step_dy * target_dy) / (step_len * target_len)
        scored.append((alignment, action.distance_cm, progress, -action.end_position.distance(target), action))
    best = max(scored, key=lambda item: item[:4])
    if best[2] <= 1e-3 and current_distance > 250.0:
        direction = "L" if rel_angle > 0.0 else "R"
        action = turn_action(state, f"{direction}30")
        return action, {
            "selection_rule": "turn_when_no_move_reduces_subgoal_distance",
            "relative_angle_deg": round(rel_angle, 4),
            "active_subgoal_index": state.subgoal_index,
        }
    action = best[4]
    return action, {
        "selection_rule": "shortest_direct_move_to_active_subgoal",
        "relative_angle_deg": round(rel_angle, 4),
        "active_subgoal_index": state.subgoal_index,
        "selected_progress_cm": round(best[2], 4),
        "selected_alignment": round(best[0], 6),
    }


def static_hits(
    start: Point,
    end: Point,
    static_obstacles: Sequence[Dict[str, Any]],
    static_radius: float,
    building_margin: float,
) -> List[Dict[str, Any]]:
    hits = []
    for obstacle in static_obstacles:
        if obstacle.get("bounds"):
            distance = segment_to_aabb_distance(start, end, obstacle["bounds"])
            hit = segment_intersects_aabb(start, end, obstacle["bounds"], building_margin)
        else:
            distance = point_to_segment_distance(obstacle["point"], start, end)
            hit = distance <= static_radius
        if hit:
            hits.append(
                {
                    "id": obstacle.get("id"),
                    "kind": obstacle.get("kind", "static"),
                    "type": obstacle.get("type"),
                    "distance_cm": round(distance, 4),
                }
            )
    return hits


def dynamic_hits(
    start: Point,
    end: Point,
    start_time: float,
    duration: float,
    moving_agents: Sequence[MovingAgent],
    collision_radius: float,
) -> List[Dict[str, Any]]:
    if duration <= 1e-9:
        duration = 1e-9
    end_time = start_time + duration
    hits = []
    for other in moving_agents:
        other_start = other.position_at(start_time)
        other_end = other.position_at(end_time)
        distance = relative_segment_distance(start, end, other_start, other_end)
        if distance <= collision_radius:
            hits.append(
                {
                    "id": other.id,
                    "kind": other.kind,
                    "distance_cm": round(distance, 4),
                    "other_start": other_start.as_dict(),
                    "other_end": other_end.as_dict(),
                }
            )
    return hits


def count_new_contacts(hits: Sequence[Dict[str, Any]], active_contacts: Set[str]) -> Tuple[Set[str], Set[str]]:
    current = {str(hit["id"]) for hit in hits}
    new_contacts = current - active_contacts
    return new_contacts, current


def collision_preview(
    action: Action,
    state: AgentState,
    time_s: float,
    env_mode: str,
    latency: float,
    action_duration: float,
    static_obstacles: Sequence[Dict[str, Any]],
    moving_agents: Sequence[MovingAgent],
    collision_radius: float,
    static_radius: float,
    building_margin: float,
    traffic_light_schedule: Any = None,
    agent_adaptive_green_light: bool = False,
) -> Dict[str, Any]:
    passive_hits = []
    passive_count = 0
    action_start = time_s
    if env_mode == "realtime" and latency > 0.0:
        passive_hits = dynamic_hits(state.position, state.position, time_s, latency, moving_agents, collision_radius)
        passive_count = len(passive_hits)
        action_start += latency
    action_dynamic_hits = dynamic_hits(
        state.position,
        action.end_position,
        action_start,
        action_duration,
        moving_agents,
        collision_radius,
    )
    action_static_hits = static_hits(
        state.position,
        action.end_position,
        static_obstacles,
        static_radius,
        building_margin,
    )
    traffic_checked = False
    traffic_allowed = True
    traffic_details: Dict[str, Any] = {}
    if (
        action.action_type == "move_to"
        and traffic_light_schedule
        and traffic_light_schedule.segment_uses_crosswalk(state.position, action.end_position)
    ):
        traffic_checked = True
        if agent_adaptive_green_light:
            traffic_allowed = True
            traffic_details = {"adaptive_agent_green_light": True}
        else:
            traffic_allowed, traffic_details = traffic_light_schedule.allows_crossing(action_start, action_duration)
    return {
        "predicted_collision_count": passive_count + len(action_dynamic_hits) + len(action_static_hits),
        "predicted_red_light_violation": 1 if traffic_checked and not traffic_allowed else 0,
        "traffic_checked": traffic_checked,
        "traffic_allowed": traffic_allowed,
        "traffic_details": traffic_details,
        "passive_dynamic_hits": passive_hits,
        "action_dynamic_hits": action_dynamic_hits,
        "action_static_hits": action_static_hits,
        "min_dynamic_clearance_cm": min(
            [hit["distance_cm"] for hit in passive_hits + action_dynamic_hits],
            default=float("inf"),
        ),
        "static_collision_count": len(action_static_hits),
        "dynamic_collision_count": passive_count + len(action_dynamic_hits),
    }


def safety_action(
    state: AgentState,
    route: Sequence[Point],
    time_s: float,
    env_mode: str,
    latency: float,
    action_duration: float,
    static_obstacles: Sequence[Dict[str, Any]],
    moving_agents: Sequence[MovingAgent],
    collision_radius: float,
    static_radius: float,
    building_margin: float,
    traffic_light_schedule: Any = None,
    goal_threshold: float = 250.0,
    lookahead_depth: int = 2,
    beam_width: int = 0,
    agent_adaptive_green_light: bool = False,
) -> Tuple[Action, Dict[str, Any]]:
    target = route[state.subgoal_index]
    current_distance = state.position.distance(target)
    greedy, greedy_details = greedy_action(state, route)

    def advanced_index(position: Point, subgoal_index: int) -> int:
        idx = subgoal_index
        while idx < len(route) and position.distance(route[idx]) < goal_threshold:
            idx += 1
        return idx

    def score_action(current_state: AgentState, current_time: float, action: Action) -> Tuple[Dict[str, Any], AgentState, float, float]:
        preview = collision_preview(
            action,
            current_state,
            current_time,
            env_mode,
            latency,
            action_duration,
            static_obstacles,
            moving_agents,
            collision_radius,
            static_radius,
            building_margin,
            traffic_light_schedule,
            agent_adaptive_green_light,
        )
        next_time = current_time + action_duration + (latency if env_mode == "realtime" else 0.0)
        next_subgoal = advanced_index(action.end_position, current_state.subgoal_index)
        next_state = AgentState(action.end_position, action.end_yaw_deg, next_subgoal)
        current_target = route[current_state.subgoal_index] if current_state.subgoal_index < len(route) else route[-1]
        progress = current_state.position.distance(current_target) - action.end_position.distance(current_target)
        return preview, next_state, next_time, progress

    def route_distance(current_state: AgentState) -> float:
        if current_state.subgoal_index >= len(route):
            return 0.0
        return current_state.position.distance(route[current_state.subgoal_index])

    def action_priority(action: Action, progress: float, next_state: AgentState, current_state: AgentState) -> int:
        if action.action_type == "move_to" and (progress > 1e-3 or next_state.subgoal_index > current_state.subgoal_index):
            return 0
        if action.action_type == "turn_around":
            return 1
        if action.action_type == "wait":
            return 2
        return 3

    def candidate_sort_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            item["collisions"],
            item["red"],
            item["state"].subgoal_index < len(route),
            -item["state"].subgoal_index,
            item["action_priority"],
            item["end_distance"],
            -item["progress"],
            item["path_len"],
        )

    def action_candidates(current_state: AgentState, current_time: float, depth: int) -> List[Dict[str, Any]]:
        candidates = []
        actions = all_actions(current_state)
        direct_target_action = waypoint_completion_action(current_state, route, goal_threshold)
        if direct_target_action is not None:
            actions.insert(0, direct_target_action)
        for action in actions:
            preview, next_state, next_time, progress = score_action(current_state, current_time, action)
            candidates.append(
                {
                    "state": next_state,
                    "time": next_time,
                    "first_action": action,
                    "first_preview": preview,
                    "first_progress": progress,
                    "collisions": preview["predicted_collision_count"],
                    "red": preview["predicted_red_light_violation"],
                    "progress": progress,
                    "path_len": 1,
                    "end_distance": route_distance(next_state),
                    "action_priority": action_priority(action, progress, next_state, current_state),
                }
            )
        candidates.sort(key=candidate_sort_key)
        if beam_width > 0 and depth > 1:
            candidates = candidates[:beam_width]

        expanded = []
        for candidate in candidates:
            next_state = candidate["state"]
            next_time = candidate["time"]
            expanded_candidate = {
                "state": next_state,
                "time": next_time,
                "first_action": candidate["first_action"],
                "first_preview": candidate["first_preview"],
                "first_progress": candidate["first_progress"],
                "collisions": candidate["collisions"],
                "red": candidate["red"],
                "progress": candidate["progress"],
                "path_len": 1,
                "end_distance": route_distance(next_state),
                "action_priority": candidate["action_priority"],
            }
            if depth > 1 and next_state.subgoal_index < len(route):
                followups = action_candidates(next_state, next_time, depth - 1)
                if followups:
                    next_best = followups[0]
                    expanded_candidate["collisions"] += next_best["collisions"]
                    expanded_candidate["red"] += next_best["red"]
                    expanded_candidate["progress"] += next_best["progress"]
                    expanded_candidate["path_len"] += next_best["path_len"]
                    expanded_candidate["end_distance"] = next_best["end_distance"]
            expanded.append(expanded_candidate)
        expanded.sort(key=candidate_sort_key)
        return expanded

    candidates = action_candidates(state, time_s, max(1, lookahead_depth))
    best = candidates[0]
    preview = best["first_preview"]
    return best["first_action"], {
        "selection_rule": "deep_lookahead_minimize_collisions_then_red_lights_then_route_distance",
        "greedy": greedy_details,
        "active_subgoal_index": state.subgoal_index,
        "predicted_red_light_violation": preview["predicted_red_light_violation"],
        "predicted_collision_count": preview["predicted_collision_count"],
        "predicted_progress_cm": round(best["first_progress"], 4),
        "lookahead_depth": lookahead_depth,
        "beam_width": beam_width,
        "lookahead_total_predicted_collisions": best["collisions"],
        "lookahead_total_predicted_red_lights": best["red"],
        "candidate_count": len(candidates),
        "action_priority": best["action_priority"],
    }


def load_movable_obstacle_ids(map_dir: Path) -> List[str]:
    obstacles_path = map_dir / "obstacles.json"
    if not obstacles_path.exists():
        return []
    with obstacles_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [node.get("id", "") for node in data.get("nodes", []) if str(node.get("id", "")).startswith("GEN_RT_RT_")]


def build_environment(task_file: Path, task_index: int, seed: int, difficulty: str) -> Dict[str, Any]:
    with task_file.open("r", encoding="utf-8") as f:
        tasks_data = json.load(f)
    task = tasks_data["tasks"][task_index]
    scenario = convert_task_to_scenario(task)
    route = [point_from_xy(p) for p in scenario["route_info"]["shortest_path"]]
    map_path = REPO_ROOT / scenario["map_path"]
    map_dir = map_path.parent
    static_obstacles = load_static_obstacles(map_dir)
    rng = random.Random(seed + task_index)
    total_hops = int(scenario["task"].get("total_hops", 0))
    num_regular_candidates = PED_NUM_PER_EDGE * total_hops
    regular = build_regular_pedestrians(scenario, num_regular_candidates, difficulty, rng)
    irregular = build_irregular_pedestrians(scenario, difficulty, rng, regular, static_obstacles)
    traffic_light_schedule = build_traffic_light_schedule(
        scenario["task"].get("edges", []),
        crosswalk_width=250.0,
        vehicle_phase_count=3,
        vehicle_green_duration=10.0,
        yellow_duration=2.0,
        pedestrian_green_duration=25.0,
        safety_margin=1.0,
    )
    movable_ids = load_movable_obstacle_ids(map_dir)
    ratio = DIFFICULTY_DYNAMIC_ACTIVATION_RATIO[difficulty]
    activated_movable_count = int(round(len(movable_ids) * ratio))
    falling_count = len(scenario.get("falling_objects", []))
    return {
        "task": task,
        "scenario": scenario,
        "route": route,
        "static_obstacles": static_obstacles,
        "moving_agents": regular + irregular,
        "regular_pedestrians": regular,
        "irregular_agents": irregular,
        "traffic_light_schedule": traffic_light_schedule,
        "map_dir": map_dir,
        "unmodeled_assets": {
            "movable_obstacle_ids_total": len(movable_ids),
            "movable_obstacle_ids_activated_by_ratio": activated_movable_count,
            "falling_objects_total": falling_count,
            "falling_objects_activated_by_ratio": int(round(falling_count * ratio)),
            "note": "UE-only movable/falling dynamics are recorded here but not geometrically simulated offline.",
        },
    }


def maybe_advance_subgoal(state: AgentState, route: Sequence[Point], threshold: float) -> Tuple[bool, List[int]]:
    reached = []
    while state.subgoal_index < len(route) and state.position.distance(route[state.subgoal_index]) < threshold:
        reached.append(state.subgoal_index)
        state.subgoal_index += 1
    return state.subgoal_index >= len(route), reached


def run_baseline(
    baseline: str,
    env_mode: str,
    environment: Dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
) -> Dict[str, Any]:
    route = environment["route"]
    if len(route) < 2:
        raise ValueError("Route must have at least two points.")
    state = AgentState(route[0], bearing(route[0], route[1]), 1)
    sim_time = (
        -args.internal_latency_seconds
        if env_mode == "realtime" and args.preplan_initial_latency and args.internal_latency_seconds > 0.0
        else 0.0
    )
    active_dynamic_contacts: Set[str] = set()
    active_static_contacts: Set[str] = set()
    collision_count = 0
    human_collision_count = 0
    static_collision_count = 0
    building_collision_count = 0
    passive_collision_count = 0
    red_light_violation_count = 0
    decisions: List[Dict[str, Any]] = []
    success, initial_reached = maybe_advance_subgoal(state, route, args.goal_threshold_cm)

    for step in range(args.max_steps):
        if success:
            break
        action_start_observation_time = sim_time
        if baseline == "greedy":
            action, details = greedy_action(state, route)
        else:
            action, details = safety_action(
                state,
                route,
                sim_time,
                env_mode,
                args.internal_latency_seconds,
                args.fixed_action_seconds,
                environment["static_obstacles"],
                environment["moving_agents"],
                args.collision_radius_cm,
                args.static_collision_radius_cm,
                args.building_margin_cm,
                environment["traffic_light_schedule"],
                args.goal_threshold_cm,
                args.safety_lookahead_depth,
                args.safety_beam_width,
                args.agent_adaptive_green_light,
            )

        passive_hits = []
        if env_mode == "realtime" and args.internal_latency_seconds > 0.0:
            passive_hits = dynamic_hits(
                state.position,
                state.position,
                sim_time,
                args.internal_latency_seconds,
                environment["moving_agents"],
                args.collision_radius_cm,
            )
            new_contacts, active_dynamic_contacts = count_new_contacts(passive_hits, active_dynamic_contacts)
            collision_count += len(new_contacts)
            human_collision_count += sum(1 for hit in passive_hits if str(hit["id"]) in new_contacts and hit["kind"] == "pedestrian")
            passive_collision_count += len(new_contacts)
            sim_time += args.internal_latency_seconds

        move_start = state.position
        move_end = action.end_position
        dynamic_action_hits = dynamic_hits(
            move_start,
            move_end,
            sim_time,
            args.fixed_action_seconds,
            environment["moving_agents"],
            args.collision_radius_cm,
        )
        new_dynamic_contacts, active_dynamic_contacts = count_new_contacts(dynamic_action_hits, active_dynamic_contacts)
        collision_count += len(new_dynamic_contacts)
        human_collision_count += sum(
            1 for hit in dynamic_action_hits if str(hit["id"]) in new_dynamic_contacts and hit["kind"] == "pedestrian"
        )

        static_action_hits = static_hits(
            move_start,
            move_end,
            environment["static_obstacles"],
            args.static_collision_radius_cm,
            args.building_margin_cm,
        )
        new_static_contacts, active_static_contacts = count_new_contacts(static_action_hits, active_static_contacts)
        collision_count += len(new_static_contacts)
        static_collision_count += len(new_static_contacts)
        building_collision_count += sum(
            1 for hit in static_action_hits if str(hit["id"]) in new_static_contacts and hit["kind"] == "building"
        )

        traffic_checked = False
        traffic_allowed = True
        if (
            action.action_type == "move_to"
            and environment["traffic_light_schedule"]
            and environment["traffic_light_schedule"].segment_uses_crosswalk(move_start, move_end)
        ):
            traffic_checked = True
            if args.agent_adaptive_green_light:
                traffic_allowed = True
            else:
                traffic_allowed, _ = environment["traffic_light_schedule"].allows_crossing(sim_time, args.fixed_action_seconds)
            if not traffic_allowed:
                red_light_violation_count += 1

        sim_time += args.fixed_action_seconds
        state.position = action.end_position
        state.yaw_deg = action.end_yaw_deg
        success, reached_subgoals = maybe_advance_subgoal(state, route, args.goal_threshold_cm)
        if not dynamic_action_hits and not passive_hits:
            active_dynamic_contacts = set()
        if not static_action_hits:
            active_static_contacts = set()

        decisions.append(
            {
                "step": step + 1,
                "observation_time_s": round(action_start_observation_time, 4),
                "action_start_time_s": round(sim_time - args.fixed_action_seconds, 4),
                "action_end_time_s": round(sim_time, 4),
                "action_type": action.action_type,
                "action_param": action.action_param,
                "start": move_start.as_dict(),
                "end": move_end.as_dict(),
                "yaw_end_deg": round(state.yaw_deg, 4),
                "active_subgoal_index_after": state.subgoal_index,
                "reached_subgoal_indices": reached_subgoals,
                "policy_details": details,
                "passive_dynamic_hits": passive_hits,
                "dynamic_action_hits": dynamic_action_hits,
                "static_action_hits": static_action_hits,
                "traffic_checked": traffic_checked,
                "traffic_allowed": traffic_allowed,
            }
        )

    final_distance = state.position.distance(route[-1])
    result = {
        "status": "completed",
        "condition_id": f"{baseline}__{env_mode}__offline",
        "setting_id": f"{baseline}__{env_mode}__offline",
        "baseline_policy": baseline,
        "env_mode": env_mode,
        "offline_no_ue": True,
        "task_file": args.task_file,
        "task_index": args.task_index,
        "task_id": environment["task"].get("task_id"),
        "difficulty": args.difficulty,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "fixed_action_seconds": args.fixed_action_seconds,
        "internal_latency_seconds": args.internal_latency_seconds,
        "agent_adaptive_green_light": args.agent_adaptive_green_light,
        "success": bool(success),
        "final_step": len(decisions),
        "decision_count": len(decisions),
        "sim_time_s": round(sim_time, 4),
        "wall_clock_modeled_time_s": round(sim_time + (0.0 if env_mode == "realtime" else len(decisions) * args.internal_latency_seconds), 4),
        "final_position": state.position.as_dict(),
        "final_distance_to_goal_cm": round(final_distance, 4),
        "collision_count": collision_count,
        "passive_collision_count": passive_collision_count,
        "human_collision_count": human_collision_count,
        "static_collision_count": static_collision_count,
        "building_collision_count": building_collision_count,
        "red_light_violation_count": red_light_violation_count,
        "fall_count": 0,
        "route": {"shortest_path": [p.as_dict() for p in route]},
        "environment_counts": {
            "regular_pedestrians": len(environment["regular_pedestrians"]),
            "irregular_agents": len(environment["irregular_agents"]),
            "moving_agents_modeled": len(environment["moving_agents"]),
            "static_obstacles_modeled": len(environment["static_obstacles"]),
            **environment["unmodeled_assets"],
        },
        "traffic_light_schedule": environment["traffic_light_schedule"].as_dict()
        if environment["traffic_light_schedule"]
        else None,
        "decisions": decisions,
        "created_at": datetime.now().isoformat(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["result_path"] = str(result_path.relative_to(REPO_ROOT))
    return result


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    suite_name = args.suite_name or f"code_baselines_offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir = REPO_ROOT / args.output_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    task_file = (REPO_ROOT / args.task_file).resolve()
    environment = build_environment(task_file, args.task_index, args.seed, args.difficulty)
    manifest = {
        "created_at": datetime.now().isoformat(),
        "suite_name": suite_name,
        "offline_no_ue": True,
        "reference_settings": {
            "dynamic": "results/building_collision_feedback_simple_tl_prompt_qwen3vl8b_dynamic_easy_20260630_gate_rerun_fixed_tlcheck_20260630",
            "static": "results/building_collision_feedback_simple_tl_prompt_qwen3vl8b_static_easy_20260701_static_prompt",
        },
        "args": vars(args),
        "environment_counts": {
            "regular_pedestrians": len(environment["regular_pedestrians"]),
            "irregular_agents": len(environment["irregular_agents"]),
            "moving_agents_modeled": len(environment["moving_agents"]),
            "static_obstacles_modeled": len(environment["static_obstacles"]),
            **environment["unmodeled_assets"],
        },
    }
    with (suite_dir / "experiment_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary_path = suite_dir / "summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()
    for baseline in args.baselines:
        for env_mode in args.env_modes:
            run_dir = suite_dir / "runs" / f"{baseline}__{env_mode}__offline" / "round_01"
            result = run_baseline(baseline, env_mode, environment, args, run_dir)
            summary = {
                key: result[key]
                for key in (
                    "condition_id",
                    "setting_id",
                    "baseline_policy",
                    "env_mode",
                    "offline_no_ue",
                    "success",
                    "final_step",
                    "decision_count",
                    "sim_time_s",
                    "wall_clock_modeled_time_s",
                    "final_distance_to_goal_cm",
                    "collision_count",
                    "passive_collision_count",
                    "human_collision_count",
                    "static_collision_count",
                    "building_collision_count",
                    "red_light_violation_count",
                    "fall_count",
                    "result_path",
                )
            }
            append_jsonl(summary_path, summary)
            print(json.dumps(summary, sort_keys=True))
    print(f"Wrote offline baseline suite: {suite_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
