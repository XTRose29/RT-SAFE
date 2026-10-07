#!/usr/bin/env python3
"""Replay an offline safe trajectory in UE and record camera frames/video.

This script uses UE/UnrealCV for rendering only. It does not call a VLM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from types import MethodType
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SIMWORLD_ROOT = REPO_ROOT / "SimWorld"
if str(SIMWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMWORLD_ROOT))

from base.rt_agent import RTAgent
from base.rt_communicator import RTCommunicator
from base.rt_unrealcv import RTUnrealCV
from manager.world_manager import SLOMO, TIME_ALPHA, TIME_BETA, WorldManager
from simworld.utils.vector import Vector


COLLISION_KEYS = ("human", "object", "building")


class ReplayAgent:
    def __init__(self, position: Vector, direction: Vector, speed: float = 200.0):
        self.id = 0
        self.position = position
        self.direction = direction
        self.speed = speed
        self.name = "RT_AGENT"
        self.camera_id = 1


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def vector_from_dict(point: Dict[str, float]) -> Vector:
    return Vector(float(point["x"]), float(point["y"]))


def yaw_degrees(start: Vector, end: Vector, fallback_yaw: float) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return fallback_yaw
    return math.degrees(math.atan2(dy, dx))


def install_async_unrealcv_commands(communicator: RTCommunicator, spawn_delay_s: float = 0.08) -> None:
    """Use fire-and-forget UE commands for full-world video setup/replay.

    Some packaged UE blueprint commands execute but do not reliably send a reply,
    which makes UnrealCV's synchronous request wait forever. Camera reads remain
    synchronous; this only patches setup/movement commands that do not need a
    return value.
    """

    unrealcv = communicator.unrealcv

    def async_request(cmd: str) -> None:
        with unrealcv.lock:
            unrealcv.client.request(cmd, -1)

    def spawn_bp_asset(self, prefab_path, name):
        async_request(f"vset /objects/spawn_bp_asset {prefab_path} {name}")
        time.sleep(spawn_delay_s)

    def set_location(self, loc, name):
        x, y, z = loc
        async_request(f"vset /object/{name}/location {x} {y} {z}")

    def set_orientation(self, orientation, name):
        pitch, yaw, roll = orientation
        async_request(f"vset /object/{name}/rotation {pitch} {yaw} {roll}")

    def set_scale(self, scale, name):
        x, y, z = scale
        async_request(f"vset /object/{name}/scale {x} {y} {z}")

    def set_collision(self, actor_name, has_collision):
        async_request(f"vset /object/{actor_name}/collision {has_collision}")

    def set_movable(self, actor_name, is_movable):
        async_request(f"vset /object/{actor_name}/object_mobility {is_movable}")

    def set_camera_resolution(self, camera_id, resolution):
        width, height = resolution
        async_request(f"vset /camera/{camera_id}/size {width} {height}")

    def set_camera_fov(self, camera_id, fov):
        async_request(f"vset /camera/{camera_id}/fov {fov}")

    def humanoid_set_speed(self, object_name, speed):
        async_request(f"vbp {object_name} SetMaxSpeed {speed}")

    def rt_set_pedestrian_speed(self, name, speed):
        async_request(f"vbp {name} SetMaxSpeed {speed}")

    def p_set_waypoints(self, object_name, waypoints):
        async_request(f"vbp {object_name} SetWaypoints {waypoints}")

    def p_movement_simulation(self, object_name):
        async_request(f"vbp {object_name} MovementSimulation")

    def activate_object_movement(self, name):
        async_request(f"vbp {name} ActivateMovement")

    def rt_agent_move_to(self, name, x, y, duration):
        async_request(f"vbp {name} MoveTo {x},{y} {duration}")

    def add_vehicle_signal(self, intersection_name, vehicle_signal_name):
        async_request(f"vbp {intersection_name} AddVehicleSignal {vehicle_signal_name}")

    def add_pedestrian_signal(self, intersection_name, pedestrian_signal_name):
        async_request(f"vbp {intersection_name} AddPedSignal {pedestrian_signal_name}")

    def traffic_signal_start_simulation(self, intersection_name):
        async_request(f"vbp {intersection_name} StartSimulation")

    for name, func in {
        "spawn_bp_asset": spawn_bp_asset,
        "set_location": set_location,
        "set_orientation": set_orientation,
        "set_scale": set_scale,
        "set_collision": set_collision,
        "set_movable": set_movable,
        "set_camera_resolution": set_camera_resolution,
        "set_camera_fov": set_camera_fov,
        "humanoid_set_speed": humanoid_set_speed,
        "rt_set_pedestrian_speed": rt_set_pedestrian_speed,
        "p_set_waypoints": p_set_waypoints,
        "p_movement_simulation": p_movement_simulation,
        "activate_object_movement": activate_object_movement,
        "rt_agent_move_to": rt_agent_move_to,
        "add_vehicle_signal": add_vehicle_signal,
        "add_pedestrian_signal": add_pedestrian_signal,
        "traffic_signal_start_simulation": traffic_signal_start_simulation,
    }.items():
        setattr(unrealcv, name, MethodType(func, unrealcv))


def connect_unrealcv(port: int, ip: str, retries: int, delay_s: float) -> RTUnrealCV:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return RTUnrealCV(port=port, ip=ip)
        except Exception as exc:  # UnrealCV raises different socket errors by version.
            last_error = exc
            print(f"Waiting for UE/UnrealCV ({attempt}/{retries}): {exc}")
            time.sleep(delay_s)
    raise RuntimeError(f"Could not connect to UnrealCV at {ip}:{port}") from last_error


def setup_world(
    communicator: RTCommunicator,
    trajectory_data: Dict,
    source_result: Dict,
    task_file: str,
    task_index: int,
    seed: int,
    difficulty: str,
    use_tick: bool,
) -> WorldManager:
    original_set_game_speed = communicator.set_game_speed
    if not use_tick:
        # In full-world real-time video mode the packaged UE build can block on
        # rt_set_game_speed(1). Normal speed is already the default, so skip it.
        communicator.set_game_speed = lambda scale: None
    try:
        world_manager = WorldManager(
            communicator,
            task_file_path=task_file,
            seed=seed,
            control_mode="replay",
            token_based=False,
            use_tick=use_tick,
            realtime_thinking=True,
            record_per_step=False,
            use_action_frames=False,
            prompt_style="instructional",
            pedestrians_enabled=True,
            movable_obstacles_enabled=True,
            irregular_npcs_enabled=True,
            falling_objects_enabled=True,
            max_steps=0,
        )
    finally:
        communicator.set_game_speed = original_set_game_speed

    world_manager.difficulty = world_manager._normalize_difficulty_label(difficulty)
    world_manager.difficulty_config = world_manager._get_difficulty_config()
    random.seed(seed + task_index)
    world_manager.scenario_data = world_manager.all_scenarios[task_index]
    world_manager.current_task_id = world_manager.scenario_data.get("task_id", task_index)
    world_manager._initialize_world()
    world_manager.traffic_controller.init_communicator(communicator)

    task_data = world_manager.scenario_data["task"]
    start_point = vector_from_dict(trajectory_data["route"]["start"])
    end_point = vector_from_dict(trajectory_data["route"]["goal"])
    first_step_end = vector_from_dict(trajectory_data["trajectory"][0]["end"])
    initial_direction = Vector(first_step_end.x - start_point.x, first_step_end.y - start_point.y).normalize()
    shortest_path = [Vector(x, y) for x, y in world_manager.scenario_data["route_info"]["shortest_path"][1:]]

    world_manager.agent = RTAgent(
        start_point,
        initial_direction,
        end_point,
        shortest_path,
        task_data.get("required_time", source_result.get("required_time", 1000)),
        communicator,
        llm=None,
        token_based=False,
        use_tick=True,
        time_alpha=TIME_ALPHA,
        time_beta=TIME_BETA,
        realtime_thinking=True,
        task_edges=task_data.get("edges", []),
        traffic_signals=world_manager.traffic_controller.traffic_signals,
        static_obstacles=world_manager.static_obstacles,
        record_per_step=False,
        record_dir=None,
        slomo=SLOMO,
        use_action_frames=False,
        prompt_style="instructional",
        dynamic_obstacles=world_manager.dynamic_obstacle_metadata,
    )
    world_manager.spawn_agents()
    return world_manager


def setup_agent_only(communicator: RTCommunicator, trajectory_data: Dict) -> ReplayAgent:
    start_point = vector_from_dict(trajectory_data["route"]["start"])
    first_step_end = vector_from_dict(trajectory_data["trajectory"][0]["end"])
    initial_direction = Vector(first_step_end.x - start_point.x, first_step_end.y - start_point.y).normalize()
    agent = ReplayAgent(start_point, initial_direction)
    communicator.unrealcv.set_mode("sync")
    communicator.set_game_speed(SLOMO)
    communicator.spawn_agent(agent, agent.name, "/Game/RealTimeBench/Agent/BP_RT_Agent.BP_RT_Agent_C", type="agent")
    communicator.rt_agent_adjust_speed(agent.name, agent.speed)
    time.sleep(1.0)
    return agent


def capture_frame(communicator: RTCommunicator, camera_id: int, frame_path: Path) -> bool:
    image = communicator.get_camera_observation(camera_id, "lit")
    if image is None:
        return False
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(frame_path), image))


def read_runtime_state(communicator: RTCommunicator, agent_name: str) -> Dict:
    human, obj, building, intensity, overlap_type, touched_road = communicator.get_states(agent_name)
    return {
        "human": human,
        "object": obj,
        "building": building,
        "intensity": intensity,
        "overlap_type": overlap_type,
        "touched_road": touched_road,
    }


def state_delta(current: Dict, baseline: Dict) -> Dict:
    return {key: int(current.get(key, 0) - baseline.get(key, 0)) for key in COLLISION_KEYS}


def total_collision_delta(delta: Dict) -> int:
    return sum(int(delta.get(key, 0)) for key in COLLISION_KEYS)


class RuntimeValidator:
    def __init__(
        self,
        communicator: RTCommunicator,
        agent,
        world_manager: Optional[WorldManager],
        sample_interval_s: float,
    ) -> None:
        self.communicator = communicator
        self.agent = agent
        self.world_manager = world_manager
        self.sample_interval_s = max(0.05, sample_interval_s)
        self.baseline_state: Optional[Dict] = None
        self.samples: List[Dict] = []
        self.collision_events: List[Dict] = []
        self.traffic_checks: List[Dict] = []
        self.max_collision_delta = {key: 0 for key in COLLISION_KEYS}
        self.max_intensity = 0.0
        self.touched_road_samples = 0
        self.overlap_samples: Dict[str, int] = {}
        self.last_red_light_count = getattr(agent, "red_light_violations_count", 0)

    def initialize(self) -> None:
        self.baseline_state = read_runtime_state(self.communicator, self.agent.name)
        self.sample("initial", None)

    def check_traffic_before_move(self, step: Dict, phase: str) -> None:
        before = getattr(self.agent, "red_light_violations_count", 0)
        violation = False
        if hasattr(self.agent, "_check_red_light_violation"):
            violation = bool(self.agent._check_red_light_violation())
        after = getattr(self.agent, "red_light_violations_count", before)
        self.traffic_checks.append(
            {
                "phase": phase,
                "step": step.get("step"),
                "source_action_type": step.get("action_type"),
                "agent_position": {
                    "x": round(float(getattr(self.agent.position, "x", 0.0)), 4),
                    "y": round(float(getattr(self.agent.position, "y", 0.0)), 4),
                },
                "red_light_violations_before": before,
                "red_light_violations_after": after,
                "new_violation": violation or after > before,
            }
        )

    def sample(self, phase: str, step: Optional[Dict]) -> Dict:
        if self.baseline_state is None:
            self.baseline_state = read_runtime_state(self.communicator, self.agent.name)
        current = read_runtime_state(self.communicator, self.agent.name)
        delta = state_delta(current, self.baseline_state)
        for key in COLLISION_KEYS:
            self.max_collision_delta[key] = max(self.max_collision_delta[key], delta[key])
        self.max_intensity = max(self.max_intensity, float(current.get("intensity", 0.0)))
        if int(current.get("touched_road", 0)) != 0:
            self.touched_road_samples += 1
        overlap_type = int(current.get("overlap_type", 0))
        if overlap_type:
            self.overlap_samples[str(overlap_type)] = self.overlap_samples.get(str(overlap_type), 0) + 1

        sample = {
            "phase": phase,
            "step": step.get("step") if step else None,
            "action_type": step.get("action_type") if step else None,
            "state": current,
            "collision_delta": delta,
        }
        self.samples.append(sample)
        if total_collision_delta(delta) > 0:
            self.collision_events.append(sample)
        return sample

    def sample_during(
        self,
        phase: str,
        step: Dict,
        duration_s: float,
        capture_func,
    ) -> int:
        end_time = time.time() + max(0.0, duration_s)
        count = 0
        while time.time() < end_time:
            time.sleep(min(self.sample_interval_s, max(0.0, end_time - time.time())))
            capture_func()
            self.sample(phase, step)
            count += 1
        return count

    def report(self) -> Dict:
        final_state = read_runtime_state(self.communicator, self.agent.name)
        final_delta = state_delta(final_state, self.baseline_state or final_state)
        traffic_violations = int(getattr(self.agent, "red_light_violations_count", 0))
        return {
            "runtime_validation_enabled": True,
            "collision_api": "RTCommunicator.get_states / UE rt_get_states",
            "collision_channels": list(COLLISION_KEYS),
            "baseline_state": self.baseline_state,
            "final_state": final_state,
            "final_collision_delta": final_delta,
            "max_collision_delta": self.max_collision_delta,
            "num_collision_event_samples": len(self.collision_events),
            "collision_events": self.collision_events[:200],
            "max_collision_intensity": self.max_intensity,
            "touched_road_samples": self.touched_road_samples,
            "overlap_samples": self.overlap_samples,
            "traffic_api": "RTAgent._check_red_light_violation before MOVE, matching RTAgent.execute_action",
            "red_light_violations_count": traffic_violations,
            "num_traffic_checks": len(self.traffic_checks),
            "traffic_checks": self.traffic_checks,
            "passed": (
                total_collision_delta(final_delta) == 0
                and total_collision_delta(self.max_collision_delta) == 0
                and traffic_violations == 0
                and self.touched_road_samples == 0
            ),
        }


def tick_and_capture(
    communicator: RTCommunicator,
    camera_id: int,
    frame_dir: Path,
    frame_index: int,
    duration_s: float,
    fps: float,
    sleep_after_tick_s: float,
) -> int:
    frame_dt = 1.0 / fps
    remaining = max(0.0, duration_s)
    while remaining > 1e-6:
        tick_dt = min(frame_dt, remaining)
        communicator.unrealcv.set_tick_interval(round(tick_dt, 4))
        communicator.unrealcv.tick()
        if sleep_after_tick_s > 0:
            time.sleep(sleep_after_tick_s)
        capture_frame(communicator, camera_id, frame_dir / f"frame_{frame_index:05d}.png")
        frame_index += 1
        remaining -= tick_dt
    return frame_index


def realtime_capture(
    communicator: RTCommunicator,
    camera_id: int,
    frame_dir: Path,
    frame_index: int,
    duration_s: float,
    fps: float,
) -> int:
    frame_dt = 1.0 / fps
    end_time = time.time() + max(0.0, duration_s)
    while time.time() < end_time:
        time.sleep(min(frame_dt, max(0.0, end_time - time.time())))
        capture_frame(communicator, camera_id, frame_dir / f"frame_{frame_index:05d}.png")
        frame_index += 1
    return frame_index


def make_video(frame_dir: Path, output_video: Path, fps: float) -> None:
    frame_pattern = str(frame_dir / "frame_%05d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    import subprocess

    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", default="results/offline_safe_path_20260625_214632/offline_safe_trajectory.json")
    parser.add_argument("--source-result", default="results/20260624_194515/task_1.json")
    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--sleep-after-tick", type=float, default=0.08)
    parser.add_argument("--startup-retries", type=int, default=60)
    parser.add_argument("--startup-delay", type=float, default=2.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fov", type=float, default=100.0)
    parser.add_argument("--clear-existing-agents", action="store_true")
    parser.add_argument("--agent-only", action="store_true")
    parser.add_argument("--full-world-use-tick", action="store_true")
    parser.add_argument("--no-runtime-validation", action="store_true")
    parser.add_argument("--runtime-sample-interval", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    trajectory_path = repo_root / args.trajectory
    source_result_path = repo_root / args.source_result
    trajectory_data = load_json(trajectory_path)
    source_result = load_json(source_result_path)

    if not trajectory_data.get("success"):
        raise ValueError(f"Trajectory is not marked successful: {trajectory_path}")
    if not trajectory_data.get("trajectory"):
        raise ValueError(f"Trajectory has no steps: {trajectory_path}")

    task_index = args.task_index if args.task_index is not None else int(trajectory_data.get("task_index", source_result.get("task_id", 1)))
    seed = args.seed if args.seed is not None else int(trajectory_data.get("seed", source_result.get("seed", 0)))
    difficulty = args.difficulty or trajectory_data.get("difficulty") or source_result.get("difficulty", "easy")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "results" / f"safe_trajectory_ue_video_{timestamp}"
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    frame_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    unrealcv = connect_unrealcv(args.port, args.ip, args.startup_retries, args.startup_delay)
    communicator = RTCommunicator(unrealcv)
    if not args.agent_only and not args.full_world_use_tick:
        install_async_unrealcv_commands(communicator)
    frame_index = 0
    captured_initial = False
    world_manager: Optional[WorldManager] = None
    validator: Optional[RuntimeValidator] = None
    runtime_validation_report: Optional[Dict] = None

    try:
        if args.clear_existing_agents:
            communicator.clear_agents()
        if args.agent_only:
            agent = setup_agent_only(communicator, trajectory_data)
        else:
            world_manager = setup_world(
                communicator,
                trajectory_data,
                source_result,
                args.task_file,
                task_index,
                seed,
                difficulty,
                args.full_world_use_tick,
            )
            agent = world_manager.agent
        camera_id = agent.camera_id
        communicator.unrealcv.set_camera_resolution(camera_id, (args.camera_width, args.camera_height))
        communicator.unrealcv.set_camera_fov(camera_id, args.camera_fov)

        time.sleep(1.0)
        captured_initial = capture_frame(communicator, camera_id, frame_dir / f"frame_{frame_index:05d}.png")
        frame_index += 1
        if not args.no_runtime_validation:
            validator = RuntimeValidator(
                communicator,
                agent,
                world_manager,
                args.runtime_sample_interval if args.runtime_sample_interval is not None else 1.0 / args.fps,
            )
            validator.initialize()

        current_yaw = math.degrees(math.atan2(agent.direction.y, agent.direction.x))
        for step in trajectory_data["trajectory"]:
            start = vector_from_dict(step["start"])
            end = vector_from_dict(step["end"])
            duration = float(step["duration_s"])
            agent.position = start
            if step["action_type"] == "move" and start.distance(end) > 1e-6:
                planned_direction = Vector(end.x - start.x, end.y - start.y).normalize()
                if planned_direction.length() > 0:
                    planned_yaw = math.degrees(math.atan2(planned_direction.y, planned_direction.x))
                    try:
                        agent.direction = planned_yaw
                    except TypeError:
                        agent.direction = planned_direction
                    if hasattr(agent, "_direction"):
                        agent._direction = planned_direction
                if validator is not None:
                    validator.check_traffic_before_move(step, "before_move")
                    validator.sample("before_move", step)
                current_yaw = yaw_degrees(start, end, current_yaw)
                communicator.unrealcv.set_orientation((0, current_yaw, 0), agent.name)
                communicator.rt_agent_move_to(agent.name, end, duration)
                if args.agent_only or args.full_world_use_tick:
                    frame_index = tick_and_capture(
                        communicator,
                        camera_id,
                        frame_dir,
                        frame_index,
                        duration,
                        args.fps,
                        args.sleep_after_tick,
                    )
                    if validator is not None:
                        validator.sample("after_tick_move", step)
                else:
                    if validator is not None:
                        def capture_next_frame() -> None:
                            nonlocal frame_index
                            capture_frame(communicator, camera_id, frame_dir / f"frame_{frame_index:05d}.png")
                            frame_index += 1

                        validator.sample_during("during_move", step, duration, capture_next_frame)
                    else:
                        frame_index = realtime_capture(
                            communicator,
                            camera_id,
                            frame_dir,
                            frame_index,
                            duration,
                            args.fps,
                        )
                communicator.unrealcv.set_location((end.x, end.y, 110), agent.name)
                agent.position = end
                if validator is not None:
                    validator.sample("after_move_set_location", step)
            else:
                if args.agent_only or args.full_world_use_tick:
                    frame_index = tick_and_capture(
                        communicator,
                        camera_id,
                        frame_dir,
                        frame_index,
                        duration,
                        args.fps,
                        args.sleep_after_tick,
                    )
                    if validator is not None:
                        validator.sample("after_tick_wait", step)
                else:
                    if validator is not None:
                        def capture_next_frame() -> None:
                            nonlocal frame_index
                            capture_frame(communicator, camera_id, frame_dir / f"frame_{frame_index:05d}.png")
                            frame_index += 1

                        validator.sample_during("during_wait", step, duration, capture_next_frame)
                    else:
                        frame_index = realtime_capture(
                            communicator,
                            camera_id,
                            frame_dir,
                            frame_index,
                            duration,
                            args.fps,
                        )
                agent.position = end
                if validator is not None:
                    validator.sample("after_wait", step)

        video_path = output_dir / "safe_trajectory_replay.mp4"
        make_video(frame_dir, video_path, args.fps)
        if validator is not None:
            runtime_validation_report = validator.report()
            with (output_dir / "runtime_validation_report.json").open("w", encoding="utf-8") as f:
                json.dump(runtime_validation_report, f, indent=2)

        manifest = {
            "created_at": datetime.now().isoformat(),
            "source_result": str(source_result_path.relative_to(repo_root)),
            "trajectory": str(trajectory_path.relative_to(repo_root)),
            "task_file": args.task_file,
            "task_index": task_index,
            "seed": seed,
            "difficulty": difficulty,
            "agent_only": args.agent_only,
            "full_world_use_tick": args.full_world_use_tick,
            "fps": args.fps,
            "camera_id": camera_id,
            "camera_resolution": [args.camera_width, args.camera_height],
            "num_frames": frame_index,
            "captured_initial_frame": captured_initial,
            "video": video_path.name,
            "frames_dir": frame_dir.name,
            "trajectory_summary": trajectory_data.get("summary", {}),
            "runtime_validation": runtime_validation_report,
        }
        if world_manager is not None:
            activation_ratio = world_manager._get_dynamic_activation_ratio()
            manifest["full_environment"] = {
                "num_pedestrians": len(getattr(world_manager, "pedestrians", [])),
                "num_irregular_pedestrians": len(getattr(world_manager, "irregular_pedestrians", [])),
                "num_movable_obstacles_total": len(getattr(world_manager, "movable_obstacle_ids", [])),
                "num_movable_obstacles_activated": int(round(len(getattr(world_manager, "movable_obstacle_ids", [])) * activation_ratio)),
                "num_falling_objects_total": len(getattr(world_manager, "falling_object_ids", [])),
                "num_falling_objects_activated": int(round(len(getattr(world_manager, "falling_object_ids", [])) * activation_ratio)),
                "dynamic_activation_ratio": activation_ratio,
            }
        with (output_dir / "replay_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(json.dumps({"output_dir": str(output_dir), **manifest}, indent=2))
        return 0
    finally:
        try:
            communicator.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
