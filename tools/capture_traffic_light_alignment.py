#!/usr/bin/env python3
"""Capture a real Task 5 crossing model input with synchronized UE light state.

This diagnostic does not call the VLM. It places the benchmark agent at the
recorded Task 5 crossing pose, starts the real UE traffic-light controllers,
then uses RTAgent.get_observation() so the Blueprint GetState read and camera
capture follow the exact post-change benchmark path.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.rt_agent import RTAgent
from base.rt_communicator import RTCommunicator
from base.rt_unrealcv import RTUnrealCV
from llm.prompt import USER_PROMPT, get_image_description, get_system_prompt
from manager.world_manager import WorldManager
from simworld.utils.vector import Vector


RECORDED_STEP_19_POSITION = Vector(-9487.641, 85.624)
RECORDED_STEP_19_DIRECTION = Vector(-0.1908, 0.9816).normalize()
TASK5_CROSSWALK_EXIT = Vector(-9300.0, 700.0)
TASK5_NEXT_ROUTE_POINT = Vector(-8193.4948, 700.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", default="data/map2_12roads/tasks.json")
    parser.add_argument("--task-number", type=int, default=5)
    parser.add_argument("--agent-config", default="data/agents_qwen3vl8b.json")
    parser.add_argument("--unrealcv-host", default="127.0.0.1")
    parser.add_argument("--unrealcv-port", type=int, default=9001)
    parser.add_argument(
        "--phase",
        choices=(
            "VEHICLE_GREEN",
            "VEHICLE_YELLOW",
            "ALL_RED",
            "PEDESTRIAN_WALK",
            "PEDESTRIAN_CLEARANCE",
        ),
        default=None,
        help="Force and freeze every native intersection in this phase.",
    )
    parser.add_argument("--active-vehicle-group", type=int, default=0)
    parser.add_argument(
        "--advance-seconds",
        type=float,
        default=0.0,
        help=(
            "Before the final capture, command a short agent move and advance "
            "the paused UE world for this many seconds."
        ),
    )
    parser.add_argument("--advance-distance-cm", type=float, default=200.0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to results/traffic_light_alignment_capture_<timestamp>",
    )
    return parser.parse_args()


def build_user_prompt(agent: RTAgent, observation: dict) -> str:
    relative_distance = agent.position.distance(agent.current_destination)
    current_yaw_rad = math.radians(agent.yaw)
    dx = agent.current_destination.x - agent.position.x
    dy = agent.current_destination.y - agent.position.y
    target_yaw_rad = math.atan2(dy, dx)
    relative_angle = math.degrees(target_yaw_rad - current_yaw_rad)
    if relative_angle > 180:
        relative_angle -= 360
    elif relative_angle < -180:
        relative_angle += 360
    relative_angle = -relative_angle

    return USER_PROMPT.format(
        step_num=19,
        current_position=agent.position,
        speed=agent.speed,
        direction=agent.direction,
        subgoal=agent.current_destination,
        relative_distance=relative_distance,
        relative_angle=relative_angle,
        required_time=agent.required_time,
        time_spent=0.0,
        traffic_light_context=agent._format_traffic_light_context(
            observation["traffic_light_snapshot"]
        ),
        history="",
        image_description=get_image_description(False, is_first_step=False),
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(
        args.output_dir
        or (
            "results/traffic_light_alignment_capture_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    step_dir = output_dir / "step_0019"

    unrealcv = RTUnrealCV(port=args.unrealcv_port, ip=args.unrealcv_host)
    communicator = RTCommunicator(unrealcv)
    manager = WorldManager(
        communicator,
        agent_path=args.agent_config,
        task_file_path=args.task_file,
        seed=1,
        use_tick=True,
        realtime_thinking=True,
        record_per_step=True,
    )

    try:
        manager.difficulty = "default"
        manager.scenario_data = manager.all_scenarios[args.task_number - 1]
        manager._initialize_world()
        manager._spawn_and_setup_traffic_signals()
        if args.phase:
            for intersection in manager.traffic_controller.intersections:
                unrealcv.traffic_signal_set_intersection_phase(
                    f"RT_Intersection_{intersection.id}",
                    args.phase,
                    args.active_vehicle_group,
                )
        time.sleep(3)

        path_points = manager.scenario_data["route_info"]["shortest_path"]
        route_crosswalks = manager.traffic_controller.get_route_crosswalks(
            path_points
        )
        signal_groups = (
            manager.traffic_controller.get_crosswalk_signal_groups(
                route_crosswalks
            )
        )
        intersection_names = (
            manager.traffic_controller.get_crosswalk_intersection_names(
                route_crosswalks
            )
        )

        agent = RTAgent(
            position=RECORDED_STEP_19_POSITION,
            direction=RECORDED_STEP_19_DIRECTION,
            destination=TASK5_NEXT_ROUTE_POINT,
            shortest_path=[
                TASK5_CROSSWALK_EXIT,
                TASK5_NEXT_ROUTE_POINT,
            ],
            required_time=1000,
            communicator=communicator,
            llm=None,
            token_based=False,
            use_tick=True,
            realtime_thinking=True,
            task_edges=manager.scenario_data["task"].get("edges", []),
            traffic_signals=manager.traffic_controller.traffic_signals,
            route_crosswalks=route_crosswalks,
            crosswalk_signal_groups=signal_groups,
            crosswalk_intersection_names=intersection_names,
            static_obstacles=manager.static_obstacles,
            record_per_step=True,
            record_dir=str(output_dir),
            terminate_on_touched_road=False,
        )
        agent.step_num = 19
        manager.agent = agent

        communicator.spawn_agent(
            agent,
            agent.name,
            "/Game/RealTimeBench/Agent/BP_RT_Agent.BP_RT_Agent_C",
            type="agent",
        )
        communicator.rt_agent_adjust_speed(agent.name, agent.speed)
        unrealcv.set_camera_resolution(agent.camera_id, agent.camera_resolution)
        unrealcv.set_camera_fov(agent.camera_id, agent.fov)
        time.sleep(3)

        agent.sync_ue()
        time_advance_smoke = None
        if args.advance_seconds > 0:
            before_position = [agent.position.x, agent.position.y]
            before_observation = agent.get_observation()
            before_snapshot = before_observation["traffic_light_snapshot"]
            target = agent.position + agent.direction * args.advance_distance_cm
            communicator.rt_agent_move_to(
                agent.name,
                target,
                args.advance_seconds,
            )
            unrealcv.advance_simulation_time(args.advance_seconds)
            agent.sync_ue()
            observation = agent.get_observation()
            after_snapshot = observation["traffic_light_snapshot"]
            after_position = [agent.position.x, agent.position.y]
            moved_cm = math.dist(before_position, after_position)
            before_remaining = before_snapshot.get("phase_remaining_time_s")
            after_remaining = after_snapshot.get("phase_remaining_time_s")
            traffic_advanced = (
                before_snapshot.get("observed_phase")
                != after_snapshot.get("observed_phase")
                or (
                    before_remaining is not None
                    and after_remaining is not None
                    and abs(float(before_remaining) - float(after_remaining))
                    > 0.25
                )
            )
            time_advance_smoke = {
                "advance_seconds": args.advance_seconds,
                "target": [target.x, target.y],
                "before_position": before_position,
                "after_position": after_position,
                "moved_cm": moved_cm,
                "before_phase": before_snapshot.get("observed_phase"),
                "after_phase": after_snapshot.get("observed_phase"),
                "before_remaining_time_s": before_remaining,
                "after_remaining_time_s": after_remaining,
                "traffic_advanced": traffic_advanced,
                "passed": moved_cm > 10.0 and traffic_advanced,
            }
        else:
            observation = agent.get_observation()
        snapshot = observation["traffic_light_snapshot"]
        user_prompt = build_user_prompt(agent, observation)
        system_prompt = get_system_prompt(realtime_thinking=True)

        observation["ego_view"].save(output_dir / "actual_model_input.png")
        with (output_dir / "traffic_light_snapshot.json").open(
            "w", encoding="utf-8"
        ) as snapshot_file:
            json.dump(snapshot, snapshot_file, indent=2, ensure_ascii=False)
        (output_dir / "user_prompt.txt").write_text(
            user_prompt, encoding="utf-8"
        )

        prompt_data = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "full_response": (
                "Not executed: this capture validates the real visual and "
                "prompt inputs before a VLM call."
            ),
            "traffic_light_snapshot": snapshot,
            "execution_traffic_light_snapshot": None,
            "input_images": [observation["ego_view"]],
            "char_count": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "response_time": None,
        }
        agent._record_step_data(
            19,
            observation,
            prompt_data,
            action=None,
            input_images=[observation["ego_view"]],
            feedback="Input alignment capture only; no action executed.",
            collision_details={},
        )

        summary = {
            "output_dir": str(output_dir.resolve()),
            "capture_path": "RTAgent.get_observation",
            "blueprint_state_source": snapshot.get("source"),
            "position": [agent.position.x, agent.position.y],
            "direction": [agent.direction.x, agent.direction.y],
            "subgoal": [
                agent.current_destination.x,
                agent.current_destination.y,
            ],
            "route_crosswalk_ids": [
                crosswalk.id for crosswalk in route_crosswalks
            ],
            "forced_phase": args.phase,
            "active_vehicle_group": args.active_vehicle_group,
            "traffic_light_snapshot": snapshot,
            "prompt_contains_snapshot": (
                agent._format_traffic_light_context(snapshot) in user_prompt
            ),
            "time_advance_smoke": time_advance_smoke,
        }
        with (output_dir / "capture_summary.json").open(
            "w", encoding="utf-8"
        ) as summary_file:
            json.dump(summary, summary_file, indent=2, ensure_ascii=False)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if time_advance_smoke and not time_advance_smoke["passed"]:
            raise RuntimeError(
                "Simulation clock failed to advance agent movement and native "
                "traffic state together"
            )
    finally:
        manager.cleanup()
        unrealcv.disconnect()


if __name__ == "__main__":
    main()
