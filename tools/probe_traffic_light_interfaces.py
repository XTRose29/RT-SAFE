#!/usr/bin/env python3
"""Probe the packaged RT traffic-light Blueprint interfaces through UnrealCV.

This is a diagnostic tool. It spawns only the traffic lights/intersection controllers
for one task map, calls likely read-only Blueprint getters, and prints their raw
responses. It does not start the benchmark agent or an LLM.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.rt_communicator import RTCommunicator
from base.rt_agent import RTAgent
from base.rt_unrealcv import RTUnrealCV
from manager.world_manager import WorldManager
from simworld.utils.vector import Vector


SIGNAL_GETTERS = (
    "GetState",
    "GetLightState",
    "GetCurrentState",
    "GetInformation",
    "IsVehicleGreen",
    "IsPedestrianWalk",
    "GetIsVehicleGreen",
    "GetIsPedestrianWalk",
    "GetLeftTime",
    "GetRemainingTime",
)

INTERSECTION_GETTERS = (
    "GetIntersectionState",
    "GetTrafficDiagnostics",
    "GetState",
    "GetCurrentState",
    "GetInformation",
    "GetPhase",
    "GetRemainingTime",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", default="data/map2_12roads/tasks.json")
    parser.add_argument("--task-number", type=int, default=5)
    parser.add_argument("--agent-config", default="data/agents_qwen3vl8b.json")
    parser.add_argument("--unrealcv-host", default="127.0.0.1")
    parser.add_argument("--unrealcv-port", type=int, default=9001)
    parser.add_argument("--tick-seconds", type=float, default=0.0)
    return parser.parse_args()


def call_blueprint(unrealcv: RTUnrealCV, object_name: str, function: str) -> str:
    try:
        response = unrealcv.client.request(f"vbp {object_name} {function}")
        return str(response)
    except Exception as exc:  # diagnostic output should include every failed getter
        return f"{type(exc).__name__}: {exc}"


def collect_probe(
    unrealcv: RTUnrealCV,
    manager: WorldManager,
) -> dict:
    path_points = manager.scenario_data["route_info"]["shortest_path"]
    route_crosswalks = manager.traffic_controller.get_route_crosswalks(path_points)
    signal_groups = manager.traffic_controller.get_crosswalk_signal_groups(
        route_crosswalks
    )
    relevant_signal_ids = {
        signal.id
        for signals in signal_groups.values()
        for signal in signals
    }
    relevant_signals = [
        signal
        for signal in manager.traffic_controller.traffic_signals
        if signal.id in relevant_signal_ids
    ]

    task_intersection_ids = set()
    for intersection in manager.traffic_controller.intersections:
        if any(
            crosswalk.id in signal_groups
            for crosswalk in intersection.crosswalks
        ):
            task_intersection_ids.add(intersection.id)

    signal_results = {}
    for signal in relevant_signals:
        object_name = manager.communicator.get_traffic_signal_name(signal.id)
        signal_results[object_name] = {
            "signal_id": signal.id,
            "type": signal.type,
            "lane_id": signal.lane_id,
            "crosswalk_id": signal.crosswalk_id,
            "python_state": [str(value) for value in signal.get_state()],
            "position": [signal.position.x, signal.position.y],
            "getters": {
                getter: call_blueprint(unrealcv, object_name, getter)
                for getter in SIGNAL_GETTERS
            },
        }

    intersection_results = {}
    for intersection in manager.traffic_controller.intersections:
        if intersection.id not in task_intersection_ids:
            continue
        object_name = f"RT_Intersection_{intersection.id}"
        intersection_results[object_name] = {
            getter: call_blueprint(unrealcv, object_name, getter)
            for getter in INTERSECTION_GETTERS
        }

    start = Vector(manager.scenario_data["task"]["start_point"])
    end = Vector(manager.scenario_data["task"]["end_point"])
    route = [Vector(point) for point in path_points[1:]]
    initial_direction = (route[0] - start).normalize()
    agent = RTAgent(
        position=start,
        direction=initial_direction,
        destination=end,
        shortest_path=route,
        communicator=manager.communicator,
        llm=None,
        route_crosswalks=route_crosswalks,
        crosswalk_signal_groups=signal_groups,
    )
    agent_snapshot = agent._get_traffic_light_snapshot()

    return {
        "task_intersection_ids": sorted(task_intersection_ids),
        "route_crosswalks": [
            {
                "id": crosswalk.id,
                "start": [crosswalk.start.x, crosswalk.start.y],
                "end": [crosswalk.end.x, crosswalk.end.y],
            }
            for crosswalk in route_crosswalks
        ],
        "agent_snapshot": agent_snapshot,
        "agent_prompt_context": agent._format_traffic_light_context(
            agent_snapshot
        ),
        "signals": signal_results,
        "intersections": intersection_results,
    }


def main() -> None:
    args = parse_args()
    unrealcv = RTUnrealCV(port=args.unrealcv_port, ip=args.unrealcv_host)
    communicator = RTCommunicator(unrealcv)
    manager = WorldManager(
        communicator,
        agent_path=args.agent_config,
        task_file_path=args.task_file,
        seed=1,
        use_tick=True,
    )
    try:
        task_index = args.task_number - 1
        manager.difficulty = "default"
        manager.scenario_data = manager.all_scenarios[task_index]
        manager._initialize_world()
        manager._spawn_and_setup_traffic_signals()

        print(json.dumps({"before": collect_probe(unrealcv, manager)}, indent=2))

        if args.tick_seconds > 0:
            unrealcv.set_tick_interval(1.0)
            for _ in range(max(1, round(args.tick_seconds))):
                unrealcv.tick()
            print(json.dumps({"after": collect_probe(unrealcv, manager)}, indent=2))
    finally:
        manager.cleanup()
        unrealcv.disconnect()


if __name__ == "__main__":
    main()
