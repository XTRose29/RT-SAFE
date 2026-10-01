#!/usr/bin/env python3
"""Spawn an experiment world in UE and probe rendering/ticks without LLM calls."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# The realtime environment does not use SimWorld's optional Qt city-layout
# visualizer. Avoid importing it through the package-level __init__ during a
# headless runtime preflight, matching run_openai_benchmark.py.
_simworld = types.ModuleType("simworld")
_simworld.__path__ = [str(REPO_ROOT / "SimWorld" / "simworld")]
sys.modules.setdefault("simworld", _simworld)

from base.rt_agent import RTAgent  # noqa: E402
from base.rt_communicator import RTCommunicator  # noqa: E402
from base.rt_unrealcv import RTUnrealCV  # noqa: E402
from manager.world_manager import SLOMO, TIME_ALPHA, TIME_BETA, WorldManager  # noqa: E402
from simworld.utils.vector import Vector  # noqa: E402
from utils.route_steps import route_length_to_max_steps  # noqa: E402


def _save_image(image, path: Path) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
        cv2.imwrite(str(path), arr[:, :, ::-1])
        return arr
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        cv2.imwrite(str(path), arr)
        return arr[:, :, ::-1]
    cv2.imwrite(str(path), arr)
    return arr


def _stats(arr: np.ndarray) -> str:
    return (
        f"shape={tuple(arr.shape)} min={int(arr.min())} max={int(arr.max())} "
        f"mean={float(arr.mean()):.2f} std={float(arr.std()):.2f}"
    )


def _apply_render_quality(
    unrealcv: RTUnrealCV,
    preset: str,
    exposure_bias: float | None,
) -> dict[str, object]:
    """Apply and record an explicit UE scalability profile for visual QA."""
    if preset == "inherited" and exposure_bias is None:
        return {"preset": preset, "commands": {}}

    commands: list[str] = []
    if preset != "inherited":
        quality = "3" if preset == "epic" else "2"
        commands.extend(
            (
                f"vrun sg.ViewDistanceQuality {quality}",
                f"vrun sg.AntiAliasingQuality {quality}",
                f"vrun sg.ShadowQuality {quality}",
                f"vrun sg.GlobalIlluminationQuality {quality}",
                f"vrun sg.ReflectionQuality {quality}",
                f"vrun sg.PostProcessQuality {quality}",
                f"vrun sg.TextureQuality {quality}",
                f"vrun sg.EffectsQuality {quality}",
                f"vrun sg.FoliageQuality {quality}",
                f"vrun sg.ShadingQuality {quality}",
                "vrun r.ScreenPercentage 100",
                "vrun r.PostProcessAAQuality 6",
            )
        )
    if exposure_bias is not None:
        commands.extend(
            (
                "vrun r.DefaultFeature.AutoExposure 1",
                f"vrun r.DefaultFeature.AutoExposure.Bias {exposure_bias:g}",
            )
        )
    responses = {
        command: unrealcv.client.request(command)
        for command in commands
    }
    return {"preset": preset, "commands": responses}


def _build_spawned_world(condition: dict, communicator: RTCommunicator) -> WorldManager:
    world_manager = WorldManager(
        communicator=communicator,
        agent_path=condition["agent_config_path"],
        task_file_path=condition["task_file"],
        seed=condition["seed"],
        control_mode="llm",
        token_based=condition["token_based"],
        use_tick=condition["use_tick"],
        realtime_thinking=condition["realtime_thinking"],
        record_per_step=False,
        use_action_frames=condition["use_action_frames"],
        prompt_style=condition["prompt_style"],
        max_steps=condition["max_steps"],
        results_dir="/tmp/simworld_ue_world_smoke",
    )
    world_manager._batch_suffix = f"{condition['condition_id']}__ue_smoke"

    difficulty = world_manager._normalize_difficulty_label(condition["difficulty"])
    world_manager.difficulty = difficulty
    world_manager.difficulty_config = world_manager._get_difficulty_config()
    random.seed(condition["seed"] + condition["task_index"])

    world_manager.scenario_data = world_manager.all_scenarios[condition["task_index"]]
    world_manager.current_task_id = world_manager.scenario_data.get("task_id", condition["task_index"])
    world_manager._initialize_world()

    task_data = world_manager.scenario_data["task"]
    start_point = Vector(task_data["start_point"][0], task_data["start_point"][1])
    end_point = Vector(task_data["end_point"][0], task_data["end_point"][1])
    shortest_path = [Vector(x, y) for x, y in world_manager.scenario_data["route_info"]["shortest_path"][1:]]
    first_waypoint = shortest_path[0]
    initial_direction = Vector(first_waypoint.x - start_point.x, first_waypoint.y - start_point.y).normalize()
    required_time = task_data.get("required_time", 1000)

    path_length = start_point.distance(shortest_path[0])
    for idx in range(len(shortest_path) - 1):
        path_length += shortest_path[idx].distance(shortest_path[idx + 1])
    world_manager.max_steps = (
        condition["max_steps"]
        if condition["max_steps"] >= 0
        else route_length_to_max_steps(path_length)
    )

    traffic_signals = world_manager.traffic_controller.traffic_signals
    traffic_intersections = world_manager.traffic_controller.intersections
    traffic_signal_config = {
        "green_light_duration": world_manager.traffic_controller.config["traffic.traffic_signal.green_light_duration"],
        "yellow_light_duration": world_manager.traffic_controller.config["traffic.traffic_signal.yellow_light_duration"],
        "pedestrian_green_light_duration": world_manager.traffic_controller.config[
            "traffic.traffic_signal.pedestrian_green_light_duration"
        ],
        "pedestrian_phase_duration": world_manager.traffic_controller.config["traffic.traffic_signal.pedestrian_phase_duration"],
    }

    world_manager.agent = RTAgent(
        start_point,
        initial_direction,
        end_point,
        shortest_path,
        required_time,
        communicator,
        world_manager.llm,
        token_based=world_manager.token_based,
        use_tick=world_manager.use_tick,
        time_alpha=TIME_ALPHA,
        time_beta=TIME_BETA,
        realtime_thinking=world_manager.realtime_thinking,
        task_edges=task_data.get("edges", []),
        traffic_signals=traffic_signals,
        traffic_intersections=traffic_intersections,
        traffic_signal_config=traffic_signal_config,
        static_obstacles=world_manager.static_obstacles,
        record_per_step=False,
        record_dir=None,
        slomo=SLOMO,
        use_action_frames=world_manager.use_action_frames,
        prompt_style=world_manager.prompt_style,
    )
    world_manager.spawn_agents()
    return world_manager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--ue-ip", default="127.0.0.1")
    parser.add_argument("--ue-port", type=int, default=9006)
    parser.add_argument("--ticks", type=int, default=20)
    parser.add_argument("--location-rounds", type=int, default=10)
    parser.add_argument("--location-limit", type=int, default=96)
    parser.add_argument("--out", default="/tmp/ue_world_smoke_observation.png")
    parser.add_argument(
        "--render-quality",
        choices=("inherited", "high", "epic"),
        default="inherited",
        help="Optional explicit UE scalability preset applied before capture.",
    )
    parser.add_argument(
        "--exposure-bias",
        type=float,
        help="Optional UE auto-exposure bias applied before capture.",
    )
    parser.add_argument("--no-cleanup", action="store_true", help="Disconnect without destroying spawned UE actors.")
    args = parser.parse_args()

    with open(args.condition, "r", encoding="utf-8") as f:
        condition = json.load(f)

    print(f"[world-smoke] condition={args.condition}", flush=True)
    communicator = RTCommunicator(RTUnrealCV(port=args.ue_port, ip=args.ue_ip))
    world_manager = None
    try:
        world_manager = _build_spawned_world(condition, communicator)
        agent = world_manager.agent
        print(
            "[world-smoke] spawned "
            f"pedestrians={len(world_manager.pedestrians)} irregular={len(world_manager.irregular_pedestrians)} "
            f"movable={len(getattr(world_manager, 'movable_obstacle_ids', []))} "
            f"activated_movable={len(getattr(world_manager, 'activated_movable_obstacle_ids', []))} "
            f"traffic_signals={len(world_manager.traffic_controller.traffic_signals)}",
            flush=True,
        )

        quality = _apply_render_quality(
            communicator.unrealcv,
            args.render_quality,
            args.exposure_bias,
        )
        print(f"[world-smoke] render quality: {quality}", flush=True)
        communicator.unrealcv.set_camera_resolution(
            agent.camera_id, agent.camera_resolution
        )
        communicator.unrealcv.set_camera_fov(agent.camera_id, agent.fov)
        agent.sync_ue()
        observation = agent.get_observation()
        arr = _save_image(observation["ego_view"], Path(args.out))
        print(f"[world-smoke] observation: {_stats(arr)} saved={args.out}", flush=True)
        if arr.std() < 1.0:
            raise RuntimeError("agent observation appears blank")

        objects = [str(name) for name in communicator.unrealcv.get_objects()]
        names = [name for name in objects if name.startswith(("RT_", "ProGen", "SM_", "BP_", "Road", "Sidewalk"))]
        names = names[: args.location_limit]
        if not names:
            names = objects[: args.location_limit]
        for idx in range(args.location_rounds):
            locs = communicator.unrealcv.get_location_batch(names)
            print(
                f"[world-smoke] location batch {idx + 1}/{args.location_rounds}: "
                f"count={len(locs)} first={locs[0].round(2).tolist()}",
                flush=True,
            )
            time.sleep(0.05)

        for tick_idx in range(args.ticks):
            communicator.unrealcv.tick()
            if (tick_idx + 1) % 5 == 0 or tick_idx == args.ticks - 1:
                print(f"[world-smoke] ticked {tick_idx + 1}/{args.ticks}", flush=True)
            time.sleep(0.02)

        agent.sync_ue()
        observation_after = agent.get_observation()
        output_path = Path(args.out)
        arr_after = _save_image(
            observation_after["ego_view"],
            output_path.with_name(f"{output_path.stem}_after{output_path.suffix}"),
        )
        print(f"[world-smoke] observation after ticks: {_stats(arr_after)}", flush=True)
        print("[world-smoke] ok", flush=True)
        return 0
    finally:
        if args.no_cleanup:
            pass
        elif world_manager is not None:
            world_manager.cleanup()
        else:
            try:
                communicator.clear_agents()
            except Exception:
                pass
        communicator.unrealcv.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
