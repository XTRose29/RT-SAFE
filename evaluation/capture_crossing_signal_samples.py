#!/usr/bin/env python3
"""Capture direct-facing policy views of every pedestrian-signal state."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SimWorld"))

from base.rt_communicator import RTCommunicator  # noqa: E402
from base.rt_unrealcv import RTUnrealCV  # noqa: E402
from evaluation.ue_world_smoke_test import _build_spawned_world  # noqa: E402
from simworld.utils.vector import Vector  # noqa: E402


TARGET_STATES = ("WALK", "FLASHING_DONT_WALK", "DONT_WALK")


def _normalized_pedestrian_state(value) -> str:
    return str(value or "UNKNOWN").upper().replace("'", "").replace(" ", "_")


def _route_crosswalk_pose(world_manager) -> tuple[Vector, Vector]:
    task = world_manager.scenario_data["task"]
    current = Vector(*task["start_point"])
    for edge in task.get("edges", []):
        first = Vector(*edge["node1"])
        second = Vector(*edge["node2"])
        near, far = (
            (first, second)
            if current.distance(first) <= current.distance(second)
            else (second, first)
        )
        if edge.get("type") == "crosswalk":
            return near, far
        current = far
    raise ValueError("selected task has no ordered route crosswalk")


def _as_rgb_array(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"unexpected camera image shape: {array.shape}")
    # RTCommunicator camera observations are OpenCV/BGR arrays.
    return array[:, :, ::-1]


def _write_video(images: list[Path], output: Path, fps: float = 0.75) -> None:
    if not images:
        return
    first = cv2.imread(str(images[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output}")
    try:
        for path in images:
            frame = cv2.imread(str(path))
            for _ in range(3):
                writer.write(frame)
    finally:
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", default="data/map3_15roads/tasks.json")
    parser.add_argument("--task-index", type=int, default=5)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ue-ip", default="127.0.0.1")
    parser.add_argument("--ue-port", type=int, default=9020)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fov", type=float, default=100.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=-25.0)
    parser.add_argument("--approach-offset-cm", type=float, default=100.0)
    parser.add_argument("--sample-step-seconds", type=float, default=1.0)
    parser.add_argument("--max-sim-seconds", type=float, default=140.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    condition = {
        "condition_id": "crossing_signal_visibility",
        "agent_config_path": str(Path(args.agent_config).expanduser().resolve()),
        "task_file": args.task_file,
        "task_index": args.task_index,
        "seed": args.seed,
        "difficulty": "easy",
        "token_based": False,
        "use_tick": True,
        "realtime_thinking": True,
        "record_per_step": False,
        "use_action_frames": False,
        "prompt_style": "instructional",
        "max_steps": 1,
    }
    communicator = RTCommunicator(
        RTUnrealCV(
            port=args.ue_port,
            ip=args.ue_ip,
            resolution=(args.width, args.height),
        )
    )
    world_manager = None
    captured: dict[str, dict] = {}
    try:
        world_manager = _build_spawned_world(condition, communicator)
        agent = world_manager.agent
        # The smoke-test builder intentionally constructs a minimal agent.
        # Restore the production rollout's route/signal metadata before using
        # it for an input-alignment diagnostic.
        route_points = world_manager.scenario_data["route_info"]["shortest_path"]
        agent.route_crosswalks = world_manager.traffic_controller.get_route_crosswalks(
            route_points
        )
        agent.crosswalk_signal_groups = (
            world_manager.traffic_controller.get_crosswalk_signal_groups(
                agent.route_crosswalks
            )
        )
        agent.crosswalk_intersection_names = (
            world_manager.traffic_controller.get_crosswalk_intersection_names(
                agent.route_crosswalks
            )
            if world_manager.use_intersection_phase_api
            else {}
        )
        agent.traffic_phase_timing = world_manager.traffic_phase_timing
        communicator.unrealcv.set_camera_resolution(
            agent.camera_id,
            (args.width, args.height),
        )
        # Keep the agent-side VLM input contract identical to the UE camera
        # configuration.  get_observation() deliberately rejects any mismatch.
        agent.camera_resolution = (args.width, args.height)
        agent.fov = args.fov
        agent.first_person_camera_pitch_deg = args.camera_pitch_deg
        communicator.unrealcv.set_camera_fov(agent.camera_id, args.fov)

        task_near_curb, task_far_curb = _route_crosswalk_pose(world_manager)
        route_crosswalks = list(agent.route_crosswalks or [])
        if not route_crosswalks:
            raise RuntimeError("selected task did not resolve to a route crosswalk")
        route_crosswalk = min(
            route_crosswalks,
            key=lambda crosswalk: min(
                task_near_curb.distance(crosswalk.start)
                + task_far_curb.distance(crosswalk.end),
                task_near_curb.distance(crosswalk.end)
                + task_far_curb.distance(crosswalk.start),
            ),
        )
        if task_near_curb.distance(route_crosswalk.start) <= task_near_curb.distance(
            route_crosswalk.end
        ):
            near_curb, far_curb = route_crosswalk.start, route_crosswalk.end
        else:
            near_curb, far_curb = route_crosswalk.end, route_crosswalk.start
        # A rollout only queries a crossing once that endpoint is the active
        # route subgoal. Mirror that state after teleporting to the diagnostic
        # curb pose so the snapshot selects the same route-relevant signal.
        agent.current_destination = near_curb
        agent.sync_settle_seconds = min(agent.sync_settle_seconds, 0.1)
        crossing_direction = (far_curb - near_curb).normalize()
        sample_position = near_curb - crossing_direction * args.approach_offset_cm
        yaw = math.degrees(
            math.atan2(crossing_direction.y, crossing_direction.x)
        )
        z = float(communicator.unrealcv.get_location(agent.name)[2])

        elapsed = 0.0
        last_reported_state = None
        while elapsed <= args.max_sim_seconds and len(captured) < len(TARGET_STATES):
            # Reassert a controlled curb pose before every capture. This is a
            # rendering diagnostic; background actors and signal timing still
            # advance exactly as they do in a rollout.
            communicator.unrealcv.set_location(
                (sample_position.x, sample_position.y, z),
                agent.name,
            )
            communicator.unrealcv.set_orientation((0, yaw, 0), agent.name)
            agent.sync_ue()
            observation = agent.get_observation()
            snapshot = dict(observation.get("traffic_light_snapshot") or {})
            state = _normalized_pedestrian_state(snapshot.get("pedestrian_state"))
            if state != last_reported_state:
                print(
                    "[signal-sample] "
                    f"t={elapsed:.1f}s state={state} "
                    f"relevant={snapshot.get('relevant')} "
                    f"crosswalk={snapshot.get('crosswalk_id')} "
                    f"distance_cm={snapshot.get('distance_cm')}",
                    flush=True,
                )
                last_reported_state = state
            if state in TARGET_STATES and state not in captured:
                input_path = output_dir / f"pedestrian_signal_{state.lower()}_policy_input.png"
                observation["ego_view"].save(input_path)

                raw = communicator.get_camera_observation(agent.camera_id, "lit")
                raw_path = output_dir / f"pedestrian_signal_{state.lower()}_raw.png"
                Image.fromarray(_as_rgb_array(raw)).save(raw_path)

                labeled = observation["ego_view"].copy().convert("RGB")
                draw = ImageDraw.Draw(labeled)
                draw.rectangle((0, 0, labeled.width, 42), fill=(8, 10, 14))
                draw.text(
                    (14, 12),
                    f"DIRECT OPPOSITE-CURB VIEW | PEDESTRIAN SIGNAL: {state}",
                    fill=(255, 255, 255),
                )
                labeled_path = output_dir / f"pedestrian_signal_{state.lower()}_labeled.png"
                labeled.save(labeled_path)
                captured[state] = {
                    "sim_time_seconds": round(elapsed, 3),
                    "policy_input": str(input_path),
                    "raw_frame": str(raw_path),
                    "labeled_frame": str(labeled_path),
                    "snapshot": snapshot,
                }
                print(f"[signal-sample] captured {state}: {labeled_path}", flush=True)

            if len(captured) < len(TARGET_STATES):
                agent._advance_simulation_time(args.sample_step_seconds)
                elapsed += args.sample_step_seconds

        ordered_images = [
            Path(captured[state]["labeled_frame"])
            for state in TARGET_STATES
            if state in captured
        ]
        video_path = output_dir / "pedestrian_signal_direct_facing_samples.mp4"
        _write_video(ordered_images, video_path)
        manifest = {
            "task_file": args.task_file,
            "task_index": args.task_index,
            "seed": args.seed,
            "camera_resolution": [args.width, args.height],
            "camera_fov_degrees": args.fov,
            "camera_pitch_degrees": args.camera_pitch_deg,
            "near_curb": [near_curb.x, near_curb.y],
            "far_curb": [far_curb.x, far_curb.y],
            "sample_position": [sample_position.x, sample_position.y],
            "direct_facing_yaw_degrees": yaw,
            "captured_states": captured,
            "missing_states": [state for state in TARGET_STATES if state not in captured],
            "video": str(video_path),
        }
        manifest_path = output_dir / "pedestrian_signal_sample_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2), flush=True)
        return 0 if not manifest["missing_states"] else 2
    finally:
        if world_manager is not None:
            world_manager.cleanup()
        communicator.unrealcv.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
