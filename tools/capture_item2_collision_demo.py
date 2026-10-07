#!/usr/bin/env python3
"""Record matched Task 5 demos for safe and violating crossings."""

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

import cv2
import numpy as np

from base.rt_communicator import RTCommunicator
from base.rt_unrealcv import RTUnrealCV
from manager.world_manager import WorldManager


CONTACT_ENVELOPE_CM = 300.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("safe", "collision"), required=True)
    parser.add_argument("--task-file", default="data/map2_12roads/tasks.json")
    parser.add_argument("--task-number", type=int, default=5)
    parser.add_argument("--agent-config", default="data/agents_qwen3vl8b.json")
    parser.add_argument("--unrealcv-host", default="127.0.0.1")
    parser.add_argument("--unrealcv-port", type=int, default=9001)
    parser.add_argument("--crosswalk-id", type=int, default=2)
    parser.add_argument(
        "--demo-label",
        default=None,
        help="Filename/metadata label (for example map1_task2_task_id1).",
    )
    parser.add_argument(
        "--agent-blueprint",
        default="/Game/RealTimeBench/Agent/BP_RT_Agent.BP_RT_Agent_C",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--camera-mode",
        choices=("first-person", "overhead"),
        default="first-person",
    )
    parser.add_argument("--exposure-offset", type=float, default=2.0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def vec2(raw):
    values = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    return np.array([float(values[0]), float(values[1])], dtype=np.float64)


def get_crosswalk_signal(intersection, crosswalk_id):
    return next(
        signal
        for signal in intersection.traffic_lights
        if signal.crosswalk_id == crosswalk_id
    )


def set_pedestrian_signal(communicator, intersection, crosswalk_id, walk):
    """Set every rendered pedestrian face and verify the routed crosswalk head.

    RT12 uses combined vehicle/pedestrian heads for route crosswalks.  Updating
    only ``intersection.pedestrian_lights`` leaves the visible hand on the
    combined head unchanged, so include both signal collections here.
    """
    primary = get_crosswalk_signal(intersection, crosswalk_id)
    signals = {
        signal.id: signal
        for signal in (
            list(intersection.traffic_lights)
            + list(intersection.pedestrian_lights)
        )
    }
    phase = "PEDESTRIAN_WALK" if walk else "VEHICLE_GREEN"
    responses = []
    state = None
    # The packaged RT12 city predates the native intersection-controller
    # actor.  Drive all rendered heads as one atomic group, then verify the
    # route head directly from UE.  VEHICLE_GREEN is also the canonical
    # DON'T-WALK phase and releases the consequence vehicle in demo B.
    for _ in range(5):
        for signal in signals.values():
            name = communicator.get_traffic_signal_name(signal.id)
            if walk:
                responses.extend(
                    communicator.unrealcv.tl_set_vehicle_red(name) or []
                )
                responses.append(
                    communicator.unrealcv.tl_set_pedestrian_walk(name)
                )
            else:
                responses.append(
                    communicator.unrealcv.tl_set_pedestrian_stop(name)
                )
                responses.append(
                    communicator.unrealcv.tl_set_vehicle_green(name)
                )
        time.sleep(0.05)
        state = communicator.get_traffic_signal_state(primary.id)
        if bool(state["pedestrian_walk"]) == walk:
            break
    rendered_walk = bool(state["pedestrian_walk"])
    if rendered_walk != walk:
        raise RuntimeError(
            "Rendered pedestrian signal did not match request: "
            f"crosswalk={crosswalk_id} signal={primary.id} "
            f"requested_walk={walk} state={state} responses={responses}"
        )
    state["forced_intersection_phase"] = phase
    state["phase_response"] = responses
    return rendered_walk, state


def draw_relative_panel(canvas, trace, origin, has_vehicle):
    height, width = canvas.shape[:2]
    panel_w, panel_h = 288, 255
    left, top = width - panel_w - 18, 132
    panel = np.full((panel_h, panel_w, 3), (15, 19, 25), dtype=np.uint8)
    cv2.putText(panel, "LIVE UE ACTOR POSITIONS", (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 235, 235), 1, cv2.LINE_AA)
    center = np.array([panel_w / 2, panel_h / 2 + 12])
    scale = 0.20
    actors = [("agent", (75, 230, 110), "AGENT")]
    if has_vehicle:
        actors.append(("vehicle", (45, 180, 255), "CAR"))
    for key, color, label in actors:
        points = []
        for item in trace[-80:]:
            if item.get(key) is None:
                continue
            rel = np.array(item[key]) - origin
            point = center + np.array([rel[0], -rel[1]]) * scale
            points.append(np.int32(np.clip(point, [0, 35], [panel_w - 1, panel_h - 1])))
        if len(points) > 1:
            cv2.polylines(panel, [np.array(points)], False, color, 2, cv2.LINE_AA)
        if points:
            cv2.circle(panel, tuple(points[-1]), 7, color, -1)
            cv2.putText(panel, label, tuple(points[-1] + np.array([8, -8])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                        cv2.LINE_AA)
    cv2.putText(panel, "green: agent", (12, panel_h - 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.39, (75, 230, 110), 1, cv2.LINE_AA)
    if has_vehicle:
        cv2.putText(panel, "orange: consequence vehicle", (12, panel_h - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, (45, 180, 255), 1,
                    cv2.LINE_AA)
    canvas[top:top + panel_h, left:left + panel_w] = panel
    cv2.rectangle(canvas, (left, top), (left + panel_w, top + panel_h),
                  (220, 220, 220), 1)


def annotate(frame, mode, elapsed, signal_walk, status, progress, distance,
             fallback_count, trace, origin, show_relative_panel,
             crossing_active):
    canvas = frame.copy()
    width = canvas.shape[1]
    cv2.rectangle(canvas, (0, 0), (width, 116), (10, 13, 18), -1)
    title = (
        "DEMO A | SAFE CROSSING WITHIN WALK PHASE"
        if mode == "safe"
        else "DEMO B | WALK EXPIRES MID-CROSSING -> 100% COLLISION"
    )
    cv2.putText(canvas, title, (20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                (245, 245, 245), 2, cv2.LINE_AA)
    signal_text = "WALK" if signal_walk else "DON'T WALK"
    vehicle_text = "RED / STOP" if signal_walk else "GREEN / GO"
    signal_color = (60, 220, 80) if signal_walk else (55, 70, 245)
    on_crosswalk = crossing_active and progress < 100.0
    violation = mode == "collision" and not signal_walk and on_crosswalk
    cv2.putText(
        canvas,
        f"PEDESTRIAN: {signal_text}  |  VEHICLE: {vehicle_text}  |  ON CROSSWALK: {str(on_crosswalk).upper()}  |  VIOLATION: {str(violation).upper()}",
        (20, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.57, signal_color, 2, cv2.LINE_AA,
    )
    third = f"AGENT: {status}  |  t={elapsed:04.1f}s  |  crossing progress={progress:05.1f}%"
    if distance is not None:
        third += f"  |  car distance={distance:05.1f}cm  |  fallback={fallback_count}"
    status_color = (70, 220, 90) if "COMPLETE" in status or "BRAKED" in status else (45, 185, 255)
    cv2.putText(canvas, third, (20, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                status_color, 2, cv2.LINE_AA)
    if show_relative_panel:
        draw_relative_panel(canvas, trace, origin, distance is not None)
    return canvas


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_label = args.demo_label or f"task_number_{args.task_number}"
    video_path = output_dir / f"{demo_label}_{args.mode}_crossing_demo.mp4"
    keyframe_path = output_dir / f"{demo_label}_{args.mode}_result.png"
    metrics_path = output_dir / f"{demo_label}_{args.mode}_metrics.json"

    unrealcv = RTUnrealCV(port=args.unrealcv_port, ip=args.unrealcv_host)
    communicator = RTCommunicator(unrealcv)
    manager = WorldManager(
        communicator,
        agent_path=args.agent_config,
        task_file_path=args.task_file,
        seed=7,
        use_tick=True,
        realtime_thinking=False,
        disable_background_agents=True,
    )
    writer = None
    suffix = str(int(time.time() * 1000))
    agent_name = f"RT_CROSSING_DEMO_AGENT_{args.mode}_{suffix}"
    vehicle_name = f"RT_CROSSING_DEMO_VEHICLE_{suffix}"
    trace = []
    try:
        manager.difficulty = "default"
        manager.scenario_data = manager.all_scenarios[args.task_number - 1]
        manager._initialize_world()
        unrealcv.client.request("vrun r.DefaultFeature.AutoExposure 1")
        unrealcv.client.request(
            "vrun r.DefaultFeature.AutoExposure.Bias "
            f"{args.exposure_offset:g}"
        )
        intersection = next(
            item for item in manager.traffic_controller.intersections
            if any(crosswalk.id == args.crosswalk_id for crosswalk in item.crosswalks)
        )
        crosswalk = next(
            item for item in intersection.crosswalks if item.id == args.crosswalk_id
        )
        communicator.spawn_traffic_signals(
            manager.traffic_controller.traffic_signals,
            manager.traffic_light_model_path,
            manager.pedestrian_light_model_path,
        )
        pedestrian_walk_duration = 20.0 if args.mode == "safe" else 3.0
        for traffic_signal in manager.traffic_controller.traffic_signals:
            communicator.traffic_signal_set_duration(
                traffic_signal.id,
                10.0,
                3.0,
                pedestrian_walk_duration,
            )
        time.sleep(2.0)
        # Demo A begins with an explicit vehicle-green phase so the first-person
        # recording shows that cars move while the pedestrian waits.  Demo B
        # begins with WALK and lets the rendered UE timer expire mid-crossing.
        signal_walk, signal_state = set_pedestrian_signal(
            communicator, intersection, crosswalk.id, args.mode == "collision"
        )

        crosswalk_start = np.array([crosswalk.start.x, crosswalk.start.y], dtype=np.float64)
        crosswalk_end = np.array([crosswalk.end.x, crosswalk.end.y], dtype=np.float64)
        axis = crosswalk_end - crosswalk_start
        length = float(np.linalg.norm(axis))
        axis /= length
        start = crosswalk_start + axis * 80.0
        destination = crosswalk_end - axis * 80.0
        midpoint = (crosswalk_start + crosswalk_end) / 2.0
        road_axis = np.array([-axis[1], axis[0]])

        unrealcv.spawn_bp_asset(args.agent_blueprint, agent_name)
        unrealcv.set_location((start[0], start[1], 100.0), agent_name)
        unrealcv.set_collision(agent_name, True)
        unrealcv.set_movable(agent_name, True)

        existing_cameras = str(unrealcv.get_cameras()).split()
        response = unrealcv.client.request("vset /cameras/spawn")
        if isinstance(response, str) and response.lower().startswith("error"):
            raise RuntimeError(response)
        camera = len(existing_cameras)
        unrealcv.set_camera_resolution(camera, (1280, 720))
        forward_yaw = math.degrees(math.atan2(axis[1], axis[0]))
        if args.camera_mode == "overhead":
            unrealcv.set_camera_location(camera, (midpoint[0], midpoint[1], 3600.0))
            unrealcv.set_camera_rotation(camera, (-90.0, 0.0, 0.0))
            unrealcv.set_camera_fov(camera, 58.0)
        else:
            unrealcv.set_camera_location(camera, (start[0], start[1], 250.0))
            unrealcv.set_camera_rotation(camera, (-7.0, forward_yaw, 0.0))
            unrealcv.set_camera_fov(camera, 95.0)
        time.sleep(1.0)

        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
            (1280, 720),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open MP4 writer")

        dt = 1.0 / args.fps
        elapsed = 0.0
        walk_at = 3.0 if args.mode == "safe" else None
        move_at = 3.7 if args.mode == "safe" else 0.8
        red_at = None
        move_duration = 4.2 if args.mode == "safe" else 6.0
        move_started = False
        vehicle_spawned = False
        vehicle_braked_for_walk = False
        contact_time = None
        fallback_count = 0
        status = (
            "VEHICLE GREEN / AGENT WAITING AT CURB"
            if args.mode == "safe"
            else "WAITING FOR WALK CROSSING"
        )
        final_frame = None

        # A controlled through vehicle makes the vehicle-green phase visible
        # from the waiting agent's viewpoint.  It is stopped before WALK begins.
        if args.mode == "safe":
            launch = midpoint - road_axis * 650.0
            yaw = math.degrees(math.atan2(road_axis[1], road_axis[0]))
            unrealcv.spawn_bp_asset(
                "/Game/TrafficSystem/Vehicle/Vehicle1.Vehicle1_C",
                vehicle_name,
            )
            unrealcv.set_location((launch[0], launch[1], 5.0), vehicle_name)
            unrealcv.set_orientation((0.0, yaw, 0.0), vehicle_name)
            unrealcv.set_collision(vehicle_name, True)
            unrealcv.set_movable(vehicle_name, True)
            unrealcv.v_set_state(vehicle_name, 0.75, 0.0, 0.0)
            vehicle_spawned = True

        demo_duration = 10.0 if args.mode == "safe" else 8.0
        for frame_index in range(int(demo_duration * args.fps) + 1):
            if (
                args.mode == "safe"
                and not signal_walk
                and elapsed >= walk_at
            ):
                unrealcv.v_set_state(vehicle_name, 0.0, 1.0, 0.0)
                vehicle_braked_for_walk = True
                signal_walk, signal_state = set_pedestrian_signal(
                    communicator, intersection, crosswalk.id, True
                )
                status = "VEHICLE STOPPED / PEDESTRIAN WALK"

            if not move_started and elapsed >= move_at:
                unrealcv.rt_agent_move_to(
                    agent_name, destination[0], destination[1], move_duration
                )
                move_started = True
                status = "CROSSING ON WALK"

            # The packaged Task5 navigation mesh can stop a long MoveTo at the
            # road centre. Keep the animation command above, but place the demo
            # actor deterministically along the actual crosswalk axis each tick.
            # This makes both matched recordings differ only by signal timing.
            if move_started and contact_time is None:
                crossing_alpha = min(
                    1.0, max(0.0, (elapsed - move_at) / move_duration)
                )
                scheduled_position = start + (destination - start) * crossing_alpha
                current_z = float(unrealcv.get_location(agent_name)[2])
                unrealcv.set_location(
                    (scheduled_position[0], scheduled_position[1], current_z),
                    agent_name,
                )

            # The rendered Blueprint timer is authoritative.  Demo A gives it
            # enough WALK time for the crossing; demo B uses a short WALK and
            # waits for UE itself to report DON'T WALK before declaring a
            # violation and releasing the consequence vehicle.
            signal_state = communicator.get_traffic_signal_state(
                get_crosswalk_signal(intersection, crosswalk.id).id
            )
            observed_walk = bool(signal_state["pedestrian_walk"])
            if args.mode == "safe":
                signal_walk = observed_walk
            elif signal_walk and not observed_walk and move_started:
                signal_walk = False
                red_at = elapsed
                status = "RED-LIGHT VIOLATION DETECTED"
                agent_now = vec2(unrealcv.get_location(agent_name))
                launch = agent_now - road_axis * 450.0
                yaw = math.degrees(math.atan2(road_axis[1], road_axis[0]))
                unrealcv.spawn_bp_asset(
                    "/Game/TrafficSystem/Vehicle/Vehicle1.Vehicle1_C",
                    vehicle_name,
                )
                unrealcv.set_location((launch[0], launch[1], 5.0), vehicle_name)
                unrealcv.set_orientation((0.0, yaw, 0.0), vehicle_name)
                unrealcv.set_collision(vehicle_name, True)
                unrealcv.set_movable(vehicle_name, True)
                unrealcv.v_set_state(vehicle_name, 1.0, 0.0, 0.0)
                vehicle_spawned = True
                status = "COLLISION VEHICLE APPROACHING"
            elif args.mode == "collision":
                signal_walk = observed_walk

            agent_position = vec2(unrealcv.get_location(agent_name))
            progress = float(np.dot(agent_position - crosswalk_start, axis) / length * 100.0)
            distance = None
            vehicle_position = None
            if vehicle_spawned:
                vehicle_raw = unrealcv.get_location(vehicle_name)
                vehicle_position = vec2(vehicle_raw)
                distance = float(np.linalg.norm(agent_position - vehicle_position))
                since_red = elapsed - red_at if red_at is not None else 0.0
                if (
                    args.mode == "collision"
                    and contact_time is None
                    and since_red >= 2.0
                    and distance > CONTACT_ENVELOPE_CM
                ):
                    direction = (agent_position - vehicle_position) / max(distance, 1e-6)
                    next_position = vehicle_position + direction * min(60.0, distance)
                    z = float(vehicle_raw[2]) if len(vehicle_raw) > 2 else 5.0
                    unrealcv.set_orientation(
                        (0.0, math.degrees(math.atan2(direction[1], direction[0])), 0.0),
                        vehicle_name,
                    )
                    unrealcv.set_location((next_position[0], next_position[1], z), vehicle_name)
                    fallback_count += 1
                if (
                    args.mode == "collision"
                    and contact_time is None
                    and distance <= CONTACT_ENVELOPE_CM
                ):
                    contact_time = elapsed
                    status = "COLLISION OBSERVED / VEHICLE BRAKED"
                    unrealcv.v_set_state(vehicle_name, 0.0, 1.0, 0.0)

            if args.mode == "safe" and progress >= 90.0:
                status = "SAFE CROSSING COMPLETE"

            trace.append(
                {
                    "time_s": round(elapsed, 3),
                    "agent": agent_position.tolist(),
                    "vehicle": vehicle_position.tolist() if vehicle_position is not None else None,
                    "signal": "WALK" if signal_walk else "DONT_WALK",
                    "primary_signal_id": signal_state["signal_id"],
                    "rendered_pedestrian_walk": bool(
                        signal_state["pedestrian_walk"]
                    ),
                    "signal_source": signal_state["source"],
                    "progress_pct": round(progress, 2),
                    "distance_cm": round(distance, 2) if distance is not None else None,
                    "status": status,
                    "crossing_active": move_started and progress < 100.0,
                }
            )
            if args.camera_mode == "first-person":
                agent_raw = unrealcv.get_location(agent_name)
                camera_yaw = forward_yaw
                if args.mode == "safe" and not move_started and vehicle_position is not None:
                    toward_vehicle = vehicle_position - agent_position
                    camera_yaw = math.degrees(
                        math.atan2(toward_vehicle[1], toward_vehicle[0])
                    )
                    if signal_walk:
                        blend = min(1.0, max(0.0, (elapsed - walk_at) / 0.6))
                        yaw_delta = (forward_yaw - camera_yaw + 180.0) % 360.0 - 180.0
                        camera_yaw += yaw_delta * blend
                elif vehicle_position is not None and args.mode == "collision":
                    toward_vehicle = vehicle_position - agent_position
                    threat_yaw = math.degrees(
                        math.atan2(toward_vehicle[1], toward_vehicle[0])
                    )
                    yaw_delta = (threat_yaw - forward_yaw + 180.0) % 360.0 - 180.0
                    look_blend = min(1.0, max(0.0, (elapsed - red_at) / 0.6))
                    camera_yaw = forward_yaw + yaw_delta * look_blend
                camera_xy = agent_position + axis * 25.0
                unrealcv.set_camera_location(
                    camera,
                    (camera_xy[0], camera_xy[1], float(agent_raw[2]) + 155.0),
                )
                unrealcv.set_camera_rotation(camera, (-7.0, camera_yaw, 0.0))

            frame = unrealcv.get_image(camera, "lit", "direct")
            annotated = annotate(
                frame, args.mode, elapsed, signal_walk, status, progress,
                distance, fallback_count, trace, midpoint,
                args.camera_mode == "overhead",
                move_started and progress < 100.0,
            )
            writer.write(annotated)
            if final_frame is None and (
                (args.mode == "safe" and status == "SAFE CROSSING COMPLETE")
                or (args.mode == "collision" and contact_time is not None)
            ):
                final_frame = annotated.copy()

            if frame_index < int(demo_duration * args.fps):
                unrealcv.advance_simulation_time(dt, time_scale=1.0)
                elapsed += dt

        if final_frame is not None:
            cv2.imwrite(str(keyframe_path), final_frame)
        safe_complete = (
            args.mode == "safe"
            and max(item["progress_pct"] for item in trace) >= 90.0
            and all(
                item["signal"] == "WALK"
                for item in trace
                if item["crossing_active"] and item["progress_pct"] < 90.0
            )
        )
        violation_observed = any(
            item["signal"] == "DONT_WALK"
            and 0.0 < item["progress_pct"] < 100.0
            for item in trace
        )
        collision_complete = (
            args.mode == "collision"
            and violation_observed
            and contact_time is not None
        )
        metrics = {
            "task": demo_label,
            "task_file": args.task_file,
            "task_number": args.task_number,
            "scenario_task_id": manager.scenario_data.get("task_id"),
            "crosswalk_id": crosswalk.id,
            "mode": args.mode,
            "camera_mode": args.camera_mode,
            "passed": safe_complete if args.mode == "safe" else collision_complete,
            "pedestrian_signal_sequence": list(dict.fromkeys(item["signal"] for item in trace)),
            "configured_pedestrian_walk_duration_s": pedestrian_walk_duration,
            "primary_signal_id": signal_state["signal_id"],
            "signal_source": signal_state["source"],
            "red_light_violation": args.mode == "collision" and violation_observed,
            "red_light_collision_probability": 1.0 if args.mode == "collision" else None,
            "maximum_crossing_progress_pct": max(item["progress_pct"] for item in trace),
            "contact_time_s": round(contact_time, 3) if contact_time is not None else None,
            "minimum_center_distance_cm": min(
                (item["distance_cm"] for item in trace if item["distance_cm"] is not None),
                default=None,
            ),
            "kinematic_fallback_advance_count": fallback_count,
            "vehicle_green_motion_shown": args.mode == "safe",
            "vehicle_braked_before_walk": (
                vehicle_braked_for_walk if args.mode == "safe" else None
            ),
            "video": video_path.name,
            "keyframe": keyframe_path.name if final_frame is not None else None,
            "trace": trace,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in metrics.items() if key != "trace"}, indent=2))
        if not metrics["passed"]:
            raise RuntimeError(f"{args.mode} demo did not satisfy its pass condition")
    finally:
        if writer is not None:
            writer.release()
        unrealcv.disconnect()


if __name__ == "__main__":
    main()
