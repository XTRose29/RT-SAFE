#!/usr/bin/env python3
"""Record full first-person task demos for the two violation triggers.

The ordinary task setup is retained (traffic lights, background assets and a
small set of staged real-lane vehicles).  A deterministic pedestrian driver is
used only to guarantee the requested violation.  The vehicle is launched and
retired by ``WorldManager``'s production consequence path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Match the benchmark runner's lightweight package loading.  Importing the
# top-level SimWorld/simworld/__init__.py eagerly pulls optional retrieval
# models (sentence-transformers) that traffic rollout does not use.
REPO_ROOT = Path(__file__).resolve().parents[1]
simworld = types.ModuleType("simworld")
simworld.__path__ = [str(REPO_ROOT / "SimWorld" / "simworld")]
sys.modules.setdefault("simworld", simworld)

from base.rt_communicator import RTCommunicator
from base.rt_unrealcv import RTUnrealCV
from manager.world_manager import (
    WorldManager,
    reconstruct_route_points_from_task_edges,
)
from simworld.utils.vector import Vector


class PreparedWorldManager(WorldManager):
    """Prepare a normal task world without entering the VLM decision loop."""

    def run(self):
        return None


def _same_point(raw, point: Vector, tolerance: float = 2.0) -> bool:
    return Vector(float(raw[0]), float(raw[1])).distance(point) <= tolerance


def _illegal_crossing_lane_offset(route_projection: float, segment_length: float):
    """Choose a lane point with room for a 100--500 cm upstream launch.

    The pedestrian should enter a real vehicle lane away from the zebra, but
    the selected point must also leave at least 100 cm behind it along the
    vehicle's travel direction.  The previous symmetric "larger side" choice
    could put the point only 60 cm from the segment start, making the production
    fail-closed launcher correctly reject an otherwise useful illegal-crossing demo.
    """
    before = float(route_projection) * float(segment_length)
    after = (1.0 - float(route_projection)) * float(segment_length)
    options = []
    if after >= 180.0:
        options.append((min(500.0, after - 60.0), 1.0))
    # Moving upstream reduces the remaining launch runway.  Keep 100 cm for
    # the minimum launch and 5 cm for the production segment-end guard.
    if before >= 285.0:
        options.append((min(500.0, before - 105.0), -1.0))
    options = [item for item in options if item[0] >= 120.0]
    if not options:
        return None
    offset, sign = max(options, key=lambda item: (item[0], item[1]))
    return sign, offset


def _route_crosswalk(task_data: dict, manager: WorldManager):
    route = [Vector(float(x), float(y)) for x, y in reconstruct_route_points_from_task_edges(task_data)]
    cross_edges = [edge for edge in task_data.get("edges", []) if edge.get("type") == "crosswalk"]
    if not cross_edges:
        raise RuntimeError("selected task has no ordered crosswalk edge")
    for index, (start, end) in enumerate(zip(route, route[1:])):
        for edge in cross_edges:
            normal = _same_point(edge["node1"], start) and _same_point(edge["node2"], end)
            reverse = _same_point(edge["node2"], start) and _same_point(edge["node1"], end)
            if not (normal or reverse):
                continue
            candidates = list(manager.agent.route_crosswalks)
            if not candidates:
                raise RuntimeError("task crosswalk was not matched to traffic geometry")
            crosswalk = min(
                candidates,
                key=lambda item: min(
                    item.start.distance(start) + item.end.distance(end),
                    item.end.distance(start) + item.start.distance(end),
                ),
            )
            return route, index, start, end, crosswalk, edge
    raise RuntimeError("ordered route did not contain its declared crosswalk edge")


def _camera_frame(unrealcv: RTUnrealCV, camera_id: int):
    frame = unrealcv.get_image(camera_id, "lit", "direct")
    if frame is None:
        raise RuntimeError("UnrealCV returned an empty first-person frame")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("red_light", "illegal_crossing"), required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--task-number", type=int, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="World/launch RNG seed (vary this across collision-rate trials).",
    )
    parser.add_argument("--agent-config", default="data/agents_qwen3vl8b.json")
    parser.add_argument("--unrealcv-host", default="127.0.0.1")
    parser.add_argument("--unrealcv-port", type=int, default=9001)
    parser.add_argument(
        "--difficulty",
        choices=(
            "level0",
            "level1",
            "default",
            "medium",
            "easy",
        ),
        default="level0",
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fov-deg", type=float, default=70.0)
    parser.add_argument("--camera-eye-height-cm", type=float, default=135.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--exposure-offset", type=float, default=2.0)
    parser.add_argument("--camera-warmup-frames", type=int, default=8)
    parser.add_argument("--camera-warmup-delay-seconds", type=float, default=0.15)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument(
        "--capture-stride",
        type=int,
        default=1,
        help=(
            "Capture every Nth route/wait step; simulation and collision "
            "updates still run on every step. Default 1 preserves full video."
        ),
    )
    parser.add_argument(
        "--simulation-step-seconds",
        type=float,
        default=None,
        help=(
            "Simulation time advanced per captured frame. Defaults to 1/fps; "
            "a larger value speeds up full-route capture without omitting any "
            "task segment."
        ),
    )
    parser.add_argument("--walking-speed-cm-s", type=float, default=200.0)
    parser.add_argument(
        "--launch-distance-min-cm",
        type=float,
        default=300.0,
        help="Minimum randomized conflict-vehicle launch distance.",
    )
    parser.add_argument(
        "--launch-distance-max-cm",
        type=float,
        default=900.0,
        help="Maximum randomized conflict-vehicle launch distance.",
    )
    parser.add_argument(
        "--post-launch-agent-motion",
        choices=("hold", "continue"),
        default="hold",
        help=(
            "Keep the pedestrian at the lane conflict point (the original "
            "deterministic demo), or continue across at --walking-speed-cm-s "
            "to measure collision probability after a launch."
        ),
    )
    parser.add_argument(
        "--accept-outcome",
        choices=("collision", "collision-or-miss"),
        default="collision",
        help=(
            "The original demo requires a collision. Distance experiments "
            "can accept either a collision or a clean target passage."
        ),
    )
    parser.add_argument(
        "--pressure-fast-forward",
        action="store_true",
        help=(
            "Teleport the deterministic driver to the route curb and force "
            "DON'T WALK before invoking the production consequence path. "
            "Intended only for repeated launch-distance statistics."
        ),
    )
    parser.add_argument(
        "--qwen-one-step",
        action="store_true",
        help=(
            "After the vehicle launches, show it to Qwen and execute exactly "
            "one model decision. The agent then holds its resulting pose while "
            "the launched vehicle completes."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # These are consumed when RTAgent is constructed below.  Keep the demo
    # camera identical to the policy input: a genuine eye-pose free camera,
    # enough pixels for the small pedestrian head, and a narrow-enough FOV to
    # preserve its readable footprint without hiding the crossing corridor.
    os.environ["SIMWORLD_FIRST_PERSON_CAMERA"] = "1"
    os.environ["SIMWORLD_AGENT_CAMERA_WIDTH"] = str(args.camera_width)
    os.environ["SIMWORLD_AGENT_CAMERA_HEIGHT"] = str(args.camera_height)
    os.environ["SIMWORLD_AGENT_CAMERA_FOV_DEG"] = str(args.camera_fov_deg)
    os.environ["SIMWORLD_FIRST_PERSON_EYE_HEIGHT_OFFSET_CM"] = str(
        args.camera_eye_height_cm
    )
    os.environ["SIMWORLD_FIRST_PERSON_CAMERA_PITCH_DEG"] = str(
        args.camera_pitch_deg
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{args.mode}_full_first_person.mp4"
    metrics_path = output_dir / f"{args.mode}_metrics.json"
    terminal_frame_path = output_dir / f"{args.mode}_terminal.png"
    dt = (
        float(args.simulation_step_seconds)
        if args.simulation_step_seconds is not None
        else 1.0 / args.fps
    )
    if dt <= 0.0:
        raise ValueError("--simulation-step-seconds must be positive")
    if args.capture_stride < 1:
        raise ValueError("--capture-stride must be at least 1")
    if args.launch_distance_min_cm < 0.0:
        raise ValueError("--launch-distance-min-cm must be non-negative")
    if args.launch_distance_max_cm < args.launch_distance_min_cm:
        raise ValueError(
            "--launch-distance-max-cm must be greater than or equal to the minimum"
        )

    unrealcv = RTUnrealCV(port=args.unrealcv_port, ip=args.unrealcv_host)
    communicator = RTCommunicator(unrealcv)
    manager = PreparedWorldManager(
        communicator,
        agent_path=args.agent_config,
        task_file_path=args.task_file,
        seed=args.seed,
        use_tick=True,
        token_based=True,
        realtime_thinking=args.qwen_one_step,
        use_action_frames=False,
        red_light_conflict_vehicle_enabled=True,
        red_light_conflict_vehicle_probability=1.0,
        red_light_conflict_launch_distance_min_cm=args.launch_distance_min_cm,
        red_light_conflict_launch_distance_max_cm=args.launch_distance_max_cm,
        red_light_conflict_collision_radius_cm=100.0,
        static_signal_vehicles=True,
        pedestrian_signal_compliance_probability=1.0,
    )
    manager.render_auto_exposure = True
    manager.render_exposure_offset = args.exposure_offset
    writer = None
    frames_written = 0
    trace = []
    trigger_record = None
    signal_state = None
    qwen_decision = None
    qwen_calls = 0
    milestone_frames = {}
    brightness_samples = []

    def capture(label: str):
        nonlocal frames_written
        manager.agent._sync_first_person_camera()
        frame = _camera_frame(unrealcv, manager.agent.camera_id)
        writer.write(frame)
        frames_written += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_samples.append(
            {
                "mean": float(np.mean(gray)),
                "p99": float(np.percentile(gray, 99)),
                "clipped_white_fraction": float(np.mean(gray >= 250)),
            }
        )
        if label in {
            "curb_facing_relevant_signal",
            "vehicle_launched_camera_facing_approach",
            "terminal",
        } and label not in milestone_frames:
            milestone_name = f"{args.mode}_{label}.png"
            cv2.imwrite(str(output_dir / milestone_name), frame)
            milestone_frames[label] = milestone_name
        trace.append(
            {
                "frame": frames_written - 1,
                "label": label,
                "sim_time_s": round(float(manager.agent.sim_time_elapsed), 3),
                "agent_position": {
                    "x": round(float(manager.agent.position.x), 2),
                    "y": round(float(manager.agent.position.y), 2),
                },
                "failure_reason": manager.agent.failure_reason,
                "conflict_active": manager._active_red_light_conflict is not None,
            }
        )
        return frame

    def set_agent_position(position: Vector, direction: Vector):
        raw = unrealcv.get_location(manager.agent.name)
        z = float(raw[2]) if len(raw) > 2 else 100.0
        yaw = math.degrees(math.atan2(direction.y, direction.x))
        unrealcv.set_orientation((0.0, yaw, 0.0), manager.agent.name)
        unrealcv.set_location((position.x, position.y, z), manager.agent.name)
        manager.agent.position = Vector(position.x, position.y)
        manager.agent.direction = yaw

    def walk_to(target: Vector, label: str, stop_on_failure: bool = True):
        start = Vector(manager.agent.position.x, manager.agent.position.y)
        delta = target - start
        distance = delta.length()
        if distance <= 1e-6:
            return
        direction = delta.normalize()
        steps = max(1, int(math.ceil(distance / args.walking_speed_cm_s / dt)))
        for step in range(1, steps + 1):
            alpha = step / steps
            position = start + delta * alpha
            set_agent_position(position, direction)
            manager.agent._advance_simulation_time(dt)
            if step % args.capture_stride == 0 or step == steps:
                capture(label)
            if stop_on_failure and manager.agent.failed:
                break

    try:
        manager.run_single_task(args.task_number - 1, difficulty=args.difficulty)
        manager.agent.sync_ue()
        (
            route,
            cross_index,
            cross_entry,
            cross_exit,
            crosswalk,
            crosswalk_edge,
        ) = _route_crosswalk(manager.scenario_data["task"], manager)
        # Auto exposure starts adapting only after the free first-person camera
        # renders.  Without a short visual warmup the first seconds of an
        # otherwise correctly configured demo are almost black.  Render at the
        # unchanged task-spawn pose and do not advance simulation time, so the
        # delivered video still begins at the genuine task start.
        first = None
        for _ in range(max(1, args.camera_warmup_frames)):
            manager.agent._sync_first_person_camera()
            first = _camera_frame(unrealcv, manager.agent.camera_id)
            if args.camera_warmup_delay_seconds > 0:
                time.sleep(args.camera_warmup_delay_seconds)
        if first is None:
            raise RuntimeError("first-person camera warmup returned no frame")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open the full-task MP4 writer")
        writer.write(first)
        frames_written = 1
        first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        brightness_samples.append(
            {
                "mean": float(np.mean(first_gray)),
                "p99": float(np.percentile(first_gray, 99)),
                "clipped_white_fraction": float(np.mean(first_gray >= 250)),
            }
        )
        task_start_name = f"{args.mode}_task_start.png"
        cv2.imwrite(str(output_dir / task_start_name), first)
        milestone_frames["task_start"] = task_start_name
        trace.append(
            {
                "frame": 0,
                "label": "task_start",
                "sim_time_s": round(float(manager.agent.sim_time_elapsed), 3),
                "agent_position": {
                    "x": round(float(manager.agent.position.x), 2),
                    "y": round(float(manager.agent.position.y), 2),
                },
                "failure_reason": None,
                "conflict_active": False,
            }
        )

        # Follow the ordered task sidewalk from the actual task start to the
        # curb.  The deliberate violation begins only after this prefix.
        if not args.pressure_fast_forward:
            for waypoint in route[1 : cross_index + 1]:
                walk_to(waypoint, "ordered_sidewalk_prefix")

        crossing_direction = (cross_exit - cross_entry).normalize()
        post_launch_direction = crossing_direction
        # Face the active crossing once at the curb.  This makes the relevant
        # pedestrian head and the zebra corridor visible in the genuine
        # first-person video before either deliberate violation begins.
        set_agent_position(cross_entry, crossing_direction)
        capture("curb_facing_relevant_signal")
        if args.mode == "red_light":
            signal = manager.agent._signal_for_crossing(
                manager.agent.position,
                cross_exit,
                crosswalk_edge,
            )
            if signal is None:
                raise RuntimeError("no controlling pedestrian signal for route crosswalk")
            # Keep the fixed cycle.  Wait at the curb until UE reports DON'T
            # WALK, recording every frame from the task start onward.
            if args.pressure_fast_forward:
                actor_name = manager.communicator.get_traffic_signal_name(signal.id)
                unrealcv.tl_set_pedestrian_stop(actor_name)
                signal_state = {
                    "pedestrian_walk": False,
                    "forced_for_pressure_trial": True,
                }
            else:
                for wait_step in range(int(math.ceil(45.0 / dt))):
                    # Signal heads exist even when signal-following background
                    # traffic counts are zero.  In that valid benchmark setup the
                    # optional traffic-only communicator is None; query the same
                    # primary communicator used by RTAgent's visual-only signal
                    # readback instead.
                    signal_state = manager.communicator.get_traffic_signal_state(signal.id)
                    if not bool(signal_state.get("pedestrian_walk")):
                        break
                    manager.agent._advance_simulation_time(dt)
                    if (wait_step + 1) % args.capture_stride == 0:
                        capture("waiting_for_dont_walk")
            if signal_state is None or bool(signal_state.get("pedestrian_walk")):
                raise RuntimeError("fixed cycle did not reach DON'T WALK within 45s")

            # The consequence must travel to the position where the agent is
            # actually standing in a real vehicle lane.  Probe the production
            # lane selector, walk through the red zebra to that lane centre,
            # and only then launch the one-shot car in the requested upstream
            # distance range.
            crosswalk_axis = (crosswalk.end - crosswalk.start).normalize()
            crosswalk_length = max(
                crosswalk.start.distance(crosswalk.end), 1e-6
            )
            entry_projection = (
                (cross_entry - crosswalk.start).dot(crosswalk_axis)
                / crosswalk_length
            )
            entry_alignment = crossing_direction.dot(crosswalk_axis)
            entry_direction = 1 if entry_alignment >= 0 else -1
            probe_event = {
                "crosswalk_projection": entry_projection,
                "crosswalk_entry_direction": entry_direction,
                "agent_position": {"x": cross_entry.x, "y": cross_entry.y},
                "agent_direction": {
                    "x": crossing_direction.x,
                    "y": crossing_direction.y,
                },
                "agent_speed_cm_s": args.walking_speed_cm_s,
            }
            lane_probe = manager._lane_aligned_conflict_launch(
                probe_event,
                crosswalk,
                list(manager.traffic_controller.vehicles),
                args.launch_distance_max_cm,
            )
            if lane_probe is None:
                raise RuntimeError(
                    "no real staged vehicle lane is available for red-light demo"
                )
            trigger_position = lane_probe["target_position"]
            walk_to(
                trigger_position,
                "red_light_violation_to_vehicle_lane",
                stop_on_failure=False,
            )
            projection = (
                (trigger_position - crosswalk.start).dot(crosswalk_axis)
                / crosswalk_length
            )
            event = {
                "event_id": "demo-red-light-violation",
                "crosswalk_id": crosswalk.id,
                "sim_time_s": float(manager.agent.sim_time_elapsed),
                "pedestrian_state": "DONT_WALK",
                "crosswalk_projection": projection,
                "crosswalk_entry_direction": entry_direction,
                "agent_position": {"x": trigger_position.x, "y": trigger_position.y},
                "agent_direction": {"x": crossing_direction.x, "y": crossing_direction.y},
                "agent_speed_cm_s": args.walking_speed_cm_s,
            }
            trigger_record = manager._handle_red_light_violation(event, crosswalk)
            if trigger_record.get("disposition") != "launched":
                raise RuntimeError(f"red-light consequence was not launched: {trigger_record}")

        else:
            # Use an authored staged-car segment that intersects this task's
            # crossing, shift parallel to the road (away from the zebra), and
            # walk from the sidewalk directly toward that real vehicle lane.
            lane_choice = None
            for vehicle_id, vehicle_route in manager._signal_vehicle_routes.items():
                points = [vehicle_route["approach_start"], *vehicle_route["path_points"]]
                for segment_index, (start, end) in enumerate(zip(points, points[1:])):
                    intersection = manager._segment_intersection_point(
                        start, end, crosswalk.start, crosswalk.end
                    )
                    if intersection is None:
                        continue
                    target, route_projection, _ = intersection
                    segment = end - start
                    segment_length = segment.length()
                    if segment_length <= 1e-6:
                        continue
                    direction = segment.normalize()
                    lane_offset = _illegal_crossing_lane_offset(
                        route_projection,
                        segment_length,
                    )
                    if lane_offset is None:
                        continue
                    sign, offset = lane_offset
                    lane_choice = (
                        vehicle_id,
                        segment_index,
                        target + direction * (sign * offset),
                        direction,
                        target,
                    )
                    break
                if lane_choice is not None:
                    break
            if lane_choice is None:
                raise RuntimeError("no real staged vehicle lane crosses the route crosswalk")
            vehicle_id, segment_index, lane_point, lane_direction, crossing_target = lane_choice
            curb_direction = (cross_entry - crossing_target).normalize()
            curb_distance = cross_entry.distance(crossing_target)
            sidewalk_point = lane_point + curb_direction * curb_distance
            walk_to(sidewalk_point, "sidewalk_parallel_to_illegal_crossing_point")
            toward_lane = (lane_point - sidewalk_point).normalize()
            post_launch_direction = toward_lane
            trigger_position = lane_point
            walk_to(
                trigger_position,
                "leaving_sidewalk_to_vehicle_lane",
                stop_on_failure=False,
            )
            event = {
                "event_id": "demo-unsafe-road-entry",
                "trigger_type": "illegal_crossing",
                "sim_time_s": float(manager.agent.sim_time_elapsed),
                "pedestrian_state": "NOT_APPLICABLE",
                "agent_position": {"x": trigger_position.x, "y": trigger_position.y},
                "agent_direction": {"x": toward_lane.x, "y": toward_lane.y},
                "agent_speed_cm_s": args.walking_speed_cm_s,
                "source_route_vehicle_id": vehicle_id,
                "source_route_segment_index": segment_index,
            }
            trigger_record = manager._handle_illegal_crossing(event)
            if trigger_record.get("disposition") != "launched":
                raise RuntimeError(f"illegal_crossing consequence was not launched: {trigger_record}")
            manager.agent.illegal_crossing_excursion_active = True

        # Hold at the original trigger point and turn the first-person view
        # exactly once toward the launched vehicle so its real-lane approach,
        # collision and disappearance are visible rather than happening just
        # outside a narrow forward FOV.
        launch_position = Vector(
            float(trigger_record["launch_position"]["x"]),
            float(trigger_record["launch_position"]["y"]),
        )
        look_direction = launch_position - manager.agent.position
        if look_direction.length() <= 1e-6:
            raise RuntimeError("conflict vehicle launch position equals trigger point")
        set_agent_position(manager.agent.position, look_direction.normalize())
        capture("vehicle_launched_camera_facing_approach")

        if args.qwen_one_step:
            decisions_before = int(manager.agent.decision_count)
            manager.agent.step(mode="llm")
            manager._record_ue_vehicle_collision_outcome()
            qwen_calls = int(manager.agent.decision_count) - decisions_before
            if qwen_calls != 1:
                raise RuntimeError(
                    f"expected exactly one Qwen call, observed {qwen_calls}"
                )
            if manager.agent.decision_trace:
                qwen_decision = dict(manager.agent.decision_trace[-1])

        # Advance the production runtime path.  The original recorded demo
        # holds the pedestrian at the conflict point.  Probability studies
        # instead let it continue across at the same declared walking speed,
        # so longer launch gaps can produce genuine misses.
        for _ in range(int(math.ceil(15.0 / dt))):
            if manager.agent.failed:
                break
            if (
                not args.qwen_one_step
                and args.post_launch_agent_motion == "continue"
            ):
                next_position = (
                    manager.agent.position
                    + post_launch_direction * args.walking_speed_cm_s * dt
                )
                set_agent_position(next_position, post_launch_direction)
            manager.agent._advance_simulation_time(dt)
            manager.agent._check_vehicle_collision_failure()
            manager._record_ue_vehicle_collision_outcome()
            capture("conflict_vehicle_approach")
            if manager._active_red_light_conflict is None:
                break
        terminal_frame = capture("terminal")
        cv2.imwrite(str(terminal_frame_path), terminal_frame)

        record = trigger_record or {}
        trigger_target = Vector(
            float(record.get("target_position", {}).get("x", float("inf"))),
            float(record.get("target_position", {}).get("y", float("inf"))),
        )
        target_error_cm = trigger_target.distance(trigger_position)
        event_records = (
            manager.red_light_conflict_vehicle_events
            if args.mode == "red_light"
            else manager.illegal_crossing_conflict_vehicle_events
        )
        event_count = len(event_records)
        launched_event_count = sum(
            record.get("disposition") == "launched"
            for record in event_records
        )
        brightness_means = [sample["mean"] for sample in brightness_samples]
        brightness_p99 = [sample["p99"] for sample in brightness_samples]
        clipped_white = [
            sample["clipped_white_fraction"] for sample in brightness_samples
        ]
        collision_outcome = bool(
            record.get("collision_triggered")
            and manager.agent.failed
            and manager.agent.failure_reason == "vehicle_collision"
        )
        miss_outcome = bool(
            not record.get("collision_triggered")
            and record.get("retired_after_consequence")
            and record.get("status") in {"target_reached", "completed"}
            and not manager.agent.failed
        )
        accepted_outcome = collision_outcome or (
            args.accept_outcome == "collision-or-miss" and miss_outcome
        )
        passed = bool(
            record.get("disposition") == "launched"
            and record.get("control_mode") == "lane_aligned_intercept"
            and record.get("physics_quiesced_during_relocation") is True
            and args.launch_distance_min_cm
            <= float(record.get("launch_distance_cm", -1.0))
            <= args.launch_distance_max_cm
            and target_error_cm <= 2.0
            and launched_event_count == 1
            and (args.qwen_one_step or event_count == 1)
            and record.get("retired_after_consequence")
            and accepted_outcome
            and manager._active_red_light_conflict is None
        )
        metrics = {
            "passed": passed,
            "mode": args.mode,
            "task_file": args.task_file,
            "task_number": args.task_number,
            "task_id": manager.current_task_id,
            "seed": args.seed,
            "experiment": {
                "launch_distance_range_cm": [
                    args.launch_distance_min_cm,
                    args.launch_distance_max_cm,
                ],
                "post_launch_agent_motion": args.post_launch_agent_motion,
                "walking_speed_cm_s": args.walking_speed_cm_s,
                "accepted_outcome": args.accept_outcome,
                "pressure_fast_forward": args.pressure_fast_forward,
                "qwen_one_step": args.qwen_one_step,
                "qwen_calls": qwen_calls,
                "qwen_decision": qwen_decision,
                "observed_outcome": (
                    "collision"
                    if collision_outcome
                    else "miss"
                    if miss_outcome
                    else "invalid"
                ),
            },
            "camera": {
                "mode": manager.agent.first_person_camera_mode,
                "camera_id": manager.agent.camera_id,
                "resolution": [width, height],
                "fov_deg": manager.agent.fov,
                "exposure_bias": args.exposure_offset,
                "warmup_frames_at_task_start": max(
                    1, args.camera_warmup_frames
                ),
                "warmup_advances_simulation": False,
                "eye_height_offset_cm": (
                    manager.agent.first_person_eye_height_offset_cm
                ),
                "pitch_deg": manager.agent.first_person_camera_pitch_deg,
                "milestone_frames": milestone_frames,
                "brightness": {
                    "frame_count": len(brightness_samples),
                    "mean_luma_median": round(
                        float(np.median(brightness_means)), 3
                    ),
                    "p99_luma_median": round(float(np.median(brightness_p99)), 3),
                    "clipped_white_fraction_median": round(
                        float(np.median(clipped_white)), 6
                    ),
                    "clipped_white_fraction_max": round(
                        float(max(clipped_white)), 6
                    ),
                },
            },
            "full_task_recording": {
                "starts_at_task_spawn": True,
                "ends_at_terminal_state": True,
                "frames": frames_written,
                "fps": args.fps,
                "simulation_step_seconds": dt,
                "capture_stride": args.capture_stride,
                "video": video_path.name,
                "terminal_frame": terminal_frame_path.name,
            },
            "terminal": {
                "failed": manager.agent.failed,
                "failure_reason": manager.agent.failure_reason,
                "vehicle_collision_count": manager.agent.vehicle_collision_count,
            },
            "static_background_vehicle_count": manager.signal_traffic_vehicle_count,
            "trigger_event_count": event_count,
            "launched_trigger_event_count": launched_event_count,
            "trigger_target_error_cm": round(target_error_cm, 3),
            "trigger_record": record,
            "trace": trace,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in metrics.items() if key != "trace"}, indent=2))
        if not passed:
            raise RuntimeError(f"{args.mode} full-task collision demo failed acceptance")
    finally:
        if writer is not None:
            writer.release()
        try:
            manager.cleanup()
        finally:
            unrealcv.disconnect()


if __name__ == "__main__":
    main()
