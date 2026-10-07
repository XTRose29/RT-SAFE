#!/usr/bin/env python3
"""Benchmark pedestrian-light readability in a fully populated UE scene.

The benchmark keeps the agent and scene fixed while varying only camera and
render settings.  It captures both WALK and DON'T WALK, measures render time,
and uses UnrealCV's object-mask view to report the visible pixel footprint of
every pedestrian head controlling the route crosswalk.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.rt_action_space import MOVE_TO, RTActionSpace
from base.rt_agent import RTAgent
from base.rt_communicator import RTCommunicator
from base.rt_unrealcv import RTUnrealCV
from manager.world_manager import WorldManager, reconstruct_route_points_from_task_edges
from simworld.utils.vector import Vector
from utils.annotate_image import annotate_image, project_waypoints_to_image


PROFILES = (
    {
        "name": "benchmark_720x640_fov100_pitch_m25",
        "resolution": (720, 640),
        "fov": 100.0,
        "camera_pitch": -25.0,
        "exposure_bias": 2.0,
        "aa_quality": 4,
    },
    {
        "name": "legacy_720x640_fov70_level",
        "resolution": (720, 640),
        "fov": 70.0,
        "camera_pitch": 0.0,
        "exposure_bias": 2.0,
        "aa_quality": 4,
    },
    {
        "name": "production_800x640_fov75_bias0",
        "resolution": (800, 640),
        "fov": 75.0,
        "exposure_bias": 0.0,
        "aa_quality": 4,
    },
    {
        "name": "production_800x640_fov75_bias0_5",
        "resolution": (800, 640),
        "fov": 75.0,
        "exposure_bias": 0.5,
        "aa_quality": 4,
    },
    {
        "name": "production_800x640_fov75_bias1",
        "resolution": (800, 640),
        "fov": 75.0,
        "exposure_bias": 1.0,
        "aa_quality": 4,
    },
    {
        "name": "wide_800x640_fov90_bias0_5",
        "resolution": (800, 640),
        "fov": 90.0,
        "exposure_bias": 0.5,
        "aa_quality": 4,
    },
    {
        "name": "balanced_960x640_fov85_bias0_5",
        "resolution": (960, 640),
        "fov": 85.0,
        "exposure_bias": 0.5,
        "aa_quality": 4,
    },
    {
        "name": "balanced_960x640_fov85_bias1",
        "resolution": (960, 640),
        "fov": 85.0,
        "exposure_bias": 1.0,
        "aa_quality": 4,
    },
    {
        "name": "balanced_960x640_fov85_bias1_5",
        "resolution": (960, 640),
        "fov": 85.0,
        "exposure_bias": 1.5,
        "aa_quality": 4,
    },
    {
        "name": "balanced_960x640_fov85_bias2",
        "resolution": (960, 640),
        "fov": 85.0,
        "exposure_bias": 2.0,
        "aa_quality": 4,
    },
    {
        "name": "clarity_1280x720_fov75_bias0_5",
        "resolution": (1280, 720),
        "fov": 75.0,
        "exposure_bias": 0.5,
        "aa_quality": 4,
    },
    {
        "name": "baseline_720x640_fov100",
        "resolution": (720, 640),
        "fov": 100.0,
        "exposure_bias": 2.0,
        "aa_quality": 4,
    },
    {
        "name": "zoom_only_720x640_fov75",
        "resolution": (720, 640),
        "fov": 75.0,
        "exposure_bias": 2.0,
        "aa_quality": 4,
    },
    {
        "name": "zoom_800x640_fov75",
        "resolution": (800, 640),
        "fov": 75.0,
        "exposure_bias": 2.0,
        "aa_quality": 4,
    },
    {
        "name": "balanced_960x640_fov75",
        "resolution": (960, 640),
        "fov": 75.0,
        "exposure_bias": 2.0,
        "aa_quality": 4,
    },
    {
        "name": "balanced_pitch_m15_960x640_fov75",
        "resolution": (960, 640),
        "fov": 75.0,
        "camera_pitch": -15.0,
        "exposure_bias": 2.0,
        "aa_quality": 4,
    },
    {
        "name": "balanced_pitch_m10_960x640_fov75",
        "resolution": (960, 640),
        "fov": 75.0,
        "camera_pitch": -10.0,
        "exposure_bias": 2.0,
        "aa_quality": 4,
    },
    {
        "name": "balanced_pitch_m15_960x640_fov80",
        "resolution": (960, 640),
        "fov": 80.0,
        "camera_pitch": -15.0,
        "exposure_bias": 2.0,
        "aa_quality": 4,
    },
    {
        "name": "balanced_bright_960x640_fov75",
        "resolution": (960, 640),
        "fov": 75.0,
        "exposure_bias": 3.0,
        "aa_quality": 4,
    },
    {
        "name": "balanced_960x640_fov80",
        "resolution": (960, 640),
        "fov": 80.0,
        "exposure_bias": 2.0,
        "aa_quality": 4,
    },
    {
        "name": "balanced_low_aa_960x640_fov75",
        "resolution": (960, 640),
        "fov": 75.0,
        "exposure_bias": 2.0,
        "aa_quality": 2,
    },
    {
        "name": "clarity_1280x720_fov70",
        "resolution": (1280, 720),
        "fov": 70.0,
        "exposure_bias": 2.0,
        "aa_quality": 4,
    },
    {
        "name": "clarity_bright_1280x720_fov70",
        "resolution": (1280, 720),
        "fov": 70.0,
        "exposure_bias": 3.0,
        "aa_quality": 4,
    },
)

MASK_COLORS_RGB = (
    (229, 35, 35),
    (35, 229, 35),
    (35, 35, 229),
    (229, 229, 35),
    (229, 35, 229),
    (35, 229, 229),
    (245, 125, 35),
    (125, 35, 245),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", default="data/map2_12roads/tasks.json")
    parser.add_argument("--task-number", type=int, default=5)
    parser.add_argument("--agent-config", default="data/agents_qwen3vl8b.json")
    parser.add_argument("--unrealcv-host", default="127.0.0.1")
    parser.add_argument("--unrealcv-port", type=int, default=9001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--difficulty",
        default="level1",
        choices=("level0", "level1", "level2", "level3", "level4"),
        help=(
            "Use level1 by default: keep the full rendered city but do not "
            "activate movable or falling-object physics during a visibility test."
        ),
    )
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--timed-frames", type=int, default=5)
    parser.add_argument("--entry-offset-cm", type=float, default=190.0)
    parser.add_argument(
        "--pedestrian-light-scale",
        type=float,
        default=None,
        help=(
            "Override the resolved production pedestrian-signal actor scale "
            "for a controlled visibility comparison."
        ),
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Run only the named profile; may be repeated.",
    )
    parser.add_argument(
        "--execute-waypoint",
        type=int,
        choices=range(1, 8),
        default=None,
        help=(
            "After capture, execute one numbered waypoint and record its exact "
            "selected, commanded, and reached world positions."
        ),
    )
    parser.add_argument(
        "--multiview-example",
        action="store_true",
        help="Capture left/center/right 100-degree views and a human-inspection sheet.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _crosswalk_entry_pose(
    route_points: list[list[float]], task_edges: list[dict], offset_cm: float
):
    if len(route_points) != len(task_edges) + 1:
        raise RuntimeError(
            "Route points no longer correspond one-to-one with the ordered task edges"
        )
    crosswalk_indexes = [
        index for index, edge in enumerate(task_edges) if edge.get("type") == "crosswalk"
    ]
    if not crosswalk_indexes:
        raise RuntimeError("Expected an ordered route containing a crosswalk edge")
    crosswalk_index = crosswalk_indexes[0]
    crosswalk_start = Vector(*route_points[crosswalk_index])
    crosswalk_end = Vector(*route_points[crosswalk_index + 1])
    if crosswalk_index > 0:
        incoming_direction = (
            crosswalk_start - Vector(*route_points[crosswalk_index - 1])
        ).normalize()
    else:
        incoming_direction = (crosswalk_end - crosswalk_start).normalize()
    viewing_direction = (crosswalk_end - crosswalk_start).normalize()
    position = crosswalk_start - incoming_direction * float(offset_cm)
    return position, viewing_direction, crosswalk_start, crosswalk_end


def _route_pedestrian_signal_heads(signal_group: list, crosswalk) -> list:
    """Return pedestrian heads physically adjacent to the route zebra.

    Dedicated heads carry the native crosswalk ID and sit immediately outside
    its two endpoints.  Prefer those exact heads and retain a nearest-endpoint
    fallback only for recordings produced by an older map generator.
    """
    exact_heads = [
        signal
        for signal in signal_group
        if (
            getattr(signal, "type", None) == "pedestrian"
            and getattr(signal, "crosswalk_id", None) == crosswalk.id
        )
    ]
    if exact_heads:
        return exact_heads

    pedestrian_heads = [
        signal
        for signal in signal_group
        if getattr(signal, "type", None) == "pedestrian"
    ]

    def endpoint_distance(signal) -> float:
        return min(
            signal.position.distance(crosswalk.start),
            signal.position.distance(crosswalk.end),
        )

    pedestrian_heads.sort(key=endpoint_distance)
    nearby = [
        signal for signal in pedestrian_heads if endpoint_distance(signal) <= 500.0
    ]
    return nearby or pedestrian_heads[:1]


def _mask_metrics(mask_bgr: np.ndarray, signals: list[dict]) -> list[dict]:
    rows = []
    pixels = mask_bgr.astype(np.int16)
    for signal in signals:
        target_bgr = np.asarray(signal["color_rgb"][::-1], dtype=np.int16)
        selected = np.max(np.abs(pixels - target_bgr), axis=2) <= 8
        ys, xs = np.nonzero(selected)
        if len(xs):
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            width = bbox[2] - bbox[0] + 1
            height = bbox[3] - bbox[1] + 1
        else:
            bbox = None
            width = height = 0
        rows.append(
            {
                "signal_id": signal["signal_id"],
                "actor_name": signal["actor_name"],
                "visible_pixels": int(len(xs)),
                "bbox": bbox,
                "bbox_width_px": int(width),
                "bbox_height_px": int(height),
            }
        )
    return rows


def _state_change_metrics(
    walk_frame: np.ndarray, stop_frame: np.ndarray
) -> dict | None:
    """Locate rendered signal pixels from the WALK/STOP frame difference.

    The traffic-signal actor's UnrealCV object mask is empty in this packaged
    scene because the visible lamp is a child mesh.  State differencing is a
    better fail-closed measurement: the scene is held on one synchronous tick
    while only the pedestrian heads are mutated.
    """
    difference = np.mean(
        np.abs(walk_frame.astype(np.int16) - stop_frame.astype(np.int16)), axis=2
    )
    changed = (difference >= 25.0).astype(np.uint8)
    grouped = cv2.dilate(changed, np.ones((3, 3), np.uint8), iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    candidates = []
    height, width = changed.shape
    for component_id in range(1, count):
        x, y, box_width, box_height, _ = stats[component_id]
        if x <= 0 or y <= 0 or x + box_width >= width or y + box_height >= height:
            continue
        selected = labels == component_id
        original = selected & (changed > 0)
        changed_pixels = int(np.count_nonzero(original))
        if changed_pixels < 2:
            continue
        mean_difference = float(np.mean(difference[original]))
        score = changed_pixels * mean_difference
        if score < 500.0:
            continue
        candidates.append(
            {
                "changed_pixels": changed_pixels,
                "mean_abs_difference": mean_difference,
                "score": score,
                "bbox": [
                    int(x),
                    int(y),
                    int(x + box_width - 1),
                    int(y + box_height - 1),
                ],
                "bbox_width_px": int(box_width),
                "bbox_height_px": int(box_height),
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])


def _color_components(frame: np.ndarray, state: str) -> list[dict]:
    """Return compact, saturated emission candidates in the upper scene."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    if state == "walk":
        selected = (hue >= 35) & (hue <= 95)
    else:
        selected = (hue <= 24) | (hue >= 172)
    selected &= saturation >= 125
    selected &= value >= 75
    selected[int(frame.shape[0] * 0.60) :, :] = False
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        selected.astype(np.uint8), 8
    )
    components = []
    for component_id in range(1, count):
        x, y, width, height, area = stats[component_id]
        if not 2 <= area <= 1200:
            continue
        components.append(
            {
                "pixels": int(area),
                "bbox": [int(x), int(y), int(x + width - 1), int(y + height - 1)],
                "bbox_width_px": int(width),
                "bbox_height_px": int(height),
                "center": [
                    float(centroids[component_id][0]),
                    float(centroids[component_id][1]),
                ],
            }
        )
    return components


def _match_rendered_light_states(
    walk_frame: np.ndarray, stop_frame: np.ndarray
) -> dict | None:
    """Match WALK green and STOP orange emissions at one screen location."""
    walk_candidates = _color_components(walk_frame, "walk")
    stop_candidates = _color_components(stop_frame, "dont_walk")
    maximum_distance = max(18.0, walk_frame.shape[1] * 0.035)
    matches = []
    for walk in walk_candidates:
        for stop in stop_candidates:
            distance = math.dist(walk["center"], stop["center"])
            if distance > maximum_distance:
                continue
            pixels = min(walk["pixels"], stop["pixels"])
            compactness = 1.0 / max(
                1.0,
                walk["bbox_width_px"] * walk["bbox_height_px"]
                + stop["bbox_width_px"] * stop["bbox_height_px"],
            )
            matches.append(
                {
                    "walk": walk,
                    "dont_walk": stop,
                    "center_distance_px": float(distance),
                    "matched_pixels": int(pixels),
                    "score": float(pixels * (1.0 + compactness) / (1.0 + distance)),
                }
            )
    if not matches:
        return None
    return max(matches, key=lambda item: item["score"])


def _save_crop(frame: np.ndarray, bbox: list[int] | None, path: Path) -> None:
    if not bbox:
        return
    x0, y0, x1, y1 = bbox
    pad = max(12, int(max(x1 - x0 + 1, y1 - y0 + 1) * 1.5))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(frame.shape[1] - 1, x1 + pad)
    y1 = min(frame.shape[0] - 1, y1 + pad)
    crop = frame[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size:
        scale = max(1, min(8, 480 // max(crop.shape[:2])))
        crop = cv2.resize(
            crop,
            (crop.shape[1] * scale, crop.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imwrite(str(path), crop)


def _contact_sheet(frames: list[tuple[str, np.ndarray]], output: Path) -> None:
    cards = []
    for label, frame in frames:
        width = 420
        height = int(round(frame.shape[0] * width / frame.shape[1]))
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        card = np.full((height + 42, width, 3), 245, dtype=np.uint8)
        card[42:] = resized
        cv2.putText(
            card,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cards.append(card)
    if not cards:
        return
    max_height = max(card.shape[0] for card in cards)
    padded = []
    for card in cards:
        if card.shape[0] < max_height:
            pad = np.full((max_height - card.shape[0], card.shape[1], 3), 245, np.uint8)
            card = np.vstack((card, pad))
        padded.append(card)
    cv2.imwrite(str(output), np.hstack(padded))


def main() -> None:
    args = parse_args()
    if args.task_number < 1:
        raise ValueError("--task-number is 1-based")
    output_dir = Path(
        args.output_dir
        or "results/pedestrian_light_visibility_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    unrealcv = RTUnrealCV(port=args.unrealcv_port, ip=args.unrealcv_host)
    communicator = RTCommunicator(unrealcv)
    manager = WorldManager(
        communicator,
        agent_path=args.agent_config,
        task_file_path=args.task_file,
        seed=args.seed,
        use_tick=True,
        realtime_thinking=True,
        disable_background_agents=False,
        red_light_conflict_vehicle_enabled=False,
    )
    results = {
        "task_file": args.task_file,
        "task_number": args.task_number,
        "full_scene": True,
        "profiles": [],
    }
    sheet_frames = []
    profiles = list(PROFILES)
    if args.profile:
        requested = set(args.profile)
        known = {profile["name"] for profile in PROFILES}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown --profile values: {unknown}; expected {sorted(known)}")
        profiles = [profile for profile in PROFILES if profile["name"] in requested]

    try:
        manager.difficulty = args.difficulty
        manager.difficulty_config = manager._get_difficulty_config()
        manager.scenario_data = manager.all_scenarios[args.task_number - 1]
        manager.current_task_id = manager.scenario_data.get("task_id")
        manager._initialize_world()
        if args.pedestrian_light_scale is not None:
            if (
                not math.isfinite(args.pedestrian_light_scale)
                or args.pedestrian_light_scale <= 0
            ):
                raise ValueError("--pedestrian-light-scale must be finite and positive")
            manager.pedestrian_light_scale = args.pedestrian_light_scale
        manager.render_auto_exposure = True
        manager.render_exposure_offset = 2.0
        manager._apply_render_settings()

        task = manager.scenario_data["task"]
        route_points = reconstruct_route_points_from_task_edges(task)
        position, direction, crosswalk_start, crosswalk_end = _crosswalk_entry_pose(
            route_points, task.get("edges", []), args.entry_offset_cm
        )
        shortest_path = [Vector(x, y) for x, y in route_points[1:]]
        route_crosswalks = manager.traffic_controller.get_route_crosswalks(route_points)
        if not route_crosswalks:
            raise RuntimeError("The selected task has no matched route crosswalk")
        groups = manager.traffic_controller.get_crosswalk_signal_groups(route_crosswalks)
        signal_group = groups.get(route_crosswalks[0].id, [])
        pedestrian_signals = _route_pedestrian_signal_heads(
            signal_group, route_crosswalks[0]
        )
        if not pedestrian_signals:
            raise RuntimeError("The route crosswalk has no pedestrian signal heads")

        if manager.signal_traffic_enabled:
            manager._stage_signal_traffic_at_crosswalk(route_crosswalks[0])
        manager._remove_disabled_background_hazards(route_points)
        manager._spawn_route_crosswalk_markings(route_crosswalks)
        traffic_config = {
            "green_light_duration": manager.traffic_controller.config[
                "traffic.traffic_signal.green_light_duration"
            ],
            "yellow_light_duration": manager.traffic_controller.config[
                "traffic.traffic_signal.yellow_light_duration"
            ],
            "pedestrian_green_light_duration": manager.traffic_controller.config[
                "traffic.traffic_signal.pedestrian_green_light_duration"
            ],
            "pedestrian_phase_duration": manager.traffic_controller.config[
                "traffic.traffic_signal.pedestrian_phase_duration"
            ],
        }
        agent = RTAgent(
            position=position,
            direction=direction,
            destination=Vector(*route_points[-1]),
            shortest_path=shortest_path,
            required_time=task.get("required_time", 1000),
            communicator=communicator,
            llm=None,
            use_tick=True,
            realtime_thinking=True,
            task_edges=task.get("edges", []),
            traffic_signals=manager.traffic_controller.traffic_signals,
            traffic_intersections=manager.traffic_controller.intersections,
            traffic_signal_config=traffic_config,
            route_crosswalks=route_crosswalks,
            crosswalk_signal_groups=groups,
            crosswalk_intersection_names={},
            static_obstacles=manager.static_obstacles,
            terminate_on_touched_road=False,
        )
        manager.agent = agent
        manager.spawn_agents()
        # BaseAgent initializes its scalar yaw to zero even when constructed
        # with a direction vector. Normal rollouts replace that yaw while
        # synchronizing the spawned actor, but this fixed-pose diagnostic must
        # explicitly retain the authored crosswalk-facing heading. Keep the
        # UE actor, route geometry, free camera, and waypoint projector on the
        # same accepted pose before comparing FOV profiles.
        capture_yaw = math.degrees(math.atan2(direction.y, direction.x))
        actor_location = unrealcv.get_location(agent.name)
        agent.position = Vector(position.x, position.y)
        agent.direction = capture_yaw
        unrealcv.set_location(
            (float(position.x), float(position.y), float(actor_location[2])),
            agent.name,
        )
        unrealcv.set_orientation((0.0, capture_yaw, 0.0), agent.name)
        agent._sync_first_person_camera()

        signal_records = []
        for index, signal in enumerate(pedestrian_signals[: len(MASK_COLORS_RGB)]):
            actor_name = communicator.get_traffic_signal_name(signal.id)
            color = MASK_COLORS_RGB[index]
            unrealcv.set_color(actor_name, color)
            signal_records.append(
                {
                    "signal_id": signal.id,
                    "actor_name": actor_name,
                    "color_rgb": color,
                    "position": [signal.position.x, signal.position.y],
                }
            )

        results.update(
            {
                "task_id": manager.current_task_id,
                "difficulty": manager.difficulty,
                "traffic_light_scale": manager.traffic_light_scale,
                "pedestrian_light_scale": manager.pedestrian_light_scale,
                "route_points": route_points,
                "crosswalk_id": route_crosswalks[0].id,
                "capture_position": [position.x, position.y],
                "capture_direction": [direction.x, direction.y],
                "crosswalk_start": [crosswalk_start.x, crosswalk_start.y],
                "crosswalk_end": [crosswalk_end.x, crosswalk_end.y],
                "scene_assets": {
                    "pedestrians": len(manager.pedestrians),
                    "irregular_npcs": len(manager.irregular_pedestrians),
                    "movable_obstacles": len(manager.movable_obstacle_ids),
                    "falling_objects": len(getattr(manager, "falling_object_ids", [])),
                    "signal_traffic_vehicles": manager.signal_traffic_vehicle_count,
                    "signal_traffic_pedestrians": manager.signal_traffic_pedestrian_count,
                },
                "route_pedestrian_signals": signal_records,
                "camera_location": list(agent.camera_location),
                "camera_rotation": list(agent.camera_rotation),
            }
        )

        base_camera_rotation = tuple(agent.camera_rotation)
        for profile in profiles:
            profile_dir = output_dir / profile["name"]
            profile_dir.mkdir(parents=True, exist_ok=True)
            width, height = profile["resolution"]
            unrealcv.set_camera_resolution(agent.camera_id, (width, height))
            unrealcv.set_camera_fov(agent.camera_id, profile["fov"])
            camera_rotation = base_camera_rotation
            if "camera_pitch" in profile:
                camera_rotation = (
                    float(profile["camera_pitch"]),
                    float(base_camera_rotation[1]),
                    float(base_camera_rotation[2]),
                )
                unrealcv.set_camera_rotation(agent.camera_id, camera_rotation)
                agent.camera_rotation = camera_rotation
            agent.fov = profile["fov"]
            commands = (
                "vrun r.DefaultFeature.AutoExposure 1",
                f"vrun r.DefaultFeature.AutoExposure.Bias {profile['exposure_bias']:g}",
                f"vrun r.PostProcessAAQuality {profile['aa_quality']}",
                "vrun r.ScreenPercentage 100",
            )
            responses = {command: unrealcv.client.request(command) for command in commands}
            for _ in range(max(0, args.warmup_frames)):
                unrealcv.get_image(agent.camera_id, "lit", "direct")

            timings = []
            final_frames = {}
            mask_metrics = None
            for state_name, walk in (("walk", True), ("dont_walk", False)):
                for signal in pedestrian_signals:
                    actor_name = communicator.get_traffic_signal_name(signal.id)
                    if walk:
                        unrealcv.tl_set_pedestrian_walk(actor_name)
                    else:
                        unrealcv.tl_set_pedestrian_stop(actor_name)
                unrealcv.get_image(agent.camera_id, "lit", "direct")
                frame_times = []
                frame = None
                for _ in range(max(1, args.timed_frames)):
                    started = time.perf_counter()
                    frame = unrealcv.get_image(agent.camera_id, "lit", "direct")
                    frame_times.append(time.perf_counter() - started)
                timings.extend(frame_times)
                final_frames[state_name] = frame
                cv2.imwrite(str(profile_dir / f"{state_name}_raw.png"), frame)
                annotated = annotate_image(
                    frame,
                    agent._find_waypoints(),
                    agent.camera_location,
                    agent.camera_rotation,
                    agent.fov,
                    route_crosswalks=route_crosswalks,
                )
                annotated.save(profile_dir / f"{state_name}_model_input.png")

                if mask_metrics is None:
                    mask = unrealcv.get_image(agent.camera_id, "object_mask", "direct")
                    cv2.imwrite(str(profile_dir / "object_mask.png"), mask)
                    mask_metrics = _mask_metrics(mask, signal_records)
                visible = max(mask_metrics, key=lambda item: item["visible_pixels"])
                _save_crop(
                    frame,
                    visible["bbox"],
                    profile_dir / f"{state_name}_best_signal_crop.png",
                )

            best = max(mask_metrics, key=lambda item: item["visible_pixels"])
            rendered_state_change = _state_change_metrics(
                final_frames["walk"], final_frames["dont_walk"]
            )
            rendered_light_match = _match_rendered_light_states(
                final_frames["walk"], final_frames["dont_walk"]
            )
            if rendered_state_change is not None:
                for state_name, frame in final_frames.items():
                    _save_crop(
                        frame,
                        rendered_state_change["bbox"],
                        profile_dir / f"{state_name}_state_change_crop.png",
                    )
            if rendered_light_match is not None:
                _save_crop(
                    final_frames["walk"],
                    rendered_light_match["walk"]["bbox"],
                    profile_dir / "walk_matched_light_crop.png",
                )
                _save_crop(
                    final_frames["dont_walk"],
                    rendered_light_match["dont_walk"]["bbox"],
                    profile_dir / "dont_walk_matched_light_crop.png",
                )
            profile_result = {
                **profile,
                "resolution": list(profile["resolution"]),
                "camera_rotation": list(camera_rotation),
                "render_commands": responses,
                "capture_seconds_median": float(statistics.median(timings)),
                "capture_seconds_p95": float(np.percentile(timings, 95)),
                "visible_signal_metrics": mask_metrics,
                "best_visible_signal": best,
                "rendered_state_change": rendered_state_change,
                "rendered_light_match": rendered_light_match,
                "scene_luminance": {
                    state_name: {
                        "mean": round(
                            float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()),
                            3,
                        ),
                        "p01": int(
                            np.percentile(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1
                            )
                        ),
                        "p50": int(
                            np.percentile(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50
                            )
                        ),
                        "p99": int(
                            np.percentile(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 99
                            )
                        ),
                        "clipped_black_fraction": round(
                            float(
                                (
                                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                    <= 3
                                ).mean()
                            ),
                            6,
                        ),
                        "clipped_white_fraction": round(
                            float(
                                (
                                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                    >= 252
                                ).mean()
                            ),
                            6,
                        ),
                    }
                    for state_name, frame in final_frames.items()
                },
                "walk_vs_stop_mean_abs_difference": (
                    rendered_state_change["mean_abs_difference"]
                    if rendered_state_change is not None
                    else None
                ),
            }
            results["profiles"].append(profile_result)
            sheet_frames.append((profile["name"], final_frames["walk"]))

        baseline_time = results["profiles"][0]["capture_seconds_median"]
        baseline_pixels = max(
            1,
            int(
                (results["profiles"][0].get("rendered_light_match") or {}).get(
                    "matched_pixels", 0
                )
            ),
        )
        for profile in results["profiles"]:
            profile["relative_render_time"] = (
                profile["capture_seconds_median"] / baseline_time
            )
            profile["relative_visible_signal_pixels"] = (
                int((profile.get("rendered_light_match") or {}).get("matched_pixels", 0))
                / baseline_pixels
            )
            profile["visibility_per_render_cost"] = (
                profile["relative_visible_signal_pixels"]
                / max(profile["relative_render_time"], 1e-6)
            )
            profile["relative_state_difference"] = 1.0
            profile["readability_per_render_cost"] = profile[
                "visibility_per_render_cost"
            ]
        eligible = [
            profile
            for profile in results["profiles"]
            if profile.get("rendered_light_match") is not None
        ]
        if eligible:
            results["recommended_profile"] = max(
                eligible, key=lambda item: item["readability_per_render_cost"]
            )["name"]

        if args.multiview_example:
            multiview_dir = output_dir / "multiview_fov100"
            multiview_dir.mkdir(parents=True, exist_ok=True)
            width, height = (720, 640)
            agent.fov = 100.0
            agent.first_person_camera_pitch_deg = -25.0
            agent.camera_resolution = (width, height)
            unrealcv.set_camera_resolution(agent.camera_id, (width, height))
            unrealcv.set_camera_fov(agent.camera_id, agent.fov)
            agent._sync_first_person_camera()
            center_ue_yaw = -float(agent.camera_rotation[1])
            candidate_waypoints = agent._find_waypoints()
            center_projection_rotation = (
                agent.first_person_camera_pitch_deg,
                -center_ue_yaw,
                0.0,
            )
            marker_pixels = project_waypoints_to_image(
                candidate_waypoints,
                agent.camera_location,
                center_projection_rotation,
                agent.fov,
                width,
                height,
            )
            if len(marker_pixels) != 7 or any(
                pixel is None for pixel in marker_pixels
            ):
                raise RuntimeError(
                    "Multiview center action image must contain all seven markers"
                )
            views = []
            view_records = []
            for label, yaw_offset in (
                ("left_context", -70.0),
                ("center_action", 0.0),
                ("right_context", 70.0),
            ):
                ue_rotation = (
                    agent.first_person_camera_pitch_deg,
                    center_ue_yaw + yaw_offset,
                    0.0,
                )
                unrealcv.set_camera_rotation(agent.camera_id, ue_rotation)
                frame = unrealcv.get_image(agent.camera_id, "lit", "direct")
                cv2.imwrite(str(multiview_dir / f"{label}.png"), frame)
                display_frame = frame
                if yaw_offset == 0.0:
                    marked = annotate_image(
                        frame,
                        candidate_waypoints,
                        agent.camera_location,
                        center_projection_rotation,
                        agent.fov,
                        route_crosswalks=route_crosswalks,
                    )
                    marked.save(multiview_dir / "center_action_marked.png")
                    display_frame = cv2.cvtColor(
                        np.asarray(marked),
                        cv2.COLOR_RGB2BGR,
                    )
                views.append((f"{label}: yaw {yaw_offset:+.0f} deg", display_frame))
                view_records.append(
                    {
                        "name": label,
                        "relative_yaw_deg": yaw_offset,
                        "ue_camera_rotation": list(ue_rotation),
                        "annotated": yaw_offset == 0.0,
                    }
                )
            _contact_sheet(
                views,
                multiview_dir / "multiview_contact_sheet.png",
            )
            agent._sync_first_person_camera()
            results["multiview_example"] = {
                "directory": str(multiview_dir),
                "resolution_per_view": [width, height],
                "horizontal_fov_deg": agent.fov,
                "center_markers": len(candidate_waypoints),
                "center_marker_pixels": [list(pixel) for pixel in marker_pixels],
                "views": view_records,
                "contact_sheet_role": "human inspection only",
            }

        if args.execute_waypoint is not None:
            # Execute from the task's true initial, route-aligned pose. The
            # visibility profiles above deliberately face the first crosswalk,
            # which is useful for signal comparison but is not the active-route
            # heading at step zero.
            step_position = Vector(*route_points[0])
            step_direction = (
                Vector(*route_points[1]) - step_position
            ).normalize()
            step_yaw = math.degrees(
                math.atan2(step_direction.y, step_direction.x)
            )
            actor_location = unrealcv.get_location(agent.name)
            agent.position = Vector(step_position.x, step_position.y)
            agent.direction = step_yaw
            agent.shortest_path = [
                Vector(x, y) for x, y in route_points[1:]
            ]
            agent.original_shortest_path = [
                Vector(point.x, point.y) for point in agent.shortest_path
            ]
            agent.route_polyline = [
                Vector(step_position.x, step_position.y),
                *[
                    Vector(point.x, point.y)
                    for point in agent.shortest_path
                ],
            ]
            agent.current_destination = agent.shortest_path[0]
            unrealcv.set_location(
                (
                    float(step_position.x),
                    float(step_position.y),
                    float(actor_location[2]),
                ),
                agent.name,
            )
            unrealcv.set_orientation((0.0, step_yaw, 0.0), agent.name)

            width, height = (720, 640)
            agent.fov = 100.0
            agent.first_person_camera_pitch_deg = -25.0
            unrealcv.set_camera_resolution(agent.camera_id, (width, height))
            unrealcv.set_camera_fov(agent.camera_id, agent.fov)
            agent.camera_resolution = (width, height)
            agent._sync_first_person_camera()
            before_frame = unrealcv.get_image(agent.camera_id, "lit", "direct")
            candidate_waypoints = agent._find_waypoints()
            marker_pixels = project_waypoints_to_image(
                candidate_waypoints,
                agent.camera_location,
                agent.camera_rotation,
                agent.fov,
                width,
                height,
            )
            if len(marker_pixels) != 7 or any(
                pixel is None for pixel in marker_pixels
            ):
                raise RuntimeError(
                    "One-step check requires all seven waypoint markers in frame: "
                    f"{marker_pixels}"
                )
            before_annotated = annotate_image(
                before_frame,
                candidate_waypoints,
                agent.camera_location,
                agent.camera_rotation,
                agent.fov,
                route_crosswalks=route_crosswalks,
            )
            before_annotated.save(output_dir / "one_step_before_policy_input.png")

            pre_action_position = Vector(agent.position.x, agent.position.y)
            action = RTActionSpace(
                action_type=MOVE_TO,
                action_param=str(args.execute_waypoint),
            )
            action_success, action_record = agent.take_action(
                action,
                candidate_waypoints,
            )
            agent.sync_ue()
            after_frame = unrealcv.get_image(agent.camera_id, "lit", "direct")
            cv2.imwrite(str(output_dir / "one_step_after_raw.png"), after_frame)
            execution = agent._waypoint_execution_record(
                action,
                pre_action_position,
            )
            selected = candidate_waypoints[args.execute_waypoint - 1]
            results["one_step_check"] = {
                "action_success": bool(action_success),
                "action_record": action_record,
                "selected_waypoint_index": args.execute_waypoint,
                "candidate_waypoints_cm": [
                    [float(point.x), float(point.y)]
                    for point in candidate_waypoints
                ],
                "marker_pixels": [list(pixel) for pixel in marker_pixels],
                "all_seven_markers_visible": True,
                "selected_marker_pixel": list(
                    marker_pixels[args.execute_waypoint - 1]
                ),
                "selected_world_target_cm": [
                    float(selected.x),
                    float(selected.y),
                ],
                "execution": execution,
            }

        _contact_sheet(sheet_frames, output_dir / "walk_contact_sheet.png")
        (output_dir / "benchmark.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
    finally:
        manager.cleanup()
        unrealcv.disconnect()


if __name__ == "__main__":
    main()
