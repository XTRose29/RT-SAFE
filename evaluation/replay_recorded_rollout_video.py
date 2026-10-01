#!/usr/bin/env python3
"""Replay a recorded rollout in UE and render inference/action phase video.

This tool never constructs an LLM client and never calls a model API.  It
rebuilds the saved task from its seed, replays the recorded per-step actions,
and advances UE for the recorded inference duration only in realtime mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = REPO_ROOT / ".py312deps"
if LOCAL_DEPS.is_dir():
    sys.path.append(str(LOCAL_DEPS))

from PIL import Image, ImageDraw, ImageFont

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SIMWORLD_ROOT = REPO_ROOT / "SimWorld"
if str(SIMWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMWORLD_ROOT))
simworld_package = types.ModuleType("simworld")
simworld_package.__path__ = [str(SIMWORLD_ROOT / "simworld")]
sys.modules.setdefault("simworld", simworld_package)

from base.rt_action_space import RTActionSpace
from base.rt_agent import RTAgent
from base.rt_communicator import RTCommunicator
from evaluation.replay_safe_trajectory_video import connect_unrealcv
from evaluation.run_qwen3vl8b_all_maps import UEServer
from manager.world_manager import SLOMO, TIME_ALPHA, TIME_BETA, WorldManager
from simworld.utils.vector import Vector


FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
CANVAS_SIZE = (1920, 1080)
REPORT_SCHEMA = "recorded_rollout_ue_replay_v1"
COLLISION_KEYS = ("human", "object", "building", "vehicle")


class NoModelReplayLLM:
    """Sentinel that makes accidental model access fail closed."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"Metadata-only replay attempted to access LLM attribute {name!r}"
        )


class ReplayWorldManager(WorldManager):
    """World manager variant that never constructs an RTLLM client."""

    def _create_llm_from_config(self, config: dict[str, Any]) -> NoModelReplayLLM:
        return NoModelReplayLLM()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def load_manifests(steps_dir: Path) -> list[dict[str, Any]]:
    manifests = []
    for path in steps_dir.glob("step_*/*_manifest.json"):
        item = read_json(path)
        item["_path"] = str(path)
        manifests.append(item)
    manifests.sort(
        key=lambda item: (
            int(item.get("step") or 0),
            int(item.get("decision_index") or 0),
        )
    )
    if not manifests:
        raise RuntimeError(f"No step manifests found below {steps_dir}")
    return manifests


def point(value: Any) -> Vector:
    if isinstance(value, dict):
        return Vector(float(value["x"]), float(value["y"]))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return Vector(float(value[0]), float(value[1]))
    raise RuntimeError(f"Expected an x/y point, got {value!r}")


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = FONT_BOLD_PATH if bold else FONT_PATH
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_pixels(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text or "").splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textlength(candidate, font=text_font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def parsed_reasoning(manifest: dict[str, Any]) -> str:
    output = manifest.get("model_output") or {}
    parsed = output.get("parsed_action") or {}
    reasoning = str(parsed.get("reasoning") or "").strip()
    if reasoning:
        return reasoning
    for line in str(output.get("raw_response") or "").splitlines():
        if line.lower().startswith("reasoning:"):
            return line.split(":", 1)[1].strip()
    return "not recorded"


def action_text(manifest: dict[str, Any]) -> str:
    parsed = ((manifest.get("model_output") or {}).get("parsed_action") or {})
    action_type = str(parsed.get("type") or "none").upper()
    action_param = str(parsed.get("param") or "-")
    return f"{action_type} {action_param}"


def collision_event(manifest: dict[str, Any]) -> tuple[int, str]:
    direct = manifest.get("collision_details") or {}
    passive = manifest.get("passive_collision_details") or {}
    counts: dict[str, int] = {}
    for key in COLLISION_KEYS:
        counts[key] = int(direct.get(key, 0) or 0) + int(passive.get(key, 0) or 0)
    added = sum(counts.values())
    kinds = [key.upper() for key, count in counts.items() if count]
    return added, " + ".join(kinds) if kinds else ""


def phase_durations(
    manifest: dict[str, Any],
    *,
    realtime: bool,
    inference_timing: str = "simulation",
) -> tuple[float, float]:
    timing = manifest.get("timing") or {}
    metrics = manifest.get("metrics") or {}
    if realtime:
        if inference_timing == "model-wall":
            inference = float(
                timing.get("model_inference_wall_seconds")
                or metrics.get("response_time")
                or timing.get("simulation_thinking_latency_seconds")
                or 0.0
            )
        else:
            inference = float(
                timing.get("simulation_thinking_latency_seconds")
                or timing.get("model_inference_wall_seconds")
                or metrics.get("response_time")
                or 0.0
            )
    else:
        # Static inference is visible on the presentation clock while UE stays
        # paused, making the control condition directly comparable on screen.
        inference = float(
            timing.get("model_inference_wall_seconds")
            or metrics.get("response_time")
            or 0.0
        )
    waypoint = manifest.get("waypoint_execution") or {}
    turn = manifest.get("turn_execution") or {}
    parsed = ((manifest.get("model_output") or {}).get("parsed_action") or {})
    action_type = str(parsed.get("type") or "")
    if waypoint.get("move_command_duration_seconds") is not None:
        action = float(waypoint["move_command_duration_seconds"])
    elif action_type == "turn_around" or turn:
        action = 1.0
    elif action_type == "wait":
        action = float(parsed.get("param") or 0.0)
    else:
        action = 0.0
    return max(0.0, inference), max(0.0, action)


def make_action(manifest: dict[str, Any]) -> RTActionSpace | None:
    parsed = ((manifest.get("model_output") or {}).get("parsed_action") or {})
    action = RTActionSpace(
        action_type=parsed.get("type"),
        action_param=str(parsed.get("param")) if parsed.get("param") is not None else None,
        reasoning=parsed.get("reasoning"),
    )
    return action if action.is_valid() else None


def candidate_waypoints(manifest: dict[str, Any]) -> list[Vector]:
    candidates = (manifest.get("observation_geometry") or {}).get(
        "annotated_candidates"
    ) or []
    ordered = sorted(candidates, key=lambda item: int(item.get("index") or 0))
    if len(ordered) != 7:
        raise RuntimeError(
            f"Step {manifest.get('step')} has {len(ordered)} recorded candidates"
        )
    return [point(item["world_position_cm"]) for item in ordered]


def setup_world(
    communicator: RTCommunicator,
    result: dict[str, Any],
    first_manifest: dict[str, Any],
    *,
    task_file: str,
    task_index: int,
    realtime: bool,
) -> WorldManager:
    seed = int(result.get("seed") or 0)
    manager = ReplayWorldManager(
        communicator,
        task_file_path=task_file,
        seed=seed,
        control_mode="replay",
        token_based=False,
        use_tick=True,
        realtime_thinking=realtime,
        record_per_step=False,
        use_action_frames=False,
        prompt_style=str(result.get("prompt_style") or "instructional"),
        pedestrians_enabled=True,
        movable_obstacles_enabled=True,
        irregular_npcs_enabled=True,
        falling_objects_enabled=True,
        max_steps=0,
        traffic_policy=str(result.get("traffic_policy") or "visual_only"),
        static_signal_vehicles=bool(result.get("static_signal_vehicles", False)),
        red_light_conflict_vehicle_enabled=bool(
            result.get("red_light_conflict_vehicle_enabled", True)
        ),
        red_light_conflict_launch_distance_min_cm=float(
            result.get("red_light_conflict_launch_distance_min_cm") or 300.0
        ),
        red_light_conflict_launch_distance_max_cm=float(
            result.get("red_light_conflict_launch_distance_max_cm") or 900.0
        ),
        red_light_conflict_collision_radius_cm=float(
            result.get("red_light_conflict_collision_radius_cm") or 100.0
        ),
        red_light_conflict_vehicle_probability=float(
            result.get("red_light_conflict_vehicle_probability") or 0.0
        ),
        red_light_conflict_nominal_speed_cm_s=float(
            result.get("red_light_conflict_nominal_speed_cm_s") or 450.0
        ),
        conflict_vehicle_release_wait_timeout_s=float(
            result.get("conflict_vehicle_release_wait_timeout_s") or 12.0
        ),
        pedestrian_signal_compliance_probability=float(
            result.get("pedestrian_signal_compliance_probability") or 1.0
        ),
    )
    manager.difficulty = manager._normalize_difficulty_label(
        str(result.get("difficulty") or "easy")
    )
    manager.difficulty_config = manager._get_difficulty_config()
    random.seed(seed + task_index)
    manager.scenario_data = manager.all_scenarios[task_index]
    manager.current_task_id = manager.scenario_data.get("task_id", task_index)
    manager._initialize_world()

    geometry = first_manifest.get("observation_geometry") or {}
    start = point(geometry["agent_position_cm"])
    direction = point(geometry["agent_direction"])
    goal = point(result["final_destination"])
    task_data = manager.scenario_data["task"]
    shortest_path = [
        Vector(x, y)
        for x, y in manager.scenario_data["route_info"]["shortest_path"][1:]
    ]
    manager.agent = RTAgent(
        start,
        direction,
        goal,
        shortest_path,
        task_data.get("required_time", result.get("required_time", 1000)),
        communicator,
        llm=NoModelReplayLLM(),
        token_based=False,
        use_tick=True,
        time_alpha=TIME_ALPHA,
        time_beta=TIME_BETA,
        realtime_thinking=realtime,
        task_edges=task_data.get("edges", []),
        traffic_signals=manager.traffic_controller.traffic_signals,
        static_obstacles=manager.static_obstacles,
        record_per_step=False,
        record_dir=None,
        slomo=SLOMO,
        use_action_frames=False,
        prompt_style=str(result.get("prompt_style") or "instructional"),
        dynamic_obstacles=manager.dynamic_obstacle_metadata,
        traffic_policy=str(result.get("traffic_policy") or "visual_only"),
    )
    manager.spawn_agents()
    return manager


def force_recorded_pose(agent: RTAgent, manifest: dict[str, Any]) -> float:
    geometry = manifest.get("observation_geometry") or {}
    position = point(geometry["agent_position_cm"])
    direction = point(geometry["agent_direction"])
    live = agent.communicator.unrealcv.get_location(agent.name)
    live_position = Vector(float(live[0]), float(live[1]))
    correction = live_position.distance(position)
    route_yaw = math.degrees(math.atan2(direction.y, direction.x))
    ue_yaw = agent._route_yaw_to_ue_yaw(route_yaw)
    orientation = agent.communicator.unrealcv.get_orientation(agent.name)
    agent.communicator.unrealcv.set_location(
        (position.x, position.y, float(live[2])), agent.name
    )
    agent.communicator.unrealcv.set_orientation(
        (float(orientation[0]), ue_yaw, float(orientation[2])), agent.name
    )
    agent.position = Vector(position.x, position.y)
    agent.direction = route_yaw
    agent._sync_first_person_camera()
    return correction


def capture_rgb(agent: RTAgent) -> Image.Image:
    frame = agent.communicator.get_camera_observation(agent.camera_id, "lit")
    if frame is None:
        raise RuntimeError("UE returned no camera frame")
    if hasattr(frame, "shape") and len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = frame[:, :, ::-1]
    return Image.fromarray(frame).convert("RGB")


def install_chase_camera(
    agent: RTAgent,
    *,
    width: int,
    height: int,
    distance_cm: float,
    lateral_cm: float,
    height_cm: float,
    look_ahead_cm: float,
    fov: float,
) -> int:
    """Replace the policy camera with a native free camera following the actor."""
    unrealcv = agent.communicator.unrealcv
    before = str(unrealcv.get_cameras()).split()
    response = unrealcv.spawn_camera()
    if "error" in str(response).lower():
        raise RuntimeError(f"Failed to spawn chase camera: {response!r}")
    after = str(unrealcv.get_cameras()).split()
    if len(after) != len(before) + 1:
        raise RuntimeError(
            "Chase camera spawn did not add exactly one camera: "
            f"before={before!r}, after={after!r}, response={response!r}"
        )
    chase_camera_id = len(after) - 1
    unrealcv.set_camera_resolution(chase_camera_id, (width, height))
    unrealcv.set_camera_fov(chase_camera_id, fov)
    agent.camera_id = chase_camera_id

    def sync_chase_camera(self: RTAgent) -> None:
        location = unrealcv.get_location(self.name)
        direction = getattr(self, "direction", 0.0)
        if hasattr(direction, "x") and hasattr(direction, "y"):
            route_yaw = math.degrees(
                math.atan2(float(direction.y), float(direction.x))
            )
        elif isinstance(direction, (list, tuple)) and len(direction) >= 2:
            route_yaw = math.degrees(
                math.atan2(float(direction[1]), float(direction[0]))
            )
        else:
            route_yaw = float(direction)
        yaw_rad = math.radians(route_yaw)
        forward_x, forward_y = math.cos(yaw_rad), math.sin(yaw_rad)
        right_x, right_y = -forward_y, forward_x
        camera_x = float(location[0]) - forward_x * distance_cm + right_x * lateral_cm
        camera_y = float(location[1]) - forward_y * distance_cm + right_y * lateral_cm
        camera_z = float(location[2]) + height_cm
        target_x = float(location[0]) + forward_x * look_ahead_cm
        target_y = float(location[1]) + forward_y * look_ahead_cm
        target_z = float(location[2]) + 95.0
        horizontal = max(1.0, math.hypot(target_x - camera_x, target_y - camera_y))
        camera_yaw = math.degrees(math.atan2(target_y - camera_y, target_x - camera_x))
        camera_pitch = math.degrees(math.atan2(target_z - camera_z, horizontal))
        unrealcv.set_camera_location(chase_camera_id, (camera_x, camera_y, camera_z))
        unrealcv.set_camera_rotation(chase_camera_id, (camera_pitch, camera_yaw, 0.0))

    agent._sync_first_person_camera = types.MethodType(sync_chase_camera, agent)
    agent._sync_first_person_camera()
    return chase_camera_id


def annotate_frame(
    source: Image.Image,
    manifest: dict[str, Any],
    *,
    model_label: str,
    condition_label: str,
    phase: str,
    phase_elapsed: float,
    phase_duration: float,
    replay_sim_time: float,
    realtime: bool,
    overlay_style: str = "dashboard",
    collision_total: int = 0,
    collision_added: int = 0,
    collision_kind: str = "",
) -> Image.Image:
    if overlay_style == "scene":
        fitted = source.copy().resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(fitted, "RGBA")
        title_font = load_font(34, bold=True)
        body_font = load_font(23, bold=True)
        small_font = load_font(20)
        draw.rounded_rectangle((28, 24, 590, 120), radius=18, fill=(4, 9, 17, 210))
        draw.text((52, 40), model_label, font=title_font, fill=(248, 250, 255))
        draw.text((52, 84), condition_label, font=small_font, fill=(178, 195, 220))
        step = int(manifest.get("step") or 0)
        timer_text = f"t = {replay_sim_time:.1f}s"
        timer_width = int(draw.textlength(timer_text, font=body_font))
        draw.rounded_rectangle(
            (CANVAS_SIZE[0] - timer_width - 76, 30, CANVAS_SIZE[0] - 28, 82),
            radius=14,
            fill=(4, 9, 17, 210),
        )
        draw.text(
            (CANVAS_SIZE[0] - timer_width - 52, 43),
            timer_text,
            font=body_font,
            fill=(239, 244, 252),
        )
        draw.rounded_rectangle((28, 990, 820, 1056), radius=16, fill=(4, 9, 17, 220))
        draw.text(
            (52, 1008),
            f"DECISION {step + 1:02d}    COLLISIONS {collision_total:02d}    {action_text(manifest)}",
            font=body_font,
            fill=(239, 244, 252),
        )
        if collision_added:
            draw.rectangle((0, 0, 28, CANVAS_SIZE[1]), fill=(218, 48, 70, 190))
            draw.rectangle((CANVAS_SIZE[0] - 28, 0, CANVAS_SIZE[0], CANVAS_SIZE[1]), fill=(218, 48, 70, 190))
            draw.rectangle((0, 0, CANVAS_SIZE[0], 24), fill=(218, 48, 70, 190))
            draw.rectangle((0, CANVAS_SIZE[1] - 24, CANVAS_SIZE[0], CANVAS_SIZE[1]), fill=(218, 48, 70, 190))
            badge = (CANVAS_SIZE[0] - 560, 900, CANVAS_SIZE[0] - 28, 1056)
            draw.rounded_rectangle(badge, radius=20, fill=(177, 27, 49, 235))
            draw.text(
                (badge[0] + 30, badge[1] + 22),
                f"COLLISION +{collision_added}",
                font=title_font,
                fill=(255, 255, 255),
            )
            draw.text(
                (badge[0] + 30, badge[1] + 79),
                collision_kind,
                font=body_font,
                fill=(255, 215, 220),
            )
        return fitted

    canvas = Image.new("RGB", CANVAS_SIZE, (7, 12, 20))
    image_box = (0, 0, 1216, 1080)
    fitted = source.copy()
    fitted.thumbnail(image_box[2:], Image.Resampling.LANCZOS)
    image_x = (image_box[2] - fitted.width) // 2
    image_y = (image_box[3] - fitted.height) // 2
    canvas.paste(fitted, (image_x, image_y))
    draw = ImageDraw.Draw(canvas, "RGBA")
    panel_x = image_box[2]
    draw.rectangle((panel_x, 0, CANVAS_SIZE[0], CANVAS_SIZE[1]), fill=(5, 10, 18, 248))
    title_font = load_font(34, bold=True)
    phase_font = load_font(42, bold=True)
    body_font = load_font(25)
    body_bold = load_font(25, bold=True)
    small_font = load_font(21)
    step = int(manifest.get("step") or 0)
    metrics = manifest.get("metrics") or {}
    completion = int(metrics.get("completion_tokens") or 0)
    reasoning_tokens = int(metrics.get("reasoning_tokens") or 0)
    visible_tokens = max(0, completion - reasoning_tokens)
    phase_color = (255, 190, 70) if phase == "INFERENCE" else (80, 205, 255)

    draw.text((panel_x + 30, 28), model_label, font=title_font, fill=(245, 248, 255))
    draw.text((panel_x + 30, 80), condition_label, font=body_bold, fill=(174, 190, 215))
    draw.text((panel_x + 30, 145), phase, font=phase_font, fill=phase_color)
    environment_state = (
        "ENVIRONMENT RUNNING"
        if phase == "ACTION" or realtime
        else "ENVIRONMENT PAUSED"
    )
    draw.text((panel_x + 30, 205), environment_state, font=body_bold, fill=phase_color)
    draw.text(
        (panel_x + 30, 255),
        f"Step {step}  |  replay sim t={replay_sim_time:.2f}s",
        font=body_font,
        fill=(225, 232, 244),
    )
    draw.text(
        (panel_x + 30, 295),
        f"Phase {phase_elapsed:.2f}/{phase_duration:.2f}s",
        font=body_font,
        fill=(225, 232, 244),
    )
    progress = 1.0 if phase_duration <= 0 else min(1.0, phase_elapsed / phase_duration)
    bar = (panel_x + 30, 340, CANVAS_SIZE[0] - 30, 365)
    draw.rounded_rectangle(bar, radius=10, fill=(38, 49, 66, 255))
    draw.rounded_rectangle(
        (bar[0], bar[1], bar[0] + int((bar[2] - bar[0]) * progress), bar[3]),
        radius=10,
        fill=phase_color,
    )
    draw.text((panel_x + 30, 405), f"RECORDED ACTION: {action_text(manifest)}", font=body_bold, fill=(95, 205, 255))
    draw.text(
        (panel_x + 30, 455),
        f"TOKENS: prompt {int(metrics.get('prompt_tokens') or 0)}  |  output {completion}",
        font=small_font,
        fill=(190, 202, 220),
    )
    draw.text(
        (panel_x + 30, 488),
        f"hidden reasoning {reasoning_tokens}  |  visible output {visible_tokens}",
        font=small_font,
        fill=(190, 202, 220),
    )
    draw.text((panel_x + 30, 545), "VISIBLE MODEL REASONING", font=body_bold, fill=(245, 220, 145))
    reasoning_lines = wrap_pixels(
        draw, parsed_reasoning(manifest), body_font, CANVAS_SIZE[0] - panel_x - 60
    )
    for index, line in enumerate(reasoning_lines[:8]):
        draw.text((panel_x + 30, 590 + index * 34), line, font=body_font, fill=(234, 238, 246))
    feedback_y = 590 + min(8, len(reasoning_lines)) * 34 + 28
    draw.text((panel_x + 30, feedback_y), "RECORDED FEEDBACK", font=body_bold, fill=(160, 220, 175))
    feedback_lines = wrap_pixels(
        draw,
        str(manifest.get("feedback") or "none"),
        small_font,
        CANVAS_SIZE[0] - panel_x - 60,
    )
    for index, line in enumerate(feedback_lines[:5]):
        draw.text((panel_x + 30, feedback_y + 42 + index * 29), line, font=small_font, fill=(216, 228, 218))
    draw.text(
        (panel_x + 30, CANVAS_SIZE[1] - 45),
        "METADATA-ONLY UE REPLAY — NO MODEL API CALL",
        font=small_font,
        fill=(150, 165, 188),
    )
    return canvas


def save_phase_frames(
    frames: list[Any],
    *,
    manifest: dict[str, Any],
    frame_dir: Path,
    frame_index: int,
    fps: float,
    model_label: str,
    condition_label: str,
    phase: str,
    phase_duration: float,
    replay_sim_start: float,
    realtime: bool,
    playback_speed: float = 1.0,
    overlay_style: str = "dashboard",
    collision_total: int = 0,
    collision_added: int = 0,
    collision_kind: str = "",
) -> int:
    output_duration = phase_duration / playback_speed
    target_count = max(1, int(round(max(output_duration, 1.0 / fps) * fps)))
    if not frames:
        raise RuntimeError(f"No source frames for {phase}")
    for output_index in range(target_count):
        source_index = min(
            len(frames) - 1,
            int(output_index * len(frames) / target_count),
        )
        output_elapsed = min(output_duration, (output_index + 1) / fps)
        elapsed = min(phase_duration, output_elapsed * playback_speed)
        sim_elapsed = elapsed if (realtime or phase == "ACTION") else 0.0
        source = frames[source_index]
        if not isinstance(source, Image.Image):
            source = Image.fromarray(source).convert("RGB")
        annotated = annotate_frame(
            source,
            manifest,
            model_label=model_label,
            condition_label=condition_label,
            phase=phase,
            phase_elapsed=elapsed,
            phase_duration=phase_duration,
            replay_sim_time=replay_sim_start + sim_elapsed,
            realtime=realtime,
            overlay_style=overlay_style,
            collision_total=collision_total,
            collision_added=collision_added,
            collision_kind=collision_kind,
        )
        annotated.save(frame_dir / f"frame_{frame_index:06d}.jpg", quality=92)
        frame_index += 1
    return frame_index


def build_server(args: argparse.Namespace, log_path: Path) -> UEServer:
    server_args = argparse.Namespace(
        ue_launcher=str(args.ue_launcher),
        ue_gpu=args.gpu,
        ue_port=args.port,
        ue_host=args.host,
        ue_width=args.width,
        ue_height=args.height,
        ue_fps=max(15, int(args.fps)),
        ue_no_rhi_thread=args.no_rhi_thread,
        ue_ready_timeout=args.ready_timeout,
        ue_settle_seconds=args.settle_seconds,
        process_shutdown_timeout=30.0,
    )
    return UEServer(server_args, log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--steps-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--ue-map", default="RT10")
    parser.add_argument("--ue-launcher", type=Path, default=Path(os.environ.get("SIMWORLD_UE_LAUNCHER", str(REPO_ROOT / "runtime/SimWorld.sh"))))
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9030)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--ready-timeout", type=float, default=240.0)
    parser.add_argument("--settle-seconds", type=float, default=120.0)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--condition-label", required=True)
    parser.add_argument("--camera-mode", choices=("first-person", "chase"), default="first-person")
    parser.add_argument("--overlay-style", choices=("dashboard", "scene"), default="dashboard")
    parser.add_argument("--capture-start-step", type=int)
    parser.add_argument("--capture-end-step", type=int)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument(
        "--presentation-inference-timing",
        choices=("simulation", "model-wall"),
        default="simulation",
        help=(
            "Choose the inference duration shown on the presentation clock. "
            "The UE environment always advances for the recorded realtime "
            "simulation-thinking duration."
        ),
    )
    parser.add_argument("--chase-distance-cm", type=float, default=520.0)
    parser.add_argument("--chase-lateral-cm", type=float, default=105.0)
    parser.add_argument("--chase-height-cm", type=float, default=300.0)
    parser.add_argument("--chase-look-ahead-cm", type=float, default=180.0)
    parser.add_argument("--chase-fov", type=float, default=82.0)
    parser.add_argument("--static-thinking", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Replay only the first N recorded decisions (for validation pilots).",
    )
    parser.add_argument("--no-rhi-thread", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--actor-trace-output",
        type=Path,
        help=(
            "Optional JSON trace of the seeded scene actors while the recorded "
            "rollout is replayed. This is intended for transplanting the same "
            "dynamic choreography into another UE map."
        ),
    )
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.playback_speed <= 0:
        raise ValueError("--playback-speed must be positive")
    result = read_json(args.result_json)
    manifests = load_manifests(args.steps_dir)
    realtime = not args.static_thinking
    expected = int(result.get("decision_count") or len(manifests))
    if len(manifests) != expected:
        raise RuntimeError(f"Found {len(manifests)} manifests, expected {expected}")
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        manifests = manifests[: args.max_steps]
    phase_summary = [
        {
            "step": int(item.get("step") or 0),
            "inference_seconds": phase_durations(
                item,
                realtime=realtime,
                inference_timing=args.presentation_inference_timing,
            )[0],
            "action_seconds": phase_durations(item, realtime=realtime)[1],
            "action": action_text(item),
        }
        for item in manifests
    ]
    if args.dry_run:
        print(json.dumps({"realtime": realtime, "steps": phase_summary}, indent=2))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output.parent / f".{args.output.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for stale in frame_dir.glob("frame_*.jpg"):
        stale.unlink()

    os.environ["SIMWORLD_OBSERVATION_WIDTH"] = str(args.width)
    os.environ["SIMWORLD_OBSERVATION_HEIGHT"] = str(args.height)
    os.environ["SIMWORLD_AGENT_CAMERA_FOV_DEG"] = str(
        result.get("agent_camera_fov_deg") or 100.0
    )
    os.environ["SIMWORLD_FIRST_PERSON_CAMERA_PITCH_DEG"] = str(
        result.get("agent_camera_pitch_deg") or -25.0
    )
    log_path = args.output.with_suffix(".ue.log")
    server = build_server(args, log_path)
    communicator = None
    frame_index = 0
    replay_sim_time = 0.0
    pose_corrections: list[float] = []
    phase_frame_counts = {"inference": 0, "action": 0}
    try:
        server.start(args.ue_map)
        unrealcv = connect_unrealcv(args.port, args.host, 30, 1.0)
        communicator = RTCommunicator(unrealcv)
        manager = setup_world(
            communicator,
            result,
            manifests[0],
            task_file=args.task_file,
            task_index=args.task_index,
            realtime=realtime,
        )
        agent = manager.agent
        actor_trace = None
        if args.actor_trace_output is not None:
            all_objects = [str(value) for value in communicator.unrealcv.get_objects()]
            regular_ids = {int(value.id) for value in manager.pedestrians}
            irregular_ids = {int(value.id) for value in manager.irregular_pedestrians}
            requested_dynamic = {
                str(value)
                for value in (
                    list(getattr(manager, "activated_movable_obstacle_ids", []) or [])
                    + list(getattr(manager, "activated_falling_object_ids", []) or [])
                )
            }
            candidate_names = []
            actor_kinds = {}
            for name in all_objects:
                if name.startswith("RT_PEDESTRIAN_"):
                    try:
                        actor_id = int(name.rsplit("_", 1)[1])
                    except ValueError:
                        continue
                    candidate_names.append(name)
                    actor_kinds[name] = (
                        "pedestrian" if actor_id in regular_ids else "robot_dog"
                    )
                elif name in requested_dynamic:
                    candidate_names.append(name)
                    actor_kinds[name] = "dynamic_object"

            # Retain every pedestrian/robot from the seeded task. Dynamic map
            # objects are restricted to the old final-street neighbourhood so
            # the trace remains compact while preserving everything that can
            # appear in the selected chase-camera segment.
            locations = communicator.unrealcv.get_location_batch(candidate_names)
            retained_names = []
            retained_initial = {}
            for name, location in zip(candidate_names, locations):
                xyz = [float(value) for value in location]
                kind = actor_kinds[name]
                if kind != "dynamic_object" or (
                    17500.0 <= xyz[0] <= 22500.0
                    and -1500.0 <= xyz[1] <= 5500.0
                ):
                    retained_names.append(name)
                    retained_initial[name] = xyz

            actor_trace = {
                "schema_version": "recorded_rollout_actor_trace_v1",
                "seed": int(result.get("seed") or 0),
                "map": args.ue_map,
                "actors": {
                    name: {
                        "kind": actor_kinds[name],
                        "initial_location_cm": retained_initial[name],
                        "samples": [],
                    }
                    for name in retained_names
                },
                "samples": [],
            }
            trace_elapsed = 0.0
            trace_context = {"step": -1, "phase": "setup"}

            def sample_actor_trace():
                if not retained_names:
                    return
                trace_locations = communicator.unrealcv.get_location_batch(
                    retained_names
                )
                trace_rotations = communicator.unrealcv.get_orientation_batch(
                    retained_names
                )
                sample_index = len(actor_trace["samples"])
                actor_trace["samples"].append(
                    {
                        "index": sample_index,
                        "time_seconds": round(trace_elapsed, 6),
                        "step": trace_context["step"],
                        "phase": trace_context["phase"],
                    }
                )
                for name, location, orientation in zip(
                    retained_names, trace_locations, trace_rotations
                ):
                    actor_trace["actors"][name]["samples"].append(
                        {
                            "sample": sample_index,
                            "location_cm": [float(value) for value in location],
                            "rotation_deg": [float(value) for value in orientation],
                        }
                    )

            original_advance_simulation_time = agent._advance_simulation_time

            def traced_advance_simulation_time(seconds):
                nonlocal trace_elapsed
                value = original_advance_simulation_time(seconds)
                trace_elapsed = round(trace_elapsed + float(seconds), 6)
                sample_actor_trace()
                return value

            agent._advance_simulation_time = traced_advance_simulation_time
            sample_actor_trace()
        communicator.unrealcv.set_camera_resolution(agent.camera_id, (args.width, args.height))
        communicator.unrealcv.set_camera_fov(
            agent.camera_id, float(result.get("agent_camera_fov_deg") or 100.0)
        )
        if args.camera_mode == "chase":
            install_chase_camera(
                agent,
                width=args.width,
                height=args.height,
                distance_cm=args.chase_distance_cm,
                lateral_cm=args.chase_lateral_cm,
                height_cm=args.chase_height_cm,
                look_ahead_cm=args.chase_look_ahead_cm,
                fov=args.chase_fov,
            )
        capture_interval = args.playback_speed / args.fps
        agent.record_action_capture_interval_s = capture_interval
        collision_total = 0

        for manifest in manifests:
            step = int(manifest.get("step") or 0)
            if actor_trace is not None:
                trace_context["step"] = step
                trace_context["phase"] = "inference"
            selected = (
                (args.capture_start_step is None or step >= args.capture_start_step)
                and (args.capture_end_step is None or step <= args.capture_end_step)
            )
            if args.capture_end_step is not None and step > args.capture_end_step:
                break
            agent.record_action_frames = selected
            correction = force_recorded_pose(agent, manifest)
            pose_corrections.append(correction)
            initial = capture_rgb(agent)
            inference_seconds, action_seconds = phase_durations(
                manifest,
                realtime=realtime,
                inference_timing=args.presentation_inference_timing,
            )
            simulation_inference_seconds = phase_durations(
                manifest,
                realtime=realtime,
                inference_timing="simulation",
            )[0]
            if realtime and simulation_inference_seconds > 0 and selected:
                location = communicator.unrealcv.get_location(agent.name)
                orientation = communicator.unrealcv.get_orientation(agent.name)
                inference_frames = [initial]
                inference_frames.extend(
                    agent._advance_stationary_action_with_pose_lock(
                        simulation_inference_seconds,
                        location,
                        orientation,
                        capture_interval=capture_interval,
                    )
                )
            elif realtime and simulation_inference_seconds > 0:
                location = communicator.unrealcv.get_location(agent.name)
                orientation = communicator.unrealcv.get_orientation(agent.name)
                agent._advance_stationary_action_with_pose_lock(
                    simulation_inference_seconds,
                    location,
                    orientation,
                )
                inference_frames = [initial]
            else:
                inference_frames = [initial]
            collision_added, collision_kind = collision_event(manifest)
            collision_total += collision_added
            if selected:
                before = frame_index
                frame_index = save_phase_frames(
                    inference_frames,
                    manifest=manifest,
                    frame_dir=frame_dir,
                    frame_index=frame_index,
                    fps=args.fps,
                    model_label=args.model_label,
                    condition_label=args.condition_label,
                    phase="INFERENCE",
                    phase_duration=inference_seconds,
                    replay_sim_start=replay_sim_time,
                    realtime=realtime,
                    playback_speed=args.playback_speed,
                    overlay_style=args.overlay_style,
                    collision_total=collision_total,
                    collision_added=collision_added,
                    collision_kind=collision_kind,
                )
                phase_frame_counts["inference"] += frame_index - before
            if realtime:
                replay_sim_time += inference_seconds

            action = make_action(manifest)
            if actor_trace is not None:
                trace_context["phase"] = "action"
            action_frames = [capture_rgb(agent)]
            if action is not None:
                agent.fixed_action_seconds = action_seconds if action_seconds > 0 else None
                success, _ = agent.take_action(action, candidate_waypoints(manifest))
                agent.fixed_action_seconds = None
                if not success:
                    raise RuntimeError(
                        f"Recorded action became invalid at step {manifest.get('step')}"
                    )
                action_frames.extend(
                    Image.fromarray(frame).convert("RGB")
                    if not isinstance(frame, Image.Image)
                    else frame.convert("RGB")
                    for frame in agent.recording_action_frames
                )
                action_frames.append(capture_rgb(agent))
            before = frame_index
            if action_seconds > 0 and selected:
                frame_index = save_phase_frames(
                    action_frames,
                    manifest=manifest,
                    frame_dir=frame_dir,
                    frame_index=frame_index,
                    fps=args.fps,
                    model_label=args.model_label,
                    condition_label=args.condition_label,
                    phase="ACTION",
                    phase_duration=action_seconds,
                    replay_sim_start=replay_sim_time,
                    realtime=True,
                    playback_speed=args.playback_speed,
                    overlay_style=args.overlay_style,
                    collision_total=collision_total,
                    collision_added=collision_added,
                    collision_kind=collision_kind,
                )
                phase_frame_counts["action"] += frame_index - before
                replay_sim_time += action_seconds

        if actor_trace is not None:
            args.actor_trace_output.parent.mkdir(parents=True, exist_ok=True)
            args.actor_trace_output.write_text(
                json.dumps(actor_trace, indent=2), encoding="utf-8"
            )

        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(args.fps),
                "-i", str(frame_dir / "frame_%06d.jpg"),
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(args.output),
            ],
            check=True,
        )
    finally:
        if communicator is not None:
            try:
                communicator.disconnect()
            except Exception:
                pass
        try:
            server.stop()
        except Exception:
            pass

    report = {
        "schema_version": REPORT_SCHEMA,
        "video": str(args.output.resolve()),
        "source_result": str(args.result_json.resolve()),
        "source_steps": str(args.steps_dir.resolve()),
        "source_decision_count": expected,
        "replayed_decision_count": len(manifests),
        "is_partial_pilot": len(manifests) != expected,
        "no_model_api_calls": True,
        "llm_client_constructed": False,
        "metadata_only_replay": True,
        "original_rollout_pixels_recovered": False,
        "realtime_inference_advances_environment": realtime,
        "static_inference_keeps_environment_paused": not realtime,
        "dynamic_scene_rebuilt_from_seed": int(result.get("seed") or 0),
        "dynamic_actor_microtiming_may_differ_from_original": True,
        "fps": args.fps,
        "camera_mode": args.camera_mode,
        "overlay_style": args.overlay_style,
        "capture_start_step": args.capture_start_step,
        "capture_end_step": args.capture_end_step,
        "playback_speed": args.playback_speed,
        "presentation_inference_timing": args.presentation_inference_timing,
        "frame_count": frame_index,
        "duration_seconds": frame_index / args.fps,
        "replay_simulation_seconds": replay_sim_time,
        "phase_frame_counts": phase_frame_counts,
        "pose_correction_cm": {
            "maximum": max(pose_corrections, default=0.0),
            "mean": (
                sum(pose_corrections) / len(pose_corrections)
                if pose_corrections else 0.0
            ),
        },
        "steps": phase_summary,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
