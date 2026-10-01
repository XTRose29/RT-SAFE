#!/usr/bin/env python3
"""Render exact per-decision rollout inputs as an annotated MP4."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
VIDEO_REPORT_SCHEMA_VERSION = "all_policy_inputs_sim_time_v2"


def load_font(size: int, bold: bool = False):
    path = FONT_BOLD_PATH if bold else FONT_PATH
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def classify_events(manifest: dict) -> list[str]:
    details = manifest.get("collision_details") or {}
    passive_details = manifest.get("passive_collision_details") or {}
    feedback = str(manifest.get("feedback") or "").lower()
    events: list[str] = []

    for kind in ("human", "object", "building", "vehicle"):
        count = int(details.get(kind) or 0)
        passive_count = int(passive_details.get(kind) or 0)
        if (
            kind == "vehicle"
            and passive_count
            and "vehicle collision occurred during model inference" in feedback
        ):
            # Terminal inference-time impacts retain the total vehicle count in
            # collision_details for termination reporting. Do not present that
            # same impact as both active and passive in the video.
            count = max(0, count - passive_count)
        if count:
            events.append(f"{kind} collision x{count}")
        if passive_count:
            events.append(f"passive {kind} collision x{passive_count}")
    if int(details.get("touched_road") or 0):
        events.append("touched road")
    if "red light" in feedback:
        events.append("red-light violation")
    if "fell" in feedback:
        events.append("fall")
    if "water" in feedback:
        events.append("water/slip")
    if "oil" in feedback:
        events.append("oil/slip")

    structured = manifest.get("safety_events") or {}
    for event in structured.get("red_light_violations", []):
        state = str(event.get("pedestrian_state") or "DON'T WALK")
        launch = event.get("conflict_vehicle") or {}
        if launch.get("status") in {"launched", "completed", "impact"}:
            distance_m = float(launch.get("launch_distance_cm") or 0.0) / 100.0
            events.append(
                f"red light + vehicle launch {distance_m:.1f}m ({state})"
            )
        else:
            events.append(f"red-light violation ({state})")
    for event in structured.get("illegal_crossing_violations", []):
        launch = event.get("conflict_vehicle") or {}
        if launch.get("status") in {"launched", "completed", "impact"}:
            distance_m = float(launch.get("launch_distance_cm") or 0.0) / 100.0
            events.append(f"illegal crossing + vehicle launch {distance_m:.1f}m")
        else:
            events.append("illegal crossing")

    # Exact duplicate labels add no value in the video overlay.
    return list(dict.fromkeys(events))


def structured_event_times(manifest: dict) -> list[float]:
    structured = manifest.get("safety_events") or {}
    times = []
    for key in ("red_light_violations", "illegal_crossing_violations"):
        for event in structured.get(key, []):
            value = event.get("sim_time_s")
            if value is not None:
                times.append(float(value))
    return times


def prepare_frame_directory(frame_dir: Path) -> None:
    """Create the render directory and remove frames from older attempts."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in frame_dir.glob("frame_*.png"):
        stale_frame.unlink()


def is_turn_action(manifest: dict) -> bool:
    """Use the parsed action, not potentially stale execution metadata."""
    action = (
        (manifest.get("model_output") or {}).get("parsed_action") or {}
    )
    return action.get("type") == "turn_around"


def current_policy_image_name(source_names: list[str], step: object) -> str:
    """Return the current marked view from a valid policy-image sequence."""
    if not source_names:
        raise RuntimeError(
            f"Expected at least one policy image (history plus current) at "
            f"step {step}, got 0"
        )
    return source_names[-1]


def policy_input_grid_shape(max_input_count: int) -> tuple[int, int]:
    """Return a stable, compact grid that can show every policy input."""
    if max_input_count < 1:
        raise RuntimeError(
            f"Expected at least one policy image per decision, got "
            f"{max_input_count}"
        )
    if max_input_count == 1:
        columns = 1
    elif max_input_count <= 4:
        columns = 2
    else:
        columns = 4
    return columns, math.ceil(max_input_count / columns)


def simulation_frame_durations(manifests: list[dict]) -> list[float]:
    """Hold each input bundle until the next decision on the simulation clock."""
    starts = [
        float((manifest.get("timing") or {}).get("sim_time_start_seconds") or 0.0)
        for manifest in manifests
    ]
    final_end = float(
        (manifests[-1].get("timing") or {}).get("sim_time_end_seconds")
        or starts[-1]
    )
    durations = [
        starts[index + 1] - start
        for index, start in enumerate(starts[:-1])
    ]
    durations.append(final_end - starts[-1])
    for index, duration in enumerate(durations):
        if duration <= 0:
            raise RuntimeError(
                f"Non-positive simulation display duration at frame {index}: "
                f"{duration:.6f}s"
            )
    return durations


def ffconcat_quote(path: Path) -> str:
    """Quote an absolute path for ffmpeg's concat-demuxer syntax."""
    return str(path.resolve()).replace("'", "'\\''")


def probe_video_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def fit_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "..."
    while text and draw.textbbox((0, 0), text + suffix, font=font)[2] > max_width:
        text = text[:-1]
    return text + suffix

def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
) -> list[str]:
    """Wrap overlay text without discarding safety-event information."""
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--fps", default=2.0, type=float)
    parser.add_argument(
        "--policy-label",
        default="deterministic constant-speed greedy-to-goal code baseline",
    )
    parser.add_argument("--header-label", default="EXACT AGENT INPUT")
    parser.add_argument(
        "--learned-agent-model-was-run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--supplemental-events", type=Path)
    parser.add_argument(
        "--report-output",
        type=Path,
        help=(
            "Optional report path. Defaults to rollout_input_video_report.json "
            "beside the video."
        ),
    )
    args = parser.parse_args()

    supplemental_by_step: dict[int, list[dict]] = {}
    if args.supplemental_events:
        supplemental = json.loads(args.supplemental_events.read_text(encoding="utf-8"))
        for item in supplemental:
            supplemental_by_step.setdefault(int(item["step"]), []).append(item)

    manifest_paths = sorted(args.steps_dir.glob("step_*/*_manifest.json"))
    manifests = []
    for path in manifest_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = path
        manifests.append(data)
    manifests.sort(key=lambda item: (int(item.get("step", 0)), int(item.get("decision_index", 0))))
    if not manifests:
        raise SystemExit(f"No step manifests found below {args.steps_dir}")

    frame_dir = args.output.parent / "annotated_input_frames"
    prepare_frame_directory(frame_dir)
    event_records = []
    action_counts: dict[str, int] = {}
    render_timings: list[dict] = []

    title_font = load_font(19, bold=True)
    body_font = load_font(17)
    small_font = load_font(14)
    event_font = load_font(17, bold=True)

    input_counts = [
        len(manifest.get("input_images") or []) for manifest in manifests
    ]
    max_input_count = max(input_counts)
    grid_columns, grid_rows = policy_input_grid_shape(max_input_count)
    first_source_names = manifests[0].get("input_images") or []
    first_source_path = (
        manifests[0]["_path"].parent
        / current_policy_image_name(
            first_source_names, manifests[0].get("step")
        )
    )
    with Image.open(first_source_path) as first_source:
        source_width, source_height = first_source.size
    header_height = 92
    input_label_height = 30
    footer_height = 220
    canvas_width = source_width * grid_columns
    canvas_height = (
        header_height
        + grid_rows * (input_label_height + source_height)
        + footer_height
    )

    for frame_index, manifest in enumerate(manifests):
        source_names = manifest.get("input_images") or []
        current_name = current_policy_image_name(
            source_names, manifest.get("step")
        )
        source_paths = [
            manifest["_path"].parent / name for name in source_names
        ]
        source_path = manifest["_path"].parent / current_name
        frame = Image.new(
            "RGB", (canvas_width, canvas_height), (20, 20, 20)
        )
        bundle_draw = ImageDraw.Draw(frame, "RGBA")
        for input_index, input_path in enumerate(source_paths):
            with Image.open(input_path) as source:
                policy_image = source.convert("RGB")
            if policy_image.size != (source_width, source_height):
                raise RuntimeError(
                    f"Policy input size changed at step {manifest.get('step')}: "
                    f"{input_path} is {policy_image.size}, expected "
                    f"{(source_width, source_height)}"
                )
            column = input_index % grid_columns
            row = input_index // grid_columns
            x = column * source_width
            label_y = (
                header_height
                + row * (input_label_height + source_height)
            )
            image_y = label_y + input_label_height
            frame.paste(policy_image, (x, image_y))
            role = (
                "CURRENT"
                if input_index == len(source_paths) - 1
                else "HISTORY"
            )
            bundle_draw.rectangle(
                (
                    x,
                    label_y,
                    x + source_width - 1,
                    image_y - 1,
                ),
                fill=(5, 5, 5, 255),
            )
            bundle_draw.text(
                (x + 10, label_y + 6),
                (
                    f"MODEL INPUT {input_index + 1}/{len(source_paths)}"
                    f" | {role}"
                ),
                fill=(
                    (255, 230, 120)
                    if role == "CURRENT"
                    else (210, 210, 210)
                ),
                font=small_font,
            )

        draw = ImageDraw.Draw(frame, "RGBA")
        width, height = frame.size
        timing = manifest.get("timing") or {}
        step = int(manifest.get("step") or 0)
        input_render = timing.get("input_render") or {}
        if input_render:
            render_timings.append({"step": step, **input_render})
        start_time = float(timing.get("sim_time_start_seconds") or 0.0)
        end_time = float(timing.get("sim_time_end_seconds") or start_time)
        decision = int(manifest.get("decision_index") or step + 1)
        action = ((manifest.get("model_output") or {}).get("parsed_action") or {})
        action_type = str(action.get("type") or "none").upper()
        action_param = str(action.get("param") or "-")
        action_key = f"{action_type} {action_param}"
        action_counts[action_key] = action_counts.get(action_key, 0) + 1
        events = classify_events(manifest)
        event_times = structured_event_times(manifest)
        waypoint_execution = manifest.get("waypoint_execution") or {}
        turn_execution = manifest.get("turn_execution") or {}
        supplemental_events = supplemental_by_step.get(step, [])
        for item in supplemental_events:
            events.extend(str(event) for event in item.get("events", []))
            if item.get("sim_time_s") is not None:
                event_times.append(float(item["sim_time_s"]))

        draw.rectangle((0, 0, width, 92), fill=(0, 0, 0, 205))
        task_label = f"SCENARIO {args.task_id}"
        if args.task_index is not None:
            task_label += f" (INDEX {args.task_index})"
        heading = (
            f"{args.header_label} | {task_label} | {args.difficulty.upper()} | "
            f"STEP {step:02d} | t={start_time:.1f}s"
        )
        draw.text((14, 10), fit_text(draw, heading, title_font, width - 28), fill=(255, 255, 255), font=title_font)
        action_label = f"ACTION TAKEN: {action_type}  param={action_param}"
        if waypoint_execution:
            distance_m = float(
                waypoint_execution.get("selected_waypoint_distance_cm") or 0.0
            ) / 100.0
            duration_s = float(
                waypoint_execution.get("move_command_duration_seconds") or 0.0
            )
            action_label = (
                f"ACTION: MOVE_TO {action_param} -> RED DOT {action_param} | "
                f"target={distance_m:.2f}m | walk={duration_s:.1f}s"
            )
        elif action_type == "TURN_AROUND" and turn_execution:
            action_label = (
                f"ACTION: TURN {action_param} | "
                f"verified={float(turn_execution.get('verified_delta_from_input_deg') or 0.0):+.1f}deg | "
                f"translation={float(turn_execution.get('verified_position_drift_cm') or 0.0):.1f}cm"
            )
        draw.text(
            (14, 42),
            fit_text(draw, action_label, body_font, width - 28),
            fill=(105, 210, 255),
            font=body_font,
        )
        render_s = input_render.get("total_wall_seconds")
        render_label = (
            f"Input render: {float(render_s):.3f}s | "
            f"resolution={frame.width}x{frame.height}"
            if render_s is not None else f"resolution={frame.width}x{frame.height}"
        )
        draw.text(
            (14, 68),
            fit_text(
                draw,
                (
                    f"Exact current image; {len(source_names) - 1} prior-action "
                    "history frame(s). "
                    if len(source_names) > 1
                    else "Exact current policy image. "
                ) + render_label,
                small_font,
                width - 28,
            ),
            fill=(220, 220, 220),
            font=small_font,
        )

        if events:
            event_heading = (
                "POST-ACTION SAFETY EVENT"
                if event_times
                else "SAFETY EVENT IN THIS STEP"
            )
            event_lines: list[str] = []
            for event in events:
                event_lines.extend(
                    wrap_text(draw, event.upper(), event_font, width - 28)
                )
            event_line_height = 24
            banner_height = 65 + event_line_height * len(event_lines)
            banner_top = height - banner_height
            draw.rectangle(
                (0, banner_top, width, height),
                fill=(155, 0, 0, 220),
            )
            draw.text(
                (14, banner_top + 8),
                event_heading,
                fill=(255, 255, 255),
                font=event_font,
            )
            for line_index, event_line in enumerate(event_lines):
                draw.text(
                    (14, banner_top + 32 + event_line_height * line_index),
                    event_line,
                    fill=(255, 255, 255),
                    font=event_font,
                )
            draw.text(
                (14, height - 25),
                fit_text(
                    draw,
                    (
                        (
                            f"input t={start_time:.1f}s | event t={min(event_times):.1f}s"
                            if event_times
                            else f"t={start_time:.1f}-{end_time:.1f}s"
                        )
                        + (
                            f" | endpoint error={float(waypoint_execution.get('selected_waypoint_endpoint_error_cm')):.1f}cm"
                            if waypoint_execution.get("selected_waypoint_endpoint_error_cm") is not None
                            else ""
                        )
                    ),
                    small_font,
                    width - 28,
                ),
                fill=(255, 230, 150),
                font=small_font,
            )
            event_records.append(
                {
                    "step": step,
                    "decision_index": decision,
                    "sim_time_start_seconds": start_time,
                    "sim_time_end_seconds": end_time,
                    "events": events,
                    "feedback": manifest.get("feedback"),
                    "collision_details": manifest.get("collision_details"),
                    "passive_collision_details": manifest.get(
                        "passive_collision_details"
                    ),
                    "supplemental_event_sources": supplemental_events,
                    "source_input_image": str(source_path),
                    "source_input_images": [
                        str(path) for path in source_paths
                    ],
                }
            )
        else:
            reached = waypoint_execution.get("reached_selected_waypoint")
            result_fill = (0, 70, 20, 190) if reached is not False else (135, 75, 0, 210)
            draw.rectangle((0, height - 42, width, height), fill=result_fill)
            if waypoint_execution:
                endpoint_error = float(
                    waypoint_execution.get("selected_waypoint_endpoint_error_cm") or 0.0
                )
                actual_m = float(
                    waypoint_execution.get("actual_displacement_cm") or 0.0
                ) / 100.0
                result_text = (
                    f"RESULT: {'REACHED MARKED DOT' if reached else 'BLOCKED / DID NOT REACH DOT'}"
                    f" | moved={actual_m:.2f}m | endpoint error={endpoint_error:.1f}cm"
                )
            elif turn_execution:
                result_text = (
                    f"RESULT: TURN {action_param} VERIFIED | heading delta="
                    f"{float(turn_execution.get('verified_delta_from_input_deg') or 0.0):+.1f}deg"
                    f" | input-pose drift="
                    f"{float(turn_execution.get('verified_position_drift_cm') or 0.0):.1f}cm"
                )
            else:
                result_text = (
                    f"No safety event reported after this action (through t={end_time:.1f}s)"
                )
            draw.text(
                (14, height - 31),
                fit_text(draw, result_text, small_font, width - 28),
                fill=(230, 255, 230),
                font=small_font,
            )

        frame.save(frame_dir / f"frame_{frame_index:04d}.png", compress_level=3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_durations = simulation_frame_durations(manifests)
    sim_time_start = float(
        (manifests[0].get("timing") or {}).get(
            "sim_time_start_seconds"
        )
        or 0.0
    )
    sim_time_end = float(
        (manifests[-1].get("timing") or {}).get(
            "sim_time_end_seconds"
        )
        or sim_time_start
    )
    sim_time_duration = sim_time_end - sim_time_start
    concat_path = frame_dir / "simulation_timeline.ffconcat"
    concat_lines = ["ffconcat version 1.0"]
    concat_time_base_seconds = 0.001
    for frame_index, duration in enumerate(frame_durations):
        frame_path = frame_dir / f"frame_{frame_index:04d}.png"
        concat_lines.append(f"file '{ffconcat_quote(frame_path)}'")
        concat_lines.append("option framerate 1000")
        encoded_duration = duration
        if frame_index == len(frame_durations) - 1:
            # The repeated terminal PNG contributes one 1ms packet. Shorten
            # its hold by that packet so container duration equals sim time.
            encoded_duration -= concat_time_base_seconds
        if encoded_duration <= 0:
            raise RuntimeError(
                "Final simulation interval is too short for the 1ms video "
                f"time base: {duration:.6f}s"
            )
        concat_lines.append(f"duration {encoded_duration:.9f}")
    # concat requires the final file again for its duration directive to apply.
    final_frame_path = frame_dir / f"frame_{len(manifests) - 1:04d}.png"
    concat_lines.append(f"file '{ffconcat_quote(final_frame_path)}'")
    concat_lines.append("option framerate 1000")
    concat_path.write_text(
        "\n".join(concat_lines) + "\n", encoding="utf-8"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-fps_mode", "passthrough",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-bf", "0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-video_track_timescale", "1000",
            str(args.output),
        ],
        check=True,
    )
    encoded_video_duration = probe_video_duration(args.output)
    duration_alignment_error = encoded_video_duration - sim_time_duration
    if abs(duration_alignment_error) > 0.002:
        raise RuntimeError(
            "Encoded video duration does not match the simulation timeline: "
            f"video={encoded_video_duration:.6f}s, "
            f"simulation={sim_time_duration:.6f}s, "
            f"error={duration_alignment_error:+.6f}s"
        )

    report = {
        "video_report_schema_version": VIDEO_REPORT_SCHEMA_VERSION,
        "task_id": args.task_id,
        "task_index": args.task_index,
        "difficulty": args.difficulty,
        "policy": args.policy_label,
        "learned_agent_model_was_run": args.learned_agent_model_was_run,
        "rollout_agent_body_was_run": True,
        "rollout_environment_was_run": True,
        "source_frames_are_exact_policy_inputs": True,
        "video_uses_current_policy_input_image": True,
        "video_includes_every_policy_input_image": True,
        "policy_inputs_are_displayed_in_model_order": True,
        "rendered_policy_input_regions_are_unscaled_and_unobscured": True,
        "annotation_overlays_are_outside_policy_input_pixels": True,
        "history_images_are_recorded_prior_action_frames": True,
        "current_policy_image_is_last_input": True,
        "policy_input_image_counts": [
            len(manifest.get("input_images") or []) for manifest in manifests
        ],
        "video_frames_are_postprocessed_copies_with_overlays": True,
        "source_step_directory": str(args.steps_dir),
        "video": str(args.output),
        "fps": args.fps,
        "timing_mode": "simulation_time_variable_frame_duration",
        "frame_count": len(manifests),
        "video_duration_seconds": encoded_video_duration,
        "sim_time_start_seconds": sim_time_start,
        "sim_time_end_seconds": sim_time_end,
        "sim_time_duration_seconds": sim_time_duration,
        "video_to_sim_time_duration_error_seconds": duration_alignment_error,
        "per_decision_display_duration_seconds": frame_durations,
        "policy_input_native_resolution": [
            source_width,
            source_height,
        ],
        "policy_input_grid": {
            "columns": grid_columns,
            "rows": grid_rows,
            "max_input_count": max_input_count,
            "canvas_resolution": [canvas_width, canvas_height],
        },
        "action_counts": action_counts,
        "safety_event_count": len(event_records),
        "safety_events": event_records,
        "structured_safety_events_by_step": [
            {
                "step": int(manifest.get("step") or 0),
                "events": manifest.get("safety_events") or {},
            }
            for manifest in manifests
            if manifest.get("safety_events")
        ],
        "per_step_input_render_timings": render_timings,
        "mean_input_render_wall_seconds": (
            sum(float(item["total_wall_seconds"]) for item in render_timings)
            / len(render_timings)
            if render_timings else None
        ),
        "waypoint_alignment": {
            "move_steps": sum(
                1 for manifest in manifests if manifest.get("waypoint_execution")
            ),
            "selected_waypoint_matched_command_count": sum(
                1
                for manifest in manifests
                if (manifest.get("waypoint_execution") or {}).get(
                    "selected_waypoint_matches_command"
                )
            ),
            "reached_selected_waypoint_count": sum(
                1
                for manifest in manifests
                if (manifest.get("waypoint_execution") or {}).get(
                    "reached_selected_waypoint"
                )
            ),
            "turn_steps": sum(
                1 for manifest in manifests if is_turn_action(manifest)
            ),
            "turn_verified_stationary_count": sum(
                1
                for manifest in manifests
                if is_turn_action(manifest)
                and manifest.get("turn_execution")
                and float(
                    (manifest.get("turn_execution") or {}).get(
                        "verified_position_drift_cm",
                        float("inf"),
                    )
                ) <= 1.0
            ),
        },
    }
    report_path = args.report_output or args.output.with_name(
        "rollout_input_video_report.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
