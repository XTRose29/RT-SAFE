#!/usr/bin/env python3
"""Render a high-quality, simulation-time-faithful rollout storyboard.

The source rollouts retain observations only at decision boundaries, not at
every Unreal Engine tick.  This renderer therefore holds each annotated
before/after pair for its exact interval on the saved simulation clock.  The
encoded MP4 duration is checked against that clock to millisecond precision.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CANVAS_SIZE = (2560, 1440)
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
REPORT_SCHEMA = "annotated_before_after_simulation_timeline_v1"


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = FONT_BOLD_PATH if bold else FONT_PATH
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def wrap_pixels(
    draw: ImageDraw.ImageDraw,
    value: str,
    text_font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap text according to rendered pixel width."""
    paragraphs = str(value or "").splitlines() or [""]
    result: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            result.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textlength(candidate, font=text_font) <= max_width:
                current = candidate
            else:
                result.append(current)
                current = word
        result.append(current)
    return result


def fit_image(source_path: Path, size: tuple[int, int]) -> Image.Image:
    """Return a letterboxed RGB image without cropping source pixels."""
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    fitted = Image.new("RGB", size, (8, 12, 18))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    fitted.paste(image, (x, y))
    return fitted


def point_xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict) and value.get("x") is not None and value.get("y") is not None:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def action_label(manifest: dict[str, Any]) -> tuple[str, str]:
    parsed = ((manifest.get("model_output") or {}).get("parsed_action") or {})
    feedback = str(manifest.get("feedback") or "")
    action_type = str(parsed.get("type") or "unknown").upper()
    action_param = str(parsed.get("param") or "-")
    reasoning = str(parsed.get("reasoning") or "")
    if not reasoning:
        raw = str((manifest.get("model_output") or {}).get("raw_response") or "")
        for line in raw.splitlines():
            if line.lower().startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
                break
    if action_type == "TURN_AROUND":
        return f"TURN {action_param}", reasoning
    if action_type == "MOVE_TO":
        target = (manifest.get("waypoint_execution") or {}).get(
            "selected_waypoint_distance_cm"
        )
        suffix = f" ({float(target) / 100.0:.1f} m target)" if target is not None else ""
        return f"MOVE_TO waypoint {action_param}{suffix}", reasoning
    if action_type == "WAIT":
        return f"WAIT {action_param} s", reasoning
    if action_type == "UNKNOWN" and "during model inference" in feedback.lower():
        return "NO ACTION EXECUTED — COLLISION DURING INFERENCE", reasoning
    return f"{action_type} {action_param}", reasoning


def safety_labels(manifest: dict[str, Any]) -> list[str]:
    details = manifest.get("collision_details") or {}
    passive = manifest.get("passive_collision_details") or {}
    feedback = str(manifest.get("feedback") or "").lower()
    labels: list[str] = []
    for kind in ("human", "object", "building", "vehicle"):
        count = int(details.get(kind) or 0)
        passive_count = int(passive.get(kind) or 0)
        if (
            kind == "vehicle"
            and passive_count
            and "vehicle collision occurred during model inference" in feedback
        ):
            count = max(0, count - passive_count)
        if count:
            labels.append(f"{kind.upper()} COLLISION ×{count}")
        if passive_count:
            labels.append(f"PASSIVE {kind.upper()} COLLISION ×{passive_count}")
    if int(details.get("touched_road") or 0):
        labels.append("ROAD CONTACT")
    if "traffic-rule violation" in feedback or "illegal crossing" in feedback:
        labels.append("ILLEGAL CROSSING")
    if "red-light" in feedback or "red light" in feedback:
        labels.append("RED-LIGHT VIOLATION")
    if int(details.get("fall") or 0) or "fell" in feedback:
        labels.append("FALL")
    if int(details.get("oil") or 0) or "oil" in feedback:
        labels.append("OIL / SLOWDOWN")
    if int(details.get("water") or 0) or "water" in feedback or "slip" in feedback:
        labels.append("WATER / SLIP")
    return list(dict.fromkeys(labels))


def frame_durations(manifests: list[dict[str, Any]]) -> tuple[list[float], float, float]:
    starts = [
        float((item.get("timing") or {}).get("sim_time_start_seconds") or 0.0)
        for item in manifests
    ]
    final_end = float(
        (manifests[-1].get("timing") or {}).get("sim_time_end_seconds")
        or starts[-1]
    )
    durations = [starts[index + 1] - value for index, value in enumerate(starts[:-1])]
    durations.append(final_end - starts[-1])
    for index, duration in enumerate(durations):
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError(
                f"Invalid simulation duration at decision {index + 1}: {duration}"
            )
    return durations, starts[0], final_end


def quote_concat(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def probe(path: Path, entry: str) -> str:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            entry,
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def render_frame(
    manifest: dict[str, Any],
    *,
    index: int,
    total: int,
    model_label: str,
    condition_label: str,
    outcome_label: str,
) -> tuple[Image.Image, bool, list[str]]:
    canvas = Image.new("RGB", CANVAS_SIZE, (12, 18, 27))
    draw = ImageDraw.Draw(canvas, "RGBA")
    title_font = font(38, bold=True)
    header_font = font(27, bold=True)
    body_font = font(27)
    body_bold = font(27, bold=True)
    small_font = font(23)
    feedback_font = font(28, bold=True)

    manifest_path = Path(manifest["_path"])
    source_names = list(manifest.get("input_images") or [])
    if not source_names:
        raise RuntimeError(f"No input images listed in {manifest_path}")
    before_path = manifest_path.parent / source_names[-1]
    output_name = manifest.get("output_image")
    after_path = manifest_path.parent / str(output_name) if output_name else None
    missing_output = after_path is None or not after_path.is_file()

    image_size = (945, 840)
    image_y = 136
    left_x = 317
    right_x = 1298
    before = fit_image(before_path, image_size)
    canvas.paste(before, (left_x, image_y))
    if not missing_output:
        after = fit_image(after_path, image_size)
        canvas.paste(after, (right_x, image_y))
    else:
        draw.rectangle(
            (right_x, image_y, right_x + image_size[0], image_y + image_size[1]),
            fill=(30, 35, 43, 255),
        )
        warning = "POST-ACTION FRAME\nNOT SAVED"
        bbox = draw.multiline_textbbox((0, 0), warning, font=title_font, spacing=12, align="center")
        text_x = right_x + (image_size[0] - (bbox[2] - bbox[0])) // 2
        text_y = image_y + (image_size[1] - (bbox[3] - bbox[1])) // 2
        draw.multiline_text(
            (text_x, text_y), warning, font=title_font, fill=(255, 180, 90), spacing=12, align="center"
        )

    timing = manifest.get("timing") or {}
    start = float(timing.get("sim_time_start_seconds") or 0.0)
    end = float(timing.get("sim_time_end_seconds") or start)
    step = int(manifest.get("step") or 0)
    action, reasoning = action_label(manifest)
    feedback = str(manifest.get("feedback") or "No environment feedback recorded.")
    labels = safety_labels(manifest)

    draw.rectangle((0, 0, CANVAS_SIZE[0], 100), fill=(4, 9, 17, 245))
    title = f"{model_label}  |  {condition_label}  |  TASK 0"
    draw.text((38, 20), title, font=title_font, fill=(242, 247, 255))
    outcome_color = (105, 230, 145) if outcome_label == "SUCCESS" else (255, 120, 105)
    outcome_width = draw.textlength(outcome_label, font=header_font)
    draw.text((CANVAS_SIZE[0] - outcome_width - 38, 30), outcome_label, font=header_font, fill=outcome_color)

    draw.rectangle((left_x, 100, left_x + image_size[0], 136), fill=(20, 72, 120, 255))
    draw.rectangle((right_x, 100, right_x + image_size[0], 136), fill=(22, 102, 68, 255))
    draw.text((left_x + 14, 105), "BEFORE ACTION — CURRENT MODEL VIEW", font=small_font, fill=(255, 255, 255))
    draw.text((right_x + 14, 105), "AFTER ACTION — ENVIRONMENT RESULT", font=small_font, fill=(255, 255, 255))
    draw.rectangle((left_x - 2, image_y - 2, left_x + image_size[0] + 2, image_y + image_size[1] + 2), outline=(85, 170, 240, 255), width=3)
    draw.rectangle((right_x - 2, image_y - 2, right_x + image_size[0] + 2, image_y + image_size[1] + 2), outline=(80, 205, 135, 255), width=3)

    panel_top = 997
    draw.rectangle((24, panel_top, CANVAS_SIZE[0] - 24, CANVAS_SIZE[1] - 20), fill=(6, 11, 19, 242), outline=(67, 81, 102, 255), width=2)
    draw.text(
        (48, panel_top + 18),
        f"STEP {step}  •  DECISION {index + 1}/{total}  •  SIM t={start:.2f}–{end:.2f}s  •  Δ={end - start:.2f}s  •  1× SIMULATION TIME",
        font=body_bold,
        fill=(235, 240, 250),
    )
    draw.text((48, panel_top + 62), f"ACTION: {action}", font=body_bold, fill=(95, 205, 255))

    geometry = manifest.get("observation_geometry") or {}
    before_xy = point_xy(geometry.get("agent_position_cm"))
    after_xy = point_xy((manifest.get("output_camera") or {}).get("location"))
    waypoint = manifest.get("waypoint_execution") or {}
    before_xy = point_xy(waypoint.get("pre_action_position_cm")) or before_xy
    after_xy = point_xy(waypoint.get("post_action_position_cm")) or after_xy or before_xy
    if before_xy and after_xy:
        location = (
            f"LOCATION: ({before_xy[0] / 100.0:.2f}, {before_xy[1] / 100.0:.2f}) m"
            f"  →  ({after_xy[0] / 100.0:.2f}, {after_xy[1] / 100.0:.2f}) m"
        )
    else:
        location = "LOCATION: unavailable"
    draw.text((48, panel_top + 104), location, font=body_font, fill=(210, 220, 235))

    reason_lines = wrap_pixels(draw, f"REASONING: {reasoning or 'not recorded'}", small_font, 2460)
    for line_index, line in enumerate(reason_lines[:2]):
        draw.text((48, panel_top + 147 + line_index * 30), line, font=small_font, fill=(180, 192, 210))

    feedback_top = panel_top + 214
    event_color = (148, 28, 30, 238) if labels else (18, 75, 50, 238)
    draw.rounded_rectangle(
        (42, feedback_top, CANVAS_SIZE[0] - 42, CANVAS_SIZE[1] - 38),
        radius=14,
        fill=event_color,
    )
    event_prefix = "  •  ".join(labels)
    if event_prefix:
        draw.text((62, feedback_top + 12), event_prefix, font=body_bold, fill=(255, 238, 175))
        feedback_y = feedback_top + 53
    else:
        feedback_y = feedback_top + 18
    feedback_lines = wrap_pixels(draw, f"FEEDBACK: {feedback}", feedback_font, 2410)
    for line_index, line in enumerate(feedback_lines[:4]):
        draw.text((62, feedback_y + line_index * 36), line, font=feedback_font, fill=(255, 255, 255))

    return canvas, missing_output, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--condition-label", required=True)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--preset", default="slow")
    args = parser.parse_args()

    result = read_json(args.result_json)
    manifest_paths = sorted(args.steps_dir.glob("step_*/*_manifest.json"))
    manifests: list[dict[str, Any]] = []
    for path in manifest_paths:
        manifest = read_json(path)
        manifest["_path"] = str(path)
        manifests.append(manifest)
    manifests.sort(key=lambda item: (int(item.get("step") or 0), int(item.get("decision_index") or 0)))
    if not manifests:
        raise SystemExit(f"No step manifests found below {args.steps_dir}")
    expected = int(result.get("decision_count") or len(manifests))
    if len(manifests) != expected:
        raise RuntimeError(f"Found {len(manifests)} manifests, expected {expected}")

    durations, sim_start, sim_end = frame_durations(manifests)
    outcome = "SUCCESS" if result.get("success") else f"FAILED: {str(result.get('termination_reason') or 'unknown').upper()}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    missing_outputs: list[int] = []
    safety_steps: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix=f".{args.output.stem}_", dir=args.output.parent) as temporary:
        frame_dir = Path(temporary)
        for index, manifest in enumerate(manifests):
            frame, missing, labels = render_frame(
                manifest,
                index=index,
                total=len(manifests),
                model_label=args.model_label,
                condition_label=args.condition_label,
                outcome_label=outcome,
            )
            frame_path = frame_dir / f"frame_{index:04d}.png"
            frame.save(frame_path, compress_level=3)
            if missing:
                missing_outputs.append(int(manifest.get("step") or index))
            if labels:
                safety_steps.append(
                    {
                        "step": int(manifest.get("step") or index),
                        "labels": labels,
                        "feedback": manifest.get("feedback"),
                    }
                )

        concat = frame_dir / "timeline.ffconcat"
        lines = ["ffconcat version 1.0"]
        packet_seconds = 0.001
        for index, duration in enumerate(durations):
            frame_path = frame_dir / f"frame_{index:04d}.png"
            lines.append(f"file '{quote_concat(frame_path)}'")
            lines.append("option framerate 1000")
            encoded_duration = duration - (packet_seconds if index == len(durations) - 1 else 0.0)
            if encoded_duration <= 0:
                raise RuntimeError(f"Final interval {duration:.6f}s is too short")
            lines.append(f"duration {encoded_duration:.9f}")
        final_frame = frame_dir / f"frame_{len(manifests) - 1:04d}.png"
        lines.append(f"file '{quote_concat(final_frame)}'")
        lines.append("option framerate 1000")
        concat.write_text("\n".join(lines) + "\n", encoding="utf-8")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat),
                "-fps_mode",
                "passthrough",
                "-c:v",
                "libx264",
                "-crf",
                str(args.crf),
                "-preset",
                args.preset,
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-video_track_timescale",
                "1000",
                str(args.output),
            ],
            check=True,
        )

    duration = float(probe(args.output, "format=duration"))
    resolution = probe(args.output, "stream=width,height").splitlines()
    expected_duration = sim_end - sim_start
    error = duration - expected_duration
    if abs(error) > 0.002:
        raise RuntimeError(
            f"Video duration {duration:.6f}s differs from simulation duration "
            f"{expected_duration:.6f}s by {error:+.6f}s"
        )
    if resolution != [str(CANVAS_SIZE[0]), str(CANVAS_SIZE[1])]:
        raise RuntimeError(f"Unexpected encoded resolution: {resolution}")

    report = {
        "schema_version": REPORT_SCHEMA,
        "video": str(args.output.resolve()),
        "result_json": str(args.result_json.resolve()),
        "steps_dir": str(args.steps_dir.resolve()),
        "model_label": args.model_label,
        "condition_label": args.condition_label,
        "outcome": outcome,
        "decisions": len(manifests),
        "source_is_decision_boundary_snapshots_not_continuous_ue_capture": True,
        "timing_mode": "one_video_second_per_simulation_second_variable_frame_duration",
        "simulation_start_seconds": sim_start,
        "simulation_end_seconds": sim_end,
        "simulation_duration_seconds": expected_duration,
        "encoded_duration_seconds": duration,
        "duration_error_seconds": error,
        "resolution": list(CANVAS_SIZE),
        "codec": "H.264/libx264",
        "crf": args.crf,
        "preset": args.preset,
        "missing_output_frame_steps": missing_outputs,
        "safety_event_steps": safety_steps,
        "per_decision_duration_seconds": durations,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
