#!/usr/bin/env python3
"""Build an MP4 from one benchmark run's recorded agent-input frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def _load_traffic_snapshot(frame_path: Path) -> dict | None:
    step_name = frame_path.parent.name
    snapshot_path = frame_path.parent / f"{step_name}_traffic_light_snapshot.json"
    if not snapshot_path.exists():
        return None
    with snapshot_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _permission_color(permission: str) -> tuple[int, int, int]:
    if permission == "CROSS":
        return (70, 220, 70)
    if permission == "CLEAR_ONLY":
        return (0, 210, 255)
    return (70, 70, 255)


def _annotate_traffic(frame, frame_path: Path):
    snapshot = _load_traffic_snapshot(frame_path)
    if snapshot is None:
        return frame

    model_input = snapshot.get("model_input") or {}
    execution = snapshot.get("action_execution") or {}
    if not execution.get("relevant") and model_input.get("relevant"):
        execution = model_input
    override = snapshot.get("execution_override")
    step_name = frame_path.parent.name.replace("step_", "")

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 146), (8, 8, 8), -1)
    frame = cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)

    if not model_input.get("relevant"):
        lines = [
            (f"STEP {step_name} | ROUTE TRAFFIC SIGNAL: NOT RELEVANT", (210, 210, 210)),
            ("Agent is outside the guarded crossing approach.", (210, 210, 210)),
        ]
    else:
        permission = str(model_input.get("route_crossing_permission", "UNKNOWN"))
        pedestrian_state = str(model_input.get("pedestrian_state", "UNKNOWN"))
        display_permission = permission.replace("_", " ")
        display_pedestrian_state = pedestrian_state.replace("_", " ")
        remaining = model_input.get("remaining_time_s")
        remaining_text = f" | {remaining:.1f}s left" if isinstance(remaining, (int, float)) else ""
        on_crossing = bool(
            model_input.get(
                "agent_on_crosswalk",
                model_input.get("on_crosswalk"),
            )
        )
        admitted = bool(model_input.get("crossing_admitted_on_walk"))
        execution_state = str(execution.get("pedestrian_state", "UNKNOWN"))
        execution_permission = str(
            execution.get("route_crossing_permission", "UNKNOWN")
        )
        display_execution_state = execution_state.replace("_", " ")
        display_execution_permission = execution_permission.replace("_", " ")
        occupied = bool(execution.get("pedestrian_occupied"))
        clearance_extended = bool(execution.get("clearance_extended"))
        legacy_occupancy_hold = bool(execution.get("legacy_occupancy_hold"))
        extension_elapsed = execution.get("clearance_extension_elapsed_s")
        extension_text = (
            "OCCUPANCY HOLD"
            if legacy_occupancy_hold
            else
            f"EXTENDED {extension_elapsed:.1f}s"
            if clearance_extended and isinstance(extension_elapsed, (int, float))
            else "NOMINAL"
        )
        execution_signals = execution.get("signal_states", [])
        vehicle_states = [
            str(item.get("vehicle_state", "UNKNOWN"))
            for item in execution_signals
            if item.get("vehicle_state") != "NOT_APPLICABLE"
        ]
        vehicle_text = (
            "HELD RED"
            if legacy_occupancy_hold
            else "GREEN ACTIVE"
            if any(item.get("vehicle_green") for item in execution_signals)
            else "NO GREEN"
            if execution_signals
            and all(not item.get("vehicle_green") for item in execution_signals)
            else "ALL RED"
            if vehicle_states and all(state == "RED" for state in vehicle_states)
            else "/".join(sorted(set(vehicle_states))) or "UNKNOWN"
        )
        if override:
            gate_text = "GATE: MOVE BLOCKED -> WAIT"
            gate_color = (70, 70, 255)
        elif execution_permission == "CLEAR_ONLY":
            gate_text = "GATE: ALLOW CLEARING"
            gate_color = _permission_color(execution_permission)
        else:
            gate_text = "GATE: ALLOW"
            gate_color = _permission_color(execution_permission)
        lines = [
            (
                f"STEP {step_name} | PED: {display_pedestrian_state} | "
                f"PERM: {display_permission}{remaining_text}",
                _permission_color(permission),
            ),
            (
                f"ROADWAY: {'INSIDE' if on_crossing else 'OUTSIDE'} | "
                f"WALK ADMISSION: {'YES' if admitted else 'NO'}",
                (235, 235, 235),
            ),
            (
                f"OCCUPANCY: {'ACTIVE' if occupied else 'CLEAR'} | "
                f"CLEARANCE: {extension_text} | VEHICLES: {vehicle_text}",
                (0, 210, 255) if occupied else (235, 235, 235),
            ),
            (
                f"EXEC: {display_execution_state} / "
                f"{display_execution_permission} | {gate_text}",
                gate_color,
            ),
        ]

    for index, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (14, 28 + index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("steps_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument(
        "--annotate-traffic",
        action="store_true",
        help="overlay pedestrian signal, permission, admission, and gate state",
    )
    parser.add_argument(
        "--current-frame-only",
        action="store_true",
        help="use only the last (current-view) input image recorded for each step",
    )
    parser.add_argument(
        "--append-final-output",
        action="store_true",
        help=(
            "append the final post-action camera frame so the demo includes "
            "the terminal/goal state rather than ending before the last move"
        ),
    )
    parser.add_argument(
        "--full-action-timeline",
        action="store_true",
        help=(
            "for every step, include its current model-view frame, the "
            "separately recorded action frames, and the post-action frame"
        ),
    )
    parser.add_argument(
        "--raw-first-person",
        action="store_true",
        help=(
            "build from the natural first-person demo stream rather than "
            "alternating annotated model inputs with raw action frames"
        ),
    )
    parser.add_argument(
        "--preview-frame",
        type=int,
        help="zero-based annotated frame index to also save as a PNG",
    )
    parser.add_argument("--preview-output", type=Path)
    args = parser.parse_args()

    frames = sorted(args.steps_dir.glob("step_*/step_*_input_frame_*.png"))
    if args.full_action_timeline:
        frames = []
        for step_dir in sorted(args.steps_dir.glob("step_*")):
            if not step_dir.is_dir():
                continue
            inputs = sorted(
                step_dir.glob(
                    "step_*_demo_input_frame.png"
                    if args.raw_first_person
                    else "step_*_input_frame_*.png"
                )
            )
            actions = sorted(step_dir.glob("step_*_action_frame_*.png"))
            outputs = sorted(
                step_dir.glob(
                    "step_*_demo_output_frame.png"
                    if args.raw_first_person
                    else "step_*_output_frame.png"
                )
            )
            if inputs:
                frames.append(inputs[-1])
            frames.extend(actions)
            if outputs:
                frames.append(outputs[-1])
    elif args.current_frame_only:
        current_frames = {}
        for frame_path in frames:
            current_frames[frame_path.parent] = frame_path
        frames = [current_frames[key] for key in sorted(current_frames)]
    if args.append_final_output and frames:
        final_step = frames[-1].parent
        outputs = sorted(
            final_step.glob(
                "step_*_demo_output_frame.png"
                if args.raw_first_person
                else "step_*_output_frame.png"
            )
        )
        if not outputs:
            raise SystemExit(
                f"--append-final-output requested but no output frame exists in {final_step}"
            )
        frames.append(outputs[-1])
    if not frames:
        raise SystemExit(f"no input frames found under {args.steps_dir}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise SystemExit(f"could not read {frames[0]}")
    height, width = first.shape[:2]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit("OpenCV could not initialize the MP4 writer")

    try:
        for frame_index, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise SystemExit(f"could not read {frame_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            if args.annotate_traffic:
                frame = _annotate_traffic(frame, frame_path)
            if args.preview_frame == frame_index:
                if args.preview_output is None:
                    raise SystemExit("--preview-frame requires --preview-output")
                args.preview_output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.preview_output), frame):
                    raise SystemExit(f"could not write {args.preview_output}")
            writer.write(frame)
    finally:
        writer.release()

    print(f"wrote {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
