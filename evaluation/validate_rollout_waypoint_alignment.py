#!/usr/bin/env python3
"""Validate numbered waypoint, executed target, and sequential-frame alignment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


EXPECTED_DISTANCE_CM = {
    1: 100.0,
    2: 200.0,
    3: 200.0,
    4: 200.0,
    5: 400.0,
    6: 400.0,
    7: 400.0,
}
EXPECTED_ANGLE_DEG = {
    1: 0.0,
    2: 0.0,
    3: 45.0,
    4: -45.0,
    5: 0.0,
    6: 45.0,
    7: -45.0,
}
WALK_SPEED_CM_PER_SECOND = 200.0
OIL_SPEED_MULTIPLIER = 0.5


def distance(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def circular_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def heading_degrees(direction: dict) -> float:
    return math.degrees(math.atan2(float(direction["y"]), float(direction["x"])))


def signed_circular_delta(after: float, before: float) -> float:
    return (after - before + 180.0) % 360.0 - 180.0


def expected_move_duration_seconds(
    distance_cm: float,
    oil_slowdown_pending: bool = False,
) -> float:
    speed = WALK_SPEED_CM_PER_SECOND
    if oil_slowdown_pending:
        speed *= OIL_SPEED_MULTIPLIER
    return distance_cm / speed


def pixel_mae(first: Path, second: Path) -> float | None:
    if not first.is_file() or not second.is_file():
        return None
    with Image.open(first) as first_image, Image.open(second) as second_image:
        first_rgb = first_image.convert("RGB")
        second_rgb = second_image.convert("RGB")
        if first_rgb.size != second_rgb.size:
            return None
        channel_means = ImageStat.Stat(ImageChops.difference(first_rgb, second_rgb)).mean
    return float(sum(channel_means) / len(channel_means))


def has_safety_explanation(manifest: dict) -> bool:
    for key in ("collision_details", "passive_collision_details"):
        details = manifest.get(key) or {}
        if any(int(details.get(kind) or 0) > 0 for kind in ("human", "object", "building", "vehicle")):
            return True
        if int(details.get("touched_road") or 0) > 0:
            return True
        if int(details.get("off_route_move_no_progress") or 0) > 0:
            return True
    feedback = str(manifest.get("feedback") or "").lower()
    return any(
        term in feedback
        for term in (
            "collision",
            "slip",
            "water",
            "oil",
            "fell",
            "road",
            "made no progress",
        )
    )


def action_was_executed(manifest: dict) -> bool:
    """Distinguish parsed-but-rejected model output from executed actions."""
    return str(manifest.get("feedback") or "") not in {
        "Invalid action (not executed)",
        "Parse error (no executable action)",
    }


def retained_history_source_frames(
    recorded_action_frames: list[str],
    next_history_inputs: list[str],
) -> list[str] | None:
    """Select comparable sources, or None when sparse retention pruned them."""
    if not next_history_inputs:
        return []
    if not recorded_action_frames:
        return None
    return recorded_action_frames[-len(next_history_inputs):]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--geometry-tolerance-cm", default=0.1, type=float)
    parser.add_argument("--sequence-tolerance-cm", default=1.0, type=float)
    args = parser.parse_args()

    paths = sorted(args.steps_dir.glob("step_*/*_manifest.json"))
    manifests = []
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["_path"] = path
        manifests.append(manifest)
    manifests.sort(key=lambda item: (int(item.get("step", 0)), int(item.get("decision_index", 0))))
    if not manifests:
        raise SystemExit(f"No manifests found below {args.steps_dir}")

    failures: list[dict] = []
    step_records: list[dict] = []
    move_steps = 0
    reached_steps = 0
    safety_explained_misses = 0
    sequential_checks = 0
    pixel_maes: list[float] = []
    history_pixel_maes: list[float] = []
    history_action_frame_checks = 0
    history_action_frame_unverifiable = 0
    turn_steps = 0
    aligned_turn_steps = 0
    stationary_turn_checks = 0
    stationary_turn_matches = 0
    oil_slowdown_pending = False
    oil_slowed_move_steps = 0
    unexecuted_action_steps = 0

    def fail(step: int, check: str, actual=None, expected=None):
        failures.append({"step": step, "check": check, "actual": actual, "expected": expected})

    for manifest_index, manifest in enumerate(manifests):
        step = int(manifest.get("step") or 0)
        geometry = manifest.get("observation_geometry") or {}
        agent = geometry.get("agent_position_cm")
        candidates = geometry.get("annotated_candidates") or []
        if len(candidates) != 7:
            fail(step, "seven_annotated_candidates", len(candidates), 7)

        candidates_by_index = {int(item["index"]): item for item in candidates if "index" in item}
        for index, expected_distance in EXPECTED_DISTANCE_CM.items():
            candidate = candidates_by_index.get(index)
            if candidate is None or agent is None:
                fail(step, f"candidate_{index}_geometry_present", False, True)
                continue
            world_distance = distance(agent, candidate["world_position_cm"])
            recorded_distance = float(candidate.get("distance_cm") or 0.0)
            recorded_angle = float(candidate.get("relative_angle_deg") or 0.0)
            if abs(world_distance - expected_distance) > args.geometry_tolerance_cm:
                fail(step, f"candidate_{index}_world_distance_cm", world_distance, expected_distance)
            if abs(recorded_distance - expected_distance) > args.geometry_tolerance_cm:
                fail(step, f"candidate_{index}_label_distance_cm", recorded_distance, expected_distance)
            if circular_difference(recorded_angle, EXPECTED_ANGLE_DEG[index]) > 0.1:
                fail(step, f"candidate_{index}_relative_angle_deg", recorded_angle, EXPECTED_ANGLE_DEG[index])
            agent_direction = geometry.get("agent_direction")
            if agent_direction:
                bearing = math.degrees(math.atan2(
                    float(candidate["world_position_cm"]["y"]) - float(agent["y"]),
                    float(candidate["world_position_cm"]["x"]) - float(agent["x"]),
                ))
                # Prompt angles are positive to the visual left; UE yaw grows
                # toward the visual right, so the world angle is heading - bearing.
                world_angle = signed_circular_delta(heading_degrees(agent_direction), bearing)
                if circular_difference(world_angle, EXPECTED_ANGLE_DEG[index]) > 0.5:
                    fail(step, f"candidate_{index}_world_side_matches_prompt_angle",
                         world_angle, EXPECTED_ANGLE_DEG[index])
        pixels = {index: (candidates_by_index.get(index) or {}).get("image_pixel") for index in EXPECTED_ANGLE_DEG}
        for left, center, right in ((3, 2, 4), (6, 5, 7)):
            if all(pixels.get(index) for index in (left, center, right)) and not (
                pixels[left][0] < pixels[center][0] < pixels[right][0]
            ):
                fail(step, f"markers_{left}_{center}_{right}_drawn_left_to_right",
                     [pixels[index][0] for index in (left, center, right)], "increasing image x")

        action = ((manifest.get("model_output") or {}).get("parsed_action") or {})
        action_executed = action_was_executed(manifest)
        if not action_executed:
            unexecuted_action_steps += 1
        execution = manifest.get("waypoint_execution")
        turn_execution = manifest.get("turn_execution")
        step_record = {
            "step": step,
            "action": action,
            "waypoint_execution": execution,
            "turn_execution": turn_execution,
        }
        if execution:
            move_steps += 1
            selected_index = int(execution["selected_image_waypoint_index"])
            candidate = candidates_by_index.get(selected_index)
            selected = execution.get("selected_image_waypoint_cm")
            commanded = execution.get("commanded_waypoint_cm")
            pre_action = execution.get("pre_action_position_cm")
            expected_distance = EXPECTED_DISTANCE_CM.get(selected_index)
            if str(action.get("param")) != str(selected_index):
                fail(step, "action_description_matches_selected_dot", action.get("param"), selected_index)
            if candidate is None or selected is None:
                fail(step, "selected_dot_geometry_present", False, True)
            elif distance(candidate["world_position_cm"], selected) > args.geometry_tolerance_cm:
                fail(step, "selected_dot_matches_annotated_candidate", selected, candidate["world_position_cm"])
            if selected is None or commanded is None or distance(selected, commanded) > args.geometry_tolerance_cm:
                fail(step, "selected_dot_matches_commanded_target", commanded, selected)
            if not execution.get("selected_waypoint_matches_command"):
                fail(step, "runtime_selected_waypoint_matches_command", False, True)
            if agent is None or pre_action is None or distance(agent, pre_action) > args.geometry_tolerance_cm:
                fail(step, "input_agent_position_matches_move_start", pre_action, agent)
            if expected_distance is not None:
                actual_distance = float(execution.get("selected_waypoint_distance_cm") or 0.0)
                if abs(actual_distance - expected_distance) > args.geometry_tolerance_cm:
                    fail(step, "selected_distance_matches_meter_description", actual_distance, expected_distance)
                expected_duration = expected_move_duration_seconds(
                    expected_distance,
                    oil_slowdown_pending,
                )
                if oil_slowdown_pending:
                    oil_slowed_move_steps += 1
                actual_duration = float(execution.get("move_command_duration_seconds") or 0.0)
                if abs(actual_duration - expected_duration) > 0.01:
                    fail(step, "move_duration_matches_distance_at_constant_speed", actual_duration, expected_duration)
            # Oil applies to exactly the next movement. Turns and waits leave
            # the pending slowdown armed; executing any move consumes it.
            oil_slowdown_pending = False
            if execution.get("reached_selected_waypoint"):
                reached_steps += 1
            elif has_safety_explanation(manifest):
                safety_explained_misses += 1
            else:
                fail(
                    step,
                    "unexplained_failure_to_reach_selected_dot",
                    execution.get("selected_waypoint_endpoint_error_cm"),
                    f"<= {execution.get('endpoint_tolerance_cm', 25.0)} cm or a recorded safety event",
                )

        if manifest_index + 1 < len(manifests):
            following = manifests[manifest_index + 1]
            following_geometry = following.get("observation_geometry") or {}
            following_agent = following_geometry.get("agent_position_cm")
            if action.get("type") == "turn_around" and action_executed:
                turn_steps += 1
                before_direction = geometry.get("agent_direction")
                after_direction = following_geometry.get("agent_direction")
                param = str(action.get("param") or "")
                if before_direction and after_direction and len(param) >= 2:
                    # UE heading increases toward the visual right, so L lowers it.
                    expected_delta = float(param[1:]) * (-1.0 if param[0] == "L" else 1.0)
                    actual_delta = signed_circular_delta(
                        heading_degrees(after_direction), heading_degrees(before_direction)
                    )
                    step_record["turn_heading_delta_deg"] = actual_delta
                    if abs(actual_delta - expected_delta) <= 8.0:
                        aligned_turn_steps += 1
                    else:
                        fail(step, "turn_label_matches_next_input_heading", actual_delta, expected_delta)
                else:
                    fail(step, "turn_heading_geometry_present", False, True)
                if agent is None or following_agent is None:
                    fail(step, "turn_stationary_geometry_present", False, True)
                elif not turn_execution or any(
                    turn_execution.get(key) is None
                    for key in (
                        "observation_position_cm",
                        "verified_end_position_cm",
                        "verified_position_drift_cm",
                    )
                ):
                    # Older rollouts predate the detailed turn-position fields.
                    # Their next observation occurs after another inference
                    # interval, so it cannot prove action-time stationarity.
                    # Heading alignment above remains independently auditable.
                    step_record["turn_stationary_source"] = (
                        "not_recorded_in_legacy_manifest"
                    )
                else:
                    stationary_turn_checks += 1
                    start_position = turn_execution.get(
                        "observation_position_cm"
                    )
                    verified_position = turn_execution.get(
                        "verified_end_position_cm"
                    )
                    verified_drift = turn_execution.get(
                        "verified_position_drift_cm"
                    )
                    input_start_error = distance(agent, start_position)
                    verified_start_error = distance(
                        start_position,
                        verified_position,
                    )
                    next_input_error = distance(
                        verified_position,
                        following_agent,
                    )
                    step_record["turn_input_to_recorded_observation_error_cm"] = (
                        input_start_error
                    )
                    step_record[
                        "turn_command_start_drift_from_input_cm"
                    ] = float(
                        (turn_execution or {}).get(
                            "command_start_position_drift_from_input_cm",
                            0.0,
                        )
                    )
                    step_record["turn_verified_position_drift_cm"] = float(
                        verified_drift
                    )
                    step_record["turn_to_next_input_position_error_cm"] = (
                        next_input_error
                    )
                    if (
                        input_start_error <= args.sequence_tolerance_cm
                        and verified_start_error <= args.sequence_tolerance_cm
                        and float(verified_drift) <= args.sequence_tolerance_cm
                        and next_input_error <= args.sequence_tolerance_cm
                    ):
                        stationary_turn_matches += 1
                    else:
                        fail(
                            step,
                            "turn_is_rotation_only_and_matches_next_input",
                            {
                                "input_start_error_cm": input_start_error,
                                "verified_start_error_cm": verified_start_error,
                                "verified_drift_cm": float(verified_drift),
                                "next_input_error_cm": next_input_error,
                            },
                            f"all <= {args.sequence_tolerance_cm} cm",
                        )
            if execution and execution.get("post_action_position_cm") and following_agent:
                sequential_checks += 1
                sequential_error = distance(execution["post_action_position_cm"], following_agent)
                step_record["next_input_position_error_cm"] = sequential_error
                if sequential_error > args.sequence_tolerance_cm:
                    fail(step, "post_action_position_matches_next_input", sequential_error, f"<= {args.sequence_tolerance_cm} cm")

            output_name = manifest.get("output_image")
            next_inputs = following.get("input_images") or []
            recorded_action_frames = manifest.get("recording_action_frames") or []
            next_history_inputs = next_inputs[:-1]
            if next_history_inputs:
                expected_history = retained_history_source_frames(
                    recorded_action_frames,
                    next_history_inputs,
                )
                if expected_history is None:
                    history_action_frame_unverifiable += len(
                        next_history_inputs
                    )
                    step_record["history_action_frame_source"] = (
                        "not_retained_by_sparse_artifact_policy"
                    )
                elif len(expected_history) != len(next_history_inputs):
                    fail(
                        step,
                        "recorded_action_frame_history_count",
                        len(next_history_inputs),
                        len(expected_history),
                    )
                history_maes_for_step = []
                for recorded_name, history_name in zip(
                    expected_history or [], next_history_inputs
                ):
                    history_mae = pixel_mae(
                        manifest["_path"].parent / recorded_name,
                        following["_path"].parent / history_name,
                    )
                    if history_mae is None:
                        fail(
                            step,
                            "recorded_action_frame_history_decodable",
                            False,
                            True,
                        )
                        continue
                    history_action_frame_checks += 1
                    history_pixel_maes.append(history_mae)
                    history_maes_for_step.append(history_mae)
                    if history_mae > 0.01:
                        fail(
                            step,
                            "recorded_action_frame_matches_next_history_input",
                            history_mae,
                            "<= 0.01 pixel MAE",
                        )
                if history_maes_for_step:
                    step_record[
                        "recorded_action_to_next_history_pixel_mae_0_to_255"
                    ] = history_maes_for_step
            if output_name and next_inputs:
                mae = pixel_mae(
                    manifest["_path"].parent / output_name,
                    following["_path"].parent / next_inputs[-1],
                )
                if mae is not None:
                    pixel_maes.append(mae)
                    step_record["output_to_next_input_pixel_mae_0_to_255"] = mae
        if int((manifest.get("collision_details") or {}).get("oil") or 0) > 0:
            oil_slowdown_pending = True
        step_records.append(step_record)

    report = {
        "passed": not failures,
        "source_step_directory": str(args.steps_dir),
        "manifest_count": len(manifests),
        "move_steps": move_steps,
        "meter_labels_unchanged": {"dot_1": "1m", "dots_2_to_4": "2m", "dots_5_to_7": "4m"},
        "constant_walk_speed_cm_per_second": WALK_SPEED_CM_PER_SECOND,
        "oil_slowed_move_steps": oil_slowed_move_steps,
        "unexecuted_action_steps": unexecuted_action_steps,
        "selected_dot_command_matches": move_steps - sum(1 for item in failures if item["check"] in {"selected_dot_matches_commanded_target", "runtime_selected_waypoint_matches_command"}),
        "reached_selected_dot_steps": reached_steps,
        "safety_explained_blocked_steps": safety_explained_misses,
        "sequential_position_checks": sequential_checks,
        "mean_output_to_next_input_pixel_mae_0_to_255": (
            sum(pixel_maes) / len(pixel_maes) if pixel_maes else None
        ),
        "history_action_frame_checks": history_action_frame_checks,
        "history_action_frame_unverifiable": (
            history_action_frame_unverifiable
        ),
        "turn_steps_with_following_observation": turn_steps,
        "turn_label_heading_matches": aligned_turn_steps,
        "turn_position_stationary_checks": stationary_turn_checks,
        "turn_position_stationary_matches": stationary_turn_matches,
        "mean_recorded_action_to_next_history_pixel_mae_0_to_255": (
            sum(history_pixel_maes) / len(history_pixel_maes)
            if history_pixel_maes else None
        ),
        "failure_count": len(failures),
        "failures": failures,
        "steps": step_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "steps"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
