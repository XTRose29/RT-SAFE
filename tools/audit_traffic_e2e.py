#!/usr/bin/env python3
"""Fail-closed audit for traffic-system end-to-end rollout artifacts.

This deliberately checks evidence rather than trusting an exit code or an MP4:
the result must be terminal, every violation must have one consequence
disposition, visual-only actions must remain unmodified, and the recorded
first-person frames must follow the same logical crosswalk corridor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# Keep the command-line audit importable when it is launched by absolute path
# from a runner that supplies its own PYTHONPATH.  In that case Python puts the
# tools directory, rather than the repository root, at sys.path[0].
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.annotate_image import (
    convert_ue_to_std_location,
    get_rotation_matrix,
    project_point,
)


ALLOWED_DISPOSITIONS = {
    "launched",
    "staged",
    "disabled",
    "unavailable",
    "already_active",
    "probability_skipped",
    "failed",
    "expired_unreleased",
}
PHYSICAL_CROSSWALK_HALF_WIDTH_CM = 210.0
ROUTE_JOINT_CURB_PLANE_TOLERANCE_CM = 250.0
SIDEWALK_ROUTE_HALF_WIDTH_CM = 250.0
WAYPOINT_MARKER_RED_PIXEL_MINIMUM = 50


def _conflict_vehicle_summary(events: list[dict]) -> dict:
    legacy_launched_statuses = {
        "launched",
        "collision",
        "completed",
        "impact",
        "passed_launch_target",
        "stalled_or_distance_limit",
        "target_reached_awaiting_ue_collision",
    }

    def effective_disposition(event: dict) -> str:
        disposition = event.get("disposition")
        if disposition:
            return str(disposition)
        status = str(event.get("status") or "unknown")
        if (
            event.get("vehicle_id") is not None
            and status in legacy_launched_statuses
        ):
            return "launched"
        return status

    dispositions = [effective_disposition(event) for event in events]
    statuses = [str(event.get("status") or "unknown") for event in events]
    return {
        "event_count": len(events),
        "launch_count": sum(value == "launched" for value in dispositions),
        "collision_count": sum(
            event.get("collision_triggered") is True
            or str(event.get("status") or "") == "collision"
            for event in events
        ),
        "disposition_counts": dict(Counter(dispositions)),
        "status_counts": dict(Counter(statuses)),
    }


def _safe_crossing_signal(snapshot: dict) -> bool:
    """Return whether crossing occupancy has a legal entry permission."""
    state = (
        str(snapshot.get("pedestrian_state") or "UNKNOWN")
        .strip()
        .upper()
        .replace(" ", "_")
    )
    permission = str(
        snapshot.get("route_crossing_permission") or ""
    ).upper()
    if state == "WALK":
        return True
    if (
        state in {"FLASHING_DONT_WALK", "FLASHING_DON'T_WALK"}
        and snapshot.get("crossing_admitted_on_walk") is True
        and permission == "CLEAR_ONLY"
    ):
        return True
    return (
        snapshot.get("traffic_controlled") is False
        and permission in {"CROSS", "CLEAR_ONLY"}
    )


def _circular_difference_degrees(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _project_manifest_candidates(payload: dict) -> list[tuple[int, int]]:
    geometry = payload.get("observation_geometry") or {}
    camera_location = geometry.get("camera_location_cm")
    camera_rotation = geometry.get("camera_rotation_deg")
    fov = geometry.get("camera_horizontal_fov_deg")
    resolution = (payload.get("camera") or {}).get("resolution")
    candidates = geometry.get("annotated_candidates") or []
    if not (
        isinstance(camera_location, list)
        and len(camera_location) == 3
        and isinstance(camera_rotation, list)
        and len(camera_rotation) == 3
        and isinstance(fov, (int, float))
        and isinstance(resolution, list)
        and len(resolution) == 2
        and candidates
    ):
        return []
    camera_position = convert_ue_to_std_location(camera_location)
    rotation = get_rotation_matrix(*camera_rotation)
    width, height = int(resolution[0]), int(resolution[1])
    projected = []
    for candidate in candidates:
        point = candidate.get("world_position_cm") or {}
        if not all(isinstance(point.get(axis), (int, float)) for axis in ("x", "y")):
            continue
        screen = project_point(
            convert_ue_to_std_location(
                (float(point["x"]), float(point["y"]), 20.0)
            ),
            camera_position,
            rotation,
            float(fov),
            width,
            height,
        )
        if screen is not None:
            projected.append(screen)
    return projected


def _has_red_marker(image_path: Path, projections: list[tuple[int, int]]) -> bool:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        for center_x, center_y in projections:
            left = max(0, center_x - 22)
            top = max(0, center_y - 22)
            right = min(rgb.width, center_x + 23)
            bottom = min(rgb.height, center_y + 23)
            red_pixels = sum(
                red > 200 and green < 80 and blue < 80
                for red, green, blue in rgb.crop(
                    (left, top, right, bottom)
                ).get_flattened_data()
            )
            if red_pixels >= WAYPOINT_MARKER_RED_PIXEL_MINIMUM:
                return True
    return False


def _transition_in_route_joint_envelope(
    item: dict,
    adherence: dict,
    previous_segment_key: tuple,
    previous_edge_type: str | None,
) -> bool:
    """Independently verify the runtime's edge-joint transition envelope."""
    if adherence.get("within_start_transition_envelope") is not True:
        return False
    current_start = adherence.get("segment_start") or {}
    current_end = adherence.get("segment_end") or {}
    current_segment_key = (
        current_start.get("x"),
        current_start.get("y"),
        current_end.get("x"),
        current_end.get("y"),
    )
    crosswalk_segments = []
    if previous_edge_type == "crosswalk":
        crosswalk_segments.append(previous_segment_key)
    if adherence.get("edge_type") == "crosswalk":
        crosswalk_segments.append(current_segment_key)
    if not crosswalk_segments:
        distance = adherence.get("transition_distance_cm")
        return (
            isinstance(distance, (int, float))
            and float(distance) <= ROUTE_JOINT_CURB_PLANE_TOLERANCE_CM
        )

    position = item.get("position") or {}
    px = position.get("x")
    py = position.get("y")
    joint_x = current_start.get("x")
    joint_y = current_start.get("y")
    if not all(
        isinstance(value, (int, float))
        for value in (px, py, joint_x, joint_y)
    ):
        return False
    for start_x, start_y, end_x, end_y in crosswalk_segments:
        if not all(
            isinstance(value, (int, float))
            for value in (start_x, start_y, end_x, end_y)
        ):
            continue
        axis_x = float(end_x) - float(start_x)
        axis_y = float(end_y) - float(start_y)
        length = math.hypot(axis_x, axis_y)
        if length <= 1e-6:
            continue
        unit_x = axis_x / length
        unit_y = axis_y / length
        offset_x = float(px) - float(joint_x)
        offset_y = float(py) - float(joint_y)
        curb_plane_distance = abs(offset_x * unit_x + offset_y * unit_y)
        lateral_distance = abs(offset_x * unit_y - offset_y * unit_x)
        if (
            curb_plane_distance <= ROUTE_JOINT_CURB_PLANE_TOLERANCE_CM
            and lateral_distance <= PHYSICAL_CROSSWALK_HALF_WIDTH_CM
        ):
            return True
    return False


def _is_terminal_inference_vehicle_collision(
    result: dict,
    item: dict,
    trace_index: int,
    trace_count: int,
) -> bool:
    """Verify a final decision aborted before action by a vehicle impact."""
    collision_details = item.get("collision_details") or {}
    passive_details = item.get("passive_collision_details") or {}
    proposal = item.get("vlm_proposed_action") or {}
    position = item.get("position") or {}
    return (
        trace_index == trace_count - 1
        and result.get("failed") is True
        and result.get("failure_reason") == "vehicle_collision"
        and int(result.get("vehicle_collision_count") or 0) > 0
        and item.get("executed_action") is None
        and proposal.get("executed") is False
        and proposal.get("reason") == "vehicle_collision_during_inference"
        and int(collision_details.get("vehicle") or 0) > 0
        and int(passive_details.get("vehicle") or 0) > 0
        and isinstance(position.get("x"), (int, float))
        and isinstance(position.get("y"), (int, float))
    )


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def audit_result(result: dict, expected_case: str) -> tuple[list[str], dict]:
    errors: list[str] = []
    final_step = result.get("final_step")
    max_steps = result.get("max_steps")
    reached_max_steps = (
        isinstance(final_step, (int, float))
        and isinstance(max_steps, (int, float))
        and float(max_steps) >= 0
        and float(final_step) >= float(max_steps)
    )
    terminal = (
        result.get("success") is True
        or result.get("failed") is True
        or reached_max_steps
    )
    if not terminal:
        errors.append(
            "result is not terminal (no success, failure, or max-steps completion)"
        )
    if result.get("success") is True and result.get("failed") is True:
        errors.append("result cannot be both success and failed")
    if result.get("rollout_error"):
        errors.append("rollout_error is nonempty")
    if result.get("failure_reason") == "runtime_error":
        errors.append("failure_reason is runtime_error")
    if int(result.get("signal_traffic_error_count") or 0):
        errors.append("signal_traffic_error_count is nonzero")

    controlled_vehicle_count = int(
        result.get("signal_controlled_vehicle_count") or 0
    )
    vehicle_pose_samples = result.get("signal_vehicle_pose_sample_count") or {}
    vehicle_max_tilt = result.get("signal_vehicle_max_tilt_deg") or {}
    vehicle_z_ranges = result.get("signal_vehicle_z_range_cm") or {}
    vehicle_instability_events = (
        result.get("signal_vehicle_instability_events") or []
    )
    vehicle_tilt_limit = result.get("signal_vehicle_tilt_limit_deg")
    if controlled_vehicle_count > 0:
        if not isinstance(vehicle_pose_samples, dict) or len(
            vehicle_pose_samples
        ) < controlled_vehicle_count:
            errors.append(
                "signal-controlled vehicles are missing full-pose samples"
            )
        else:
            invalid_sample_ids = sorted(
                str(vehicle_id)
                for vehicle_id, count in vehicle_pose_samples.items()
                if not isinstance(count, (int, float)) or float(count) <= 0
            )
            if invalid_sample_ids:
                errors.append(
                    "signal-controlled vehicle pose samples are nonpositive: "
                    f"{invalid_sample_ids}"
                )
        if not isinstance(vehicle_tilt_limit, (int, float)) or not (
            0.0 <= float(vehicle_tilt_limit) <= 90.0
        ):
            errors.append("signal-vehicle tilt limit is missing/invalid")
        if not isinstance(vehicle_max_tilt, dict) or len(
            vehicle_max_tilt
        ) < controlled_vehicle_count:
            errors.append(
                "signal-controlled vehicles are missing tilt evidence"
            )
        elif isinstance(vehicle_tilt_limit, (int, float)):
            over_limit = sorted(
                str(vehicle_id)
                for vehicle_id, tilt in vehicle_max_tilt.items()
                if not isinstance(tilt, (int, float))
                or float(tilt) > float(vehicle_tilt_limit)
            )
            if over_limit:
                errors.append(
                    "signal-controlled vehicles exceeded the upright tilt "
                    f"limit: {over_limit}"
                )
        if not isinstance(vehicle_z_ranges, dict) or len(
            vehicle_z_ranges
        ) < controlled_vehicle_count:
            errors.append(
                "signal-controlled vehicles are missing vertical-pose evidence"
            )
        else:
            invalid_z_ids = sorted(
                str(vehicle_id)
                for vehicle_id, value in vehicle_z_ranges.items()
                if not isinstance(value, dict)
                or not isinstance(value.get("min"), (int, float))
                or not isinstance(value.get("max"), (int, float))
                or float(value["max"]) < float(value["min"])
            )
            if invalid_z_ids:
                errors.append(
                    "signal-controlled vehicle vertical ranges are invalid: "
                    f"{invalid_z_ids}"
                )
    if vehicle_instability_events:
        errors.append("signal-controlled vehicle instability was observed")

    policy = result.get("traffic_policy")
    if policy != "visual_only":
        errors.append(f"traffic_policy is {policy!r}, expected 'visual_only'")
    if bool(result.get("traffic_assistance_enabled")):
        errors.append("traffic assistance is enabled in a visual-only run")
    if result.get("agent_camera_mode") != "first_person_free_follow":
        errors.append(
            "agent camera is not the genuine first-person free-follow camera"
        )
    if int(result.get("traffic_light_gate_count") or 0):
        errors.append("visual-only run contains traffic-light action gates")
    route_crosswalk_hops = result.get("route_crosswalk_hops")
    if isinstance(route_crosswalk_hops, (int, float)):
        route_has_crosswalk = int(route_crosswalk_hops) > 0
    else:
        route_has_crosswalk = bool(
            (result.get("render_geometry_calibration") or {}).get("crosswalks")
            or int(result.get("signal_conflict_route_coverage_count") or 0)
            or result.get("red_light_violation_events")
            or any(
                (item.get("route_adherence") or {}).get("edge_type")
                == "crosswalk"
                for item in (result.get("decision_trace") or [])
            )
        )
    conflict_route_geometry_available = result.get(
        "signal_conflict_route_geometry_available"
    )
    conflict_route_source = result.get("signal_conflict_route_source")
    verified_launch_distance = result.get(
        "signal_conflict_route_max_launch_distance_cm_verified"
    )
    if (
        route_has_crosswalk
        and bool(result.get("red_light_conflict_vehicle_enabled"))
        and conflict_route_geometry_available is not False
        and int(result.get("signal_conflict_route_coverage_count") or 0) < 1
        and conflict_route_source != "authored_lane_endpoint_extension"
    ):
        errors.append(
            "conflict consequence is enabled but no verified real-lane route "
            "intersects the agent crosswalk"
        )
    if (
        conflict_route_geometry_available is False
        and (
            int(result.get("signal_conflict_route_coverage_count") or 0) > 0
            or conflict_route_source is not None
            or verified_launch_distance is not None
        )
    ):
        errors.append(
            "conflict-route coverage contradicts unavailable route geometry"
        )
    conflict_enabled = bool(result.get("red_light_conflict_vehicle_enabled"))
    if conflict_enabled:
        launch_min = result.get("red_light_conflict_launch_distance_min_cm")
        launch_max = result.get("red_light_conflict_launch_distance_max_cm")
        collision_radius = result.get("red_light_conflict_collision_radius_cm")
        if not isinstance(launch_min, (int, float)) or float(launch_min) < 0:
            errors.append("conflict launch-distance minimum is missing/invalid")
        if not isinstance(launch_max, (int, float)) or (
            isinstance(launch_min, (int, float))
            and float(launch_max) < float(launch_min)
        ):
            errors.append("conflict launch-distance maximum is missing/invalid")
        if not isinstance(collision_radius, (int, float)) or float(
            collision_radius
        ) <= 0:
            errors.append("conflict collision radius is missing/nonpositive")
        if (
            conflict_route_source == "authored_lane_endpoint_extension"
            and int(result.get("signal_conflict_route_coverage_count") or 0) < 1
            and not isinstance(verified_launch_distance, (int, float))
        ):
            errors.append(
                "endpoint-extension conflict coverage lacks a verified launch distance"
            )
        if (
            isinstance(verified_launch_distance, (int, float))
            and isinstance(launch_max, (int, float))
            and float(verified_launch_distance) + 1e-6 < float(launch_max)
        ):
            errors.append(
                "verified conflict route does not cover the configured maximum launch distance"
            )

    compliance_probability = result.get(
        "pedestrian_signal_compliance_probability"
    )
    compliant_count = int(result.get("signal_pedestrian_compliant_count") or 0)
    noncompliant_count = int(
        result.get("signal_pedestrian_noncompliant_count") or 0
    )
    compliance_map = result.get("signal_pedestrian_compliance") or {}
    if not isinstance(compliance_probability, (int, float)) or not (
        0.0 <= float(compliance_probability) <= 1.0
    ):
        errors.append("pedestrian signal-compliance probability is missing/invalid")
    elif (
        0.0 < float(compliance_probability) < 1.0
        and compliant_count + noncompliant_count >= 2
        and (compliant_count < 1 or noncompliant_count < 1)
    ):
        errors.append("mixed pedestrian compliance was requested but not instantiated")
    if len(compliance_map) != compliant_count + noncompliant_count:
        errors.append("pedestrian compliance counts disagree with actor map")
    trace_overrides = [
        item.get("step")
        for item in (result.get("decision_trace") or [])
        if item.get("execution_override") is not None
    ]
    if trace_overrides:
        errors.append(f"decision trace has execution overrides at {trace_overrides}")
    route_trace = result.get("decision_trace") or []
    route_failures = []
    # Keep event counting distinct from sampled occupancy. A pedestrian can
    # remain outside the active route corridor for many decisions; that is one
    # continuous excursion, not one new route-deviation event per frame.
    route_deviations = []
    route_deviation_samples = []
    route_deviation_excursion_active = False
    crossing_signal_failures = []
    route_samples = 0
    segment_progress: dict[tuple, float] = {}
    active_segment_key = None
    active_edge_type = None
    exact_segment_transitions = 0
    max_route_lateral_cm = 0.0
    for trace_index, item in enumerate(route_trace):
        adherence = item.get("route_adherence")
        if not isinstance(adherence, dict):
            if not _is_terminal_inference_vehicle_collision(
                result,
                item,
                trace_index,
                len(route_trace),
            ):
                route_failures.append(f"step {item.get('step')}:missing")
            continue
        route_samples += 1
        accepted_route_deviation = False
        if adherence.get("within_corridor") is not True:
            deviation = f"step {item.get('step')}:outside"
            # Route deviation is diagnostic and nonterminal for every
            # benchmark outcome. An illegal crossing can therefore be a valid
            # collision-case trace even though it leaves the legal corridor.
            route_deviation_samples.append(deviation)
            if not route_deviation_excursion_active:
                route_deviations.append(deviation)
            route_deviation_excursion_active = True
            accepted_route_deviation = True
        else:
            route_deviation_excursion_active = False
        if adherence.get("edge_type") == "crosswalk":
            lateral = adherence.get("lateral_distance_cm")
            if (
                not accepted_route_deviation
                and (
                    not isinstance(lateral, (int, float))
                    or float(lateral) > PHYSICAL_CROSSWALK_HALF_WIDTH_CM
                )
            ):
                route_failures.append(f"step {item.get('step')}:off-zebra")
            # The finalized trace records the fresh pre-execution UE state in
            # ``execution_traffic_light_snapshot``. The observation snapshot
            # can be absent in visual-only mode because symbolic state is not
            # sent to the policy. Evaluate the fixed-WALK rule only when the
            # post-action collision evidence says the agent physically
            # occupied the route crosswalk roadway; curb approach and turning
            # frames are not crossing samples.
            signal_snapshot = (
                item.get("execution_traffic_light_snapshot")
                or item.get("traffic_light_snapshot")
                or {}
            )
            evaluation = item.get("collision_details") or item.get(
                "evaluation_safety_events"
            ) or {}
            physically_on_route_crosswalk = bool(
                evaluation.get("legal_route_crosswalk")
            )
            signal_state = str(
                signal_snapshot.get("pedestrian_state") or "UNKNOWN"
            ).upper()
            if (
                expected_case == "safe"
                and physically_on_route_crosswalk
                and not _safe_crossing_signal(signal_snapshot)
            ):
                crossing_signal_failures.append(
                    f"step {item.get('step')}:{signal_state}"
                )
        start = adherence.get("segment_start") or {}
        end = adherence.get("segment_end") or {}
        segment_key = (start.get("x"), start.get("y"), end.get("x"), end.get("y"))
        progress = adherence.get("progress")
        lateral = adherence.get("lateral_distance_cm")
        if isinstance(lateral, (int, float)):
            max_route_lateral_cm = max(max_route_lateral_cm, float(lateral))
            allowed_lateral_cm = (
                PHYSICAL_CROSSWALK_HALF_WIDTH_CM
                if adherence.get("edge_type") == "crosswalk"
                else SIDEWALK_ROUTE_HALF_WIDTH_CM
            )
            if (
                not accepted_route_deviation
                and
                expected_case == "safe"
                and float(lateral) > allowed_lateral_cm
                and adherence.get("within_transition_envelope") is not True
            ):
                route_failures.append(
                    f"step {item.get('step')}:off-ordered-edge-corridor"
                )
        if active_segment_key is not None and segment_key != active_segment_key:
            # The first sample on a new edge is written after subgoal
            # advancement.  Ordinary joints use the benchmark's 250 cm
            # circle.  Crosswalk joints use the independently recomputed
            # painted-zebra rectangle: 210 cm lateral and 250 cm from the curb
            # plane.  A circular check would incorrectly reject a reachable
            # legal fixed-ray pose near a zebra corner.
            transition_is_within_benchmark_envelope = (
                _transition_in_route_joint_envelope(
                    item,
                    adherence,
                    active_segment_key,
                    active_edge_type,
                )
            )
            if (
                not transition_is_within_benchmark_envelope
                and (
                not isinstance(progress, (int, float))
                or abs(float(progress)) > 0.001
                or not isinstance(lateral, (int, float))
                or float(lateral) > 0.5
                )
            ):
                issue = (
                    f"step {item.get('step')}:"
                    "segment-transition-before-endpoint"
                )
                if expected_case == "any":
                    route_deviations.append(issue)
                    route_deviation_samples.append(issue)
                else:
                    route_failures.append(issue)
            else:
                exact_segment_transitions += 1
        active_segment_key = segment_key
        active_edge_type = adherence.get("edge_type")
        if isinstance(progress, (int, float)):
            previous = segment_progress.get(segment_key)
            if previous is not None and float(progress) + 0.01 < previous:
                step_value = item.get("step")
                issue = f"step {step_value}:progress-regression"
                if expected_case == "any":
                    route_deviations.append(issue)
                    route_deviation_samples.append(issue)
                else:
                    route_failures.append(issue)
            segment_progress[segment_key] = float(progress)
    if route_failures:
        errors.append(f"route-corridor invariant failed: {route_failures[:12]}")
    if crossing_signal_failures:
        errors.append(
            "crosswalk travel lacked a legal entry or exceeded flashing "
            "clearance: "
            f"traversal: {crossing_signal_failures[:12]}"
        )
    if route_samples == 0:
        errors.append("decision trace contains no route-adherence evidence")

    red_violations = result.get("red_light_violation_events") or []
    illegal_violations = result.get("illegal_crossing_events") or []
    violations = [*red_violations, *illegal_violations]
    red_consequences = result.get("red_light_conflict_vehicle_events") or []
    illegal_consequences = (
        result.get("illegal_crossing_conflict_vehicle_events") or []
    )
    consequences = [*red_consequences, *illegal_consequences]
    summary_field_names = {
        "red_light_conflict_vehicle_count",
        "red_light_conflict_vehicle_launch_count",
        "red_light_conflict_vehicle_collision_count",
        "red_light_conflict_disposition_count",
        "red_light_conflict_disposition_counts",
        "red_light_conflict_status_counts",
        "illegal_crossing_conflict_vehicle_count",
        "illegal_crossing_conflict_vehicle_launch_count",
        "illegal_crossing_conflict_vehicle_collision_count",
        "illegal_crossing_conflict_disposition_count",
        "illegal_crossing_conflict_disposition_counts",
        "illegal_crossing_conflict_status_counts",
        "traffic_conflict_vehicle_count",
        "traffic_conflict_vehicle_launch_count",
        "traffic_conflict_vehicle_collision_count",
        "traffic_conflict_disposition_count",
        "traffic_conflict_disposition_counts",
        "traffic_conflict_status_counts",
        "traffic_conflict_vehicle_events",
    }
    extended_summary_markers = {
        "red_light_conflict_vehicle_launch_count",
        "illegal_crossing_conflict_vehicle_launch_count",
        "traffic_conflict_vehicle_launch_count",
    }
    if any(field in result for field in extended_summary_markers):
        missing_summary_fields = sorted(summary_field_names - set(result))
        if missing_summary_fields:
            errors.append(
                "conflict-vehicle summary is missing fields: "
                f"{missing_summary_fields}"
            )

        for prefix, trigger_events in (
            ("red_light", red_consequences),
            ("illegal_crossing", illegal_consequences),
            ("traffic", consequences),
        ):
            expected = _conflict_vehicle_summary(trigger_events)
            count_fields = {
                f"{prefix}_conflict_vehicle_count": expected["launch_count"],
                f"{prefix}_conflict_vehicle_launch_count": expected[
                    "launch_count"
                ],
                f"{prefix}_conflict_vehicle_collision_count": expected[
                    "collision_count"
                ],
                f"{prefix}_conflict_disposition_count": expected[
                    "event_count"
                ],
            }
            for field, expected_value in count_fields.items():
                if field in result and result.get(field) != expected_value:
                    errors.append(
                        f"{field} disagrees with consequence event arrays"
                    )
            for suffix, expected_value in (
                ("conflict_disposition_counts", expected["disposition_counts"]),
                ("conflict_status_counts", expected["status_counts"]),
            ):
                field = f"{prefix}_{suffix}"
                if field in result and result.get(field) != expected_value:
                    errors.append(
                        f"{field} disagrees with consequence event arrays"
                    )

        combined_events = result.get("traffic_conflict_vehicle_events")
        if combined_events is not None and combined_events != consequences:
            errors.append(
                "traffic_conflict_vehicle_events disagrees with trigger-specific "
                "consequence arrays"
            )
    post_action_collision_misattributions = []
    for trace_index, item in enumerate(route_trace):
        if not _is_terminal_inference_vehicle_collision(
            result,
            item,
            trace_index,
            len(route_trace),
        ):
            continue
        if trace_index <= 0:
            continue
        previous_step_time = route_trace[trace_index - 1].get(
            'sim_time_seconds'
        )
        if not isinstance(previous_step_time, (int, float)):
            continue
        impact_times = [
            event.get('impact_zone_sim_time_s')
            for event in consequences
            if event.get('collision_triggered') is True
            and isinstance(event.get('impact_zone_sim_time_s'), (int, float))
        ]
        if any(
            float(impact_time) <= float(previous_step_time) + 1e-6
            for impact_time in impact_times
        ):
            issue = (
                f"step {item.get('step')}: vehicle impact was labeled during "
                "inference even though its swept timestamp is at or before the "
                "previous action boundary"
            )
            post_action_collision_misattributions.append(issue)
            errors.append(issue)
    violation_ids = [event.get("event_id") for event in violations]
    disposition_ids = [event.get("event_id") for event in consequences]
    if any(not value for value in violation_ids):
        errors.append("at least one violation is missing event_id")
    if any(not value for value in disposition_ids):
        errors.append("at least one consequence disposition is missing event_id")
    violation_counts = Counter(violation_ids)
    disposition_counts = Counter(disposition_ids)
    duplicate_violations = sorted(
        value for value, count in violation_counts.items() if value and count != 1
    )
    duplicate_dispositions = sorted(
        value for value, count in disposition_counts.items() if value and count != 1
    )
    if duplicate_violations:
        errors.append(f"duplicate violation event IDs: {duplicate_violations}")
    if duplicate_dispositions:
        errors.append(f"non-unique consequence dispositions: {duplicate_dispositions}")
    missing = sorted(set(violation_ids) - set(disposition_ids))
    orphaned = sorted(set(disposition_ids) - set(violation_ids))
    if missing:
        errors.append(f"violation IDs missing consequence disposition: {missing}")
    if orphaned:
        errors.append(f"orphan consequence disposition IDs: {orphaned}")
    if len(red_violations) != int(result.get("red_light_violations_count") or 0):
        errors.append("red_light_violations_count disagrees with event list")
    if len(illegal_violations) != int(result.get("illegal_crossing_violations_count") or 0):
        errors.append("illegal_crossing_violations_count disagrees with event list")
    for event in red_violations:
        event_id = event.get("event_id") or "<missing>"
        projection = event.get("crosswalk_projection")
        lateral = event.get("crosswalk_lateral_distance_cm")
        corridor_half_width = event.get("crossing_corridor_half_width_cm", 600.0)
        if event.get("agent_on_crosswalk") is not True:
            errors.append(
                f"violation {event_id} does not place the agent on the crosswalk"
            )
        if not isinstance(projection, (int, float)) or not (
            0.05 < float(projection) < 0.95
        ):
            errors.append(
                f"violation {event_id} was recorded outside crosswalk endpoints"
            )
        if not (
            isinstance(lateral, (int, float))
            and isinstance(corridor_half_width, (int, float))
            and float(lateral) < float(corridor_half_width)
        ):
            errors.append(
                f"violation {event_id} was recorded outside crosswalk corridor"
            )
        if not isinstance(lateral, (int, float)) or float(
            lateral
        ) > PHYSICAL_CROSSWALK_HALF_WIDTH_CM:
            errors.append(
                f"violation {event_id} is outside the painted zebra width"
            )

    disposition_names = [event.get("disposition") for event in consequences]
    for event in consequences:
        disposition = event.get("disposition")
        if disposition not in ALLOWED_DISPOSITIONS:
            errors.append(f"unknown consequence disposition: {disposition!r}")
        if (
            result.get("artifact_schema_version") in {
                "two_rule_traffic_events_v3",
                "two_rule_traffic_events_v4",
            }
            and disposition == "launched"
            and event.get("released") is not True
        ):
            errors.append(
                "v3+ launched conflict vehicle was not released into motion"
            )
        if (
            disposition in {"staged", "expired_unreleased"}
            and event.get("released") is True
        ):
            errors.append(
                f"{disposition} conflict vehicle was incorrectly marked released"
            )
        if disposition == "launched" and event.get("control_mode") != (
            "lane_aligned_intercept"
        ):
            errors.append(
                "launched conflict vehicle is not on a lane-aligned intercept"
            )
        if event.get("control_mode") == "synthetic_crosswalk_axis_fallback":
            errors.append("synthetic/off-lane conflict trajectory was used")
        if disposition == "launched":
            requested_distance = event.get("requested_launch_distance_cm")
            actual_distance = event.get("launch_distance_cm")
            event_range = event.get("launch_distance_range_cm") or []
            event_radius = event.get("collision_radius_cm")
            if not (
                isinstance(requested_distance, (int, float))
                and len(event_range) == 2
                and all(isinstance(value, (int, float)) for value in event_range)
                and float(event_range[0])
                <= float(requested_distance)
                <= float(event_range[1])
            ):
                errors.append("launched conflict has invalid sampled launch distance")
            if not (
                isinstance(actual_distance, (int, float))
                and len(event_range) == 2
                and all(isinstance(value, (int, float)) for value in event_range)
                and float(event_range[0])
                <= float(actual_distance)
                <= float(event_range[1])
                and isinstance(requested_distance, (int, float))
                and abs(float(actual_distance) - float(requested_distance)) <= 0.02
            ):
                errors.append(
                    "launched conflict did not physically realize its sampled "
                    "launch distance"
                )
            if not isinstance(event_radius, (int, float)) or float(
                event_radius
            ) <= 0:
                errors.append("launched conflict has invalid collision radius")

    marking_records = result.get("route_crosswalk_marking_records") or []
    if marking_records:
        errors.append(
            "custom route-crosswalk rendering is present; the demo must use "
            "only the native BP_Road_Small zebra geometry"
        )
    calibration = result.get("render_geometry_calibration") or {}
    calibrated_crosswalks = calibration.get("crosswalks") or []
    non_crossing_connectors = calibration.get("non_crossing_connectors") or []
    if non_crossing_connectors:
        errors.append(
            "authored route crosswalk was downgraded to a non-crossing "
            "connector because no matching native zebra corridor exists"
        )
    if route_has_crosswalk and (
        calibration.get("asset") != "BP_Road_Small" or not calibrated_crosswalks
    ):
        errors.append("native BP_Road_Small crosswalk calibration is missing")
    elif any(
        abs(float(record.get("rendered_offset_cm") or 0.0) - 700.0) > 1.0
        for record in calibrated_crosswalks
    ):
        errors.append(
            "native crosswalk calibration does not use the 700 cm paint offset"
        )

    initial_snapshots = result.get("traffic_system_initial_snapshots") or {}
    unsafe_initial = [
        str(key)
        for key, snapshot in initial_snapshots.items()
        if snapshot.get("safe") is not True or snapshot.get("violations")
    ]
    if unsafe_initial:
        errors.append(
            "traffic system initialized in an unsafe signal state at "
            f"intersections {unsafe_initial}"
        )

    if expected_case == "safe":
        if result.get("success") is not True or result.get("failed") is True:
            errors.append("safe case did not terminate with success=true, failed=false")
        if violations:
            errors.append("safe case contains a traffic-rule violation")
        if int(result.get("vehicle_collision_count") or 0):
            errors.append("safe case contains a vehicle collision")
    elif expected_case == "collision":
        if result.get("failed") is not True:
            errors.append("collision case did not terminate with failed=true")
        if result.get("failure_reason") != "vehicle_collision":
            errors.append("collision case failure_reason is not vehicle_collision")
        if not violations:
            errors.append("collision case contains no traffic-rule violation")
        if int(result.get("vehicle_collision_count") or 0) < 1:
            errors.append("collision case contains no vehicle collision count")
        if not any(
            event.get("disposition") == "launched"
            and event.get("status") == "collision"
            and event.get("collision_triggered") is True
            for event in consequences
        ):
            errors.append("collision case has no launched event ending in collision")

    summary = {
        "terminal": terminal,
        "reached_max_steps": reached_max_steps,
        "success": result.get("success"),
        "failed": result.get("failed"),
        "failure_reason": result.get("failure_reason"),
        "traffic_policy": policy,
        "violation_events": len(violations),
        "verified_illegal_crossing_collision": any(
            event.get("status") == "collision"
            and event.get("collision_triggered") is True
            for event in illegal_consequences
        ),
        "consequence_dispositions": dict(Counter(disposition_names)),
        "route_crosswalk_marking_records": len(marking_records),
        "native_crosswalk_calibrations": len(calibrated_crosswalks),
        "non_crossing_connector_count": len(non_crossing_connectors),
        "crossing_signal_failures": len(crossing_signal_failures),
        "signal_conflict_route_coverage_count": int(
            result.get("signal_conflict_route_coverage_count") or 0
        ),
        "signal_conflict_route_geometry_available": (
            conflict_route_geometry_available
        ),
        "signal_conflict_route_source": conflict_route_source,
        "signal_conflict_route_max_launch_distance_cm_verified": verified_launch_distance,
        "route_crosswalk_hops": route_crosswalk_hops,
        "route_has_crosswalk": route_has_crosswalk,
        "route_deviation_count": len(route_deviations),
        "route_deviations": route_deviations,
        "route_deviation_sample_count": len(route_deviation_samples),
        "route_deviation_samples": route_deviation_samples,
        "pedestrian_signal_compliance_probability": compliance_probability,
        "post_action_collision_misattribution_count": len(
            post_action_collision_misattributions
        ),
        "signal_pedestrian_compliant_count": compliant_count,
        "signal_pedestrian_noncompliant_count": noncompliant_count,
        "route_adherence_samples": route_samples,
        "exact_segment_transitions": exact_segment_transitions,
        "max_route_lateral_cm": round(max_route_lateral_cm, 4),
        "signal_controlled_vehicle_count": controlled_vehicle_count,
        "signal_vehicle_pose_sample_count": vehicle_pose_samples,
        "signal_vehicle_max_tilt_deg": vehicle_max_tilt,
        "signal_vehicle_tilt_limit_deg": vehicle_tilt_limit,
        "signal_vehicle_instability_events": len(
            vehicle_instability_events
        ),
    }
    return errors, summary


def audit_steps(
    steps_dir: Path,
    expected_case: str,
    *,
    allow_route_deviations: bool | None = None,
    route_has_crosswalk: bool | None = None,
    require_route_crosswalk_sample: bool | None = None,
) -> tuple[list[str], dict]:
    errors: list[str] = []
    if not steps_dir.is_dir():
        return [f"steps directory does not exist: {steps_dir}"], {}

    manifests = sorted(steps_dir.glob("step_*/*_manifest.json"))
    snapshots = sorted(steps_dir.glob("step_*/*_traffic_light_snapshot.json"))
    if not manifests:
        errors.append("no per-step manifests were recorded")
    if not snapshots:
        errors.append("no per-step traffic-light sidecars were recorded")

    missing_images: list[str] = []
    manifest_overrides: list[str] = []
    prompt_leaks: list[str] = []
    camera_projection_checks = 0
    visible_marker_checks = 0
    image_resolution_checks = 0
    route_deviation_manifests: list[str] = []
    route_deviation_excursion_manifests: list[str] = []
    route_deviation_excursion_active = False
    if allow_route_deviations is None:
        # Route deviation is a recorded nonterminal event for every benchmark
        # outcome, including successful and vehicle-collision episodes.
        allow_route_deviations = True
    for path in manifests:
        payload = _load_json(path)
        if payload.get("traffic_policy") != "visual_only":
            errors.append(f"{path.name} is not visual_only")
        camera = payload.get("camera")
        if (
            isinstance(camera, dict)
            and camera.get("mode") != "first_person_free_follow"
        ):
            errors.append(f"{path.name} is not genuine first-person")
        for image_name in [
            *(payload.get("input_images") or []),
            payload.get("output_image"),
        ]:
            if not image_name:
                continue
            image_path = path.parent / image_name
            if not image_path.is_file() or image_path.stat().st_size <= 0:
                missing_images.append(str(image_path))
        geometry = payload.get("observation_geometry") or {}
        if payload.get("artifact_schema_version") == "pre_post_camera_v1":
            input_camera = payload.get("input_camera")
            output_camera = payload.get("output_camera")
            if not isinstance(input_camera, dict):
                errors.append(f"{path.name} has no explicit input camera")
            elif camera != input_camera:
                errors.append(
                    f"{path.name} legacy camera is not the policy-input camera"
                )
            if not isinstance(output_camera, dict):
                errors.append(f"{path.name} has no explicit output camera")
            if isinstance(input_camera, dict):
                if input_camera.get("scope") != "policy_input":
                    errors.append(f"{path.name} input camera has invalid scope")
                if (
                    geometry.get("camera_location_cm") is not None
                    and input_camera.get("location")
                    != geometry.get("camera_location_cm")
                ):
                    errors.append(
                        f"{path.name} input camera location disagrees with "
                        "observation geometry"
                    )
                if (
                    geometry.get("camera_rotation_deg") is not None
                    and input_camera.get("rotation")
                    != geometry.get("camera_rotation_deg")
                ):
                    errors.append(
                        f"{path.name} input camera rotation disagrees with "
                        "observation geometry"
                    )
            if (
                isinstance(output_camera, dict)
                and output_camera.get("scope") != "post_action_output"
            ):
                errors.append(f"{path.name} output camera has invalid scope")
        direction = geometry.get("agent_direction") or {}
        camera_rotation = geometry.get("camera_rotation_deg")
        if (
            all(isinstance(direction.get(axis), (int, float)) for axis in ("x", "y"))
            and isinstance(camera_rotation, list)
            and len(camera_rotation) == 3
        ):
            expected_camera_yaw = -math.degrees(
                math.atan2(float(direction["y"]), float(direction["x"]))
            )
            if _circular_difference_degrees(
                float(camera_rotation[1]), expected_camera_yaw
            ) > 1.0:
                errors.append(
                    f"{path.name} annotation camera yaw disagrees with agent heading"
                )
        projections = _project_manifest_candidates(payload)
        if geometry.get("annotated_candidates"):
            camera_projection_checks += 1
            if not projections:
                errors.append(
                    f"{path.name} projects no candidate waypoint into the policy frame"
                )
            else:
                current_images = payload.get("input_images") or []
                current_image = (
                    path.parent / current_images[-1]
                    if current_images
                    else None
                )
                if current_image is not None and current_image.is_file():
                    visible_marker_checks += 1
                    try:
                        if not _has_red_marker(current_image, projections):
                            errors.append(
                                f"{path.name} policy image has no red marker at "
                                "any recorded candidate projection"
                            )
                    except OSError as exc:
                        errors.append(
                            f"{path.name} policy image cannot be decoded: {exc}"
                        )
        resolution = (payload.get("camera") or {}).get("resolution")
        current_images = payload.get("input_images") or []
        if (
            isinstance(resolution, list)
            and len(resolution) == 2
            and current_images
            and (path.parent / current_images[-1]).is_file()
        ):
            image_resolution_checks += 1
            try:
                with Image.open(path.parent / current_images[-1]) as image:
                    if list(image.size) != [
                        int(resolution[0]),
                        int(resolution[1]),
                    ]:
                        errors.append(
                            f"{path.name} policy image resolution {list(image.size)} "
                            f"disagrees with camera metadata {resolution}"
                        )
            except OSError:
                pass
        prompt = payload.get("prompt") or {}
        if str(prompt.get("traffic_light_context") or "").strip():
            prompt_leaks.append(path.name)
        if prompt.get("system") and (
            "when they fall inside the first-person camera frame"
            not in str(prompt.get("system"))
        ):
            errors.append(
                f"{path.name} system prompt omits projection-clipping contract"
            )
        user_prompt = str(prompt.get("user") or "")
        all_markers_visible = (
            (payload.get("observation_geometry") or {}).get(
                "all_waypoint_markers_visible"
            )
            is True
        )
        states_all_markers_visible = (
            "all seven candidate waypoint projections marked in red"
            in user_prompt
        )
        if prompt.get("user") and not (
            "can be clipped by the first-person camera" in user_prompt
            or (all_markers_visible and states_all_markers_visible)
        ):
            errors.append(
                f"{path.name} user prompt omits the marker-visibility contract"
            )
        if payload.get("safety_filter_overrode_action") is True:
            manifest_overrides.append(path.name)
        adherence = payload.get("route_adherence")
        if not isinstance(adherence, dict):
            errors.append(f"{path.name} has no route-adherence evidence")
        elif adherence.get("within_corridor") is not True:
            if allow_route_deviations:
                route_deviation_manifests.append(path.name)
                if not route_deviation_excursion_active:
                    route_deviation_excursion_manifests.append(path.name)
                route_deviation_excursion_active = True
            else:
                errors.append(
                    f"{path.name} left the legal sidewalk/crosswalk corridor"
                )
        else:
            route_deviation_excursion_active = False
            lateral = float(adherence.get("lateral_distance_cm") or 0.0)
            if (
                adherence.get("edge_type") == "crosswalk"
                and lateral > PHYSICAL_CROSSWALK_HALF_WIDTH_CM
            ):
                errors.append(f"{path.name} is outside the painted zebra width")
            if (
                expected_case == "safe"
                and adherence.get("edge_type") != "crosswalk"
                and lateral > SIDEWALK_ROUTE_HALF_WIDTH_CM
                and adherence.get("within_transition_envelope") is not True
            ):
                errors.append(
                    f"{path.name} is outside the authored sidewalk corridor"
                )
    if missing_images:
        errors.append(f"missing/empty recorded images: {missing_images[:8]}")
    if prompt_leaks:
        errors.append(f"visual-only manifests expose symbolic traffic context: {prompt_leaks}")
    if manifest_overrides:
        errors.append(f"visual-only manifests contain safety overrides: {manifest_overrides}")

    by_crosswalk: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    sidecar_overrides: list[str] = []
    invalid_clearance_holds: list[str] = []
    crossing_samples = 0
    for path in snapshots:
        payload = _load_json(path)
        if payload.get("execution_override") is not None:
            sidecar_overrides.append(path.name)
        for stage_name in ("model_input", "action_execution"):
            stage = payload.get(stage_name) or {}
            hold_active = bool(stage.get("legacy_occupancy_hold")) or bool(
                stage.get("clearance_extended")
            )
            # WALK has a fixed end time. No occupancy or clearance state may
            # extend the rendered pedestrian WALK interval.
            if hold_active:
                invalid_clearance_holds.append(f"{path.name}:{stage_name}")
        snapshot = payload.get("model_input") or {}
        if not bool(snapshot.get("agent_on_crosswalk")):
            continue
        crossing_samples += 1
        if expected_case == "safe" and not _safe_crossing_signal(snapshot):
            errors.append(
                f"{path.name} records crosswalk travel without a legal "
                "entry or beyond flashing clearance"
            )
        crosswalk_id = str(snapshot.get("crosswalk_id"))
        projection = snapshot.get("crosswalk_projection")
        lateral = snapshot.get("crosswalk_lateral_distance_cm")
        if not isinstance(projection, (int, float)):
            errors.append(f"{path.name} crosswalk sample has no numeric projection")
            continue
        if not isinstance(lateral, (int, float)):
            errors.append(f"{path.name} crosswalk sample has no numeric lateral distance")
            continue
        corridor_half_width = float(
            snapshot.get("crossing_corridor_half_width_cm") or 600.0
        )
        if float(lateral) > corridor_half_width:
            errors.append(
                f"{path.name} is logically on crosswalk but outside its corridor"
            )
        step = int(path.parent.name.split("_")[1])
        by_crosswalk[crosswalk_id].append((step, float(projection), float(lateral)))
    if sidecar_overrides:
        errors.append(f"visual-only sidecars contain execution overrides: {sidecar_overrides}")
    if invalid_clearance_holds:
        errors.append(
            "occupancy/clearance state extended the fixed WALK phase: "
            f"{invalid_clearance_holds}"
        )
    if require_route_crosswalk_sample is None:
        require_route_crosswalk_sample = expected_case in {"safe", "collision"}
    if (
        route_has_crosswalk is not False
        and require_route_crosswalk_sample
        and crossing_samples == 0
    ):
        errors.append("no first-person step was recorded on a route crosswalk")

    traversed = []
    for crosswalk_id, samples in by_crosswalk.items():
        projections = [sample[1] for sample in samples]
        span = max(projections) - min(projections)
        if span >= 0.40:
            traversed.append(crosswalk_id)
    if route_has_crosswalk is not False and expected_case == "safe" and not traversed:
        errors.append("safe case has no recorded crosswalk traversal span >= 0.40")

    summary = {
        "manifests": len(manifests),
        "sidecars": len(snapshots),
        "crossing_samples": crossing_samples,
        "camera_projection_checks": camera_projection_checks,
        "visible_marker_checks": visible_marker_checks,
        "image_resolution_checks": image_resolution_checks,
        "route_deviation_manifest_count": len(route_deviation_manifests),
        "route_deviation_manifests": route_deviation_manifests,
        "route_deviation_excursion_count": len(
            route_deviation_excursion_manifests
        ),
        "route_deviation_excursion_manifests": (
            route_deviation_excursion_manifests
        ),
        "crosswalk_projection_spans": {
            key: round(
                max(value[1] for value in samples)
                - min(value[1] for value in samples),
                4,
            )
            for key, samples in by_crosswalk.items()
        },
        "traversed_crosswalk_ids": traversed,
    }
    return errors, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--steps-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=("any", "safe", "collision"),
        default="any",
        dest="expected_case",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = _load_json(args.result)
    result_errors, result_summary = audit_result(result, args.expected_case)
    step_errors, step_summary = audit_steps(
        args.steps_dir,
        args.expected_case,
        allow_route_deviations=True,
        route_has_crosswalk=bool(result_summary.get("route_has_crosswalk")),
        require_route_crosswalk_sample=(
            args.expected_case == "safe"
            or (
                args.expected_case == "collision"
                and not bool(
                    result_summary.get("verified_illegal_crossing_collision")
                )
            )
        ),
    )
    errors = [*result_errors, *step_errors]
    report = {
        "passed": not errors,
        "case": args.expected_case,
        "result": str(args.result),
        "steps_dir": str(args.steps_dir),
        "result_summary": result_summary,
        "step_summary": step_summary,
        "errors": errors,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
