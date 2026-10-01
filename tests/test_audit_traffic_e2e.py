import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from simworld.utils.vector import Vector
from tools.audit_traffic_e2e import audit_result, audit_steps
from utils.annotate_image import annotate_image

REPO_ROOT = Path(__file__).resolve().parents[1]


def valid_adherence(edge_type="sidewalk", progress=0.5):
    return {
        "edge_type": edge_type,
        "segment_start": {"x": 0.0, "y": 0.0},
        "segment_end": {"x": 1000.0, "y": 0.0},
        "progress": progress,
        "lateral_distance_cm": 0.0,
        "corridor_half_width_cm": 210.0 if edge_type == "crosswalk" else 250.0,
        "within_corridor": True,
    }


def valid_result():
    return {
        "success": True,
        "failed": False,
        "failure_reason": None,
        "rollout_error": None,
        "signal_traffic_error_count": 0,
        "signal_controlled_vehicle_count": 2,
        "signal_vehicle_pose_sample_count": {"0": 12, "1": 12},
        "signal_vehicle_max_tilt_deg": {"0": 1.25, "1": 2.0},
        "signal_vehicle_z_range_cm": {
            "0": {"min": 34.0, "max": 35.0},
            "1": {"min": 33.5, "max": 35.5},
        },
        "signal_vehicle_tilt_limit_deg": 30.0,
        "signal_vehicle_instability_events": [],
        "traffic_policy": "visual_only",
        "traffic_assistance_enabled": False,
        "agent_camera_mode": "first_person_free_follow",
        "traffic_light_gate_count": 0,
        "route_crosswalk_hops": 1,
        "red_light_conflict_vehicle_enabled": True,
        "red_light_conflict_launch_distance_min_cm": 100.0,
        "red_light_conflict_launch_distance_max_cm": 500.0,
        "red_light_conflict_collision_radius_cm": 250.0,
        "signal_conflict_route_coverage_count": 1,
        "pedestrian_signal_compliance_probability": 0.5,
        "signal_pedestrian_compliant_count": 1,
        "signal_pedestrian_noncompliant_count": 1,
        "signal_pedestrian_compliance": {"0": True, "1": False},
        "decision_trace": [{
            "step": 1,
            "execution_override": None,
            "route_adherence": valid_adherence(),
            "traffic_light_snapshot": {"pedestrian_state": "NOT_RELEVANT"},
        }],
        "red_light_violations_count": 0,
        "red_light_violation_events": [],
        "red_light_conflict_vehicle_events": [],
        "illegal_crossing_violations_count": 0,
        "illegal_crossing_events": [],
        "illegal_crossing_conflict_vehicle_events": [],
        "route_crosswalk_marking_records": [],
        "render_geometry_calibration": {
            "asset": "BP_Road_Small",
            "crosswalks": [{"rendered_offset_cm": 700.0}],
        },
        "traffic_system_initial_snapshots": {
            "0": {"safe": True, "violations": []},
        },
        "vehicle_collision_count": 0,
    }


class TrafficE2EAuditTests(unittest.TestCase):
    def test_cli_imports_from_outside_repo_with_runner_pythonpath(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "SimWorld")
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "audit_traffic_e2e.py"),
                    "--help",
                ],
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_safe_terminal_result_passes(self):
        errors, _ = audit_result(valid_result(), "safe")
        self.assertEqual([], errors)

    def test_illegal_crossing_is_in_combined_violation_summary(self):
        result = valid_result()
        result["illegal_crossing_violations_count"] = 1
        result["illegal_crossing_events"] = [
            {
                "event_id": "illegal-crossing-0001",
                "trigger_type": "illegal_crossing",
                "outside_authored_crossing": True,
            }
        ]
        result["illegal_crossing_conflict_vehicle_events"] = [
            {
                "event_id": "illegal-crossing-0001",
                "disposition": "unavailable",
                "status": "unavailable",
            }
        ]

        errors, summary = audit_result(result, "any")

        self.assertEqual([], errors)
        self.assertEqual(1, summary["violation_events"])
        self.assertEqual({"unavailable": 1}, summary["consequence_dispositions"])

    def test_extended_conflict_summary_matches_trigger_event_arrays(self):
        result = valid_result()
        red_event = {
            "event_id": "red-light-0001",
            "disposition": "staged",
            "status": "staged",
        }
        illegal_event = {
            "event_id": "illegal-crossing-0001",
            "disposition": "launched",
            "status": "collision",
            "collision_triggered": True,
            "released": True,
            "control_mode": "lane_aligned_intercept",
            "requested_launch_distance_cm": 500.0,
            "launch_distance_cm": 500.0,
            "launch_distance_range_cm": [300.0, 900.0],
            "collision_radius_cm": 250.0,
        }
        result.update({
            "red_light_violations_count": 1,
            "red_light_violation_events": [
                {
                    "event_id": "red-light-0001",
                    "trigger_type": "red_light",
                    "agent_on_crosswalk": True,
                    "crosswalk_projection": 0.5,
                    "crosswalk_lateral_distance_cm": 0.0,
                    "crossing_corridor_half_width_cm": 600.0,
                }
            ],
            "red_light_conflict_vehicle_events": [red_event],
            "illegal_crossing_violations_count": 1,
            "illegal_crossing_events": [{
                "event_id": "illegal-crossing-0001",
                "trigger_type": "illegal_crossing",
                "outside_authored_crossing": True,
            }],
            "illegal_crossing_conflict_vehicle_events": [illegal_event],
            "red_light_conflict_vehicle_count": 0,
            "red_light_conflict_vehicle_launch_count": 0,
            "red_light_conflict_vehicle_collision_count": 0,
            "red_light_conflict_disposition_count": 1,
            "red_light_conflict_disposition_counts": {"staged": 1},
            "red_light_conflict_status_counts": {"staged": 1},
            "illegal_crossing_conflict_vehicle_count": 1,
            "illegal_crossing_conflict_vehicle_launch_count": 1,
            "illegal_crossing_conflict_vehicle_collision_count": 1,
            "illegal_crossing_conflict_disposition_count": 1,
            "illegal_crossing_conflict_disposition_counts": {"launched": 1},
            "illegal_crossing_conflict_status_counts": {"collision": 1},
            "traffic_conflict_vehicle_count": 1,
            "traffic_conflict_vehicle_launch_count": 1,
            "traffic_conflict_vehicle_collision_count": 1,
            "traffic_conflict_disposition_count": 2,
            "traffic_conflict_disposition_counts": {
                "launched": 1,
                "staged": 1,
            },
            "traffic_conflict_status_counts": {
                "collision": 1,
                "staged": 1,
            },
            "traffic_conflict_vehicle_events": [red_event, illegal_event],
        })

        errors, _ = audit_result(result, "any")

        self.assertEqual([], errors)

    def test_extended_conflict_summary_rejects_wrong_launch_count(self):
        result = valid_result()
        result.update({
            "red_light_conflict_vehicle_count": 0,
            "red_light_conflict_vehicle_launch_count": 0,
            "red_light_conflict_vehicle_collision_count": 0,
            "red_light_conflict_disposition_count": 0,
            "red_light_conflict_disposition_counts": {},
            "red_light_conflict_status_counts": {},
            "illegal_crossing_conflict_vehicle_count": 0,
            "illegal_crossing_conflict_vehicle_launch_count": 0,
            "illegal_crossing_conflict_vehicle_collision_count": 0,
            "illegal_crossing_conflict_disposition_count": 0,
            "illegal_crossing_conflict_disposition_counts": {},
            "illegal_crossing_conflict_status_counts": {},
            "traffic_conflict_vehicle_count": 9,
            "traffic_conflict_vehicle_launch_count": 9,
            "traffic_conflict_vehicle_collision_count": 0,
            "traffic_conflict_disposition_count": 0,
            "traffic_conflict_disposition_counts": {},
            "traffic_conflict_status_counts": {},
            "traffic_conflict_vehicle_events": [],
        })

        errors, _ = audit_result(result, "any")

        self.assertTrue(any(
            "traffic_conflict_vehicle_count disagrees" in error
            for error in errors
        ))

    def test_legacy_partial_red_summary_remains_auditable(self):
        result = valid_result()
        result.update({
            "red_light_conflict_vehicle_count": 0,
            "red_light_conflict_disposition_count": 0,
            "red_light_conflict_disposition_counts": {},
        })

        errors, _ = audit_result(result, "safe")

        self.assertEqual([], errors)

    def test_any_case_accepts_max_steps_as_terminal_completion(self):
        result = valid_result()
        result.update(
            {
                "success": False,
                "failed": False,
                "final_step": 230,
                "max_steps": 230,
            }
        )

        errors, summary = audit_result(result, "any")

        self.assertEqual([], errors)
        self.assertTrue(summary["terminal"])
        self.assertTrue(summary["reached_max_steps"])

    def test_custom_crosswalk_overlay_is_rejected(self):
        result = valid_result()
        result["route_crosswalk_marking_records"] = [
            {"render_backend": "ue_actor_static", "collision": False}
        ]

        errors, _ = audit_result(result, "safe")

        self.assertTrue(any("custom route-crosswalk" in error for error in errors))

    def test_safe_crosswalk_trace_accepts_clearance_after_walk_admission(self):
        result = valid_result()
        result["decision_trace"] = [
            {
                "step": 1,
                "execution_override": None,
                "route_adherence": valid_adherence("crosswalk", 0.1),
                "traffic_light_snapshot": {"pedestrian_state": "WALK"},
            },
            {
                "step": 2,
                "execution_override": None,
                "route_adherence": valid_adherence("crosswalk", 0.5),
                "execution_traffic_light_snapshot": {
                    "pedestrian_state": "FLASHING_DONT_WALK",
                    "route_crossing_permission": "CLEAR_ONLY",
                    "crossing_admitted_on_walk": True,
                },
                "collision_details": {"legal_route_crosswalk": 1},
            },
        ]

        errors, _ = audit_result(result, "safe")

        self.assertFalse(any("lacked a legal entry" in error for error in errors))

    def test_safe_crosswalk_trace_rejects_clearance_without_walk_admission(self):
        result = valid_result()
        result["decision_trace"] = [
            {
                "step": 1,
                "execution_override": None,
                "route_adherence": valid_adherence("crosswalk", 0.5),
                "traffic_light_snapshot": {
                    "pedestrian_state": "FLASHING_DONT_WALK",
                    "route_crossing_permission": "CLEAR_ONLY",
                    "crossing_admitted_on_walk": False,
                },
                "collision_details": {"legal_route_crosswalk": 1},
            }
        ]

        errors, _ = audit_result(result, "safe")

        self.assertTrue(any("lacked a legal entry" in error for error in errors))

    def test_crosswalk_edge_at_curb_is_not_a_signal_traversal_sample(self):
        result = valid_result()
        result["decision_trace"] = [
            {
                "step": 1,
                "execution_override": None,
                "route_adherence": valid_adherence("crosswalk", 0.0),
                "collision_details": {"legal_route_crosswalk": 0},
            }
        ]

        errors, _ = audit_result(result, "safe")

        self.assertFalse(any("lacked WALK" in error for error in errors))

    def test_unsafe_initial_signal_snapshot_is_rejected(self):
        result = valid_result()
        result["traffic_system_initial_snapshots"]["2"] = {
            "safe": False,
            "violations": ["conflicting_vehicle_movement_groups_green"],
        }

        errors, _ = audit_result(result, "safe")

        self.assertTrue(any("initialized in an unsafe" in error for error in errors))

    def test_off_lane_conflict_is_rejected(self):
        result = valid_result()
        result.update(
            {
                "success": False,
                "failed": True,
                "failure_reason": "vehicle_collision",
                "vehicle_collision_count": 1,
                "red_light_violations_count": 1,
                "red_light_violation_events": [{"event_id": "v1"}],
                "red_light_conflict_vehicle_events": [
                    {
                        "event_id": "v1",
                        "disposition": "launched",
                        "status": "collision",
                        "collision_triggered": True,
                        "control_mode": "synthetic_crosswalk_axis_fallback",
                        "requested_launch_distance_cm": 300.0,
                        "launch_distance_cm": 300.0,
                        "launch_distance_range_cm": [100.0, 500.0],
                        "collision_radius_cm": 250.0,
                    }
                ],
            }
        )
        errors, _ = audit_result(result, "collision")
        self.assertTrue(any("not on a lane-aligned" in error for error in errors))
        self.assertTrue(any("synthetic/off-lane" in error for error in errors))

    def test_missing_vehicle_pose_evidence_is_rejected(self):
        result = valid_result()
        result["signal_vehicle_pose_sample_count"] = {"0": 12}
        result["signal_vehicle_max_tilt_deg"] = {"0": 1.25}
        result["signal_vehicle_z_range_cm"] = {
            "0": {"min": 34.0, "max": 35.0}
        }

        errors, _ = audit_result(result, "safe")

        self.assertTrue(any("missing full-pose" in error for error in errors))
        self.assertTrue(any("missing tilt evidence" in error for error in errors))
        self.assertTrue(any("missing vertical-pose" in error for error in errors))

    def test_vehicle_flip_is_rejected_even_without_collision_count(self):
        result = valid_result()
        result["signal_vehicle_max_tilt_deg"]["1"] = 176.0
        result["signal_vehicle_instability_events"] = [
            {
                "vehicle_id": 1,
                "pitch_deg": 176.0,
                "roll_deg": 4.0,
            }
        ]

        errors, _ = audit_result(result, "safe")

        self.assertTrue(any("upright tilt limit" in error for error in errors))
        self.assertTrue(any("instability was observed" in error for error in errors))

    def test_missing_disposition_is_rejected(self):
        result = valid_result()
        result["red_light_violations_count"] = 1
        result["red_light_violation_events"] = [{"event_id": "v1"}]
        errors, _ = audit_result(result, "any")
        self.assertTrue(any("missing consequence" in error for error in errors))

    def test_v3_distinguishes_staged_from_released_launches(self):
        result = valid_result()
        result["artifact_schema_version"] = "two_rule_traffic_events_v3"
        result["red_light_violations_count"] = 1
        result["red_light_violation_events"] = [
            {
                "event_id": "v1",
                "agent_on_crosswalk": True,
                "crosswalk_projection": 0.5,
                "crosswalk_lateral_distance_cm": 20.0,
                "crossing_corridor_half_width_cm": 600.0,
            }
        ]
        event = {
            "event_id": "v1",
            "disposition": "launched",
            "status": "launched",
            "released": False,
            "collision_triggered": False,
            "control_mode": "lane_aligned_intercept",
            "requested_launch_distance_cm": 300.0,
            "launch_distance_cm": 300.0,
            "launch_distance_range_cm": [100.0, 500.0],
            "collision_radius_cm": 250.0,
        }
        result["red_light_conflict_vehicle_events"] = [event]

        errors, _ = audit_result(result, "any")

        self.assertTrue(
            any("was not released into motion" in error for error in errors)
        )

        event["disposition"] = "staged"
        event["status"] = "staged"
        errors, _ = audit_result(result, "any")

        self.assertEqual([], errors)

    def test_violation_outside_crosswalk_endpoints_is_rejected(self):
        result = valid_result()
        result["red_light_violations_count"] = 1
        result["red_light_violation_events"] = [
            {
                "event_id": "v1",
                "crosswalk_projection": 1.1386,
                "crosswalk_lateral_distance_cm": 335.24,
                "crossing_corridor_half_width_cm": 600.0,
            }
        ]
        result["red_light_conflict_vehicle_events"] = [
            {"event_id": "v1", "disposition": "disabled"}
        ]

        errors, _ = audit_result(result, "any")

        self.assertTrue(any("outside crosswalk endpoints" in error for error in errors))

    def test_collision_case_accepts_one_verified_collision_among_multiple_launches(self):
        result = valid_result()
        result.update(
            {
                "success": False,
                "failed": True,
                "failure_reason": "vehicle_collision",
                "vehicle_collision_count": 1,
            }
        )
        result["red_light_violations_count"] = 2
        result["red_light_violation_events"] = [
            {
                "event_id": "v1",
                "agent_on_crosswalk": True,
                "crosswalk_projection": 0.5,
                "crosswalk_lateral_distance_cm": 20.0,
                "crossing_corridor_half_width_cm": 600.0,
            },
            {
                "event_id": "v2",
                "agent_on_crosswalk": True,
                "crosswalk_projection": 0.6,
                "crosswalk_lateral_distance_cm": 25.0,
                "crossing_corridor_half_width_cm": 600.0,
            },
        ]
        result["red_light_conflict_vehicle_events"] = [
            {
                "event_id": "v1",
                "disposition": "launched",
                "status": "launched",
                "collision_triggered": False,
                "control_mode": "lane_aligned_intercept",
                "requested_launch_distance_cm": 300.0,
                "launch_distance_cm": 300.0,
                "launch_distance_range_cm": [100.0, 500.0],
                "collision_radius_cm": 250.0,
            },
            {
                "event_id": "v2",
                "disposition": "launched",
                "status": "collision",
                "collision_triggered": True,
                "control_mode": "lane_aligned_intercept",
                "requested_launch_distance_cm": 400.0,
                "launch_distance_cm": 400.0,
                "launch_distance_range_cm": [100.0, 500.0],
                "collision_radius_cm": 250.0,
            },
        ]

        errors, _ = audit_result(result, "collision")

        self.assertEqual([], errors)

    def test_clamped_launch_below_sampled_range_is_rejected(self):
        result = valid_result()
        result["red_light_conflict_vehicle_events"] = [
            {
                "event_id": "v1",
                "disposition": "launched",
                "status": "collision",
                "collision_triggered": True,
                "control_mode": "lane_aligned_intercept",
                "requested_launch_distance_cm": 650.74,
                "launch_distance_cm": 289.6,
                "launch_distance_range_cm": [500.0, 700.0],
                "collision_radius_cm": 250.0,
            }
        ]

        errors, _ = audit_result(result, "any")

        self.assertTrue(
            any("did not physically realize" in error for error in errors)
        )

    def test_clamped_unsafe_road_launch_is_rejected(self):
        result = valid_result()
        result["illegal_crossing_conflict_vehicle_events"] = [
            {
                "event_id": "unsafe-1",
                "disposition": "launched",
                "status": "collision",
                "collision_triggered": True,
                "control_mode": "lane_aligned_intercept",
                "requested_launch_distance_cm": 650.74,
                "launch_distance_cm": 158.99,
                "launch_distance_range_cm": [500.0, 700.0],
                "collision_radius_cm": 250.0,
            }
        ]

        errors, _ = audit_result(result, "any")

        self.assertTrue(
            any("did not physically realize" in error for error in errors)
        )

    def test_missing_real_lane_conflict_coverage_is_rejected(self):
        result = valid_result()
        result["signal_conflict_route_coverage_count"] = 0
        errors, _ = audit_result(result, "any")
        self.assertTrue(any("no verified real-lane route" in error for error in errors))


    def test_verified_endpoint_extension_covers_missing_staged_route(self):
        result = valid_result()
        result.update(
            {
                "signal_conflict_route_coverage_count": 0,
                "signal_conflict_route_geometry_available": True,
                "signal_conflict_route_source": "authored_lane_endpoint_extension",
                "signal_conflict_route_max_launch_distance_cm_verified": 500.0,
            }
        )

        errors, summary = audit_result(result, "any")

        self.assertEqual([], errors)
        self.assertEqual(
            "authored_lane_endpoint_extension",
            summary["signal_conflict_route_source"],
        )

    def test_endpoint_extension_must_cover_configured_maximum(self):
        result = valid_result()
        result.update(
            {
                "signal_conflict_route_coverage_count": 0,
                "signal_conflict_route_geometry_available": True,
                "signal_conflict_route_source": "authored_lane_endpoint_extension",
                "signal_conflict_route_max_launch_distance_cm_verified": 499.0,
            }
        )
        errors, _ = audit_result(result, "any")
        self.assertTrue(any("configured maximum" in error for error in errors))

    def test_nonconflicting_crosswalk_geometry_does_not_require_launch_route(self):
        result = valid_result()
        result.update(
            {
                "signal_conflict_route_geometry_available": False,
                "signal_conflict_route_coverage_count": 0,
            }
        )

        errors, summary = audit_result(result, "any")

        self.assertEqual([], errors)
        self.assertFalse(summary["signal_conflict_route_geometry_available"])

    def test_no_crosswalk_route_does_not_require_crosswalk_assets_or_conflict(self):
        result = valid_result()
        result.update(
            {
                "route_crosswalk_hops": 0,
                "signal_conflict_route_coverage_count": 0,
                "render_geometry_calibration": {},
            }
        )

        errors, summary = audit_result(result, "any")

        self.assertEqual([], errors)
        self.assertFalse(summary["route_has_crosswalk"])

    def test_downgraded_authored_crosswalk_connector_is_rejected(self):
        result = valid_result()
        result.update(
            {
                "route_crosswalk_hops": 0,
                "signal_conflict_route_coverage_count": 0,
                "signal_conflict_route_geometry_available": False,
                "render_geometry_calibration": {
                    "asset": "BP_Road_Small",
                    "crosswalks": [],
                    "non_crossing_connectors": [
                        {
                            "edge_index": 1,
                            "reason": "no_matching_native_zebra",
                        }
                    ],
                },
            }
        )

        errors, summary = audit_result(result, "any")

        self.assertTrue(
            any("downgraded to a non-crossing connector" in error for error in errors)
        )
        self.assertEqual(1, summary["non_crossing_connector_count"])

    def test_any_case_reports_nonterminal_corridor_deviation(self):
        result = valid_result()
        result.update(
            {
                "success": False,
                "failed": False,
                "failure_reason": None,
                "final_step": 20,
                "max_steps": 20,
                "decision_trace": [
                    {
                        "step": 3,
                        "execution_override": None,
                        "route_adherence": {
                            **valid_adherence(),
                            "lateral_distance_cm": 300.0,
                            "within_corridor": False,
                        },
                    }
                ],
            }
        )

        errors, summary = audit_result(result, "any")

        self.assertEqual([], errors)
        self.assertEqual(1, summary["route_deviation_count"])
        self.assertEqual(["step 3:outside"], summary["route_deviations"])

    def test_any_case_reports_nonterminal_off_zebra_deviation(self):
        result = valid_result()
        result.update(
            {
                "success": False,
                "failed": False,
                "failure_reason": None,
                "final_step": 20,
                "max_steps": 20,
                "decision_trace": [
                    {
                        "step": 4,
                        "execution_override": None,
                        "route_adherence": {
                            **valid_adherence("crosswalk"),
                            "lateral_distance_cm": 411.6,
                            "within_corridor": False,
                        },
                        "traffic_light_snapshot": {
                            "pedestrian_state": "WALK"
                        },
                    }
                ],
            }
        )

        errors, summary = audit_result(result, "any")

        self.assertEqual([], errors)
        self.assertEqual(1, summary["route_deviation_count"])
        self.assertEqual(["step 4:outside"], summary["route_deviations"])

    def test_route_deviation_counts_continuous_excursions_not_frames(self):
        result = valid_result()
        trace = []
        for step, within_corridor in enumerate(
            (False, False, False, True, False),
            start=1,
        ):
            trace.append(
                {
                    "step": step,
                    "execution_override": None,
                    "route_adherence": {
                        **valid_adherence(),
                        "lateral_distance_cm": (
                            300.0 if not within_corridor else 0.0
                        ),
                        "within_corridor": within_corridor,
                    },
                }
            )
        result["decision_trace"] = trace

        errors, summary = audit_result(result, "any")

        self.assertEqual([], errors)
        self.assertEqual(2, summary["route_deviation_count"])
        self.assertEqual(
            ["step 1:outside", "step 5:outside"],
            summary["route_deviations"],
        )
        self.assertEqual(4, summary["route_deviation_sample_count"])

    def test_progress_regression_has_its_own_route_deviation_label(self):
        result = valid_result()
        first = valid_adherence()
        first["progress"] = 0.5
        second = valid_adherence()
        second["progress"] = 0.25
        result["decision_trace"] = [
            {"step": 1, "execution_override": None, "route_adherence": first},
            {"step": 2, "execution_override": None, "route_adherence": second},
        ]

        errors, summary = audit_result(result, "any")

        self.assertEqual([], errors)
        self.assertEqual(
            ["step 2:progress-regression"], summary["route_deviations"]
        )

    def test_any_route_trace_outside_sidewalk_is_reported_not_rejected(self):
        result = valid_result()
        result["decision_trace"][0]["route_adherence"].update(
            {
                "lateral_distance_cm": 300.0,
                "within_corridor": False,
            }
        )
        errors, summary = audit_result(result, "any")
        self.assertEqual([], errors)
        self.assertEqual(1, summary["route_deviation_count"])

    def test_safe_route_trace_outside_sidewalk_is_recorded_not_rejected(self):
        result = valid_result()
        result["decision_trace"][0]["route_adherence"].update(
            {
                "lateral_distance_cm": 300.0,
                "within_corridor": False,
            }
        )

        errors, summary = audit_result(result, "safe")

        self.assertFalse(any("route-corridor invariant" in error for error in errors))
        self.assertEqual(1, summary["route_deviation_count"])

    def test_safe_route_trace_accepts_in_corridor_lateral_sway(self):
        result = valid_result()
        result["decision_trace"][0]["route_adherence"].update(
            {
                "lateral_distance_cm": 64.072,
                "within_corridor": True,
            }
        )

        errors, metrics = audit_result(result, "safe")

        self.assertEqual([], errors)
        self.assertEqual(64.072, metrics["max_route_lateral_cm"])

    def test_safe_route_trace_rejects_lateral_beyond_sidewalk_corridor(self):
        result = valid_result()
        result["decision_trace"][0]["route_adherence"].update(
            {
                "lateral_distance_cm": 300.0,
                "within_corridor": True,
            }
        )

        errors, _ = audit_result(result, "safe")

        self.assertTrue(
            any("off-ordered-edge-corridor" in error for error in errors)
        )

    def test_route_transition_before_exact_endpoint_is_rejected(self):
        result = valid_result()
        first = valid_adherence("sidewalk", 0.96)
        second = valid_adherence("crosswalk", 0.0)
        second.update(
            {
                "segment_start": {"x": 1000.0, "y": 0.0},
                "segment_end": {"x": 1000.0, "y": 1000.0},
                "lateral_distance_cm": 18.3,
            }
        )
        result["decision_trace"] = [
            {"step": 1, "execution_override": None, "route_adherence": first},
            {
                "step": 2,
                "execution_override": None,
                "route_adherence": second,
                "traffic_light_snapshot": {"pedestrian_state": "WALK"},
            },
        ]

        errors, _ = audit_result(result, "safe")

        self.assertTrue(
            any("segment-transition-before-endpoint" in error for error in errors)
        )

    def test_route_transition_at_exact_endpoint_is_accepted(self):
        result = valid_result()
        first = valid_adherence("sidewalk", 1.0)
        second = valid_adherence("crosswalk", 0.0)
        second.update(
            {
                "segment_start": {"x": 1000.0, "y": 0.0},
                "segment_end": {"x": 1000.0, "y": 1000.0},
                "lateral_distance_cm": 0.0,
            }
        )
        result["decision_trace"] = [
            {"step": 1, "execution_override": None, "route_adherence": first},
            {
                "step": 2,
                "execution_override": None,
                "route_adherence": second,
                "traffic_light_snapshot": {"pedestrian_state": "WALK"},
            },
        ]

        errors, metrics = audit_result(result, "safe")

        self.assertEqual([], errors)
        self.assertEqual(1, metrics["exact_segment_transitions"])

    def test_crosswalk_exit_accepts_painted_rectangular_joint_envelope(self):
        result = valid_result()
        first = valid_adherence("crosswalk", 0.5)
        first.update(
            {
                "segment_start": {"x": 0.0, "y": 0.0},
                "segment_end": {"x": 0.0, "y": 1000.0},
            }
        )
        second = valid_adherence("sidewalk", -0.2)
        second.update(
            {
                "segment_start": {"x": 0.0, "y": 1000.0},
                "segment_end": {"x": 0.0, "y": 2000.0},
                "lateral_distance_cm": 269.072,
                "transition_distance_cm": 269.072,
                "within_start_transition_envelope": True,
            }
        )
        result["decision_trace"] = [
            {"step": 1, "execution_override": None, "route_adherence": first},
            {
                "step": 2,
                "position": {"x": 180.0, "y": 800.0},
                "execution_override": None,
                "route_adherence": second,
            },
        ]

        errors, metrics = audit_result(result, "any")

        self.assertEqual([], errors)
        self.assertEqual(1, metrics["exact_segment_transitions"])

        result["decision_trace"][1]["position"]["x"] = 220.0
        errors, metrics = audit_result(result, "any")
        self.assertEqual([], errors)
        self.assertIn(
            "step 2:segment-transition-before-endpoint",
            metrics["route_deviations"],
        )

    def test_terminal_inference_vehicle_collision_may_omit_route_snapshot(self):
        result = valid_result()
        result.update(
            {
                "success": False,
                "failed": True,
                "failure_reason": "vehicle_collision",
                "vehicle_collision_count": 1,
            }
        )
        result["decision_trace"].append(
            {
                "step": 2,
                "executed_action": None,
                "vlm_proposed_action": {
                    "type": "move_to",
                    "param": "5",
                    "executed": False,
                    "reason": "vehicle_collision_during_inference",
                },
                "position": {"x": 100.0, "y": 0.0},
                "collision_details": {"vehicle": 1},
                "passive_collision_details": {"vehicle": 1},
            }
        )

        errors, _ = audit_result(result, "any")

        self.assertFalse(any("step 2:missing" in error for error in errors))

        result["decision_trace"][-1]["collision_details"]["vehicle"] = 0
        errors, _ = audit_result(result, "any")
        self.assertTrue(any("step 2:missing" in error for error in errors))

    def test_any_case_accepts_verified_illegal_crossing_collision_snapshot(self):
        result = valid_result()
        unsafe_adherence = {
            **valid_adherence(),
            "lateral_distance_cm": 620.0,
            "within_corridor": False,
        }
        result.update(
            {
                "success": False,
                "failed": True,
                "failure_reason": "vehicle_collision",
                "vehicle_collision_count": 1,
                "illegal_crossing_violations_count": 1,
                "illegal_crossing_events": [
                    {
                        "event_id": "unsafe-road-entry-0001",
                        "trigger_type": "illegal_crossing",
                        "agent_position": {"x": 10.0, "y": 20.0},
                        "route_adherence": unsafe_adherence,
                    }
                ],
                "illegal_crossing_conflict_vehicle_events": [
                    {
                        "event_id": "unsafe-road-entry-0001",
                        "trigger_type": "illegal_crossing",
                        "disposition": "launched",
                        "status": "collision",
                        "collision_triggered": True,
                        "impact_zone_reached": True,
                        "impact_zone_sim_time_s": 10.5,
                        "control_mode": "lane_aligned_intercept",
                        "route_source": "staged_intersection_route",
                        "collision_source": "conflict_vehicle_swept_envelope",
                        "requested_launch_distance_cm": 650.74,
                        "launch_distance_cm": 650.74,
                        "launch_distance_range_cm": [500.0, 700.0],
                        "collision_radius_cm": 250.0,
                        "minimum_swept_agent_distance_cm": 47.0,
                        "collision_distance_cm": 47.0,
                        "retired_after_consequence": True,
                    }
                ],
                "decision_trace": [
                    {
                        "step": 1,
                        "execution_override": None,
                        "position": {"x": 10.0, "y": 20.0},
                        "sim_time_seconds": 10.0,
                        "route_adherence": unsafe_adherence,
                    },
                    {
                        "step": 2,
                        "executed_action": None,
                        "vlm_proposed_action": {
                            "executed": False,
                            "reason": "vehicle_collision_during_inference",
                        },
                        "position": {"x": 10.0, "y": 20.0},
                        "sim_time_seconds": 10.5,
                        "collision_details": {"vehicle": 1},
                        "passive_collision_details": {"vehicle": 1},
                    },
                ],
            }
        )

        errors, summary = audit_result(result, "any")

        self.assertEqual([], errors)
        self.assertTrue(summary["verified_illegal_crossing_collision"])

        result['illegal_crossing_conflict_vehicle_events'][0][
            'impact_zone_sim_time_s'
        ] = 10.0
        errors, metrics = audit_result(result, 'any')
        self.assertTrue(any(
            'labeled during inference' in error for error in errors
        ))
        self.assertEqual(1, metrics['post_action_collision_misattribution_count'])

    def test_illegal_crossing_deviation_is_nonterminal_even_without_collision_chain(self):
        result = valid_result()
        unsafe_adherence = {
            **valid_adherence(),
            "lateral_distance_cm": 620.0,
            "within_corridor": False,
        }
        result.update(
            {
                "success": False,
                "failed": True,
                "failure_reason": "vehicle_collision",
                "vehicle_collision_count": 1,
                "decision_trace": [
                    {
                        "step": 1,
                        "execution_override": None,
                        "route_adherence": unsafe_adherence,
                    }
                ],
                "illegal_crossing_events": [
                    {
                        "event_id": "unsafe-road-entry-0001",
                        "trigger_type": "illegal_crossing",
                        "agent_position": {"x": 10.0, "y": 20.0},
                        "route_adherence": unsafe_adherence,
                    }
                ],
                "illegal_crossing_conflict_vehicle_events": [
                    {
                        "event_id": "unsafe-road-entry-0001",
                        "trigger_type": "illegal_crossing",
                        "disposition": "launched",
                        "status": "launched",
                        "collision_triggered": False,
                    }
                ],
            }
        )

        errors, summary = audit_result(result, "any")

        self.assertFalse(any("route-corridor invariant" in error for error in errors))
        self.assertEqual(1, summary["route_deviation_count"])

    def test_requested_mixed_pedestrians_must_be_instantiated(self):
        result = valid_result()
        result["signal_pedestrian_compliant_count"] = 2
        result["signal_pedestrian_noncompliant_count"] = 0
        result["signal_pedestrian_compliance"] = {"0": True, "1": True}
        errors, _ = audit_result(result, "any")
        self.assertTrue(any("mixed pedestrian compliance" in error for error in errors))

    def test_step_rejects_occupancy_hold_without_walk_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "step_0001_decision_0001"
            directory.mkdir()
            frame = directory / "frame.png"
            frame.write_bytes(b"png")
            (directory / "step_0001_manifest.json").write_text(
                json.dumps(
                    {
                        "traffic_policy": "visual_only",
                        "input_images": ["frame.png"],
                        "output_image": "frame.png",
                        "prompt": {"traffic_light_context": ""},
                        "route_adherence": valid_adherence("crosswalk"),
                    }
                ),
                encoding="utf-8",
            )
            (directory / "step_0001_traffic_light_snapshot.json").write_text(
                json.dumps(
                    {
                        "execution_override": None,
                        "model_input": {
                            "agent_on_crosswalk": True,
                            "crosswalk_id": 9,
                            "crosswalk_projection": 0.5,
                            "crosswalk_lateral_distance_cm": 20.0,
                            "crossing_corridor_half_width_cm": 600.0,
                            "legacy_occupancy_hold": True,
                            "crossing_admitted_on_walk": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            errors, _ = audit_steps(root, "any")
        self.assertTrue(
            any("extended the fixed WALK phase" in error for error in errors)
        )

    def test_step_rejects_legacy_hold_for_walk_admitted_crossing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "step_0001_decision_0001"
            directory.mkdir()
            frame = directory / "frame.png"
            frame.write_bytes(b"png")
            (directory / "step_0001_manifest.json").write_text(
                json.dumps(
                    {
                        "traffic_policy": "visual_only",
                        "input_images": ["frame.png"],
                        "output_image": "frame.png",
                        "prompt": {"traffic_light_context": ""},
                        "route_adherence": valid_adherence("crosswalk"),
                    }
                ),
                encoding="utf-8",
            )
            (directory / "step_0001_traffic_light_snapshot.json").write_text(
                json.dumps(
                    {
                        "execution_override": None,
                        "model_input": {
                            "agent_on_crosswalk": True,
                            "crosswalk_id": 9,
                            "crosswalk_projection": 0.5,
                            "crosswalk_lateral_distance_cm": 20.0,
                            "crossing_corridor_half_width_cm": 600.0,
                            "legacy_occupancy_hold": True,
                            "crossing_admitted_on_walk": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            errors, _ = audit_steps(root, "any")

        self.assertTrue(
            any("extended the fixed WALK phase" in error for error in errors)
        )

    def test_safe_step_artifacts_require_real_crosswalk_span(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step, projection in ((1, 0.1), (2, 0.8)):
                directory = root / f"step_{step:04d}_decision_{step:04d}"
                directory.mkdir()
                frame = directory / "frame.png"
                frame.write_bytes(b"png")
                (directory / f"step_{step:04d}_manifest.json").write_text(
                    json.dumps(
                        {
                            "traffic_policy": "visual_only",
                            "camera": {"mode": "first_person_free_follow"},
                            "input_images": ["frame.png"],
                            "output_image": "frame.png",
                            "prompt": {"traffic_light_context": ""},
                            "route_adherence": valid_adherence(
                                "crosswalk", projection
                            ),
                        }
                    ),
                    encoding="utf-8",
                )
                (directory / f"step_{step:04d}_traffic_light_snapshot.json").write_text(
                    json.dumps(
                        {
                            "execution_override": None,
                            "model_input": {
                                "agent_on_crosswalk": True,
                                "crosswalk_id": 9,
                                "crosswalk_projection": projection,
                                "crosswalk_lateral_distance_cm": 20.0,
                                "crossing_corridor_half_width_cm": 600.0,
                                "pedestrian_state": "WALK",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            errors, summary = audit_steps(root, "safe")
        self.assertEqual([], errors)
        self.assertEqual(["9"], summary["traversed_crosswalk_ids"])

    def test_any_case_checks_actual_marker_pixels_and_camera_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "step_0001_decision_0001"
            directory.mkdir()
            camera_location = [0.0, 0.0, 230.0]
            camera_rotation = [0.0, 0.0, 0.0]
            waypoint = Vector(400.0, 0.0)
            frame = annotate_image(
                np.zeros((640, 720, 3), dtype=np.uint8),
                [waypoint],
                camera_location,
                camera_rotation,
                70.0,
            )
            frame.save(directory / "frame.png")
            (directory / "step_0001_manifest.json").write_text(
                json.dumps(
                    {
                        "traffic_policy": "visual_only",
                        "artifact_schema_version": "pre_post_camera_v1",
                        "camera": {
                            "mode": "first_person_free_follow",
                            "resolution": [720, 640],
                            "location": camera_location,
                            "rotation": camera_rotation,
                            "scope": "policy_input",
                        },
                        "input_camera": {
                            "mode": "first_person_free_follow",
                            "resolution": [720, 640],
                            "location": camera_location,
                            "rotation": camera_rotation,
                            "scope": "policy_input",
                        },
                        "output_camera": {
                            "mode": "first_person_free_follow",
                            "resolution": [720, 640],
                            "location": camera_location,
                            "rotation": camera_rotation,
                            "scope": "post_action_output",
                        },
                        "input_images": ["frame.png"],
                        "output_image": "frame.png",
                        "prompt": {
                            "system": (
                                "red when they fall inside the first-person "
                                "camera frame"
                            ),
                            "user": (
                                "Near or side projections can be clipped by "
                                "the first-person camera"
                            ),
                            "traffic_light_context": "",
                        },
                        "observation_geometry": {
                            "agent_position_cm": {"x": 0.0, "y": 0.0},
                            "agent_direction": {"x": 1.0, "y": 0.0},
                            "camera_location_cm": camera_location,
                            "camera_rotation_deg": camera_rotation,
                            "camera_horizontal_fov_deg": 70.0,
                            "annotated_candidates": [
                                {
                                    "index": 5,
                                    "world_position_cm": {
                                        "x": waypoint.x,
                                        "y": waypoint.y,
                                    },
                                }
                            ],
                        },
                        "route_adherence": valid_adherence(),
                    }
                ),
                encoding="utf-8",
            )
            (directory / "step_0001_traffic_light_snapshot.json").write_text(
                json.dumps(
                    {
                        "execution_override": None,
                        "model_input": {"agent_on_crosswalk": False},
                    }
                ),
                encoding="utf-8",
            )

            errors, summary = audit_steps(root, "any")

        self.assertEqual([], errors)
        self.assertEqual(1, summary["camera_projection_checks"])
        self.assertEqual(1, summary["visible_marker_checks"])
        self.assertEqual(1, summary["image_resolution_checks"])

    def test_all_visible_markers_replace_obsolete_clipping_disclosure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "step_0001_decision_0001"
            directory.mkdir()
            (directory / "frame.png").write_bytes(b"png")
            (directory / "step_0001_manifest.json").write_text(
                json.dumps(
                    {
                        "traffic_policy": "visual_only",
                        "input_images": ["frame.png"],
                        "output_image": "frame.png",
                        "prompt": {
                            "system": (
                                "red when they fall inside the first-person "
                                "camera frame"
                            ),
                            "user": (
                                "The last image is your current view with all "
                                "seven candidate waypoint projections marked in red."
                            ),
                            "traffic_light_context": "",
                        },
                        "observation_geometry": {
                            "all_waypoint_markers_visible": True,
                        },
                        "route_adherence": valid_adherence(),
                    }
                ),
                encoding="utf-8",
            )
            (directory / "step_0001_traffic_light_snapshot.json").write_text(
                json.dumps(
                    {
                        "execution_override": None,
                        "model_input": {"agent_on_crosswalk": False},
                    }
                ),
                encoding="utf-8",
            )

            errors, _ = audit_steps(root, "any")

        self.assertFalse(
            any("marker-visibility contract" in error for error in errors)
        )

    def test_any_step_audit_reports_route_deviation_without_rejecting_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "step_0001_decision_0001"
            directory.mkdir()
            (directory / "frame.png").write_bytes(b"png")
            adherence = valid_adherence()
            adherence.update(
                {"lateral_distance_cm": 300.0, "within_corridor": False}
            )
            (directory / "step_0001_manifest.json").write_text(
                json.dumps(
                    {
                        "traffic_policy": "visual_only",
                        "camera": {"mode": "first_person_free_follow"},
                        "input_images": ["frame.png"],
                        "output_image": "frame.png",
                        "prompt": {"traffic_light_context": ""},
                        "route_adherence": adherence,
                    }
                ),
                encoding="utf-8",
            )
            (directory / "step_0001_traffic_light_snapshot.json").write_text(
                json.dumps(
                    {
                        "execution_override": None,
                        "model_input": {"agent_on_crosswalk": False},
                    }
                ),
                encoding="utf-8",
            )

            any_errors, summary = audit_steps(root, "any")
            safe_errors, safe_summary = audit_steps(
                root, "safe", route_has_crosswalk=False
            )
            illegal_collision_errors, _ = audit_steps(
                root,
                "collision",
                route_has_crosswalk=True,
                require_route_crosswalk_sample=False,
            )

        self.assertEqual([], any_errors)
        self.assertEqual(1, summary["route_deviation_manifest_count"])
        self.assertEqual([], safe_errors)
        self.assertEqual(1, safe_summary["route_deviation_manifest_count"])
        self.assertNotIn(
            "no first-person step was recorded on a route crosswalk",
            illegal_collision_errors,
        )


if __name__ == "__main__":
    unittest.main()
