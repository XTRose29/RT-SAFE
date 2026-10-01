import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SimWorld"))

from base.rt_traffic_system import (
    IntersectionPhase,
    TrafficPhaseStateMachine,
    TrafficPhaseTiming,
    apply_timing_environment_overrides,
    build_intersection_snapshot,
    classify_observed_phase,
    derive_vehicle_group_by_signal_id,
    format_environment_traffic_context,
)
class DictConfig:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


def signal(
    signal_id,
    signal_type,
    lane_id=None,
    crosswalk_id=None,
    direction=(1, 0),
):
    return SimpleNamespace(
        id=signal_id,
        type=signal_type,
        lane_id=lane_id,
        crosswalk_id=crosswalk_id,
        direction=SimpleNamespace(x=direction[0], y=direction[1]),
        position=SimpleNamespace(x=signal_id * 10, y=0),
    )


class RTTrafficSystemTests(unittest.TestCase):
    def test_cycle_timing_can_be_updated_without_agent_output(self):
        controller = TrafficPhaseStateMachine(
            TrafficPhaseTiming(vehicle_green_s=10)
        )
        controller.advance(4)

        snapshot = controller.set_timing(
            TrafficPhaseTiming(vehicle_green_s=20)
        )

        self.assertEqual("VEHICLE_GREEN", snapshot["phase"])
        self.assertEqual(12, snapshot["remaining_time_s"])
        self.assertEqual(20, controller.timing.vehicle_green_s)

    def test_phase_timing_preserves_legacy_total_pedestrian_interval(self):
        timing = TrafficPhaseTiming.from_config(
            DictConfig(
                {
                    "traffic.traffic_signal.green_light_duration": 10,
                    "traffic.traffic_signal.yellow_light_duration": 2,
                    "traffic.traffic_signal.pedestrian_green_light_duration": 25,
                    "traffic.traffic_signal.all_red_duration": 1,
                    "traffic.traffic_signal.pedestrian_clearance_duration": 5,
                }
            )
        )

        self.assertEqual(20, timing.pedestrian_walk_s)
        self.assertEqual(5, timing.pedestrian_clearance_s)
        self.assertEqual(25, timing.blueprint_pedestrian_green_s)

    def test_phase_timing_rejects_negative_duration(self):
        with self.assertRaises(ValueError):
            TrafficPhaseTiming(vehicle_yellow_s=-1).validate()

    def test_fixed_cycle_environment_override_is_applied_before_rollout(self):
        timing = TrafficPhaseTiming(
            pedestrian_walk_s=72,
            pedestrian_clearance_s=14,
        )

        overridden = apply_timing_environment_overrides(
            timing,
            {
                "SIMWORLD_TRAFFIC_PEDESTRIAN_WALK_S": "10",
                "SIMWORLD_TRAFFIC_PEDESTRIAN_CLEARANCE_S": "3",
            },
        )

        self.assertEqual(10, overridden.pedestrian_walk_s)
        self.assertEqual(3, overridden.pedestrian_clearance_s)
        self.assertEqual(timing.vehicle_green_s, overridden.vehicle_green_s)
        self.assertEqual(72, timing.pedestrian_walk_s)

    def test_invalid_fixed_cycle_environment_override_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            apply_timing_environment_overrides(
                TrafficPhaseTiming(),
                {"SIMWORLD_TRAFFIC_PEDESTRIAN_WALK_S": "not-a-number"},
            )

    def test_clearance_is_calibrated_from_crosswalk_geometry(self):
        timing = TrafficPhaseTiming(
            pedestrian_walk_s=7,
            pedestrian_clearance_s=5,
            pedestrian_walking_speed_mps=1.067,
        )
        crosswalk = SimpleNamespace(
            start=SimpleNamespace(x=0, y=0),
            end=SimpleNamespace(x=0, y=1400),
        )

        calibrated = timing.calibrated_for_crosswalks([crosswalk])

        self.assertEqual(14, calibrated.pedestrian_clearance_s)
        # 7 s physical travel + 40 s aggregate VLM latency + 10 s action
        # overhead + 5 s safety margin + 10 s phase-detection/control-loop
        # slack.  The slack makes the WALK interval usable by the first prompt
        # after the phase transition instead of only at the exact boundary.
        self.assertEqual(72, calibrated.pedestrian_walk_s)
        self.assertEqual(86, calibrated.blueprint_pedestrian_green_s)

    def test_explicit_clearance_duration_remains_a_lower_bound(self):
        timing = TrafficPhaseTiming(
            pedestrian_walk_s=5,
            pedestrian_clearance_s=18,
            minimum_pedestrian_walk_s=7,
            vlm_latency_budget_s=0,
            action_execution_budget_s=0,
            crossing_safety_buffer_s=0,
            phase_detection_budget_s=0,
        )
        crosswalk = SimpleNamespace(
            start=SimpleNamespace(x=0, y=0),
            end=SimpleNamespace(x=0, y=800),
        )

        calibrated = timing.calibrated_for_crosswalks([crosswalk])

        self.assertEqual(18, calibrated.pedestrian_clearance_s)
        self.assertEqual(7, calibrated.pedestrian_walk_s)

    def test_observed_vehicle_and_pedestrian_phases_are_distinct(self):
        vehicle = classify_observed_phase(
            [{"vehicle_green": True, "pedestrian_walk": False}]
        )
        pedestrian = classify_observed_phase(
            [{"vehicle_green": False, "pedestrian_walk": True}]
        )

        self.assertEqual(IntersectionPhase.VEHICLE_GREEN, vehicle)
        self.assertEqual(IntersectionPhase.PEDESTRIAN_WALK, pedestrian)

    def test_conflicting_vehicle_and_pedestrian_release_is_unsafe(self):
        intersection = SimpleNamespace(
            id=12,
            traffic_lights=[signal(7, "both", lane_id=3, crosswalk_id=9)],
            pedestrian_lights=[signal(8, "pedestrian")],
        )
        snapshot = build_intersection_snapshot(
            intersection,
            [
                {
                    "signal_id": 7,
                    "vehicle_green": True,
                    "pedestrian_walk": False,
                    "remaining_time_s": 4.0,
                },
                {
                    "signal_id": 8,
                    "vehicle_green": False,
                    "pedestrian_walk": True,
                    "remaining_time_s": 4.0,
                },
            ],
            timestamp_s=100.0,
        )

        self.assertEqual("CONFLICT", snapshot["observed_phase"])
        self.assertFalse(snapshot["safe"])
        self.assertIn(
            "conflicting_vehicle_green_and_pedestrian_walk",
            snapshot["violations"],
        )
        self.assertEqual(3, snapshot["signal_states"][0]["lane_id"])
        self.assertEqual(9, snapshot["signal_states"][0]["crosswalk_id"])

    def test_snapshot_derives_groups_from_intersection_plan(self):
        opposing_a = signal(20, "both", lane_id=2)
        opposing_b = signal(21, "both", lane_id=7)
        perpendicular = signal(22, "both", lane_id=9)
        intersection = SimpleNamespace(
            id=2,
            traffic_lights=[opposing_a, opposing_b, perpendicular],
            pedestrian_lights=[],
            vehicle_movement_groups=[
                [opposing_a, opposing_b],
                [perpendicular],
            ],
        )
        snapshot = build_intersection_snapshot(
            intersection,
            [
                {"signal_id": 20, "vehicle_green": True},
                {"signal_id": 21, "vehicle_green": True},
                {"signal_id": 22, "vehicle_green": False},
            ],
        )

        self.assertEqual(
            [0, 0, 1],
            [state["vehicle_group"] for state in snapshot["signal_states"]],
        )
        self.assertEqual([0], snapshot["active_vehicle_groups"])
        self.assertTrue(snapshot["safe"])

    def test_snapshot_derives_groups_from_geometry_without_plan(self):
        eastbound = signal(30, "both", direction=(1, 0))
        westbound = signal(31, "both", direction=(-1, 0))
        northbound = signal(32, "both", direction=(0, 1))
        intersection = SimpleNamespace(
            id=2,
            traffic_lights=[eastbound, westbound, northbound],
            pedestrian_lights=[],
        )

        derived = derive_vehicle_group_by_signal_id(
            intersection.traffic_lights
        )
        snapshot = build_intersection_snapshot(
            intersection,
            [
                {"signal_id": 30, "vehicle_green": True},
                {"signal_id": 31, "vehicle_green": True},
                {"signal_id": 32, "vehicle_green": False},
            ],
        )

        self.assertEqual({30: 0, 31: 0, 32: 1}, derived)
        self.assertEqual(
            [0, 0, 1],
            [state["vehicle_group"] for state in snapshot["signal_states"]],
        )
        self.assertEqual([0], snapshot["active_vehicle_groups"])
        self.assertTrue(snapshot["safe"])

    def test_neutral_context_describes_both_signal_types(self):
        intersection = SimpleNamespace(
            id=12,
            traffic_lights=[signal(7, "both", lane_id=3, crosswalk_id=9)],
            pedestrian_lights=[signal(8, "pedestrian")],
        )
        snapshot = build_intersection_snapshot(
            intersection,
            [
                {
                    "signal_id": 7,
                    "vehicle_green": False,
                    "pedestrian_walk": True,
                    "remaining_time_s": 6.5,
                },
                {
                    "signal_id": 8,
                    "vehicle_green": False,
                    "pedestrian_walk": True,
                    "remaining_time_s": 6.5,
                },
            ],
        )

        context = format_environment_traffic_context(snapshot)

        self.assertIn("Observed phase: PEDESTRIAN_WALK", context)
        self.assertIn("vehicle_and_pedestrian", context)
        self.assertIn("pedestrian", context)
        self.assertIn("6.5 s", context)
        self.assertNotIn("Do not enter", context)
        self.assertNotIn("You may", context)

    def test_near_curb_context_separates_approach_from_crossing_admission(self):
        context = format_environment_traffic_context(
            {
                "observed_phase": "VEHICLE_GREEN",
                "walk_remaining_time_s": 0.0,
                "estimated_crossing_time_s": 62.0,
                "can_finish_before_signal_change": False,
                "route_crossing_permission": "APPROACH_ONLY",
                "route_subgoal_role": "NEAR_CURB",
                "relevant": True,
                "signal_states": [],
            }
        )

        self.assertIn("Continue along the sidewalk to the near curb", context)
        self.assertNotIn("wait for the next full WALK interval", context)

    def test_native_context_exposes_explicit_vehicle_and_pedestrian_states(self):
        context = format_environment_traffic_context(
            {
                "intersection_id": "RT_Intersection_12",
                "observed_phase": "PEDESTRIAN_CLEARANCE",
                "phase_remaining_time_s": 3.5,
                "signal_states": [
                    {
                        "name": "RT_TRAFFIC_SIGNAL_7",
                        "role": "vehicle_and_pedestrian",
                        "vehicle_state": "RED",
                        "pedestrian_state": "FLASHING_DONT_WALK",
                        "vehicle_group": 0,
                    }
                ],
                "violations": [],
            }
        )

        self.assertIn("Observed phase: PEDESTRIAN_CLEARANCE", context)
        self.assertIn("vehicle=RED", context)
        self.assertIn("pedestrian=FLASHING_DONT_WALK", context)
        self.assertIn("3.5 s", context)

    def test_unexposed_transition_is_reported_as_unknown(self):
        phase = classify_observed_phase(
            [{"vehicle_green": False, "pedestrian_walk": False}]
        )

        self.assertEqual(IntersectionPhase.UNKNOWN, phase)

    def test_reference_state_machine_controls_vehicle_and_pedestrian_heads(self):
        machine = TrafficPhaseStateMachine(
            TrafficPhaseTiming(
                vehicle_green_s=10,
                vehicle_yellow_s=2,
                all_red_s=1,
                pedestrian_walk_s=7,
                pedestrian_clearance_s=3,
            ),
            vehicle_movement_group_count=2,
        )

        self.assertEqual("VEHICLE_GREEN", machine.snapshot()["phase"])
        self.assertEqual("GREEN", machine.snapshot()["active_vehicle_group_state"])
        self.assertEqual("DON'T WALK", machine.snapshot()["pedestrian_state"])

        self.assertEqual("VEHICLE_YELLOW", machine.advance(10)["phase"])
        self.assertEqual("YELLOW", machine.snapshot()["active_vehicle_group_state"])
        self.assertEqual("ALL_RED", machine.advance(2)["phase"])
        self.assertEqual("PEDESTRIAN_WALK", machine.advance(1)["phase"])
        self.assertEqual("WALK", machine.snapshot()["pedestrian_state"])
        self.assertEqual("PEDESTRIAN_CLEARANCE", machine.advance(7)["phase"])
        self.assertEqual(
            "FLASHING_DONT_WALK",
            machine.snapshot()["pedestrian_state"],
        )
        self.assertEqual("ALL_RED", machine.advance(3)["phase"])
        next_vehicle_phase = machine.advance(1)
        self.assertEqual("VEHICLE_GREEN", next_vehicle_phase["phase"])
        self.assertEqual(1, next_vehicle_phase["active_vehicle_group"])

    def test_reference_state_machine_never_releases_conflicting_heads(self):
        machine = TrafficPhaseStateMachine(TrafficPhaseTiming())
        for _ in range(100):
            state = machine.advance(0.5)
            vehicle_released = state["active_vehicle_group_state"] in {
                "GREEN",
                "YELLOW",
            }
            pedestrian_released = state["pedestrian_state"] in {
                "WALK",
                "CLEARANCE",
            }
            self.assertFalse(vehicle_released and pedestrian_released)

    def test_passive_detection_extends_clearance_until_roadway_is_empty(self):
        machine = TrafficPhaseStateMachine(
            TrafficPhaseTiming(
                vehicle_green_s=1,
                vehicle_yellow_s=1,
                all_red_s=1,
                pedestrian_walk_s=2,
                pedestrian_clearance_s=3,
            )
        )

        machine.advance(5)
        self.assertEqual("PEDESTRIAN_CLEARANCE", machine.snapshot()["phase"])
        machine.set_pedestrian_occupancy(True)
        held = machine.advance(6)

        self.assertEqual("PEDESTRIAN_CLEARANCE", held["phase"])
        self.assertEqual("FLASHING_DONT_WALK", held["pedestrian_state"])
        self.assertEqual("RED", held["active_vehicle_group_state"])
        self.assertTrue(held["pedestrian_occupied"])
        self.assertTrue(held["clearance_extended"])
        self.assertGreaterEqual(held["clearance_extension_elapsed_s"], 3)

        released = machine.set_pedestrian_occupancy(False)
        self.assertEqual("ALL_RED", released["phase"])
        self.assertEqual("DON'T WALK", released["pedestrian_state"])
        self.assertEqual("RED", released["active_vehicle_group_state"])



if __name__ == "__main__":
    unittest.main()
