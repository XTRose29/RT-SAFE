import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SimWorld"))

from simworld.traffic.base.traffic_signal import TrafficSignalState
from simworld.traffic.base.intersection import Intersection
from simworld.traffic.base.traffic_signal import TrafficSignal
from simworld.traffic.manager.intersection_manager import IntersectionManager
from simworld.utils.vector import Vector


class LegacyIntersectionClearanceTests(unittest.TestCase):
    def _light(self, signal_id, lane_id, state):
        light = SimpleNamespace(id=signal_id, lane_id=lane_id, state=state)
        light.get_state = lambda: light.state
        return light

    def test_vehicle_green_waits_for_pedestrian_head_red_readback(self):
        vehicle_head = self._light(
            1,
            6,
            (
                TrafficSignalState.VEHICLE_RED,
                TrafficSignalState.PEDESTRIAN_RED,
            ),
        )
        pedestrian_head = self._light(
            2,
            None,
            (
                TrafficSignalState.VEHICLE_RED,
                TrafficSignalState.PEDESTRIAN_GREEN,
            ),
        )
        intersection = SimpleNamespace(
            id=12,
            traffic_lights=[vehicle_head],
            pedestrian_lights=[pedestrian_head],
            cycle_count=0,
            all_traffic_lights_red=lambda: True,
        )
        intersection.increment_cycle_count = Mock()

        manager = IntersectionManager.__new__(IntersectionManager)
        manager.intersections = [intersection]
        manager.python_states = {}
        manager.logger = Mock()
        communicator = Mock()

        manager.update_intersections(communicator)

        communicator.traffic_signal_switch_to.assert_has_calls(
            [
                call(vehicle_head.id, "all red"),
                call(pedestrian_head.id, "all red"),
            ]
        )
        intersection.increment_cycle_count.assert_not_called()

        communicator.reset_mock()
        pedestrian_head.state = (
            TrafficSignalState.VEHICLE_RED,
            TrafficSignalState.PEDESTRIAN_RED,
        )

        manager.update_intersections(communicator)

        communicator.traffic_signal_switch_to.assert_called_once_with(
            vehicle_head.id, "green"
        )
        intersection.increment_cycle_count.assert_called_once_with()

    def test_opposing_heads_share_one_movement_group(self):
        intersection = Intersection.__new__(Intersection)
        intersection.traffic_lights = [
            TrafficSignal(Vector(0, 0), Vector(1, 0), lane_id=2),
            TrafficSignal(Vector(0, 0), Vector(-1, 0), lane_id=7),
            TrafficSignal(Vector(0, 0), Vector(0, 1), lane_id=9),
        ]
        intersection.vehicle_movement_groups = []

        intersection._build_vehicle_movement_groups()

        self.assertEqual(
            [[light.lane_id for light in group]
             for group in intersection.vehicle_movement_groups],
            [[2, 7], [9]],
        )
        self.assertEqual(
            [light.vehicle_group for light in intersection.traffic_lights],
            [0, 0, 1],
        )

    def test_group_waits_for_green_readback_then_releases_next_group(self):
        red = (
            TrafficSignalState.VEHICLE_RED,
            TrafficSignalState.PEDESTRIAN_RED,
        )
        green = (
            TrafficSignalState.VEHICLE_GREEN,
            TrafficSignalState.PEDESTRIAN_RED,
        )
        opposing_a = self._light(10, 2, red)
        opposing_b = self._light(11, 7, red)
        perpendicular = self._light(12, 9, red)
        intersection = SimpleNamespace(
            id=2,
            traffic_lights=[opposing_a, opposing_b, perpendicular],
            pedestrian_lights=[],
            vehicle_movement_groups=[
                [opposing_a, opposing_b],
                [perpendicular],
            ],
            cycle_count=0,
            active_vehicle_group_index=None,
            active_vehicle_group_seen_green=False,
        )
        intersection.all_traffic_lights_red = lambda: all(
            light.state == red for light in intersection.traffic_lights
        )
        intersection.increment_cycle_count = lambda: setattr(
            intersection, "cycle_count", intersection.cycle_count + 1
        )
        intersection.has_completed_cycle = lambda: (
            intersection.cycle_count >= 2
        )
        intersection.reset_cycle_count = lambda: setattr(
            intersection, "cycle_count", 0
        )

        manager = IntersectionManager.__new__(IntersectionManager)
        manager.intersections = [intersection]
        manager.python_states = {}
        manager.logger = Mock()
        communicator = Mock()

        manager.update_intersections(communicator)
        communicator.traffic_signal_switch_to.assert_has_calls(
            [call(10, "green"), call(11, "green")]
        )
        self.assertEqual(intersection.cycle_count, 1)

        # UE has not acknowledged green yet: do not skip the first group.
        communicator.reset_mock()
        manager.update_intersections(communicator)
        communicator.traffic_signal_switch_to.assert_not_called()

        opposing_a.state = green
        opposing_b.state = green
        manager.update_intersections(communicator)
        opposing_a.state = red
        opposing_b.state = red
        manager.update_intersections(communicator)

        communicator.traffic_signal_switch_to.assert_called_once_with(
            12, "green"
        )
        self.assertEqual(intersection.cycle_count, 2)

    def test_timed_group_preserves_yellow_and_all_red_before_next_release(self):
        red = (
            TrafficSignalState.VEHICLE_RED,
            TrafficSignalState.PEDESTRIAN_RED,
        )
        green = (
            TrafficSignalState.VEHICLE_GREEN,
            TrafficSignalState.PEDESTRIAN_RED,
        )
        first_a = self._light(20, 2, red)
        first_b = self._light(21, 7, red)
        second = self._light(22, 9, red)
        intersection = SimpleNamespace(
            id=2,
            traffic_lights=[first_a, first_b, second],
            pedestrian_lights=[],
            vehicle_movement_groups=[[first_a, first_b], [second]],
            cycle_count=0,
            active_vehicle_group_index=None,
            active_vehicle_group_seen_green=False,
            active_vehicle_group_elapsed_s=0.0,
        )
        intersection.all_traffic_lights_red = lambda: all(
            light.state == red for light in intersection.traffic_lights
        )
        intersection.increment_cycle_count = lambda: setattr(
            intersection, "cycle_count", intersection.cycle_count + 1
        )
        intersection.has_completed_cycle = lambda: (
            intersection.cycle_count >= 2
        )
        intersection.reset_cycle_count = lambda: setattr(
            intersection, "cycle_count", 0
        )

        manager = IntersectionManager.__new__(IntersectionManager)
        manager.intersections = [intersection]
        manager.python_states = {}
        manager.logger = Mock()
        manager.config = {
            "traffic.traffic_signal.green_light_duration": 2,
            "traffic.traffic_signal.yellow_light_duration": 1,
            "traffic.traffic_signal.all_red_duration": 1,
        }
        communicator = Mock()

        manager.update_intersections(communicator, delta_time_s=0)
        communicator.reset_mock()
        first_a.state = green
        first_b.state = green
        manager.update_intersections(communicator, delta_time_s=1)
        manager.update_intersections(communicator, delta_time_s=1)
        first_a.state = red
        first_b.state = red
        manager.update_intersections(communicator, delta_time_s=1)
        communicator.traffic_signal_switch_to.assert_not_called()

        manager.update_intersections(communicator, delta_time_s=1)
        communicator.traffic_signal_switch_to.assert_called_once_with(
            22, "green"
        )


if __name__ == "__main__":
    unittest.main()
