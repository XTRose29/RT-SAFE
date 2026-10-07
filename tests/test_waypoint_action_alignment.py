import math
import unittest
from types import SimpleNamespace

from base.rt_action_space import MOVE_TO, WAIT, RTActionSpace
from base.rt_agent import POLICY_ACTION_FRAME_INTERVAL_S, RTAgent
from base.rt_communicator import RTCommunicator
from simworld.utils.vector import Vector


class WaypointActionAlignmentTests(unittest.TestCase):
    def make_agent(self):
        agent = RTAgent.__new__(RTAgent)
        agent.position = Vector(0.0, 0.0)
        agent._direction = Vector(1.0, 0.0)
        agent._yaw = 0.0
        agent.speed = 200.0
        agent.fixed_action_seconds = None
        agent.use_action_frames = False
        agent.communicator = SimpleNamespace(rt_agent_move_to=lambda *args: None)
        agent.name = "RT_AGENT"
        agent._advance_simulation_time = lambda seconds: None
        agent.last_selected_image_waypoint = None
        agent.last_commanded_waypoint = None
        agent.last_move_command_duration_seconds = None
        agent.last_move_execution_mode = None
        return agent

    def test_non_turn_action_clears_previous_turn_execution_metadata(self):
        agent = self.make_agent()
        agent.last_turn_execution = {"requested_param": "L90"}
        agent.last_action_override = None
        agent.last_execution_traffic_light_snapshot = None
        agent.recording_action_frames = []
        agent.wait_count = 0
        agent.wait = lambda duration: float(duration)

        success, record = agent.take_action(
            RTActionSpace(action_type=WAIT, action_param="1"),
            [],
        )

        self.assertTrue(success)
        self.assertEqual("Wait 1s", record)
        self.assertIsNone(agent.last_turn_execution)

    def test_action_frame_policy_input_is_chronological_and_ends_current(self):
        agent = self.make_agent()
        first = object()
        second = object()
        current = object()
        agent.use_action_frames = True
        agent.action_frames = [first, None, second]
        agent.image_history = [object()]

        images = agent._decision_input_images(current)

        self.assertEqual([first, second, current], images)

    def test_action_frame_policy_input_has_no_benchmark_count_cap(self):
        agent = self.make_agent()
        frames = [object() for _ in range(12)]
        current = object()
        agent.use_action_frames = True
        agent.action_frames = frames

        images = agent._decision_input_images(current)

        self.assertEqual(frames + [current], images)

    def test_policy_action_frames_default_to_half_second_cadence(self):
        self.assertEqual(0.5, POLICY_ACTION_FRAME_INTERVAL_S)

    def test_numbered_waypoint_distances_and_angles_match_prompt(self):
        agent = self.make_agent()
        waypoints = agent._find_waypoints()
        expected = [
            (100.0, 0.0),
            (200.0, 0.0),
            (200.0, 45.0),
            (200.0, -45.0),
            (400.0, 0.0),
            (400.0, 45.0),
            (400.0, -45.0),
        ]
        self.assertEqual(7, len(waypoints))
        for waypoint, (expected_distance, expected_angle) in zip(waypoints, expected):
            distance = math.hypot(waypoint.x, waypoint.y)
            # Prompt angles are positive to the visual left, which is UE -Y
            # when facing +X (left-handed world).
            angle = -math.degrees(math.atan2(waypoint.y, waypoint.x))
            self.assertAlmostEqual(expected_distance, distance, delta=0.01)
            self.assertAlmostEqual(expected_angle, angle, delta=0.01)

    def test_four_meter_move_commands_rendered_target_for_two_seconds(self):
        agent = self.make_agent()
        captured = {}

        def capture(name, waypoint, duration):
            captured.update(name=name, waypoint=waypoint, duration=duration)

        agent.communicator.rt_agent_move_to = capture
        waypoint = agent._find_waypoints()[4]
        action_seconds = agent.move_to(waypoint)

        self.assertIs(waypoint, captured["waypoint"])
        self.assertAlmostEqual(2.0, captured["duration"])
        self.assertAlmostEqual(2.0, action_seconds)
        self.assertAlmostEqual(waypoint.x, agent.last_commanded_waypoint.x)
        self.assertAlmostEqual(waypoint.y, agent.last_commanded_waypoint.y)

    def test_execution_audit_confirms_selected_dot_and_command_match(self):
        agent = self.make_agent()
        selected = agent._find_waypoints()[4]
        agent.last_selected_image_waypoint = selected
        agent.last_commanded_waypoint = Vector(selected.x, selected.y)
        agent.last_move_command_duration_seconds = 2.0
        agent.position = Vector(selected.x, selected.y)

        record = agent._waypoint_execution_record(
            RTActionSpace(action_type=MOVE_TO, action_param="5"),
            Vector(0.0, 0.0),
        )

        self.assertTrue(record["selected_waypoint_matches_command"])
        self.assertTrue(record["reached_selected_waypoint"])
        self.assertAlmostEqual(400.0, record["selected_waypoint_distance_cm"])
        self.assertAlmostEqual(0.0, record["selected_waypoint_endpoint_error_cm"])

    def test_uniform_rollout_mode_ends_at_rendered_waypoint(self):
        agent = self.make_agent()
        positions = []
        stops = []
        agent.uniform_greedy_movement = True
        agent.communicator.unrealcv = SimpleNamespace(
            get_location=lambda name: (0.0, 0.0, 100.0),
            set_location=lambda location, name: positions.append(tuple(location)),
            humanoid_stop=lambda name: stops.append(name),
        )
        agent._advance_simulation_time = lambda seconds: None
        waypoint = agent._find_waypoints()[4]

        action_seconds = agent.move_to(waypoint)

        self.assertAlmostEqual(2.0, action_seconds)
        self.assertEqual((400.0, 0.0, 100.0), positions[-1])
        self.assertEqual(["RT_AGENT"], stops)
        self.assertEqual(
            "uniform_world_space_interpolation",
            agent.last_move_execution_mode,
        )

    def test_subgoal_angle_sign_matches_image_and_turn_commands(self):
        agent = self.make_agent()
        # Facing UE +X, the visual left is -Y (left-handed world).
        left_target = Vector(0.0, -100.0)
        right_target = Vector(0.0, 100.0)

        self.assertAlmostEqual(90.0, agent._relative_angle_to(left_target))
        self.assertAlmostEqual(-90.0, agent._relative_angle_to(right_target))
        self.assertEqual("L90", agent._turn_toward_target_action(left_target).action_param)
        self.assertEqual("R90", agent._turn_toward_target_action(right_target).action_param)

    def test_turn_execution_matches_public_left_right_convention(self):
        calls = []
        communicator = RTCommunicator.__new__(RTCommunicator)
        communicator.unrealcv = SimpleNamespace(
            p_rotate=lambda name, angle, direction: calls.append(
                (name, angle, direction)
            )
        )

        communicator.rt_agent_turn_around("RT_AGENT", 30, clockwise=False)
        communicator.rt_agent_turn_around("RT_AGENT", 60, clockwise=True)

        # p_rotate 'right' raises UE yaw, which turns the view right.
        self.assertEqual(
            [("RT_AGENT", 30, "left"), ("RT_AGENT", 60, "right")],
            calls,
        )

    def test_turn_postcondition_corrects_opposite_blueprint_result(self):
        agent = self.make_agent()
        agent.position = Vector(100.0, 200.0)
        orientation = [0.0, 10.0, 0.0]
        location = [100.0, 200.0, 90.0]

        def get_orientation(name):
            return tuple(orientation)

        def set_orientation(value, name):
            orientation[:] = value

        def get_location(name):
            return tuple(location)

        def set_location(value, name):
            location[:] = value

        def raw_opposite_turn(name, angle, clockwise):
            # Reproduce the intermittent packaged-Blueprint failure observed
            # on a road overlap: R30 returned the opposite (-30 degree) yaw
            # change and the nominally in-place action walked roughly two
            # metres.
            orientation[1] -= 30.0
            location[0] += 120.0
            location[1] += 160.0

        agent._last_ue_orientation = tuple(orientation)
        agent.communicator = SimpleNamespace(
            unrealcv=SimpleNamespace(
                get_orientation=get_orientation,
                set_orientation=set_orientation,
                get_location=get_location,
                set_location=set_location,
            ),
            rt_agent_turn_around=raw_opposite_turn,
        )

        agent.turn_around(30, clockwise=True)

        self.assertAlmostEqual(40.0, orientation[1])
        self.assertAlmostEqual(-30.0, agent.last_turn_execution[
            "raw_blueprint_delta_from_input_deg"
        ])
        self.assertAlmostEqual(30.0, agent.last_turn_execution[
            "verified_delta_from_input_deg"
        ])
        self.assertTrue(agent.last_turn_execution[
            "post_turn_correction_applied"
        ])
        self.assertEqual([100.0, 200.0, 90.0], location)
        self.assertAlmostEqual(
            200.0,
            agent.last_turn_execution["raw_blueprint_position_drift_cm"],
        )
        self.assertAlmostEqual(
            0.0,
            agent.last_turn_execution["verified_position_drift_cm"],
        )
        self.assertTrue(agent.last_turn_execution[
            "post_turn_position_correction_applied"
        ])
        self.assertEqual(
            {"x": 100.0, "y": 200.0},
            agent.last_turn_execution["observation_position_cm"],
        )
        self.assertEqual(
            {"x": 100.0, "y": 200.0},
            agent.last_turn_execution["verified_end_position_cm"],
        )
        self.assertAlmostEqual(
            0.0,
            agent.last_turn_execution[
                "command_start_position_drift_from_input_cm"
            ],
        )

    def test_ordered_route_turn_locks_root_motion_each_controller_interval(self):
        agent = self.make_agent()
        agent.position = Vector(100.0, 200.0)
        agent.task_edges = [(Vector(0.0, 0.0), Vector(1.0, 0.0))]
        agent.first_person_actor_base_height_cm = None
        location = [100.0, 200.0, 90.0]
        orientation = [0.0, 10.0, 0.0]
        advances = []

        def set_location(value, name):
            location[:] = value

        def set_orientation(value, name):
            orientation[:] = value

        def advance(seconds):
            advances.append(seconds)
            location[0] += 100.0

        agent.communicator = SimpleNamespace(
            unrealcv=SimpleNamespace(
                get_location=lambda name: tuple(location),
                set_location=set_location,
                get_orientation=lambda name: tuple(orientation),
                set_orientation=set_orientation,
            ),
            rt_agent_move_to=lambda *args: None,
            rt_agent_turn_around=lambda name, angle, clockwise: (
                orientation.__setitem__(1, orientation[1] + angle)
            ),
        )
        agent._advance_simulation_time = advance

        agent.turn_around(30, clockwise=True)

        self.assertEqual([0.25, 0.25, 0.25, 0.25], advances)
        self.assertEqual([100.0, 200.0, 90.0], location)
        self.assertAlmostEqual(30.0, orientation[1])
        self.assertAlmostEqual(
            100.0,
            agent.last_turn_execution["raw_blueprint_position_drift_cm"],
        )
        self.assertAlmostEqual(
            0.0,
            agent.last_turn_execution["verified_position_drift_cm"],
        )
        self.assertTrue(
            agent.last_turn_execution[
                "post_turn_position_correction_applied"
            ]
        )


if __name__ == "__main__":
    unittest.main()
