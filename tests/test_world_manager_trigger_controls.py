import inspect
import math
import random
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from types import ModuleType, SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

simworld = ModuleType("simworld")
simworld.__path__ = [str(REPO_ROOT / "SimWorld" / "simworld")]
sys.modules["simworld"] = simworld

from manager.world_manager import (
    LEVEL_DIFFICULTY_CONFIGS,
    SCRIPTED_PEDESTRIAN_MOTION_POLICY,
    WorldManager,
    build_scripted_pedestrian_recovery_detour,
    build_scripted_pedestrian_patrol_loop,
    offset_scripted_pedestrian_route,
    rotate_scripted_pedestrian_waypoints_from_position,
    summarize_conflict_vehicle_events,
)
from simworld.utils.vector import Vector


class WorldManagerTriggerControlTests(unittest.TestCase):
    CASES = (
        ("pedestrians_enabled", "_get_pedestrian_activation_ratio"),
        ("movable_obstacles_enabled", "_get_movable_obstacle_activation_ratio"),
        ("irregular_npcs_enabled", "_get_irregular_activation_ratio"),
        ("falling_objects_enabled", "_get_falling_object_activation_ratio"),
    )

    @classmethod
    def make_manager(cls):
        manager = WorldManager.__new__(WorldManager)
        manager.difficulty = "easy"
        manager.difficulty_config = {
            "pedestrian_ratio": 0.6,
            "movable_obstacle_ratio": 0.6,
            "irregular_ratio": 0.6,
            "falling_ratio": 0.6,
        }
        for control, _ in cls.CASES:
            setattr(manager, control, True)
        return manager

    def test_controls_override_only_the_selected_ratio(self):
        for disabled_control, disabled_method in self.CASES:
            with self.subTest(control=disabled_control):
                manager = self.make_manager()
                self.assertEqual(getattr(manager, disabled_method)(), 0.6)
                setattr(manager, disabled_control, False)
                self.assertEqual(getattr(manager, disabled_method)(), 0.0)

                for control, method in self.CASES:
                    if control != disabled_control:
                        self.assertEqual(getattr(manager, method)(), 0.6)

    def test_legacy_difficulty_ratios_remain_60_80_100_percent(self):
        for difficulty, expected in (
            ("easy", 0.6),
            ("medium", 0.8),
            ("default", 1.0),
        ):
            manager = self.make_manager()
            manager.difficulty = difficulty
            manager.difficulty_config = {}
            for _, method in self.CASES:
                self.assertEqual(getattr(manager, method)(), expected)

    def test_legacy_benchmark_difficulties_use_uniform_pedestrian_speed(self):
        manager = self.make_manager()
        for difficulty in ("easy", "medium", "default"):
            with self.subTest(difficulty=difficulty):
                manager.difficulty = difficulty
                manager.difficulty_config = manager._get_difficulty_config()
                self.assertEqual([100], manager._get_pedestrian_speed_options())

    def test_opposing_pedestrian_flows_use_opposite_sidewalk_lanes(self):
        forward = offset_scripted_pedestrian_route(
            [Vector(0, 0), Vector(1000, 0)],
            110.0,
        )
        reverse = offset_scripted_pedestrian_route(
            [Vector(1000, 0), Vector(0, 0)],
            110.0,
        )

        self.assertEqual([-110.0, -110.0], [point.y for point in forward])
        self.assertEqual([110.0, 110.0], [point.y for point in reverse])

    def test_non_loop_patrol_returns_on_opposite_right_hand_lane(self):
        patrol, placement_indices = build_scripted_pedestrian_patrol_loop(
            [Vector(0, 0), Vector(1000, 0)],
            150.0,
        )

        self.assertEqual(
            [
                (0.0, -150.0),
                (1000.0, -150.0),
                (1000.0, 150.0),
                (0.0, 150.0),
            ],
            [(point.x, point.y) for point in patrol],
        )
        self.assertEqual([0, 2], placement_indices)

    def test_non_loop_patrol_never_reverses_on_same_segment(self):
        patrol, _ = build_scripted_pedestrian_patrol_loop(
            [Vector(0, 0), Vector(1000, 0), Vector(1000, 1000)],
            150.0,
        )

        patrol_arms = list(zip(patrol, patrol[1:]))
        horizontal_directions = [
            end.x - start.x
            for start, end in patrol_arms
            if abs(end.y - start.y) < 1e-6
            and abs(end.x - start.x) > 500.0
        ]
        self.assertIn(1000.0, horizontal_directions)
        self.assertIn(-1000.0, horizontal_directions)
        forward_y = next(
            start.y
            for start, end in patrol_arms
            if end.x - start.x > 500.0
        )
        return_y = next(
            start.y
            for start, end in patrol_arms
            if end.x - start.x < -500.0
        )
        self.assertAlmostEqual(-150.0, forward_y)
        self.assertAlmostEqual(150.0, return_y)

    def test_pedestrian_turn_uses_constant_segment_lanes_with_bevel(self):
        shifted = offset_scripted_pedestrian_route(
            [Vector(0, 0), Vector(1000, 0), Vector(1000, 1000)],
            110.0,
        )

        self.assertEqual(4, len(shifted))
        self.assertAlmostEqual(1000.0, shifted[1].x)
        self.assertAlmostEqual(-110.0, shifted[1].y)
        self.assertAlmostEqual(1110.0, shifted[2].x)
        self.assertAlmostEqual(0.0, shifted[2].y)
        self.assertAlmostEqual(1110.0, shifted[3].x)
        self.assertAlmostEqual(1000.0, shifted[3].y)

    def test_maintained_beveled_turn_clears_pedestrian_capsules(self):
        lane_offset = SCRIPTED_PEDESTRIAN_MOTION_POLICY[
            "right_hand_lane_offset_cm"
        ]
        blocking_distance = SCRIPTED_PEDESTRIAN_MOTION_POLICY[
            "observed_capsule_blocking_distance_cm"
        ]
        corner_clearance = math.sqrt(2.0) * lane_offset

        self.assertGreater(corner_clearance, blocking_distance)
        self.assertAlmostEqual(
            round(corner_clearance, 2),
            SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                "minimum_beveled_corner_center_clearance_cm"
            ],
        )

    def test_pedestrian_loop_offsets_closing_arm(self):
        shifted = offset_scripted_pedestrian_route(
            [
                Vector(0, 0),
                Vector(1000, 0),
                Vector(1000, 1000),
                Vector(0, 1000),
            ],
            110.0,
            is_loop=True,
        )

        self.assertEqual(8, len(shifted))
        self.assertAlmostEqual(-110.0, shifted[6].x)
        self.assertAlmostEqual(1000.0, shifted[6].y)
        self.assertAlmostEqual(-110.0, shifted[7].x)
        self.assertAlmostEqual(0.0, shifted[7].y)

    def test_stalled_scripted_controller_restarts_without_teleport(self):
        actor = SimpleNamespace(
            id=7,
            waypoints=[Vector(900.0, 200.0)],
        )

        class UnrealCV:
            position = [100.0, 200.0, 110.0]

            def get_location_batch(self, names):
                return [list(self.position) for _ in names]

        class Communicator:
            unrealcv = UnrealCV()

            def __init__(self):
                self.restarted = []
                self.waypoint_resets = []

            @staticmethod
            def get_pedestrian_name(actor_id):
                return f"RT_Pedestrian_{actor_id}"

            def start_pedestrians_simulation(self, actors):
                self.restarted.extend(actors)

            def set_pedestrians_waypoints(self, actors):
                self.waypoint_resets.extend(actors)

        manager = WorldManager.__new__(WorldManager)
        manager.pedestrians = [actor]
        manager.irregular_pedestrians = []
        manager.agent = SimpleNamespace(sim_time_elapsed=0.0)
        manager.communicator = Communicator()
        manager.static_obstacles = [
            {'id': 'RT_BENCH_1', 'type': 'bench', 'x': 400.0, 'y': 200.0},
        ]
        manager.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
        manager._scripted_pedestrians_started = True
        manager._scripted_pedestrian_motion_state = {}
        manager._scripted_pedestrian_motion_events = []
        manager._next_scripted_pedestrian_motion_sample_s = 0.0

        manager._maintain_scripted_pedestrian_motion()
        manager.agent.sim_time_elapsed = 2.0
        manager._maintain_scripted_pedestrian_motion()
        self.assertEqual([], manager.communicator.restarted)

        manager.agent.sim_time_elapsed = (
            SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                'controller_restart_after_s'
            ]
        )
        manager._maintain_scripted_pedestrian_motion()

        self.assertEqual([actor], manager.communicator.restarted)
        self.assertEqual([actor], manager.communicator.waypoint_resets)
        self.assertEqual(
            [100.0, 200.0, 110.0],
            manager.communicator.unrealcv.position,
        )
        summary = manager._scripted_pedestrian_motion_summary()
        self.assertEqual('stall_context_v1', summary['diagnostic_schema_version'])
        self.assertEqual(1, len(summary['events']))
        event = summary['events'][0]
        self.assertEqual('stalled_controller', event['trigger'])
        self.assertEqual([100.0, 200.0], event['position_cm'])
        self.assertEqual([900.0, 200.0], event['forward_waypoint_cm'])
        self.assertEqual(800.0, event['forward_waypoint_distance_cm'])
        self.assertIsNone(event['nearest_scripted_actor'])
        self.assertEqual('RT_BENCH_1', event['nearest_static_obstacle']['actor_id'])
        self.assertEqual(300.0, event['nearest_static_obstacle']['center_distance_cm'])
        self.assertEqual('reload_forward_waypoints', event['recovery_action'])

    def test_waypoint_reload_continues_from_nearest_forward_segment(self):
        waypoints = [
            Vector(0.0, 0.0),
            Vector(1000.0, 0.0),
            Vector(1000.0, 1000.0),
            Vector(0.0, 1000.0),
        ]

        continued = rotate_scripted_pedestrian_waypoints_from_position(
            waypoints,
            Vector(1000.0, 500.0),
        )
        after_endpoint = rotate_scripted_pedestrian_waypoints_from_position(
            waypoints,
            Vector(1000.0, 1000.0),
        )

        self.assertEqual(Vector(1000.0, 1000.0), continued[0])
        self.assertEqual(Vector(0.0, 1000.0), after_endpoint[0])
        self.assertCountEqual(waypoints, continued)
        self.assertCountEqual(waypoints, after_endpoint)

    def test_local_detour_physically_bypasses_static_blocker(self):
        obstacle = Vector(300.0, 0.0)
        detour = build_scripted_pedestrian_recovery_detour(
            Vector(0.0, 0.0),
            Vector(1000.0, 0.0),
            actor_id='RT_PEDESTRIAN_12',
            blocking_obstacle_position=obstacle,
            static_obstacle_positions=[obstacle],
            agent_position=Vector(5000.0, 5000.0),
            legal_point=lambda point: abs(point.y) <= 500.0,
        )

        self.assertTrue(detour)
        self.assertLessEqual(
            len(detour),
            SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                'local_detour_max_waypoints'
            ],
        )
        self.assertTrue(all(
            point.distance(obstacle) >=
            SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                'local_detour_static_clearance_cm'
            ]
            for point in detour
        ))
        self.assertTrue(all(abs(point.y) <= 500.0 for point in detour))

    def test_local_detour_is_rejected_outside_pedestrian_geometry(self):
        obstacle = Vector(300.0, 0.0)
        detour = build_scripted_pedestrian_recovery_detour(
            Vector(0.0, 0.0),
            Vector(1000.0, 0.0),
            actor_id='RT_PEDESTRIAN_12',
            blocking_obstacle_position=obstacle,
            static_obstacle_positions=[obstacle],
            agent_position=Vector(5000.0, 5000.0),
            legal_point=lambda point: False,
        )

        self.assertEqual([], detour)

    def test_stall_recovery_injects_detour_without_recreation(self):
        actor = SimpleNamespace(
            id=12,
            waypoints=[Vector(900.0, 0.0), Vector(1800.0, 0.0)],
        )

        class UnrealCV:
            position = [100.0, 0.0, 110.0]

            def get_location_batch(self, names):
                return [list(self.position) for _ in names]

        class Communicator:
            unrealcv = UnrealCV()

            def __init__(self):
                self.restarted = []
                self.waypoint_resets = []

            @staticmethod
            def get_pedestrian_name(actor_id):
                return f'RT_PEDESTRIAN_{actor_id}'

            def start_pedestrians_simulation(self, actors):
                self.restarted.extend(actors)

            def set_pedestrians_waypoints(self, actors):
                self.waypoint_resets.extend(actors)

        manager = WorldManager.__new__(WorldManager)
        manager.pedestrians = [actor]
        manager.agent = SimpleNamespace(
            sim_time_elapsed=0.0,
            position=Vector(5000.0, 5000.0),
            _is_within_authored_sidewalk=lambda point: abs(point.y) <= 500.0,
            _is_within_marked_crosswalk=lambda point: False,
        )
        manager.communicator = Communicator()
        manager.static_obstacles = [
            {'id': 'TREE', 'type': 'tree', 'x': 400.0, 'y': 0.0},
        ]
        manager.logger = SimpleNamespace(
            debug=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        )
        manager._scripted_pedestrians_started = True
        manager._scripted_pedestrian_motion_state = {}
        manager._scripted_pedestrian_motion_events = []
        manager._next_scripted_pedestrian_motion_sample_s = 0.0

        manager._maintain_scripted_pedestrian_motion()
        manager.agent.sim_time_elapsed = 6.0
        manager._maintain_scripted_pedestrian_motion()

        summary = manager._scripted_pedestrian_motion_summary()
        self.assertEqual(1, summary['local_detour_count'])
        self.assertGreater(summary['local_detour_waypoint_count'], 0)
        self.assertEqual(
            'inject_local_detour_waypoints',
            summary['events'][0]['recovery_action'],
        )
        self.assertNotEqual(Vector(900.0, 0.0), actor.waypoints[0])
        self.assertEqual([actor], manager.communicator.waypoint_resets)
        self.assertEqual([actor], manager.communicator.restarted)

    def test_two_failed_restarts_reset_in_place_and_verify_motion(self):
        actor = SimpleNamespace(
            id=9,
            position=Vector(0.0, 0.0),
            direction=Vector(1.0, 0.0),
            waypoints=[
                Vector(1000.0, 0.0),
                Vector(1000.0, 1000.0),
                Vector(0.0, 1000.0),
                Vector(0.0, 0.0),
            ],
            speed=100.0,
        )

        class UnrealCV:
            position = [500.0, 0.0, 110.0]

            def __init__(self):
                self.destroyed = []
                self.garbage_collection_count = 0
                self.live_names = {'RT_PEDESTRIAN_9'}

            def get_location_batch(self, names):
                return [list(self.position) for _ in names]

            def get_location(self, name):
                return list(self.position)

            def destroy(self, name):
                self.destroyed.append(name)

            def clean_garbage(self):
                self.garbage_collection_count += 1
                self.live_names.clear()

            def get_objects(self):
                return list(self.live_names)

        class Communicator:
            unrealcv = UnrealCV()

            def __init__(self):
                self.spawned = []
                self.restarted = []
                self.waypoint_resets = []
                self.speed_resets = []

            @staticmethod
            def get_pedestrian_name(actor_id):
                return f"RT_PEDESTRIAN_{actor_id}"

            def spawn_pedestrians(self, actors, type):
                self.spawned.append((actors[0], type))

            def start_pedestrians_simulation(self, actors):
                self.restarted.extend(actors)

            def set_pedestrians_waypoints(self, actors):
                self.waypoint_resets.extend(actors)

            def set_pedestrians_speed(self, actors):
                self.speed_resets.extend(actors)

        manager = WorldManager.__new__(WorldManager)
        manager.pedestrians = [actor]
        manager.agent = SimpleNamespace(
            sim_time_elapsed=0.0,
            position=Vector(10000.0, 10000.0),
        )
        manager.communicator = Communicator()
        manager.logger = SimpleNamespace(
            debug=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        )
        manager._scripted_pedestrians_started = True
        manager._scripted_pedestrian_motion_state = {}
        manager._next_scripted_pedestrian_motion_sample_s = 0.0

        manager._maintain_scripted_pedestrian_motion()
        manager.agent.sim_time_elapsed = 6.0
        manager._maintain_scripted_pedestrian_motion()
        self.assertEqual([], manager.communicator.spawned)

        manager.agent.sim_time_elapsed = 12.0
        manager._maintain_scripted_pedestrian_motion()

        self.assertEqual([], manager.communicator.unrealcv.destroyed)
        self.assertEqual(0, manager.communicator.unrealcv.garbage_collection_count)
        self.assertEqual([], manager.communicator.spawned)
        self.assertEqual([actor], manager.communicator.speed_resets)
        self.assertEqual([actor, actor], manager.communicator.waypoint_resets)
        self.assertEqual([actor, actor], manager.communicator.restarted)
        self.assertEqual(Vector(500.0, 0.0), actor.position)
        self.assertEqual(Vector(1000.0, 0.0), actor.waypoints[0])

        manager.communicator.unrealcv.position = [600.0, 0.0, 110.0]
        manager.agent.sim_time_elapsed = 14.0
        manager._maintain_scripted_pedestrian_motion()
        summary = manager._scripted_pedestrian_motion_summary()
        self.assertEqual(0, summary['controller_recreation_count'])
        self.assertEqual(0, summary['controller_recreation_verified_motion_count'])
        self.assertEqual(1, summary['controller_in_place_recovery_count'])
        self.assertEqual(
            1,
            summary['controller_in_place_recovery_verified_motion_count'],
        )
        self.assertFalse(
            summary['actors'][0][
                'controller_in_place_recovery_pending_verification'
            ]
        )

    def test_in_place_recovery_is_deferred_near_agent(self):
        actor = SimpleNamespace(
            id=10,
            position=Vector(0.0, 0.0),
            direction=Vector(1.0, 0.0),
            waypoints=[Vector(1000.0, 0.0), Vector(0.0, 0.0)],
            speed=100.0,
        )

        class UnrealCV:
            position = [500.0, 0.0, 110.0]

            def __init__(self):
                self.destroyed = []

            def get_location_batch(self, names):
                return [list(self.position) for _ in names]

            def destroy(self, name):
                self.destroyed.append(name)

        class Communicator:
            unrealcv = UnrealCV()

            def __init__(self):
                self.spawned = []

            @staticmethod
            def get_pedestrian_name(actor_id):
                return f"RT_PEDESTRIAN_{actor_id}"

            def spawn_pedestrians(self, actors, type):
                self.spawned.append((actors[0], type))

            def set_pedestrians_waypoints(self, actors):
                pass

            def start_pedestrians_simulation(self, actors):
                pass

        manager = WorldManager.__new__(WorldManager)
        manager.pedestrians = [actor]
        manager.agent = SimpleNamespace(
            sim_time_elapsed=0.0,
            position=Vector(600.0, 0.0),
        )
        manager.communicator = Communicator()
        manager.logger = SimpleNamespace(
            debug=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        )
        manager._scripted_pedestrians_started = True
        manager._scripted_pedestrian_motion_state = {}
        manager._next_scripted_pedestrian_motion_sample_s = 0.0

        for sim_time in (0.0, 6.0, 12.0):
            manager.agent.sim_time_elapsed = sim_time
            manager._maintain_scripted_pedestrian_motion()

        summary = manager._scripted_pedestrian_motion_summary()
        self.assertEqual([], manager.communicator.unrealcv.destroyed)
        self.assertEqual([], manager.communicator.spawned)
        self.assertEqual(0, summary['controller_recreation_deferred_count'])
        self.assertEqual(
            1,
            summary['controller_in_place_recovery_deferred_count'],
        )

    def test_reloaded_patrol_must_leave_endpoint_radius_before_recycling(self):
        actor = SimpleNamespace(
            id=11,
            waypoints=[Vector(1000.0, 0.0), Vector(0.0, 0.0)],
        )

        class UnrealCV:
            position = [0.0, 0.0, 110.0]

            def get_location_batch(self, names):
                return [list(self.position) for _ in names]

        class Communicator:
            unrealcv = UnrealCV()

            def __init__(self):
                self.restarted = []

            @staticmethod
            def get_pedestrian_name(actor_id):
                return f"RT_PEDESTRIAN_{actor_id}"

            def set_pedestrians_waypoints(self, actors):
                pass

            def start_pedestrians_simulation(self, actors):
                self.restarted.extend(actors)

        manager = WorldManager.__new__(WorldManager)
        manager.pedestrians = [actor]
        manager.agent = SimpleNamespace(sim_time_elapsed=0.0)
        manager.communicator = Communicator()
        manager.logger = SimpleNamespace(
            debug=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        )
        manager._scripted_pedestrians_started = True
        manager._scripted_pedestrian_motion_state = {}
        manager._next_scripted_pedestrian_motion_sample_s = 0.0

        manager._maintain_scripted_pedestrian_motion()
        manager.agent.sim_time_elapsed = 2.0
        manager._maintain_scripted_pedestrian_motion()
        self.assertEqual(1, len(manager.communicator.restarted))

        manager.communicator.unrealcv.position = [50.0, 0.0, 110.0]
        manager.agent.sim_time_elapsed = 4.0
        manager._maintain_scripted_pedestrian_motion()
        self.assertEqual(1, len(manager.communicator.restarted))

        manager.communicator.unrealcv.position = [200.0, 0.0, 110.0]
        manager.agent.sim_time_elapsed = 6.0
        manager._maintain_scripted_pedestrian_motion()
        self.assertTrue(
            manager._scripted_pedestrian_motion_state[
                'RT_PEDESTRIAN_11'
            ]['endpoint_recycle_armed']
        )

    def test_completed_scripted_patrol_recycles_without_stall_delay(self):
        actor = SimpleNamespace(
            id=8,
            waypoints=[Vector(100.0, 200.0), Vector(300.0, 400.0)],
        )

        class UnrealCV:
            position = [300.0, 400.0, 110.0]

            def get_location_batch(self, names):
                return [list(self.position) for _ in names]

        class Communicator:
            unrealcv = UnrealCV()

            def __init__(self):
                self.restarted = []
                self.waypoint_resets = []

            @staticmethod
            def get_pedestrian_name(actor_id):
                return f"RT_Pedestrian_{actor_id}"

            def start_pedestrians_simulation(self, actors):
                self.restarted.extend(actors)

            def set_pedestrians_waypoints(self, actors):
                self.waypoint_resets.extend(actors)

        manager = WorldManager.__new__(WorldManager)
        manager.pedestrians = [actor]
        manager.agent = SimpleNamespace(sim_time_elapsed=0.0)
        manager.communicator = Communicator()
        manager.logger = SimpleNamespace(
            debug=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        )
        manager._scripted_pedestrians_started = True
        manager._scripted_pedestrian_motion_state = {}
        manager._next_scripted_pedestrian_motion_sample_s = 0.0

        manager._maintain_scripted_pedestrian_motion()
        manager.agent.sim_time_elapsed = 2.0
        manager._maintain_scripted_pedestrian_motion()

        self.assertEqual([actor], manager.communicator.waypoint_resets)
        self.assertEqual([actor], manager.communicator.restarted)
        summary = manager._scripted_pedestrian_motion_summary()
        self.assertEqual(1, summary['endpoint_recycle_count'])
        self.assertEqual(0, summary['stalled_controller_restart_count'])


    def test_static_obstacle_config_describes_internal_geometry_not_model_input(self):
        for config in LEVEL_DIFFICULTY_CONFIGS.values():
            self.assertIn(
                "load_static_obstacles_for_internal_geometry",
                config,
            )
            self.assertNotIn(
                "include_static_obstacles_in_agent_context",
                config,
            )

    def test_conflict_vehicle_launch_probability_is_one_for_all_difficulties(self):
        for difficulty, expected in (
            ("easy", 1.0),
            ("medium", 1.0),
            ("default", 1.0),
        ):
            manager = self.make_manager()
            manager.difficulty = difficulty
            manager._red_light_conflict_vehicle_probability_override = None
            self.assertEqual(
                manager._get_conflict_vehicle_launch_probability(), expected
            )

    def test_conflict_vehicle_probability_override_is_preserved(self):
        manager = self.make_manager()
        manager.difficulty = "easy"
        manager._red_light_conflict_vehicle_probability_override = 0.25
        self.assertEqual(
            manager._get_conflict_vehicle_launch_probability(), 0.25
        )

    def test_conflict_vehicle_rngs_are_reproducible_and_task_scoped(self):
        manager = self.make_manager()
        manager.seed = 0

        manager._reset_task_conflict_rngs(0)
        task_zero = (
            manager._traffic_consequence_rng.random(),
            manager._conflict_vehicle_rng.uniform(300.0, 900.0),
        )
        manager._reset_task_conflict_rngs(0)
        repeated_task_zero = (
            manager._traffic_consequence_rng.random(),
            manager._conflict_vehicle_rng.uniform(300.0, 900.0),
        )
        manager._reset_task_conflict_rngs(8)
        task_eight = (
            manager._traffic_consequence_rng.random(),
            manager._conflict_vehicle_rng.uniform(300.0, 900.0),
        )

        self.assertEqual(task_zero, repeated_task_zero)
        self.assertNotEqual(task_zero, task_eight)

    def test_pedestrians_are_fully_configured_one_at_a_time(self):
        events = []

        class Communicator:
            def spawn_agent(self, *args, **kwargs):
                events.append(("agent",))

            def rt_agent_adjust_speed(self, *args):
                events.append(("agent_speed",))

            def spawn_pedestrians(self, pedestrians, type):
                events.append(("spawn", pedestrians[0].id, type))

            def set_pedestrians_waypoints(self, pedestrians):
                events.append(("waypoints", pedestrians[0].id))

            def set_pedestrians_speed(self, pedestrians):
                events.append(("speed", pedestrians[0].id))

            def start_pedestrians_simulation(self, pedestrians):
                events.append(("start", tuple(item.id for item in pedestrians)))

        manager = WorldManager.__new__(WorldManager)
        manager.communicator = Communicator()
        manager.agent = SimpleNamespace(name="RT_AGENT", speed=200, evaluator=None)
        manager.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
        manager.pedestrians = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
        manager.irregular_pedestrians = []
        manager.dynamic_obstacle_metadata = {}
        manager.selected_movable_obstacle_ids = []
        manager.selected_falling_object_ids = []
        manager.signal_traffic_enabled = False
        manager._remove_misplaced_lane_furniture = lambda: None
        manager._spawn_and_setup_traffic_signals = lambda: None
        manager._initialize_integrated_traffic_recorder = lambda: None

        with patch("manager.world_manager.time.sleep", return_value=None):
            manager.spawn_agents()

        self.assertEqual(
            events,
            [
                ("agent",),
                ("agent_speed",),
                ("spawn", 1, 2),
                ("waypoints", 1),
                ("speed", 1),
                ("spawn", 2, 2),
                ("waypoints", 2),
                ("speed", 2),
                ("start", (1, 2)),
            ],
        )

    def test_hard_is_human_facing_alias_for_default(self):
        manager = self.make_manager()
        self.assertEqual(manager._normalize_difficulty_label('hard'), 'default')

    def test_evaluation_result_serializes_canonical_termination_reason(self):
        source = inspect.getsource(WorldManager.save_evaluation_data)
        self.assertIn('"termination_reason": termination_reason', source)

    def test_conflict_vehicle_summary_counts_only_released_launches(self):
        summary = summarize_conflict_vehicle_events(
            [
                {
                    "event_id": "red-1",
                    "disposition": "staged",
                    "status": "staged",
                    "vehicle_id": 1,
                    "released": False,
                },
                {
                    "event_id": "red-2",
                    "disposition": "launched",
                    "status": "completed",
                    "vehicle_id": 2,
                    "released": True,
                },
            ],
            [
                {
                    "event_id": "illegal-1",
                    "disposition": "launched",
                    "status": "collision",
                    "vehicle_id": 3,
                    "released": True,
                    "collision_triggered": True,
                },
                {
                    "event_id": "illegal-2",
                    "disposition": "probability_skipped",
                    "status": "probability_skipped",
                },
            ],
        )

        self.assertEqual(4, summary["event_count"])
        self.assertEqual(2, summary["launch_count"])
        self.assertEqual(1, summary["collision_count"])
        self.assertEqual(
            {
                "launched": 2,
                "probability_skipped": 1,
                "staged": 1,
            },
            summary["disposition_counts"],
        )
        self.assertEqual(
            {
                "collision": 1,
                "completed": 1,
                "probability_skipped": 1,
                "staged": 1,
            },
            summary["status_counts"],
        )

    def test_conflict_vehicle_summary_supports_legacy_terminal_records(self):
        summary = summarize_conflict_vehicle_events(
            [{"vehicle_id": 7, "status": "collision"}],
            [{"status": "unavailable"}],
        )

        self.assertEqual(1, summary["launch_count"])
        self.assertEqual(1, summary["collision_count"])
        self.assertEqual(
            {"launched": 1, "unavailable": 1},
            summary["disposition_counts"],
        )

    def test_evaluation_result_reports_trigger_and_combined_launch_counts(self):
        source = inspect.getsource(WorldManager.save_evaluation_data)
        for field in (
            '"red_light_conflict_vehicle_launch_count"',
            '"illegal_crossing_conflict_vehicle_launch_count"',
            '"traffic_conflict_vehicle_launch_count"',
            '"traffic_conflict_vehicle_collision_count"',
            '"traffic_conflict_disposition_counts"',
            '"traffic_conflict_status_counts"',
            '"traffic_conflict_vehicle_events"',
        ):
            self.assertIn(field, source)

    def test_selected_dynamic_actors_leave_static_input_and_gain_metadata(self):
        manager = self.make_manager()
        manager.difficulty_config = {
            "movable_obstacle_ratio": 0.5,
            "falling_ratio": 0.5,
        }
        manager.movable_obstacle_ids = ["movable_1", "movable_2"]
        manager.falling_object_ids = ["falling_1", "falling_2"]
        manager.map_static_obstacles = [
            {"id": "movable_1", "type": "box", "x": 1, "y": 2, "z": 3},
            {"id": "movable_2", "type": "ball", "x": 4, "y": 5, "z": 6},
            {"id": "falling_1", "type": "box", "x": 7, "y": 8, "z": 9},
            {"id": "falling_2", "type": "box", "x": 10, "y": 11, "z": 12},
            {"id": "static_1", "type": "bench", "x": 13, "y": 14, "z": 0},
        ]
        manager.static_obstacles = list(manager.map_static_obstacles)
        manager.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

        manager._prepare_dynamic_obstacle_selection()

        selected = set(manager.selected_movable_obstacle_ids)
        selected.update(manager.selected_falling_object_ids)
        remaining = {item["id"] for item in manager.static_obstacles}

        self.assertEqual(len(manager.selected_movable_obstacle_ids), 1)
        self.assertEqual(len(manager.selected_falling_object_ids), 1)
        self.assertEqual(set(manager.dynamic_obstacle_metadata), selected)
        self.assertTrue(selected.isdisjoint(remaining))
        self.assertEqual(selected | remaining, {
            "movable_1", "movable_2", "falling_1", "falling_2", "static_1",
        })
        self.assertIn("static_1", remaining)
        for actor_id in manager.selected_movable_obstacle_ids:
            self.assertEqual(
                manager.dynamic_obstacle_metadata[actor_id]["kind"],
                "movable_obstacle",
            )
        for actor_id in manager.selected_falling_object_ids:
            self.assertEqual(
                manager.dynamic_obstacle_metadata[actor_id]["kind"],
                "falling_object",
            )

    def test_disabled_dynamic_controls_keep_actors_in_static_input(self):
        manager = self.make_manager()
        manager.movable_obstacles_enabled = False
        manager.falling_objects_enabled = False
        manager.movable_obstacle_ids = ["movable_1"]
        manager.falling_object_ids = ["falling_1"]
        manager.map_static_obstacles = [
            {"id": "movable_1", "type": "box", "x": 1, "y": 2, "z": 3},
            {"id": "falling_1", "type": "box", "x": 4, "y": 5, "z": 6},
        ]
        manager.static_obstacles = list(manager.map_static_obstacles)
        manager.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

        manager._prepare_dynamic_obstacle_selection()

        self.assertEqual(manager.selected_movable_obstacle_ids, [])
        self.assertEqual(manager.selected_falling_object_ids, [])
        self.assertEqual(manager.dynamic_obstacle_metadata, {})
        self.assertEqual(manager.static_obstacles, manager.map_static_obstacles)

    def _selection_for_ratio(self, ratio, ambient_seed):
        random.seed(ambient_seed)
        manager = self.make_manager()
        manager.seed = 23
        manager.current_task_id = "task_0"
        manager.difficulty_config = {
            "movable_obstacle_ratio": ratio,
            "falling_ratio": ratio,
        }
        manager.movable_obstacle_ids = [
            f"movable_{index}" for index in range(10)
        ]
        manager.falling_object_ids = [
            f"falling_{index}" for index in range(10)
        ]
        all_ids = (
            manager.movable_obstacle_ids + manager.falling_object_ids
        )
        manager.map_static_obstacles = [
            {"id": actor_id, "type": "test", "x": 0, "y": 0, "z": 0}
            for actor_id in all_ids
        ]
        manager.static_obstacles = list(manager.map_static_obstacles)
        manager.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

        manager._prepare_dynamic_obstacle_selection()

        return (
            set(manager.selected_movable_obstacle_ids),
            set(manager.selected_falling_object_ids),
        )

    def test_dynamic_selection_is_seed_scoped_and_nested(self):
        easy = self._selection_for_ratio(0.6, ambient_seed=101)
        repeated_easy = self._selection_for_ratio(0.6, ambient_seed=999)
        medium = self._selection_for_ratio(0.8, ambient_seed=202)
        default = self._selection_for_ratio(1.0, ambient_seed=303)

        self.assertEqual(easy, repeated_easy)
        for easy_ids, medium_ids, default_ids in zip(
            easy, medium, default
        ):
            self.assertEqual(len(easy_ids), 6)
            self.assertEqual(len(medium_ids), 8)
            self.assertEqual(len(default_ids), 10)
            self.assertLessEqual(easy_ids, medium_ids)
            self.assertLessEqual(medium_ids, default_ids)
