import inspect
import json
import os
import random
import sys
import tempfile
import unittest
import numpy as np
from unittest.mock import MagicMock, call, patch
from pathlib import Path
from types import SimpleNamespace
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SimWorld"))

from base.rt_agent import (
    BUILDING_COLLISION_REPEAT_LIMIT,
    RTAgent,
    TRAFFIC_POLICY_SAFETY_ASSISTED,
    TRAFFIC_POLICY_VISUAL_ONLY,
)
from base.rt_action_space import MOVE_TO, WAIT, RTActionSpace
from base.rt_communicator import RTCommunicator
from base.rt_evaluator import RTEvaluator
from base.rt_traffic_controller import RTTrafficController
from base.rt_traffic_system import TrafficPhaseTiming
from manager.world_manager import (
    BenchmarkAlignmentError,
    SCRIPTED_PEDESTRIAN_MOTION_POLICY,
    WorldManager,
    align_task_crosswalks_to_rendered_geometry,
    _env_optional_nonnegative_int,
    rendered_native_crosswalk_segments,
    reconstruct_route_points_from_task_edges,
    validate_selected_task_crosswalk_preflight,
)
from tools.capture_triggered_collision_task import _illegal_crossing_lane_offset
from llm.prompt import USER_PROMPT
from utils.annotate_image import annotate_image
from simworld.agent.vehicle import Vehicle, VehicleState
from simworld.agent.pedestrian import PedestrianState
from simworld.config import Config
from simworld.map.map import Map, Node
from simworld.traffic.base.traffic_signal import TrafficSignalState
from simworld.traffic.controller.traffic_controller import TrafficController
from simworld.traffic.manager.pedestrian_manager import PedestrianManager
from simworld.traffic.manager.intersection_manager import IntersectionManager
from simworld.traffic.manager.vehicle_manager import VehicleManager
from simworld.utils.vector import Vector
def test_illegal_crossing_uses_the_rendered_zebra_boundary():
    agent = object.__new__(RTAgent)
    agent.traffic_crosswalks = [
        SimpleNamespace(
            id=19,
            start=Vector(9300.0, -9300.0),
            end=Vector(9300.0, -10700.0),
        )
    ]
    agent.route_crosswalks = []

    assert agent._is_within_marked_crosswalk(Vector(9099.4, -10031.9))
    assert not agent._is_within_marked_crosswalk(Vector(9089.9, -10031.9))




def test_background_density_environment_overrides():
    with patch.dict(
        os.environ,
        {
            "SIMWORLD_BACKGROUND_PEDESTRIANS_PER_EDGE": "6",
            "SIMWORLD_BACKGROUND_IRREGULAR_NPC_MAX": "4",
        },
    ):
        assert _env_optional_nonnegative_int(
            "SIMWORLD_BACKGROUND_PEDESTRIANS_PER_EDGE", 20
        ) == 6
        assert _env_optional_nonnegative_int(
            "SIMWORLD_BACKGROUND_IRREGULAR_NPC_MAX", None
        ) == 4


def test_signalized_preflight_rejects_downgraded_authored_crosswalk():
    scenario = {
        'task_id': 2,
        'render_geometry_calibration': {
            'non_crossing_connectors': [
                {
                    'edge_index': 1,
                    'reason': 'no_matching_native_zebra',
                }
            ],
        },
    }

    with np.testing.assert_raises_regex(
        BenchmarkAlignmentError,
        'task_id=2.*edge_indices=\\[1\\]',
    ):
        validate_selected_task_crosswalk_preflight(
            scenario,
            signal_traffic_enabled=True,
        )


def test_alignment_preflight_error_is_explicitly_nonretryable():
    assert BenchmarkAlignmentError.retryable is False


def test_nonsignalized_preflight_allows_downgraded_connector():
    validate_selected_task_crosswalk_preflight(
        {
            'task_id': 2,
            'render_geometry_calibration': {
                'non_crossing_connectors': [{'edge_index': 1}],
            },
        },
        signal_traffic_enabled=False,
    )


def test_world_manager_passes_strict_api_error_setting_to_llm():
    manager = object.__new__(WorldManager)

    with patch('manager.world_manager.RTLLM') as llm_class:
        manager._create_llm_from_config({
            'model': 'qwen3-vl-8b',
            'provider': 'self-hosted',
            'url': 'http://127.0.0.1:30001/v1',
            'raise_on_api_error': True,
        })

    self_kwargs = llm_class.call_args.kwargs
    assert self_kwargs['raise_on_api_error'] is True


def test_world_manager_passes_recorded_sampling_policy_to_llm():
    manager = object.__new__(WorldManager)

    with patch('manager.world_manager.RTLLM') as llm_class:
        manager._create_llm_from_config({
            'model': 'qwen3-vl-8b',
            'provider': 'self-hosted',
            'temperature': 0.7,
            'top_p': 1.0,
            'seed': 42,
        })

    kwargs = llm_class.call_args.kwargs
    assert kwargs['temperature'] == 0.7
    assert kwargs['top_p'] == 1.0
    assert kwargs['seed'] == 42


def test_background_density_environment_override_rejects_negative_values():
    with patch.dict(
        os.environ,
        {"SIMWORLD_BACKGROUND_PEDESTRIANS_PER_EDGE": "-1"},
    ):
        try:
            _env_optional_nonnegative_int(
                "SIMWORLD_BACKGROUND_PEDESTRIANS_PER_EDGE", 20
            )
        except ValueError as exc:
            assert "non-negative integer" in str(exc)
        else:
            raise AssertionError("negative background density must fail closed")


def test_illegal_crossing_lane_offset_preserves_upstream_launch_runway():
    # A crossing near the segment start used to select the larger upstream
    # side and leave only 60 cm behind the trigger point.  The downstream side
    # keeps a real 100--500 cm launch runway.
    sign, offset = _illegal_crossing_lane_offset(0.2, 1000.0)
    assert sign == 1.0
    assert offset == 500.0
    assert 0.2 * 1000.0 + offset >= 100.0


def test_illegal_crossing_lane_offset_rejects_segment_without_safe_runway():
    assert _illegal_crossing_lane_offset(0.5, 180.0) is None


def test_lane_conflict_skips_crossing_without_requested_launch_runway():
    manager = object.__new__(WorldManager)
    short_vehicle = SimpleNamespace(id=0)
    full_vehicle = SimpleNamespace(id=1)
    manager.red_light_conflict_nominal_speed_cm_s = 450.0
    manager._signal_vehicle_routes = {
        0: {
            'approach_start': Vector(-290.0, 0.0),
            'path_points': [Vector(100.0, 0.0)],
            'incoming_lane': SimpleNamespace(id=0),
            'outgoing_lane': SimpleNamespace(id=1),
        },
        1: {
            'approach_start': Vector(-900.0, 100.0),
            'path_points': [Vector(100.0, 100.0)],
            'incoming_lane': SimpleNamespace(id=2),
            'outgoing_lane': SimpleNamespace(id=3),
        },
    }
    crosswalk = SimpleNamespace(
        start=Vector(0.0, -200.0),
        end=Vector(0.0, 300.0),
        road_id=4,
    )

    selected = manager._lane_aligned_conflict_launch(
        {
            'crosswalk_projection': 0.0,
            'crosswalk_entry_direction': 1,
            'agent_position': {'x': 0.0, 'y': -200.0},
            'agent_speed_cm_s': 200.0,
        },
        crosswalk,
        [short_vehicle, full_vehicle],
        650.0,
    )

    assert selected is not None
    assert selected['vehicle'].id == 1
    assert selected['launch_distance_cm'] == 650.0
    assert selected['launch_position'].distance(Vector(-650.0, 100.0)) < 1e-6


def test_lane_conflict_is_unavailable_when_no_route_has_requested_runway():
    manager = object.__new__(WorldManager)
    vehicle = SimpleNamespace(id=0)
    manager.red_light_conflict_nominal_speed_cm_s = 450.0
    manager._signal_vehicle_routes = {
        0: {
            'approach_start': Vector(-290.0, 0.0),
            'path_points': [Vector(100.0, 0.0)],
            'incoming_lane': SimpleNamespace(id=0),
            'outgoing_lane': SimpleNamespace(id=1),
        },
    }
    manager.traffic_controller = SimpleNamespace(lanes=[])
    crosswalk = SimpleNamespace(
        start=Vector(0.0, -200.0),
        end=Vector(0.0, 300.0),
        road_id=4,
    )

    selected = manager._lane_aligned_conflict_launch(
        {
            'crosswalk_projection': 0.0,
            'crosswalk_entry_direction': 1,
            'agent_position': {'x': 0.0, 'y': -200.0},
            'agent_speed_cm_s': 200.0,
        },
        crosswalk,
        [vehicle],
        650.0,
    )

    assert selected is None


class FakeCommunicator:
    def __init__(self, states):
        self.states = states
        self.legacy_calls = []

    def get_traffic_signal_state(self, signal_id):
        return {
            "signal_id": signal_id,
            "object_name": f"RT_TRAFFIC_SIGNAL_{signal_id}",
            "vehicle_green": False,
            "pedestrian_walk": self.states[signal_id],
            "remaining_time_s": 7.5,
            "source": "ue_blueprint:GetState",
            "raw": {},
        }

    def get_canonical_intersection_state(self, intersection_name):
        return {
            "intersection_id": intersection_name,
            "observed_phase": "PEDESTRIAN_CLEARANCE",
            "phase_remaining_time_s": 3.5,
            "safe": True,
            "violations": [],
            "signal_states": [
                {
                    "name": "RT_TRAFFIC_SIGNAL_7",
                    "role": "vehicle_and_pedestrian",
                    "vehicle_state": "RED",
                    "pedestrian_state": "FLASHING_DONT_WALK",
                    "vehicle_group": 0,
                },
                {
                    "name": "RT_TRAFFIC_SIGNAL_8",
                    "role": "pedestrian",
                    "vehicle_state": "NOT_APPLICABLE",
                    "pedestrian_state": "FLASHING_DONT_WALK",
                },
            ],
            "source": "ue_controller:GetIntersectionState",
            "capabilities": {"pedestrian_clearance_exposed": True},
            "raw": {"active_vehicle_group": "0"},
        }

    def set_traffic_signal_pedestrian_stop(self, signal_id):
        self.legacy_calls.append(("pedestrian_stop", signal_id))

    def set_traffic_signal_pedestrian_walk(self, signal_id):
        self.legacy_calls.append(("pedestrian_walk", signal_id))

    def set_traffic_signal_vehicle_stop(self, signal_id):
        self.legacy_calls.append(("vehicle_stop", signal_id))


def make_crosswalk():
    return SimpleNamespace(
        id=42,
        start=Vector(-9300, -700),
        end=Vector(-9300, 700),
    )


def make_signal(signal_id, x, y, crosswalk_id=None):
    return SimpleNamespace(
        id=signal_id,
        type="pedestrian",
        position=Vector(x, y),
        crosswalk_id=crosswalk_id,
    )


def make_agent(position, destination, states, violation_callback=None):
    crosswalk = make_crosswalk()
    signals = [
        make_signal(7, -9350, -650),
        make_signal(8, -9250, 650),
    ]
    return RTAgent(
        position=position,
        direction=Vector(-1, 0),
        destination=destination,
        shortest_path=[destination],
        communicator=FakeCommunicator(states),
        llm=None,
        route_crosswalks=[crosswalk],
        crosswalk_signal_groups={crosswalk.id: signals},
        task_edges=[{
            "node1": [position.x, position.y],
            "node2": [destination.x, destination.y],
            "type": "sidewalk",
        }],
        red_light_violation_callback=violation_callback,
        traffic_policy=TRAFFIC_POLICY_SAFETY_ASSISTED,
    )


class TrafficLightAlignmentTests(unittest.TestCase):
    def test_task_crosswalk_preserves_packaged_native_zebra_offset(self):
        task_path = REPO_ROOT / "data" / "map1_10roads" / "tasks.json"
        source = json.loads(task_path.read_text(encoding="utf-8"))["tasks"][1]

        aligned = align_task_crosswalks_to_rendered_geometry(source, 700.0)

        crosswalk = aligned["edges"][1]
        self.assertEqual([-9300.0, -700.0], crosswalk["node1"])
        self.assertEqual([-10700.0, -700.0], crosswalk["node2"])
        self.assertEqual([-10700.0, -4688.4384], aligned["start_point"])
        self.assertEqual([-9300.0, -1377.6817], aligned["end_point"])
        self.assertEqual(-700.0, aligned["route_info"]["shortest_path"][1][1])
        self.assertEqual(-700.0, aligned["pedestrian_routes"][0]["route"][0][1])
        self.assertEqual(-700.0, source["edges"][1]["node1"][1])
        calibration = aligned["render_geometry_calibration"]
        self.assertEqual("BP_Road_Small", calibration["asset"])
        self.assertEqual(1, len(calibration["crosswalks"]))
        self.assertAlmostEqual(
            700.0,
            calibration["crosswalks"][0]["rendered_offset_cm"],
        )
        self.assertEqual(
            [0.0, 0.0],
            calibration["crosswalks"][0]["shift_cm"],
        )

    def test_same_side_intersection_connector_is_authored_as_sidewalk(self):
        task_path = REPO_ROOT / "data" / "map1_10roads" / "tasks.json"
        roads_path = task_path.parent / "roads.json"
        source = next(
            task
            for task in json.loads(
                task_path.read_text(encoding="utf-8")
            )["tasks"]
            if task["task_id"] == 2
        )
        roads_payload = json.loads(roads_path.read_text(encoding="utf-8"))
        segments = rendered_native_crosswalk_segments(
            roads_payload,
            700.0,
            700.0,
        )
        centers = sorted({
            (
                float(road[endpoint]["x"]) * 100.0,
                float(road[endpoint]["y"]) * 100.0,
            )
            for road in roads_payload["roads"]
            for endpoint in ("start", "end")
        })

        self.assertEqual(0, source["crosswalk_hops"])
        self.assertEqual("sidewalk", source["edges"][1]["type"])

        aligned = align_task_crosswalks_to_rendered_geometry(
            source,
            700.0,
            intersection_centers=centers,
            rendered_crosswalk_segments=segments,
        )

        self.assertEqual(0, aligned["crosswalk_hops"])
        self.assertEqual("sidewalk", aligned["edges"][1]["type"])
        self.assertEqual(
            [],
            aligned["render_geometry_calibration"][
                "non_crossing_connectors"
            ],
        )
        validate_selected_task_crosswalk_preflight(
            aligned,
            signal_traffic_enabled=True,
        )
        self.assertEqual(
            reconstruct_route_points_from_task_edges(aligned),
            aligned["route_info"]["shortest_path"],
        )

    def test_adjacent_road_corner_connections_are_sidewalks(self):
        graph = Map.__new__(Map)
        graph.nodes = set()
        graph.edges = set()
        graph.adjacency_list = {}
        first = Node(Vector(-700.0, -700.0), Vector(1.0, 0.0), "intersection")
        second = Node(Vector(700.0, -700.0), Vector(1.0, 0.0), "intersection")
        graph.add_node(first)
        graph.add_node(second)

        graph._connect_adjacent_roads(1500.0)

        self.assertEqual(1, len(graph.edges))
        self.assertEqual("sidewalk", next(iter(graph.edges)).type)

    def test_generated_crosswalk_edges_exactly_match_native_zebras(self):
        config = Config(str(REPO_ROOT / "config.yaml"))

        for roads_path in sorted((REPO_ROOT / "data").glob("map*/roads.json")):
            with self.subTest(map_name=roads_path.parent.name):
                roads_payload = json.loads(
                    roads_path.read_text(encoding="utf-8")
                )
                graph = Map(config)
                graph.initialize_map_from_file(
                    str(roads_path),
                    sidewalk_offset=700.0,
                    crosswalk_offset=700.0,
                )

                def edge_key(start, end):
                    return tuple(sorted((
                        (round(float(start[0]), 6), round(float(start[1]), 6)),
                        (round(float(end[0]), 6), round(float(end[1]), 6)),
                    )))

                expected = {
                    edge_key(segment["start"], segment["end"])
                    for segment in rendered_native_crosswalk_segments(
                        roads_payload,
                        700.0,
                        700.0,
                    )
                }
                actual = {
                    edge_key(
                        (edge.node1.position.x, edge.node1.position.y),
                        (edge.node2.position.x, edge.node2.position.y),
                    )
                    for edge in graph.edges
                    if edge.type == "crosswalk"
                }

                self.assertEqual(expected, actual)

    def test_grouped_fixed_walk_uses_one_transition_until_clock_expires(self):
        signals = [SimpleNamespace(id=7), SimpleNamespace(id=8)]
        intersection = SimpleNamespace(
            traffic_lights=[signals[0]],
            pedestrian_lights=[signals[1]],
            vehicle_movement_groups=[[signals[0]]],
            python_pedestrian_phase="walk",
            python_pedestrian_phase_elapsed_s=0.0,
        )
        controller = object.__new__(IntersectionManager)
        controller.intersections = [intersection]
        controller.python_states = {}
        controller.config = {
            "traffic.traffic_signal.pedestrian_green_light_duration": 20.0,
        }
        controller.logger = MagicMock()
        communicator = MagicMock()
        timing = SimpleNamespace(
            pedestrian_walk_s=10.0,
            pedestrian_clearance_s=3.0,
        )

        controller.update_intersections(
            communicator,
            delta_time_s=4.0,
            timing=timing,
        )

        self.assertEqual("walk", intersection.python_pedestrian_phase)
        communicator.traffic_signal_switch_to.assert_not_called()
        self.assertEqual(
            (
                TrafficSignalState.VEHICLE_RED,
                TrafficSignalState.PEDESTRIAN_GREEN,
            ),
            controller.python_states[7],
        )
        self.assertEqual(
            (
                TrafficSignalState.VEHICLE_RED,
                TrafficSignalState.PEDESTRIAN_GREEN,
            ),
            controller.python_states[8],
        )
        communicator.reset_mock()

        controller.update_intersections(
            communicator,
            delta_time_s=6.0,
            timing=timing,
        )

        self.assertEqual("clearance", intersection.python_pedestrian_phase)
        self.assertEqual(2, communicator.traffic_signal_switch_to.call_count)
        self.assertTrue(all(
            call.args[1] == "all red"
            for call in communicator.traffic_signal_switch_to.call_args_list
        ))

    def test_real_vehicle_route_intersection_matches_crosswalk_geometry(self):
        crosswalk = make_crosswalk()
        crossing_route = [
            Vector(-10000, -200),
            Vector(-8600, -200),
        ]
        parallel_route = [
            Vector(-9500, -1200),
            Vector(-9500, 1200),
        ]

        self.assertTrue(
            WorldManager._route_intersects_crosswalk(
                crossing_route,
                crosswalk,
            )
        )
        self.assertFalse(
            WorldManager._route_intersects_crosswalk(
                parallel_route,
                crosswalk,
            )
        )

    def test_route_crosswalk_markings_follow_existing_crosswalk_geometry(self):
        crosswalk = SimpleNamespace(
            id='route-1',
            start=Vector(-10700, -700),
            end=Vector(-9300, -700),
        )

        specs = WorldManager._route_crosswalk_marking_specs(crosswalk)

        self.assertEqual(7, len(specs))
        self.assertTrue(all(spec['location'][1] == -700 for spec in specs))
        self.assertGreater(specs[0]['location'][0], -10700)
        self.assertLess(specs[-1]['location'][0], -9300)
        self.assertTrue(all(spec['orientation'][1] == 0.0 for spec in specs))
        self.assertTrue(all(spec['scale'][0] < spec['scale'][1] for spec in specs))

    def test_route_crosswalk_visuals_default_to_persistent_ue_actors(self):
        crosswalk = SimpleNamespace(
            id=42,
            start=Vector(-9300, -700),
            end=Vector(-9300, 700),
        )
        unrealcv = MagicMock()
        manager = object.__new__(WorldManager)
        manager.communicator = SimpleNamespace(unrealcv=unrealcv)
        manager.logger = MagicMock()

        with patch.dict(os.environ, {'SIMWORLD_RENDER_ROUTE_CROSSWALKS': '1'}):
            manager._spawn_route_crosswalk_markings([crosswalk])

        self.assertEqual(7, len(manager.route_crosswalk_marking_ids))
        self.assertEqual(7, len(manager.route_crosswalk_marking_records))
        self.assertEqual(7, unrealcv.spawn_bp_asset.call_count)
        self.assertTrue(all(
            record['render_backend'] == 'ue_actor_static'
            for record in manager.route_crosswalk_marking_records
        ))

    def test_experimental_ue_crosswalk_actors_remain_non_colliding(self):
        crosswalk = SimpleNamespace(
            id=42,
            start=Vector(-9300, -700),
            end=Vector(-9300, 700),
        )
        unrealcv = MagicMock()
        manager = object.__new__(WorldManager)
        manager.communicator = SimpleNamespace(unrealcv=unrealcv)
        manager.logger = MagicMock()

        with patch.dict(os.environ, {
            'SIMWORLD_RENDER_ROUTE_CROSSWALKS': '1',
            'SIMWORLD_RENDER_ROUTE_CROSSWALK_UE_ACTORS': '1',
        }):
            manager._spawn_route_crosswalk_markings([crosswalk])

        self.assertEqual(7, len(manager.route_crosswalk_marking_ids))
        self.assertEqual(7, unrealcv.spawn_bp_asset.call_count)
        self.assertEqual(7, unrealcv.set_collision.call_count)
        self.assertTrue(
            all(call.args[1] is False for call in unrealcv.set_collision.call_args_list)
        )
        self.assertEqual(7, unrealcv.set_movable.call_count)
        self.assertTrue(
            all(call.args[1] is False for call in unrealcv.set_movable.call_args_list)
        )

    def test_rgb_crosswalk_overlay_is_disabled_for_persistent_ue_actors(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        crosswalk = SimpleNamespace(
            id=42,
            start=Vector(300, 0),
            end=Vector(900, 0),
        )

        with patch.dict(os.environ, {
            'SIMWORLD_RENDER_ROUTE_CROSSWALKS': '1',
            'SIMWORLD_RENDER_ROUTE_CROSSWALK_UE_ACTORS': '1',
        }):
            annotated = annotate_image(
                image,
                [],
                camera_location=(0, 0, 100),
                camera_rotation=(0, 0, 0),
                camera_fov=90,
                route_crosswalks=[crosswalk],
            )

        self.assertEqual(0, int(np.asarray(annotated).sum()))

    def test_rgb_crosswalk_overlay_uses_same_logical_route_geometry(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        crosswalk = SimpleNamespace(
            id=42,
            start=Vector(300, 0),
            end=Vector(900, 0),
        )

        with patch.dict(os.environ, {
            'SIMWORLD_RENDER_ROUTE_CROSSWALKS': '1',
            'SIMWORLD_RENDER_ROUTE_CROSSWALK_UE_ACTORS': '0',
        }):
            annotated = annotate_image(
                image,
                [],
                camera_location=(0, 0, 100),
                camera_rotation=(0, 0, 0),
                camera_fov=90,
                route_crosswalks=[crosswalk],
            )

        rgb = np.asarray(annotated)
        painted = np.all(rgb == np.array([238, 238, 224]), axis=2)
        self.assertGreater(int(painted.sum()), 0)

    def test_rollout_transport_error_is_an_explicit_failed_terminal_result(self):
        manager = object.__new__(WorldManager)
        manager.max_steps = 5
        manager.control_mode = 'llm'
        manager.signal_traffic_drain_sim_time_s = 0.0
        manager.rollout_error = None
        manager.logger = MagicMock()
        manager.save_evaluation_data = MagicMock()
        manager.agent = SimpleNamespace(
            step_num=0,
            success=False,
            failed=False,
            failure_reason=None,
            step=MagicMock(side_effect=ConnectionError('socket is closed')),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            'runtime/UE communication error',
        ):
            manager.run()

        self.assertTrue(manager.agent.failed)
        self.assertFalse(manager.agent.success)
        self.assertEqual('runtime_error', manager.agent.failure_reason)
        self.assertEqual(
            {'type': 'ConnectionError', 'message': 'socket is closed'},
            manager.rollout_error,
        )
        manager.save_evaluation_data.assert_called_once_with()

    def test_vehicle_collision_counter_uses_ue_collision_api(self):
        communicator = object.__new__(RTCommunicator)
        communicator.unrealcv = SimpleNamespace(
            get_collision_num=lambda name: json.dumps(
                {
                    "HumanCollision": 0,
                    "ObjectCollision": 0,
                    "BuildingCollision": 0,
                    "VehicleCollision": 2,
                }
            )
        )

        self.assertEqual(
            2,
            communicator.get_vehicle_collision_number("RT_AGENT"),
        )

    def test_non_vehicle_collision_counters_use_ue_states_api(self):
        communicator = object.__new__(RTCommunicator)
        communicator.unrealcv = SimpleNamespace(
            rt_get_states=lambda name: json.dumps({
                "HumanCollision": 1,
                "ObjectCollision": 2,
                "BuildingCollision": 3,
                "Intensity": 4.5,
                "OverlapType": 0,
                "TouchedRoad": 1,
            })
        )

        self.assertEqual(
            (1, 2, 3, 4.5, 0, 1),
            communicator.get_states("RT_AGENT"),
        )

    def test_new_vehicle_collision_marks_agent_failed_once(self):
        counts = iter([0, 1, 1])
        agent = object.__new__(RTAgent)
        agent.name = "RT_AGENT"
        agent.communicator = SimpleNamespace(
            get_vehicle_collision_number=lambda _: next(counts)
        )
        agent.last_ue_collision_count = {"vehicle": 0}
        agent.vehicle_collision_count = 0
        agent.collision_count = 0
        agent.success = True
        agent.failed = False
        agent.failure_reason = None
        agent.logger = SimpleNamespace(error=lambda *args: None)
        agent.red_light_conflict_active = True

        self.assertEqual(0, agent._check_vehicle_collision_failure())
        self.assertEqual(1, agent._check_vehicle_collision_failure())
        self.assertTrue(agent.failed)
        self.assertFalse(agent.success)
        self.assertEqual("vehicle_collision", agent.failure_reason)
        self.assertEqual(1, agent.vehicle_collision_count)
        self.assertEqual(1, agent.collision_count)
        self.assertEqual(0, agent._check_vehicle_collision_failure())
        self.assertEqual(1, agent.vehicle_collision_count)

    def test_vehicle_collision_is_ignored_without_launched_vehicle(self):
        counts = iter([0, 1])
        agent = object.__new__(RTAgent)
        agent.name = "RT_AGENT"
        agent.communicator = SimpleNamespace(
            get_vehicle_collision_number=lambda _: next(counts)
        )
        agent.last_ue_collision_count = {"vehicle": 0}
        agent.vehicle_collision_count = 0
        agent.collision_count = 0
        agent.success = True
        agent.failed = False
        agent.failure_reason = None
        agent.red_light_conflict_active = False
        agent.logger = SimpleNamespace(
            error=lambda *args: None,
            warning=lambda *args: None,
        )

        self.assertEqual(0, agent._check_vehicle_collision_failure())
        self.assertEqual(0, agent._check_vehicle_collision_failure())
        self.assertFalse(agent.failed)
        self.assertTrue(agent.success)
        self.assertEqual(0, agent.vehicle_collision_count)
        self.assertEqual(0, agent.collision_count)
        self.assertEqual(1, agent.last_ue_collision_count['vehicle'])

    def test_swept_launched_vehicle_collision_survives_vehicle_retirement(self):
        agent = object.__new__(RTAgent)
        agent.name = "RT_AGENT"
        agent.communicator = SimpleNamespace(
            get_vehicle_collision_number=lambda _: 0
        )
        agent.last_ue_collision_count = {"vehicle": 0}
        agent.vehicle_collision_count = 0
        agent.collision_count = 0
        agent.success = True
        agent.failed = False
        agent.failure_reason = None
        agent.red_light_conflict_active = False
        agent.logger = MagicMock()

        agent.queue_swept_vehicle_collision()

        self.assertEqual(1, agent._check_vehicle_collision_failure())
        self.assertTrue(agent.failed)
        self.assertFalse(agent.success)
        self.assertEqual("vehicle_collision", agent.failure_reason)
        self.assertEqual(1, agent.vehicle_collision_count)
        self.assertEqual(1, agent.collision_count)
        self.assertEqual(0, agent._check_vehicle_collision_failure())

    def test_post_action_recovery_vehicle_impact_stays_on_current_step(self):
        agent = object.__new__(RTAgent)
        collision_checks = iter((0, 1))
        agent._check_vehicle_collision_failure = lambda: next(collision_checks)
        agent._handle_building_collision_recovery = MagicMock(
            return_value='building recovered'
        )
        agent.success = True
        agent.failed = False
        agent.failure_reason = None
        details = {'building': 1, 'vehicle': 0}

        feedback, recovered = agent._finalize_post_action_collisions(
            SimpleNamespace(action_type='move_to', action_param='5'),
            'Move to target',
            'illegal crossing',
            details,
        )

        self.assertTrue(recovered)
        self.assertEqual(1, details['vehicle'])
        self.assertTrue(agent.failed)
        self.assertFalse(agent.success)
        self.assertEqual('vehicle_collision', agent.failure_reason)
        self.assertIn('struck by a vehicle', feedback)
        agent._handle_building_collision_recovery.assert_called_once()

    def test_vehicle_impact_precedes_building_recovery(self):
        agent = object.__new__(RTAgent)
        agent._check_vehicle_collision_failure = lambda: 1
        agent._handle_building_collision_recovery = MagicMock()
        agent.success = True
        agent.failed = False
        agent.failure_reason = None
        details = {'building': 1, 'vehicle': 0}

        feedback, recovered = agent._finalize_post_action_collisions(
            SimpleNamespace(action_type='move_to', action_param='5'),
            'Move to target',
            'collision',
            details,
        )

        self.assertFalse(recovered)
        self.assertEqual(1, details['vehicle'])
        self.assertEqual('vehicle_collision', agent.failure_reason)
        self.assertIn('struck by a vehicle', feedback)
        agent._handle_building_collision_recovery.assert_not_called()

    def test_existing_vehicle_impact_also_precedes_building_recovery(self):
        agent = object.__new__(RTAgent)
        agent._check_vehicle_collision_failure = lambda: 0
        agent._handle_building_collision_recovery = MagicMock()
        agent.success = False
        agent.failed = True
        agent.failure_reason = 'vehicle_collision'
        details = {'building': 1, 'vehicle': 1}

        feedback, recovered = agent._finalize_post_action_collisions(
            SimpleNamespace(action_type='move_to', action_param='5'),
            'Move to target',
            'vehicle collision',
            details,
        )

        self.assertFalse(recovered)
        self.assertEqual(1, details['vehicle'])
        self.assertEqual('vehicle collision', feedback)
        agent._handle_building_collision_recovery.assert_not_called()

    def test_swept_vehicle_distance_detects_between_sample_impact(self):
        distance = WorldManager._swept_minimum_separation_cm(
            Vector(-500, 0),
            Vector(500, 0),
            Vector(0, 0),
            Vector(0, 0),
        )

        self.assertAlmostEqual(0.0, distance)

    def test_legacy_conflict_vehicle_swept_impact_is_terminal(self):
        queued = []
        updates = []
        vehicle = SimpleNamespace(
            id=4,
            position=Vector(500, 0),
            state=VehicleState.MOVING,
            set_attributes=lambda *values: updates.append(values),
        )
        record = {
            "minimum_agent_distance_cm": 500.0,
            "impact_zone_reached": False,
        }
        manager = object.__new__(WorldManager)
        manager.agent = SimpleNamespace(
            position=Vector(0, 0),
            sim_time_elapsed=5.0,
            queue_swept_vehicle_collision=lambda: queued.append(True),
        )
        manager.signal_traffic_communicator = SimpleNamespace(
            update_vehicle=lambda *args: updates.append(args)
        )
        manager.conflict_vehicle_impact_radius_cm = 150.0
        manager.vehicle_impact_zone_count = 0
        active = {
            "vehicle": vehicle,
            "record": record,
            "launch_position": Vector(-500, 0),
            "direction": Vector(1, 0),
            "launch_sim_time_s": 0.0,
            "previous_vehicle_position": Vector(-500, 0),
            "previous_agent_position": Vector(0, 0),
            "target_position": Vector(1000, 0),
        }
        manager._active_red_light_conflict = active
        manager._retire_conflict_vehicle = MagicMock()

        manager._update_red_light_conflict_vehicle()

        self.assertEqual([True], queued)
        self.assertEqual("collision", record["status"])
        self.assertEqual(
            "swept_actor_trajectory",
            record["impact_zone_detection_source"],
        )
        self.assertEqual(
            "launched_vehicle_swept_radius",
            record["collision_authority"],
        )
        self.assertEqual(0.0, record["minimum_agent_distance_cm"])
        manager._retire_conflict_vehicle.assert_called_once_with(
            active,
            "collision",
        )

    def test_conflict_vehicle_despawns_after_passing_launch_time_agent_pose(self):
        destroyed = []
        lane = SimpleNamespace(vehicles=[])
        lane.remove_vehicle = lambda vehicle: lane.vehicles.remove(vehicle)
        vehicle = SimpleNamespace(
            id=6,
            position=Vector(100, 0),
            state=VehicleState.MOVING,
            current_lane=lane,
            set_attributes=lambda *values: None,
        )
        lane.vehicles.append(vehicle)
        vehicle_manager = SimpleNamespace(vehicles=[vehicle])
        record = {
            "vehicle_name": "RT_SIGNAL_VEHICLE_6",
            "minimum_agent_distance_cm": 500.0,
            "impact_zone_reached": False,
        }
        active = {
            "vehicle": vehicle,
            "record": record,
            "launch_position": Vector(-500, 0),
            "target_position": Vector(0, 0),
            "direction": Vector(1, 0),
            "launch_sim_time_s": 0.0,
            "previous_vehicle_position": Vector(-100, 0),
            # The agent has moved away; completion must still use its pose at
            # violation time rather than chasing its current position forever.
            "previous_agent_position": Vector(0, 1000),
        }
        manager = object.__new__(WorldManager)
        manager.agent = SimpleNamespace(
            position=Vector(0, 1000),
            sim_time_elapsed=2.0,
        )
        manager.traffic_controller = SimpleNamespace(
            vehicle_manager=vehicle_manager,
        )
        manager.signal_traffic_communicator = SimpleNamespace(
            unrealcv=SimpleNamespace(
                destroy=lambda name: destroyed.append(name)
            ),
            get_vehicle_name=lambda vehicle_id: f"RT_SIGNAL_VEHICLE_{vehicle_id}",
            update_vehicle=lambda *args: None,
        )
        manager.conflict_vehicle_impact_radius_cm = 50.0
        manager.vehicle_impact_zone_count = 0
        manager._retired_signal_vehicle_ids = set()
        manager._active_vehicle_conflicts = [active]
        manager._active_red_light_conflict = active

        manager._update_red_light_conflict_vehicle()

        self.assertEqual(["RT_SIGNAL_VEHICLE_6"], destroyed)
        self.assertEqual([], lane.vehicles)
        self.assertEqual([], vehicle_manager.vehicles)
        self.assertEqual({6: vehicle}, manager._retired_conflict_vehicles)
        self.assertEqual({6}, manager._retired_signal_vehicle_ids)
        self.assertEqual("completed", record["status"])
        self.assertEqual("passed_launch_target", record["completion_reason"])
        self.assertTrue(record["passed_target_position"])
        self.assertTrue(record["despawned"])
        self.assertIsNone(manager._active_red_light_conflict)

    def test_signal_spawn_uses_configured_scale_by_asset_type(self):
        scales = []
        communicator = object.__new__(RTCommunicator)
        communicator.unrealcv = SimpleNamespace(
            spawn_bp_asset=lambda *args: None,
            set_location=lambda *args: None,
            set_orientation=lambda *args: None,
            set_scale=lambda scale, name: scales.append((name, scale)),
            set_collision=lambda *args: None,
            set_movable=lambda *args: None,
        )
        signals = [
            SimpleNamespace(
                id=1,
                type="both",
                position=Vector(0, 0),
                direction=Vector(1, 0),
            ),
            SimpleNamespace(
                id=2,
                type="pedestrian",
                position=Vector(0, 0),
                direction=Vector(1, 0),
            ),
        ]

        communicator.spawn_traffic_signals(
            signals,
            traffic_light_scale=1.2,
            pedestrian_light_scale=1.3,
        )

        self.assertEqual(
            [
                ("RT_TRAFFIC_SIGNAL_1", (1.2, 1.2, 1.2)),
                ("RT_TRAFFIC_SIGNAL_2", (1.3, 1.3, 1.3)),
            ],
            scales,
        )

    def test_same_head_green_walk_conflict_is_repaired_before_snapshot(self):
        state = {"pedestrian_walk": True}
        communicator = SimpleNamespace(
            get_traffic_signal_state=lambda signal_id: {
                "signal_id": signal_id,
                "vehicle_green": True,
                "pedestrian_walk": state["pedestrian_walk"],
            },
            set_traffic_signal_pedestrian_stop=lambda signal_id: state.update(
                pedestrian_walk=False
            ),
        )
        agent = object.__new__(RTAgent)
        agent.communicator = communicator
        agent.logger = SimpleNamespace(warning=lambda *args: None)

        aligned = agent._read_aligned_traffic_signal_state(9)

        self.assertFalse(aligned["pedestrian_walk"])
        self.assertEqual("repaired", aligned["state_repair"]["status"])

    def test_vehicle_rechecks_signal_after_red_stop_even_outside_end_threshold(self):
        vehicle = MagicMock()
        vehicle.id = 2
        vehicle.state = VehicleState.STOPPED
        vehicle.stop_reason = 'red_signal'
        vehicle.current_lane = SimpleNamespace(id=9)
        vehicle.waypoints = []
        vehicle.advance_route_waypoints.return_value = None
        vehicle.get_attributes.return_value = (1.0, 0.0, 0.0)
        vehicle.is_close_to_object.return_value = False
        vehicle.is_close_to_end.return_value = False
        vehicle.nearest_static_obstacle_ahead.return_value = None
        next_lane = SimpleNamespace(id=6)
        intersection = MagicMock()
        intersection.get_traffic_light_state.return_value = (
            TrafficSignalState.VEHICLE_GREEN,
            20.0,
        )
        intersection_controller = MagicMock()
        intersection_controller.get_waypoints_for_vehicle.return_value = (
            next_lane,
            [Vector(100, 0)],
            intersection,
            False,
        )
        communicator = MagicMock()
        manager = object.__new__(VehicleManager)
        manager.vehicles = [vehicle]
        manager.last_states = {}
        manager.logger = MagicMock()

        VehicleManager.update_vehicles(
            manager,
            communicator,
            intersection_controller,
            [],
            static_obstacles=[],
        )

        vehicle.add_waypoint.assert_called_once()
        vehicle.change_to_next_lane.assert_called_once_with(next_lane)
        self.assertIsNone(vehicle.stop_reason)

    def test_world_controlled_conflict_vehicle_skips_normal_avoidance(self):
        vehicle = MagicMock()
        vehicle.id = 2
        vehicle.get_attributes.return_value = (1.0, 0.0, 0.0)
        communicator = MagicMock()
        manager = object.__new__(VehicleManager)
        manager.vehicles = [vehicle]
        manager.last_states = {}
        manager.logger = MagicMock()

        VehicleManager.update_vehicles(
            manager,
            communicator,
            MagicMock(),
            [SimpleNamespace(id='benchmark-agent')],
            static_obstacles=[{'id': 'road-block'}],
            externally_controlled_vehicle_ids={2},
        )

        vehicle.advance_route_waypoints.assert_not_called()
        vehicle.is_close_to_object.assert_not_called()
        vehicle.nearest_static_obstacle_ahead.assert_not_called()
        vehicle.set_attributes.assert_not_called()
        communicator.update_vehicles.assert_called_once_with(
            {2: (1.0, 0.0, 0.0)}
        )

    def test_generated_tree_blocking_live_vehicle_route_is_removed(self):
        tree = {
            'id': 'GEN_RT_BP_Tree6_61',
            'type': 'BP_Tree6_C',
            'x': -13101.8,
            'y': -935.4,
        }
        vehicle = MagicMock()
        vehicle.id = 2
        vehicle.state = VehicleState.MOVING
        vehicle.stop_reason = None
        vehicle.current_lane = SimpleNamespace(id=6)
        vehicle.waypoints = [Vector(-13100, -400)]
        vehicle.advance_route_waypoints.return_value = None
        vehicle.is_close_to_object.return_value = False
        vehicle.is_close_to_end.return_value = False
        vehicle.nearest_static_obstacle_ahead.return_value = tree
        vehicle.get_attributes.return_value = (0.1, 0.0, 0.85)
        communicator = MagicMock()
        manager = object.__new__(VehicleManager)
        manager.vehicles = [vehicle]
        manager.last_states = {}
        manager.logger = MagicMock()
        manager.removed_generated_route_furniture = []
        obstacles = [tree]

        VehicleManager.update_vehicles(
            manager,
            communicator,
            MagicMock(),
            [],
            static_obstacles=obstacles,
        )

        communicator.unrealcv.destroy.assert_called_once_with(tree['id'])
        communicator.unrealcv.clean_garbage.assert_called_once()
        self.assertEqual([], obstacles)
        self.assertEqual(1, len(manager.removed_generated_route_furniture))
        vehicle.set_attributes.assert_not_called()

    def test_real_road_debris_is_not_removed_as_map_furniture(self):
        self.assertFalse(
            VehicleManager._is_generated_route_furniture(
                {
                    'id': 'GEN_RT_RT_RoadBlock_3',
                    'type': 'BP_RoadBlock_C',
                }
            )
        )

    def test_prompt_marks_waypoints_whose_path_hits_static_geometry(self):
        agent = make_agent(
            position=Vector(0, 0),
            destination=Vector(1000, 0),
            states={7: False, 8: False},
        )
        agent.static_obstacles = [
            {'id': 'TREE_TEST', 'type': 'tree', 'x': 200, 'y': 0}
        ]

        context = agent._format_static_obstacle_context(
            [Vector(200, 0), Vector(0, 200)]
        )

        self.assertIn('Waypoint 1: BLOCKED', context)
        self.assertIn('TREE_TEST', context)
        self.assertNotIn('Waypoint 2: BLOCKED', context)

    def test_vehicle_dynamic_avoidance_follows_planned_route_corridor(self):
        vehicle = object.__new__(Vehicle)
        vehicle.id = 0
        vehicle.position = Vector(0, 0)
        vehicle.direction = 0.0
        vehicle.length = 400.0
        vehicle.width = 200.0
        vehicle.config = {
            'traffic.distance_between_objects': 400.0,
            'traffic.detection_angle': 50.0,
        }
        adjacent_vehicle = SimpleNamespace(
            id=1,
            position=Vector(700, 500),
            direction=Vector(1, 0),
            length=400.0,
            width=200.0,
        )
        route_vehicle = SimpleNamespace(
            id=2,
            position=Vector(700, 0),
            direction=Vector(1, 0),
            length=400.0,
            width=200.0,
        )
        route = [Vector(500, 0), Vector(1000, 0)]

        self.assertFalse(
            vehicle.is_close_to_object(
                [vehicle, adjacent_vehicle],
                [],
                planned_waypoints=route,
            )
        )
        self.assertTrue(
            vehicle.is_close_to_object(
                [vehicle, route_vehicle],
                [],
                planned_waypoints=route,
            )
        )

    def test_traffic_controller_updates_sparse_actor_ids_by_identity(self):
        vehicle_zero = SimpleNamespace(id=0, position=None, direction=None)
        vehicle_two = SimpleNamespace(id=2, position=None, direction=None)
        pedestrian_three = SimpleNamespace(id=3, position=None, direction=None)
        controller = SimpleNamespace(
            vehicles=[vehicle_zero, vehicle_two],
            pedestrians=[pedestrian_three],
            traffic_signals=[],
            communicator=SimpleNamespace(
                get_position_and_direction=lambda *_: {
                    ('vehicle', 2): (Vector(12, 34), 56.0),
                    ('pedestrian', 3): (Vector(78, 90), 12.0),
                }
            ),
        )

        TrafficController.update_states(controller)

        self.assertEqual(Vector(12, 34), vehicle_two.position)
        self.assertEqual(56.0, vehicle_two.direction)
        self.assertEqual(Vector(78, 90), pedestrian_three.position)
        self.assertEqual(12.0, pedestrian_three.direction)
        self.assertIsNone(vehicle_zero.position)

    def test_vehicle_detects_only_static_obstacles_in_forward_swept_path(self):
        lane = SimpleNamespace(
            id=1,
            start=Vector(0, 0),
            end=Vector(2000, 0),
            direction=Vector(1, 0),
        )
        config = {
            'traffic.vehicle.steering_pid.kp': 0.15,
            'traffic.vehicle.steering_pid.ki': 0.005,
            'traffic.vehicle.steering_pid.kd': 0.12,
            'traffic.vehicle.max_steering': 0.5,
            'traffic.vehicle.static_obstacle_detection_distance': 800,
            'traffic.vehicle.static_obstacle_lateral_margin': 120,
        }
        vehicle = Vehicle(
            Vector(0, 0),
            Vector(1, 0),
            lane,
            '/Game/TestVehicle',
            config,
            length=400,
            width=200,
        )
        ahead = {'id': 'ahead', 'x': 600, 'y': 50}
        self.assertEqual(
            ahead,
            vehicle.nearest_static_obstacle_ahead(
                [
                    {'id': 'behind', 'x': -400, 'y': 0},
                    {'id': 'sidewalk', 'x': 500, 'y': 500},
                    ahead,
                ]
            ),
        )
        self.assertIsNone(
            vehicle.nearest_static_obstacle_ahead(
                [
                    {'id': 'behind', 'x': -400, 'y': 0},
                    {'id': 'sidewalk', 'x': 500, 'y': 500},
                ]
            )
        )

    def test_lane_furniture_distance_uses_finite_route_segment(self):
        self.assertAlmostEqual(
            25.0,
            WorldManager._point_segment_distance(
                Vector(50, 25),
                Vector(0, 0),
                Vector(100, 0),
            ),
        )
        self.assertAlmostEqual(
            50.0,
            WorldManager._point_segment_distance(
                Vector(150, 0),
                Vector(0, 0),
                Vector(100, 0),
            ),
        )

    def test_vehicle_static_obstacle_detection_follows_turning_route(self):
        lane = SimpleNamespace(
            id=1,
            start=Vector(0, 0),
            end=Vector(300, 0),
            direction=Vector(1, 0),
        )
        config = {
            'traffic.vehicle.steering_pid.kp': 0.15,
            'traffic.vehicle.steering_pid.ki': 0.005,
            'traffic.vehicle.steering_pid.kd': 0.12,
            'traffic.vehicle.max_steering': 0.5,
            'traffic.vehicle.static_obstacle_detection_distance': 800,
            'traffic.vehicle.static_obstacle_lateral_margin': 120,
        }
        vehicle = Vehicle(
            Vector(0, 0),
            Vector(1, 0),
            lane,
            '/Game/TestVehicle',
            config,
            length=400,
            width=200,
        )
        turning_path = [Vector(300, 0), Vector(300, 500)]
        on_route = {'id': 'on-turn', 'x': 300, 'y': 400}

        self.assertEqual(
            on_route,
            vehicle.nearest_static_obstacle_ahead(
                [
                    # Visually straight ahead, but the vehicle turns before it.
                    {'id': 'past-curb', 'x': 650, 'y': 0},
                    on_route,
                ],
                planned_waypoints=turning_path,
            ),
        )
        self.assertIsNone(
            vehicle.nearest_static_obstacle_ahead(
                [{'id': 'past-curb', 'x': 650, 'y': 0}],
                planned_waypoints=turning_path,
            )
        )

    def test_vehicle_route_progress_keeps_curve_recovery_target(self):
        lane = SimpleNamespace(
            id=1,
            start=Vector(0, 0),
            end=Vector(2000, 0),
            direction=Vector(1, 0),
        )
        config = {
            'traffic.vehicle.steering_pid.kp': 0.15,
            'traffic.vehicle.steering_pid.ki': 0.005,
            'traffic.vehicle.steering_pid.kd': 0.12,
            'traffic.vehicle.max_steering': 0.5,
        }
        vehicle = Vehicle(
            Vector(500, 700),
            Vector(1, 0),
            lane,
            '/Game/TestVehicle',
            config,
            length=400,
            width=200,
        )
        curve = [
            Vector(0, 0),
            Vector(250, 250),
            Vector(500, 500),
            Vector(750, 750),
            Vector(1000, 1000),
            Vector(3000, 0),
        ]
        vehicle.waypoints = list(curve)

        target = vehicle.advance_route_waypoints()

        # A short look-ahead sample on the curve is retained for recovery.  The
        # old behind-dot loop deleted every curve sample and targeted (3000, 0).
        self.assertEqual(Vector(750, 750), target)
        self.assertEqual(curve[2:], vehicle.waypoints)
        remaining = list(vehicle.waypoints)
        self.assertEqual(target, vehicle.advance_route_waypoints())
        self.assertEqual(remaining, vehicle.waypoints)

    def test_vehicle_route_progress_drops_reached_points(self):
        lane = SimpleNamespace(
            id=1,
            start=Vector(0, 0),
            end=Vector(2000, 0),
            direction=Vector(1, 0),
        )
        config = {
            'traffic.vehicle.steering_pid.kp': 0.15,
            'traffic.vehicle.steering_pid.ki': 0.005,
            'traffic.vehicle.steering_pid.kd': 0.12,
            'traffic.vehicle.max_steering': 0.5,
        }
        vehicle = Vehicle(
            Vector(510, 5),
            Vector(1, 0),
            lane,
            '/Game/TestVehicle',
            config,
            length=400,
            width=200,
        )
        vehicle.waypoints = [
            Vector(0, 0),
            Vector(500, 0),
            Vector(700, 0),
            Vector(900, 0),
        ]

        target = vehicle.advance_route_waypoints()

        self.assertEqual(Vector(900, 0), target)
        self.assertEqual([Vector(700, 0), Vector(900, 0)], vehicle.waypoints)

    def test_vehicle_slows_until_it_recovers_outgoing_lane_center(self):
        lane = SimpleNamespace(
            id=3,
            start=Vector(-8300, 2000),
            end=Vector(-8300, -4000),
            direction=Vector(0, -1),
        )
        config = {
            'traffic.vehicle.steering_pid.kp': 0.15,
            'traffic.vehicle.steering_pid.ki': 0.005,
            'traffic.vehicle.steering_pid.kd': 0.12,
            'traffic.vehicle.max_steering': 0.5,
            'traffic.vehicle.lane_deviation': 70,
        }
        vehicle = Vehicle(
            Vector(-8060, 1700),
            Vector(0, -1),
            lane,
            '/Game/TestVehicle',
            config,
            length=400,
            width=200,
        )

        throttle, brake, steering, changed = vehicle.compute_control(
            Vector(-8300, 400),
            0.1,
        )

        self.assertTrue(changed)
        self.assertEqual(0.10, throttle)
        self.assertEqual(0, brake)
        self.assertAlmostEqual(0.85, abs(float(steering)))

    def test_completed_agent_drains_signal_traffic_until_all_cars_clear(self):
        manager = WorldManager.__new__(WorldManager)
        manager.signal_traffic_enabled = True
        manager.static_signal_vehicles = True
        manager._active_vehicle_conflicts = [
            {'vehicle': SimpleNamespace(id=0)},
        ]
        manager.signal_vehicle_crossing_status = {
            0: {
                'entered_intersection': True,
                'cleared_intersection': False,
            },
            1: {
                'entered_intersection': True,
                'cleared_intersection': True,
            },
        }
        manager.signal_traffic_drain_sim_time_s = 0.0
        manager.signal_traffic_drain_completed = False
        manager.logger = MagicMock()
        advances = []

        def advance(seconds):
            advances.append(seconds)
            if len(advances) == 2:
                manager.signal_vehicle_crossing_status[0][
                    'cleared_intersection'
                ] = True

        manager.agent = SimpleNamespace(
            _advance_simulation_time=advance,
        )

        with patch.dict(
            os.environ,
            {'SIMWORLD_SIGNAL_TRAFFIC_DRAIN_SECONDS': '5'},
        ):
            manager._drain_signal_traffic_after_agent_completion()

        self.assertEqual([0.5, 0.5], advances)
        self.assertEqual(1.0, manager.signal_traffic_drain_sim_time_s)
        self.assertTrue(manager.signal_traffic_drain_completed)

    def test_completed_agent_does_not_drain_never_admitted_staged_cars(self):
        manager = WorldManager.__new__(WorldManager)
        manager.signal_traffic_enabled = True
        manager.signal_vehicle_crossing_status = {
            0: {
                'entered_intersection': False,
                'cleared_intersection': False,
            },
            1: {
                'entered_intersection': True,
                'cleared_intersection': True,
            },
        }
        manager.signal_traffic_drain_sim_time_s = 0.0
        manager.signal_traffic_drain_completed = False
        manager.logger = MagicMock()
        manager.agent = SimpleNamespace(
            _advance_simulation_time=MagicMock(),
        )

        manager._drain_signal_traffic_after_agent_completion()

        manager.agent._advance_simulation_time.assert_not_called()
        self.assertEqual(0.0, manager.signal_traffic_drain_sim_time_s)
        self.assertTrue(manager.signal_traffic_drain_completed)

    def test_completed_agent_does_not_drain_static_background_car(self):
        manager = WorldManager.__new__(WorldManager)
        manager.signal_traffic_enabled = True
        manager.static_signal_vehicles = True
        manager._active_vehicle_conflicts = []
        manager.signal_vehicle_crossing_status = {
            0: {
                'entered_intersection': True,
                'cleared_intersection': False,
            },
        }
        manager.signal_traffic_drain_sim_time_s = 0.0
        manager.signal_traffic_drain_completed = False
        manager.logger = MagicMock()
        manager.agent = SimpleNamespace(
            _advance_simulation_time=MagicMock(),
        )

        manager._drain_signal_traffic_after_agent_completion()

        manager.agent._advance_simulation_time.assert_not_called()
        self.assertEqual(0.0, manager.signal_traffic_drain_sim_time_s)
        self.assertTrue(manager.signal_traffic_drain_completed)

    def test_red_light_violation_without_real_lane_records_unavailable(self):
        class FakeVehicle:
            def __init__(self):
                self.id = 3
                self.position = Vector(0, 0)
                self._direction = Vector(1, 0)
                self.state = None
                self.attributes = None

            @property
            def direction(self):
                return self._direction

            @direction.setter
            def direction(self, yaw):
                import math
                self._direction = Vector(
                    math.cos(math.radians(yaw)),
                    math.sin(math.radians(yaw)),
                )

            def set_attributes(self, throttle, brake, steering):
                self.attributes = (throttle, brake, steering)

        calls = []
        unrealcv = SimpleNamespace(
            set_location=lambda *args: calls.append(("location", args)),
            set_orientation=lambda *args: calls.append(("orientation", args)),
            set_collision=lambda *args: calls.append(("collision", args)),
            set_movable=lambda *args: calls.append(("movable", args)),
        )
        vehicle = FakeVehicle()
        manager = object.__new__(WorldManager)
        manager.traffic_controller = SimpleNamespace(vehicles=[vehicle])
        manager.signal_traffic_communicator = SimpleNamespace(
            unrealcv=unrealcv,
            get_vehicle_name=lambda vehicle_id: f"RT_SIGNAL_VEHICLE_{vehicle_id}",
            update_vehicle=lambda *args: calls.append(("state", args)),
        )
        manager.red_light_conflict_vehicle_events = []
        manager._active_red_light_conflict = None
        manager.red_light_conflict_launch_distance_min_cm = 450.0
        manager.red_light_conflict_launch_distance_max_cm = 450.0
        manager.red_light_conflict_collision_radius_cm = 250.0
        manager.red_light_conflict_vehicle_probability = 1.0
        manager.red_light_conflict_nominal_speed_cm_s = 450.0
        manager._traffic_consequence_rng = random.Random(1)
        manager.logger = SimpleNamespace(
            error=lambda *args: None,
            warning=lambda *args: None,
        )

        manager._handle_red_light_violation(
            {
                "event_id": "red-light-0001",
                "crosswalk_id": 42,
                "sim_time_s": 12.5,
                "pedestrian_state": "DON'T WALK",
                "crosswalk_projection": 0.5,
                "agent_position": {"x": -9300, "y": 0},
            },
            make_crosswalk(),
        )

        event = manager.red_light_conflict_vehicle_events[0]
        self.assertEqual("unavailable", event["status"])
        self.assertEqual("unavailable", event["disposition"])
        self.assertEqual(
            "no_lane_aligned_conflict_route",
            event["unavailable_reason"],
        )
        self.assertIsNone(vehicle.attributes)
        self.assertFalse(calls)
        self.assertIsNone(manager._active_red_light_conflict)

    def test_endpoint_crosswalk_uses_same_road_lane_extension(self):
        vehicle = SimpleNamespace(id=7)
        incoming_lane = SimpleNamespace(
            id=16,
            road_id=8,
            start=Vector(19600, 12300),
            end=Vector(19600, 17700),
        )
        unrelated_lane = SimpleNamespace(
            id=99,
            road_id=9,
            start=Vector(19300, 19000),
            end=Vector(20700, 19000),
        )
        crosswalk = SimpleNamespace(
            id=16,
            road_id=8,
            start=Vector(19300, 19300),
            end=Vector(20700, 19300),
        )
        manager = object.__new__(WorldManager)
        manager._signal_vehicle_routes = {}
        manager.traffic_controller = SimpleNamespace(
            vehicles=[vehicle],
            lanes=[unrelated_lane, incoming_lane],
        )
        manager.red_light_conflict_nominal_speed_cm_s = 450.0

        selected = manager._lane_aligned_conflict_launch(
            {
                'crosswalk_projection': 0.0,
                'crosswalk_entry_direction': 1,
                'agent_position': {'x': 19300.0, 'y': 19300.0},
                'agent_direction': {'x': 1.0, 'y': 0.0},
                'agent_speed_cm_s': 200.0,
            },
            crosswalk,
            [vehicle],
            300.0,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(7, selected['vehicle'].id)
        self.assertEqual(16, selected['route']['incoming_lane'].id)
        self.assertEqual(
            'authored_lane_endpoint_extension',
            selected['route']['route_source'],
        )
        self.assertAlmostEqual(19600.0, selected['target_position'].x)
        self.assertAlmostEqual(19300.0, selected['target_position'].y)
        self.assertAlmostEqual(19600.0, selected['launch_position'].x)
        self.assertAlmostEqual(19000.0, selected['launch_position'].y)
        self.assertAlmostEqual(300.0, selected['launch_distance_cm'])
        self.assertAlmostEqual(1600.0, selected['lane_endpoint_extension_cm'])
    def test_endpoint_crosswalk_uses_nearest_lane_after_signal_expires(self):
        """A violation near the far curb still gets a real-lane vehicle."""
        vehicle = SimpleNamespace(id=7)
        nearer_lane = SimpleNamespace(
            id=18,
            road_id=9,
            start=Vector(7700.0, -10400.0),
            end=Vector(2300.0, -10400.0),
        )
        lane = SimpleNamespace(
            id=19,
            road_id=9,
            start=Vector(2300.0, -9600.0),
            end=Vector(7700.0, -9600.0),
        )
        crosswalk = SimpleNamespace(
            id=19,
            road_id=9,
            start=Vector(9300.0, -9300.0),
            end=Vector(9300.0, -10700.0),
        )
        manager = object.__new__(WorldManager)
        manager._signal_vehicle_routes = {}
        manager.traffic_controller = SimpleNamespace(
            vehicles=[vehicle],
            lanes=[lane, nearer_lane],
        )
        manager.red_light_conflict_nominal_speed_cm_s = 450.0
        selected = manager._lane_aligned_conflict_launch(
            {
                'crosswalk_projection': 0.856,
                'crosswalk_entry_direction': 1,
                'agent_position': {'x': 9300.0, 'y': -10498.4},
                'agent_speed_cm_s': 200.0,
            },
            crosswalk,
            [vehicle],
            752.21,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(19, selected['route']['incoming_lane'].id)
        self.assertEqual(
            'authored_lane_endpoint_extension',
            selected['route']['route_source'],
        )
        self.assertTrue(selected['target_behind_agent'])
        self.assertAlmostEqual(
            -898.4,
            selected['signed_agent_distance_to_target_cm'],
        )
        self.assertAlmostEqual(752.21, selected['launch_distance_cm'])


    def test_all_native_task_crosswalks_support_full_launch_range(self):
        checked = set()
        vehicle = SimpleNamespace(id=0)
        for map_name in (
            'map1_10roads',
            'map2_12roads',
            'map3_15roads',
            'map4_18roads',
            'map5_20roads',
        ):
            controller = RTTrafficController(
                Config(str(REPO_ROOT / 'config.yaml')),
                str(REPO_ROOT / 'data' / map_name / 'roads.json'),
                seed=1,
                num_vehicles=3,
                num_pedestrians=0,
            )
            tasks = json.loads(
                (REPO_ROOT / 'data' / map_name / 'tasks.json').read_text(
                    encoding='utf-8'
                )
            )['tasks']
            manager = object.__new__(WorldManager)
            manager._signal_vehicle_routes = {}
            manager.traffic_controller = controller
            manager.red_light_conflict_nominal_speed_cm_s = 450.0
            for source_task in tasks:
                task = align_task_crosswalks_to_rendered_geometry(
                    source_task,
                    700.0,
                    intersection_centers=[
                        [item.center.x, item.center.y]
                        for item in controller.intersections
                    ],
                )
                crosswalks = controller.get_route_crosswalks(
                    task['route_info']['shortest_path'],
                    task_edges=task['edges'],
                )
                for crosswalk in crosswalks:
                    if getattr(crosswalk, 'road_id', None) is None:
                        continue
                    key = (map_name, crosswalk.id)
                    if key in checked:
                        continue
                    checked.add(key)
                    axis = crosswalk.end - crosswalk.start
                    for entry_direction in (-1, 1):
                        for projection in (0.1, 0.5, 0.9):
                            position = crosswalk.start + axis * projection
                            selected = manager._lane_aligned_conflict_launch(
                                {
                                    'crosswalk_projection': projection,
                                    'crosswalk_entry_direction': entry_direction,
                                    'agent_position': {
                                        'x': position.x,
                                        'y': position.y,
                                    },
                                    'agent_speed_cm_s': 200.0,
                                },
                                crosswalk,
                                [vehicle],
                                900.0,
                            )
                            self.assertIsNotNone(
                                selected,
                                (map_name, crosswalk.id, entry_direction, projection),
                            )
                            self.assertEqual(
                                'authored_lane_endpoint_extension',
                                selected['route']['route_source'],
                            )
                            self.assertAlmostEqual(
                                900.0, selected['launch_distance_cm']
                            )
        self.assertGreater(len(checked), 0)

    def test_red_light_conflict_uses_next_real_lane_crossing(self):
        class FakeVehicle:
            def __init__(self, vehicle_id):
                self.id = vehicle_id
                self.position = Vector(0, 0)
                self._direction = Vector(1, 0)
                self.state = None
                self.attributes = None

            @property
            def direction(self):
                return self._direction

            @direction.setter
            def direction(self, yaw):
                import math
                self._direction = Vector(
                    math.cos(math.radians(yaw)),
                    math.sin(math.radians(yaw)),
                )

            def set_attributes(self, throttle, brake, steering):
                self.attributes = (throttle, brake, steering)

        calls = []
        vehicles = [FakeVehicle(0), FakeVehicle(1)]
        manager = object.__new__(WorldManager)
        manager.traffic_controller = SimpleNamespace(vehicles=vehicles)
        manager._signal_vehicle_routes = {
            0: {
                'approach_start': Vector(-10000, -200),
                'path_points': [Vector(-8600, -200)],
                'incoming_lane': SimpleNamespace(id=10),
                'outgoing_lane': SimpleNamespace(id=11),
            },
            1: {
                'approach_start': Vector(-10000, 400),
                'path_points': [Vector(-8600, 400)],
                'incoming_lane': SimpleNamespace(id=20),
                'outgoing_lane': SimpleNamespace(id=21),
            },
        }
        manager.signal_traffic_communicator = SimpleNamespace(
            unrealcv=SimpleNamespace(
                get_location=lambda name: [0.0, 0.0, 72.0],
                set_physics=lambda *args: calls.append(('physics', args)),
                set_location=lambda *args: calls.append(('location', args)),
                set_orientation=lambda *args: calls.append(('orientation', args)),
                set_collision=lambda *args: calls.append(('collision', args)),
                set_movable=lambda *args: calls.append(('movable', args)),
            ),
            get_vehicle_name=lambda vehicle_id: f'RT_SIGNAL_VEHICLE_{vehicle_id}',
            update_vehicle=lambda *args: calls.append(('state', args)),
        )
        manager.red_light_conflict_vehicle_events = []
        manager._active_red_light_conflict = None
        manager.red_light_conflict_launch_distance_min_cm = 250.0
        manager.red_light_conflict_launch_distance_max_cm = 250.0
        manager.red_light_conflict_collision_radius_cm = 250.0
        manager.red_light_conflict_vehicle_probability = 1.0
        manager.red_light_conflict_nominal_speed_cm_s = 450.0
        manager._traffic_consequence_rng = random.Random(1)
        manager.logger = SimpleNamespace(error=lambda *args: None)

        manager._handle_red_light_violation(
            {
                'crosswalk_id': 42,
                'sim_time_s': 12.5,
                'pedestrian_state': "DON'T WALK",
                'crosswalk_projection': 50.0 / 1400.0,
                'crosswalk_entry_direction': 1,
                'agent_direction': {'x': 0.0, 'y': 1.0},
                'agent_speed_cm_s': 200.0,
                'agent_position': {'x': -9300, 'y': -650},
            },
            make_crosswalk(),
        )

        event = manager.red_light_conflict_vehicle_events[0]
        self.assertEqual('lane_aligned_intercept', event['control_mode'])
        self.assertEqual(0, event['vehicle_id'])
        self.assertEqual(10, event['incoming_lane_id'])
        self.assertEqual(11, event['outgoing_lane_id'])
        self.assertAlmostEqual(-9300.0, event['target_position']['x'])
        self.assertAlmostEqual(-200.0, event['target_position']['y'])
        self.assertAlmostEqual(250.0, event['launch_distance_cm'])
        self.assertAlmostEqual(-9550.0, event['launch_position']['x'])
        self.assertAlmostEqual(-200.0, event['launch_position']['y'])
        self.assertEqual(72.0, event['launch_z_cm'])
        self.assertTrue(event['physics_quiesced_during_relocation'])
        self.assertLess(
            calls.index(('physics', ('RT_SIGNAL_VEHICLE_0', False))),
            next(index for index, call in enumerate(calls) if call[0] == 'location'),
        )
        self.assertNotIn(
            ('physics', ('RT_SIGNAL_VEHICLE_0', True)),
            calls,
        )
        self.assertIn(
            ('collision', ('RT_SIGNAL_VEHICLE_0', True)),
            calls,
        )
        self.assertTrue(event['collision_during_approach'])
        self.assertEqual(
            'ue_counter_or_launched_vehicle_swept_radius',
            event['collision_authority'],
        )
        self.assertEqual((0.0, 1.0, 0.0), vehicles[0].attributes)
        self.assertFalse(event['physics_during_approach'])
        self.assertTrue(event['collision_during_approach'])
        self.assertEqual(
            'lane_aligned_kinematic_intercept',
            event['motion_mode'],
        )
        self.assertEqual('collision_envelope_intercept', event['release_mode'])
        self.assertAlmostEqual(
            250.0,
            event['release_trigger_distance_cm'],
            places=2,
        )
        self.assertFalse(event['released'])
        self.assertEqual('staged', event['disposition'])
        self.assertEqual('staged', event['status'])
        self.assertEqual(1, len(manager._active_vehicle_conflicts))

        manager._handle_red_light_violation(
            {
                'event_id': 'red-light-overlapping',
                'crosswalk_id': 42,
                'sim_time_s': 12.6,
                'pedestrian_state': "DON'T WALK",
                'crosswalk_projection': 50.0 / 1400.0,
                'crosswalk_entry_direction': 1,
                'agent_direction': {'x': 0.0, 'y': 1.0},
                'agent_speed_cm_s': 200.0,
                'agent_position': {'x': -9300, 'y': -650},
            },
            make_crosswalk(),
        )

        overlapping = manager.red_light_conflict_vehicle_events[1]
        self.assertEqual(1, overlapping['vehicle_id'])
        self.assertNotEqual('already_active', overlapping['disposition'])
        self.assertEqual(2, len(manager._active_vehicle_conflicts))

        manager.agent = SimpleNamespace(
            position=Vector(-9300, -500),
            sim_time_elapsed=14.0,
            failed=False,
            success=True,
            failure_reason=None,
            vehicle_collision_count=0,
            collision_count=0,
        )
        manager._update_red_light_conflict_vehicle()

        self.assertFalse(event['released'])
        self.assertEqual('staged', event['disposition'])
        self.assertEqual('staged', event['status'])

        manager.agent.position = Vector(-9300, -300)
        manager.agent.sim_time_elapsed = 14.5
        manager._update_red_light_conflict_vehicle()

        self.assertTrue(event['released'])
        self.assertEqual('launched', event['disposition'])
        self.assertEqual('launched', event['status'])
        self.assertEqual((0.0, 1.0, 0.0), vehicles[0].attributes)
        self.assertAlmostEqual(
            100.0,
            event['agent_distance_to_target_at_release_cm'],
        )

    def test_staged_conflict_vehicle_timeout_forces_launch(self):
        manager = object.__new__(WorldManager)
        manager.red_light_conflict_collision_radius_cm = 100.0
        manager.red_light_conflict_release_wait_timeout_s = 12.0
        manager.agent = SimpleNamespace(
            position=Vector(500.0, 0.0),
            sim_time_elapsed=13.0,
            red_light_conflict_active=True,
        )
        vehicle = SimpleNamespace(
            id=4,
            position=Vector(-300.0, 0.0),
            state=VehicleState.STOPPED,
            set_attributes=MagicMock(),
        )
        record = {
            'minimum_agent_distance_cm': 800.0,
            'impact_zone_reached': False,
            'collision_triggered': False,
            'released': False,
            'disposition': 'staged',
            'status': 'staged',
            'release_trigger_distance_cm': 100.0,
            'release_wait_timeout_s': 12.0,
        }
        active = {
            'vehicle': vehicle,
            'record': record,
            'launch_position': Vector(-300.0, 0.0),
            'target_position': Vector(0.0, 0.0),
            'direction': Vector(1.0, 0.0),
            'launch_sim_time_s': 0.0,
        }
        manager._active_vehicle_conflicts = [active]
        manager._active_red_light_conflict = active
        manager.signal_traffic_communicator = SimpleNamespace(
            update_vehicle=MagicMock(),
        )
        manager._retire_conflict_vehicle = MagicMock(
            side_effect=lambda _active, status: record.update(status=status)
        )

        manager._update_red_light_conflict_vehicle()

        self.assertTrue(record['released'])
        self.assertEqual('launched', record['disposition'])
        self.assertEqual('launched', record['status'])
        self.assertEqual(13.0, record['release_wait_elapsed_s'])
        self.assertEqual(
            'release_wait_timeout',
            record['release_reason'],
        )
        self.assertEqual(
            500.0,
            record['agent_distance_to_target_at_release_cm'],
        )
        manager._retire_conflict_vehicle.assert_not_called()
        self.assertEqual([active], manager._active_vehicle_conflicts)
        self.assertIs(manager._active_red_light_conflict, active)

    def test_conflict_vehicle_ue_launch_error_records_failed_disposition(self):
        class FakeVehicle:
            def __init__(self):
                self.id = 0
                self.position = Vector(0, 0)
                self._direction = Vector(1, 0)
                self.state = None

            @property
            def direction(self):
                return self._direction

            @direction.setter
            def direction(self, _yaw):
                self._direction = Vector(1, 0)

            def set_attributes(self, *_args):
                return None

        vehicle = FakeVehicle()
        manager = object.__new__(WorldManager)
        manager.traffic_controller = SimpleNamespace(vehicles=[vehicle])
        manager._signal_vehicle_routes = {
            0: {
                'approach_start': Vector(-10000, -200),
                'path_points': [Vector(-8600, -200)],
                'incoming_lane': SimpleNamespace(id=10),
                'outgoing_lane': SimpleNamespace(id=11),
            },
        }
        manager.signal_traffic_communicator = SimpleNamespace(
            unrealcv=SimpleNamespace(
                get_location=lambda name: [0.0, 0.0, 72.0],
                set_physics=MagicMock(),
                set_collision=MagicMock(),
                set_location=MagicMock(
                    side_effect=ConnectionError('UE socket closed')
                ),
            ),
            get_vehicle_name=lambda vehicle_id: (
                f'RT_SIGNAL_VEHICLE_{vehicle_id}'
            ),
        )
        manager.red_light_conflict_vehicle_events = []
        manager._active_red_light_conflict = None
        manager.red_light_conflict_launch_distance_min_cm = 250.0
        manager.red_light_conflict_launch_distance_max_cm = 250.0
        manager.red_light_conflict_collision_radius_cm = 250.0
        manager.red_light_conflict_vehicle_probability = 1.0
        manager.red_light_conflict_nominal_speed_cm_s = 450.0
        manager._traffic_consequence_rng = random.Random(1)
        manager.logger = MagicMock()

        with self.assertRaisesRegex(ConnectionError, 'UE socket closed'):
            manager._handle_red_light_violation(
                {
                    'event_id': 'red-light-0001',
                    'crosswalk_id': 42,
                    'sim_time_s': 12.5,
                    'pedestrian_state': "DON'T WALK",
                    'crosswalk_projection': 50.0 / 1400.0,
                    'crosswalk_entry_direction': 1,
                    'agent_direction': {'x': 0.0, 'y': 1.0},
                    'agent_speed_cm_s': 200.0,
                    'agent_position': {'x': -9300, 'y': -650},
                },
                make_crosswalk(),
            )

        event = manager.red_light_conflict_vehicle_events[0]
        self.assertEqual('failed', event['disposition'])
        self.assertEqual('failed', event['status'])
        self.assertEqual('ConnectionError', event['error_type'])
        self.assertEqual('lane_aligned_intercept', event['control_mode'])
        self.assertIsNone(manager._active_red_light_conflict)

    def test_conflict_vehicle_collision_envelope_is_terminal_at_1_5m(self):
        manager = object.__new__(WorldManager)
        manager.red_light_conflict_collision_radius_cm = 150.0
        agent = object.__new__(RTAgent)
        agent.position = Vector(150, 0)
        agent.sim_time_elapsed = 4.0
        agent.failed = False
        agent.success = True
        agent.failure_reason = None
        agent.vehicle_collision_count = 0
        agent.collision_count = 0
        agent.name = 'RT_AGENT'
        agent.communicator = SimpleNamespace(
            get_vehicle_collision_number=lambda _: 0,
        )
        agent.last_ue_collision_count = {'vehicle': 0}
        agent._pending_swept_vehicle_collisions = 0
        agent.red_light_conflict_active = True
        agent.logger = MagicMock()
        manager.agent = agent
        vehicle = SimpleNamespace(
            id=4,
            position=Vector(0, 0),
            state=VehicleState.MOVING,
            set_attributes=lambda *args: None,
        )
        record = {
            'minimum_agent_distance_cm': 450.0,
            'impact_zone_reached': False,
            'collision_triggered': False,
            'released': True,
            'status': 'launched',
        }
        manager._active_red_light_conflict = {
            'vehicle': vehicle,
            'record': record,
            'launch_position': Vector(-100, 0),
            'direction': Vector(1, 0),
            'launch_sim_time_s': 0.0,
            'release_sim_time_s': 0.0,
            'target_position': Vector(1000, 0),
        }
        manager.signal_traffic_communicator = SimpleNamespace(
            update_vehicle=lambda *args: None,
        )
        manager.vehicle_impact_zone_count = 0
        manager._retire_conflict_vehicle = MagicMock(
            side_effect=lambda *_: setattr(
                agent,
                'red_light_conflict_active',
                False,
            )
        )
        manager.logger = MagicMock()

        manager._update_red_light_conflict_vehicle()

        self.assertFalse(agent.failed)
        self.assertEqual(1, agent._pending_swept_vehicle_collisions)
        self.assertEqual(1, agent._check_vehicle_collision_failure())
        self.assertTrue(agent.failed)
        self.assertFalse(agent.success)
        self.assertEqual('vehicle_collision', agent.failure_reason)
        self.assertEqual('collision', record['status'])
        self.assertEqual(
            'conflict_vehicle_distance_envelope',
            record['impact_zone_detection_source'],
        )
        self.assertEqual(150.0, record['collision_distance_cm'])
        self.assertEqual(
            'launched_vehicle_swept_radius',
            record['collision_authority'],
        )

    def test_ue_vehicle_counter_marks_active_conflict_as_collision(self):
        manager = object.__new__(WorldManager)
        manager.agent = SimpleNamespace(
            failure_reason='vehicle_collision',
            vehicle_collision_count=1,
            sim_time_elapsed=12.5,
        )
        record = {
            'status': 'impact_zone_reached',
            'impact_zone_reached': True,
            'collision_triggered': False,
        }
        active = {'record': record}
        manager._active_red_light_conflict = active
        manager._active_vehicle_conflicts = [active]
        manager._retire_conflict_vehicle = MagicMock(
            side_effect=lambda _, status: record.update(status=status)
        )

        self.assertTrue(manager._record_ue_vehicle_collision_outcome())

        self.assertTrue(record['collision_triggered'])
        self.assertEqual('unreal_engine_counter', record['collision_source'])
        self.assertEqual('unreal_engine_counter', record['collision_authority'])
        self.assertEqual(1, record['ue_vehicle_collision_count'])
        self.assertEqual(12.5, record['collision_sim_time_s'])
        self.assertEqual('collision', record['status'])
        manager._retire_conflict_vehicle.assert_called_once_with(
            active,
            'collision',
        )
        self.assertEqual([], manager._active_vehicle_conflicts)
        self.assertIsNone(manager._active_red_light_conflict)

    def test_impact_zone_without_ue_counter_is_not_recorded_as_collision(self):
        manager = object.__new__(WorldManager)
        manager.agent = SimpleNamespace(
            failure_reason=None,
            vehicle_collision_count=0,
            sim_time_elapsed=12.5,
        )
        record = {
            'status': 'impact_zone_reached',
            'impact_zone_reached': True,
            'collision_triggered': False,
        }
        active = {'record': record}
        manager._active_red_light_conflict = active
        manager._active_vehicle_conflicts = [active]
        manager._retire_conflict_vehicle = MagicMock()

        self.assertFalse(manager._record_ue_vehicle_collision_outcome())

        self.assertFalse(record['collision_triggered'])
        self.assertNotIn('collision_source', record)
        manager._retire_conflict_vehicle.assert_not_called()
        self.assertIs(manager._active_red_light_conflict, active)

    def test_conflict_vehicle_swept_envelope_catches_between_tick_impact(self):
        manager = object.__new__(WorldManager)
        manager.red_light_conflict_collision_radius_cm = 100.0
        manager.agent = SimpleNamespace(
            position=Vector(0, 0),
            sim_time_elapsed=5.0,
            failed=False,
            success=True,
            failure_reason=None,
            vehicle_collision_count=0,
            collision_count=0,
        )
        vehicle = SimpleNamespace(
            id=5,
            position=Vector(300, 0),
            state=VehicleState.MOVING,
            set_attributes=lambda *args: None,
        )
        record = {
            'minimum_agent_distance_cm': 300.0,
            'impact_zone_reached': False,
            'collision_triggered': False,
            'released': True,
            'status': 'launched',
        }
        manager._active_red_light_conflict = {
            'vehicle': vehicle,
            'record': record,
            'launch_position': Vector(-300, 0),
            'direction': Vector(1, 0),
            'launch_sim_time_s': 0.0,
            'release_sim_time_s': 0.0,
            'last_vehicle_position': Vector(-300, 0),
            'last_agent_position': Vector(0, 0),
            'target_position': Vector(1000, 0),
        }
        manager.signal_traffic_communicator = SimpleNamespace(
            update_vehicle=lambda *args: None,
        )
        manager._retire_conflict_vehicle = MagicMock()
        manager.logger = MagicMock()

        manager._update_red_light_conflict_vehicle()

        self.assertTrue(manager.agent.failed)
        self.assertEqual('vehicle_collision', manager.agent.failure_reason)
        self.assertEqual('collision', record['status'])
        self.assertEqual(
            'conflict_vehicle_swept_envelope',
            record['impact_zone_detection_source'],
        )
        self.assertAlmostEqual(0.0, record['impact_zone_distance_cm'])
        self.assertEqual(
            'launched_vehicle_swept_radius',
            record['collision_authority'],
        )

    def test_noncompliant_signal_pedestrian_can_enter_on_red(self):
        pedestrian = SimpleNamespace(
            id=7,
            position=Vector(0, 0),
            direction=Vector(1, 0),
            speed=100.0,
            current_sidewalk=SimpleNamespace(id=1),
            state=PedestrianState.MOVE_FORWARD,
            stop_reason=None,
            waypoints=[Vector(10, 0)],
            follows_traffic_signal=False,
            is_close_to_end=lambda _: True,
            add_waypoint=MagicMock(),
            change_to_next_sidewalk=MagicMock(),
            pop_waypoint=MagicMock(),
            compute_control=lambda _: (0, None),
        )
        crosswalk = SimpleNamespace(
            id=19,
            start=Vector(0, 0),
            end=Vector(0, 600),
        )
        intersection = SimpleNamespace(
            get_crosswalk_light_state=lambda _: (
                TrafficSignalState.PEDESTRIAN_RED,
                0.0,
            )
        )
        controller = SimpleNamespace(
            get_waypoints_for_pedestrian=lambda *_: (
                SimpleNamespace(id=3),
                crosswalk,
                [Vector(0, 600)],
                intersection,
            )
        )
        manager = object.__new__(PedestrianManager)
        manager.pedestrians = [pedestrian]
        manager.signal_entry_events = []
        manager.config = {
            'traffic.pedestrian.agent_yield_distance': 250.0,
            'traffic.pedestrian.agent_yield_lateral_margin': 130.0,
            'traffic.pedestrian.waypoint_distance_threshold': 200.0,
        }
        manager.logger = MagicMock()

        manager.update_pedestrians(MagicMock(), controller, [])

        pedestrian.add_waypoint.assert_called_once()
        pedestrian.change_to_next_sidewalk.assert_called_once()
        self.assertEqual(1, len(manager.signal_entry_events))
        event = manager.signal_entry_events[0]
        self.assertTrue(event['entered_against_signal'])
        self.assertFalse(event['follows_traffic_signal'])

    def test_signal_compliance_probability_produces_mixed_population(self):
        flags = PedestrianManager._signal_compliance_flags(2, 0.5)

        self.assertEqual(1, sum(flags))
        self.assertEqual(1, len(flags) - sum(flags))

    def test_legacy_red_light_violation_emits_event_and_callback(self):
        events = []
        edge = {
            'type': 'crosswalk',
            'node1': [-9300, -700],
            'node2': [-9300, 700],
        }
        signal = SimpleNamespace(id=7, state=None)
        agent = object.__new__(RTAgent)
        agent.task_edges = [edge]
        agent.position = Vector(-9300, 0)
        agent.direction = 90.0
        agent.speed = 200.0
        agent.sim_time_elapsed = 12.5
        agent.traffic_signals = [signal]
        agent.last_execution_traffic_light_snapshot = {}
        agent.violated_crosswalks = set()
        agent.red_light_violations_count = 0
        agent.red_light_violation_events = []
        agent.red_light_violation_callback = (
            lambda event, crosswalk: events.append((event, crosswalk))
        )
        agent.logger = MagicMock()
        agent._refresh_traffic_signal_states = lambda: None
        agent._signal_for_crossing = lambda *_: signal
        agent._traffic_signal_snapshot = lambda *_args, **_kwargs: (
            (
                TrafficSignalState.VEHICLE_GREEN,
                TrafficSignalState.PEDESTRIAN_RED,
            ),
            5.0,
            'legacy-test',
        )

        violated = agent._check_red_light_violation(Vector(-9300, -400))

        self.assertTrue(violated)
        self.assertEqual(1, agent.red_light_violations_count)
        self.assertEqual(1, len(agent.red_light_violation_events))
        self.assertEqual('red-light-0001', events[0][0]['event_id'])
        self.assertTrue(events[0][0]['legacy_fallback'])
        self.assertEqual(events[0][0]['crosswalk_id'], events[0][1].id)

    def test_legacy_path_entry_outside_crosswalk_is_not_violation(self):
        events = []
        edge = {
            'type': 'crosswalk',
            'node1': [-9300, -700],
            'node2': [-9300, 700],
        }
        signal = SimpleNamespace(id=7, state=None)
        agent = object.__new__(RTAgent)
        agent.task_edges = [edge]
        agent.position = Vector(-9300, -900)
        agent.direction = 90.0
        agent.speed = 200.0
        agent.sim_time_elapsed = 12.5
        agent.traffic_signals = [signal]
        agent.last_execution_traffic_light_snapshot = {}
        agent.violated_crosswalks = set()
        agent.red_light_violations_count = 0
        agent.red_light_violation_events = []
        agent.red_light_violation_callback = (
            lambda event, crosswalk: events.append((event, crosswalk))
        )
        agent.logger = MagicMock()
        agent._refresh_traffic_signal_states = lambda: None
        agent._signal_for_crossing = lambda *_: signal
        agent._traffic_signal_snapshot = lambda *_args, **_kwargs: (
            (
                TrafficSignalState.VEHICLE_GREEN,
                TrafficSignalState.PEDESTRIAN_RED,
            ),
            5.0,
            'legacy-test',
        )

        violated = agent._check_red_light_violation(Vector(-9300, -400))

        self.assertFalse(violated)
        self.assertEqual(0, agent.red_light_violations_count)
        self.assertEqual([], agent.red_light_violation_events)
        self.assertEqual([], events)

    def test_visual_only_synced_crosswalk_never_uses_legacy_schedule(self):
        agent = make_agent(
            position=Vector(-9300, 0),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
        )
        signal = SimpleNamespace(id=8, state=None)
        agent.traffic_policy = TRAFFIC_POLICY_VISUAL_ONLY
        agent.task_edges = [{
            'type': 'crosswalk',
            'node1': [-9300, -700],
            'node2': [-9300, 700],
        }]
        agent.traffic_signals = [signal]
        agent.last_execution_traffic_light_snapshot = {}
        agent._refresh_traffic_signal_states = MagicMock()
        agent._signal_for_crossing = MagicMock(return_value=signal)
        agent._traffic_signal_snapshot = MagicMock(return_value=(
            (
                TrafficSignalState.VEHICLE_GREEN,
                TrafficSignalState.PEDESTRIAN_RED,
            ),
            18.4,
            'local_pedestrian_schedule',
        ))

        violated = agent._check_red_light_violation(Vector(-9300, 400))

        self.assertFalse(violated)
        self.assertEqual(0, agent.red_light_violations_count)
        self.assertEqual([], agent.red_light_violation_events)
        agent._refresh_traffic_signal_states.assert_not_called()
        agent._signal_for_crossing.assert_not_called()
        agent._traffic_signal_snapshot.assert_not_called()

    def test_red_light_conflict_vehicle_can_be_disabled_independently(self):
        manager = object.__new__(WorldManager)
        manager.red_light_conflict_vehicle_enabled = False
        manager.red_light_conflict_vehicle_events = []
        manager._active_red_light_conflict = None
        manager.traffic_controller = SimpleNamespace(
            vehicles=[SimpleNamespace(id=0)]
        )

        manager._handle_red_light_violation(
            {
                "crosswalk_id": 42,
                "sim_time_s": 12.5,
                "pedestrian_state": "DON'T WALK",
                "crosswalk_projection": 0.5,
                "agent_position": {"x": -9300, "y": 0},
            },
            make_crosswalk(),
        )

        self.assertEqual(
            'disabled',
            manager.red_light_conflict_vehicle_events[0]['disposition'],
        )
        self.assertIsNone(manager._active_red_light_conflict)

    def test_red_light_conflict_vehicle_probability_records_skip(self):
        manager = object.__new__(WorldManager)
        manager.red_light_conflict_vehicle_enabled = True
        manager.red_light_conflict_vehicle_probability = 0.0
        manager._traffic_consequence_rng = random.Random(7)
        manager._traffic_consequence_rng_seed = 1_234_567
        manager._traffic_consequence_draw_count = 0
        manager.red_light_conflict_vehicle_events = []
        manager._active_red_light_conflict = None
        manager.traffic_controller = SimpleNamespace(
            vehicles=[SimpleNamespace(id=0)]
        )

        manager._handle_red_light_violation(
            {
                'event_id': 'red-light-0001',
                'crosswalk_id': 42,
                'sim_time_s': 12.5,
                'pedestrian_state': "DON'T WALK",
                'crosswalk_projection': 0.5,
            },
            make_crosswalk(),
        )

        event = manager.red_light_conflict_vehicle_events[0]
        self.assertEqual('red-light-0001', event['event_id'])
        self.assertEqual('probability_skipped', event['disposition'])
        self.assertEqual('probability_skipped', event['status'])
        self.assertEqual(0.0, event['vehicle_probability'])
        self.assertEqual(1_234_567, event['vehicle_probability_rng_seed'])
        self.assertEqual(0, event['vehicle_probability_draw_index'])
        self.assertEqual(1, manager._traffic_consequence_draw_count)
        self.assertIsNone(manager._active_red_light_conflict)

    def test_probability_one_never_skips_even_for_high_draw(self):
        manager = object.__new__(WorldManager)
        manager.red_light_conflict_vehicle_enabled = True
        manager.red_light_conflict_vehicle_probability = 1.0
        manager._traffic_consequence_rng = SimpleNamespace(
            random=lambda: 0.999999,
        )
        manager._traffic_consequence_rng_seed = 7_654_321
        manager._traffic_consequence_draw_count = 0
        manager._active_red_light_conflict = None
        manager.traffic_controller = SimpleNamespace(vehicles=[])
        manager.logger = SimpleNamespace(warning=lambda *args: None)
        events = []

        event = manager._handle_conflict_trigger(
            {
                'event_id': 'illegal-crossing-0001',
                'crosswalk_id': 42,
                'sim_time_s': 12.5,
            },
            make_crosswalk(),
            events,
            trigger_type='illegal_crossing',
        )

        self.assertEqual('unavailable', event['disposition'])
        self.assertNotEqual('probability_skipped', event['status'])
        self.assertEqual(1.0, event['vehicle_probability'])
        self.assertEqual(0.999999, event['vehicle_probability_draw'])
        self.assertEqual(7_654_321, event['vehicle_probability_rng_seed'])
        self.assertEqual(0, event['vehicle_probability_draw_index'])
        self.assertEqual(1, manager._traffic_consequence_draw_count)

    def test_missing_conflict_vehicle_actor_is_respawned_before_collision_setup(self):
        calls = []
        objects = []
        vehicle = SimpleNamespace(
            id=9,
            position=Vector(0, 0),
            direction=0.0,
            state=None,
            vehicle_reference='/Game/TrafficSystem/Vehicle/Vehicle3.Vehicle3_C',
            set_attributes=lambda *args: calls.append(('attributes', args)),
        )

        def spawn(reference, name):
            calls.append(('spawn', (reference, name)))
            objects.append(name)

        unrealcv = SimpleNamespace(
            get_objects=lambda: list(objects),
            spawn_bp_asset=spawn,
            set_location=lambda *args: calls.append(('location', args)),
            set_orientation=lambda *args: calls.append(('orientation', args)),
            set_collision=lambda *args: calls.append(('collision', args)),
            set_movable=lambda *args: calls.append(('movable', args)),
        )
        manager = object.__new__(WorldManager)
        manager.red_light_conflict_vehicle_enabled = True
        manager.traffic_controller = SimpleNamespace(vehicles=[vehicle])
        manager.signal_traffic_communicator = SimpleNamespace(
            unrealcv=unrealcv,
            get_vehicle_name=lambda vehicle_id: f'RT_SIGNAL_VEHICLE_{vehicle_id}',
            update_vehicle=lambda *args: calls.append(('state', args)),
        )
        manager.red_light_conflict_vehicle_events = []
        manager.illegal_crossing_conflict_vehicle_events = []
        manager.vehicle_conflict_events = []
        manager._active_red_light_conflict = None
        manager._active_vehicle_conflicts = []
        manager._retired_signal_vehicle_ids = set()
        manager._conflict_vehicle_rng = SimpleNamespace(
            uniform=lambda minimum, maximum: 400.0
        )
        manager.logger = SimpleNamespace(
            error=lambda *args: None,
            warning=lambda *args: None,
        )

        manager._handle_red_light_violation(
            {
                'event_id': 'red-light-respawn',
                'crosswalk_id': 42,
                'sim_time_s': 10.0,
                'pedestrian_state': "DON'T WALK",
                'agent_position': {'x': -9300, 'y': 0},
            },
            make_crosswalk(),
        )

        event = manager.red_light_conflict_vehicle_events[0]
        self.assertTrue(event['vehicle_actor_respawned'])
        self.assertEqual('RT_SIGNAL_VEHICLE_9', objects[0])
        self.assertTrue(any(call[0] == 'spawn' for call in calls))
        self.assertTrue(any(call[0] == 'collision' for call in calls))

    def test_stale_conflict_vehicle_actor_is_respawned_after_location_error(self):
        calls = []
        name = 'RT_SIGNAL_VEHICLE_0'
        vehicle = SimpleNamespace(
            id=0,
            position=Vector(0.0, 0.0),
            direction=Vector(1.0, 0.0),
            state=None,
            vehicle_reference='/Game/TrafficSystem/Vehicle/Vehicle3.Vehicle3_C',
            set_attributes=lambda *args: calls.append(('attributes', args)),
        )
        location_calls = 0
        objects = [name]
        destroyed = False
        post_destroy_polls = 0

        def get_location(actor_name):
            nonlocal location_calls
            location_calls += 1
            calls.append(('get_location', actor_name))
            if location_calls == 1:
                raise ValueError("could not convert string to float: 'error'")
            return [0.0, 0.0, 72.0]

        def get_objects():
            nonlocal post_destroy_polls
            calls.append(('get_objects', tuple(objects)))
            if destroyed:
                post_destroy_polls += 1
                if post_destroy_polls >= 2:
                    objects.clear()
            return list(objects)

        def destroy(actor_name):
            nonlocal destroyed
            calls.append(('destroy', (actor_name,)))
            destroyed = True


        unrealcv = SimpleNamespace(
            get_objects=get_objects,
            get_location=get_location,
            destroy=destroy,
            spawn_bp_asset=lambda *args: calls.append(('spawn', args)),
            set_physics=lambda *args: calls.append(('physics', args)),
            set_location=lambda *args: calls.append(('location', args)),
            set_orientation=lambda *args: calls.append(('orientation', args)),
            set_collision=lambda *args: calls.append(('collision', args)),
            set_movable=lambda *args: calls.append(('movable', args)),
        )
        manager = object.__new__(WorldManager)
        manager.traffic_controller = SimpleNamespace(vehicles=[vehicle])
        manager._signal_vehicle_routes = {
            0: {
                'approach_start': Vector(-10000.0, -200.0),
                'path_points': [Vector(-8600.0, -200.0)],
                'incoming_lane': SimpleNamespace(id=10),
                'outgoing_lane': SimpleNamespace(id=11),
            },
        }
        manager.signal_traffic_communicator = SimpleNamespace(
            unrealcv=unrealcv,
            get_vehicle_name=lambda vehicle_id: f'RT_SIGNAL_VEHICLE_{vehicle_id}',
            update_vehicle=lambda *args: calls.append(('state', args)),
        )
        manager.red_light_conflict_vehicle_enabled = True
        manager.red_light_conflict_vehicle_events = []
        manager._active_red_light_conflict = None
        manager._active_vehicle_conflicts = []
        manager._retired_signal_vehicle_ids = set()
        manager.red_light_conflict_launch_distance_min_cm = 250.0
        manager.red_light_conflict_launch_distance_max_cm = 250.0
        manager.red_light_conflict_collision_radius_cm = 250.0
        manager.red_light_conflict_vehicle_probability = 1.0
        manager.red_light_conflict_nominal_speed_cm_s = 450.0
        manager._traffic_consequence_rng = random.Random(1)
        manager.logger = MagicMock()

        manager._handle_red_light_violation(
            {
                'event_id': 'red-light-stale-respawn',
                'crosswalk_id': 42,
                'sim_time_s': 12.5,
                'pedestrian_state': "DON'T WALK",
                'crosswalk_projection': 50.0 / 1400.0,
                'crosswalk_entry_direction': 1,
                'agent_direction': {'x': 0.0, 'y': 1.0},
                'agent_speed_cm_s': 200.0,
                'agent_position': {'x': -9300.0, 'y': -650.0},
            },
            make_crosswalk(),
        )

        event = manager.red_light_conflict_vehicle_events[0]
        self.assertEqual('lane_aligned_intercept', event['control_mode'])
        self.assertEqual(2, location_calls)
        self.assertTrue(event['vehicle_actor_respawned'])
        self.assertIn(('destroy', (name,)), calls)
        self.assertIn(
            ('spawn', (vehicle.vehicle_reference, name)),
            calls,
        )

        destroy_index = calls.index(('destroy', (name,)))
        spawn_index = calls.index(('spawn', (vehicle.vehicle_reference, name)))
        self.assertGreater(spawn_index, destroy_index)
        self.assertTrue(any(
            call[0] == 'get_objects' for call in calls[destroy_index + 1:spawn_index]
        ))
    def test_illegal_crossing_emits_one_event_per_continuous_entry(self):
        events = []
        agent = object.__new__(RTAgent)
        agent.position = Vector(300, 100)
        agent._direction = Vector(1, 0)
        agent.last_action_start_position = Vector(100, 100)
        agent.route_crosswalks = []
        agent.sim_time_elapsed = 7.25
        agent.speed = 200.0
        agent.illegal_crossing_excursion_active = False
        agent.illegal_crossing_violations_count = 0
        agent.illegal_crossing_events = []
        agent.illegal_crossing_callback = events.append
        agent.logger = SimpleNamespace(
            error=lambda *args: None,
            exception=lambda *args: None,
        )

        self.assertTrue(agent._check_illegal_crossing_trigger(1, 0))
        self.assertFalse(agent._check_illegal_crossing_trigger(1, 1))
        self.assertEqual(1, agent.illegal_crossing_violations_count)
        self.assertEqual('illegal-crossing-0001', events[0]['event_id'])
        self.assertEqual('illegal_crossing', events[0]['trigger_type'])
        self.assertTrue(events[0]['outside_authored_crossing'])

    def test_illegal_crossing_builds_virtual_crossing(self):
        manager = object.__new__(WorldManager)
        manager.red_light_conflict_vehicle_probability = 1.0
        manager.red_light_conflict_vehicle_events = []
        manager.illegal_crossing_conflict_vehicle_events = []
        virtual_crossing = SimpleNamespace(
            id='illegal-crossing-lane-2-0',
            lane_distance_cm=125.0,
        )
        manager._road_entry_conflict_crosswalk = MagicMock(
            return_value=virtual_crossing
        )
        expected = {'disposition': 'launched'}
        manager._handle_conflict_trigger = MagicMock(return_value=expected)

        result = manager._handle_illegal_crossing(
            {
                'event_id': 'illegal-crossing-0001',
                'trigger_type': 'illegal_crossing',
                'sim_time_s': 8.0,
                'agent_position': {'x': 1000.0, 'y': 2000.0},
                'agent_direction': {'x': 0.0, 'y': 1.0},
            },
        )

        self.assertIs(result, expected)
        manager._road_entry_conflict_crosswalk.assert_called_once()
        routed_event = manager._handle_conflict_trigger.call_args.args[0]
        self.assertEqual(
            'illegal-crossing-lane-2-0',
            routed_event['crosswalk_id'],
        )
        self.assertEqual(125.0, routed_event['vehicle_lane_distance_cm'])
        self.assertEqual(
            {'x': 0.0, 'y': 1.0},
            routed_event['agent_direction'],
        )
        self.assertIs(
            virtual_crossing,
            manager._handle_conflict_trigger.call_args.args[1],
        )
        self.assertIs(
            manager.illegal_crossing_conflict_vehicle_events,
            manager._handle_conflict_trigger.call_args.args[2],
        )

    def test_native_controller_module_path_can_match_full_city_package(self):
        self.assertEqual(
            '/Script/SimWorld.RTTrafficIntersectionController',
            WorldManager._native_intersection_controller_path(),
        )
        with patch.dict(
            os.environ,
            {
                'SIMWORLD_INTERSECTION_CONTROLLER_MODEL_PATH':
                    '/Script/gym_citynav.RTTrafficIntersectionController'
            },
        ):
            self.assertEqual(
                '/Script/gym_citynav.RTTrafficIntersectionController',
                WorldManager._native_intersection_controller_path(),
            )

    def test_task5_route_segment_matches_map_crosswalk(self):
        crosswalk = make_crosswalk()
        fake_controller = SimpleNamespace(crosswalks=[crosswalk])
        route = [
            [-7797.6773, -700],
            [-9300, -700],
            [-9300, 700],
            [-8193.4948, 700],
        ]

        matched = RTTrafficController.get_route_crosswalks(
            fake_controller,
            route,
        )

        self.assertEqual([crosswalk.id], [item.id for item in matched])

    def test_task_edge_crosswalk_is_retained_when_no_road_zebra_matches(self):
        fake_controller = SimpleNamespace(crosswalks=[])
        route = [[7498.2395, -700], [700, -700], [-700, -700]]
        edges = [
            {
                "node1": [9300, -700],
                "node2": [700, -700],
                "type": "sidewalk",
            },
            {
                "node1": [700, -700],
                "node2": [-700, -700],
                "type": "crosswalk",
            },
        ]

        matched = RTTrafficController.get_route_crosswalks(
            fake_controller,
            route,
            task_edges=edges,
        )

        self.assertEqual(1, len(matched))
        self.assertEqual("task_edge_1", matched[0].id)
        self.assertEqual(Vector(700, -700), matched[0].start)
        self.assertEqual(Vector(-700, -700), matched[0].end)

    def test_every_signalled_zebra_has_two_endpoint_pedestrian_heads(self):
        config = Config(str(REPO_ROOT / "config.yaml"))
        endpoint_offset = config[
            "traffic.traffic_signal.pedestrian_light_endpoint_offset"
        ]
        lateral_offset = config[
            "traffic.traffic_signal.pedestrian_light_lateral_offset"
        ]

        for roads_file in sorted((REPO_ROOT / "data").glob("map*/roads.json")):
            controller = RTTrafficController(
                config,
                str(roads_file),
                seed=1,
                num_vehicles=0,
                num_pedestrians=0,
            )
            checked_crosswalks = 0
            for intersection in controller.intersections:
                if not intersection.traffic_lights:
                    self.assertFalse(intersection.pedestrian_lights)
                    continue

                for crosswalk in intersection.crosswalks:
                    heads = [
                        light
                        for light in intersection.pedestrian_lights
                        if light.crosswalk_id == crosswalk.id
                    ]
                    self.assertEqual(
                        2,
                        len(heads),
                        f"{roads_file.parent.name} crosswalk {crosswalk.id}",
                    )
                    direction = (crosswalk.end - crosswalk.start).normalize()
                    normal = Vector(direction.y, -direction.x)
                    expected = (
                        (
                            (
                                crosswalk.start
                                - direction * endpoint_offset
                                + normal * lateral_offset
                            ),
                            direction,
                        ),
                        (
                            (
                                crosswalk.end
                                + direction * endpoint_offset
                                - normal * lateral_offset
                            ),
                            direction * -1,
                        ),
                    )
                    for position, facing in expected:
                        self.assertTrue(
                            any(
                                head.position == position
                                and head.direction == facing
                                for head in heads
                            ),
                            f"missing endpoint head at {position}",
                        )
                    checked_crosswalks += 1
            self.assertGreater(checked_crosswalks, 0, roads_file.parent.name)

    def test_pedestrian_signal_heads_clear_scripted_corner_lanes(self):
        config = Config(str(REPO_ROOT / "config.yaml"))
        lateral_offset = float(config[
            "traffic.traffic_signal.pedestrian_light_lateral_offset"
        ])
        lane_offset = SCRIPTED_PEDESTRIAN_MOTION_POLICY[
            "right_hand_lane_offset_cm"
        ]
        blocking_distance = SCRIPTED_PEDESTRIAN_MOTION_POLICY[
            "observed_capsule_blocking_distance_cm"
        ]

        self.assertGreater(
            lateral_offset - lane_offset,
            blocking_distance,
        )

    def test_map1_ordered_routes_distinguish_signalized_and_terminal_zebras(self):
        controller = RTTrafficController(
            Config(str(REPO_ROOT / "config.yaml")),
            str(REPO_ROOT / "data" / "map1_10roads" / "roads.json"),
            seed=1,
            num_vehicles=3,
            num_pedestrians=2,
        )
        tasks = json.loads(
            (
                REPO_ROOT / "data" / "map1_10roads" / "tasks.json"
            ).read_text(encoding="utf-8")
        )["tasks"]
        manager = object.__new__(WorldManager)
        manager.traffic_controller = controller

        for task_id, collision_capable in ((1, True), (3, False)):
            source_task = next(
                item for item in tasks if item["task_id"] == task_id
            )
            task = align_task_crosswalks_to_rendered_geometry(
                source_task,
                700.0,
                intersection_centers=[
                    [item.center.x, item.center.y]
                    for item in controller.intersections
                ],
            )
            route_crosswalks = controller.get_route_crosswalks(
                reconstruct_route_points_from_task_edges(task),
                task_edges=task["edges"],
            )
            self.assertEqual(
                1,
                len(route_crosswalks),
                f"task_id={task_id} must have one route crosswalk",
            )
            crosswalk = route_crosswalks[0]
            intersection = controller._get_route_crosswalk_intersection(
                crosswalk
            )
            incoming_lanes = [
                lane for lane in intersection.lane_mapping
                if any(
                    light.lane_id == lane.id
                    for light in intersection.traffic_lights
                )
            ]
            has_conflict_route = any(
                manager._incoming_lane_has_crosswalk_route(
                    lane,
                    intersection,
                    crosswalk,
                )
                for lane in incoming_lanes
            )
            self.assertEqual(
                collision_capable,
                bool(incoming_lanes),
                f"task_id={task_id} signalization must match its real intersection",
            )
            self.assertEqual(
                collision_capable,
                has_conflict_route,
                f"task_id={task_id} collision capability must match geometry",
            )

    def test_same_side_connector_is_not_a_route_crosswalk(self):
        controller = RTTrafficController(
            Config(str(REPO_ROOT / "config.yaml")),
            str(REPO_ROOT / "data" / "map1_10roads" / "roads.json"),
            seed=1,
            num_vehicles=3,
            num_pedestrians=0,
        )
        tasks = json.loads(
            (
                REPO_ROOT / "data" / "map1_10roads" / "tasks.json"
            ).read_text(encoding="utf-8")
        )["tasks"]
        source_task = next(item for item in tasks if item["task_id"] == 2)
        task = align_task_crosswalks_to_rendered_geometry(
            source_task,
            700.0,
            intersection_centers=[
                [item.center.x, item.center.y]
                for item in controller.intersections
            ],
        )
        route_crosswalks = controller.get_route_crosswalks(
            task["route_info"]["shortest_path"],
            task_edges=task["edges"],
        )
        self.assertEqual([], route_crosswalks)

        manager = object.__new__(WorldManager)
        manager.traffic_controller = controller
        manager.signal_conflict_route_vehicle_ids = []
        manager.signal_conflict_route_geometry_available = False
        manager._signal_vehicle_routes = {}
        manager.signal_vehicle_crossing_status = {}
        manager._count_route_blockers = lambda *_: 0
        manager.logger = SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        )

        self.assertFalse(manager.signal_conflict_route_vehicle_ids)
        self.assertFalse(manager.signal_conflict_route_geometry_available)
        source_task = next(
            item for item in tasks if item["task_id"] == 3
        )
        task = align_task_crosswalks_to_rendered_geometry(
            source_task,
            700.0,
            intersection_centers=[
                [item.center.x, item.center.y]
                for item in controller.intersections
            ],
        )
        endpoint_crosswalks = controller.get_route_crosswalks(
            task["route_info"]["shortest_path"],
            task_edges=task["edges"],
        )
        self.assertEqual(1, len(endpoint_crosswalks))
        self.assertEqual(18, endpoint_crosswalks[0].id)
        manager.red_light_conflict_launch_distance_max_cm = 900.0
        manager.red_light_conflict_nominal_speed_cm_s = 450.0
        manager._stage_signal_traffic_at_crosswalk(endpoint_crosswalks[0])
        self.assertFalse(manager.signal_conflict_route_vehicle_ids)
        self.assertTrue(manager.signal_conflict_route_geometry_available)
        self.assertEqual(
            'authored_lane_endpoint_extension',
            manager.signal_conflict_route_source,
        )
        self.assertEqual(
            900.0,
            manager.signal_conflict_route_max_launch_distance_cm_verified,
        )

    def test_snapshot_and_prompt_use_unanimous_ue_dont_walk(self):
        agent = make_agent(
            position=Vector(-7797.6773, -700),
            destination=Vector(-9300, -700),
            states={7: False, 8: False},
        )

        snapshot = agent._get_traffic_light_snapshot()
        context = agent._format_traffic_light_context(snapshot)

        self.assertTrue(snapshot["relevant"])
        self.assertEqual("DON'T WALK", snapshot["pedestrian_state"])
        self.assertEqual("ue_blueprint:GetState", snapshot["source"])
        self.assertIn("TRAFFIC SYSTEM OBSERVATION", context)
        self.assertIn("pedestrian=DONT_WALK", context)
        self.assertIn("Painted crosswalk centerline", context)
        self.assertIn("[-9300.0, -700.0] -> [-9300.0, 700.0]", context)
        self.assertIn("TURN toward it before any MOVE", context)
        self.assertIn("do not take a diagonal shortcut", context)
        self.assertNotIn("Do not enter", context)

    def test_unsignalized_route_crosswalk_is_not_a_red_light_violation(self):
        crosswalk = make_crosswalk()
        unrelated_signal = make_signal(
            99,
            -5000,
            -5000,
            crosswalk_id=7,
        )
        agent = RTAgent(
            position=Vector(-9300, -700),
            direction=Vector(0, 1),
            destination=Vector(-9300, 700),
            shortest_path=[Vector(-9300, 700)],
            communicator=FakeCommunicator({99: False}),
            llm=None,
            route_crosswalks=[crosswalk],
            crosswalk_signal_groups={crosswalk.id: []},
            traffic_signals=[unrelated_signal],
            traffic_policy=TRAFFIC_POLICY_SAFETY_ASSISTED,
        )

        approach = agent._get_traffic_light_snapshot()
        self.assertTrue(approach["relevant"])
        self.assertFalse(approach["traffic_controlled"])
        self.assertEqual("UNCONTROLLED", approach["pedestrian_state"])
        self.assertEqual("CROSS", approach["route_crossing_permission"])
        self.assertFalse(agent._should_gate_crosswalk_entry(Vector(-9300, -300)))

        agent.position = Vector(-9300, -300)
        crossing = agent._get_traffic_light_snapshot()
        self.assertEqual("CLEAR_ONLY", crossing["route_crossing_permission"])
        self.assertEqual(0, agent.red_light_violations_count)
        self.assertEqual([], agent.red_light_violation_events)
        self.assertIn("uncontrolled", agent._format_traffic_light_context(crossing))

    def test_route_crosswalk_signal_is_primary_even_when_another_head_is_closer(self):
        crosswalk = make_crosswalk()
        signals = [
            make_signal(7, -9000, -700, crosswalk_id=7),
            make_signal(8, -9300, 650, crosswalk_id=crosswalk.id),
        ]
        agent = RTAgent(
            position=Vector(-8900, -700),
            direction=Vector(-1, 0),
            destination=Vector(-9300, -700),
            shortest_path=[Vector(-9300, -700)],
            communicator=FakeCommunicator({7: False, 8: False}),
            llm=None,
            route_crosswalks=[crosswalk],
            crosswalk_signal_groups={crosswalk.id: signals},
        )

        snapshot = agent._get_traffic_light_snapshot()

        self.assertEqual(8, snapshot["primary_signal_id"])

    def test_native_intersection_snapshot_drives_prompt_phase_and_countdown(self):
        crosswalk = make_crosswalk()
        signals = [
            make_signal(7, -9350, -650),
            make_signal(8, -9250, 650),
        ]
        agent = RTAgent(
            position=Vector(-9300, -700),
            direction=Vector(0, 1),
            destination=Vector(-9300, 700),
            shortest_path=[Vector(-9300, 700)],
            communicator=FakeCommunicator({7: False, 8: False}),
            llm=None,
            route_crosswalks=[crosswalk],
            crosswalk_signal_groups={crosswalk.id: signals},
            crosswalk_intersection_names={
                crosswalk.id: "RT_Intersection_12"
            },
        )

        snapshot = agent._get_traffic_light_snapshot()
        context = agent._format_traffic_light_context(snapshot)

        self.assertEqual("PEDESTRIAN_CLEARANCE", snapshot["observed_phase"])
        self.assertEqual("FLASHING_DONT_WALK", snapshot["pedestrian_state"])
        self.assertEqual(3.5, snapshot["phase_remaining_time_s"])
        self.assertEqual("ue_controller:GetIntersectionState", snapshot["source"])
        self.assertIn("Observed phase: PEDESTRIAN_CLEARANCE", context)
        self.assertIn("3.5 s", context)
        self.assertIn("pedestrian=FLASHING_DONT_WALK", context)
        self.assertEqual("STOP", snapshot["route_crossing_permission"])

    def test_starting_during_flashing_clearance_is_a_violation(self):
        crosswalk = make_crosswalk()
        signals = [
            make_signal(7, -9350, -650),
            make_signal(8, -9250, 650),
        ]
        agent = RTAgent(
            position=Vector(-9300, -300),
            direction=Vector(0, 1),
            destination=Vector(-9300, 700),
            shortest_path=[Vector(-9300, 700)],
            communicator=FakeCommunicator({7: False, 8: False}),
            llm=None,
            route_crosswalks=[crosswalk],
            crosswalk_signal_groups={crosswalk.id: signals},
            crosswalk_intersection_names={crosswalk.id: "RT_Intersection_12"},
        )

        snapshot = agent._get_traffic_light_snapshot()

        self.assertTrue(snapshot["agent_on_crosswalk"])
        self.assertEqual(
            "CLEAR_ONLY",
            snapshot["route_crossing_permission"],
        )
        self.assertFalse(snapshot["crossing_admitted_on_walk"])
        self.assertEqual(1, agent.red_light_violations_count)
        self.assertEqual(
            "entered_without_walk",
            agent.red_light_violation_events[0]["violation_reason"],
        )
        self.assertFalse(agent._should_gate_crosswalk_entry(Vector(-9300, 0)))

    def test_legacy_clearance_tail_updates_render_and_prompt_semantics(self):
        stopped = []
        communicator = FakeCommunicator({7: True, 8: True})
        communicator.set_traffic_signal_pedestrian_stop = stopped.append
        original_get_state = communicator.get_traffic_signal_state
        communicator.get_traffic_signal_state = lambda signal_id: {
            **original_get_state(signal_id),
            "remaining_time_s": 3.5,
        }
        crosswalk = make_crosswalk()
        signals = [
            make_signal(7, -9350, -650),
            make_signal(8, -9250, 650),
        ]
        agent = RTAgent(
            position=Vector(-9300, -700),
            direction=Vector(0, 1),
            destination=Vector(-9300, 700),
            shortest_path=[Vector(-9300, 700)],
            communicator=communicator,
            llm=None,
            route_crosswalks=[crosswalk],
            crosswalk_signal_groups={crosswalk.id: signals},
            traffic_phase_timing=TrafficPhaseTiming(
                pedestrian_walk_s=7,
                pedestrian_clearance_s=14,
            ),
        )

        snapshot = agent._get_traffic_light_snapshot()

        self.assertEqual("PEDESTRIAN_CLEARANCE", snapshot["observed_phase"])
        self.assertEqual("FLASHING_DONT_WALK", snapshot["pedestrian_state"])
        self.assertEqual("STOP", snapshot["route_crossing_permission"])
        self.assertEqual([7, 8], stopped)

    def test_legacy_walk_countdown_excludes_clearance_tail(self):
        communicator = FakeCommunicator({7: True, 8: True})
        original_get_state = communicator.get_traffic_signal_state
        communicator.get_traffic_signal_state = lambda signal_id: {
            **original_get_state(signal_id),
            "remaining_time_s": 44.0,
        }
        crosswalk = make_crosswalk()
        signals = [
            make_signal(7, -9350, -650),
            make_signal(8, -9250, 650),
        ]
        agent = RTAgent(
            position=Vector(-9300, -700),
            direction=Vector(0, 1),
            destination=Vector(-9300, 700),
            shortest_path=[Vector(-9300, 700)],
            communicator=communicator,
            llm=None,
            route_crosswalks=[crosswalk],
            crosswalk_signal_groups={crosswalk.id: signals},
            traffic_phase_timing=TrafficPhaseTiming(
                pedestrian_walk_s=90,
                pedestrian_clearance_s=14,
                vlm_latency_budget_s=15,
            ),
        )

        snapshot = agent._get_traffic_light_snapshot()
        context = agent._format_traffic_light_context(snapshot)

        self.assertEqual(44.0, snapshot["phase_remaining_time_s"])
        self.assertEqual(30.0, snapshot["walk_remaining_time_s"])
        self.assertFalse(snapshot["can_finish_before_signal_change"])
        self.assertIn("Legal WALK time remaining", context)
        self.assertIn("remaining WALK time is insufficient", context)
        self.assertIn("WALK must remain active until the far curb", context)

    def test_legacy_vehicle_phase_does_not_publish_wrong_cycle_countdown(self):
        communicator = FakeCommunicator({7: False, 8: False})
        communicator.get_traffic_signal_state = lambda signal_id: {
            "signal_id": signal_id,
            "vehicle_green": True,
            "pedestrian_walk": False,
            # The legacy head exposes its combined local-cycle timer here,
            # which is not the grouped controller's vehicle-green countdown.
            "remaining_time_s": 104.0,
        }
        crosswalk = make_crosswalk()
        signals = [
            make_signal(7, -9350, -650),
            make_signal(8, -9250, 650),
        ]
        agent = RTAgent(
            position=Vector(-9300, -700),
            direction=Vector(0, 1),
            destination=Vector(-9300, 700),
            shortest_path=[Vector(-9300, 700)],
            communicator=communicator,
            llm=None,
            route_crosswalks=[crosswalk],
            crosswalk_signal_groups={crosswalk.id: signals},
            traffic_phase_timing=TrafficPhaseTiming(),
        )

        snapshot = agent._get_traffic_light_snapshot()
        context = agent._format_traffic_light_context(snapshot)

        self.assertEqual("VEHICLE_GREEN", snapshot["observed_phase"])
        self.assertIsNone(snapshot["phase_remaining_time_s"])
        self.assertFalse(
            snapshot["capabilities"]["vehicle_phase_countdown_exposed"]
        )
        self.assertNotIn("SPaT time remaining", context)

    def test_crosswalk_context_deactivates_after_far_curb(self):
        crosswalk = make_crosswalk()
        signals = [
            make_signal(7, -9350, -650),
            make_signal(8, -9250, 650),
        ]
        next_sidewalk_subgoal = Vector(-8193.4948, 700)
        agent = RTAgent(
            position=Vector(-9300, 700),
            direction=Vector(1, 0),
            destination=next_sidewalk_subgoal,
            shortest_path=[next_sidewalk_subgoal],
            communicator=FakeCommunicator({7: True, 8: True}),
            llm=None,
            route_crosswalks=[crosswalk],
            crosswalk_signal_groups={crosswalk.id: signals},
        )
        agent._admitted_crosswalks.add(crosswalk.id)

        snapshot = agent._get_traffic_light_snapshot()

        self.assertFalse(snapshot["relevant"])
        self.assertEqual("NOT_RELEVANT", snapshot["route_crossing_permission"])

    def test_far_curb_releases_admission_before_later_sidewalk_slip(self):
        crosswalk = make_crosswalk()
        signals = [
            make_signal(7, -9350, -650),
            make_signal(8, -9250, 650),
        ]
        next_sidewalk_subgoal = Vector(-8193.4948, 700)
        agent = RTAgent(
            position=Vector(-9300, 850),
            direction=Vector(1, 0),
            destination=next_sidewalk_subgoal,
            shortest_path=[next_sidewalk_subgoal],
            communicator=FakeCommunicator({7: False, 8: False}),
            llm=None,
            route_crosswalks=[crosswalk],
            crosswalk_signal_groups={crosswalk.id: signals},
        )
        agent._admitted_crosswalks.add(crosswalk.id)
        agent._crosswalk_entry_directions[crosswalk.id] = 1

        snapshot = agent._get_traffic_light_snapshot()

        self.assertFalse(snapshot["relevant"])
        self.assertNotIn(crosswalk.id, agent._admitted_crosswalks)
        self.assertNotIn(crosswalk.id, agent._crosswalk_entry_directions)

        # A later off-center sidewalk position near the same crosswalk must
        # not resurrect the completed traversal under DON'T WALK.
        agent.position = Vector(-8612, 500)
        later_snapshot = agent._get_traffic_light_snapshot()
        self.assertFalse(later_snapshot["relevant"])
        self.assertEqual(0, agent.red_light_violations_count)

    def test_legacy_occupancy_keeps_vehicles_red_without_extending_walk(self):
        calls = []
        communicator = SimpleNamespace(
            set_intersection_pedestrian_occupancy=(
                lambda *_: (_ for _ in ()).throw(RuntimeError("legacy UE"))
            ),
            set_traffic_signal_vehicle_stop=(
                lambda signal_id: calls.append(("vehicle_stop", signal_id))
            ),
            set_traffic_signal_pedestrian_walk=(
                lambda signal_id: calls.append(("pedestrian_walk", signal_id))
            ),
            set_traffic_signal_pedestrian_stop=(
                lambda signal_id: calls.append(("pedestrian_stop", signal_id))
            ),
        )
        crosswalk = make_crosswalk()
        agent = object.__new__(RTAgent)
        agent.communicator = communicator
        # The released full-city runtime has no native intersection mapping;
        # its compatibility path must protect traffic without extending WALK.
        agent.crosswalk_intersection_names = {}
        agent.crosswalk_signal_groups = {
            crosswalk.id: [
                make_signal(7, -9350, -650),
                make_signal(8, -9250, 650),
            ]
        }
        agent._legacy_occupancy_crosswalks = set()
        agent.logger = SimpleNamespace(
            debug=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        )

        agent._sync_native_crosswalk_occupancy(crosswalk, True)

        self.assertEqual(
            [
                ("vehicle_stop", 7),
                ("vehicle_stop", 8),
            ],
            calls,
        )
        self.assertNotIn(crosswalk.id, agent._legacy_occupancy_crosswalks)

        agent._sync_native_crosswalk_occupancy(crosswalk, False)

        self.assertEqual([("vehicle_stop", 7), ("vehicle_stop", 8)], calls)
        self.assertNotIn(crosswalk.id, agent._legacy_occupancy_crosswalks)

    def test_snapshot_does_not_rearm_hold_from_agent_roadway_occupancy(self):
        communicator = FakeCommunicator({7: False, 8: False})
        communicator.set_traffic_signal_pedestrian_walk = (
            lambda signal_id: communicator.states.__setitem__(signal_id, True)
        )
        crosswalk = make_crosswalk()
        signals = [
            make_signal(7, -9350, -650),
            make_signal(8, -9250, 650),
        ]
        agent = RTAgent(
            position=Vector(-9300, 0),
            direction=Vector(0, 1),
            destination=Vector(-9300, 700),
            shortest_path=[Vector(-9300, 700)],
            communicator=communicator,
            llm=None,
            route_crosswalks=[crosswalk],
            crosswalk_signal_groups={crosswalk.id: signals},
            crosswalk_intersection_names={},
        )

        snapshot = agent._get_traffic_light_snapshot()

        self.assertTrue(snapshot["agent_on_crosswalk"])
        self.assertFalse(snapshot["pedestrian_occupied"])
        self.assertFalse(snapshot["legacy_occupancy_hold"])
        self.assertEqual("DON'T WALK", snapshot["pedestrian_state"])
        self.assertEqual("CLEAR_ONLY", snapshot["route_crossing_permission"])
        self.assertEqual(1, agent.red_light_violations_count)
        self.assertEqual(1, len(agent.red_light_violation_events))
        event = agent.red_light_violation_events[0]
        self.assertEqual("red-light-0001", event["event_id"])
        self.assertTrue(event["agent_on_crosswalk"])
        self.assertTrue(event["crossing_in_progress"])
        self.assertEqual(0.5, event["crosswalk_projection"])
        self.assertEqual(0.0, event["crosswalk_lateral_distance_cm"])
        self.assertEqual("STOP", event["entry_permission"])
        self.assertEqual("STOP", event["route_crossing_permission_at_entry"])
        self.assertEqual("CLEAR_ONLY", event["route_crossing_permission"])
        self.assertFalse(event["agent_entry_legal"])
        self.assertEqual(1, event["crosswalk_entry_direction"])
        self.assertFalse(
            any(state["vehicle_green"] for state in snapshot["signal_states"])
        )

    def test_route_crosswalk_maps_to_native_intersection_actor(self):
        crosswalk = make_crosswalk()
        fake_controller = SimpleNamespace(
            intersections=[SimpleNamespace(id=12, crosswalks=[crosswalk])]
        )

        names = RTTrafficController.get_crosswalk_intersection_names(
            fake_controller,
            [crosswalk],
        )

        self.assertEqual({42: "RT_Intersection_12"}, names)

    def test_task_edge_crosswalk_maps_to_nearest_signalized_intersection(self):
        crosswalk = SimpleNamespace(
            id="task_edge_1",
            start=Vector(700, -700),
            end=Vector(-700, -700),
        )
        near_signal = make_signal(7, 0, -700)
        far_signal = make_signal(8, 10000, -700)
        fake_controller = SimpleNamespace(
            intersections=[
                SimpleNamespace(
                    id=0,
                    center=Vector(0, 0),
                    crosswalks=[],
                    traffic_lights=[near_signal],
                    pedestrian_lights=[],
                ),
                SimpleNamespace(
                    id=1,
                    center=Vector(10000, 0),
                    crosswalks=[],
                    traffic_lights=[far_signal],
                    pedestrian_lights=[],
                ),
            ]
        )

        names = RTTrafficController.get_crosswalk_intersection_names(
            fake_controller,
            [crosswalk],
        )
        groups = RTTrafficController.get_crosswalk_signal_groups(
            fake_controller,
            [crosswalk],
        )

        self.assertEqual(
            {"task_edge_1": "RT_Intersection_0"},
            names,
        )
        self.assertEqual([near_signal], groups["task_edge_1"])

    def test_inconsistent_blueprint_states_fail_safe(self):
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: False},
        )

        snapshot = agent._get_traffic_light_snapshot()

        self.assertFalse(snapshot["consistent"])
        self.assertEqual("CONFLICT", snapshot["pedestrian_state"])

    def test_dont_walk_gate_replaces_crosswalk_entry_with_wait(self):
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: False, 8: False},
        )
        waited = []
        agent.wait = lambda duration: waited.append(duration) or duration

        success, record = agent.take_action(
            RTActionSpace(action_type=MOVE_TO, action_param="1"),
            [Vector(-9300, -500)],
        )

        self.assertTrue(success)
        self.assertIn("Traffic-light gate", record)
        self.assertEqual([1], waited)
        self.assertEqual(1, agent.traffic_light_gate_count)
        self.assertEqual(1, agent.wait_count)
        self.assertEqual(0, agent.move_count)
        self.assertEqual(0, agent.red_light_violations_count)
        self.assertEqual("WAIT", agent.last_action_type)
        self.assertEqual("1", agent.last_action_param)
        self.assertEqual(
            "DON'T WALK",
            agent.last_execution_traffic_light_snapshot["pedestrian_state"],
        )
        self.assertEqual(
            "pedestrian_signal_not_walk",
            agent.last_action_override["reason"],
        )

    def test_visual_only_is_default_and_executes_dont_walk_move_unmodified(self):
        crosswalk = make_crosswalk()
        signals = [
            make_signal(7, -9350, -650),
            make_signal(8, -9250, 650),
        ]
        agent = RTAgent(
            position=Vector(-9300, -700),
            direction=Vector(0, 1),
            destination=Vector(-9300, 700),
            shortest_path=[Vector(-9300, 700)],
            communicator=FakeCommunicator({7: False, 8: False}),
            llm=None,
            route_crosswalks=[crosswalk],
            crosswalk_signal_groups={crosswalk.id: signals},
        )
        requested = Vector(-9700, -300)
        moved = []
        agent.move_to = lambda waypoint: moved.append(waypoint) or 1

        success, _ = agent.take_action(
            RTActionSpace(action_type=MOVE_TO, action_param="1"),
            [requested],
        )

        self.assertTrue(success)
        self.assertEqual(TRAFFIC_POLICY_VISUAL_ONLY, agent.traffic_policy)
        self.assertEqual([requested], moved)
        self.assertEqual(0, agent.traffic_light_gate_count)
        self.assertIsNone(agent.last_action_override)
        self.assertEqual(
            "DON'T WALK",
            agent.last_execution_traffic_light_snapshot["pedestrian_state"],
        )
        self.assertNotIn(crosswalk.id, agent._admitted_crosswalks)

    def test_visual_only_walk_entry_stays_legal_if_clearance_starts_during_inference(self):
        events = []
        crosswalk = make_crosswalk()
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
            violation_callback=lambda event, crossed: events.append(
                (event, crossed)
            ),
        )
        agent.traffic_policy = TRAFFIC_POLICY_VISUAL_ONLY
        requested = Vector(-9300, -300)

        def move_to(waypoint):
            agent.position = waypoint
            return 1

        agent.move_to = move_to

        success, _ = agent.take_action(
            RTActionSpace(action_type=MOVE_TO, action_param="1"),
            [requested],
        )

        self.assertTrue(success)
        self.assertEqual("WALK", agent.last_execution_traffic_light_snapshot[
            "pedestrian_state"
        ])
        self.assertIn(crosswalk.id, agent._admitted_crosswalks)
        self.assertEqual(0, agent.traffic_light_gate_count)

        # Realtime thinking advances the scene into clearance before the next
        # observation. The legal WALK admission remains valid for clearance.
        agent.crosswalk_intersection_names[crosswalk.id] = (
            "RT_Intersection_12"
        )
        agent.communicator.states = {7: False, 8: False}
        clearance = agent._get_traffic_light_snapshot()

        self.assertEqual("CLEAR_ONLY", clearance["route_crossing_permission"])
        self.assertTrue(clearance["crossing_admitted_on_walk"])
        self.assertEqual(0, agent.red_light_violations_count)
        self.assertEqual([], events)

    def test_visual_only_prompt_has_no_symbolic_traffic_section(self):
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: False, 8: False},
        )
        agent.traffic_policy = TRAFFIC_POLICY_VISUAL_ONLY
        snapshot_reader = MagicMock(side_effect=AssertionError(
            "visual-only prompt must not query symbolic traffic state"
        ))
        agent._get_traffic_light_snapshot = snapshot_reader

        context = agent.format_traffic_light_context([])
        prompt = USER_PROMPT.format(
            step_num=0,
            current_position=agent.position,
            speed=200,
            direction=agent.direction,
            subgoal=agent.current_destination,
            relative_distance=1400.0,
            relative_angle=0.0,
            required_time=1000,
            time_spent=0.0,
            timing_context="realtime",
            route_context=agent._format_active_route_context(),
            traffic_light_section="",
            history="",
            image_description="current image",
        )

        self.assertEqual("", context)
        self.assertNotIn("Traffic-light context", prompt)
        self.assertNotIn("Permission:", prompt)
        self.assertNotIn("Current active edge", prompt)
        self.assertNotIn("This is a signalized crosswalk", prompt)
        self.assertNotIn("This is an uncontrolled crosswalk", prompt)
        snapshot_reader.assert_not_called()

    def test_walk_allows_crosswalk_entry(self):
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
        )
        moved = []
        agent.move_to = lambda waypoint: moved.append(waypoint) or 1

        success, record = agent.take_action(
            RTActionSpace(action_type=MOVE_TO, action_param="1"),
            [Vector(-9300, -500)],
        )

        self.assertTrue(success)
        self.assertIn("Move to", record)
        self.assertEqual(1, len(moved))
        self.assertEqual(0, agent.traffic_light_gate_count)
        self.assertEqual(1, agent.move_count)
        self.assertEqual(0, agent.red_light_violations_count)
        self.assertEqual("MOVE", agent.last_action_type)

    def test_ue_walk_admission_is_not_overridden_by_legacy_local_schedule(self):
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
        )
        moved = []
        agent.move_to = lambda waypoint: moved.append(waypoint) or 1
        legacy_reads = []
        agent._traffic_signal_snapshot = lambda *args, **kwargs: (
            legacy_reads.append((args, kwargs))
            or (
                (
                    TrafficSignalState.VEHICLE_RED,
                    TrafficSignalState.PEDESTRIAN_RED,
                ),
                10.0,
                "local_pedestrian_schedule",
            )
        )

        success, _ = agent.take_action(
            RTActionSpace(action_type=MOVE_TO, action_param="1"),
            [Vector(-9300, -500)],
        )

        self.assertTrue(success)
        self.assertEqual(1, len(moved))
        self.assertEqual([], legacy_reads)
        self.assertEqual(0, agent.red_light_violations_count)

    def test_visual_only_legacy_violation_dispatches_consequence_callback(self):
        events = []
        agent = make_agent(
            position=Vector(-9300, -800),
            destination=Vector(-9300, 700),
            states={7: False, 8: False},
            violation_callback=lambda event, crosswalk: events.append(
                (event, crosswalk)
            ),
        )
        agent.traffic_policy = TRAFFIC_POLICY_VISUAL_ONLY
        agent.route_crosswalks = []
        agent.traffic_signals = list(
            next(iter(agent.crosswalk_signal_groups.values()))
        )
        for signal in agent.traffic_signals:
            signal.state = (
                TrafficSignalState.VEHICLE_RED,
                TrafficSignalState.PEDESTRIAN_RED,
            )
        agent.task_edges = [
            {
                "type": "crosswalk",
                "node1": [-9300, -700],
                "node2": [-9300, 700],
            }
        ]
        agent._traffic_signal_snapshot = lambda *args, **kwargs: (
            (
                TrafficSignalState.VEHICLE_RED,
                TrafficSignalState.PEDESTRIAN_RED,
            ),
            10.0,
            "local_pedestrian_schedule",
        )

        violated = agent._check_red_light_violation(Vector(-9300, -500))

        self.assertTrue(violated)
        self.assertEqual(1, agent.red_light_violations_count)
        self.assertEqual(1, len(agent.red_light_violation_events))
        self.assertEqual(1, len(events))
        self.assertEqual("red-light-0001", events[0][0]["event_id"])
        self.assertEqual(
            agent.red_light_violation_events[0]["event_id"],
            events[0][0]["event_id"],
        )
        self.assertEqual(7, events[0][0]["signal_id"])
        self.assertEqual(Vector(-9300, -700), events[0][1].start)
        self.assertEqual(Vector(-9300, 700), events[0][1].end)

    def test_visual_only_prefers_native_route_scorer_over_legacy_fallback(self):
        events = []
        agent = make_agent(
            position=Vector(-9300, -800),
            destination=Vector(-9300, 700),
            states={7: False, 8: False},
            violation_callback=lambda *args: events.append(args),
        )
        agent.traffic_policy = TRAFFIC_POLICY_VISUAL_ONLY
        agent.traffic_signals = list(
            next(iter(agent.crosswalk_signal_groups.values()))
        )
        agent.task_edges = [
            {
                "type": "crosswalk",
                "node1": [-9300, -700],
                "node2": [-9300, 700],
            }
        ]
        synchronized_reads = []
        agent._get_traffic_light_snapshot = lambda: (
            synchronized_reads.append(True) or {"relevant": True}
        )
        agent._traffic_signal_snapshot = lambda *args, **kwargs: self.fail(
            "legacy scorer must not run for a native route crosswalk"
        )

        violated = agent._check_red_light_violation(Vector(-9300, -500))

        self.assertFalse(violated)
        self.assertEqual([True], synchronized_reads)
        self.assertEqual([], events)
        self.assertEqual(0, agent.red_light_violations_count)

    def test_route_relevant_near_curb_approach_ignores_legacy_geometry_id(self):
        agent = make_agent(
            position=Vector(-8900, -700),
            destination=Vector(-9300, -700),
            states={7: False, 8: False},
        )
        # dev-main task edges synthesize a coordinate-string ID, whereas the
        # synchronized route crosswalk uses the native numeric ID (42 here).
        # They describe the same geometry and must not be evaluated twice.
        agent.task_edges = [
            {
                "type": "crosswalk",
                "node1": [-9300, -700],
                "node2": [-9300, 700],
            }
        ]
        moved = []
        agent.move_to = lambda waypoint: moved.append(waypoint) or 1
        legacy_reads = []
        agent._traffic_signal_snapshot = lambda *args, **kwargs: (
            legacy_reads.append((args, kwargs))
            or (
                (
                    TrafficSignalState.VEHICLE_RED,
                    TrafficSignalState.PEDESTRIAN_RED,
                ),
                10.0,
                "local_pedestrian_schedule",
            )
        )

        success, _ = agent.take_action(
            RTActionSpace(action_type=MOVE_TO, action_param="1"),
            [Vector(-9300, -300)],
        )

        self.assertTrue(success)
        self.assertEqual([Vector(-9300, -700)], moved)
        self.assertEqual(
            "APPROACH_ONLY",
            agent.last_execution_traffic_light_snapshot[
                "route_crossing_permission"
            ],
        )
        self.assertEqual([], legacy_reads)
        self.assertEqual(0, agent.red_light_violations_count)

    def test_near_curb_subgoal_is_approach_only_and_cannot_be_skipped(self):
        agent = make_agent(
            position=Vector(-8935, -700),
            destination=Vector(-9300, -700),
            states={7: True, 8: True},
        )
        snapshot = agent._get_traffic_light_snapshot()
        moved = []
        agent.move_to = lambda waypoint: moved.append(waypoint) or 1

        success, _ = agent.take_action(
            RTActionSpace(action_type=MOVE_TO, action_param="1"),
            [Vector(-9300, -300)],
        )

        self.assertTrue(success)
        self.assertEqual("NEAR_CURB", snapshot["route_subgoal_role"])
        self.assertEqual("APPROACH_ONLY", snapshot["route_crossing_permission"])
        self.assertIn(
            "Permission: APPROACH_ONLY",
            agent._format_traffic_light_context(snapshot),
        )
        self.assertEqual(Vector(-9300, -700), moved[0])
        self.assertEqual(
            "near_curb_approach_clamp",
            agent.last_action_override["reason"],
        )
        self.assertNotIn(42, agent._admitted_crosswalks)

    def test_walk_projects_diagonal_move_onto_crosswalk_centerline(self):
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
        )
        moved = []
        agent.move_to = lambda waypoint: moved.append(waypoint) or 1

        success, _ = agent.take_action(
            RTActionSpace(action_type=MOVE_TO, action_param="1"),
            [Vector(-9700, -300)],
        )

        self.assertTrue(success)
        self.assertEqual(-9300, moved[0].x)
        self.assertEqual(-300, moved[0].y)
        self.assertEqual(
            "crosswalk_centerline_tracking",
            agent.last_action_override["reason"],
        )

    def test_clear_only_wait_is_replaced_with_forward_clearance(self):
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
        )
        agent.move_to = lambda waypoint: 1
        agent.take_action(
            RTActionSpace(action_type=MOVE_TO, action_param="1"),
            [Vector(-9300, -500)],
        )
        agent.position = Vector(-9300, -300)
        moved = []
        agent.move_to = lambda waypoint: moved.append(waypoint) or 1

        success, record = agent.take_action(
            RTActionSpace(action_type=WAIT, action_param="3"),
            [],
        )

        self.assertTrue(success)
        self.assertEqual(1, len(moved))
        self.assertEqual(-9300, moved[0].x)
        self.assertEqual(100, moved[0].y)
        self.assertIn("instead of waiting in roadway", record)
        self.assertEqual("MOVE", agent.last_action_type)
        self.assertEqual("clear_only_no_wait", agent.last_action_override["reason"])

    def test_clear_only_prompt_never_instructs_waiting_in_roadway(self):
        agent = make_agent(
            position=Vector(-9300, -300),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
        )
        snapshot = agent._get_traffic_light_snapshot()
        snapshot["can_finish_before_signal_change"] = False

        context = agent._format_traffic_light_context(snapshot)

        self.assertEqual("CLEAR_ONLY", snapshot["route_crossing_permission"])
        self.assertIn("do NOT wait in the traffic lane", context)
        self.assertNotIn("wait for the next full WALK interval", context)

    def test_walk_authorizes_entry_even_when_countdown_budget_is_short(self):
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
        )
        agent.traffic_phase_timing = TrafficPhaseTiming(
            vlm_latency_budget_s=40,
            action_execution_budget_s=10,
            crossing_safety_buffer_s=5,
        )

        blocked = agent._should_gate_crosswalk_entry(Vector(-9300, -500))

        self.assertFalse(blocked)
        snapshot = agent.last_execution_traffic_light_snapshot
        self.assertFalse(snapshot["can_finish_before_signal_change"])
        self.assertEqual("CROSS", snapshot["route_crossing_permission"])
        self.assertIn(42, agent._admitted_crosswalks)
        self.assertNotIn(42, agent._legacy_occupancy_crosswalks)

    def test_admitted_crossing_clamps_overshoot_to_sidewalk_landing(self):
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
        )
        agent.move_to = lambda waypoint: 1
        agent.take_action(
            RTActionSpace(action_type=MOVE_TO, action_param="1"),
            [Vector(-9300, -500)],
        )
        agent.position = Vector(-9600, 0)
        agent.communicator.states = {7: False, 8: False}
        moved = []
        agent.move_to = lambda waypoint: moved.append(waypoint) or 1

        success, _ = agent.take_action(
            RTActionSpace(action_type=MOVE_TO, action_param="1"),
            [Vector(-9800, 1200)],
        )

        self.assertTrue(success)
        self.assertEqual(-9300, moved[0].x)
        self.assertEqual(850, moved[0].y)
        self.assertAlmostEqual(
            1.1071,
            agent.last_action_override["executed_progress"],
            places=4,
        )

    def test_lateral_drift_does_not_release_admitted_roadway_occupancy(self):
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
        )
        self.assertFalse(
            agent._should_gate_crosswalk_entry(Vector(-9300, -500))
        )

        # More than six metres off the centerline, but still between curb
        # planes.  This is a bad trajectory, not a cleared roadway.
        agent.position = Vector(-10000, 0)
        agent.communicator.states = {7: False, 8: False}
        snapshot = agent._get_traffic_light_snapshot()

        self.assertFalse(snapshot["agent_on_crosswalk"])
        self.assertTrue(snapshot["crossing_in_progress"])
        self.assertTrue(snapshot["agent_in_crossing_roadway"])
        self.assertEqual("CLEAR_ONLY", snapshot["route_crossing_permission"])
        self.assertFalse(snapshot["legacy_occupancy_hold"])
        self.assertIn(42, agent._admitted_crosswalks)
        # The severe lateral drift remains a route-adherence failure, but a
        # Outside the painted crossing, red-light attribution does not apply;
        # the separate illegal-crossing detector evaluates roadway occupancy.
        self.assertEqual(0, agent.red_light_violations_count)

    def test_crossing_does_not_extend_legacy_walk_during_long_action(self):
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
        )
        self.assertFalse(
            agent._should_gate_crosswalk_entry(Vector(-9300, -500))
        )
        advances = []
        agent.communicator.unrealcv = SimpleNamespace(
            advance_simulation_time=(
                lambda seconds, slomo: advances.append((seconds, slomo))
            )
        )
        agent.communicator.legacy_calls.clear()

        agent._advance_simulation_time(1.2)

        self.assertEqual([(1.2, 1)], advances)
        self.assertEqual(1.2, agent.sim_time_elapsed)
        self.assertEqual([], agent.communicator.legacy_calls)

    def test_signal_traffic_controller_ticks_during_long_agent_action(self):
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
        )
        advances = []
        controller_ticks = []
        agent.communicator.unrealcv = SimpleNamespace(
            advance_simulation_time=(
                lambda seconds, slomo: advances.append((seconds, slomo))
            )
        )
        agent.simulation_step_callback = controller_ticks.append
        agent.simulation_step_callback_interval_s = 0.5

        agent._advance_simulation_time(1.2)

        self.assertEqual(
            [(0.5, 1), (0.5, 1), (0.2, 1)],
            advances,
        )
        self.assertEqual([0.5, 0.5, 0.2], controller_ticks)
        self.assertEqual(1.2, agent.sim_time_elapsed)

    def test_vehicle_spawn_is_quiescent_until_transform_is_complete(self):
        events = []
        unrealcv = SimpleNamespace(
            spawn_bp_asset=lambda *args: events.append(('spawn', args)),
            set_physics=lambda *args: events.append(('physics', args)),
            set_collision=lambda *args: events.append(('collision', args)),
            set_location=lambda *args: events.append(('location', args)),
            set_orientation=lambda *args: events.append(('orientation', args)),
            set_scale=lambda *args: events.append(('scale', args)),
            set_movable=lambda *args: events.append(('movable', args)),
        )
        communicator = RTCommunicator(unrealcv)
        vehicle = SimpleNamespace(
            id=9,
            vehicle_reference='/Game/TestVehicle.TestVehicle_C',
            position=Vector(1200, -400),
            direction=Vector(1, 0),
        )

        with patch.dict(
            os.environ,
            {'SIMWORLD_ACTOR_MUTATION_SETTLE_SECONDS': '0.2'},
        ), patch(
            'simworld.communicator.communicator.time.sleep',
            side_effect=lambda seconds: events.append(('sleep', (seconds,))),
        ):
            communicator.spawn_vehicles([vehicle])

        self.assertEqual(
            [
                ('spawn', ('/Game/TestVehicle.TestVehicle_C', 'GEN_BP_Vehicle_9')),
                ('physics', ('GEN_BP_Vehicle_9', False)),
                ('collision', ('GEN_BP_Vehicle_9', False)),
                ('sleep', (0.2,)),
            ],
            events[:4],
        )
        self.assertEqual(
            [
                ('movable', ('GEN_BP_Vehicle_9', True)),
                ('sleep', (0.2,)),
                ('collision', ('GEN_BP_Vehicle_9', True)),
                ('sleep', (0.2,)),
                ('physics', ('GEN_BP_Vehicle_9', True)),
                ('sleep', (0.2,)),
            ],
            events[-6:],
        )

    def test_vehicle_poll_preserves_full_pose_for_safety_audit(self):
        information = {
            'VLocations': 'GEN_BP_Vehicle_9X=1200.5 Y=-400.25 Z=36.75',
            'VRotations': 'GEN_BP_Vehicle_9P=-2.5 Y=91.0 R=4.25',
            'PLocations': '',
            'PRotations': '',
            'LStates': '',
            'ALocations': '',
            'ARotations': '',
            'SLocations': '',
            'SRotations': '',
        }
        communicator = RTCommunicator(
            SimpleNamespace(
                get_informations=lambda _manager: json.dumps(information)
            )
        )

        result = communicator.get_position_and_direction(vehicle_ids=[9])

        position, yaw = result[('vehicle', 9)]
        self.assertAlmostEqual(1200.5, position.x)
        self.assertAlmostEqual(-400.25, position.y)
        self.assertAlmostEqual(91.0, yaw)
        self.assertEqual(
            {
                'name': 'GEN_BP_Vehicle_9',
                'location': [1200.5, -400.25, 36.75],
                'rotation': [-2.5, 91.0, 4.25],
            },
            communicator.last_vehicle_transforms[9],
        )

    def test_vehicle_flip_is_braked_and_marks_rollout_failed(self):
        vehicle = SimpleNamespace(
            id=9,
            position=Vector(1200.0, -400.0),
            stop_reason=None,
            state=None,
            set_attributes=MagicMock(),
        )
        update_vehicle = MagicMock()
        manager = object.__new__(WorldManager)
        manager.traffic_controller = SimpleNamespace(
            vehicles=[vehicle],
            pedestrians=[],
        )
        manager.signal_traffic_communicator = SimpleNamespace(
            last_vehicle_transforms={
                9: {
                    'name': 'RT_SIGNAL_VEHICLE_9',
                    'location': [1200.0, -400.0, 36.0],
                    'rotation': [1.0, 91.0, 174.0],
                }
            },
            update_vehicle=update_vehicle,
        )
        manager.signal_vehicle_initial_positions = {}
        manager.signal_vehicle_max_displacement_cm = {}
        manager.signal_vehicle_pose_sample_count = {}
        manager.signal_vehicle_max_tilt_deg = {}
        manager.signal_vehicle_z_range_cm = {}
        manager.signal_vehicle_instability_events = []
        manager._unstable_signal_vehicle_ids = set()
        manager.signal_vehicle_tilt_limit_deg = 30.0
        manager.signal_vehicle_stop_reasons = {}
        manager._signal_vehicle_routes = {}
        manager.signal_vehicle_crossing_status = {}
        manager.signal_traffic_min_distance_cm = {
            'vehicle_vehicle': float('inf'),
            'vehicle_pedestrian': float('inf'),
            'vehicle_agent': float('inf'),
            'pedestrian_agent': float('inf'),
        }
        manager._retired_signal_vehicle_ids = set()
        manager.agent = SimpleNamespace(
            position=Vector(0.0, 0.0),
            sim_time_elapsed=12.5,
            success=True,
            failed=False,
            failure_reason=None,
        )
        manager.logger = MagicMock()

        manager._update_signal_traffic_safety_metrics()

        vehicle.set_attributes.assert_called_once_with(0.0, 1.0, 0.0)
        update_vehicle.assert_called_once_with(9, 0.0, 1.0, 0.0)
        self.assertEqual(VehicleState.STOPPED, vehicle.state)
        self.assertEqual('physics_instability', vehicle.stop_reason)
        self.assertTrue(manager.agent.failed)
        self.assertFalse(manager.agent.success)
        self.assertEqual(
            'signal_vehicle_instability',
            manager.agent.failure_reason,
        )
        self.assertEqual(174.0, manager.signal_vehicle_max_tilt_deg[9])
        self.assertEqual(1, manager.signal_vehicle_pose_sample_count[9])
        self.assertEqual(1, len(manager.signal_vehicle_instability_events))

    def test_dont_walk_blocks_nearby_illegal_crossing_outside_painted_crosswalk(self):
        agent = make_agent(
            position=Vector(-8900, -700),
            destination=Vector(-9300, 700),
            states={7: False, 8: False},
        )

        blocked = agent._should_gate_crosswalk_entry(Vector(-8900, -500))

        self.assertTrue(blocked)

    def test_dont_walk_blocks_intersection_scale_illegal_crossing(self):
        agent = make_agent(
            position=Vector(-8450, -700),
            destination=Vector(-9300, 700),
            states={7: False, 8: False},
        )

        blocked = agent._should_gate_crosswalk_entry(Vector(-8450, -500))

        self.assertTrue(blocked)

    def test_walk_admission_remains_legal_during_flashing_clearance(self):
        events = []
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
            violation_callback=lambda event, crosswalk: events.append(
                (event, crosswalk)
            ),
        )

        self.assertFalse(
            agent._should_gate_crosswalk_entry(Vector(-9300, -500))
        )
        agent.position = Vector(-9300, -300)
        agent.crosswalk_intersection_names[42] = "RT_Intersection_12"
        agent.communicator.states = {7: False, 8: False}
        snapshot = agent._get_traffic_light_snapshot()

        self.assertTrue(snapshot["agent_on_crosswalk"])
        self.assertTrue(snapshot["crossing_admitted_on_walk"])
        self.assertEqual("CLEAR_ONLY", snapshot["route_crossing_permission"])
        self.assertEqual(0, agent.red_light_violations_count)
        self.assertEqual([], events)
        self.assertNotIn(42, agent._legacy_occupancy_crosswalks)

        # Re-reading the same crossing state does not duplicate the event.
        agent._get_traffic_light_snapshot()
        self.assertEqual(0, agent.red_light_violations_count)
        self.assertEqual([], events)

    def test_visual_only_walk_entry_remains_legal_during_flashing_clearance(self):
        events = []
        agent = make_agent(
            position=Vector(-9300, -300),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
            violation_callback=lambda event, crosswalk: events.append(
                (event, crosswalk)
            ),
        )
        agent.traffic_policy = TRAFFIC_POLICY_VISUAL_ONLY

        walk = agent._get_traffic_light_snapshot()
        agent.crosswalk_intersection_names[42] = "RT_Intersection_12"
        agent.communicator.states = {7: False, 8: False}
        clearance = agent._get_traffic_light_snapshot()

        self.assertTrue(walk["crossing_admitted_on_walk"])
        self.assertNotIn(42, agent._legacy_occupancy_crosswalks)
        self.assertEqual("CLEAR_ONLY", clearance["route_crossing_permission"])
        self.assertEqual(0, agent.red_light_violations_count)
        self.assertEqual([], events)

    def test_walk_admission_remains_legal_after_clearance_ends(self):
        events = []
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: True},
            violation_callback=lambda event, crosswalk: events.append(
                (event, crosswalk)
            ),
        )

        self.assertFalse(
            agent._should_gate_crosswalk_entry(Vector(-9300, -500))
        )
        agent.position = Vector(-9300, -300)
        crosswalk = next(
            crosswalk
            for crosswalk in agent.route_crosswalks
            if crosswalk.id == 42
        )
        snapshot = agent._finalize_route_crossing_permission(
            {
                "pedestrian_state": "DONT_WALK",
                "consistent": True,
                "safe": False,
            },
            crosswalk,
        )

        self.assertTrue(snapshot["crossing_in_progress"])
        self.assertEqual("CLEAR_ONLY", snapshot["route_crossing_permission"])
        self.assertEqual(0, agent.red_light_violations_count)
        self.assertEqual([], events)

    def test_observed_unauthorized_roadway_entry_is_counted_once(self):
        agent = make_agent(
            position=Vector(-9300, -300),
            destination=Vector(-9300, 700),
            states={7: False, 8: False},
        )

        first = agent._get_traffic_light_snapshot()
        second = agent._get_traffic_light_snapshot()

        self.assertTrue(first["agent_on_crosswalk"])
        self.assertEqual("CLEAR_ONLY", first["route_crossing_permission"])
        self.assertEqual("CLEAR_ONLY", second["route_crossing_permission"])
        self.assertIn(first["pedestrian_state"], {"DONT_WALK", "DON'T WALK"})
        self.assertEqual(1, agent.red_light_violations_count)
        self.assertEqual(
            "entered_without_walk",
            agent.red_light_violation_events[0]["violation_reason"],
        )

    def test_outside_crosswalk_roadway_occupancy_is_not_a_red_light_violation(self):
        agent = make_agent(
            position=Vector(-8900, -300),
            destination=Vector(-9300, 700),
            states={7: False, 8: False},
        )

        snapshot = agent._get_traffic_light_snapshot()

        self.assertFalse(snapshot["agent_on_crosswalk"])
        self.assertTrue(snapshot["crossing_in_progress"])
        self.assertEqual("CLEAR_ONLY", snapshot["route_crossing_permission"])
        self.assertEqual(0, agent.red_light_violations_count)
        self.assertEqual([], agent.red_light_violation_events)

    def test_conflicting_signal_states_fail_safe_at_execution(self):
        agent = make_agent(
            position=Vector(-9300, -700),
            destination=Vector(-9300, 700),
            states={7: True, 8: False},
        )

        blocked = agent._should_gate_crosswalk_entry(Vector(-9300, -500))

        self.assertTrue(blocked)
        self.assertEqual(
            "CONFLICT",
            agent.last_execution_traffic_light_snapshot["pedestrian_state"],
        )

    def test_rt_communicator_parses_packaged_blueprint_get_state(self):
        communicator = object.__new__(RTCommunicator)
        communicator.unrealcv = SimpleNamespace(
            tl_get_state=lambda _: (
                '{"green light":"false","ped green":"true","ped time":"9.25"}'
            )
        )

        state = communicator.get_traffic_signal_state(3)

        self.assertFalse(state["vehicle_green"])
        self.assertTrue(state["pedestrian_walk"])
        self.assertEqual(9.25, state["remaining_time_s"])

    def test_rt_communicator_configures_upgraded_intersection_phase_plan(self):
        calls = []
        communicator = object.__new__(RTCommunicator)
        communicator.unrealcv = SimpleNamespace(
            traffic_signal_configure_phase_plan=lambda *args: calls.append(args)
        )
        timing = TrafficPhaseTiming(
            vehicle_green_s=10,
            vehicle_yellow_s=2,
            all_red_s=1,
            pedestrian_walk_s=20,
            pedestrian_clearance_s=5,
        )

        communicator.configure_intersection_phase_plan(
            "RT_Intersection_12",
            timing,
        )

        self.assertEqual(
            [("RT_Intersection_12", 10, 2, 1, 20, 5)],
            calls,
        )

    def test_rt_communicator_parses_canonical_intersection_state(self):
        communicator = object.__new__(RTCommunicator)
        communicator.unrealcv = SimpleNamespace(
            traffic_signal_get_intersection_state=lambda _: json.dumps(
                {
                    "schema_version": 1,
                    "intersection_id": 12,
                    "phase": "vehicle_yellow",
                    "remaining_time_s": 1.5,
                    "safe": True,
                    "violations": [],
                    "signal_states": [
                        {
                            "signal_id": 7,
                            "vehicle_state": "YELLOW",
                            "pedestrian_state": "DON'T WALK",
                        }
                    ],
                    "pedestrian_occupied": True,
                    "clearance_extended": True,
                    "clearance_extension_elapsed_s": 4.25,
                }
            )
        )

        state = communicator.get_canonical_intersection_state(
            "RT_Intersection_12"
        )

        self.assertEqual("VEHICLE_YELLOW", state["observed_phase"])
        self.assertEqual(1.5, state["phase_remaining_time_s"])
        self.assertTrue(state["capabilities"]["vehicle_yellow_exposed"])
        self.assertEqual("ue_controller:GetIntersectionState", state["source"])
        self.assertTrue(state["pedestrian_occupied"])
        self.assertTrue(state["clearance_extended"])
        self.assertEqual(4.25, state["clearance_extension_elapsed_s"])

    def test_rt_communicator_updates_passive_pedestrian_occupancy(self):
        calls = []
        communicator = object.__new__(RTCommunicator)
        communicator.unrealcv = SimpleNamespace(
            traffic_signal_set_pedestrian_occupancy=(
                lambda *args: calls.append(args)
            )
        )

        communicator.set_intersection_pedestrian_occupancy(
            "RT_Intersection_12",
            True,
        )

        self.assertEqual([("RT_Intersection_12", True)], calls)

    def test_rt_communicator_maps_legacy_occupancy_hold_to_render_heads(self):
        calls = []
        communicator = object.__new__(RTCommunicator)
        communicator.unrealcv = SimpleNamespace(
            tl_set_pedestrian_walk=(
                lambda object_name: calls.append(("walk", object_name))
            ),
            tl_set_vehicle_red=(
                lambda object_name: calls.append(("vehicle_red", object_name))
            ),
        )

        communicator.set_traffic_signal_pedestrian_walk(8)
        communicator.set_traffic_signal_vehicle_stop(8)

        self.assertEqual(
            [
                ("walk", "RT_TRAFFIC_SIGNAL_8"),
                ("vehicle_red", "RT_TRAFFIC_SIGNAL_8"),
            ],
            calls,
        )

    def test_rt_communicator_decodes_unrealcv_string_outputs(self):
        communicator = object.__new__(RTCommunicator)
        communicator.unrealcv = SimpleNamespace(
            traffic_signal_get_intersection_state=lambda _: json.dumps(
                {
                    "schema_version": "1",
                    "intersection_id": "RT_Intersection_12",
                    "Phase": "PEDESTRIAN_WALK",
                    "remaining_time_s": "12.5",
                    "safe": "false",
                    "violations": '["forced_test_violation"]',
                    "signal_states": json.dumps(
                        [
                            {
                                "name": "RT_TRAFFIC_SIGNAL_7",
                                "vehicle_state": "RED",
                                "pedestrian_state": "WALK",
                            }
                        ]
                    ),
                }
            )
        )

        state = communicator.get_canonical_intersection_state(
            "RT_Intersection_12"
        )

        self.assertFalse(state["safe"])
        self.assertEqual(["forced_test_violation"], state["violations"])
        self.assertEqual("WALK", state["signal_states"][0]["pedestrian_state"])

    def test_model_prompt_uses_same_synchronized_ue_snapshot_as_execution_gate(self):
        agent = object.__new__(RTAgent)
        agent.traffic_policy = TRAFFIC_POLICY_SAFETY_ASSISTED
        agent.position = Vector(-10659.3, -876.9)
        agent._direction = Vector(1.0, 0.0)
        synchronized = {
            "relevant": True,
            "pedestrian_state": "DON'T WALK",
            "source": "ue_blueprint:GetState",
        }
        agent._get_traffic_light_snapshot = MagicMock(
            return_value=synchronized
        )
        agent._format_traffic_light_context = MagicMock(
            return_value="SYNCHRONIZED UE STATE: DON'T WALK"
        )
        agent._refresh_traffic_signal_states = MagicMock(
            side_effect=AssertionError(
                "route-relevant prompt must not use the local schedule"
            )
        )

        context = RTAgent.format_traffic_light_context(agent, waypoints=[])

        self.assertIn("SYNCHRONIZED UE STATE: DON'T WALK", context)
        agent._get_traffic_light_snapshot.assert_called_once_with()
        agent._format_traffic_light_context.assert_called_once_with(
            synchronized
        )
        agent._refresh_traffic_signal_states.assert_not_called()

    def test_recording_saves_alignment_sidecar_and_human_qa_image(self):
        agent = make_agent(
            position=Vector(-7797.6773, -700),
            destination=Vector(-9300, -700),
            states={7: False, 8: False},
        )
        snapshot = agent._get_traffic_light_snapshot()
        with tempfile.TemporaryDirectory() as record_dir:
            agent.record_dir = record_dir
            agent._record_step_data(
                step_num=0,
                observation={
                    "ego_view": Image.new("RGB", (64, 48), "white"),
                    "waypoints": [],
                    "traffic_light_snapshot": snapshot,
                },
                prompt_data={
                    "system_prompt": "system",
                    "user_prompt": "user",
                    "full_response": "response",
                    "traffic_light_snapshot": snapshot,
                    "execution_traffic_light_snapshot": snapshot,
                    "observation_geometry": {
                        "camera_location_cm": [1.0, 2.0, 3.0],
                        "camera_rotation_deg": [4.0, 5.0, 6.0],
                    },
                },
                action=None,
                input_images=[Image.new("RGB", (64, 48), "white")],
            )

            step_dir = Path(record_dir) / "step_0000"
            sidecar = step_dir / "step_0000_traffic_light_snapshot.json"
            debug_image = step_dir / "step_0000_traffic_light_debug.png"
            self.assertTrue(sidecar.exists())
            self.assertTrue(debug_image.exists())
            recorded = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(
                "DON'T WALK",
                recorded["model_input"]["pedestrian_state"],
            )
            manifest = json.loads(
                (step_dir / "step_0000_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                "pre_post_camera_v1",
                manifest["artifact_schema_version"],
            )
            self.assertEqual([1.0, 2.0, 3.0], manifest["input_camera"]["location"])
            self.assertEqual([4.0, 5.0, 6.0], manifest["input_camera"]["rotation"])
            self.assertEqual(manifest["input_camera"], manifest["camera"])
            self.assertEqual(
                list(agent.camera_location),
                manifest["output_camera"]["location"],
            )
            self.assertEqual("policy_input", manifest["camera"]["scope"])
            self.assertEqual(
                "post_action_output",
                manifest["output_camera"]["scope"],
            )

    def test_recording_persists_temporal_vlm_frames_in_demo_timeline(self):
        agent = make_agent(
            position=Vector(-7797.6773, -700),
            destination=Vector(-9300, -700),
            states={7: False, 8: False},
        )
        agent.use_action_frames = True
        agent.recording_action_frames = []
        agent.action_frames = [
            Image.new("RGB", (64, 48), "red"),
            Image.new("RGB", (64, 48), "green"),
        ]
        snapshot = agent._get_traffic_light_snapshot()
        with tempfile.TemporaryDirectory() as record_dir:
            agent.record_dir = record_dir
            agent._record_step_data(
                step_num=0,
                observation={
                    "ego_view": Image.new("RGB", (64, 48), "white"),
                    "raw_first_person_view": Image.new(
                        "RGB", (64, 48), "white"
                    ),
                    "waypoints": [],
                    "traffic_light_snapshot": snapshot,
                },
                prompt_data={
                    "system_prompt": "system",
                    "user_prompt": "user",
                    "full_response": "response",
                    "traffic_light_snapshot": snapshot,
                },
                action=None,
                input_images=[Image.new("RGB", (64, 48), "white")],
            )

            step_dir = Path(record_dir) / "step_0000"
            manifest = json.loads(
                (step_dir / "step_0000_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [
                    "step_0000_action_frame_00.png",
                    "step_0000_action_frame_01.png",
                ],
                manifest["demo_action_images"],
            )
            for frame_name in manifest["demo_action_images"]:
                self.assertTrue((step_dir / frame_name).exists())


class RouteCorridorInvariantTests(unittest.TestCase):
    @staticmethod
    def _point_on_edge(point, edge, tolerance=1.0):
        point = Vector(*point)
        start = Vector(*edge["node1"])
        end = Vector(*edge["node2"])
        segment = end - start
        length_sq = segment.x ** 2 + segment.y ** 2
        if length_sq <= 1e-9:
            return point.distance(start) <= tolerance
        offset = point - start
        projection = (
            offset.x * segment.x + offset.y * segment.y
        ) / length_sq
        if projection < -1e-6 or projection > 1.0 + 1e-6:
            return False
        projection = max(0.0, min(1.0, projection))
        return point.distance(start + segment * projection) <= tolerance

    def test_every_task_in_every_map_reconstructs_on_ordered_edges(self):
        task_files = sorted((REPO_ROOT / "data").glob("map*/tasks.json"))
        self.assertGreaterEqual(len(task_files), 5)
        checked_tasks = 0

        for task_file in task_files:
            tasks = json.loads(task_file.read_text(encoding="utf-8"))["tasks"]
            controller = RTTrafficController(
                Config(str(REPO_ROOT / "config.yaml")),
                str(task_file.parent / "roads.json"),
                seed=1,
                num_vehicles=0,
                num_pedestrians=0,
            )
            roads_payload = json.loads(
                (task_file.parent / "roads.json").read_text(encoding="utf-8")
            )
            roads = roads_payload["roads"]
            centers = sorted({
                (
                    float(road[endpoint]["x"]) * 100.0,
                    float(road[endpoint]["y"]) * 100.0,
                )
                for road in roads
                for endpoint in ("start", "end")
            })
            for source_task in tasks:
                task = align_task_crosswalks_to_rendered_geometry(
                    source_task,
                    700.0,
                    intersection_centers=centers,
                    rendered_crosswalk_segments=(
                        rendered_native_crosswalk_segments(
                            roads_payload,
                            700.0,
                            700.0,
                        )
                    ),
                )
                self.assertEqual(
                    [],
                    task["render_geometry_calibration"][
                        "non_crossing_connectors"
                    ],
                    f"stale authored crosswalk: {task_file.parent.name} "
                    f"task {task['task_id']}",
                )
                validate_selected_task_crosswalk_preflight(
                    task,
                    signal_traffic_enabled=True,
                )
                route = reconstruct_route_points_from_task_edges(task)
                edges = task["edges"]
                self.assertEqual(
                    len(edges) + 1,
                    len(route),
                    f"{task_file.parent.name} task {task['task_id']}",
                )
                self.assertEqual(task["start_point"], route[0])
                self.assertEqual(task["end_point"], route[-1])
                for index, edge in enumerate(edges):
                    self.assertTrue(
                        self._point_on_edge(route[index], edge),
                        f"route start outside edge {index}: "
                        f"{task_file.parent.name} task {task['task_id']}",
                    )
                    self.assertTrue(
                        self._point_on_edge(route[index + 1], edge),
                        f"route end outside edge {index}: "
                        f"{task_file.parent.name} task {task['task_id']}",
                    )
                reconstructed_crosswalks = sum(
                    1 for edge in edges if edge.get("type") == "crosswalk"
                )
                self.assertEqual(
                    int(task.get("crosswalk_hops") or 0),
                    reconstructed_crosswalks,
                )
                matched_crosswalks = controller.get_route_crosswalks(
                    route,
                    task_edges=edges,
                )
                self.assertEqual(
                    reconstructed_crosswalks,
                    len(matched_crosswalks),
                    f"unmatched crosswalk: {task_file.parent.name} "
                    f"task {task['task_id']}",
                )
                checked_tasks += 1

        self.assertEqual(36, checked_tasks)

    def test_task4_stale_shortest_path_is_rebuilt_through_crosswalk(self):
        task_file = REPO_ROOT / "data" / "map1_10roads" / "tasks.json"
        tasks = json.loads(task_file.read_text(encoding="utf-8"))["tasks"]
        task = tasks[3]

        route = reconstruct_route_points_from_task_edges(task)

        self.assertEqual([9300.0, -17485.5475], route[0])
        self.assertIn([700.0, -10700.0], route)
        self.assertIn([700.0, -9300.0], route)
        self.assertNotIn([9300.0, -9300.0], route[1:-1])

    def _route_agent(self, position, start, end, edge_type):
        agent = RTAgent.__new__(RTAgent)
        agent.position = position
        agent.current_destination = end
        agent.route_polyline = [start, end]
        agent.original_shortest_path = [end]
        agent.shortest_path = [end]
        agent.task_edges = [{
            "node1": [start.x, start.y],
            "node2": [end.x, end.y],
            "type": edge_type,
        }]
        agent.name = "agent"
        agent.communicator = MagicMock()
        agent.communicator.get_states.return_value = (0, 0, 0, 0.0, 0, 0)
        agent.traffic_roads = []
        agent.traffic_sidewalks = []
        agent.last_ue_collision_count = {
            "human": 0,
            "object": 0,
            "building": 0,
            "vehicle": 0,
            "intensity": 0.0,
            "touched_road": 0,
        }
        agent.collision_count = 0
        agent.collision_type_counts = {"human": 0, "object": 0, "building": 0}
        agent.route_crosswalks = []
        agent.illegal_crossing_violations_count = 0
        agent.illegal_crossing_events = []
        agent.illegal_crossing_callback = None
        agent.illegal_crossing_excursion_active = False
        agent.fall_count = agent.oil_count = agent.water_count = 0
        agent.decelerated = agent.slipping = False
        return agent

    def _road_detection_agent(self, outside_sidewalk_cm, ue_touched_road):
        start = Vector(-9300.0, 9300.0)
        end = Vector(-3430.4674, 9300.0)
        position = Vector(-9000.0, 9500.0 + outside_sidewalk_cm)
        agent = self._route_agent(position, start, end, "sidewalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "MOVE"
        agent.last_action_start_position = Vector(-9000.0, 9490.0)
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 12.5
        agent.step_num = 13
        agent.direction = 90.0
        agent.speed = 200.0
        agent.illegal_crossing_callback = MagicMock(
            return_value={
                "event_id": "illegal-crossing-0001",
                "disposition": "launched",
                "status": "launched",
            }
        )
        agent.traffic_sidewalks = [SimpleNamespace(start=start, end=end)]
        agent.traffic_roads = [SimpleNamespace(
            start=Vector(-10000.0, 10000.0),
            end=Vector(0.0, 10000.0),
        )]
        agent.communicator.get_states.return_value = (
            0,
            0,
            0,
            0.0,
            0,
            ue_touched_road,
        )
        return agent

    def _crosswalk_detection_agent(self, outside_crosswalk_cm):
        start = Vector(0.0, 0.0)
        end = Vector(0.0, 1000.0)
        position = Vector(210.0 + outside_crosswalk_cm, 200.0)
        agent = self._route_agent(position, start, end, "crosswalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "MOVE"
        agent.last_action_start_position = Vector(200.0, 200.0)
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 12.5
        agent.step_num = 13
        agent.direction = 90.0
        agent.speed = 200.0
        agent.illegal_crossing_callback = MagicMock(return_value={
            "event_id": "illegal-crossing-0001",
            "disposition": "launched",
            "status": "launched",
        })
        agent.route_crosswalks = [SimpleNamespace(
            id=42,
            start=start,
            end=end,
        )]
        agent.traffic_roads = [SimpleNamespace(
            start=Vector(0.0, 200.0),
            end=Vector(1000.0, 200.0),
        )]
        return agent

    def test_python_only_road_detection_never_launches_inside_old_hysteresis(self):
        agent = self._road_detection_agent(
            outside_sidewalk_cm=25.0,
            ue_touched_road=0,
        )

        _, collisions, _ = agent.get_feedback()

        self.assertEqual(1, collisions["geometric_touched_road"])
        self.assertEqual(0, collisions["ue_touched_road"])
        self.assertEqual(0, collisions["legal_authored_sidewalk"])
        self.assertEqual(0, collisions["unsafe_touched_road"])
        self.assertIsNone(collisions["illegal_crossing_detection_source"])
        self.assertEqual([], agent.illegal_crossing_events)
        agent.illegal_crossing_callback.assert_not_called()

    def test_python_only_road_detection_never_launches_outside_old_hysteresis(self):
        agent = self._road_detection_agent(
            outside_sidewalk_cm=26.0,
            ue_touched_road=0,
        )

        _, collisions, _ = agent.get_feedback()

        self.assertEqual(1, collisions["geometric_touched_road"])
        self.assertEqual(0, collisions["unsafe_touched_road"])
        self.assertEqual(0, collisions["ue_unsafe_touched_road"])
        self.assertIsNone(collisions["illegal_crossing_detection_source"])
        self.assertEqual([], agent.illegal_crossing_events)
        agent.illegal_crossing_callback.assert_not_called()

    def test_ue_touched_road_launches_outside_pedestrian_geometry(self):
        agent = self._road_detection_agent(
            outside_sidewalk_cm=10.0,
            ue_touched_road=1,
        )

        _, collisions, _ = agent.get_feedback()

        self.assertEqual(1, collisions["unsafe_touched_road"])
        self.assertEqual(1, collisions["ue_unsafe_touched_road"])
        self.assertEqual(
            "ue_touched_road",
            collisions["illegal_crossing_detection_source"],
        )
        self.assertEqual(
            "ue_touched_road",
            agent.illegal_crossing_events[0]["detection_source"],
        )
        agent.illegal_crossing_callback.assert_called_once()

    def test_ue_touched_road_does_not_launch_on_authored_sidewalk(self):
        agent = self._road_detection_agent(
            outside_sidewalk_cm=-10.0,
            ue_touched_road=1,
        )

        _, collisions, _ = agent.get_feedback()

        self.assertEqual(1, collisions["ue_touched_road"])
        self.assertEqual(1, collisions["legal_authored_sidewalk"])
        self.assertEqual(0, collisions["unsafe_touched_road"])
        self.assertEqual(0, collisions["ue_unsafe_touched_road"])
        self.assertIsNone(collisions["illegal_crossing_detection_source"])
        self.assertEqual([], agent.illegal_crossing_events)
        agent.illegal_crossing_callback.assert_not_called()

    def test_python_only_crosswalk_geometry_never_launches(self):
        inside_band = self._crosswalk_detection_agent(
            outside_crosswalk_cm=25.0,
        )
        outside_band = self._crosswalk_detection_agent(
            outside_crosswalk_cm=26.0,
        )

        _, inside_collisions, _ = inside_band.get_feedback()
        _, outside_collisions, _ = outside_band.get_feedback()

        self.assertEqual(0, inside_collisions["legal_marked_crosswalk"])
        self.assertEqual(0, inside_collisions["unsafe_touched_road"])
        inside_band.illegal_crossing_callback.assert_not_called()
        self.assertEqual(0, outside_collisions["unsafe_touched_road"])
        outside_band.illegal_crossing_callback.assert_not_called()

    def test_geometric_road_occupancy_covers_intersection_mouths(self):
        agent = RTAgent.__new__(RTAgent)
        agent.traffic_roads = [
            SimpleNamespace(
                start=Vector(-10000.0, 10000.0),
                end=Vector(0.0, 10000.0),
            ),
            SimpleNamespace(
                start=Vector(-10000.0, 0.0),
                end=Vector(-10000.0, 10000.0),
            ),
        ]
        self.assertFalse(agent._is_geometrically_on_road(Vector(-9300.0, 9300.0)))
        self.assertTrue(agent._is_geometrically_on_road(Vector(-9303.0, 9652.0)))
        self.assertTrue(agent._is_geometrically_on_road(Vector(-10039.0, 1562.0)))

    def test_authored_sidewalk_corridor_wins_over_overlapping_road_guard(self):
        agent = RTAgent.__new__(RTAgent)
        agent.traffic_roads = [
            SimpleNamespace(
                start=Vector(10000.0, 0.0),
                end=Vector(10000.0, 10000.0),
            )
        ]
        agent.traffic_sidewalks = [
            SimpleNamespace(
                start=Vector(9300.0, 700.0),
                end=Vector(9300.0, 9300.0),
            )
        ]

        self.assertTrue(
            agent._is_within_authored_sidewalk(Vector(9481.7, 1300.0))
        )
        self.assertFalse(
            agent._is_geometrically_on_road(Vector(9481.7, 1300.0))
        )
        self.assertTrue(
            agent._is_geometrically_on_road(Vector(9600.0, 1300.0))
        )

    def test_ue_road_bit_does_not_make_authored_sidewalk_illegal(self):
        start = Vector(9300.0, 700.0)
        end = Vector(9300.0, 5208.3981)
        position = Vector(9481.7, 1300.0)
        agent = self._route_agent(position, start, end, "sidewalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "WAIT"
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 101.98
        agent.step_num = 18
        agent.direction = 90.0
        agent.speed = 200.0
        agent.illegal_crossing_callback = MagicMock()
        agent.traffic_roads = [
            SimpleNamespace(
                start=Vector(10000.0, 0.0),
                end=Vector(10000.0, 10000.0),
            )
        ]
        agent.traffic_sidewalks = [
            SimpleNamespace(start=start, end=Vector(9300.0, 9300.0))
        ]
        agent.communicator.get_states.return_value = (0, 0, 0, 0.0, 0, 1)

        feedback, collisions, _ = agent.get_feedback()

        self.assertEqual("Action completed successfully.", feedback)
        self.assertEqual(1, collisions["touched_road"])
        self.assertEqual(0, collisions["geometric_touched_road"])
        self.assertEqual(1, collisions["legal_authored_sidewalk"])
        self.assertEqual(0, collisions["unsafe_touched_road"])
        self.assertEqual(0, agent.illegal_crossing_violations_count)
        agent.illegal_crossing_callback.assert_not_called()

    def test_legal_route_crosswalk_occupancy_is_not_unsafe_road_feedback(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        agent = self._route_agent(Vector(400.0, 0.0), start, end, "crosswalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "WAIT"
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 5.0
        agent.step_num = 4
        agent.direction = 0.0
        agent.speed = 200.0
        agent.illegal_crossing_events = []
        agent.illegal_crossing_excursion_active = False
        agent.illegal_crossing_callback = MagicMock()
        agent.route_crosswalks = [SimpleNamespace(start=start, end=end)]
        agent.traffic_roads = [SimpleNamespace(start=start, end=end)]

        feedback, collisions, _ = agent.get_feedback()

        self.assertEqual("Action completed successfully.", feedback)
        self.assertEqual(1, collisions["touched_road"])
        self.assertEqual(1, collisions["geometric_touched_road"])
        self.assertEqual(1, collisions["legal_route_crosswalk"])
        self.assertEqual(0, collisions["unsafe_touched_road"])
        self.assertEqual(0, agent.illegal_crossing_violations_count)
        agent.illegal_crossing_callback.assert_not_called()

    def test_any_authored_crosswalk_is_legal_independent_of_route(self):
        route_start = Vector(-1000.0, -1000.0)
        route_end = Vector(-500.0, -1000.0)
        position = Vector(0.0, 400.0)
        alternate = SimpleNamespace(
            id=19,
            start=Vector(0.0, 0.0),
            end=Vector(0.0, 1000.0),
        )
        agent = self._route_agent(
            position,
            route_start,
            route_end,
            "sidewalk",
        )
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "MOVE"
        agent.last_action_start_position = Vector(0.0, -100.0)
        agent.last_move_execution_mode = "ue_physical_off_route"
        agent.last_commanded_waypoint = position
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 5.0
        agent.step_num = 4
        agent.direction = 90.0
        agent.speed = 200.0
        agent.illegal_crossing_events = []
        agent.illegal_crossing_excursion_active = False
        agent.illegal_crossing_callback = MagicMock()
        agent.traffic_crosswalks = [alternate]
        agent.traffic_roads = [
            SimpleNamespace(
                start=Vector(-1000.0, 400.0),
                end=Vector(1000.0, 400.0),
            )
        ]
        agent.communicator.get_states.return_value = (0, 0, 0, 0.0, 0, 1)

        feedback, collisions, _ = agent.get_feedback()

        self.assertEqual("Action completed successfully.", feedback)
        self.assertNotIn("You are on the road", feedback)
        self.assertEqual(0, collisions["legal_route_crosswalk"])
        self.assertEqual(1, collisions["legal_marked_crosswalk"])
        self.assertEqual(1, collisions["legal_non_route_crosswalk"])
        self.assertEqual(0, collisions["unsafe_touched_road"])
        self.assertEqual(0, agent.illegal_crossing_violations_count)
        self.assertEqual([], agent.illegal_crossing_events)
        agent.illegal_crossing_callback.assert_not_called()

    def test_physical_non_route_crosswalk_uses_its_own_signal_snapshot(self):
        route_crosswalk = SimpleNamespace(
            id=18,
            start=Vector(700.0, -10700.0),
            end=Vector(700.0, -9300.0),
        )
        alternate = SimpleNamespace(
            id=19,
            start=Vector(9300.0, -9300.0),
            end=Vector(9300.0, -10700.0),
        )
        signal = make_signal(34, 10600.0, -9400.0, crosswalk_id=19)
        communicator = FakeCommunicator({34: False})
        agent = RTAgent(
            position=Vector(9321.0, -10312.4),
            direction=Vector(0.0, 1.0),
            destination=Vector(700.0, -9300.0),
            shortest_path=[Vector(700.0, -10700.0), Vector(700.0, -9300.0)],
            communicator=communicator,
            llm=None,
            route_crosswalks=[route_crosswalk],
            traffic_crosswalks=[route_crosswalk, alternate],
            crosswalk_signal_groups={18: [], 19: [signal]},
            traffic_signals=[signal],
            traffic_policy=TRAFFIC_POLICY_VISUAL_ONLY,
        )

        snapshot = agent._get_traffic_light_snapshot()

        self.assertEqual(19, snapshot["crosswalk_id"])
        self.assertTrue(snapshot["traffic_controlled"])
        self.assertEqual("DON'T WALK", snapshot["pedestrian_state"])
        self.assertEqual("CLEAR_ONLY", snapshot["route_crossing_permission"])
        self.assertEqual(1, agent.red_light_violations_count)
        self.assertEqual(19, agent.red_light_violation_events[0]["crosswalk_id"])
        self.assertEqual(-1, agent.red_light_violation_events[0]["crosswalk_entry_direction"])

    def test_route_crosswalk_wins_over_adjacent_crosswalk_path_overlap(self):
        route_crosswalk = SimpleNamespace(
            id=21,
            start=Vector(-29300.0, 700.0),
            end=Vector(-30700.0, 700.0),
        )
        adjacent_crosswalk = SimpleNamespace(
            id=12,
            start=Vector(-29300.0, -700.0),
            end=Vector(-29300.0, 700.0),
        )
        route_signal = make_signal(50, -29400.0, -600.0, crosswalk_id=21)
        adjacent_signal = make_signal(52, -30600.0, -600.0, crosswalk_id=12)
        communicator = FakeCommunicator({50: True, 52: False})
        position = Vector(-29300.0, 876.374)
        next_waypoint = Vector(-29500.0044, 529.9664)
        agent = RTAgent(
            position=position,
            direction=Vector(-0.5, -0.866),
            destination=route_crosswalk.end,
            shortest_path=[route_crosswalk.end],
            communicator=communicator,
            llm=None,
            route_crosswalks=[route_crosswalk],
            traffic_crosswalks=[adjacent_crosswalk, route_crosswalk],
            crosswalk_signal_groups={
                12: [adjacent_signal],
                21: [route_signal],
            },
            traffic_signals=[adjacent_signal, route_signal],
            traffic_policy=TRAFFIC_POLICY_VISUAL_ONLY,
        )

        self.assertEqual(
            12,
            agent._marked_crosswalk_for_path(position, next_waypoint).id,
        )
        self.assertFalse(agent._should_gate_crosswalk_entry(next_waypoint))
        self.assertEqual(
            21,
            agent.last_execution_traffic_light_snapshot['crosswalk_id'],
        )
        self.assertIn(21, agent._admitted_crosswalks)
        self.assertNotIn(12, agent._admitted_crosswalks)

    def test_sidewalk_corner_wins_over_adjacent_crosswalk_signal_envelope(self):
        route_crosswalk = SimpleNamespace(
            id=0,
            start=Vector(9300.0, 700.0),
            end=Vector(9300.0, -700.0),
        )
        adjacent_crosswalk = SimpleNamespace(
            id=5,
            start=Vector(10700.0, 700.0),
            end=Vector(9300.0, 700.0),
        )
        sidewalk_start = Vector(9300.0, 700.0)
        sidewalk_end = Vector(9300.0, 5208.3981)
        signal = make_signal(11, 10700.0, 700.0, crosswalk_id=5)
        communicator = FakeCommunicator({11: False})
        agent = RTAgent(
            position=Vector(9481.7, 900.0),
            direction=Vector(0.0, 1.0),
            destination=sidewalk_end,
            shortest_path=[sidewalk_end],
            communicator=communicator,
            llm=None,
            task_edges=[{
                "node1": [sidewalk_start.x, sidewalk_start.y],
                "node2": [sidewalk_end.x, sidewalk_end.y],
                "type": "sidewalk",
            }],
            route_crosswalks=[route_crosswalk],
            traffic_crosswalks=[route_crosswalk, adjacent_crosswalk],
            traffic_sidewalks=[SimpleNamespace(
                start=sidewalk_start,
                end=Vector(9300.0, 9300.0),
            )],
            crosswalk_signal_groups={0: [], 5: [signal]},
            traffic_signals=[signal],
            traffic_policy=TRAFFIC_POLICY_VISUAL_ONLY,
        )
        agent.route_polyline = [sidewalk_start, sidewalk_end]

        sidewalk_snapshot = agent._get_traffic_light_snapshot()

        self.assertFalse(sidewalk_snapshot["relevant"])
        self.assertEqual(0, agent.red_light_violations_count)

        # Another 118.3 cm laterally leaves the authored sidewalk while
        # remaining inside crosswalk 5, which must then own evaluation.
        agent.position = Vector(9600.0, 900.0)
        crossing_snapshot = agent._get_traffic_light_snapshot()

        self.assertEqual(5, crossing_snapshot["crosswalk_id"])
        self.assertEqual(1, agent.red_light_violations_count)

    def test_crosswalk_candidates_are_centered_and_monotonic(self):
        start = Vector(700.0, -10700.0)
        end = Vector(700.0, -9300.0)
        agent = self._route_agent(Vector(700.0, -10500.0), start, end, "crosswalk")

        waypoints = agent._find_waypoints()

        self.assertEqual(7, len(waypoints))
        self.assertTrue(all(abs(point.x - 700.0) < 1e-6 for point in waypoints))
        self.assertTrue(all(point.y > agent.position.y for point in waypoints))
        self.assertEqual(sorted(point.y for point in waypoints), [
            point.y for point in waypoints
        ])

    def test_sidewalk_candidate_corrects_lateral_drift_to_centerline(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        agent = self._route_agent(Vector(100.0, 180.0), start, end, "sidewalk")

        waypoints = agent._find_waypoints()

        self.assertTrue(all(abs(point.y) < 1e-6 for point in waypoints))
        self.assertTrue(all(point.x > agent.position.x for point in waypoints))

    def test_subgoal_radius_matches_benchmark_clean_ray_action_space(self):
        agent = RTAgent.__new__(RTAgent)
        agent.shortest_path = [Vector(0.0, 0.0), Vector(100.0, 0.0)]
        self.assertEqual(250.0, agent._current_subgoal_reach_threshold())

    def test_crosswalk_corridor_matches_painted_zebra_width(self):
        agent = RTAgent.__new__(RTAgent)
        agent.position = Vector(9081.7, -700.0)
        agent.current_destination = Vector(9300.0, 700.0)
        agent.route_polyline = [
            Vector(4281.6998, -700.0),
            Vector(9300.0, -700.0),
            Vector(9300.0, 700.0),
            Vector(9300.0, 5208.3981),
        ]
        agent.original_shortest_path = agent.route_polyline[1:]
        agent.shortest_path = agent.route_polyline[2:]
        agent.task_edges = [
            {
                "node1": [4281.6998, -700.0],
                "node2": [9300.0, -700.0],
                "type": "sidewalk",
            },
            {
                "node1": [9300.0, -700.0],
                "node2": [9300.0, 700.0],
                "type": "crosswalk",
            },
            {
                "node1": [9300.0, 700.0],
                "node2": [9300.0, 5208.3981],
                "type": "sidewalk",
            },
        ]

        snapshot = agent._route_adherence_snapshot()

        self.assertEqual("crosswalk", snapshot["edge_type"])
        self.assertAlmostEqual(218.3, snapshot["lateral_distance_cm"])
        self.assertEqual(210, snapshot["corridor_half_width_cm"])
        self.assertFalse(snapshot["within_transition_envelope"])
        self.assertFalse(snapshot["within_corridor"])

        agent.position = Vector(9468.0716, -803.5227)
        behind_curb = agent._route_adherence_snapshot()
        self.assertAlmostEqual(-0.073945, behind_curb["progress"], places=5)
        self.assertAlmostEqual(
            197.395,
            behind_curb["transition_distance_cm"],
            places=2,
        )
        self.assertTrue(behind_curb["within_transition_envelope"])
        self.assertTrue(behind_curb["within_corridor"])

        agent.position = Vector(9040.0, -700.0)
        outside = agent._route_adherence_snapshot()
        self.assertFalse(outside["within_transition_envelope"])
        self.assertFalse(outside["within_corridor"])

    def test_sidewalk_to_crosswalk_joint_uses_painted_zebra_threshold(self):
        agent = RTAgent.__new__(RTAgent)
        agent.position = Vector(9081.7, -700.0)
        agent.current_destination = Vector(9300.0, -700.0)
        agent.route_polyline = [
            Vector(4281.6998, -700.0),
            Vector(9300.0, -700.0),
            Vector(9300.0, 700.0),
        ]
        agent.original_shortest_path = agent.route_polyline[1:]
        agent.shortest_path = agent.route_polyline[1:]
        agent.task_edges = [
            {
                "node1": [4281.6998, -700.0],
                "node2": [9300.0, -700.0],
                "type": "sidewalk",
            },
            {
                "node1": [9300.0, -700.0],
                "node2": [9300.0, 700.0],
                "type": "crosswalk",
            },
        ]

        self.assertEqual(210.0, agent._current_subgoal_reach_threshold())
        self.assertGreater(
            agent.position.distance(agent.current_destination),
            agent._current_subgoal_reach_threshold(),
        )

        # The next 400 cm fixed ray passes the joint by only 181.7 cm and is
        # therefore both reachable and inside the visible zebra envelope.
        agent.position = Vector(9481.7, -700.0)
        self.assertLess(
            agent.position.distance(agent.current_destination),
            agent._current_subgoal_reach_threshold(),
        )

    def test_crosswalk_far_curb_uses_rectangular_reachable_envelope(self):
        agent = RTAgent.__new__(RTAgent)
        agent.position = Vector(9481.7, 500.0)
        agent.current_destination = Vector(9300.0, 700.0)
        agent.route_polyline = [
            Vector(9300.0, -700.0),
            Vector(9300.0, 700.0),
            Vector(9300.0, 5208.3981),
        ]
        agent.original_shortest_path = agent.route_polyline[1:]
        agent.shortest_path = agent.route_polyline[1:]
        agent.task_edges = [
            {
                "node1": [9300.0, -700.0],
                "node2": [9300.0, 700.0],
                "type": "crosswalk",
            },
            {
                "node1": [9300.0, 700.0],
                "node2": [9300.0, 5208.3981],
                "type": "sidewalk",
            },
        ]

        # Euclidean distance is 270.2 cm, but both crosswalk-frame components
        # are legal and reachable: 181.7 cm lateral, 200 cm from the curb.
        self.assertGreater(agent.position.distance(agent.current_destination), 250)
        self.assertTrue(agent._has_reached_current_subgoal())
        self.assertTrue(
            agent._route_adherence_snapshot()["within_end_transition_envelope"]
        )

        # The symmetric 400 cm overshoot is legal as well, so the ray lattice
        # cannot strand the actor on the crossing.
        agent.position = Vector(9481.7, 900.0)
        overshoot = agent._route_adherence_snapshot()
        self.assertGreater(overshoot["progress"], 1.01)
        self.assertTrue(overshoot["within_end_transition_envelope"])
        self.assertTrue(overshoot["within_corridor"])

        agent.position = Vector(9511.0, 500.0)
        self.assertFalse(agent._has_reached_current_subgoal())

    def test_active_edge_end_admits_benchmark_clean_ray_overshoot(self):
        agent = RTAgent.__new__(RTAgent)
        agent.position = Vector(-9401.761, -700.0)
        agent.current_destination = Vector(-9300.0, -700.0)
        agent.route_polyline = [
            Vector(-700.0, -700.0),
            Vector(-9300.0, -700.0),
            Vector(-9300.0, -6500.0),
        ]
        agent.original_shortest_path = agent.route_polyline[1:]
        agent.shortest_path = agent.route_polyline[1:]
        agent.task_edges = [
            {
                "node1": [-700.0, -700.0],
                "node2": [-9300.0, -700.0],
                "type": "sidewalk",
            },
            {
                "node1": [-9300.0, -700.0],
                "node2": [-9300.0, -6500.0],
                "type": "sidewalk",
            },
        ]

        snapshot = agent._route_adherence_snapshot()

        self.assertGreater(snapshot["progress"], 1.01)
        self.assertAlmostEqual(
            101.761,
            snapshot["end_transition_distance_cm"],
        )
        self.assertFalse(snapshot["within_start_transition_envelope"])
        self.assertTrue(snapshot["within_end_transition_envelope"])
        self.assertTrue(snapshot["within_corridor"])

        agent.position = Vector(-9600.0, -700.0)
        outside = agent._route_adherence_snapshot()
        self.assertFalse(outside["within_end_transition_envelope"])
        self.assertFalse(outside["within_corridor"])

    def test_production_agent_uses_benchmark_clean_ray_fan(self):
        agent = self._route_agent(
            Vector(9300.0, -9273.292),
            Vector(9300.0, -4473.2925),
            Vector(9300.0, -9300.0),
            "sidewalk",
        )
        agent._direction = Vector(0.0, -1.0)

        waypoints = agent._find_waypoints()

        for expected, point in zip(
            [100.0, 200.0, 200.0, 200.0, 400.0, 400.0, 400.0],
            waypoints,
        ):
            self.assertAlmostEqual(
                expected,
                agent.position.distance(point),
                places=3,
            )
        self.assertAlmostEqual(-9373.292, waypoints[0].y)
        self.assertNotEqual(9300.0, waypoints[2].x)

    def test_actual_off_corridor_position_continues_episode(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        agent = self._route_agent(Vector(100.0, 300.0), start, end, "sidewalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "WAIT"
        agent.success = False
        agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 4.25
        agent.step_num = 3

        feedback, _, _ = agent.get_feedback()

        self.assertFalse(agent.failed)
        self.assertIsNone(agent.failure_reason)
        self.assertEqual("Action completed successfully.", feedback)

    def test_off_route_position_does_not_hide_building_collision(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        agent = self._route_agent(Vector(100.0, 300.0), start, end, "sidewalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.communicator.get_states.return_value = (0, 0, 1, 0.0, 0, 0)
        agent.last_action_type = "WAIT"
        agent.success = False
        agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 4.25
        agent.step_num = 3

        feedback, collisions, _ = agent.get_feedback()

        self.assertIn("collided with a building", feedback)
        self.assertEqual(1, collisions["building"])
        self.assertEqual(1, collisions["ue_building_delta"])

    def test_ue_human_and_object_collision_deltas_are_counted(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        agent = self._route_agent(Vector(100.0, 300.0), start, end, "sidewalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.communicator.get_states.return_value = (1, 2, 0, 0.0, 0, 0)
        agent.last_action_type = "WAIT"
        agent.success = False
        agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 4.25
        agent.step_num = 3

        feedback, collisions, _ = agent.get_feedback()

        self.assertIn("collided with a human", feedback)
        self.assertIn("collided with an object", feedback)
        self.assertEqual(1, collisions["human"])
        self.assertEqual(2, collisions["object"])
        self.assertEqual(3, agent.collision_count)
        self.assertEqual(1, agent.collision_type_counts["human"])
        self.assertEqual(2, agent.collision_type_counts["object"])

    def test_mapped_building_proximity_without_ue_collision_is_endpoint_miss(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        agent = self._route_agent(Vector(140.0, 300.0), start, end, "sidewalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "MOVE"
        agent.last_move_execution_mode = "ue_physical_off_route"
        agent.last_action_start_position = Vector(100.0, 300.0)
        agent.last_commanded_waypoint = Vector(500.0, 300.0)
        agent.evaluator = RTEvaluator.__new__(RTEvaluator)
        agent.static_obstacles = [{
            "bounds": {
                "min_x": 200.0,
                "max_x": 300.0,
                "min_y": 250.0,
                "max_y": 350.0,
            },
        }]
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 5.0
        agent.step_num = 4

        feedback, collisions, _ = agent.get_feedback()

        self.assertNotIn("collided with a building", feedback)
        self.assertEqual(0, collisions["building"])
        self.assertEqual(0, collisions["ue_building_delta"])
        self.assertEqual(0, collisions["ue_physical_block_building"])
        self.assertIsNone(collisions["ue_physical_block_contact_kind"])
        self.assertEqual(1, collisions["mapped_building_proximity"])
        self.assertEqual(1, collisions["off_route_move_no_progress"])

    def test_partial_move_without_building_evidence_is_endpoint_miss(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        agent = self._route_agent(Vector(120.0, 300.0), start, end, "sidewalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "MOVE"
        agent.last_move_execution_mode = "ue_physical_off_route"
        agent.last_action_start_position = Vector(100.0, 300.0)
        agent.last_commanded_waypoint = Vector(500.0, 300.0)
        agent.evaluator = RTEvaluator.__new__(RTEvaluator)
        agent.static_obstacles = [{
            "bounds": {
                "min_x": 200.0,
                "max_x": 300.0,
                "min_y": 1000.0,
                "max_y": 1100.0,
            },
        }]
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 5.0
        agent.step_num = 4

        feedback, collisions, _ = agent.get_feedback()

        self.assertNotIn("collided with a building", feedback)
        self.assertEqual(0, collisions["building"])
        self.assertEqual(1, collisions["off_route_move_no_progress"])

    def test_ue_blocked_move_near_live_pedestrian_is_human_collision(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        agent = self._route_agent(Vector(140.0, 300.0), start, end, "sidewalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "MOVE"
        agent.last_move_execution_mode = "ue_physical_off_route"
        agent.last_action_start_position = Vector(100.0, 300.0)
        agent.last_commanded_waypoint = Vector(500.0, 300.0)
        # The contact may displace the pedestrian before post-action sampling;
        # preserve the pre-move UE center as the authoritative label source.
        agent.last_physical_move_actor_positions = {
            "RT_PEDESTRIAN_7": Vector(450.0, 300.0),
        }
        agent.evaluator = SimpleNamespace(_get_npc_positions=lambda: {})
        agent.static_obstacles = []
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 5.0
        agent.step_num = 4

        feedback, collisions, _ = agent.get_feedback()

        self.assertIn("collided with a human", feedback)
        self.assertEqual(1, collisions["human"])
        self.assertEqual(1, collisions["ue_physical_block_human"])
        self.assertEqual("human", collisions["ue_physical_block_contact_kind"])
        self.assertEqual(0, collisions["off_route_move_no_progress"])

    def test_ue_blocked_move_near_live_movable_is_object_collision(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        agent = self._route_agent(Vector(140.0, 300.0), start, end, "sidewalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "MOVE"
        agent.last_move_execution_mode = "ue_physical_off_route"
        agent.last_action_start_position = Vector(100.0, 300.0)
        agent.last_commanded_waypoint = Vector(500.0, 300.0)
        actor_name = "GEN_RT_RT_Box_7"
        agent.dynamic_obstacles = {
            actor_name: {"kind": "movable_obstacle"},
        }
        agent.evaluator = SimpleNamespace(
            _get_npc_positions=lambda: {
                actor_name: Vector(250.0, 300.0),
            },
        )
        agent.static_obstacles = []
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 5.0
        agent.step_num = 4

        feedback, collisions, _ = agent.get_feedback()

        self.assertIn("collided with an object", feedback)
        self.assertEqual(1, collisions["object"])
        self.assertEqual(1, collisions["ue_physical_block_object"])
        self.assertEqual("object", collisions["ue_physical_block_contact_kind"])
        self.assertEqual(0, collisions["off_route_move_no_progress"])


    def test_zero_displacement_without_ue_contact_is_not_building_collision(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        position = Vector(100.0, 300.0)
        agent = self._route_agent(position, start, end, "sidewalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "MOVE"
        agent.last_move_execution_mode = "ue_physical_off_route"
        agent.last_action_start_position = Vector(position.x, position.y)
        agent.last_commanded_waypoint = Vector(500.0, 300.0)
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 5.0
        agent.step_num = 4

        feedback, collisions, _ = agent.get_feedback()

        self.assertNotIn("collided with a building", feedback)
        self.assertIn("did not reach its waypoint", feedback)
        self.assertEqual(0, collisions["building"])
        self.assertEqual(1, collisions["off_route_move_no_progress"])

    def test_final_destination_suppresses_endpoint_miss_reassess_wording(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        agent = self._route_agent(Vector(end.x, end.y), start, end, "sidewalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "MOVE"
        agent.last_move_execution_mode = "ue_physical_off_route"
        agent.last_action_start_position = Vector(end.x, end.y)
        agent.last_commanded_waypoint = Vector(end.x + 500.0, end.y)
        agent.evaluator = RTEvaluator.__new__(RTEvaluator)
        agent.static_obstacles = []
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 5.0
        agent.step_num = 4

        feedback, collisions, _ = agent.get_feedback()

        self.assertIn("reached the final destination", feedback)
        self.assertNotIn("Reassess the route", feedback)
        self.assertEqual(1, collisions["off_route_move_no_progress"])
        self.assertTrue(agent.success)

    def test_zero_displacement_with_ue_building_state_is_collision(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        position = Vector(100.0, 300.0)
        agent = self._route_agent(position, start, end, "sidewalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.communicator.get_states.return_value = (0, 0, 1, 0.0, 0, 0)
        agent.last_action_type = "MOVE"
        agent.last_move_execution_mode = "ue_physical_off_route"
        agent.last_action_start_position = Vector(position.x, position.y)
        agent.last_commanded_waypoint = Vector(500.0, 300.0)
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 5.0
        agent.step_num = 4

        feedback, collisions, _ = agent.get_feedback()

        self.assertIn("collided with a building", feedback)
        self.assertEqual(1, collisions["building"])
        self.assertEqual(1, collisions["ue_building_delta"])
        self.assertEqual(0, collisions["mapped_building_proximity"])
        self.assertEqual(0, collisions["off_route_move_no_progress"])

    def test_off_sidewalk_corridor_notifies_vehicle_lane_consequence(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        agent = self._route_agent(Vector(100.0, 300.0), start, end, "sidewalk")
        events = []
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "WAIT"
        agent.success = False
        agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 4.25
        agent.direction = 90.0
        agent.speed = 200.0
        agent.illegal_crossing_events = []
        agent.illegal_crossing_callback = events.append
        agent.communicator.get_states.return_value = (0, 0, 0, 0.0, 0, 1)
        agent.traffic_roads = [SimpleNamespace(
            start=Vector(0.0, 300.0),
            end=Vector(1000.0, 300.0),
        )]

        agent.get_feedback()
        agent.get_feedback()

        self.assertEqual(1, len(events))
        self.assertEqual("illegal_crossing", events[0]["trigger_type"])
        self.assertEqual({"x": 100.0, "y": 300.0}, events[0]["agent_position"])
        self.assertTrue(events[0]["outside_authored_crossing"])

    def test_off_sidewalk_corridor_away_from_road_does_not_launch_vehicle(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        agent = self._route_agent(Vector(100.0, 300.0), start, end, "sidewalk")
        callback = MagicMock()
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "WAIT"
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 4.25
        agent.step_num = 3
        agent.direction = 0.0
        agent.speed = 200.0
        agent.illegal_crossing_events = []
        agent.illegal_crossing_excursion_active = False
        agent.illegal_crossing_excursion_active = False
        agent.illegal_crossing_callback = callback
        agent.traffic_roads = [SimpleNamespace(
            start=Vector(0.0, 1000.0),
            end=Vector(1000.0, 1000.0),
        )]

        feedback, collisions, _ = agent.get_feedback()

        self.assertEqual(0, collisions["touched_road"])
        self.assertEqual([], agent.illegal_crossing_events)
        self.assertEqual(0, agent.illegal_crossing_violations_count)
        callback.assert_not_called()

    def test_outside_crosswalk_roadway_entry_notifies_vehicle_consequence_once(self):
        start = Vector(0.0, 0.0)
        end = Vector(0.0, 1000.0)
        agent = self._route_agent(
            Vector(300.0, 200.0),
            start,
            end,
            "crosswalk",
        )
        events = []
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "WAIT"
        agent.success = False
        agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 4.25
        agent.step_num = 3
        agent.direction = 90.0
        agent.speed = 200.0
        agent.illegal_crossing_events = []
        agent.illegal_crossing_excursion_active = False
        agent.illegal_crossing_callback = events.append
        agent.communicator.get_states.return_value = (0, 0, 0, 0.0, 0, 1)

        agent.route_crosswalks = [
            SimpleNamespace(start=start, end=end)
        ]
        agent.traffic_roads = [
            SimpleNamespace(start=Vector(0.0, 200.0), end=Vector(1000.0, 200.0))
        ]
        agent.get_feedback()
        agent.get_feedback()

        self.assertEqual(1, len(events))
        self.assertEqual("illegal_crossing", events[0]["trigger_type"])
        self.assertTrue(events[0]["outside_authored_crossing"])
        self.assertFalse(agent.failed)

    def test_launched_road_entry_vehicle_is_not_revealed_in_feedback(self):
        start = Vector(0.0, 0.0)
        end = Vector(1000.0, 0.0)
        agent = self._route_agent(Vector(100.0, 300.0), start, end, "sidewalk")
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "MOVE"
        agent.success = False
        agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 4.25
        agent.direction = 90.0
        agent.speed = 200.0
        agent.illegal_crossing_events = []
        agent.illegal_crossing_callback = lambda event: {
            "event_id": event["event_id"],
            "disposition": "launched",
            "status": "launched",
        }
        agent.communicator.get_states.return_value = (0, 0, 0, 0.0, 0, 1)
        agent.traffic_roads = [SimpleNamespace(
            start=Vector(0.0, 300.0),
            end=Vector(1000.0, 300.0),
        )]

        feedback, _, _ = agent.get_feedback()

        self.assertEqual("Action completed successfully.", feedback)
        self.assertNotIn("Do not wait", feedback)
        self.assertNotIn("move promptly", feedback)
        self.assertFalse(agent.failed)
        self.assertTrue(agent.illegal_crossing_excursion_active)
        self.assertNotIn("vehicle", feedback.lower())

    def test_geometric_road_entry_does_not_record_illegal_crossing(self):
        start = Vector(-9300.0, 9300.0)
        end = Vector(-3430.4674, 9300.0)
        agent = self._route_agent(
            Vector(-9303.0, 10058.8),
            start,
            end,
            "sidewalk",
        )
        conflict = {
            "event_id": "illegal-crossing-0001",
            "disposition": "unavailable",
            "status": "unavailable",
        }
        unsafe_callback = MagicMock(return_value=conflict)
        red_light_callback = MagicMock()
        agent._check_vehicle_collision_failure = lambda: 0
        agent.last_action_type = "MOVE"
        agent.last_action_start_position = Vector(-9303.0, 9652.0)
        agent.success = agent.failed = False
        agent.failure_reason = None
        agent.logger = MagicMock()
        agent.sim_time_elapsed = 12.5
        agent.step_num = 13
        agent.direction = 90.0
        agent.speed = 200.0
        agent.illegal_crossing_events = []
        agent.illegal_crossing_excursion_active = False
        agent.illegal_crossing_excursion_active = False
        agent.illegal_crossing_callback = unsafe_callback
        agent.red_light_violation_callback = red_light_callback
        agent.traffic_roads = [SimpleNamespace(
            start=Vector(-10000.0, 10000.0),
            end=Vector(0.0, 10000.0),
        )]

        feedback, collisions, _ = agent.get_feedback()

        self.assertEqual("Action completed successfully.", feedback)
        self.assertNotIn("Do not wait", feedback)
        self.assertNotIn("move promptly", feedback)
        self.assertEqual(1, collisions["touched_road"])
        self.assertEqual(0, collisions["ue_touched_road"])
        self.assertEqual(1, collisions["geometric_touched_road"])
        self.assertEqual(0, collisions["unsafe_touched_road"])
        self.assertEqual([], agent.illegal_crossing_events)
        self.assertEqual(0, agent.illegal_crossing_violations_count)
        unsafe_callback.assert_not_called()
        red_light_callback.assert_not_called()

    def test_building_recovery_preserves_continuous_illegal_crossing_episode(self):
        agent = RTAgent.__new__(RTAgent)
        agent.name = "RT_AGENT"
        agent.position = Vector(200.0, 300.0)
        agent.direction = 0.0
        agent.shortest_path = [Vector(500.0, 300.0)]
        agent.current_destination = Vector(500.0, 300.0)
        agent.success = False
        agent.previous_distance = 300.0
        agent.step_num = 8
        agent._last_ue_location = (200.0, 300.0, 110.0)
        agent._last_ue_orientation = (0.0, 0.0, 0.0)
        agent.illegal_crossing_excursion_active = True
        agent.last_ue_collision_count = {
            "building": 0,
            "unsafe_touched_road": 1,
        }
        agent._remember_collision_free_state("post_action")
        self.assertTrue(
            agent.last_collision_free_state[
                "illegal_crossing_excursion_active"
            ]
        )
        self.assertEqual(
            1, agent.last_collision_free_state["unsafe_touched_road"]
        )

        # Reproduce the transient blocked endpoint sampled by get_feedback()
        # before building rollback returns to the checkpoint.
        agent.illegal_crossing_excursion_active = False
        agent.last_ue_collision_count["unsafe_touched_road"] = 0
        agent.communicator = SimpleNamespace(
            unrealcv=SimpleNamespace(
                set_location=MagicMock(),
                set_orientation=MagicMock(),
            )
        )
        agent.sync_ue = MagicMock()

        with patch("base.rt_agent.time.sleep"):
            restored = agent._restore_last_collision_free_state()

        self.assertTrue(restored)
        self.assertTrue(agent.illegal_crossing_excursion_active)
        self.assertEqual(1, agent.last_ue_collision_count["unsafe_touched_road"])
        self.assertFalse(agent._check_illegal_crossing_trigger(1, 1))

    def test_third_repeated_building_collision_terminates_episode(self):
        agent = RTAgent.__new__(RTAgent)
        agent.last_building_collision_action_key = None
        agent.consecutive_same_building_action_count = 0
        agent.building_collision_recovery_count = 0
        agent.building_collision_recovery_events = []
        agent.last_collision_free_state = None
        agent.step_num = 2
        agent.decision_count = 2
        agent.success = False
        agent.failed = False
        agent.failure_reason = None
        agent.stuck = False
        agent.stuck_reason = None
        agent._advance_simulation_time = lambda _seconds: None
        agent._restore_last_collision_free_state = lambda: True
        action = SimpleNamespace(action_type="MOVE_TO", action_param=3)

        for collision_index in range(1, BUILDING_COLLISION_REPEAT_LIMIT + 1):
            agent._handle_building_collision_recovery(
                action,
                "move",
                "collision",
                {"building": 1},
            )
            self.assertEqual(collision_index, agent.consecutive_same_building_action_count)
            self.assertEqual(
                collision_index >= BUILDING_COLLISION_REPEAT_LIMIT,
                agent.failed,
            )

        self.assertEqual(3, BUILDING_COLLISION_REPEAT_LIMIT)
        self.assertEqual("repeated_building_collision", agent.failure_reason)
        self.assertEqual("repeated_building_collision_action", agent.stuck_reason)

    def test_building_recovery_removes_rolled_back_subgoal_feedback(self):
        agent = RTAgent.__new__(RTAgent)
        agent.last_building_collision_action_key = None
        agent.consecutive_same_building_action_count = 0
        agent.building_collision_recovery_count = 0
        agent.building_collision_recovery_events = []
        agent.last_collision_free_state = None
        agent.step_num = 2
        agent.decision_count = 2
        agent.success = False
        agent.failed = False
        agent.failure_reason = None
        agent.stuck = False
        agent.stuck_reason = None
        agent._advance_simulation_time = lambda _seconds: None
        agent._restore_last_collision_free_state = lambda: True
        action = SimpleNamespace(action_type="MOVE_TO", action_param=3)

        feedback = agent._handle_building_collision_recovery(
            action,
            "move",
            "You have reached the subgoal. Proceed to the next waypoint. You collided with a building.",
            {"building": 1},
        )

        self.assertNotIn("reached the subgoal", feedback)
        self.assertIn("You collided with a building.", feedback)
        self.assertIn("Building collision recovery", feedback)

    def test_rolled_back_destination_feedback_removes_final_success(self):
        feedback = RTAgent._remove_rolled_back_destination_feedback(
            "You have reached the final destination. You collided with a building."
        )

        self.assertEqual("You collided with a building.", feedback)

    def test_building_rollback_rewrites_stale_road_location_warning(self):
        feedback = RTAgent._remove_rolled_back_destination_feedback(
            "You collided with a building. You are on the road outside a "
            "marked crossing. Do not wait in the roadway; move promptly to "
            "the nearest sidewalk. Traffic-rule violation recorded.",
            restored_unsafe_touched_road=False,
        )

        self.assertNotIn("You are on the road", feedback)
        self.assertIn("attempted move entered the roadway", feedback)
        self.assertIn("Traffic-rule violation", feedback)

    def test_building_rollback_keeps_road_warning_for_unsafe_checkpoint(self):
        warning = (
            "You are on the road outside a marked crossing. Do not wait in the "
            "roadway; move promptly to the nearest sidewalk. "
        )
        feedback = RTAgent._remove_rolled_back_destination_feedback(
            warning,
            restored_unsafe_touched_road=True,
        )

        self.assertEqual(warning, feedback)

    def test_building_collision_streak_counts_different_world_actions(self):
        agent = RTAgent.__new__(RTAgent)
        agent.last_building_collision_action_key = None
        agent.consecutive_same_building_action_count = 0
        agent.building_collision_recovery_count = 0
        agent.building_collision_recovery_events = []
        agent.last_collision_free_state = None
        agent.step_num = 2
        agent.decision_count = 2
        agent.success = False
        agent.failed = False
        agent.failure_reason = None
        agent.stuck = False
        agent.stuck_reason = None
        agent._advance_simulation_time = lambda _seconds: None
        agent._restore_last_collision_free_state = lambda: True
        action = SimpleNamespace(action_type="MOVE_TO", action_param=5)

        agent._handle_building_collision_recovery(
            action,
            "Move to Vector(x=100.0, y=200.0)",
            "collision",
            {"building": 1},
        )
        agent._handle_building_collision_recovery(
            action,
            "Move to Vector(x=100.0, y=200.0)",
            "collision",
            {"building": 1},
        )
        self.assertEqual(2, agent.consecutive_same_building_action_count)

        feedback = agent._handle_building_collision_recovery(
            action,
            "Move to Vector(x=600.0, y=700.0)",
            "collision",
            {"building": 1},
        )

        self.assertEqual(3, agent.building_collision_recovery_count)
        self.assertEqual(3, agent.consecutive_same_building_action_count)
        self.assertTrue(agent.failed)
        self.assertIn("(3/3)", feedback)

    def test_non_building_decision_resets_building_collision_streak(self):
        agent = RTAgent.__new__(RTAgent)
        agent.consecutive_same_building_action_count = 2
        agent.last_building_collision_action_key = "MOVE_TO:target"
        agent.last_building_collision_world_target = Vector(100.0, 200.0)
        agent.last_building_collision_action_type = MOVE_TO
        agent._check_vehicle_collision_failure = lambda: 0

        feedback, recovered = agent._finalize_post_action_collisions(
            None,
            "",
            "ok",
            {"building": 0, "vehicle": 0},
        )

        self.assertEqual("ok", feedback)
        self.assertFalse(recovered)
        self.assertEqual(0, agent.consecutive_same_building_action_count)
        self.assertIsNone(agent.last_building_collision_action_key)
        self.assertIsNone(agent.last_building_collision_world_target)
        self.assertIsNone(agent.last_building_collision_action_type)

    def test_building_collision_streak_counts_subcentimeter_target_jitter(self):
        agent = RTAgent.__new__(RTAgent)
        agent.last_building_collision_action_key = None
        agent.last_building_collision_world_target = None
        agent.last_building_collision_action_type = None
        agent.consecutive_same_building_action_count = 0
        agent.building_collision_recovery_count = 0
        agent.building_collision_recovery_events = []
        agent.last_collision_free_state = None
        agent.step_num = 2
        agent.decision_count = 2
        agent.success = False
        agent.failed = False
        agent.failure_reason = None
        agent.stuck = False
        agent.stuck_reason = None
        agent._advance_simulation_time = lambda _seconds: None
        agent._restore_last_collision_free_state = lambda: True
        action = SimpleNamespace(action_type=MOVE_TO, action_param=5)

        targets = (
            Vector(21521.4096, 18848.9306),
            Vector(21521.4096, 18848.9316),
            Vector(21521.4102, 18848.9309),
        )
        for expected_count, target in enumerate(targets, start=1):
            agent.last_commanded_waypoint = target
            feedback = agent._handle_building_collision_recovery(
                action,
                f"Move to {target}",
                "collision",
                {"building": 1},
            )
            self.assertEqual(
                expected_count,
                agent.consecutive_same_building_action_count,
            )

        self.assertTrue(agent.failed)
        self.assertEqual("repeated_building_collision", agent.failure_reason)
        self.assertIn("(3/3)", feedback)
        self.assertEqual(
            [1, 2, 3],
            [
                event["consecutive_building_collision_count"]
                for event in agent.building_collision_recovery_events
            ],
        )

    def test_illegal_crossing_launches_one_real_lane_vehicle(self):
        class FakeVehicle:
            def __init__(self):
                self.id = 0
                self.position = Vector(0.0, 0.0)
                self._direction = Vector(1.0, 0.0)
                self.state = None
                self.attributes = None

            @property
            def direction(self):
                return self._direction

            @direction.setter
            def direction(self, yaw):
                self._direction = Vector(
                    np.cos(np.radians(yaw)),
                    np.sin(np.radians(yaw)),
                )

            def set_attributes(self, throttle, brake, steering):
                self.attributes = (throttle, brake, steering)

        calls = []
        vehicle = FakeVehicle()
        manager = object.__new__(WorldManager)
        manager.traffic_controller = SimpleNamespace(vehicles=[vehicle])
        manager._signal_vehicle_routes = {
            0: {
                'approach_start': Vector(-1000.0, 0.0),
                'path_points': [Vector(1000.0, 0.0)],
                'incoming_lane': SimpleNamespace(id=10),
                'outgoing_lane': SimpleNamespace(id=11),
            },
        }
        manager.signal_traffic_communicator = SimpleNamespace(
            unrealcv=SimpleNamespace(
                get_location=lambda name: [0.0, 0.0, 72.0],
                set_physics=lambda *args: calls.append(('physics', args)),
                set_location=lambda *args: calls.append(('location', args)),
                set_orientation=lambda *args: calls.append(('orientation', args)),
                set_collision=lambda *args: calls.append(('collision', args)),
                set_movable=lambda *args: calls.append(('movable', args)),
            ),
            get_vehicle_name=lambda vehicle_id: f'RT_SIGNAL_VEHICLE_{vehicle_id}',
            update_vehicle=lambda *args: calls.append(('state', args)),
        )
        manager.red_light_conflict_vehicle_enabled = True
        manager.red_light_conflict_launch_distance_min_cm = 250.0
        manager.red_light_conflict_launch_distance_max_cm = 250.0
        manager.red_light_conflict_collision_radius_cm = 250.0
        manager.red_light_conflict_vehicle_probability = 1.0
        manager.red_light_conflict_nominal_speed_cm_s = 450.0
        manager._traffic_consequence_rng = random.Random(1)
        manager.illegal_crossing_conflict_vehicle_events = []
        manager._active_red_light_conflict = None
        manager.logger = MagicMock()

        manager._handle_illegal_crossing(
            {
                'event_id': 'illegal-crossing-0001',
                'sim_time_s': 5.0,
                'agent_position': {'x': 0.0, 'y': 200.0},
                'agent_direction': {'x': 0.0, 'y': -1.0},
                'agent_speed_cm_s': 200.0,
            }
        )

        event = manager.illegal_crossing_conflict_vehicle_events[0]
        self.assertEqual('launched', event['disposition'])
        self.assertEqual('illegal_crossing', event['trigger_type'])
        self.assertEqual('lane_aligned_intercept', event['control_mode'])
        self.assertAlmostEqual(250.0, event['launch_distance_cm'])
        self.assertAlmostEqual(0.0, event['target_position']['x'])
        self.assertAlmostEqual(0.0, event['target_position']['y'])
        self.assertEqual(72.0, event['launch_z_cm'])
        self.assertTrue(event['physics_quiesced_during_relocation'])
        self.assertEqual(1, len(manager.illegal_crossing_conflict_vehicle_events))

    def test_illegal_crossing_uses_authored_lane_without_staged_route(self):
        vehicle = SimpleNamespace(id=7)
        lane = SimpleNamespace(
            id=22,
            road_id=5,
            start=Vector(-1200.0, 0.0),
            end=Vector(1200.0, 0.0),
        )
        manager = object.__new__(WorldManager)
        manager.traffic_controller = SimpleNamespace(
            vehicles=[vehicle],
            lanes=[lane],
        )
        manager._signal_vehicle_routes = {}
        manager.red_light_conflict_nominal_speed_cm_s = 450.0
        manager.red_light_conflict_collision_radius_cm = 250.0

        crossing = manager._road_entry_conflict_crosswalk({
            'agent_position': {'x': 2200.0, 'y': 200.0},
        })
        self.assertIsNotNone(crossing)
        selected = manager._lane_aligned_conflict_launch(
            {
                'agent_position': {'x': 2200.0, 'y': 200.0},
                'agent_direction': {'x': 0.0, 'y': -1.0},
                'agent_speed_cm_s': 200.0,
            },
            crossing,
            [vehicle],
            900.0,
        )

        self.assertEqual('illegal-crossing-authored-lane-22', crossing.id)
        self.assertAlmostEqual(200.0, crossing.lane_distance_cm)
        self.assertIsNotNone(selected)
        self.assertEqual(
            'authored_lane_illegal_crossing',
            selected['route']['route_source'],
        )
        self.assertEqual(Vector(1300.0, 0.0), selected['launch_position'])
        self.assertEqual(Vector(2200.0, 0.0), selected['target_position'])
        self.assertAlmostEqual(1000.0, selected['lane_endpoint_extension_cm'])

    def test_diagonal_illegal_crossing_releases_passed_authored_lane_target(self):
        """Use full pedestrian motion, not only the virtual crossing axis."""
        vehicle = SimpleNamespace(id=0)
        lane = SimpleNamespace(
            id=11,
            road_id=5,
            start=Vector(-9600.0, 7700.0),
            end=Vector(-9600.0, 2300.0),
        )
        manager = object.__new__(WorldManager)
        manager.traffic_controller = SimpleNamespace(
            vehicles=[vehicle],
            lanes=[lane],
        )
        manager._signal_vehicle_routes = {}
        manager.red_light_conflict_nominal_speed_cm_s = 450.0
        manager.red_light_conflict_collision_radius_cm = 100.0
        event = {
            'agent_position': {'x': -9501.323, 'y': -1486.278},
            'movement_direction': {'x': -0.7484, 'y': -0.6632},
            'agent_direction': {'x': -0.7314, 'y': -0.6820},
            'agent_speed_cm_s': 200.0,
        }

        crossing = manager._road_entry_conflict_crosswalk(event)
        self.assertIsNotNone(crossing)
        self.assertEqual('illegal-crossing-authored-lane-11', crossing.id)

        selected = manager._lane_aligned_conflict_launch(
            event,
            crossing,
            [vehicle],
            480.21,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(
            'authored_lane_illegal_crossing',
            selected['route']['route_source'],
        )
        self.assertEqual(Vector(-9600.0, -700.0), selected['target_position'])
        self.assertTrue(selected['target_behind_agent'])
        self.assertLess(selected['signed_agent_distance_to_target_cm'], 0.0)
        self.assertEqual(0.0, selected['agent_distance_to_target_cm'])

    def test_far_passed_illegal_crossing_still_uses_authored_lane(self):
        vehicle = SimpleNamespace(id=0)
        lane = SimpleNamespace(
            id=11,
            road_id=5,
            start=Vector(0.0, -1200.0),
            end=Vector(0.0, 1200.0),
        )
        crossing = SimpleNamespace(
            id='illegal-crossing-authored-lane-11',
            road_id=5,
            start=Vector(-300.0, 0.0),
            end=Vector(300.0, 0.0),
            authored_lane=lane,
            lane_target=Vector(0.0, 0.0),
        )
        manager = object.__new__(WorldManager)
        manager._signal_vehicle_routes = {}
        manager.red_light_conflict_nominal_speed_cm_s = 450.0

        selected = manager._lane_aligned_conflict_launch(
            {
                'agent_position': {'x': 400.0, 'y': 0.0},
                'movement_direction': {'x': 1.0, 'y': 0.0},
                'agent_direction': {'x': 1.0, 'y': 0.0},
                'agent_speed_cm_s': 200.0,
            },
            crossing,
            [vehicle],
            900.0,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(
            'authored_lane_illegal_crossing',
            selected['route']['route_source'],
        )
        self.assertTrue(selected['target_behind_agent'])
        self.assertEqual(Vector(0.0, -900.0), selected['launch_position'])

    def test_illegal_crossing_near_lane_start_uses_road_mouth_runway(self):
        """Keep the rollout's diagonal lane-18 launch on its authored axis."""
        vehicle = SimpleNamespace(id=0)
        lane = SimpleNamespace(
            id=18,
            road_id=9,
            start=Vector(7700.0, -10400.0),
            end=Vector(2300.0, -10400.0),
        )
        manager = object.__new__(WorldManager)
        manager.traffic_controller = SimpleNamespace(
            vehicles=[vehicle],
            lanes=[lane],
        )
        manager._signal_vehicle_routes = {}
        manager.red_light_conflict_nominal_speed_cm_s = 450.0
        manager.red_light_conflict_collision_radius_cm = 100.0
        event = {
            'agent_position': {'x': 7683.23, 'y': -10490.472},
            'movement_direction': {'x': -0.7665, 'y': 0.6423},
            'agent_direction': {'x': -0.7071, 'y': 0.7071},
            'agent_speed_cm_s': 200.0,
        }

        crossing = manager._road_entry_conflict_crosswalk(event)
        self.assertIsNotNone(crossing)
        self.assertEqual('illegal-crossing-authored-lane-18', crossing.id)

        selected = manager._lane_aligned_conflict_launch(
            event,
            crossing,
            [vehicle],
            392.72,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(
            'authored_lane_illegal_crossing',
            selected['route']['route_source'],
        )
        self.assertEqual(Vector(7683.23, -10400.0), selected['target_position'])
        self.assertEqual(Vector(8075.95, -10400.0), selected['launch_position'])
        self.assertAlmostEqual(
            375.95,
            selected['lane_start_upstream_extension_cm'],
        )
        self.assertFalse(selected['target_behind_agent'])
        self.assertLessEqual(selected['agent_distance_to_target_cm'], 100.0)

    def test_illegal_crossing_retries_authored_lane_when_turn_segment_is_short(self):
        vehicle = SimpleNamespace(id=0)
        authored_lane = SimpleNamespace(
            id=21,
            road_id=10,
            start=Vector(200.0, 1500.0),
            end=Vector(200.0, 500.0),
        )
        manager = object.__new__(WorldManager)
        manager.traffic_controller = SimpleNamespace(
            vehicles=[vehicle],
            lanes=[authored_lane],
        )
        manager._signal_vehicle_routes = {
            0: {
                'approach_start': Vector(-100.0, 0.0),
                'path_points': [Vector(100.0, 0.0)],
                'incoming_lane': SimpleNamespace(id=12),
                'outgoing_lane': SimpleNamespace(id=20),
            },
        }
        manager.red_light_conflict_nominal_speed_cm_s = 450.0
        manager.red_light_conflict_collision_radius_cm = 250.0
        event = {
            'agent_position': {'x': 0.0, 'y': 0.0},
            'agent_direction': {'x': 0.0, 'y': 1.0},
            'agent_speed_cm_s': 200.0,
        }

        closest_crossing = manager._road_entry_conflict_crosswalk(event)
        self.assertEqual('illegal-crossing-lane-0-0', closest_crossing.id)
        self.assertIsNone(closest_crossing.authored_lane)

        selected = manager._lane_aligned_conflict_launch(
            event,
            closest_crossing,
            [vehicle],
            752.21,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(21, selected['route']['incoming_lane'].id)
        self.assertEqual(
            'authored_lane_illegal_crossing',
            selected['route']['route_source'],
        )
        self.assertAlmostEqual(752.21, selected['launch_distance_cm'])
        self.assertEqual(Vector(200.0, 752.21), selected['launch_position'])
        self.assertEqual(Vector(200.0, 0.0), selected['target_position'])

    def test_illegal_crossing_finds_lane_across_full_roadway_width(self):
        vehicle = SimpleNamespace(id=7)
        lanes = [
            SimpleNamespace(
                id=18,
                road_id=9,
                start=Vector(7700.0, -10400.0),
                end=Vector(2300.0, -10400.0),
            ),
            SimpleNamespace(
                id=19,
                road_id=9,
                start=Vector(2300.0, -9600.0),
                end=Vector(7700.0, -9600.0),
            ),
        ]
        manager = object.__new__(WorldManager)
        manager.traffic_controller = SimpleNamespace(
            vehicles=[vehicle],
            lanes=lanes,
        )
        manager._signal_vehicle_routes = {}
        manager.red_light_conflict_nominal_speed_cm_s = 450.0
        manager.red_light_conflict_collision_radius_cm = 250.0
        event = {
            'agent_position': {'x': 8864.482, 'y': -10422.293},
            'agent_direction': {'x': -0.5, 'y': -0.866},
            'agent_speed_cm_s': 200.0,
        }

        crossing = manager._road_entry_conflict_crosswalk(event)
        self.assertIsNotNone(crossing)
        self.assertEqual('illegal-crossing-authored-lane-19', crossing.id)
        self.assertAlmostEqual(822.293, crossing.lane_distance_cm)
        selected = manager._lane_aligned_conflict_launch(
            event,
            crossing,
            [vehicle],
            752.21,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(19, selected['route']['incoming_lane'].id)
        self.assertTrue(selected['target_behind_agent'])
        self.assertAlmostEqual(752.21, selected['launch_distance_cm'])


    def test_static_signal_vehicle_stabilization_excludes_active_conflict(self):
        vehicles = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
        unrealcv = SimpleNamespace(
            set_physics=MagicMock(),
            get_orientation=MagicMock(return_value=[17.0, 91.0, -23.0]),
            set_orientation=MagicMock(),
            set_collision=MagicMock(),
        )
        manager = object.__new__(WorldManager)
        manager.static_signal_vehicles = True
        manager.traffic_controller = SimpleNamespace(vehicles=vehicles)
        manager.signal_traffic_communicator = SimpleNamespace(
            unrealcv=unrealcv,
            get_vehicle_name=lambda vehicle_id: f'RT_SIGNAL_VEHICLE_{vehicle_id}',
        )
        manager._quiesced_static_signal_vehicle_ids = set()

        manager._stabilize_static_signal_vehicles(active_vehicle_id=2)
        manager._stabilize_static_signal_vehicles(active_vehicle_id=2)

        self.assertEqual(
            [call('RT_SIGNAL_VEHICLE_1', False)] * 2,
            unrealcv.set_physics.call_args_list,
        )
        self.assertEqual(
            [call((0.0, 91.0, 0.0), 'RT_SIGNAL_VEHICLE_1')] * 2,
            unrealcv.set_orientation.call_args_list,
        )
        self.assertEqual(
            [call('RT_SIGNAL_VEHICLE_1', False)] * 2,
            unrealcv.set_collision.call_args_list,
        )
        self.assertEqual({1}, manager._quiesced_static_signal_vehicle_ids)

    def test_active_kinematic_conflict_pose_is_reasserted_upright(self):
        vehicle = SimpleNamespace(id=2, position=Vector(-250.0, 0.0))
        unrealcv = SimpleNamespace(
            set_physics=MagicMock(),
            set_location=MagicMock(),
            set_orientation=MagicMock(),
            set_collision=MagicMock(),
        )
        manager = object.__new__(WorldManager)
        manager.agent = SimpleNamespace(sim_time_elapsed=7.0)
        manager.red_light_conflict_nominal_speed_cm_s = 450.0
        manager.signal_traffic_communicator = SimpleNamespace(unrealcv=unrealcv)
        record = {'released': True}
        active = {
            'vehicle': vehicle,
            'record': record,
            'launch_position': Vector(-250.0, 0.0),
            'direction': Vector(1.0, 0.0),
            'release_sim_time_s': 6.0,
            'vehicle_name': 'RT_SIGNAL_VEHICLE_2',
            'launch_z_cm': 72.0,
            'kinematic_motion': True,
        }

        manager._stabilize_active_kinematic_conflict_vehicle(active)

        unrealcv.set_physics.assert_called_once_with(
            'RT_SIGNAL_VEHICLE_2', False
        )
        unrealcv.set_location.assert_called_once_with(
            (200.0, 0.0, 72.0),
            'RT_SIGNAL_VEHICLE_2',
        )
        unrealcv.set_orientation.assert_called_once_with(
            (0.0, 0.0, 0.0),
            'RT_SIGNAL_VEHICLE_2',
        )
        unrealcv.set_collision.assert_called_once_with(
            'RT_SIGNAL_VEHICLE_2', True
        )
        self.assertEqual(Vector(200.0, 0.0), vehicle.position)
        self.assertTrue(record['kinematic_pose_enforced'])
        self.assertEqual(1, record['kinematic_pose_enforcement_count'])

    def test_conflict_vehicle_disappears_when_original_target_is_reached(self):
        vehicle = SimpleNamespace(
            id=3,
            position=Vector(0.0, 0.0),
            current_lane=None,
            state=VehicleState.MOVING,
            set_attributes=MagicMock(),
        )
        unrealcv = SimpleNamespace(
            set_physics=MagicMock(),
            set_collision=MagicMock(),
            set_movable=MagicMock(),
            destroy=MagicMock(),
        )
        manager = object.__new__(WorldManager)
        manager.red_light_conflict_collision_radius_cm = 50.0
        manager.agent = SimpleNamespace(
            position=Vector(0.0, 500.0),
            sim_time_elapsed=8.0,
            failed=False,
            success=False,
            failure_reason=None,
            vehicle_collision_count=0,
            collision_count=0,
            red_light_conflict_active=True,
            illegal_crossing_excursion_active=True,
        )
        manager.signal_traffic_communicator = SimpleNamespace(
            unrealcv=unrealcv,
            get_vehicle_name=lambda vehicle_id: f'RT_SIGNAL_VEHICLE_{vehicle_id}',
            update_vehicle=MagicMock(),
        )
        manager.traffic_controller = SimpleNamespace(
            vehicle_manager=SimpleNamespace(vehicles=[vehicle]),
        )
        manager._retired_signal_vehicle_ids = set()
        record = {
            'minimum_agent_distance_cm': 500.0,
            'impact_zone_reached': False,
            'collision_triggered': False,
            'released': True,
            'status': 'launched',
        }
        manager._active_red_light_conflict = {
            'vehicle': vehicle,
            'record': record,
            'launch_position': Vector(-250.0, 0.0),
            'direction': Vector(1.0, 0.0),
            'target_position': Vector(0.0, 0.0),
            'launch_sim_time_s': 5.0,
            'release_sim_time_s': 5.0,
            'last_vehicle_position': Vector(-20.0, 0.0),
            'last_agent_position': Vector(0.0, 500.0),
        }
        manager.logger = MagicMock()

        manager._update_red_light_conflict_vehicle()

        self.assertEqual(
            'target_reached_awaiting_ue_collision',
            record['status'],
        )
        self.assertTrue(record['target_reached'])
        unrealcv.destroy.assert_not_called()
        self.assertIn(vehicle, manager.traffic_controller.vehicle_manager.vehicles)
        self.assertIsNotNone(manager._active_red_light_conflict)
        self.assertFalse(manager.agent.failed)
        self.assertIsNone(manager.agent.failure_reason)

    def test_optional_background_activation_timeout_keeps_failed_asset_static(self):
        calls = []

        def activate(asset_id):
            calls.append(asset_id)
            if asset_id == 'GEN_RT_RT_Oil_227':
                raise TimeoutError('UnrealCV request timed out')

        manager = object.__new__(WorldManager)
        manager.communicator = SimpleNamespace(
            activate_object_movement=activate,
        )
        manager.logger = MagicMock()
        manager.optional_asset_activation_failures = []

        activated = manager._activate_optional_background_objects(
            ['GEN_RT_RT_Box_1', 'GEN_RT_RT_Oil_227', 'GEN_RT_RT_Ball_2'],
            'movable_obstacle',
        )

        self.assertEqual(
            ['GEN_RT_RT_Box_1', 'GEN_RT_RT_Ball_2'],
            activated,
        )
        self.assertEqual(
            ['GEN_RT_RT_Box_1', 'GEN_RT_RT_Oil_227', 'GEN_RT_RT_Ball_2'],
            calls,
        )
        self.assertEqual(1, len(manager.optional_asset_activation_failures))
        failure = manager.optional_asset_activation_failures[0]
        self.assertEqual('GEN_RT_RT_Oil_227', failure['asset_id'])
        self.assertEqual('kept_static', failure['disposition'])
        manager.logger.warning.assert_called_once()

    def test_optional_background_activation_does_not_hide_programming_errors(self):
        manager = object.__new__(WorldManager)
        manager.communicator = SimpleNamespace(
            activate_object_movement=MagicMock(
                side_effect=ValueError('invalid asset configuration')
            ),
        )
        manager.logger = MagicMock()
        manager.optional_asset_activation_failures = []

        with self.assertRaisesRegex(ValueError, 'invalid asset configuration'):
            manager._activate_optional_background_objects(
                ['GEN_RT_BAD_ASSET'],
                'movable_obstacle',
            )


if __name__ == "__main__":
    unittest.main()
