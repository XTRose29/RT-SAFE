"""Regression contracts for rtsafe_report.pdf (2026-10-15)."""
import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from base.rt_agent import RTAgent
from base.rt_unrealcv import RTUnrealCV
from manager.world_manager import WorldManager
from simworld.utils.vector import Vector


class SweepWorld:
    def __init__(self, wall=None):
        self.location = [0., 0., 110.]
        self.orientation = [0., 0., 0.]
        self.wall = wall
        self.human = 0
        self.camera = None
        self.captures = []
        self.samples = []

    def get_location(self, name): return list(self.location)
    def get_orientation(self, name): return list(self.orientation)
    def set_orientation(self, value, name): self.orientation[:] = value
    def humanoid_stop(self, name): pass
    def set_location(self, *args): pytest.fail('Executable movement must never teleport')
    def set_camera_location(self, camera_id, value): self.camera = tuple(value)
    def set_camera_rotation(self, *args): pass
    def move_actor_with_sweep(self, name, value):
        self.location[:] = value
        if self.wall is not None and value[0] > self.wall:
            self.location[0] = self.wall
            self.human += 1
        self.samples.append(tuple(self.location))
        return tuple(self.location)


def agent_in(world, start=(0., 0.), heading=0., frames=False):
    world.location[:2] = start
    comm = SimpleNamespace(
        unrealcv=world,
        get_states=lambda name: (world.human, 0, 0, 0., 0, 0),
        get_vehicle_collision_number=lambda name: 0,
    )
    a = RTAgent(Vector(*start), Vector(1, 0), Vector(10000, 0),
                [Vector(10000, 0)], 100, comm, None,
                task_edges=[{'node1': [0., 0.], 'node2': [10000., 0.], 'type': 'sidewalk'}],
                use_action_frames=frames)
    a.name = 'RT_AGENT'
    a.direction = heading
    a.camera_id = 2
    a.first_person_camera_initialized = True
    a.record_action_frames = False
    a._get_traffic_light_snapshot = lambda: {}
    a._advance_simulation_time = lambda dt: setattr(a, 'sim_time_elapsed', round(a.sim_time_elapsed + dt, 6))
    def image(*args):
        world.captures.append(world.camera)
        return np.zeros((2, 2, 3), dtype=np.uint8)
    comm.get_camera_observation = image
    return a


@pytest.mark.parametrize('angle', [-45, 0, 45])
@pytest.mark.parametrize('start', [(0., 0.), (0., 350.)])
def test_same_straight_sweep_inside_and_outside_route(angle, start):
    world = SweepWorld()
    a = agent_in(world, start, frames=True)
    target = Vector(start[0] + 400 * math.cos(math.radians(angle)),
                    start[1] + 400 * math.sin(math.radians(angle)))
    a.move_to(target)
    assert a.position.distance(target) < 0.001
    assert a.sim_time_elapsed == 2.0
    assert len(world.samples) == 20
    assert len(world.captures) == 3
    for fraction, camera in zip([.25, .5, .75], world.captures):
        expected = Vector(*start) + (target - Vector(*start)) * fraction
        assert camera[:2] == pytest.approx((expected.x, expected.y))
    assert world.camera[:2] == pytest.approx((target.x, target.y))


def test_sweep_stops_at_thin_obstacle_without_endpoint_snap_or_repeat_hits():
    world = SweepWorld(wall=155.)
    a = agent_in(world, frames=True)
    a.last_action_type = 'MOVE'
    a.last_action_start_position = Vector(0, 0)
    a.move_to(Vector(400, 0))
    assert a.position == Vector(155, 0)
    assert a.sim_time_elapsed == .8
    assert a.last_move_stop_reason == 'sweep_blocked'
    assert world.human == 1
    assert world.camera[:2] == (155., 0.)
    feedback, details, _ = a.get_feedback()
    assert details['human'] == 1  # zero intensity does not invalidate a native contact
    assert details['ue_physical_block_human'] == 0
    assert 'collided with a human' in feedback


def test_sweep_preserves_height_updated_by_native_physics():
    world = SweepWorld()
    a = agent_in(world)
    advance = a._advance_simulation_time
    sweep = world.move_actor_with_sweep

    def physics_tick(dt):
        advance(dt)
        # Native floor depenetration/gravity changes height between sweeps.
        world.location[2] += .25

    def checked_sweep(name, target):
        assert target[2] == pytest.approx(world.location[2])
        return sweep(name, target)

    a._advance_simulation_time = physics_tick
    world.move_actor_with_sweep = checked_sweep
    a.move_to(Vector(400, 0))
    assert a.position == Vector(400, 0)
    assert world.location[2] == pytest.approx(115.)


def test_entry_event_is_observed_during_the_move_that_crosses():
    a = agent_in(SweepWorld())
    events = []
    def signal():
        if a.position.x >= 150 and not events:
            events.append((a.sim_time_elapsed, a.position.x))
        return {}
    a._get_traffic_light_snapshot = signal
    a.move_to(Vector(400, 0))
    assert events == [(0.8, 160.)]
    assert events[0][0] < a.sim_time_elapsed


def test_transient_trip_keeps_contact_position_and_time_even_if_endpoint_clear():
    a = agent_in(SweepWorld())
    a.position = Vector(400, 0)
    a.last_action_type = 'MOVE'
    a.sim_time_elapsed = 2.0
    a.last_action_safety_samples = [
        {'phase': 'pre_action', 'sim_time_s': 0., 'position_cm': {'x': 0., 'y': 0.}, 'overlap_type': 0},
        {'phase': 'after_sweep', 'sim_time_s': .2, 'position_cm': {'x': 40., 'y': 0.}, 'overlap_type': 1},
        {'phase': 'post_action', 'sim_time_s': 2., 'position_cm': {'x': 400., 'y': 0.}, 'overlap_type': 0},
    ]
    _, details, _ = a.get_feedback()
    assert details['fall'] == 1
    event = details['hazard_events'][0]
    assert event['observed_sim_time_s'] == .2
    assert event['position_cm'] == {'x': 40., 'y': 0.}
    assert event['trigger_actor_id'] is None  # never invent an actor from proximity
    assert a.last_action_safety_samples == []


def test_vehicle_terminal_still_records_other_contacts_and_hazards():
    a = agent_in(SweepWorld())
    a.failed = True
    a.failure_reason = 'vehicle_collision'
    a.communicator.get_states = lambda name: (0, 1, 0, 0., 1, 0)
    a._check_vehicle_collision_failure = lambda: 1
    a._advance_simulation_time = lambda dt: pytest.fail('No recovery after terminal impact')
    feedback, details, _ = a.get_feedback()
    assert details['vehicle'] == details['object'] == details['fall'] == 1
    assert 'struck by a vehicle' in feedback
    assert not a.success


def test_native_sweep_command_is_fail_closed_on_missing_runtime_support():
    ue = RTUnrealCV.__new__(RTUnrealCV)
    calls = []
    ue.lock = threading.Lock()
    def request(command):
        calls.append(command)
        return 'error unknown function'
    ue.client = SimpleNamespace(request=request)
    with pytest.raises(RuntimeError, match='does not acknowledge'):
        ue.move_actor_with_sweep('RT_AGENT', (1, 2, 3))
    assert calls == ['vbp RT_AGENT K2_SetActorLocation (X=1.000000000,Y=2.000000000,Z=3.000000000) true false']
    with pytest.raises(ValueError):
        ue.move_actor_with_sweep('RT_AGENT', (float('nan'), 2, 3))
    assert len(calls) == 1


def test_report_task15_spawn_ball_is_removed_before_activation():
    removed = []
    m = WorldManager.__new__(WorldManager)
    ball = {'id': 'ball', 'type': 'RT_Basketball_C', 'x': -4498.399, 'y': 10729.396}
    far = {'id': 'far', 'type': 'RT_Basketball_C', 'x': -4000., 'y': 10700.}
    m.map_static_obstacles = m.static_obstacles = [ball, far]
    m.dynamic_obstacle_metadata = {'ball': {}, 'far': {}}
    m.selected_movable_obstacle_ids = ['ball', 'far']
    m.selected_falling_object_ids = []
    m.agent = SimpleNamespace(position=Vector(-4479.807,10700), dynamic_obstacles=dict(m.dynamic_obstacle_metadata),
                              evaluator=SimpleNamespace(dynamic_obstacles=dict(m.dynamic_obstacle_metadata)))
    points = {'ball': [-4498.399,10729.396,2.], 'far': [-4000.,10700.,2.]}
    m.communicator = SimpleNamespace(unrealcv=SimpleNamespace(get_location=lambda name: points[name], destroy=removed.append))
    m._clear_initial_hazards()
    assert removed == ['ball']
    assert m.selected_movable_obstacle_ids == ['far']
    assert m.initial_hazard_clearance[0]['distance_to_spawn_cm'] == pytest.approx(34.78216,abs=.001)
    assert m.agent.evaluator.static_obstacles == [far]


@pytest.mark.parametrize('case', json.loads((Path(__file__).parent/'fixtures/report_actions.json').read_text()))
def test_report_action_coordinates_execute_to_chosen_target_without_navmesh_detour(case):
    a = agent_in(SweepWorld(), (case['start']['x'],case['start']['y']))
    target = Vector(case['target']['x'],case['target']['y'])
    a.move_to(target)
    assert a.position.distance(target) < .001
    assert a.last_move_execution_mode == 'ue_swept_segment_v2'


@pytest.mark.parametrize('approach_curb_first', [False, True])
def test_real_red_light_rule_records_entry_in_current_move_once(approach_curb_first):
    from tests.test_traffic_light_alignment import make_agent
    from base.rt_agent import TRAFFIC_POLICY_VISUAL_ONLY
    world = SweepWorld()
    world.location = [-9300., -700., 110.]
    a = make_agent(Vector(-9300, -700), Vector(-9300, 700), {7: False, 8: False})
    a.name = 'RT_AGENT'
    a.traffic_policy = TRAFFIC_POLICY_VISUAL_ONLY
    a.communicator.unrealcv = world
    a.communicator.get_states = lambda name: (0,0,0,0.,0,0)
    a._sync_first_person_camera = lambda: None
    a.use_action_frames = False
    a.record_action_frames = False
    a._advance_simulation_time = lambda dt: setattr(a, 'sim_time_elapsed', round(a.sim_time_elapsed + dt, 6))
    a.step_num = 12
    a._get_traffic_light_snapshot()
    assert a.red_light_violations_count == 0
    if approach_curb_first:
        # Integrate both fixes: native sweep sampling must allow a red-light
        # approach to the curb, then attribute actual entry to the next MOVE.
        a.move_to(Vector(-9300, -500))
        a._get_traffic_light_snapshot()
        assert a.position == Vector(-9300, -500)
        assert a.red_light_violations_count == 0
        assert not a._admitted_crosswalks
    a.move_to(Vector(-9300, -300))
    assert a.red_light_violations_count == 1
    event = dict(a.red_light_violation_events[0])
    assert event['violation_reason'] == 'entered_without_walk'
    a.step_num = 13
    a._get_traffic_light_snapshot()
    assert a.red_light_violation_events == [event]


def test_initial_hazard_aborts_before_any_model_decision():
    from base.rt_agent import InitialHazardOverlapError
    a = agent_in(SweepWorld())
    a.communicator.unrealcv.set_camera_resolution = lambda *args: None
    a.communicator.unrealcv.set_camera_fov = lambda *args: None
    a.sync_ue = lambda: None
    a.communicator.get_states = lambda name: (0, 0, 0, 0., 1, 0)
    a.plan = lambda *args: pytest.fail('Invalid setup must not request an action')
    with pytest.raises(InitialHazardOverlapError) as caught:
        a.step()
    assert caught.value.retryable is False
    assert a.fall_count == 0


def test_vehicle_during_inference_preserves_simultaneous_contacts_and_trip():
    a = agent_in(SweepWorld())
    a.step_num = 1
    a.realtime_thinking = True
    a.concurrent_realtime_inference = False
    a.record_per_step = False
    a.disable_step_evaluation = True
    a.get_observation = lambda: {'ego_view': None, 'waypoints': []}
    a._remember_collision_free_state = lambda *args: None
    a._capture_stationary_pose = lambda: None
    action = SimpleNamespace(action_type='WAIT', action_param='1')
    a.plan = lambda *args: (action, 0., 0, 0, None)
    states=iter([(1, 1, 0, 0., 1, 0), (1, 1, 0, 0., 0, 0)])
    a.communicator.get_states = lambda name: next(states)
    def vehicle():
        a.failed = True
        a.failure_reason = 'vehicle_collision'
        return 1
    a._check_vehicle_collision_failure = vehicle
    a.take_action = lambda *args: pytest.fail('Terminal inference must not execute an action')
    a.step()
    details = a.decision_trace[-1]['collision_details']
    assert details['human'] == details['object'] == details['vehicle'] == details['fall'] == 1
    assert a.collision_type_counts['human'] == 1
    assert a.collision_type_counts['object'] == 1
    assert a.fall_count == 1
    assert details['hazard_events'][0]['phase'] == 'during_inference'


def test_missing_cleared_prop_restored_from_authored_inventory(tmp_path, monkeypatch):
    from manager import world_manager
    node = {'id':'cleared_ball','instance_name':'RT_Basketball_C','properties':{
        'location':{'x':10.,'y':20.,'z':2.},'orientation':{'pitch':0.,'yaw':90.,'roll':0.},
        'scale':{'x':1.,'y':1.,'z':1.}}}
    (tmp_path/'obstacles.json').write_text(json.dumps({'nodes':[node]}))
    library=tmp_path/'assets.json';library.write_text(json.dumps({'RT_Basketball_C':{'asset_path':'/Ball'}}))
    monkeypatch.setattr(world_manager,'UE_ASSET_PATH',str(library))
    calls=[];live=set()
    def spawn(asset,name):calls.append((asset,name));live.add(name)
    m=WorldManager.__new__(WorldManager)
    m.communicator=SimpleNamespace(unrealcv=SimpleNamespace(
        get_objects=lambda:list(live),spawn_bp_asset=spawn,
        set_location=lambda *args:None,set_orientation=lambda *args:None,set_scale=lambda *args:None,
        get_location=lambda name:[10.,20.,2.]))
    m._restore_missing_hazard_props(str(tmp_path/'progen_world.json'))
    assert calls==[('/Ball','cleared_ball')]
    assert m.restored_hazard_props==['cleared_ball']
    m._restore_missing_hazard_props(str(tmp_path/'progen_world.json'))
    assert len(calls)==1
    assert m.restored_hazard_props==[]


def test_spawn_clearance_skips_prop_already_removed_by_isolation_setup():
    m=WorldManager.__new__(WorldManager)
    m.map_static_obstacles=[{'id':'removed','type':'RT_Basketball_C','x':0.,'y':0.}]
    m.agent=SimpleNamespace(position=Vector(0,0))
    m.communicator=SimpleNamespace(unrealcv=SimpleNamespace(get_objects=lambda:[],
        get_location=lambda name:pytest.fail('Absent actor must not be queried')))
    m._clear_initial_hazards()
    assert m.initial_hazard_clearance==[]
