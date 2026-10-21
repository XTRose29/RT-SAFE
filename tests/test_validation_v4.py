"""Regressions reproduced in the account-backed October pilot."""
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from base.rt_communicator import RTCommunicator
from base.rt_unrealcv import RTUnrealCV
from simworld.traffic.base.traffic_signal import TrafficSignalState as S
from simworld.traffic.manager.intersection_manager import IntersectionManager
from simworld.utils.vector import Vector
from tests.test_report_regressions import SweepWorld, agent_in


def clearance_manager():
    red = (S.VEHICLE_RED, S.PEDESTRIAN_RED)
    lights = [SimpleNamespace(id=i, lane_id=i, state=red) for i in range(66)]
    for light in lights:
        light.get_state = lambda light=light: light.state
    intersection = SimpleNamespace(
        id=1, traffic_lights=lights, pedestrian_lights=[],
        python_pedestrian_phase='clearance',
        python_pedestrian_phase_elapsed_s=0., cycle_count=0,
        all_traffic_lights_red=lambda: all(light.state == red for light in lights),
        increment_cycle_count=Mock(),
    )
    manager = IntersectionManager.__new__(IntersectionManager)
    manager.intersections = [intersection]
    manager.python_states = {}
    manager.logger = Mock()
    manager.config = {'traffic.traffic_signal.pedestrian_green_light_duration': 5,
                      'traffic.traffic_signal.pedestrian_clearance_duration': 3}
    return manager, intersection


def test_clearance_does_not_resend_2376_commands_for_one_second():
    manager, intersection = clearance_manager()
    comm = Mock()
    for delta in [.1, .1, .05] * 4:
        manager.update_intersections(comm, delta_time_s=delta)
    comm.traffic_signal_switch_to.assert_not_called()
    assert intersection.python_pedestrian_phase_elapsed_s == pytest.approx(1.)


def test_clearance_repairs_drift_and_requires_fresh_red_before_release():
    manager, intersection = clearance_manager()
    intersection.python_pedestrian_phase_elapsed_s = 3.
    light = intersection.traffic_lights[3]
    light.state = (S.VEHICLE_RED, S.PEDESTRIAN_GREEN)
    comm = Mock()
    manager.update_intersections(comm, delta_time_s=.1)
    comm.traffic_signal_switch_to.assert_called_once_with(3, 'all red')
    intersection.increment_cycle_count.assert_not_called()
    light.state = (S.VEHICLE_RED, S.PEDESTRIAN_RED)
    comm.reset_mock()
    manager.update_intersections(comm, delta_time_s=.1)
    comm.traffic_signal_switch_to.assert_called_once_with(0, 'green')


def test_batch_reads_keep_actor_identity_and_rendered_state():
    ue = RTUnrealCV.__new__(RTUnrealCV)
    ue.lock = threading.Lock()
    ue.client = Mock()
    ue.client.request_batch.return_value = [
        json.dumps({'green light': False, 'ped green': True, 'ped time': 4.5}),
        json.dumps({'green light': True, 'ped green': False, 'ped time': 0}),
    ]
    comm = RTCommunicator(ue)
    states = comm.get_traffic_signal_states([12, 3])
    assert [s['signal_id'] for s in states] == [12, 3]
    assert states[0]['pedestrian_walk'] and states[0]['remaining_time_s'] == 4.5
    assert states[1]['vehicle_green']
    ue.client.request_batch.assert_called_once_with([
        'vbp RT_TRAFFIC_SIGNAL_12 GetState', 'vbp RT_TRAFFIC_SIGNAL_3 GetState'])
    ue.client.request_batch.return_value = ['{}']
    with pytest.raises(RuntimeError, match='Incomplete'):
        comm.get_traffic_signal_states([12, 3])


def test_partial_progress_within_route_is_only_an_endpoint_miss():
    a = agent_in(SweepWorld())
    a.last_action_type = 'MOVE'
    a.last_move_execution_mode = 'ue_swept_segment_v2'
    a.last_action_start_position = Vector(0, 0)
    a.last_commanded_waypoint = Vector(200, 0)
    a.position = Vector(109.69, 0)
    _, details, _ = a.get_feedback()
    assert details['waypoint_missed_without_contact'] == 1
    assert details['move_displacement_cm'] == pytest.approx(109.69)
    assert details['off_route_move_no_progress'] == 0
    assert a.collision_count == 0


def test_building_contact_retains_pose_and_does_not_teleport_or_penalize():
    world = SweepWorld()
    a = agent_in(world)
    a.position = Vector(155, 0)
    a._handle_building_collision_recovery = Mock(side_effect=AssertionError('rollback'))
    feedback, recovered = a._finalize_post_action_collisions(
        None, '', 'You collided with a building.', {'building': 1})
    assert not recovered and a.position == Vector(155, 0)
    assert a.sim_time_elapsed == 0
    assert 'remains at the contact position' in feedback


def test_first_inference_counts_contacts_even_when_output_cannot_be_parsed():
    world = SweepWorld()
    world.human = 5  # Pre-episode setup contacts must not be scored.
    a = agent_in(world)
    world.set_camera_resolution = world.set_camera_fov = lambda *args: None
    world.resume_simulation = world.pause_simulation = lambda: None
    world.set_location = lambda value, name: world.location.__setitem__(slice(None), value)
    a.realtime_thinking = a.concurrent_realtime_inference = True
    a.token_based = False
    a.disable_step_evaluation = True
    a.record_per_step = False
    a.get_observation = lambda: {'ego_view': None, 'waypoints': []}
    def plan(*args):
        world.human += 2
        return None, .1, 0, 0, {'response_time': .1}
    a.plan = plan
    a.step('llm')
    assert a.passive_collision_count == a.collision_count == 2
    assert a.decision_trace[-1]['passive_collision_details']['human'] == 2
    assert a.last_ue_collision_count['human'] == 7
    assert a.avg_response_time == .1


def test_native_time_drives_accounting_and_controller_delta():
    class ClockWorld(SweepWorld):
        game_time = 10.
        def get_actor_sim_time(self, name):
            return self.game_time
        def advance_simulation_time(self, requested, scale):
            self.game_time += requested + .03  # Actual UE resume/pause overhead.
    world = ClockWorld()
    a = agent_in(world)
    del a._advance_simulation_time
    a.use_tick = True
    deltas = []
    a.simulation_step_callback = deltas.append
    a._advance_simulation_time(.1)
    assert a.sim_time_elapsed == pytest.approx(.13)
    assert deltas == pytest.approx([.13])


def test_contact_during_world_tick_stops_before_the_next_sweep():
    world = SweepWorld()
    a = agent_in(world)
    def advance(seconds):
        a.sim_time_elapsed += seconds
        world.human += 1
        world.location[0] = 5.  # Native impact displaced the capsule.
    a._advance_simulation_time = advance
    a.move_to(Vector(400, 0))
    assert a.position == Vector(5, 0)
    assert world.samples == []
    assert a.last_move_stop_reason == 'native_contact'
    assert a.last_move_elapsed_seconds == .1


def test_replay_uses_truncated_move_duration():
    from evaluation.replay_recorded_rollout_video import phase_durations
    _, action = phase_durations({
        'waypoint_execution': {'move_command_duration_seconds': 2., 'move_elapsed_seconds': .8},
    }, realtime=True)
    assert action == .8
