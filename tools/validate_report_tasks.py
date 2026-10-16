#!/usr/bin/env python3
"""Run actual task rollouts against a disposable UE server (never a live campaign).

Only the decision policy is scripted. Observation, movement, traffic, collision
state, feedback, camera capture, and artifact writing use the production code.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'SimWorld')]
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from base.rt_unrealcv import RTUnrealCV
from base.rt_communicator import RTCommunicator
from base.rt_action_space import RTActionSpace
from manager.world_manager import WorldManager


def configure_probe_environment():
    from evaluation.run_qwen3vl8b_all_maps import configure_benchmark_environment
    configure_benchmark_environment(SimpleNamespace(
        record_png_compress_level=0, observation_width=720, observation_height=640,
        observation_fov_deg=100., observation_camera_pitch_deg=-25.,
        fast_simulation=True, concurrent_realtime_inference=False,
        red_light_conflict_vehicle=True))


class PreparedWorld(WorldManager):
    def run(self):
        pass

    def _create_llm_from_config(self, config):
        return SimpleNamespace(model_name='scripted-native-validation')


def prepare(ue, output, task_id=15):
    m = PreparedWorld(RTCommunicator(ue), agent_path='data/agents.json',
        task_file_path='data/map2_12roads/tasks.json', seed=0, use_tick=True,
        realtime_thinking=False, use_action_frames=True, record_per_step=True,
        control_mode='greedy', pedestrians_enabled=False,
        movable_obstacles_enabled=False, irregular_npcs_enabled=False,
        falling_objects_enabled=False, red_light_conflict_vehicle_enabled=False,
        results_dir=str(output))
    tasks = json.loads((ROOT/'data/map2_12roads/tasks.json').read_text())['tasks']
    m.run_single_task(next(i for i,t in enumerate(tasks) if t['task_id']==task_id), 'hard')
    return m


def install_scripted_policy(a):
    """Use the real image-history assembly but a deterministic action source."""
    original = a.plan
    a.validation_action = 'baseline'
    a.validation_inputs = []
    def plan(obs, mode):
        action, _, chars, tokens, metadata = original(obs, mode)
        metadata['input_images'] = a._decision_input_images(obs['ego_view'])
        a.validation_inputs.append(len(metadata['input_images']))
        wanted = a.validation_action
        if wanted != 'baseline':
            action = None if wanted is None else RTActionSpace(action_type=wanted[0], action_param=wanted[1])
        metadata['full_response'] = 'invalid JSON' if action is None else action.model_dump_json()
        metadata['response_time'] = 0.
        return action, 0., chars, tokens, metadata
    a.plan = plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args(); configure_probe_environment(); args.output.mkdir(parents=True,exist_ok=True)
    # The full route is a controlled static-scene harness test. Vehicle
    # consequences and changing signals are exercised by the crossing probe.
    os.environ['SIMWORLD_SIGNAL_TRAFFIC_VEHICLES']='0'
    os.environ['SIMWORLD_SIGNAL_TRAFFIC_PEDESTRIANS']='0'
    ue=RTUnrealCV(port=args.port, resolution=(720,640));ue.pause_simulation()
    result={'kind':'native_task_rollout','task_id':15,'policy':'scripted then greedy',
            'dynamic_actors_enabled':False,'static_map_hazards_enabled':True,'signal_vehicles':0,'steps':[],'passed':False}
    try:
        m=prepare(ue,args.output/'task15'); a=m.agent
        install_scripted_policy(a)
        result['initial_hazard_clearance']=m.initial_hazard_clearance
        original_capture=m.communicator.get_camera_observation
        captures=[]
        def capture(camera, *params, **kwargs):
            actor=[float(v) for v in ue.get_location(a.name)];cam=[float(x) for x in ue.get_camera_location(camera).split()]
            captures.append({'step':a.step_num,'actor':actor,'camera':cam,
                             'error_cm':math.hypot(actor[0]-cam[0],actor[1]-cam[1])})
            return original_capture(camera,*params,**kwargs)
        m.communicator.get_camera_observation=capture
        # First action matches all eight reported policies; then an explicit
        # parse-error probe tests the actual step pipeline and image retention.
        actions={0:('move_to','5'),1:None,2:('wait','1'),3:('turn_around','L30'),4:('turn_around','R30')}
        for _ in range(80):
            if a.success or a.failed:break
            a.validation_action=actions.get(a.step_num,'baseline')
            before=a.step_num
            a.step('greedy')
            trace=dict(a.decision_trace[-1])
            trace['history_frames_after_step']=len(a.action_frames)
            trace['live_actor']=[float(v) for v in ue.get_location(a.name)]
            trace['live_camera']=[float(x) for x in ue.get_camera_location(a.camera_id).split()]
            result['steps'].append(trace)
            print(json.dumps({'step':before,'position':trace['live_actor'],
                 'action':trace.get('executed_action'),'feedback':trace.get('feedback'),
                 'success':a.success,'failed':a.failed}),flush=True)
            (args.output/'progress.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        m.save_evaluation_data()
        result.update(success=a.success,failed=a.failed,failure_reason=a.failure_reason,
                      fall_count=a.fall_count,red_light_count=a.red_light_violations_count,
                      input_image_counts=a.validation_inputs,camera_samples=captures,
                      max_camera_error_cm=max(c['error_cm'] for c in captures),
                      task_record_dir=a.record_dir)
        manifests=list(Path(a.record_dir).glob('step_*/*manifest.json'))
        result['manifest_count']=len(manifests)
        result['image_step_count']=sum(bool(list(d.glob('*input_frame*.png'))) for d in Path(a.record_dir).glob('step_*'))
        first=result['steps'][0]
        checks={'first_move_no_trip':first['collision_details']['fall']==0,
                'parse_error_clears_history':result['steps'][1]['history_frames_after_step']==0,
                'next_decision_has_only_current_image':a.validation_inputs[2]==1,
                'camera_tracks_actor':result['max_camera_error_cm']<.5,
                'task_completed':a.success,
                'all_step_manifests':len(manifests)==a.step_num,
                'all_step_inputs_retained':result['image_step_count']==a.step_num}
        result['checks']=checks;result['passed']=all(checks.values())
    finally:
        (args.output/'task15-rollout.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        ue.client.disconnect()
    print(json.dumps({'passed':result['passed'],'checks':result.get('checks')}),flush=True)


if __name__=='__main__':main()
