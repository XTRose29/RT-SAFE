#!/usr/bin/env python3
"""Continue the same live task15 actor after a blind greedy policy stalls.

This is a physical route completion probe, not a scored model episode. Nothing
in the scene is respawned or teleported; only the decision policy is replaced.
"""
import argparse,json,os,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'SimWorld')]
from tools.validate_report_tasks import configure_probe_environment,install_scripted_policy
from base.rt_agent import RTAgent
from base.rt_unrealcv import RTUnrealCV
from base.rt_communicator import RTCommunicator
from simworld.utils.vector import Vector

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();configure_probe_environment();args.output.mkdir(parents=True,exist_ok=True)
u=RTUnrealCV(port=args.port,resolution=(720,640));u.pause_simulation();comm=RTCommunicator(u)
prior=json.loads(args.prior.read_text());task=next(t for t in json.loads((ROOT/'data/map2_12roads/tasks.json').read_text())['tasks'] if t['task_id']==15)
live=u.get_location('RT_AGENT');end=Vector(*task['end_point']);a=RTAgent(Vector(live[0],live[1]),Vector(-1,0),end,[end],100,comm,SimpleNamespace(model_name='scripted-native-continuation'),use_tick=True,realtime_thinking=False,use_action_frames=True,record_per_step=True,record_dir=str(args.output/'steps'),task_edges=task['edges'])
a.route_polyline=[Vector(*task['route_info']['shortest_path'][-2]),end]
a.name='RT_AGENT';a.initialize_first_person_camera();a.sync_ue();a.step_num=len(prior['steps']);a.decision_count=prior['steps'][-1]['decision'];a.sim_time_elapsed=prior['steps'][-1]['sim_time_seconds'];install_scripted_policy(a)
state=comm.get_states(a.name);a.last_ue_collision_count.update(human=state[0],object=state[1],building=state[2]);a.disable_step_evaluation=True
turn=a._turn_toward_target_action(Vector(a.position.x,9300.))
queue=[(turn.action_type,turn.action_param),('move_to','2')]
r={'prior_artifact':str(args.prior),'same_actor':'RT_AGENT','starting_live_pose':[float(v) for v in live],'scene_reloaded':False,'actor_teleported':False,'policy_intervention':'turn toward sidewalk, move 2m, then greedy','steps':[],'passed':False}
try:
 for i in range(16):
  if a.success or a.failed:break
  a.validation_action=queue[i] if i<len(queue) else 'baseline';a.step('greedy')
  r['steps'].append(a.decision_trace[-1]);print(json.dumps({'step':a.step_num,'position':[a.position.x,a.position.y],'success':a.success,'feedback':a.decision_trace[-1]['feedback']}),flush=True)
 r.update(passed=a.success,success=a.success,failed=a.failed,distance_to_task_destination_cm=a.position.distance(end),total_recorded_decisions=len(prior['steps'])+len(r['steps']))
finally:
 (args.output/'completion.json').write_text(json.dumps(r,indent=2,default=str)+'\n');u.client.disconnect()
