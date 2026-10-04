"""Extract only scientific fields needed to reconstruct the existing Task 19 comparison."""
from pathlib import Path
import json, math, hashlib, argparse
ap=argparse.ArgumentParser();ap.add_argument("--campaign",type=Path,required=True);ap.add_argument("--choreography",type=Path,required=True);args=ap.parse_args()

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=args.campaign
SPECS=[('astra','gpt-6-astra',33,44),('sol','gpt-5.6-sol',37,69)]
KINDS=['human','object','building','vehicle']
payload={'source_task':19,'source_map':'RT15','difficulty':'easy','reasoning':'low','seed':0,
 'presentation':'NYC reconstruction of the existing final-subgoal comparison; metrics are from the original run.',
 'clock':'Recorded simulation clock, including inference exposure and action execution. Uniform 6x playback.',
 'playback_speed':6,'coordinate_policy':'A single rigid rotation and translation for all source actors and both agents; no lateral scaling.',
 'models':{}}
for key,model,lo,hi in SPECS:
 folder=CAMPAIGN/model/'cells/easy_realtime_instructional_low/map3_15roads/task_019/attempt_001'
 steps=[]
 for p in sorted(folder.glob('*_steps/step_*/*_manifest.json')):
  a=json.loads(p.read_text());step=int(a.get('step') or 0)
  if not lo<=step<=hi:continue
  t=a.get('timing') or {};g=a.get('observation_geometry') or {};w=a.get('waypoint_execution') or {};turn=a.get('turn_execution') or {}
  parsed=((a.get('model_output') or {}).get('parsed_action') or {});action={k:parsed.get(k) for k in ['type','param']}
  pos=g['agent_position_cm'];direction=g['agent_direction'];end=w.get('post_action_position_cm') or pos
  yaw=math.degrees(math.atan2(direction['y'],direction['x']))
  record={'source_step':step,'source_decision':step+1,'action':action,
   'start_sim_seconds':t['sim_time_start_seconds'],'end_sim_seconds':t['sim_time_end_seconds'],
   'inference_exposure_seconds':t['simulation_thinking_latency_seconds'],'response_latency_seconds':t.get('model_inference_wall_seconds'),
   'position_cm':[pos['x'],pos['y']],'post_action_position_cm':[end['x'],end['y']],
   'yaw_degrees':yaw,'post_action_yaw_degrees':turn.get('verified_end_yaw_deg',yaw),
   'active_collisions':{k:int((a.get('collision_details') or {}).get(k) or 0) for k in KINDS},
   'passive_collisions':{k:int((a.get('passive_collision_details') or {}).get(k) or 0) for k in KINDS},
   'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  steps.append(record)
 assert len(steps)==hi-lo+1
 active=sum(sum(s['active_collisions'].values()) for s in steps);passive=sum(sum(s['passive_collisions'].values()) for s in steps)
 duration=steps[-1]['end_sim_seconds']-steps[0]['start_sim_seconds']
 payload['models'][key]={'name':model,'steps':steps,'summary':{'decisions':len(steps),'active_collisions':active,'passive_collisions':passive,'collisions':active+passive,'source_duration_seconds':duration,'video_duration_seconds':duration/6}}
assert payload['models']['astra']['summary']['collisions']==1
assert payload['models']['sol']['summary']['collisions']==17
source=args.choreography
choreo=json.loads(source.read_text());payload['source_choreography_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
(ROOT/'evidence/task19-choreography.json').write_text(json.dumps(choreo,indent=2)+'\n')
(ROOT/'evidence/task19-replay.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({k:v['summary'] for k,v in payload['models'].items()},indent=2))
