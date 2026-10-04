"""Native NYC reconstruction of the recorded Task 19 final subgoal.

RTSAFE_MODEL=astra|sol; RTSAFE_PREVIEW=1 samples the route before a full render.
Source metrics remain original RT15 measurements. NPC paths are reconstructed.
"""
import sys, os, json, math, importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nyc_scene as n, nyc_assets as a, replay_math as r
for module in (n,a,r): importlib.reload(module)
name=os.environ.get('RTSAFE_MODEL','astra')
preview=os.environ.get('RTSAFE_PREVIEW','1')=='1'
quality=os.environ.get('RTSAFE_QUALITY','raster')
payload=json.load(open(n.ROOT+'/evidence/task19-replay.json'))
choreo=json.load(open(n.ROOT+'/evidence/task19-choreography.json'))
model=payload['models'][name]
duration=model['summary']['source_duration_seconds']/6
times=([0, min(3,duration), min(7,duration), duration*.75, duration] if preview
       else [min(duration,f/30) for f in range(math.ceil((duration+1)*30))])
n.cleanup()
n.controlled_world()
n.clear_replay_corridor()
ground_cache={}
def ground(x,y):
    key=(round(x/100),round(y/100))
    if key not in ground_cache:ground_cache[key]=a.ground_at(x,y)
    return ground_cache[key]
def mapped(pos,lift=0):
    x,y=r.transform_position(pos,n.START)
    return (x,y,ground(x,y)+lift)
samples=[r.sample_agent(model,t*6) for t in times]
robot=n.skeletal('Replay_'+name,position=mapped(samples[0]['position_cm']))
poses=[];camera_poses=[]
heading=math.radians(n.YAW)
for f,s in enumerate(samples):
    loc=mapped(s['position_cm']);yaw=s['yaw_degrees']+n.YAW-90
    poses.append((f,loc,(0,yaw-90,0)))
    # The same elevated camera offset and focal length for both models.
    cam=(loc[0]-750*math.cos(heading),loc[1]-750*math.sin(heading),loc[2]+550)
    target=(loc[0]+250*math.cos(heading),loc[1]+250*math.sin(heading),loc[2]+100)
    camera_poses.append((f,cam,n.look_at(cam,target)))
animations=[];start=0
for f in range(1,len(samples)+1):
    if f==len(samples) or samples[f]['moving']!=samples[start]['moving']:
        animations.append((n.WALK if samples[start]['moving'] else n.IDLE,start,f,6 if not preview else 1))
        start=f
bindings=[(robot,poses,animations)]
for idx,item in enumerate(choreo['dynamic_objects_near_final_street']):
    a.source_object('ReplayObject_'+str(idx),item['actor_id'],mapped(item['last_location_cm'][:2]))
for idx,actor in enumerate(choreo['actors']):
    states=[r.sample_patrol(actor,s['source_sim_seconds']) for s in samples]
    is_dog=actor['kind']!='pedestrian'
    if is_dog:
        native=a.dog('ReplayNPC_'+str(idx),mapped(states[0][:2],15));anim=None
    else:
        native,walk=a.pedestrian('ReplayNPC_'+str(idx),idx,mapped(states[0][:2]));anim=[(walk,0,len(times),(6 if not preview else 1)*actor['speed_cm_s']/120)]
    actor_poses=[(f,mapped(state[:2],15 if is_dog else 0),(0,state[2]+n.YAW-90-(0 if is_dog else 90),0)) for f,state in enumerate(states)]
    bindings.append((native,actor_poses,anim))
    if is_dog:bindings.extend(a.dog_gait(native,times,6 if not preview else 1,idx*.4))
cam=n.camera('ReplayCamera',camera_poses[0][1],camera_poses[0][2],68)
check_frame=os.environ.get('RTSAFE_SAMPLE_FRAME','')
stem=name+('-samples' if preview else '')+('-check' if check_frame else '')
out=n.ROOT+'/renders/task19/'+stem;os.makedirs(out,exist_ok=True)
seq=n.sequence('task19_'+stem.replace('-','_'),cam,camera_poses,bindings,len(times))
config=n.config(out,stem,1920,1080,8 if preview else 4,quality=quality)
if check_frame:
    output=config.find_or_add_setting_by_class(n.unreal.MoviePipelineOutputSetting)
    output.use_custom_playback_range=True;output.custom_start_frame=int(check_frame);output.custom_end_frame=int(check_frame)+1
metadata={'model':name,'source':model['name'],'summary':model['summary'],'playback_speed':6,
    'preview':preview,'frame_times':times if preview else None,'ground_samples':len(ground_cache),
    'reconstruction':'Original agent poses and simulation clock; rigid NYC transform; authored NPC patrol paths.'}
with open(out+'/timing.json','w') as file:json.dump({'metadata':metadata,'frames':samples},file)
n.run_jobs([{'name':stem,'sequence':seq,'config':config}],out+'/result.json',metadata)
