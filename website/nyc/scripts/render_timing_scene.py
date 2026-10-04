"""A controlled illustration: world freeze vs continuous time, same camera/action."""
import os,sys,math,importlib,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nyc_scene as n, nyc_assets as a
importlib.reload(n);importlib.reload(a)
mode=os.environ.get('RTSAFE_TIMING','realtime')
preview=os.environ.get('RTSAFE_PREVIEW','1')=='1'
times=[0,3.5,6.9,9.15,10.5] if preview else [f/30 for f in range(330)]
n.cleanup()
n.controlled_world()
n.clear_replay_corridor()
ground_cache={}
def grounded(forward,lateral=0):
    x,y,z=n.along(forward,lateral);key=(round(x/25),round(y/25))
    if key not in ground_cache:ground_cache[key]=a.ground_at(x,y)
    return (x,y,ground_cache[key])
robot=n.skeletal('TimingRobot')
human,walk=a.pedestrian('TimingPedestrian',1,grounded(1220,35))
robot_poses=[];human_poses=[]
for f,t in enumerate(times):
    wt=max(0,t-7) if mode=='static' else t
    forward=max(0,min(400,(t-7)*400/3))
    robot_poses.append((f,grounded(forward),(0,n.YAW-90,0)))
    human_poses.append((f,grounded(1220-100*wt,35),(0,n.YAW+90,0)))
animations=[(n.IDLE,0,210,0),(n.WALK,210,300),(n.IDLE,300,330)] if not preview else n.IDLE
def actor_animation(path,rate=1):
    if mode=='static' and not preview:return [(path,0,210,0),(path,210,330,rate)]
    return path if preview else [(path,0,330,rate)]
bindings=[(robot,robot_poses,animations),(human,human_poses,actor_animation(walk,100/120))]
for idx,(forward,lateral,direction) in enumerate([(1500,150,1),(2500,350,-1),(3800,0,1)]):
    ped,anim=a.pedestrian('TimingBackground'+str(idx),idx,grounded(forward,lateral))
    poses=[]
    for f,t in enumerate(times):
        wt=max(0,t-7) if mode=='static' else t
        poses.append((f,grounded(forward+direction*90*wt,lateral),(0,n.YAW-90+(180 if direction<0 else 0),0)))
    bindings.append((ped,poses,actor_animation(anim,.75)))
for idx,(forward,lateral,speed) in enumerate([(3000,-900,-300),(-1600,-1750,280)]):
    loc=n.along(forward,lateral);road_z=a.ground_at(loc[0],loc[1]);loc=(loc[0],loc[1],road_z)
    yaw=n.YAW+(180 if speed<0 else 0);car,lift=a.vehicle('TimingCar'+str(idx),loc,yaw)
    poses=[]
    for f,t in enumerate(times):
        wt=max(0,t-7) if mode=='static' else t
        p=n.along(forward+speed*wt,lateral)
        poses.append((f,(p[0],p[1],road_z+lift),(0,yaw,0)))
    bindings.append((car,poses,None))
loc=n.along(-600,-120,430);rotation=n.look_at(loc,n.along(200,20,110))
cam=n.camera('TimingCamera',loc,rotation,68)
check_frame=os.environ.get('RTSAFE_SAMPLE_FRAME','')
stem='timing-close-'+mode+('-samples' if preview else '')+('-check' if check_frame else '')
out=n.ROOT+'/renders/'+stem;os.makedirs(out,exist_ok=True)
seq=n.sequence(stem.replace('-','_'),cam,[(0,loc,rotation),(len(times)-1,loc,rotation)],bindings,len(times))
cfg=n.config(out,stem,1920,1080,8 if preview else 4)
if check_frame:
    output=cfg.find_or_add_setting_by_class(n.unreal.MoviePipelineOutputSetting)
    output.use_custom_playback_range=True;output.custom_start_frame=int(check_frame);output.custom_end_frame=int(check_frame)+1
with open(out+'/scene.json','w') as file:
    json.dump({'camera_location':loc,'camera_rotation':rotation,'horizontal_fov':68,
        'times':times,'robot_poses':robot_poses,'human_poses':human_poses,
        'pedestrian_route':'Along the sidewalk toward the agent; no traffic-lane crossing.'},file)
n.run_jobs([{'name':stem,'sequence':seq,'config':cfg}],out+'/result.json',{'times':times if preview else None,'mode':mode,'illustrative':True})
