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
# Park path selected from native camera surveys; keep all city geometry intact.
n.START=(-4000,2850,a.ground_at(-4000,2850));n.YAW=0
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
for idx,(forward,lateral,direction) in enumerate([(1600,100,1),(2400,180,-1),(3200,0,1)]):
    ped,anim=a.pedestrian('TimingBackground'+str(idx),idx,grounded(forward,lateral))
    poses=[]
    for f,t in enumerate(times):
        wt=max(0,t-7) if mode=='static' else t
        poses.append((f,grounded(forward+direction*90*wt,lateral),(0,n.YAW-90+(180 if direction<0 else 0),0)))
    bindings.append((ped,poses,actor_animation(anim,.75)))
loc=n.along(-540,-90,315);rotation=n.look_at(loc,n.along(220,20,105))
cam=n.camera('TimingCamera',loc,rotation,68)
check_frame=os.environ.get('RTSAFE_SAMPLE_FRAME','')
stem='timing-close-'+mode+('-samples' if preview else '')+('-check' if check_frame else '')
out=n.ROOT+'/renders/'+stem;os.makedirs(out,exist_ok=True)
seq=n.sequence(stem.replace('-','_'),cam,[(0,loc,rotation),(len(times)-1,loc,rotation)],bindings,len(times))
cfg=n.config(out,stem,1920,1080,8,quality="ssgi")
if check_frame:
    output=cfg.find_or_add_setting_by_class(n.unreal.MoviePipelineOutputSetting)
    output.use_custom_playback_range=True;output.custom_start_frame=int(check_frame);output.custom_end_frame=int(check_frame)+1
with open(out+'/scene.json','w') as file:
    json.dump({'camera_location':loc,'camera_rotation':rotation,'horizontal_fov':68,
        'times':times,'robot_poses':robot_poses,'human_poses':human_poses,
        'pedestrian_route':'Along a paved Madison Square Park path toward the agent; no road or grass crossing.','location':'Madison Square Park, interior path near the fountain'},file)
n.run_jobs([{'name':stem,'sequence':seq,'config':cfg}],out+'/result.json',{'times':times if preview else None,'mode':mode,'illustrative':True})
