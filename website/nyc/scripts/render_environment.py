"""Native NYC environment tour with explicit illustrative actors and hazards."""
import sys,os,json,math,importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nyc_scene as n,nyc_assets as a
importlib.reload(n);importlib.reload(a)
preview=os.environ.get('RTSAFE_PREVIEW','1')=='1'
quality=os.environ.get('RTSAFE_QUALITY','raster')
times=[0,3,6,9,11.9666667] if preview else [f/30 for f in range(360)]
n.cleanup();n.controlled_world(False)
def position(forward,lateral=0,lift=0):
    x,y,z=n.along(forward,lateral);return(x,y,a.ground_at(x,y)+lift)
robot=n.skeletal('TourRobot',position=position(1500))
poses=[];camera_poses=[]
camera_keys=[(0,(500,0,850),(2500,-200,0)),(3.3,(1100,0,650),(2300,0,30)),
 (6,(1200,0,600),(2400,0,0)),(9,(1900,-250,1250),(3500,-1500,0)),
 (12,(1600,-250,1300),(3510,-1500,0))]
def camera_at(t):
    for (ta,la,aa),(tb,lb,ab) in zip(camera_keys,camera_keys[1:]):
        if t<=tb:
            p=max(0,min(1,(t-ta)/(tb-ta)));p=p*p*(3-2*p)
            loc=n.along(*[x+(y-x)*p for x,y in zip(la,lb)])
            target=n.along(*[x+(y-x)*p for x,y in zip(aa,ab)])
            return loc,n.look_at(loc,target)
    raise ValueError(t)
for f,t in enumerate(times):
    forward=1500+112*t;lateral=70*math.sin(t*.55)
    poses.append((f,position(forward,lateral),(0,n.YAW-90,0)))
    loc,rotation=camera_at(t)
    camera_poses.append((f,loc,rotation))
bindings=[(robot,poses,n.WALK)]
for i,(forward,lateral,speed) in enumerate([(2100,40,65),(2850,-90,-75),(3150,170,55)]):
    actor,anim=a.pedestrian('TourHuman'+str(i),i,position(forward,lateral))
    actor_poses=[(f,position(forward+t*speed,lateral),(0,n.YAW-90+(180 if speed<0 else 0),0)) for f,t in enumerate(times)]
    bindings.append((actor,actor_poses,[(anim,0,len(times),abs(speed)/120)]))
dog=a.dog('TourGo1',position(2750,80,15))
bindings.append((dog,[(f,position(2750-45*t,80,15),(0,n.YAW+180,0)) for f,t in enumerate(times)],None))
bindings.extend(a.dog_gait(dog,times))
a.prop('TourScooter',a.SCOOTER,position(1700,-220),n.YAW+20)
a.prop('TourHydrant',a.HYDRANT,position(2810,-230),n.YAW)
a.source_object('TourBox','box',position(2010,225))
a.source_object('TourBall','soccer',position(2240,-165))
a.patch('TourOil',position(1900,155),(.022,.016,.009),.12,(1.3,.8),.3)
a.patch('TourWater',position(2250,-70),(.022,.11,.16),.06,(1.25,.9),.4)
a.colored_prop('TourTrip','/Engine/BasicShapes/Cube',position(2530,150,4),(.04,.045,.049),(.9,1.1,.08),.8)
goal=position(3510,-2550,2)
a.colored_prop('TourGoal','/Engine/BasicShapes/Cylinder',goal,(.015,.42,.40),(1.6,1.6,.025),.3)
anchors={
 'start':position(1500,0,60),'goal':goal,
 'pedestrians':position(2500,40,130),'dog':position(2480,80,60),
 'scooter':position(1700,-220,80),'hydrant':position(2810,-230,90),'box':position(2010,225,30),
 'oil':position(1900,155,3),'water':position(2250,-70,3),'trip':position(2530,150,8),
 'crossing':position(3510,-1250,5),'signal':n.along(3429,-2123,632)}
route=[position(1500,0,3),position(2000,-20,3),position(2650,40,3),position(3500,-300,3),position(3510,-1250,3),goal]
cam=n.camera('TourCamera',camera_poses[0][1],camera_poses[0][2],86)
stem='environment'+('-samples' if preview else '')
out=n.ROOT+'/renders/'+stem;os.makedirs(out,exist_ok=True)
seq=n.sequence(stem.replace('-','_'),cam,camera_poses,bindings,len(times))
cfg=n.config(out,stem,1920,1080,8 if preview else 4,quality=quality)
aa=cfg.find_or_add_setting_by_class(n.unreal.MoviePipelineAntiAliasingSetting);aa.engine_warm_up_count=24;aa.render_warm_up_count=24
metadata={'times':times,'camera_poses':camera_poses,'horizontal_fov':86,'anchors':anchors,'route':route,'illustrative':True}
metadata['dynamic_anchors']={'pedestrians':[position(2100+65*t,40,130) for t in times],
    'dog':[position(2750-45*t,80,60) for t in times]}
with open(out+'/scene.json','w') as file:json.dump(metadata,file,indent=2)
n.run_jobs([{'name':stem,'sequence':seq,'config':cfg}],out+'/result.json',{'illustrative':True,'frames':len(times)})
