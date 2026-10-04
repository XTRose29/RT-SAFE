"""Scout Madison Square Park paths with a close third-person camera."""
import sys,os,json,importlib,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nyc_scene as n,nyc_assets as a
importlib.reload(n);importlib.reload(a)
n.cleanup();n.controlled_world()
out=n.ROOT+'/previews/park-scout';os.makedirs(out,exist_ok=True)
robot=n.skeletal('ScoutRobot');jobs=[];shots=[]
# A map overview provides coordinates before choosing a ground-level path.
for name,xy,yaw in [('center_east',(0,0),0),('center_north',(0,0),90),('south_path',(-4000,0),90),('north_path',(4000,0),-90),('west_path',(0,-5000),0),('east_path',(0,5000),180)]:
 z=a.ground_at(*xy);position=(*xy,z);angle=math.radians(yaw)
 loc=(xy[0]-650*math.cos(angle),xy[1]-650*math.sin(angle),z+410)
 target=(xy[0]+350*math.cos(angle),xy[1]+350*math.sin(angle),z+110)
 rotation=n.look_at(loc,target);cam=n.camera(name,loc,rotation,70)
 seq=n.sequence('park_'+name,cam,[(0,loc,rotation),(1,loc,rotation)],[(robot,[(0,position,(0,yaw-90,0)),(1,position,(0,yaw-90,0))],n.IDLE)],2)
 cfg=n.config(out+'/'+name,name,1600,900,4);jobs.append({'name':name,'sequence':seq,'config':cfg});shots.append({'name':name,'position':position,'yaw':yaw})
loc=(0,0,19000);rotation=(-90,0,0);cam=n.camera('overview',loc,rotation,80)
seq=n.sequence('park_overview',cam,[(0,loc,rotation),(1,loc,rotation)],[],2)
jobs.insert(0,{'name':'overview','sequence':seq,'config':n.config(out+'/overview','overview',1920,1080,4)})
n.run_jobs(jobs,out+'/result.json',{'shots':shots})
