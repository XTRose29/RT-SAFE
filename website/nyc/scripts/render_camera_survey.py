import sys,os,json,math,importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nyc_scene as n
importlib.reload(n)
n.unreal.log_warning('RTSAFE_NYC survey cleanup start')
n.cleanup()
n.unreal.log_warning('RTSAFE_NYC survey loading Manny')
robot=n.skeletal('SurveyRobot')
n.unreal.log_warning('RTSAFE_NYC survey actor loaded')
shots=[
 ('skyline',(62000,0,32911),(-25,180,0),68),
 ('park',(15000,-14000,18000),n.look_at((15000,-14000,18000),(0,0,0)),68),
 ('street',n.along(-280,0,178),(-4.205357,n.YAW,0),82),
 ('high_follow',n.along(-900,0,650),n.look_at(n.along(-900,0,650),n.along(350,0,100)),68),
 ('high_wide',n.along(-1400,150,1050),n.look_at(n.along(-1400,150,1050),n.along(450,0,100)),65),
 ('intersection',n.along(1500,-1200,900),n.look_at(n.along(1500,-1200,900),n.along(1900,500,0)),74),
]
jobs=[]
out=n.ROOT+'/previews/survey-v1';os.makedirs(out,exist_ok=True)
for name,loc,rot,fov in shots:
 cam=n.camera(name,loc,rot,fov)
 seq=n.sequence('survey_'+name,cam,[(0,loc,rot),(1,loc,rot)],[(robot,[(0,n.START,(0,n.YAW-90,0)),(1,n.START,(0,n.YAW-90,0))],n.IDLE)])
 jobs.append({'name':name,'sequence':seq,'config':n.config(out+'/'+name,name,1920,1080,8)})
n.run_jobs(jobs,out+'/result.json',{'shots':[{'name':a,'location':b,'rotation':c,'fov':d} for a,b,c,d in shots],'world':n.world().get_path_name()})
