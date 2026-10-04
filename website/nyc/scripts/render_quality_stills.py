import sys,os,importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nyc_scene as n
importlib.reload(n)
n.cleanup();robot=n.skeletal('QualityRobot')
out=n.ROOT+'/previews/quality-v2';os.makedirs(out,exist_ok=True)
shots=[
 ('city_detail',(25000,5000,30000),n.look_at((25000,5000,30000),(0,0,0)),62,'raster'),
 ('follow_raster',n.along(-750,0,550),n.look_at(n.along(-750,0,550),n.along(250,0,100)),68,'raster'),
 ('follow_ssgi',n.along(-750,0,550),n.look_at(n.along(-750,0,550),n.along(250,0,100)),68,'ssgi'),
 ('crossing',n.along(2000,-250,1400),n.look_at(n.along(2000,-250,1400),n.along(3700,-1000,0)),72,'raster'),
]
jobs=[]
for name,loc,rotation,fov,quality in shots:
    cam=n.camera(name,loc,rotation,fov)
    seq=n.sequence(name,cam,[(0,loc,rotation),(1,loc,rotation)],[(robot,[(0,n.START,(0,n.YAW-90,0)),(1,n.START,(0,n.YAW-90,0))],n.IDLE)],2)
    cfg=n.config(out+'/'+name,name,2560,1440,8,quality=quality)
    aa=cfg.find_or_add_setting_by_class(n.unreal.MoviePipelineAntiAliasingSetting);aa.engine_warm_up_count=24;aa.render_warm_up_count=24
    jobs.append({'name':name,'sequence':seq,'config':cfg})
n.run_jobs(jobs,out+'/result.json',{'quality_comparison':'Same native high third-person camera, raster vs screen-space indirect lighting'})
