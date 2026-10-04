"""A gentle native camera orbit over Madison Square Park; eight-second loop."""
import sys,os,math,importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nyc_scene as n
importlib.reload(n)
n.cleanup();n.controlled_world(False)
poses=[]
for frame in range(240):
    phase=2*math.pi*frame/240
    loc=(15000+1100*math.sin(phase),-14000+800*(1-math.cos(phase)),18000+250*math.sin(phase))
    poses.append((frame,loc,n.look_at(loc,(0,0,0))))
cam=n.camera('HeroCamera',poses[0][1],poses[0][2],68)
seq=n.sequence('hero_orbit',cam,poses,[],240)
out=n.ROOT+'/renders/hero';os.makedirs(out,exist_ok=True)
cfg=n.config(out,'hero',2560,1440,4,quality=os.environ.get('RTSAFE_QUALITY','raster'))
aa=cfg.find_or_add_setting_by_class(n.unreal.MoviePipelineAntiAliasingSetting);aa.engine_warm_up_count=24;aa.render_warm_up_count=24
n.run_jobs([{'name':'hero','sequence':seq,'config':cfg}],out+'/result.json',{'duration':8,'fps':30,'native_city':True})
