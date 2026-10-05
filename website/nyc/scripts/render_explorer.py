"""Render four native 360-degree viewpoints as six square camera faces each."""
import sys, os, json, importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nyc_scene as n, nyc_assets as a
importlib.reload(n);importlib.reload(a)
# Reuse the environment tour's authored actors, hazards, and route without launching its queue.
source=(Path(__file__).parent/'render_environment.py').read_text()
ns={'__file__':str(Path(__file__).parent/'render_environment.py'),'__name__':'explorer_setup'}
os.environ['RTSAFE_PREVIEW']='1'
exec(compile(source.split("cam=n.camera('TourCamera'")[0],ns['__file__'],'exec'),ns)
n.controlled_world(True)
frames=24
bindings=[]
for actor,poses,anim in ns['bindings']:
    loc,rot=poses[0][1:]
    if isinstance(anim,list): anim=[(anim[0][0],0,frames,.001)]
    elif anim:anim=[(anim,0,frames,.001)]
    bindings.append((actor,[(0,loc,rot),(frames-1,loc,rot)],anim))
# Static native vehicles make the traffic corridor legible in the 360 captures.
for i,(f,l) in enumerate([(2500,-1100),(4250,-1900),(6800,-1000)]):
    a.prop('ExplorerCar'+str(i),a.CAR,ns['position'](f,l),n.YAW+180)
locations=[('approach',n.along(850,20,300),n.YAW),('hazards',n.along(1900,-20,245),n.YAW),('crossing',n.along(3400,0,260),n.YAW-90),('park',(-4000,2850,a.ground_at(-4000,2850)+220),0)]
if os.environ.get('RTSAFE_EXPLORER_ONLY'):locations=[x for x in locations if x[0]==os.environ['RTSAFE_EXPLORER_ONLY']]
# UE camera: +X forward, +Y right, +Z up; pitch positive points up.
faces=[('px',(0,0,0)),('py',(0,90,0)),('nx',(0,180,0)),('ny',(0,-90,0)),('pz',(90,0,0)),('nz',(-90,0,0))]
out=Path(n.ROOT)/'renders/explorer';out.mkdir(parents=True,exist_ok=True)
jobs=[]
for name,loc,yaw in locations:
    cam=n.camera('Explorer_'+name,loc,faces[0][1],90)
    c=cam.get_cine_camera_component()
    film=c.get_editor_property('filmback');film.sensor_width=36;film.sensor_height=36;c.set_editor_property('filmback',film)
    c.set_editor_property('current_focal_length',18.)
    pp=c.get_editor_property('post_process_settings')
    for prop,value in [('auto_exposure_min_brightness',1.),('auto_exposure_max_brightness',1.),('auto_exposure_bias',0.),('vignette_intensity',0.),('bloom_intensity',0.),('lens_flare_intensity',0.)]:
        pp.set_editor_property('override_'+prop,True);pp.set_editor_property(prop,value)
    c.set_editor_property('post_process_settings',pp)
    camera_poses=[(i*4+j,loc,rot) for i,(_,rot) in enumerate(faces) for j in range(4)]
    seq=n.sequence('explorer_'+name,cam,camera_poses,bindings,frames)
    cfg=n.config(str(out/name),name,1536,1536,4)
    aa=cfg.find_or_add_setting_by_class(n.unreal.MoviePipelineAntiAliasingSetting);aa.engine_warm_up_count=48;aa.render_warm_up_count=32
    jobs.append({'name':name,'sequence':seq,'config':cfg})
(out/'scene.json').write_text(json.dumps({'locations':[{'id':name,'position':loc,'yaw':yaw} for name,loc,yaw in locations],'faces':[{'id':name,'rotation':rot,'frame':i*4+2} for i,(name,rot) in enumerate(faces)],'anchors':ns['anchors'],'route':ns['route'],'illustrative':True},indent=2))
n.run_jobs(jobs,str(out/'result.json'),{'native_360':True,'resolution_per_face':1536})
