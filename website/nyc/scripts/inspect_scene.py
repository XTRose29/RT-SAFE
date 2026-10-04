import sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nyc_scene as n
import unreal
rows=[];lights=[];spawners=[]
for actor in n.actors().get_all_level_actors():
    loc=actor.get_actor_location();p=(loc.x,loc.y,loc.z)
    if 'spawner' in actor.get_class().get_name().lower() or 'spawner' in actor.get_actor_label().lower():
        spawners.append({'name':actor.get_actor_label(),'class':actor.get_class().get_name(),'location':p})
    if math.dist(p[:2],n.START[:2])<6500:
        center,extent=actor.get_actor_bounds(False)
        rows.append({'name':actor.get_actor_label(),'class':actor.get_class().get_name(),'location':p,
            'bounds_center':(center.x,center.y,center.z),'bounds_extent':(extent.x,extent.y,extent.z)})
    for c in actor.get_components_by_class(unreal.LightComponent):
        item={'actor':actor.get_actor_label(),'class':c.get_class().get_name(),'location':p}
        for prop in ('intensity','light_color','indirect_lighting_intensity','volumetric_scattering_intensity'):
            try:item[prop]=str(c.get_editor_property(prop))
            except:pass
        lights.append(item)
with open(n.ROOT+'/evidence/scene-inspection.json','w') as f:json.dump({'near_route':rows,'lights':lights,'spawners':spawners},f,indent=2)
unreal.log_warning('RTSAFE_NYC inspected '+str(len(rows))+' nearby actors')
