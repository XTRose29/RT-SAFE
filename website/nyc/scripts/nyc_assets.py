"""Native scene actors. Asset edits are confined to /Game/RTSafeNYC."""
import math,os
import unreal
import nyc_scene as n

MALE='/Game/Character/Player/Male/Meshes/SKM_PlayerMale'
FEMALE='/Game/Character/Player/Female/Meshes/SKM_PlayerFemale_Body'
MALE_WALK='/Game/Crowd/Character/Anims/Loco/MTN_N_Walk_F'
FEMALE_WALK='/Game/Character/Player/Female/Anims/Locomotion/FP_Walk_F'
# Native full-stride cycles. Root tracks are fixed in unsaved copies.
SCOOTER='/Game/CityAssetsKit/Assets/MM00130_Scooter/SM_MM00130_Scooter_fbx'
HYDRANT='/Game/CityAssetsKit/Assets/MM00281_Fire_Hydrant/SM_MM00281_Fire_Hydrant'
TRASH='/Game/CityAssetsKit/Assets/MM00289_Street_Trash/SM_MM00289-Trashbag_01_fbx'
CAR='/Game/Vehicle/vehCar_vehicle07/Mesh/SM_vehCar_vehicle07'

def pedestrian(name,index=0,position=n.START):
    actor=n.skeletal(name,FEMALE if index%2 else MALE,position,scale=1)
    if index%2:
        body=actor.get_component_by_class(unreal.SkeletalMeshComponent)
        head_actor=n.skeletal(name+'_Head',FEMALE.replace('_Body','_Head'),position)
        head_actor.attach_to_actor(actor,'',unreal.AttachmentRule.KEEP_WORLD,unreal.AttachmentRule.KEEP_WORLD,unreal.AttachmentRule.KEEP_WORLD,False)
        head=head_actor.get_component_by_class(unreal.SkeletalMeshComponent)
        head.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        head.set_leader_pose_component(body,True,False)
    return actor,FEMALE_WALK if index%2 else MALE_WALK

def prop(name,path,position,yaw=0,scale=(1,1,1)):
    mesh=unreal.load_asset(path)
    if not mesh:raise RuntimeError('Missing native prop '+path)
    actor=n.actors().spawn_actor_from_class(unreal.StaticMeshActor,n.vec(position),n.rot((0,yaw,0)))
    actor.set_actor_label('RTSafeNYC_'+name);actor.set_actor_scale3d(n.vec(scale))
    c=actor.get_component_by_class(unreal.StaticMeshComponent)
    c.set_editor_property('mobility',unreal.ComponentMobility.MOVABLE);c.set_static_mesh(mesh)
    c.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION);actor.set_actor_enable_collision(False)
    return actor

def dog(name,position):
    cls=unreal.load_class(None,'/Game/Agent/Robot/Core/BP_Go1Robot.BP_Go1Robot_C')
    if not cls:raise RuntimeError('Missing native Go1 robot')
    actor=n.actors().spawn_actor_from_class(cls,n.vec(position),n.rot((0,n.YAW,0)))
    actor.set_actor_label('RTSafeNYC_'+name);actor.set_actor_enable_collision(False);actor.set_actor_tick_enabled(False)
    material=unreal.load_asset('/Game/Agent/Robot/go1/M_Go1_Body')
    for c in actor.get_components_by_class(unreal.StaticMeshComponent):
        c.set_editor_property('mobility',unreal.ComponentMobility.MOVABLE)
        c.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        if material:
            for slot in range(c.get_num_materials()):c.set_material(slot,material)
    return actor

def dog_gait(actor,times,rate=1,phase=0):
    """Presentation gait on the native Go1 links; source XY movement is unchanged."""
    tracks=[]
    for component in actor.get_components_by_class(unreal.SceneComponent):
        name=component.get_name()
        if not (name.startswith('MjBody_') and name.endswith(('_thigh','_calf'))):continue
        component.set_editor_property('mobility',unreal.ComponentMobility.MOVABLE)
        leg=name.split('_')[1];front=leg.startswith('F');left=leg.endswith('L')
        hip=(18.8 if front else -18.8,-12.7 if left else 12.7,27)
        offset=0 if leg in ('FL','RR') else math.pi
        poses=[]
        for f,t in enumerate(times):
            cycle=math.sin(t*rate*1.4*2*math.pi+offset+phase)
            upper=30+18*cycle;lower=-30+18*cycle
            if name.endswith('_thigh'):loc=hip;pitch=upper
            else:
                theta=math.radians(upper)
                loc=(hip[0]+21*math.sin(theta),hip[1],hip[2]-21*math.cos(theta));pitch=lower
            poses.append((f,loc,(pitch,0,0)))
        tracks.append((component,poses,None))
    return tracks

def material(name,color,roughness=.6,metallic=0):
    """A dynamic parameter instance leaves the loaded city materials untouched."""
    import builtins
    key=tuple(color)+(roughness,metallic)
    cache=getattr(builtins,'_rtsafe_nyc_palette',{})
    if key in cache:return cache[key]
    parent=unreal.load_asset('/Engine/BasicShapes/BasicShapeMaterial')
    mat=unreal.MaterialLibrary.create_dynamic_material_instance(n.world(),parent)
    mat.set_vector_parameter_value('Color',unreal.LinearColor(*color,1))
    mat.set_scalar_parameter_value('Roughness',roughness)
    mat.set_scalar_parameter_value('Metallic',metallic)
    cache[key]=mat;builtins._rtsafe_nyc_palette=cache;n.keep(mat)
    return mat

def patch(name,position,color,roughness=.1,size=(1.7,1.1),metallic=.15):
    """A shallow irregular wet-surface patch made from native geometry."""
    mat=material('M_'+name,color,roughness,metallic)
    parent=None
    for i,(dx,dy,sx,sy) in enumerate([(0,0,1,1),(42,14,.75,.65),(-42,-8,.65,.8),(8,38,.55,.65)]):
        actor=prop(name+str(i),'/Engine/BasicShapes/Cylinder',(position[0]+dx,position[1]+dy,position[2]+.6+i*.06),23+i*31,(size[0]*sx,size[1]*sy,.008))
        actor.get_component_by_class(unreal.StaticMeshComponent).set_material(0,mat)
        if parent is None:parent=actor
    return parent

def ground_at(x,y,fallback=n.START[2]):
    try:
        hit=unreal.SystemLibrary.line_trace_single(n.world(),n.vec((x,y,fallback+1800)),n.vec((x,y,fallback-3000)),unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,True,[],unreal.DrawDebugTrace.NONE,True)
        if hit and hit.to_tuple()[0]:
            z=float(hit.to_tuple()[4].z)
            if abs(z-fallback)<500:return z
    except Exception:
        pass
    return fallback

def colored_prop(name,path,position,color,scale=(1,1,1),roughness=.65):
    actor=prop(name,path,position,scale=scale)
    actor.get_component_by_class(unreal.StaticMeshComponent).set_material(0,material('M_'+name,color,roughness))
    return actor

def source_object(name,kind,position):
    """Type-matched native objects at recorded terminal XY locations.

    Source physics trajectories were not recorded; these remain stationary.
    """
    x,y,z=position;k=kind.lower()
    if 'bottle' in k:
        return prop(name,'/Game/Downtown_West/Assets/props/prop_food/SM_bottle_seasaltsoda',(x,y,z),yaw=22)
    if 'water' in k:return patch(name,(x,y,z),(.035,.12,.16),.08,(.6,.45),.35)
    if 'box' in k:
        box=colored_prop(name,'/Engine/BasicShapes/Cube',(x,y,z+16),(.38,.22,.10),(.40,.36,.32))
        colored_prop(name+'_Tape','/Engine/BasicShapes/Cube',(x,y,z+32.2),(.61,.47,.28),(.06,.363,.007))
        return box
    radius=3.4 if 'tennis' in k else 12 if 'basket' in k else 11
    color=(.45,.62,.025) if 'tennis' in k else (.52,.12,.015) if 'basket' in k else (.7,.73,.76)
    return colored_prop(name,'/Engine/BasicShapes/Sphere',(x,y,z+radius),color,(radius/50,)*3,.62)

def vehicle(name,position,yaw=n.YAW):
    actor=prop(name,CAR,position,yaw=yaw)
    glass=prop(name+'_Glass','/Game/Vehicle/vehCar_vehicle07/Mesh/Transparent/SM_All_Trans_vehCar_vehicle07',position,yaw=yaw)
    glass.attach_to_actor(actor,'',unreal.AttachmentRule.KEEP_WORLD,unreal.AttachmentRule.KEEP_WORLD,unreal.AttachmentRule.KEEP_WORLD,False)
    center,extent=actor.get_actor_bounds(False)
    lift=position[2]-(center.z-extent.z)
    actor.set_actor_location(n.vec((position[0],position[1],position[2]+lift)),False,False)
    return actor,lift


def walk_rate(animation_path, speed_cm_s):
    """Match cycle speed to the authored path after removing root translation."""
    native_speed = 190.7 if animation_path == FEMALE_WALK else 140.0
    return abs(speed_cm_s) / native_speed
