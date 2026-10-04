"""Reusable native Unreal / Movie Render Queue helpers for the RT-Safe NYC film."""
import json, math, os, time, builtins
import unreal

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
START = (-14392.951180142312, 20740.030905653883, -218.69718170166016)
YAW = 28.338361740112305
MANNY = '/Game/Interactable_InteractionKitVol3/Demo/Characters/Mannequins/Meshes/SKM_Manny_Simple'
IDLE = '/Game/Interactable_InteractionKitVol3/Demo/Characters/Mannequins/Animations/Manny/MM_Idle'
WALK = '/Game/Interactable_InteractionKitVol3/Demo/Characters/Mannequins/Animations/Manny/MM_Walk_InPlace'

def keep(asset):
    if not hasattr(builtins,'_rtsafe_nyc_assets'):builtins._rtsafe_nyc_assets=[]
    builtins._rtsafe_nyc_assets.append(asset)
    return asset

def vec(p): return unreal.Vector(*map(float,p))
def rot(p): return unreal.Rotator(float(p[2]),float(p[0]),float(p[1]))
def world(): return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
def actors(): return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
def along(forward, lateral=0, height=0):
    a=math.radians(YAW);x,y,z=START
    return (x+math.cos(a)*forward-math.sin(a)*lateral,y+math.sin(a)*forward+math.cos(a)*lateral,z+height)
def look_at(origin,target):
    d=[b-a for a,b in zip(origin,target)]
    return (math.degrees(math.atan2(d[2],math.hypot(d[0],d[1]))),math.degrees(math.atan2(d[1],d[0])),0)
def cleanup():
    for actor in getattr(builtins,'_rtsafe_nyc_hidden_furniture',[]):
        actor.set_actor_hidden_in_game(False);actor.set_is_temporarily_hidden_in_editor(False)
    builtins._rtsafe_nyc_hidden_furniture=[]
    for actor in actors().get_all_level_actors():
        if actor.get_actor_label().startswith('RTSafeNYC_') or isinstance(actor,unreal.LevelSequenceActor):
            actor.destroy_actor()
def controlled_world(enabled=True):
    """Use explicit foreground trajectories for the two matched comparisons."""
    for actor in actors().get_all_level_actors():
        if actor.get_actor_label() in ('MassTrafficVehicleSpawner','MassCrowdSpawner'):
            actor.set_editor_property('auto_spawn_on_begin_play',not enabled)
def clear_replay_corridor():
    """Keep NYC background poles from introducing new obstacles into source poses."""
    hidden=[]
    for actor in actors().get_all_level_actors():
        if actor.get_actor_label() in ('SW_RoadProp_lamp_59','SW_RoadProp_lamp_737'):
            actor.set_actor_hidden_in_game(True);actor.set_is_temporarily_hidden_in_editor(True)
            hidden.append(actor)
    builtins._rtsafe_nyc_hidden_furniture=hidden
def skeletal(label,mesh_path=MANNY,position=START,yaw=YAW-90,scale=1):
    unreal.log_warning('RTSAFE_NYC loading mesh '+mesh_path)
    mesh=unreal.load_asset(mesh_path)
    if mesh is None: raise RuntimeError('Missing mesh '+mesh_path)
    actor=actors().spawn_actor_from_class(unreal.SkeletalMeshActor,vec(position),rot((0,yaw,0)))
    actor.set_actor_label('RTSafeNYC_'+label)
    actor.set_actor_scale3d(vec((scale,scale,scale)))
    c=actor.get_component_by_class(unreal.SkeletalMeshComponent)
    c.set_skeletal_mesh_asset(mesh)
    c.set_editor_property('mobility',unreal.ComponentMobility.MOVABLE)
    c.set_editor_property('visibility_based_anim_tick_option',unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
    c.set_editor_property('bounds_scale',4)
    c.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    actor.set_actor_enable_collision(False)
    return actor
def key_transform(binding,poses):
    section=binding.add_track(unreal.MovieScene3DTransformTrack).add_section()
    section.set_start_frame_bounded(False);section.set_end_frame_bounded(False)
    channels=section.get_all_channels()
    for frame,loc,rotation in poses:
        values=list(loc)+[rotation[2],rotation[0],rotation[1]]
        for c,v in zip(channels,values): c.add_key(unreal.FrameNumber(frame),float(v),interpolation=unreal.MovieSceneKeyInterpolation.LINEAR)
def in_place_animation(path):
    # The UE 5.8 preview Sequencer mixer can bypass Force Root Lock.
    # https://issues.unrealengine.com/issue/UE-386453
    # Strip translation from a separate, unsaved animation copy instead.
    # Original marketplace assets are never edited or saved.
    import hashlib
    if not hasattr(builtins,'_rtsafe_in_place'):builtins._rtsafe_in_place={}
    cache=builtins._rtsafe_in_place
    if path in cache:return cache[path]
    source=unreal.load_asset(path)
    if not isinstance(source,unreal.AnimSequence):return source
    name='InPlace_'+hashlib.sha1(path.encode()).hexdigest()[:10]+'_'+str(int(time.time()*1000))
    anim=unreal.AssetToolsHelpers.get_asset_tools().duplicate_asset(name,'/Game/RTSafeNYC',source)
    model=anim.get_editor_property('data_model_interface')
    controller=anim.get_editor_property('controller')
    if model.is_valid_bone_track_name('root'):
        count=model.get_number_of_keys()
        first=unreal.AnimationLibrary.get_bone_pose_for_time(source,'root',0,False)
        controller.open_bracket('Create in-place presentation copy',False)
        ok=controller.set_bone_track_keys('root',[first.translation]*count,[first.rotation]*count,[first.scale3d]*count,False)
        controller.close_bracket(False)
        assert ok, 'Could not lock the copied root track: '+path
    anim.set_editor_property('enable_root_motion',False)
    anim.set_editor_property('force_root_lock',True)
    keep(anim);cache[path]=anim
    return anim

def animation(binding,path,start,end,rate=1,offset=0):
    anim=in_place_animation(path)
    if not anim:return
    tracks=[t for t in binding.get_tracks() if isinstance(t,unreal.MovieSceneSkeletalAnimationTrack)]
    track=tracks[0] if tracks else binding.add_track(unreal.MovieSceneSkeletalAnimationTrack)
    section=track.add_section()
    params=section.get_editor_property('params');params.set_editor_property('animation',anim)
    params.set_editor_property('play_rate',unreal.MovieSceneTimeWarpExtensions.make_time_warp(float(rate)))
    params.set_editor_property('start_frame_offset',unreal.FrameNumber(offset))
    section.set_editor_property('params',params)
    # Unreal resizes the section when PlayRate changes. Restore authored bounds last.
    section.set_range(start,end)
    assert section.get_start_frame()==start and section.get_end_frame()==end
def camera(name,location,rotation,fov=68):
    actor=actors().spawn_actor_from_class(unreal.CineCameraActor,vec(location),rot(rotation))
    actor.set_actor_label('RTSafeNYC_'+name)
    c=actor.get_cine_camera_component()
    c.set_editor_property('current_focal_length',float(c.filmback.sensor_width)/(2*math.tan(math.radians(fov/2))))
    focus=c.get_editor_property('focus_settings');focus.set_editor_property('focus_method',unreal.CameraFocusMethod.DISABLE);c.set_editor_property('focus_settings',focus)
    return actor
def sequence(name,camera_actor,camera_poses,bindings,frames=2,fps=30):
    name=name+'_'+str(int(time.time()*1000))
    package='/Game/RTSafeNYC';path=package+'/'+name
    seq=unreal.AssetToolsHelpers.get_asset_tools().create_asset(name,package,unreal.LevelSequence,unreal.LevelSequenceFactoryNew())
    seq.set_display_rate(unreal.FrameRate(fps,1));seq.set_playback_start(0);seq.set_playback_end(frames)
    cb=seq.add_possessable(camera_actor);key_transform(cb,camera_poses)
    bound_actors={}
    for actor,poses,anim in bindings:
        b=seq.add_possessable(actor)
        if isinstance(actor,unreal.ActorComponent):
            parent=bound_actors.get(actor.get_owner().get_path_name())
            if parent is not None:b.set_parent(parent)
        else:bound_actors[actor.get_path_name()]=b
        key_transform(b,poses)
        if isinstance(anim,list):
            for entry in anim:animation(b,*entry)
        elif anim:animation(b,anim,0,frames)
    cut=seq.add_track(unreal.MovieSceneCameraCutTrack).add_section();cut.set_range(0,frames)
    cut.set_camera_binding_id(unreal.MovieSceneSequenceExtensions.get_binding_id(seq,cb))
    keep(seq)  # PIE resolves this registered asset in memory; no global library scan.
    return path+'.'+name
def config(output,stem,width=1920,height=1080,spatial=8,temporal=1,quality='raster'):
    os.makedirs(output,exist_ok=True)
    cfg=unreal.MoviePipelinePrimaryConfig()
    cfg.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    cfg.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    o=cfg.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    o.output_directory=unreal.DirectoryPath(output);o.output_resolution=unreal.IntPoint(width,height)
    o.file_name_format=stem+'_{frame_number}';o.override_existing_output=True
    aa=cfg.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.spatial_sample_count=spatial;aa.temporal_sample_count=temporal;aa.override_anti_aliasing=True
    aa.anti_aliasing_method=unreal.AntiAliasingMethod.AAM_NONE
    aa.engine_warm_up_count=64;aa.render_warm_up_count=64
    c=cfg.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    c.start_console_commands=['DisableAllScreenMessages','stat none','showflag.OnScreenDebug 0',
        'r.MotionBlurQuality 0','r.MotionBlur.Amount 0','r.Nanite 0','r.Shadow.Virtual.Enable 0',
        'r.DynamicGlobalIlluminationMethod 0','r.ReflectionMethod 0','r.Lumen.DiffuseIndirect.Allow 0',
        'r.Lumen.Reflections.Allow 0','r.ScreenPercentage 100','r.Streaming.LimitPoolSizeToVRAM 1',
        'r.Shadow.MaxResolution 4096','r.MaxAnisotropy 16','r.Tonemapper.Sharpen 0.4']
    if quality=='ssgi':c.start_console_commands=list(c.start_console_commands)+['r.DynamicGlobalIlluminationMethod 2','r.ReflectionMethod 2','r.SSGI.Quality 4','r.SSR.Quality 4']
    return cfg
def run_jobs(jobs,done,metadata=None):
    subsystem=unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    state={'index':0,'active':None,'wait':0,'results':[],'started':time.time(),'jobs':jobs,'subsystem':subsystem}
    # Keep all UObjects alive until the asynchronous queue has completed.
    import builtins
    builtins._rtsafe_nyc_render_state=state
    def tick(delta):
        if state['active'] is not None:return
        if state['wait']>0:state['wait']-=1;return
        if state['index']>=len(jobs):
            unreal.unregister_slate_post_tick_callback(state['handle'])
            payload={'success':all(x['success'] for x in state['results']),'jobs':state['results'],'elapsed':time.time()-state['started'],'metadata':metadata}
            with open(done,'w') as f:json.dump(payload,f,indent=2)
            unreal.log_warning('RTSAFE_NYC completed '+done);return
        spec=jobs[state['index']];state['index']+=1
        console=spec['config'].find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
        remaining=[]
        for command in console.start_console_commands:
            parts=command.split()
            try:
                target=float(parts[1])
                if not parts[0].startswith('r.'):raise ValueError()
                current=unreal.SystemLibrary.get_console_variable_float_value(parts[0])
                if abs(current-target)>1e-5:unreal.SystemLibrary.execute_console_command(world(),command)
            except (ValueError,IndexError):remaining.append(command)
        console.start_console_commands=remaining
        queue=subsystem.get_queue()
        for job in list(queue.get_jobs()):queue.delete_job(job)
        job=queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.map=unreal.SoftObjectPath(world().get_path_name().split('.')[0])
        job.sequence=unreal.SoftObjectPath(spec['sequence']);job.set_configuration(spec['config'])
        executor=unreal.MoviePipelinePIEExecutor();state['active']=executor;start=time.time()
        def finished(ex,success):
            state['results'].append({'name':spec['name'],'success':bool(success),'seconds':time.time()-start})
            state['active']=None;state['wait']=60
            unreal.log_warning('RTSAFE_NYC finished '+spec['name']+' '+str(success))
        executor.on_executor_finished_delegate.add_callable(finished)
        unreal.log_warning('RTSAFE_NYC starting '+spec['name'])
        subsystem.render_queue_with_executor_instance(executor)
    state['handle']=unreal.register_slate_post_tick_callback(tick)
    return state
