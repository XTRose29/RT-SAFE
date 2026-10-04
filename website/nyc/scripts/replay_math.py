"""Source-space replay timing; independent of Unreal and rendering quality."""
import math

def lerp(a,b,p): return a+(b-a)*p
def angle_lerp(a,b,p): return a+((b-a+180)%360-180)*p

def sample_agent(model,elapsed_sim_seconds):
    """Hold the agent through inference while the world clock advances normally."""
    steps=model['steps'];start=steps[0]['start_sim_seconds'];end=steps[-1]['end_sim_seconds']
    source_time=min(end,max(start,start+elapsed_sim_seconds))
    step=steps[-1]
    for candidate in steps:
        if source_time<candidate['end_sim_seconds']:
            step=candidate;break
    inference_end=min(step['end_sim_seconds'],step['start_sim_seconds']+step['inference_exposure_seconds'])
    action_span=max(1e-9,step['end_sim_seconds']-inference_end)
    progress=min(1,max(0,(source_time-inference_end)/action_span))
    position=[lerp(a,b,progress) for a,b in zip(step['position_cm'],step['post_action_position_cm'])]
    yaw=angle_lerp(step['yaw_degrees'],step['post_action_yaw_degrees'],progress)
    active=source_time>=inference_end
    moving=active and math.dist(step['position_cm'],step['post_action_position_cm'])>1
    return {'source_sim_seconds':source_time,'source_step':step['source_step'],
        'decision':steps.index(step)+1,'phase':'action' if active else 'inference',
        'position_cm':position,'yaw_degrees':yaw,'moving':moving,'finished':source_time>=end}

def transform_position(position,origin,source_origin=(20700,700),source_yaw=90,world_yaw=28.338361740112305):
    """One rigid transform preserves distances for agents and all foreground actors."""
    a=math.radians(world_yaw-source_yaw);dx=position[0]-source_origin[0];dy=position[1]-source_origin[1]
    return (origin[0]+dx*math.cos(a)-dy*math.sin(a),origin[1]+dx*math.sin(a)+dy*math.cos(a))

def sample_patrol(actor,source_sim_seconds):
    """Advance the authored patrol: initial approach once, then a closed waypoint loop.

    These are reconstructed authored paths, not measured per-frame rigid-body poses.
    """
    initial=actor['initial_position_cm'];points=actor['waypoints_cm']
    if not points:return (*initial,0)
    progress=max(0,source_sim_seconds)*actor['speed_cm_s']
    lead=math.dist(initial,points[0])
    if progress<=lead and lead>1e-8:
        a,b=initial,points[0];p=progress/lead
        return lerp(a[0],b[0],p),lerp(a[1],b[1],p),math.degrees(math.atan2(b[1]-a[1],b[0]-a[0]))
    progress=max(0,progress-lead)
    segments=[(a,b,math.dist(a,b)) for a,b in zip(points,points[1:]+points[:1]) if math.dist(a,b)>1e-8]
    length=sum(s[2] for s in segments)
    if not length:return (*points[0],0)
    progress%=length
    for a,b,d in segments:
        if progress<=d:
            p=progress/d
            return lerp(a[0],b[0],p),lerp(a[1],b[1],p),math.degrees(math.atan2(b[1]-a[1],b[0]-a[0]))
        progress-=d
    return (*points[0],0)

def counts_reported_by(model,elapsed_sim_seconds):
    """Report completed phase counts; logs do not identify exact within-phase contact times."""
    t=model['steps'][0]['start_sim_seconds']+elapsed_sim_seconds
    active=passive=0
    for step in model['steps']:
        inference_end=step['start_sim_seconds']+step['inference_exposure_seconds']
        if t>=inference_end:passive+=sum(step['passive_collisions'].values())
        if t>=step['end_sim_seconds']:active+=sum(step['active_collisions'].values())
    return {'active':active,'passive':passive,'total':active+passive}

def validate_replay(payload,choreography):
    for name,model in payload['models'].items():
        first=model['steps'][0];dt=first['inference_exposure_seconds']
        a=sample_agent(model,dt*.2);b=sample_agent(model,dt*.8)
        assert a['position_cm']==b['position_cm']
        assert b['source_sim_seconds']>a['source_sim_seconds']
        finish=sample_agent(model,model['summary']['source_duration_seconds']+1)
        assert finish['finished']
        assert math.dist(finish['position_cm'],model['steps'][-1]['post_action_position_cm'])<1e-5
        total=counts_reported_by(model,model['summary']['source_duration_seconds']+1)
        assert total['total']==model['summary']['collisions']
        for step in model['steps']:
            p,q=step['position_cm'],step['post_action_position_cm']
            assert abs(math.dist(p,q)-math.dist(transform_position(p,(0,0)),transform_position(q,(0,0))))<1e-8
    for actor in choreography['actors']:
        assert math.dist(sample_patrol(actor,0)[:2],actor['initial_position_cm'])<1e-8
    return {'source_metrics':'passed','world_clock_advances_during_inference':'passed',
        'agent_holds_during_inference':'passed','rigid_transform_preserves_action_distances':'passed',
        'source_initial_actor_positions':'passed','last_agent_positions':'passed'}

if __name__=='__main__':
    from pathlib import Path
    import json
    root=Path(__file__).resolve().parents[1]
    result=validate_replay(json.loads((root/'evidence/task19-replay.json').read_text()),json.loads((root/'evidence/task19-choreography.json').read_text()))
    (root/'evidence/replay-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
