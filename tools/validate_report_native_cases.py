#!/usr/bin/env python3
"""Controlled physical counter/overlap probes on an isolated UE instance."""
import argparse,json,math,os,sys,uuid
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'SimWorld')]
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from base.rt_agent import RTAgent, InitialHazardOverlapError
from base.rt_unrealcv import RTUnrealCV
from base.rt_communicator import RTCommunicator
from simworld.utils.vector import Vector
from tools.validate_report_tasks import install_scripted_policy,configure_probe_environment


def require_passing_cases(results):
    """A probe process must fail when a physical check fails, not just log it."""
    failed = [row.get('case', '<unnamed>') for row in results if row.get('passed') is not True]
    if not results or failed:
        raise RuntimeError('Native validation failed: ' + ', '.join(failed or ['no cases recorded']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--hazards-only',action='store_true');p.add_argument('--rays-only',action='store_true');p.add_argument('--contacts-only',action='store_true');p.add_argument('--initial-only',action='store_true');p.add_argument('--inference-only',action='store_true');p.add_argument('--clear-parked-cars',action='store_true');p.add_argument('--skip-vehicle',action='store_true',help='Skip the direct-car-push negative control')
    p.add_argument('--recovery-only', action='store_true')
    args=p.parse_args();configure_probe_environment();args.output.mkdir(parents=True,exist_ok=True)
    u=RTUnrealCV(port=args.port,resolution=(720,640));u.pause_simulation();comm=RTCommunicator(u)
    if args.clear_parked_cars:
        removed=[]
        for name in comm.freeze_generated_decorative_vehicles():
            point=u.get_location(name)
            if math.hypot(point[0]+4800,point[1])<1000:
                removed.append({'actor':name,'position':[float(v) for v in point]});u.destroy(name)
        (args.output/'controlled_scene_clearance.json').write_text(json.dumps(removed,indent=2)+'\n')
    assets=json.loads((ROOT/'data/ue_assets.json').read_text()); results=[]
    created=[]
    camera_samples=[]
    active_agent=[None]
    original_capture=comm.get_camera_observation
    def capture(camera,*params,**kwargs):
        a=active_agent[0]
        if a is not None:
            actor=[float(v) for v in u.get_location(a.name)]
            pose=[float(v) for v in u.get_camera_location(camera).split()]
            camera_samples.append({'actor':actor,'camera':pose,'error_cm':math.hypot(actor[0]-pose[0],actor[1]-pose[1])})
        return original_capture(camera,*params,**kwargs)
    comm.get_camera_observation=capture
    def spawn(asset,point,prefix='PROBE'):
        name=prefix+'_'+uuid.uuid4().hex[:8];u.spawn_bp_asset(asset,name);created.append(name);u.set_location(point,name);return name
    def agent(label):
        name=spawn('/Game/RealTimeBench/Agent/BP_RT_Agent.BP_RT_Agent_C',(-5000,0,110),'RT_PROBE_AGENT')
        a=RTAgent(Vector(-5000,0),Vector(1,0),Vector(10000,0),[Vector(10000,0)],100,comm,
                  SimpleNamespace(model_name='scripted-native-validation'),use_tick=True,realtime_thinking=False,
                  use_action_frames=True,record_per_step=True,record_dir=str(args.output/label),
                  task_edges=[{'node1':[-6000,0],'node2':[10000,0],'type':'sidewalk'}])
        a.name=name;a.initialize_first_person_camera();a.disable_step_evaluation=True
        install_scripted_policy(a)
        camera_samples.clear();active_agent[0]=a
        return a
    def step(a,kind,param):
        a.validation_action=(kind,param);a.step('greedy');return a.decision_trace[-1]
    def record(label,a,extra=None):
        row={'case':label,'state':comm.get_states(a.name),'vehicle_counter':comm.get_vehicle_collision_number(a.name),
             'position':[float(v) for v in u.get_location(a.name)],'camera':u.get_camera_location(a.camera_id),
             'camera_samples':list(camera_samples),'trace':a.decision_trace,'sweep_samples':getattr(a,'last_swept_move_samples',[])}
        if extra:row.update(extra)
        results.append(row);(args.output/'cases.json').write_text(json.dumps(results,indent=2,default=str)+'\n');print(json.dumps({'case':label,'state':row['state'],'extra':extra},default=str),flush=True)
    def cleanup():
        for name in created:u.destroy(name)
        created.clear();u.clean_garbage()
    try:
        if args.recovery_only:
            for kind in ['human', 'transient_oil']:
                label='recovery_'+kind;a=agent(label)
                original_plan=a.plan
                state={'ball':None,'human':None,'oil':None,'started':None,'injected':False}
                def plan(obs,mode):
                    if state['ball'] is None:
                        state['ball']=spawn(assets['RT_Basketball_C']['asset_path'],(-5035,0,2),'PROBE_PROP')
                    return original_plan(obs,mode)
                a.plan=plan
                if kind=='human':
                    state['human']=spawn('/Game/RealTimeBench/Agent/RT_Base_Pedestrian.RT_Base_Pedestrian_C',(-4700,0,110),'RT_PEDESTRIAN_PROBE')
                    u.humanoid_stop(state['human'])
                def during_recovery(delta):
                    if getattr(a,'_safety_sampling_phase',None)!='during_recovery':return
                    if state['started'] is None:state['started']=a.sim_time_elapsed
                    elapsed=a.sim_time_elapsed-state['started']
                    if not state['injected']:
                        state['injected']=True
                        if kind=='human':
                            u.move_actor_with_sweep(state['human'],(-5000,0,110))
                        else:
                            u.destroy(state['ball']);created.remove(state['ball'])
                            state['oil']=spawn(assets['RT_Oil_C']['asset_path'],(-5000,0,2),'PROBE_PROP')
                    if kind=='transient_oil' and state['oil'] is not None and elapsed>=.5:
                        u.destroy(state['oil']);created.remove(state['oil']);state['oil']=None
                a.simulation_step_callback=during_recovery
                trace=step(a,'wait','1');d=trace['collision_details']
                native_count=comm.get_states(a.name)[0]
                if kind=='human':
                    passed=bool(d.get('recovery_collision_details',{}).get('human',0)>0
                                and a.last_ue_collision_count['human']==native_count
                                and a.passive_collision_count==0)
                else:
                    passed=bool(d['oil']==1 and any(e['kind']=='oil' and e['phase']=='during_recovery' for e in d['hazard_events'])
                                and comm.get_states(a.name)[4]==0)
                record(label,a,{'passed':passed,'injected':state['injected'],'native_human_counter':native_count})
                cleanup()
            return
        if args.inference_only:
            a=agent('inference_overlap');step(a,'wait','1')
            original_plan=a.plan
            def plan(obs,mode):
                spawn(assets['RT_Basketball_C']['asset_path'],(-5035,0,2),'PROBE_PROP')
                return original_plan(obs,mode)
            a.plan=plan;a.realtime_thinking=True
            os.environ['CODE_BASELINE_THINKING_SECONDS']='0.1'
            trace=step(a,'wait','1');d=trace['collision_details']
            record('inference_overlap',a,{'passed':bool(d['fall']==1 and d['hazard_events'][0]['phase']=='during_inference')})
            return
        for choice in ([] if args.hazards_only or args.contacts_only or args.initial_only else range(1,8)):
            label=f'waypoint_{choice}';a=agent(label)
            step(a,'move_to',str(choice))
            target=a.last_commanded_waypoint
            camera=[float(v) for v in u.get_camera_location(a.camera_id).split()]
            error=a.position.distance(target)
            camera_error=math.hypot(camera[0]-a.position.x,camera[1]-a.position.y)
            record(label,a,{'target':[target.x,target.y],'endpoint_error_cm':error,
                           'camera_error_cm':camera_error,
                           'passed':error<.5 and camera_error<.5 and all(s['error_cm']<.5 for s in camera_samples)})
            cleanup()
        if args.rays_only:return
        for blocking in ([] if args.hazards_only or args.initial_only else [False,True,"parked"]):
            label='blocked_by_static_prop_near_human' if blocking=='parked' else 'human_block' if blocking else 'nearby_human_no_contact';a=agent(label)
            if blocking=='parked':
                barrier=spawn(assets['BP_Trash_bin_a_C']['asset_path'],(-4800,0,0),'PROBE_STATIC_BARRIER')
                u.set_physics(barrier,False);u.set_collision(barrier,True)
            human=spawn('/Game/RealTimeBench/Agent/RT_Base_Pedestrian.RT_Base_Pedestrian_C',((-4800,280,110) if blocking=='parked' else (-4700,0 if blocking else 310,110)),'RT_PEDESTRIAN_PROBE')
            u.humanoid_stop(human)
            trace=step(a,'move_to','5');d=trace['collision_details'];displacement=a.position.x+5000
            record(label,a,{'displacement_cm':displacement,'human_actor':human,
                           'human_distance_to_endpoint_cm':math.hypot(u.get_location(human)[0]-a.position.x,u.get_location(human)[1]-a.position.y),
                           'passed':bool(d['human']==0 and a.position.distance(a.last_commanded_waypoint)>25 and math.hypot(u.get_location(human)[0]-a.position.x,u.get_location(human)[1]-a.position.y)<325) if blocking=='parked' else bool(d['human']>0 and displacement<400) if blocking else bool(d['human']==0 and abs(displacement-400)<.5)})
            cleanup()
        if args.contacts_only:return
        for kind in ([] if args.initial_only else ['RT_Basketball_C','RT_Bottle1_C','RT_Oil_C','RT_WaterPuddle_C']):
            label='path_'+kind;a=agent(label)
            prop=spawn(assets[kind]['asset_path'],(-4800,0,2),'PROBE_PROP')
            start_state=comm.get_states(a.name)
            trace=step(a,'move_to','5');events=trace['collision_details'].get('hazard_events',[])
            after=comm.get_states(a.name);prop_pose=u.get_location(prop)
            wait=step(a,'wait','1');turn=step(a,'turn_around','L30')
            record(label,a,{'start_state':start_state,'post_move_state':after,'prop_pose':prop_pose,
                           'hazard_events_in_move':events,'later_fall_count':sum(t['collision_details'].get('fall',0) for t in [wait,turn]),
                           'contact_observed':bool(events),'passed':bool(events)})
            cleanup()
        a=agent('initial_overlap_rejection')
        prop=spawn(assets['RT_Basketball_C']['asset_path'],(-5035,0,2),'PROBE_PROP')
        # Allow physics to establish the injected trip contact. Do not poll
        # GetStates here: the native trip flag can disappear on the next read.
        u.advance_simulation_time(.5, 1.)
        observed=[];original_states=comm.get_states
        def logged_states(name):
            state=original_states(name);observed.append(state);return state
        comm.get_states=logged_states;rejected=False
        try:step(a,'wait','1')
        except InitialHazardOverlapError:rejected=True
        finally:comm.get_states=original_states
        record('initial_overlap_rejection',a,{'initial_state':observed[0],
               'setup_requested_sim_seconds':.5,'setup_state_reads':0,'rejected':rejected,
               'fall_count':a.fall_count,'passed':rejected})
        cleanup()
        if args.initial_only or args.skip_vehicle:return
        # Native vehicle collision plus an oil overlap, both created physically.
        a=agent('vehicle_and_hazard');step(a,'wait','1')
        vehicle=spawn('/Game/TrafficSystem/Vehicle/Vehicle1.Vehicle1_C',(-5300,0,30),'RT_SIGNAL_VEHICLE_PROBE')
        u.set_collision(vehicle,True);a.red_light_conflict_active=True
        prop=spawn(assets['RT_Oil_C']['asset_path'],(-5000,0,2),'PROBE_PROP')
        before=comm.get_vehicle_collision_number(a.name)
        u.move_actor_with_sweep(vehicle,(-5000,0,30))
        trace=step(a,'wait','1');d=trace['collision_details']
        record('vehicle_and_hazard',a,{'vehicle_actor':vehicle,'vehicle_counter_before':before,
               'passed':bool(d.get('vehicle',0)>0 and d.get('oil',0)>0),'failed':a.failed})
    finally:
        cleanup();u.client.disconnect()
        if sys.exc_info()[0] is None:
            require_passing_cases(results)

if __name__=='__main__':main()
