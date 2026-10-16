#!/usr/bin/env python3
"""Real signal-entry and launched-vehicle probe in a disposable map2 task scene."""
import argparse,json,math,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'SimWorld')]
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from tools.validate_report_tasks import prepare,install_scripted_policy,configure_probe_environment
from base.rt_unrealcv import RTUnrealCV
from simworld.utils.vector import Vector


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();configure_probe_environment();args.output.mkdir(parents=True,exist_ok=True)
    u=RTUnrealCV(port=args.port,resolution=(720,640));u.pause_simulation();r={'passed':False}
    try:
        m=prepare(u,args.output/'scene',task_id=14);a=m.agent;install_scripted_policy(a)
        r['restored_hazard_props']=m.restored_hazard_props
        # Controlled initial condition at task14's actual crossing start.
        crossing=a.route_crosswalks[0]
        delta=crossing.end-crossing.start
        yaw=math.degrees(math.atan2(delta.y,delta.x))
        u.humanoid_stop(a.name);u.set_location((crossing.start.x,crossing.start.y,110),a.name)
        u.set_orientation((0,a._route_yaw_to_ue_yaw(yaw),0),a.name)
        a.sync_ue();r['crosswalks']=[{'id':c.id,'start':[c.start.x,c.start.y],'end':[c.end.x,c.end.y]} for c in a.route_crosswalks]
        # Keep the real signal heads at DON'T WALK during the entry probe.
        # Physics still advances; only the automatic signal schedule is held.
        callback=a.simulation_step_callback
        a.simulation_step_callback=None
        for signals in a.crosswalk_signal_groups.values():
            for signal in signals:m.communicator.set_traffic_signal_pedestrian_stop(signal.id)
        r['before_snapshot']=a._get_traffic_light_snapshot()
        a.validation_action=('move_to','5');a.step('greedy')
        r['entry_step']=a.decision_trace[-1];r['entry_events']=[dict(e) for e in a.red_light_violation_events]
        r['after_snapshot']=a._get_traffic_light_snapshot()
        a.validation_action=('wait','1');a.step('greedy');r['next_step']=a.decision_trace[-1]
        r['red_light_count_after_next_step']=a.red_light_violations_count
        r['red_light_passed']=bool(len(r['entry_events'])==1 and a.red_light_violations_count==1)
        # Launch through the production consequence controller. This is an
        # explicit trigger in a controlled test, not a fabricated collision.
        a.simulation_step_callback=callback
        m.red_light_conflict_vehicle_enabled=True;m.red_light_conflict_vehicle_probability=1.
        event=dict(r['entry_events'][0]) if r['entry_events'] else None
        if event:
            m._handle_red_light_violation(event,a.route_crosswalks[0])
            r['launch_events']=m.red_light_conflict_vehicle_events
            asset=json.loads((ROOT/'data/ue_assets.json').read_text())['RT_Oil_C']['asset_path']
            u.spawn_bp_asset(asset,'REPORT_SIMULTANEOUS_OIL');u.set_location((a.position.x,a.position.y,2),'REPORT_SIMULTANEOUS_OIL')
            r['vehicle_wait_steps']=[]
            for i in range(15):
                if a.failed:break
                a.validation_action=('wait','3' if i==0 else '1');a.step('greedy');m._record_ue_vehicle_collision_outcome()
                r['vehicle_wait_steps'].append(a.decision_trace[-1])
                print(json.dumps({'wait':i,'failed':a.failed,'vehicle':a.vehicle_collision_count,'oil':a.oil_count,'position':[a.position.x,a.position.y]}),flush=True)
            r['vehicle_events']=m.red_light_conflict_vehicle_events
            r['vehicle_failed']=a.failed and a.failure_reason=='vehicle_collision'
            r['simultaneous_vehicle_oil']=any(t['collision_details'].get('vehicle',0)>0 and t['collision_details'].get('oil',0)>0 for t in r['vehicle_wait_steps'])
            u.destroy('REPORT_SIMULTANEOUS_OIL')
        r['passed']=r.get('red_light_passed',False) and r.get('simultaneous_vehicle_oil',False)
    finally:
        (args.output/'crossing.json').write_text(json.dumps(r,indent=2,default=str)+'\n');u.client.disconnect()
    print(json.dumps({'passed':r['passed'],'red_light_passed':r.get('red_light_passed'),'vehicle_failed':r.get('vehicle_failed'),'simultaneous_vehicle_oil':r.get('simultaneous_vehicle_oil')}),flush=True)

if __name__=='__main__':main()
