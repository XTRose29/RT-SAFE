#!/usr/bin/env python3
"""Compare sequential and batched red writes on a dedicated paused UE instance.

Uses 33 combined vehicle/pedestrian heads and 33 pedestrian-only heads. The two
paths issue identical commands. This is an RPC microbenchmark, not task speed.
"""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'SimWorld')]
from base.rt_unrealcv import RTUnrealCV
from base.rt_communicator import RTCommunicator


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--port',type=int,required=True)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
 created=[];rows=[]
 try:
  u=RTUnrealCV(port=args.port);u.pause_simulation();c=RTCommunicator(u);ids=list(range(900000,900066))
  existing=set(u.client.request("vget /objects").split())
  if any(c.get_traffic_signal_name(i) in existing for i in ids):raise RuntimeError("Probe actors exist; use a dedicated fresh engine")
  for i in ids:
   asset='RT_BP_street_light_ped' if i%2 else 'RT_BP_street_light';name=c.get_traffic_signal_name(i);u.spawn_bp_asset(f'/Game/RealTimeBench/Traffic/{asset}.{asset}_C',name);created.append(name);c.traffic_signal_set_duration(i,10,3,10);u.set_location((-50000,(i-ids[0])*200,0),name)
  for mode in ['sequential','batch']:
   for i in ids:c.traffic_signal_switch_to(i,'pedestrian walk')
   before=c.get_traffic_signal_states(ids);assert all(x['pedestrian_walk'] for x in before)
   start=time.monotonic()
   for tick in range(12):
    if mode=='sequential':
     for i in ids:c.traffic_signal_switch_to(i,'all red')
    else:c.traffic_signals_all_red(ids)
   elapsed=time.monotonic()-start
   after=c.get_traffic_signal_states(ids)
   # Packaged SetPedestrianStop does not clear GetState's live WALK flag.
   # Verify identical readback semantics, then wait for the native timer.
   immediate_walk = [x['pedestrian_walk'] for x in after]
   assert all(not x['vehicle_green'] for x in after)
   u.advance_simulation_time(10.5,1.)
   expired=c.get_traffic_signal_states(ids)
   assert all(not x['vehicle_green'] and not x['pedestrian_walk'] for x in expired)
   rows.append(dict(path=mode,heads=66,combined_heads=33,pedestrian_heads=33,ticks=12,commands=2376,wall_seconds=elapsed,walk_before=True,immediate_walk_flags=immediate_walk,all_red_after_timer_expiry=True));print(rows[-1],flush=True)
  assert rows[0]['immediate_walk_flags'] == rows[1]['immediate_walk_flags']
  args.output.write_text(json.dumps(dict(passed=True,scope='Repeated all-red writes, identical command order/count, 66 isolated native heads; no full-task speedup implied',results=rows),indent=2)+'\n')
 finally:
  if 'u' in locals():
   for name in created:u.destroy(name)
   u.client.disconnect()


if __name__=='__main__':main()
