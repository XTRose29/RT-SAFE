#!/usr/bin/env python3
"""Compare the old and current clearance RPC paths on isolated native heads.

Requires a dedicated, paused UE instance. This measures only signal polling and
clearance writes, not end-to-end task throughput or model performance.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'SimWorld')]
from base.rt_unrealcv import RTUnrealCV
from base.rt_communicator import RTCommunicator
from simworld.traffic.base.traffic_signal import TrafficSignalState as S
from simworld.traffic.manager.intersection_manager import IntersectionManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ue = RTUnrealCV(port=args.port)
    comm = RTCommunicator(ue)
    ue.pause_simulation()
    ids = list(range(900000, 900066))
    existing = set(ue.client.request('vget /objects').split())
    names = [comm.get_traffic_signal_name(i) for i in ids]
    if existing.intersection(names):
        raise RuntimeError('Probe names already exist; use a fresh dedicated engine')
    created = []
    red = (S.VEHICLE_RED, S.PEDESTRIAN_RED)
    lights = [SimpleNamespace(id=i, lane_id=i, state=red) for i in ids]
    for light in lights:
        light.get_state = lambda light=light: light.state
    junction = SimpleNamespace(id=1, traffic_lights=lights, pedestrian_lights=[],
        python_pedestrian_phase='clearance', python_pedestrian_phase_elapsed_s=0., cycle_count=0)
    manager = IntersectionManager.__new__(IntersectionManager)
    manager.intersections = [junction]
    manager.python_states = {}
    manager.logger = logging.getLogger('clearance-probe')
    manager.config = {'traffic.traffic_signal.pedestrian_green_light_duration': 5,
                      'traffic.traffic_signal.pedestrian_clearance_duration': 3}
    writes = []
    original_switch = comm.traffic_signal_switch_to
    def switch(i, state='green'):
        writes.append((i, state))
        return original_switch(i, state)
    comm.traffic_signal_switch_to = switch
    def refresh(batched):
        states = (comm.get_traffic_signal_states(ids) if batched
                  else [comm.get_traffic_signal_state(i) for i in ids])
        for light, state in zip(lights, states):
            light.state = (S.VEHICLE_GREEN if state['vehicle_green'] else S.VEHICLE_RED,
                           S.PEDESTRIAN_GREEN if state['pedestrian_walk'] else S.PEDESTRIAN_RED)
        return states
    rows = []
    try:
        for i, name in zip(ids, names):
            ue.spawn_bp_asset('/Game/RealTimeBench/Traffic/RT_BP_street_light.RT_BP_street_light_C', name)
            created.append(name)
            ue.set_location((-50000, (i - ids[0]) * 200, 0), name)
            original_switch(i, 'all red')
        refresh(True)
        assert all(light.state == red for light in lights), 'Initial native heads are not all red'
        for legacy in (True, False):
            writes.clear()
            began = time.perf_counter()
            for delta in [.1, .1, .05] * 4:
                refresh(not legacy)
                if legacy:
                    # Exact former clearance branch: three synchronous commands
                    # per head even when its just-observed state is already red.
                    for i in ids:
                        comm.traffic_signal_switch_to(i, 'all red')
                else:
                    manager.update_intersections(comm, delta_time_s=delta)
            elapsed = time.perf_counter() - began
            refresh(True)
            assert all(light.state == red for light in lights), 'Native red state changed'
            rows.append({'path': 'former_sequential_read_and_reassert' if legacy else 'batched_read_and_repair_only',
                         'wall_seconds': elapsed, 'heads': len(ids), 'ticks': 12,
                         'native_state_reads': 12 * len(ids), 'all_red_switches': len(writes),
                         'all_red_commands': 3 * len(writes), 'all_heads_verified_red': True})
        assert rows[0]['all_red_commands'] == 2376 and rows[1]['all_red_commands'] == 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'passed': True, 'scope': __doc__, 'results': rows}, indent=2) + '\n')
        print(json.dumps(rows), flush=True)
    finally:
        for name in created:
            ue.destroy(name)
        ue.client.disconnect()


if __name__ == '__main__':
    main()
