#!/usr/bin/env python3
"""Exercise swept rays, actual contact and camera poses in an isolated UE server.

The supplied port MUST belong to a separate validation server, never a live
benchmark. Creates two uniquely named actors and destroys only those actors.
No model requests, scene reloads, or changes to runtime files are made.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'SimWorld')]
from base.rt_agent import RTAgent
from base.rt_communicator import RTCommunicator
from base.rt_unrealcv import RTUnrealCV
from simworld.utils.vector import Vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    ue = RTUnrealCV(port=args.port, resolution=(720, 640))
    ue.pause_simulation()
    comm = RTCommunicator(ue)
    suffix = uuid.uuid4().hex[:8]
    name, blocker = 'REPORT_AGENT_'+suffix, 'REPORT_BLOCK_'+suffix
    report = {'schema': 'execution_alignment_v3', 'rays': [], 'passed': False}
    start = Vector(-5000, 0)
    try:
        ue.spawn_bp_asset('/Game/RealTimeBench/Agent/BP_RT_Agent.BP_RT_Agent_C', name)
        ue.set_location((start.x, start.y, 110), name)
        a = RTAgent(start, Vector(1,0), Vector(10000,0), [Vector(10000,0)],
                    100, comm, None, use_tick=True, realtime_thinking=False,
                    use_action_frames=True,
                    task_edges=[{'node1': [-6000, 0], 'node2': [10000, 0], 'type': 'sidewalk'}])
        a.name = name
        a.initialize_first_person_camera()
        for angle, length in [(0,100), (0,200), (-45,200), (45,200), (0,400), (-45,400), (45,400)]:
            ue.humanoid_stop(name)
            ue.set_location((start.x,start.y,110), name)  # setup only
            ue.set_orientation((0,0,0), name)
            a.position = Vector(start.x,start.y)
            a.direction = 0
            target = start + Vector(math.cos(math.radians(angle)),math.sin(math.radians(angle))) * length
            a.move_to(target)
            cross_track = max(abs((v['actual_cm']['x']-start.x)*(target.y-start.y)
                                 -(v['actual_cm']['y']-start.y)*(target.x-start.x))/length
                              for v in a.last_swept_move_samples)
            result = {'angle_deg':angle,'length_cm':length,
                      'endpoint_error_cm':a.position.distance(target),
                      'max_cross_track_error_cm':cross_track,
                      'camera_error_cm':math.hypot(a.camera_location[0]-a.position.x,a.camera_location[1]-a.position.y),
                      'action_frames':len(a.action_frames),
                      'sweep_samples':a.last_swept_move_samples}
            report['rays'].append(result)
            assert result['endpoint_error_cm'] < .5, result
            assert cross_track < .5, result
            assert result['camera_error_cm'] < .5, result
        ue.spawn_bp_asset('/Game/RealTimeBench/Agent/BP_RT_Agent.BP_RT_Agent_C', blocker)
        ue.set_location((start.x+300,start.y,110), blocker)
        ue.set_location((start.x,start.y,110), name)
        a.position = Vector(start.x,start.y)
        a.direction = 0
        baseline = comm.get_states(name)
        a.last_ue_collision_count.update(human=baseline[0],object=baseline[1],building=baseline[2])
        a.last_action_type = 'MOVE'
        a.last_action_start_position = Vector(start.x,start.y)
        a.move_to(Vector(start.x+400,start.y))
        feedback, details, _ = a.get_feedback()
        report['contact'] = {'feedback':feedback,'details':details,
                             'displacement_cm':a.position.distance(start),
                             'position':a._vector_record(a.position),
                             'camera':a.camera_location,
                             'sweep_samples':a.last_swept_move_samples}
        assert 0 < report['contact']['displacement_cm'] < 300
        assert details['human'] > 0 and details['ue_physical_block_human'] == 0
        assert abs(a.camera_location[0]-a.position.x) < .5
        report['passed'] = True
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        for actor in (name,blocker):
            ue.destroy(actor)
        ue.client.disconnect()
    print(json.dumps({'passed':report['passed'],'rays':len(report['rays']),'output':str(args.output)}))


if __name__ == '__main__':
    main()
