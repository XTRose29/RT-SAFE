#!/usr/bin/env python3
"""Compare a frozen-height negative control with production sweeps in native UE.

Requires a disposable RT12 server with clear ground at (-5000, 0). Creates and
destroys only its own actors. No cameras or model inference are needed to
isolate the effect of floor depenetration between physics ticks and sweeps.
"""
import argparse
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'SimWorld')]
from base.rt_agent import RTAgent
from base.rt_communicator import RTCommunicator
from base.rt_unrealcv import RTUnrealCV
from simworld.utils.vector import Vector
from tools.validate_report_tasks import configure_probe_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    configure_probe_environment()
    ue = RTUnrealCV(port=args.port, resolution=(720, 640))
    ue.pause_simulation()
    comm = RTCommunicator(ue)
    original = ue.move_actor_with_sweep
    results = []
    actor = None
    try:
        for mode in ('frozen_height_negative_control', 'production_live_height'):
            actor = 'REPORT_HEIGHT_' + uuid.uuid4().hex[:8]
            ue.spawn_bp_asset('/Game/RealTimeBench/Agent/BP_RT_Agent.BP_RT_Agent_C', actor)
            # Reproduce a pose below the floor's supported capsule height.
            ue.set_location((-5000., 0., 85.), actor)
            start, target = Vector(-5000, 0), Vector(-4600, 0)
            agent = RTAgent(
                start, Vector(1, 0), Vector(10000, 0), [Vector(10000, 0)],
                100, comm, None, use_tick=True, realtime_thinking=False,
                use_action_frames=False,
                task_edges=[{'node1': [-6000, 0], 'node2': [10000, 0], 'type': 'sidewalk'}],
            )
            agent.name = actor
            agent.last_action_type = 'MOVE'
            agent.last_action_start_position = start
            agent.last_action_safety_samples = []
            agent._collect_action_safety = True
            calls = []

            def sweep(name, requested):
                live = [float(v) for v in ue.get_location(name)]
                executed = tuple(requested)
                if mode == 'frozen_height_negative_control':
                    executed = (requested[0], requested[1], 85.)
                actual = original(name, executed)
                calls.append({'live': live, 'requested': list(requested),
                              'executed_target': list(executed), 'actual': list(actual)})
                return actual

            ue.move_actor_with_sweep = sweep
            agent.move_to(target)
            feedback, details, _ = agent.get_feedback()
            ue.move_actor_with_sweep = original
            results.append({'mode': mode, 'endpoint_error_cm': agent.position.distance(target),
                            'feedback': feedback, 'details': details, 'calls': calls})
            ue.destroy(actor)
            actor = None
        assert results[0]['endpoint_error_cm'] > 25, 'Negative control did not reproduce floor blocking'
        assert results[1]['endpoint_error_cm'] < .5, 'Production controller failed the clear move'
        assert all(abs(c['requested'][2] - c['live'][2]) < .01 for c in results[1]['calls'])
    finally:
        ue.move_actor_with_sweep = original
        if actor is not None:
            ue.destroy(actor)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str) + '\n')
        ue.client.disconnect()


if __name__ == '__main__':
    main()
