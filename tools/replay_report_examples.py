#!/usr/bin/env python3
"""Replay exported report poses/actions in their matching, disposable UE maps.

Historical dynamic actor positions and prior hazard occupancy are not restored.
The original map, task route, pose, route heading, and selected action are restored.
Each example gets a fresh UE process; never point this at a campaign port.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'SimWorld')]
from base.rt_unrealcv import RTUnrealCV
from evaluation.run_qwen3vl8b_all_maps import DEFAULT_MAPS, UEServer
from simworld.utils.vector import Vector
from tools.validate_report_tasks import (
    configure_probe_environment, install_scripted_policy, prepare,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples-root', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--ue-launcher', required=True)
    args = parser.parse_args()
    configure_probe_environment()
    os.environ['SIMWORLD_SIGNAL_TRAFFIC_VEHICLES'] = '0'
    os.environ['SIMWORLD_SIGNAL_TRAFFIC_PEDESTRIANS'] = '0'
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.cases:
        source = next(args.examples_root.glob(f'*/{name}/case.json'))
        case = json.loads(source.read_text())
        old = json.loads(Path(case['source']['manifest']).read_text())
        map_name = case['source']['map']
        map_spec = next(spec for spec in DEFAULT_MAPS if spec.name == map_name)
        task_id = int(case['source']['task'].split('_')[-1])
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        os.environ['SIMWORLD_UE_EXTRA_ARGS'] = f'-UserDir={folder / "ue-user"}'
        launch = SimpleNamespace(
            ue_launcher=args.ue_launcher, ue_gpu=args.gpu, ue_port=args.port,
            ue_host='127.0.0.1', ue_width=1280, ue_height=720, ue_fps=30,
            ue_no_rhi_thread=True, ue_ready_timeout=180, ue_settle_seconds=5,
            process_shutdown_timeout=30,
        )
        server = UEServer(launch, folder / 'engine.log')
        ue = None
        result = {
            'case_id': name, 'map': map_name, 'task': task_id,
            'source_case_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_manifest_sha256': hashlib.sha256(Path(case['source']['manifest']).read_bytes()).hexdigest(),
            'scope': 'matching-map pose/action replay; historical dynamic actors and occupancy not restored',
            'passed': False,
        }
        try:
            server.start(map_spec.ue_map)
            ue = RTUnrealCV(port=args.port, resolution=(720, 640))
            ue.pause_simulation()
            manager = prepare(ue, folder / 'scene', task_id, map_name)
            agent = manager.agent
            install_scripted_policy(agent)
            geometry = old['observation_geometry']
            position = geometry['agent_position_cm']
            z = geometry['camera_location_cm'][2] - 135.
            # Controlled initial condition, never an action endpoint correction.
            ue.humanoid_stop(agent.name)
            ue.set_location((position['x'], position['y'], z), agent.name)
            # Historical camera yaw uses a different sign convention from
            # actor yaw. Derive the route heading from the exported marker
            # ray (unrounded), then use the production route-to-UE conversion.
            straight = geometry['annotated_candidates'][0]['world_position_cm']
            route_yaw = math.degrees(math.atan2(straight['y'] - position['y'],
                                                straight['x'] - position['x']))
            ue.set_orientation((0., agent._route_yaw_to_ue_yaw(route_yaw), 0.), agent.name)
            agent.sync_ue()
            route = old['route_adherence']
            if route:
                start = Vector(**route['segment_start'])
                end = Vector(**route['segment_end'])
                index = next(i for i in range(len(agent.route_polyline) - 1)
                             if agent.route_polyline[i].distance(start) < 1
                             and agent.route_polyline[i + 1].distance(end) < 1)
                agent.shortest_path = agent.original_shortest_path[index:].copy()
                agent.current_destination = agent.shortest_path[0]
            agent.step_num = case['source']['step']
            agent.decision_count = case['source']['decision'] - 1
            agent.sim_time_elapsed = case['outcome']['sim_time_start_s']
            agent.start_time = time.time()
            agent.disable_step_evaluation = True
            # Freeze the scheduler for red-entry replay, keeping native heads red.
            if name.startswith('red_light'):
                agent.simulation_step_callback = None
                for signals in agent.crosswalk_signal_groups.values():
                    for signal in signals:
                        manager.communicator.set_traffic_signal_pedestrian_stop(signal.id)
            samples = []
            original_capture = manager.communicator.get_camera_observation

            def capture(camera, *params, **kwargs):
                actor = [float(v) for v in ue.get_location(agent.name)]
                pose = [float(v) for v in ue.get_camera_location(camera).split()]
                samples.append({'actor': actor, 'camera': pose,
                                'error_cm': math.hypot(actor[0] - pose[0], actor[1] - pose[1])})
                return original_capture(camera, *params, **kwargs)

            manager.communicator.get_camera_observation = capture
            action = case['model_output']['action']
            agent.validation_action = (action['type'], action['param'])
            agent.step('greedy')
            trace = agent.decision_trace[-1]
            move_samples = list(getattr(agent, 'last_swept_move_samples', []))
            result.update(
                action=action, start=position,
                initial_hazard_clearance=manager.initial_hazard_clearance,
                historical_actual=(old.get('waypoint_execution') or {}).get('post_action_position_cm'),
                historical_collisions=old['collision_details'],
                trace=trace, camera_samples=samples, sweep_samples=move_samples,
                actual={'x': agent.position.x, 'y': agent.position.y},
            )
            checks = {'camera_tracks_actor': all(s['error_cm'] < .5 for s in samples)}
            details = trace['collision_details']
            checks['no_proximity_labels'] = all(details.get('ue_physical_block_' + k, 0) == 0
                                               for k in ('human', 'object', 'building'))
            if action['type'] == 'move_to':
                target = next(c['world_position_cm'] for c in geometry['annotated_candidates']
                              if c['index'] == int(action['param']))
                target = Vector(**target)
                commanded = agent.last_commanded_waypoint
                dx, dy = commanded.x - position['x'], commanded.y - position['y']
                length = math.hypot(dx, dy)
                cross_track = max(abs((s['actual_cm']['x'] - position['x']) * dy
                                      - (s['actual_cm']['y'] - position['y']) * dx) / length
                                  for s in move_samples)
                blocked = any(s['sweep_blocked'] for s in move_samples)
                error = agent.position.distance(commanded)
                result.update(historical_target={'x': target.x, 'y': target.y},
                              commanded={'x': commanded.x, 'y': commanded.y},
                              endpoint_error_cm=error, max_cross_track_error_cm=cross_track,
                              blocked=blocked)
                checks.update(selected_marker_matches_export=commanded.distance(target) < .5,
                              swept_path_follows_selected_ray=cross_track < 1.,
                              reaches_target_or_native_sweep_blocks=error < .5 or blocked)
            if name.startswith('red_light'):
                result['entry_events'] = list(agent.red_light_violation_events)
                agent.validation_action = ('wait', '1')
                agent.step('greedy')
                result['next_trace'] = agent.decision_trace[-1]
                checks['same_action_red_event'] = len(result['entry_events']) == 1
                checks['no_duplicate_after_wait'] = agent.red_light_violations_count == 1
            if name == 'trip_0021':
                checks['first_move_no_trip'] = details['fall'] == 0
            result['checks'] = checks
            result['passed'] = all(checks.values())
        except Exception as exc:
            result['error'] = repr(exc)
            raise
        finally:
            (folder / 'result.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
            results.append(result)
            (args.output / 'results.json').write_text(json.dumps(results, indent=2, default=str) + '\n')
            if ue is not None:
                ue.client.disconnect()
            server.stop()
        print(json.dumps({k: result[k] for k in ('case_id', 'passed', 'checks')}), flush=True)
    if not all(r['passed'] for r in results):
        raise SystemExit('One or more native replay checks failed; inspect results.json')


if __name__ == '__main__':
    main()
