#!/usr/bin/env python3
"""Export compact validation evidence without account details or machine paths."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(suite):
    rows = []
    for cell in sorted(suite.glob('cells/*/*/task_*')):
        status_path = cell / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        for attempt in sorted(cell.glob('attempt_*')):
            validation_file = attempt / 'validation.json'
            validation = json.loads(validation_file.read_text()) if validation_file.exists() else {}
            candidates = []
            for path in attempt.glob('task_*.json'):
                data = json.loads(path.read_text())
                if 'decision_trace' in data:
                    candidates.append((path, data))
            if len(candidates) > 1:
                raise ValueError(f'Ambiguous result in {attempt.name}')
            path, result = candidates[0] if candidates else (None, {})
            step_paths = sorted(attempt.glob('*_steps/*/*_manifest.json'))
            steps = [json.loads(p.read_text()) for p in step_paths]
            accepted = bool(validation and not validation.get('validation_issues')
                            and validation.get('return_code') == 0 and result
                            and not result.get('rollout_error'))
            log = attempt / 'rollout.log'
            log_text = log.read_text(errors='replace') if log.exists() else ''
            if accepted:
                outcome = 'goal_reached' if result.get('success') else 'completed_task_failure'
            elif '429 Too Many Requests' in log_text:
                outcome = 'incomplete_provider_http_429'
            elif 'UnrealCV connection lost' in log_text:
                outcome = 'incomplete_engine_connection_lost'
            elif validation:
                outcome = 'incomplete_infrastructure_or_validation_failure'
            else:
                outcome = 'not_finalized'
            completed_metrics = None
            if accepted:
                completed_metrics = {k: result.get(k) for k in [
                    'success', 'termination_reason', 'collision_count', 'passive_collision_count',
                    'decision_count', 'avg_response_time', 'sim_time', 'safe_success', 'benchmark_metrics']}
                if completed_metrics['safe_success'] is None:
                    indicator = (result.get('benchmark_metrics') or {}).get('safe_success_indicator')
                    if indicator is not None:
                        completed_metrics['safe_success'] = bool(indicator)
            errors = [abs(s['timing']['native_game_time_delta_seconds'] - s['timing']['sim_time_delta_seconds'])
                      for s in steps if s.get('timing', {}).get('native_game_time_delta_seconds') is not None]
            contacts = []
            for index, step in enumerate(steps):
                active = step.get('collision_details') or {}
                passive = step.get('passive_collision_details') or {}
                kinds = ['human', 'object', 'building', 'vehicle', 'fall', 'oil', 'water']
                waypoint = step.get('waypoint_execution') or {}
                if any(active.get(k) or passive.get(k) for k in kinds) or waypoint.get('move_stop_reason'):
                    feedback = step.get('feedback', '')
                    next_prompt = steps[index + 1].get('prompt', {}).get('user', '') if index + 1 < len(steps) else None
                    contacts.append({'step': step['step'],
                        'active': {k: active.get(k, 0) for k in kinds},
                        'passive': {k: passive.get(k, 0) for k in kinds},
                        'feedback': feedback,
                        'feedback_in_next_prompt': feedback in next_prompt if next_prompt is not None else None,
                        'move_stop_reason': waypoint.get('move_stop_reason'),
                        'move_elapsed_seconds': waypoint.get('move_elapsed_seconds'),
                        'move_command_duration_seconds': waypoint.get('move_command_duration_seconds'),
                        'actual_displacement_cm': waypoint.get('actual_displacement_cm')})
            rows.append({'task_id': int(cell.name.split('_')[-1]), 'map': cell.parent.name,
                'condition': cell.parent.parent.name, 'attempt': int(attempt.name.split('_')[-1]),
                'cell_state': status.get('state'), 'accepted_completed_episode': accepted,
                'outcome': outcome, 'recorded_decisions': len(steps),
                'model': result.get('model'), 'provider': result.get('provider'),
                'reasoning_effort': result.get('reasoning_effort'),
                'max_native_vs_accounted_delta_error_s': max(errors) if errors else None,
                'contacts_and_stops': contacts,
                'completed_episode_metrics': completed_metrics,
                'result_sha256': digest(path) if path else None,
                'rollout_log_sha256': digest(log) if log.exists() else None,
                'manifest_sha256': [digest(p) for p in step_paths]})
    return {'scope': 'Small execution-validation pilot, not a leaderboard. Incomplete attempts have no episode scores.',
            'completed_episodes': sum(r['accepted_completed_episode'] for r in rows), 'attempts': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'completed_episodes': report['completed_episodes'], 'attempts': len(report['attempts'])}))


if __name__ == '__main__':
    main()
