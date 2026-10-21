#!/usr/bin/env python3
"""Audit completed action windows from saved artifacts; never contact UE.

Request/response spans may overlap for batches. Their sum is diagnostic, not
an additive wall-time decomposition. Native exposure comes from UE timestamps.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
import statistics
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def covered_seconds(intervals):
    """Length of the union, so concurrently queued requests count once."""
    total = 0.0
    end = float('-inf')
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def audit(attempt: Path, engine: Path):
    windows = {}
    rollout = attempt / 'rollout.log'
    for line in rollout.read_text().splitlines():
        match = re.search(r'Step (\d+), (Action: Action:|Success:)', line)
        if match:
            timestamp = dt.datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=dt.timezone.utc).timestamp()
            window = windows.setdefault(int(match[1]), {})
            window['start' if match[2].startswith('Action') else 'end'] = timestamp
    windows = {step: w for step, w in windows.items() if 'start' in w and 'end' in w}
    manifests = {}
    for path in sorted(attempt.glob('*_steps/*/*_manifest.json')):
        data = json.loads(path.read_text())
        manifests[int(data['step'])] = (path, data)
    pending = {}
    requests = collections.defaultdict(list)
    pattern = re.compile(r'^\[([^]]+)\].*?(?:Request: (\d+):(.*)|Response id: (\d+))$')
    snapshot = engine.read_bytes()
    for line in snapshot.decode(errors='replace').splitlines():
        match = pattern.match(line)
        if not match:
            continue
        timestamp = dt.datetime.strptime(match[1], '%Y.%m.%d-%H.%M.%S:%f').replace(tzinfo=dt.timezone.utc).timestamp()
        if match[2]:
            pending[match[2]] = (timestamp, match[3])
        else:
            request = pending.pop(match[4], None)
            if request:
                start, command = request
                for step, window in windows.items():
                    if window['start'] <= start and timestamp <= window['end']:
                        requests[step].append((start, timestamp, command))
                        break
    rows = []
    for step, window in sorted(windows.items()):
        if step not in manifests:
            continue
        path, data = manifests[step]
        categories = collections.defaultdict(list)
        intervals = collections.defaultdict(list)
        for start, end, command in requests[step]:
            if re.fullmatch(r'vbp RT_TRAFFIC_SIGNAL_\d+ GetState', command):
                category = 'traffic_signal_read'
            elif re.fullmatch(r'vbp RT_TRAFFIC_SIGNAL_\d+ (SwitchVehicleFrontRed|SwitchVehicleBackRed|SetPedestrianStop)', command):
                category = 'traffic_signal_all_red_write'
            elif '/action/game/' in command:
                category = 'pause_resume'
            elif 'GetGameTimeSinceCreation' in command:
                category = 'native_clock_read'
            else:
                category = 'other'
            categories[category].append(end - start)
            intervals[category].append((start, end))
        timing = data.get('timing', {})
        native_delta = timing.get('native_game_time_delta_seconds')
        simulated_delta = timing.get('sim_time_delta_seconds')
        rows.append({
            'step': step, 'decision': step + 1,
            'manifest_sha256': sha256(path),
            'action': data.get('model_output', {}).get('parsed_action'),
            'feedback': data.get('feedback'),
            'logged_action_wall_s': window['end'] - window['start'],
            'timing': timing,
            'native_vs_accounted_delta_error_s': (abs(native_delta - simulated_delta)
                if native_delta is not None and simulated_delta is not None else None),
            'waypoint_execution': {k: v for k, v in (data.get('waypoint_execution') or {}).items() if k != 'sweep_samples'},
            'engine_requests': len(requests[step]),
            'engine_request_span_union_s': covered_seconds(
                [(start, end) for start, end, _ in requests[step]]),
            'categories': {key: {'requests': len(values), 'summed_spans_s': sum(values),
                'span_union_s': covered_seconds(intervals[key]),
                'median_span_ms': statistics.median(values) * 1000}
                for key, values in sorted(categories.items())},
        })
    return {
        'method': 'Read-only matching of engine request/response IDs within completed action windows. Batch spans can overlap; do not interpret their sum as an additive wall-time decomposition.',
        'engine_log_snapshot_bytes': len(snapshot),
        'engine_log_snapshot_sha256': hashlib.sha256(snapshot).hexdigest(),
        'rollout_log_sha256': sha256(rollout),
        'completed_recorded_actions': len(rows), 'actions': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--engine-log', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.attempt, args.engine_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'completed_recorded_actions': report['completed_recorded_actions']}))


if __name__ == '__main__':
    main()
