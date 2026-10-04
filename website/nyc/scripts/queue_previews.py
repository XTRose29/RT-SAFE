"""Render camera surveys and scene samples in the isolated Unreal editor."""
import argparse
import json
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--start', type=int, default=0, help='Resume from a scene preview index, 0–4')
args = ap.parse_args()

def wait_ready():
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            status = json.loads((root / 'runtime/editor-status.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            status = {}
        if status.get('ready') and time.time() - status.get('updated', 0) < 20:
            return
        time.sleep(2)
    raise RuntimeError('Editor is not ready for a preview')

def render(name, script, env, result):
    wait_ready()
    started = time.time()
    env.update(RTSAFE_PREVIEW='1', RTSAFE_QUALITY='raster', RTSAFE_SAMPLE_FRAME='')
    command = {'id': name + '-' + str(int(started)), 'script': str(root / 'scripts' / script), 'env': env}
    temp = root / 'runtime/command-next.json'
    temp.write_text(json.dumps(command))
    temp.replace(root / 'runtime/command.json')
    print('Started', name, flush=True)
    deadline = time.monotonic() + 3600
    path = root / result
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_mtime >= started:
            record = json.loads(path.read_text())
            if not record['success']:
                raise RuntimeError('Render failed: ' + str(path))
            print('Finished', name, flush=True)
            time.sleep(5)
            return
        error = root / 'runtime/command-error.txt'
        if error.exists() and error.stat().st_mtime >= started:
            raise RuntimeError(error.read_text())
        time.sleep(3)
    raise TimeoutError(str(path))

for job in [
    ('survey', 'render_camera_survey.py', {}, 'previews/survey-v1/result.json'),
    ('quality', 'render_quality_stills.py', {}, 'previews/quality-v2/result.json'),
]:
    path = root / job[3]
    if not path.exists() or not json.loads(path.read_text()).get('success'):
        render(*job)

jobs = [
    ('timing-rt-preview', 'render_timing_scene.py', {'RTSAFE_TIMING': 'realtime'}, 'renders/timing-realtime-samples/result.json'),
    ('timing-static-preview', 'render_timing_scene.py', {'RTSAFE_TIMING': 'static'}, 'renders/timing-static-samples/result.json'),
    ('environment-preview', 'render_environment.py', {}, 'renders/environment-samples/result.json'),
    ('astra-preview', 'render_task19_replay.py', {'RTSAFE_MODEL': 'astra'}, 'renders/task19/astra-samples/result.json'),
    ('sol-preview', 'render_task19_replay.py', {'RTSAFE_MODEL': 'sol'}, 'renders/task19/sol-samples/result.json'),
]
for job in jobs[args.start:]:
    render(*job)
print('Camera surveys and scene previews complete', flush=True)
