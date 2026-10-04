"""Encode completed native takes, render the film edit, and validate the exports."""
from pathlib import Path
import json,subprocess,sys,time,shutil,importlib.util
ROOT=Path(__file__).resolve().parents[1];NYC=ROOT/'nyc';PY=sys.executable
batch=json.loads((NYC/'runtime/render-batch.json').read_text());started=batch['started']
def wait(relative):
 p=NYC/relative;deadline=time.monotonic()+6*3600
 while time.monotonic()<deadline:
  if p.exists() and p.stat().st_mtime>=started:
   data=json.loads(p.read_text())
   if data['success']:return
   raise RuntimeError('Native render failed: '+relative)
  error=NYC/'runtime/command-error.txt'
  if error.exists() and error.stat().st_mtime>=started:raise RuntimeError(error.read_text())
  time.sleep(5)
 raise TimeoutError(relative)
def run(script,*args):
 print('Running',script,*args,flush=True);subprocess.run([PY,str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True)
wait('renders/timing-close-static/result.json');run('build-nyc-media.py','timing')
wait('renders/environment/result.json');run('compose-nyc-environment.py')
wait('renders/task19/sol/result.json');run('build-nyc-media.py','comparison')
wait('renders/hero/result.json')
run('build-nyc-media.py','hero')
run('export-nyc-stills.py')
run('build-recorded-comparison.py')
for name in ['title','timing','environment','examples','behavior','completion','comparison','reasoning','learning']:run('render-film-nyc.py','--scene',name)
run('render-film-nyc.py','--stills');run('assemble-film-nyc.py');run('verify-film-nyc.py')
print('NYC video exports complete and validated',flush=True)
