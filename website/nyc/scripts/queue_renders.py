"""Render approved native takes sequentially in the isolated editor."""
import argparse,json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument('--only',choices=['timing','environment','astra','sol','hero'])
ap.add_argument('--start-at',choices=['timing-realtime','timing-static','environment','astra','sol','hero'])
ap.add_argument('--resume',action='store_true',help='Keep the batch timestamp for already completed takes')
args=ap.parse_args()
if not args.resume or not (root/'runtime/render-batch.json').exists():
 (root/'runtime/render-batch.json').write_text(json.dumps({'started':time.time(),'only':args.only},indent=2))
jobs=[
 ('timing-realtime','render_timing_scene.py',{'RTSAFE_TIMING':'realtime'},'renders/timing-close-realtime/result.json','timing'),
 ('timing-static','render_timing_scene.py',{'RTSAFE_TIMING':'static'},'renders/timing-close-static/result.json','timing'),
 ('environment','render_environment.py',{},'renders/environment/result.json','environment'),
 ('astra','render_task19_replay.py',{'RTSAFE_MODEL':'astra'},'renders/task19/astra/result.json','astra'),
 ('sol','render_task19_replay.py',{'RTSAFE_MODEL':'sol'},'renders/task19/sol/result.json','sol'),
 ('hero','render_hero.py',{},'renders/hero/result.json','hero'),
]
if args.start_at:jobs=jobs[next(i for i,j in enumerate(jobs) if j[0]==args.start_at):]
for name,script,env,result,group in jobs:
 if args.only and args.only!=group:continue
 ready_deadline=time.monotonic()+180
 while time.monotonic()<ready_deadline:
  try:status=json.loads((root/'runtime/editor-status.json').read_text())
  except (FileNotFoundError,json.JSONDecodeError):status={}
  if status.get('ready') and time.time()-status.get('updated',0)<20:break
  time.sleep(2)
 else:raise RuntimeError('Editor is not ready for a new take')
 started=time.time();env.update(RTSAFE_PREVIEW='0',RTSAFE_QUALITY='raster',RTSAFE_SAMPLE_FRAME='')
 command={'id':name+'-full-'+str(int(started)),'script':str(root/'scripts'/script),'env':env}
 temp=root/'runtime/command-next.json';temp.write_text(json.dumps(command));temp.replace(root/'runtime/command.json')
 print('Started',name,flush=True)
 deadline=time.monotonic()+7200
 while time.monotonic()<deadline:
  p=root/result
  if p.exists() and p.stat().st_mtime>=started:
   record=json.loads(p.read_text())
   if not record['success']:raise RuntimeError(str(record))
   print('Finished',name,'in',round(time.time()-started,1),'seconds',flush=True);break
  error=root/'runtime/command-error.txt'
  if error.exists() and error.stat().st_mtime>=started:raise RuntimeError(error.read_text())
  time.sleep(5)
 else:raise TimeoutError(name)
 time.sleep(10)
print('Native render queue complete',flush=True)
