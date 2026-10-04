"""Render the inspected contact/walk revision, then refresh the environment tour."""
from pathlib import Path
import json,time
r=Path(__file__).resolve().parents[1]
jobs=[('timing-close-static','render_timing_scene.py',{'RTSAFE_TIMING':'static'}),('timing-close-realtime','render_timing_scene.py',{'RTSAFE_TIMING':'realtime'}),('environment','render_environment.py',{'RTSAFE_QUALITY':'ssgi'})]
for name,script,env in jobs:
    started=time.time();p=r/'runtime/command-next.json'
    p.write_text(json.dumps({'id':'motion-revision-'+name+'-'+str(started),'script':str(r/'scripts'/script),'env':{'RTSAFE_PREVIEW':'0','RTSAFE_SAMPLE_FRAME':'',**env}}));p.replace(r/'runtime/command.json');print('Started',name,flush=True)
    result=r/f'renders/{name}/result.json'
    while True:
        if result.exists() and result.stat().st_mtime>started:
            d=json.loads(result.read_text());assert d['success'],d;print('Finished',name,d['elapsed'],flush=True);break
        e=r/'runtime/command-error.txt'
        if e.exists() and e.stat().st_mtime>started:raise RuntimeError(e.read_text())
        if time.time()-started>3600:raise TimeoutError(name)
        time.sleep(3)
    time.sleep(5)
print('Native motion revision completed',flush=True)
