"""Curate the exact first-person observations/action snapshots for the selected replay."""
from pathlib import Path
import argparse,hashlib,json
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);args=ap.parse_args()
data=json.loads((ROOT/'nyc/evidence/task19-replay.json').read_text())
data['presentation']='Original first-person snapshots with short eased dissolves; input then holds during inference. Original simulation and collision-report timing.'
data['display_transition_seconds']=0.14
data['frame_policy']='Hold the recorded input during inference. Show saved action snapshots in file order during action, with equal spacing when exact capture times are unavailable. Hold the final output on completion. No synthesized intermediate frames.'
data.pop('coordinate_policy',None)
count=0
for name,model in data['models'].items():
 folder=args.campaign/model['name']/'cells/easy_realtime_instructional_low/map3_15roads/task_019/attempt_001'
 manifests={json.loads(p.read_text())['step']:p for p in folder.glob('*_steps/step_*/*_manifest.json')}
 for step in model['steps']:
  p=manifests[step['source_step']];assert hashlib.sha256(p.read_bytes()).hexdigest()==step['source_sha256'];m=json.loads(p.read_text())
  sources={'input':[m['input_images'][0]],'action':m['recording_action_frames'],'output':[m['output_image']]}
  step['recorded_frames']={}
  for kind,names in sources.items():
   entries=[]
   for i,file in enumerate(names):
    source=p.parent/file;dest=ROOT/'public/media/recorded/frames'/f'{name}-{step["source_step"]:03}-{kind}-{i:02}.webp';dest.parent.mkdir(parents=True,exist_ok=True)
    im=Image.open(source).convert('RGB');im.save(dest,lossless=True,method=4)
    entries.append({'path':str(dest.relative_to(ROOT/'public')),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'pixel_sha256':hashlib.sha256(im.tobytes()).hexdigest(),'size':list(im.size)});count+=1
   step['recorded_frames'][kind]=entries
(ROOT/'public/data/recorded-replay.json').write_text(json.dumps(data,indent=2)+'\n')
print('Curated',count,'lossless source frames')
