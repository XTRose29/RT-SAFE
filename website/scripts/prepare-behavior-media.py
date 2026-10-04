"""Curate recorded input frames and manuscript behavior profiles for presentation."""
from pathlib import Path
import argparse,csv,hashlib,json
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--radar-csv',type=Path,required=True);args=ap.parse_args()
replay=json.loads((ROOT/'nyc/evidence/task19-replay.json').read_text())
for name,m in replay['models'].items():
 folder=args.campaign/m['name']/'cells/easy_realtime_instructional_low/map3_15roads/task_019/attempt_001'
 manifests={json.loads(p.read_text()).get('step'):p for p in folder.glob('*_steps/step_*/*_manifest.json')}
 for step in m['steps']:
  p=manifests[step['source_step']];assert hashlib.sha256(p.read_bytes()).hexdigest()==step['source_sha256']
  d=json.loads(p.read_text());image=p.parent/d['input_images'][0]
  dest=ROOT/'public/media/nyc/inputs'/f'{name}-{step["source_step"]:03}.webp';dest.parent.mkdir(exist_ok=True)
  Image.open(image).convert('RGB').resize((360,320),Image.Resampling.LANCZOS).save(dest,quality=87)
  step['input_image']=str(dest.relative_to(ROOT/'public'))
  step['commanded_distance_m']=(d.get('waypoint_execution') or {}).get('commanded_waypoint_distance_cm',0)/100
  step['recorded_feedback']=d.get('feedback','')
  step['input_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
(ROOT/'nyc/evidence/task19-replay.json').write_text(json.dumps(replay,indent=2)+'\n')
(ROOT/'public/data/nyc').mkdir(parents=True,exist_ok=True)
(ROOT/'public/data/nyc/task19-replay.json').write_text(json.dumps(replay,indent=2)+'\n')
rows=list(csv.DictReader(args.radar_csv.open()));models=json.loads((ROOT/'app/data.json').read_text())['models']
profiles=[]
for row in rows:
 vals=[1/float(row['collisions']),1/float(row['latency']),1/float(row['decisions_per_episode']),float(row['movement_per_step']),float(row['wait_per_step']),float(row['turn_frequency'])]
 profiles.append({'name':row['model'],'raw':vals,'average':next(m['average'] for m in models if m['name']==row['model'])})
maxima=[max(m['raw'][j] for m in profiles) for j in range(6)]
for m in profiles:m['values']=[x/y for x,y in zip(m['raw'],maxima)]
result={'source':'Supplied manuscript Figure 3 and Appendix B.3; curated RQ1 action counts. All difficulties, provider-default effort, 108 episodes per model.','axes':['Fewer collisions','Quicker decisions','Fewer decisions','Longer commanded moves','More waiting','More turning'],'normalization':'Reciprocals of mean collisions, latency and decisions; action measures normalized to the maximum across all eight models. Radar area is not a safety score.','profiles':profiles,'table_source':'Supplied manuscript Table 5','radar_source':'Supplied manuscript Figure 3 / Appendix B.3; RQ1 raw action aggregates'}
(ROOT/'app/behavior.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'public/data/behavior.json').write_text(json.dumps(result,indent=2)+'\n')
print('Curated',sum(len(m['steps']) for m in replay['models'].values()),'recorded observations and 8 behavior profiles')
