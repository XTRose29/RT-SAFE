"""Build the original first-person replay with the existing game-style HUD."""
from pathlib import Path
import importlib.util,json,functools,argparse,bisect
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def module(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
media=module('build-nyc-media');hud=module('replay-presentation');v=media.v
DATA=json.loads((ROOT/'public/data/recorded-replay.json').read_text());OUT=ROOT/'public/media/recorded'
@functools.lru_cache(160)
def load(path):return Image.open(ROOT/'public'/path).convert('RGB')
@functools.lru_cache(4)
def frame_schedule(name):
 model=DATA['models'][name];origin=model['steps'][0]['start_sim_seconds'];events={}
 for step in model['steps']:
  frames=step['recorded_frames'];start=step['start_sim_seconds'];end=step['end_sim_seconds']
  events[(start-origin)/6]=frames['input'][0]['path']
  action_start=start+step['inference_exposure_seconds'];span=max(.001,end-action_start)
  captures=frames['action'] or frames['output']
  for i,entry in enumerate(captures):events[(action_start+span*i/len(captures)-origin)/6]=entry['path']
  events[(end-origin)/6]=frames['output'][0]['path']
 return sorted(events.items())
def frame_for(model,t):
 name=next(name for name,value in DATA['models'].items() if value is model)
 events=frame_schedule(name);index=max(0,bisect.bisect_right([at for at,_ in events],t)-1)
 at,path=events[index];current=load(path)
 if index and t-at<.14:
  previous=load(events[index-1][1])
  if previous.size!=current.size:previous=previous.resize(current.size,Image.Resampling.LANCZOS)
  span=min(.14,(events[index+1][0]-at)*.8) if index+1<len(events) else .14
  return Image.blend(previous,current,v.ease((t-at)/max(.001,span)))
 return current
def panel(name,t):
 model=DATA['models'][name];return hud.compose(frame_for(model,t),model,name,t,v,source='recorded')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stills',action='store_true');args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if args.stills:
  folder=ROOT/'nyc/previews/original-input';folder.mkdir(parents=True,exist_ok=True)
  for name in DATA['models']:
   for t in [0,1.1,35.8]:panel(name,t).save(folder/f'{name}-{t}.png')
  return
 for name in DATA['models']:
  writer=media.writer(OUT/f'{name}.mp4',width=960,height=1000,crf=19)
  for i in range(1323):
   im=panel(name,i/30);writer.stdin.write(im.tobytes())
   if i==0:im.save(OUT/f'{name}.webp',quality=93)
  media.finish(writer);print('Rendered original replay:',name,flush=True)
 clips=[media.film.Clip(OUT/f'{name}.mp4') for name in ('astra','sol')]
 writer=media.writer(ROOT/'public/media/rt-safe-original-comparison.mp4',crf=19)
 for i in range(1323):
  im=Image.new('RGB',(1920,1080),v.INK)
  for k,clip in enumerate(clips):im.paste(clip.frame(i/30),(960*k,0))
  v.line(im,[(959,0),(959,1000)],v.WHITE,3)
  v.text(im,(30,1015),'Original first-person observations and action snapshots · RT15 / Task 19 · 6× recorded simulation timeline',23,v.WHITE,500)
  v.text(im,(30,1050),'Short dissolves smooth recorded snapshots; input then holds during inference. Collision alerts use original report times.',18,v.PALE,400)
  writer.stdin.write(im.tobytes())
  if i==0:im.save(OUT/'comparison-poster.webp',quality=93)
 media.finish(writer);print('Exported original 44.1-second comparison',flush=True)
if __name__=='__main__':main()
