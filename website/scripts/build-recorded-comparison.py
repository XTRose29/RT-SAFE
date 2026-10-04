"""Build the original first-person replay with the existing game-style HUD."""
from pathlib import Path
import importlib.util,json,functools,argparse
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def module(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
media=module('build-nyc-media');hud=module('replay-presentation');v=media.v
DATA=json.loads((ROOT/'public/data/recorded-replay.json').read_text());OUT=ROOT/'public/media/recorded'
@functools.lru_cache(160)
def load(path):return Image.open(ROOT/'public'/path).convert('RGB')
def frame_for(model,t):
 state=hud.replay.sample_agent(model,t*6);step=model['steps'][state['decision']-1];frames=step['recorded_frames']
 if state['finished']:entry=frames['output'][0]
 elif state['phase']=='inference':entry=frames['input'][0]
 else:
  start=step['start_sim_seconds']+step['inference_exposure_seconds'];span=max(.001,step['end_sim_seconds']-start)
  progress=max(0,min(.999,(state['source_sim_seconds']-start)/span))
  captures=frames['action'] or frames['output'];entry=captures[int(progress*len(captures))]
 return load(entry['path'])
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
  v.text(im,(30,1050),'Input held during inference; snapshots are not continuous video. Collision alerts mark source-log report times.',18,v.PALE,400)
  writer.stdin.write(im.tobytes())
  if i==0:im.save(OUT/'comparison-poster.webp',quality=93)
 media.finish(writer);print('Exported original 44.1-second comparison',flush=True)
if __name__=='__main__':main()
