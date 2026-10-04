"""Export the staged opening and environment introduction for the website."""
from pathlib import Path
import json,subprocess,importlib.util
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'video/nyc-90s';MEDIA=ROOT/'public/media/nyc'
tl=json.loads((OUT/'timeline.json').read_text())
spec=importlib.util.spec_from_file_location('soundtrack',ROOT/'scripts/music-soundtrack.py')
soundtrack=importlib.util.module_from_spec(spec);spec.loader.exec_module(soundtrack)
if not soundtrack.TRACK.exists():soundtrack.generate()
def seconds(s):
 h,m,s=s.split(':');return int(h)*3600+int(m)*60+float(s)
def stamp(t):
 ms=round(t*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}.{ms%1000:03}'
for name,stem in [('timing','timing-story'),('environment','environment-intro')]:
 scene=next(s for s in tl['scenes'] if s['id']==name)
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(OUT/'scenes'/f'{name}.mp4'),'-ss',str(scene['start']),'-i',str(soundtrack.TRACK),'-map','0:v','-map','1:a','-c:v','copy','-af',soundtrack.music_filter(scene['duration']),'-c:a','aac','-b:a','192k','-t',str(scene['duration']),'-movflags','+faststart',str(MEDIA/f'{stem}.mp4')],check=True)
 cues=['WEBVTT','']
 for block in (OUT/'captions.vtt').read_text().split('\n\n'):
  if '-->' not in block:continue
  lines=block.strip().splitlines();a,b=map(seconds,lines[0].split(' --> '))
  if scene['start']<=a<scene['start']+scene['duration']:
   cues.extend([f'{stamp(a-scene["start"])} --> {stamp(min(scene["duration"],b-scene["start"]))}',*lines[1:],''])
 (MEDIA/f'{stem}.vtt').write_text('\n'.join(cues))
 subprocess.run(['ffmpeg','-v','error','-y','-ss','1','-i',str(MEDIA/f'{stem}.mp4'),'-frames:v','1','-c:v','libwebp','-quality','93',str(MEDIA/f'{stem}.webp')],check=True)
 print('Exported',stem,scene['duration'],'seconds')
