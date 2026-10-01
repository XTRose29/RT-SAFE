"""Generate scene narration, timestamped subtitles, and a reproducible film timeline."""
import asyncio,json,subprocess,math,re
from pathlib import Path
import edge_tts
ROOT=Path(__file__).resolve().parents[1]
VOICE='en-US-AriaNeural'
def stamp(t,sep='.'):
 ms=round(t*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}{sep}{ms%1000:03}'
async def main():
 scenes=json.loads((ROOT/'video/storyboard.json').read_text());out=ROOT/'video/audio';out.mkdir(exist_ok=True)
 async def one(s):
  p=out/(s['id']+'.mp3');events=out/(s['id']+'.words.json')
  if p.exists() and events.exists():return
  for attempt in range(3):
   try:
    words=[]
    with p.open('wb') as f:
     async for chunk in edge_tts.Communicate(s['narration'],VOICE,rate='-3%',pitch='-2Hz',boundary='WordBoundary').stream():
      if chunk['type']=='audio':f.write(chunk['data'])
      elif chunk['type']=='WordBoundary':words.append({'start':chunk['offset']/1e7,'duration':chunk['duration']/1e7,'text':chunk['text']})
    events.write_text(json.dumps(words));break
   except Exception:
    if attempt==2:raise
  print(s['id'],'narrated',flush=True)
 # Small batches avoid throttling; these are independent audio scenes.
 for i in range(0,len(scenes),3):await asyncio.gather(*(one(s) for s in scenes[i:i+3]))
 cursor=0;vtt=['WEBVTT',''];srt=[];count=1
 for s in scenes:
  p=out/(s['id']+'.mp3');duration=float(subprocess.check_output(['ffprobe','-v','quiet','-show_entries','format=duration','-of','csv=p=0',str(p)]))
  s.update(start=cursor,voiceDuration=duration,voiceOffset=.65,duration=math.ceil(max(s['minDuration'],duration+1.6)*30)/30)
  words=json.loads((out/(s['id']+'.words.json')).read_text())
  original=s['narration'].split()
  assert len(words)==len(original), 'Review subtitle alignment after narration edits'
  for word,token in zip(words,original):word['text']=token
  groups=[];group=[]
  for word in words:
   group.append(word)
   if len(group)>=9 or re.search(r'[.!?]$',word['text']):groups.append(group);group=[]
  if group:groups.append(group)
  for group in groups:
   start=cursor+.65+group[0]['start'];end=cursor+.65+group[-1]['start']+group[-1]['duration']+.1
   line=' '.join(x['text'] for x in group)
   vtt += [f'{stamp(start)} --> {stamp(end)}',line,'']
   srt += [str(count),f'{stamp(start,",")} --> {stamp(end,",")}',line,''];count+=1
  cursor+=s['duration']
 (ROOT/'video/timeline.json').write_text(json.dumps({'fps':30,'width':1920,'height':1080,'voice':VOICE,'duration':cursor,'scenes':scenes},indent=2))
 (ROOT/'public/media/rt-safe-demo.vtt').write_text('\n'.join(vtt))
 (ROOT/'video/rt-safe-demo.srt').write_text('\n'.join(srt))
 (ROOT/'video/narration.txt').write_text('\n\n'.join(f"{stamp(s['start'])}  {s['chapter']}\n{s['narration']}" for s in scenes)+'\n')
 print('Film duration:',round(cursor,2),'seconds',flush=True)
asyncio.run(main())
