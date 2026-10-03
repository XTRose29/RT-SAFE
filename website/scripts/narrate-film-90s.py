"""Build the fixed 90-second edit and synchronized speech/captions."""
import asyncio, json, subprocess, re, html
from pathlib import Path
import edge_tts
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'video/revision-90s'; AUDIO=OUT/'audio'
SCENES=[
 ('title',4,'RT-Safe','R T Safe.'),
 ('timing',11,'THE WORLD KEEPS MOVING','The world does not pause while an agent thinks. Static evaluation freezes the world during inference. In real time, pedestrians keep moving, and a clear path can become unsafe.'),
 ('environment',12,'NAVIGATE TO THE GOAL SAFELY','Navigate to the goal across five maps and thirty-six routes. Safety includes collisions, trip, oil and water hazards, and traffic-rule violations.'),
 ('examples',14,'RECORDED AGENT DECISIONS','These are recorded decisions. Gemini moves two meters. Sonnet also chooses a short step. Fable waits for a pedestrian. All three use the same visual observation and action interface.'),
 ('behavior',10,'MODEL BEHAVIOR','Across all difficulty levels, Sonnet records the fewest collisions. Inference exposure strongly tracks collisions, but models differ in how they move, turn, and wait.'),
 ('completion',8,'TASK COMPLETION IS NOT ENOUGH','Task completion is not enough. Success reaches ninety-four point four percent, but safe success is only zero point seven percent.'),
 ('comparison',9,'STATIC VS. REAL-TIME','On matched hard routes, real-time collisions rise twelve point three times. About eighty-four percent occur while the agent is deciding.'),
 ('reasoning',10,'LOWER / DEFAULT / HIGHER','More reasoning is not reliably safer. From default to higher effort, collisions rise from forty point seven to sixty-two per episode.'),
 ('learning',12,'AN ENVIRONMENT FOR LEARNING','R T Safe also supports offline reinforcement learning. Reward design changes safety and progress. The wander penalty achieves seventy-five percent success, while behavior cloning has the lowest collision rate.'),
]
def stamp(t,sep='.'):
 ms=round(t*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}{sep}{ms%1000:03}'
async def voice(scene):
 name,dur,chapter,text=scene;p=AUDIO/(name+'.mp3');events=AUDIO/(name+'.words.json')
 if p.exists() and events.exists():return
 for attempt in range(3):
  try:
   words=[]
   with p.open('wb') as f:
    async for c in edge_tts.Communicate(text,'en-US-AriaNeural',rate='+6%',pitch='-2Hz',boundary='WordBoundary').stream():
     if c['type']=='audio':f.write(c['data'])
     elif c['type']=='WordBoundary':words.append({'start':c['offset']/1e7,'duration':c['duration']/1e7,'text':html.unescape(c['text'])})
   events.write_text(json.dumps(words));print('Narrated',name,flush=True);return
  except Exception:
   if attempt==2:raise
async def main():
 AUDIO.mkdir(parents=True,exist_ok=True)
 for i in range(0,len(SCENES),3):await asyncio.gather(*(voice(s) for s in SCENES[i:i+3]))
 scenes=[];cursor=0;vtt=['WEBVTT',''];srt=[];count=1
 for name,dur,chapter,text in SCENES:
  p=AUDIO/(name+'.mp3');audio_duration=float(subprocess.check_output(['ffprobe','-v','quiet','-show_entries','format=duration','-of','csv=p=0',str(p)]))
  speed=max(1,audio_duration/(dur-.65));assert speed<1.25,(name,speed)
  words=json.loads((AUDIO/(name+'.words.json')).read_text());tokens=text.split()
  if len(tokens)==len(words):
   for word,token in zip(words,tokens):word['text']=token
  groups=[];group=[]
  for word in words:
   group.append(word)
   if len(group)>=8 or re.search(r'[.!?]$',word['text']):groups.append(group);group=[]
  if group:groups.append(group)
  for gi,g in enumerate(groups):
   start=cursor+.3+g[0]['start']/speed;end=min(cursor+dur-.12,cursor+.3+(g[-1]['start']+g[-1]['duration'])/speed+.08)
   if gi+1<len(groups):end=min(end,cursor+.3+groups[gi+1][0]['start']/speed-.01)
   line=' '.join(w['text'] for w in g)
   vtt.extend([f'{stamp(start)} --> {stamp(end)}',line,'']);srt.extend([str(count),f'{stamp(start,",")} --> {stamp(end,",")}',line,'']);count+=1
  scenes.append({'id':name,'chapter':chapter,'start':cursor,'duration':dur,'narration':text,'voiceOffset':.3,'voiceDuration':audio_duration,'voiceSpeed':speed});cursor+=dur
 assert cursor==90
 (OUT/'timeline.json').write_text(json.dumps({'title':'RT-Safe: Benchmarking Agent Safety in Real-Time Embodied Environment','subtitle':'The world does not pause while an agent thinks.','duration':90,'fps':30,'width':1920,'height':1080,'voice':'en-US-AriaNeural','scenes':scenes},indent=2)+'\n')
 (OUT/'captions.vtt').write_text('\n'.join(vtt));(OUT/'captions.srt').write_text('\n'.join(srt))
 (OUT/'narration.txt').write_text('\n\n'.join(f"{stamp(s['start'])}  {s['chapter']}\n{s['narration']}" for s in scenes)+'\n')
 print('Fixed duration: 90.0 seconds',[(s['id'],round(s['voiceDuration'],2),round(s['voiceSpeed'],3)) for s in scenes],flush=True)
if __name__=='__main__':asyncio.run(main())
