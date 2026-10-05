"""Render audited, silent snapshot excerpts from retained rollout manifests.

Usage: python build-model-examples.py --selection /path/to/selected.json [--stills]
Rebuild the published selections: python build-model-examples.py --workspace /path/to/rollout-archives

Selection entries identify name, difficulty, task, map, and three consecutive
steps (each with a manifest path). No model responses or prompts are published.
Requires Pillow and ffmpeg. Output uses the last input image: the current,
annotated observation, rather than earlier images from the previous action.
"""
from pathlib import Path
import argparse, bisect, functools, hashlib, json, math, subprocess
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
FPS,SPEED,W,H=30,2,960,1000
OUT=ROOT/'public/media/model-examples'
COLORS={'Astra':'#93e0df','Sol':'#ffce8d','Fable':'#c59be0','Sonnet':'#efabd0','Gemini':'#f9d575','DeepSeek':'#f4b783','Inkling':'#bfdfa2','Grok':'#ffa3af'}
MARKS={'Astra':'openai','Sol':'openai','Fable':'claude','Sonnet':'claude','Gemini':'gemini','DeepSeek':'deepseek','Inkling':'thinking-machines','Grok':'grok'}
HEADLINES={'Astra':'Consecutive moves, no turn or wait.','Sol':'Move, turn, then wait.','Fable':'Long moves followed by a wait.','Sonnet':'Quick responses through a busy sidewalk.','Gemini':'A turn followed by short moves.','DeepSeek':'Contacts reported while thinking.','Inkling':'Three consecutive four-meter moves.','Grok':'Long inference, repeated contact reports.'}
RATIONALES={'Astra':'Illustrates its tendency to move with relatively little turning or waiting.','Sol':'Shows the turning and explicit waiting that distinguish its aggregate action mix.','Fable':'Shows an explicit wait after movement, one of the behaviors in its profile.','Sonnet':'Shows response times close to its 4.9-second average and an excerpt with no recorded contacts.','Gemini':'Shows a turn and shorter commanded moves, consistent with its aggregate action mix.','DeepSeek':'Shows long inference intervals and contacts reported during inference.','Inkling':'Shows long commanded moves that cover more distance per movement decision.','Grok':'Shows response times near its aggregate average and repeated contacts during inference.'}
@functools.lru_cache(80)
def font(size,weight=500):return ImageFont.truetype(str(ROOT/f'public/fonts/space-grotesk-{weight}.ttf'),size)
def text(im,xy,s,size=24,fill='white',weight=500,anchor=None):ImageDraw.Draw(im).text(xy,str(s),font=font(size,weight),fill=fill,anchor=anchor)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command(action,distance):
 kind=action['type'];param=str(action.get('param',''))
 if kind=='move_to':return f'Move {distance:g} m · waypoint {param}'
 if kind in ('turn','turn_around'):return f'Turn {"right" if param.startswith("R") else "left"} {param[1:]}°'
 if kind=='wait':return f'Wait {param} s'
 raise ValueError(action)
def curate(c):
 name=c['name'];slug=name.lower();steps=[];schedule=[];events=[];origin=None
 for chosen in c['steps']:
  mp=Path(chosen['manifest']);
  if chosen.get('expectedSha256'):assert sha(mp)==chosen['expectedSha256'], 'Source manifest changed'
  m=json.loads(mp.read_text());t=m['timing'];start=t['sim_time_start_seconds'];end=t['sim_time_end_seconds'];inf=t['simulation_thinking_latency_seconds']
  if origin is None:origin=start
  a=m['model_output']['parsed_action'];action={k:a[k] for k in ('type','param')};assert action['type'] in ('move_to','turn','turn_around','wait')
  assert not m.get('safety_filter_overrode_action'), 'Cannot label an overridden action as the model action'
  distance=round((m.get('waypoint_execution') or {}).get('commanded_waypoint_distance_cm',0)/100,3)
  inputs=[mp.parent/f for f in m['input_images']];captures=[mp.parent/f for f in m['recording_action_frames']]
  assert all(p.is_file() for p in inputs+captures) and captures
  def evidence(p):return {'file':p.name,'sha256':sha(p)}
  schedule.append(((start-origin)/SPEED,str(inputs[-1])))
  for i,p in enumerate(captures):schedule.append(((start+inf+(end-start-inf)*i/len(captures)-origin)/SPEED,str(p)))
  schedule.append(((end-origin)/SPEED,str(captures[-1])))
  active={k:int((m.get('collision_details') or {}).get(k,0)) for k in ('human','object','building','vehicle')}
  passive={k:int((m.get('passive_collision_details') or {}).get(k,0)) for k in active}
  hazards={k:int((m.get('collision_details') or {}).get(k,0)) for k in ('fall','oil','water')}
  traffic={k:len((m.get('safety_events') or {}).get(k,[])) for k in ('red_light_violations','illegal_crossing_violations')}
  for phase,counts,at in [('inference',passive,start+inf),('action',active,end)]:
   if sum(counts.values()):events.append({'at':round((at-origin)/SPEED,6),'phase':phase,'count':sum(counts.values()),'kinds':', '.join('pedestrian' if k=='human' else k for k,v in counts.items() if v)})
  steps.append({'decision':m['decision_index'],'sourceStep':m['step'],'start':start-origin,'end':end-origin,'inference':inf,'responseSeconds':m['metrics']['response_time'],'action':action,'command':command(action,distance),'commandedMeters':distance,'active':active,'passive':passive,'hazards':hazards,'traffic':traffic,'sourceRelativePath':'/'.join(mp.parts[mp.parts.index('Safety-Claude') if 'Safety-Claude' in mp.parts else mp.parts.index('SimWorld-RealTime-results'):]),'manifestSha256':sha(mp),'inputFrames':[evidence(p) for p in inputs],'actionFrames':[evidence(p) for p in captures]})
 assert all(b['sourceStep']==a['sourceStep']+1 and abs(b['start']-a['end'])<.1 for a,b in zip(steps,steps[1:]))
 safety_reports=[]
 for s in steps:
  for kind,counts in [('hazard',s['hazards']),('traffic violation',s['traffic'])]:
   if sum(counts.values()):safety_reports.append({'at':s['end']/SPEED,'phase':'action','count':sum(counts.values()),'kinds':', '.join(k.replace('_',' ') for k,v in counts.items() if v),'kind':kind})
 duration=steps[-1]['end'];models=json.loads((ROOT/'app/data.json').read_text())['models'];full=next(m['fullName'] for m in models if m['name']==name)
 path=Path(c['steps'][0]['manifest']);parts=path.parts;campaign=next(i for i,p in enumerate(parts) if p.startswith('main_table_'))
 public={'name':name,'slug':slug,'fullName':full,'headline':HEADLINES[name],'rationale':RATIONALES[name],'difficulty':c['difficulty'],'task':c['task'],'map':c['map'],'seed':0,'reasoning':'Provider default','speed':SPEED,'sourceDuration':round(duration,3),'videoDuration':math.ceil((duration/SPEED+2)*FPS)/FPS,'meanResponse':round(sum(s['responseSeconds'] for s in steps)/len(steps),3),'contacts':sum(e['count'] for e in events),'inferenceContacts':sum(e['count'] for e in events if e['phase']=='inference'),'actionContacts':sum(e['count'] for e in events if e['phase']=='action'),'commandedMeters':sum(s['commandedMeters'] for s in steps if s['action']['type']=='move_to'),'hazards':sum(sum(s['hazards'].values()) for s in steps),'trafficViolations':sum(sum(s['traffic'].values()) for s in steps),'startSimulationSeconds':origin,'sourceCampaign':'/'.join(parts[campaign:-3]),'sourceDecisions':[s['decision'] for s in steps],'video':f'media/model-examples/{slug}.mp4','poster':f'media/model-examples/{slug}.webp','steps':steps,'events':events,'safetyReports':safety_reports}
 # Later entries supersede action snapshots at the next observation boundary.
 schedule=sorted(dict(schedule).items())
 return public,schedule
@functools.lru_cache(64)
def picture(path):return Image.open(path).convert('RGB').resize((960,853),Image.Resampling.LANCZOS)
@functools.lru_cache(8)
def logo(name):
 p=Image.open(ROOT/f'public/media/logos/{MARKS[name]}.png').convert('RGBA');p.thumbnail((35,35));return p

def panel(m,schedule,t):
 st=min(t*SPEED,m['sourceDuration']);step=next((s for s in m['steps'] if s['end']>st),m['steps'][-1]);done=st>=m['sourceDuration'];phase='inference' if st<step['start']+step['inference'] else 'action';col=COLORS[m['name']]
 idx=max(0,bisect.bisect_right([a for a,_ in schedule],t)-1);at,path=schedule[idx];frame=picture(path)
 if idx and t-at<.12:
  span=min(.12,(schedule[idx+1][0]-at)*.8) if idx+1<len(schedule) else .12
  f=min(1,max(0,(t-at)/max(.001,span)));frame=Image.blend(picture(schedule[idx-1][1]),frame,f*f*(3-2*f))
 im=Image.new('RGB',(W,H),'#091c2d');im.paste(frame,(0,73));d=ImageDraw.Draw(im,'RGBA')
 for y in range(180):d.line((0,y,W,y),fill=(7,20,36,round(225*(1-y/220))))
 for y in range(775,H):d.line((0,y,W,y),fill=(7,20,36,min(250,round(180+(y-775)*.45))))
 d.rounded_rectangle((26,23,73,70),8,fill='white');mark=logo(m['name']);im.paste(mark,(32,29),mark)
 text(im,(86,23),m['fullName'],34,weight=600)
 text(im,(30,83),f'TASK {m["task"]}  /  {m["difficulty"].upper()}  /  DEFAULT REASONING  /  {SPEED}×',16,col,600)
 text(im,(930,112),'CONTACTS IN EXCERPT',16,'#c5d5e1',600,'ra')
 reported=[e for e in m['events'] if e['at']<=t];count=sum(e['count'] for e in reported)
 text(im,(930,134),str(count).zfill(2),54,'#ff7280' if count else 'white',600,'ra')
 d.rounded_rectangle((26,124,580,161),6,fill='#122d48')
 text(im,(38,134),f'ORIGINAL INPUT · DECISION {step["decision"]}' if phase=='inference' and not done else 'ORIGINAL ACTION SNAPSHOT',16,weight=600)
 phaseLabel='Excerpt complete' if done else 'Thinking…' if phase=='inference' else 'Acting'
 bw=font(29,600).getlength(phaseLabel)+36;x=(960-bw)/2
 d.rounded_rectangle((x,696,x+bw,749),9,fill='#142d4b',outline=col,width=2);text(im,(x+18,706),phaseLabel,29,weight=600)
 if phase=='inference' and not done:text(im,(480,756),'Current input held during inference',17,'#d4e1ea',500,'ma')
 other=[e for e in m['safetyReports'] if e['at']<=t]
 alerts=sorted([dict(e,kind='collision') for e in reported]+other,key=lambda e:e['at'])
 last=alerts[-1] if alerts else None
 if last and t-last['at']<1.2:
  d.rounded_rectangle((5,5,954,994),6,outline='#ff6978',width=7)
  d.rounded_rectangle((250,222,930,309),9,fill='#872b3c',outline='#ff6978',width=2)
  text(im,(270,234),f'{last["kind"].upper()} REPORTED +{last["count"]}',28,weight=600)
  text(im,(270,275),f'{last["kinds"].capitalize()} · during {last["phase"]}',20,'#ffd4d9')
 text(im,(30,795),'MODEL OUTPUT',15,col,600)
 output='Excerpt complete' if done else step['command'] if phase=='action' else 'Choosing the next action…'
 text(im,(30,822),output,29,weight=600)
 text(im,(30,871),'ENVIRONMENT FEEDBACK',15,'#c5d5e1',600)
 feedback='No contact reported in this excerpt' if count==0 else f'{sum(e["count"] for e in reported if e["phase"]=="inference")} during inference · {sum(e["count"] for e in reported if e["phase"]=="action")} during action'
 if other:feedback+=f' · {sum(e["count"] for e in other)} hazard report'
 text(im,(30,897),feedback,21,'#ffb0b8' if count else 'white')
 text(im,(30,945),f'Decision {step["decision"]} · excerpt {m["sourceDecisions"][0]}–{m["sourceDecisions"][-1]}',20,col)
 text(im,(930,945),f'{st:.1f} / {m["sourceDuration"]:.1f} s simulation',20,anchor='ra')
 d.rounded_rectangle((30,982,930,987),2,fill='#34495c');d.rounded_rectangle((30,982,30+max(1,900*st/m['sourceDuration']),987),2,fill=col)
 return im

def render(c,stills=False):
 m,s=curate(c);OUT.mkdir(parents=True,exist_ok=True);slug=m['slug'];panel(m,s,0).save(OUT/f'{slug}.webp',quality=92)
 if stills:
  dest=ROOT.parent/'runtime/model-examples/previews';dest.mkdir(parents=True,exist_ok=True)
  for t in [0,(m['steps'][0]['inference']+.5)/SPEED, m['events'][0]['at']+.1 if m['events'] else m['sourceDuration']/SPEED]:panel(m,s,t).save(dest/f'{slug}-{t:.1f}.png')
 else:
  cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','22','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/f'{slug}.mp4')]
  p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
  for i in range(round(m['videoDuration']*FPS)):p.stdin.write(panel(m,s,i/FPS).tobytes())
  p.stdin.close();assert p.wait()==0
 print(m['name'],round(m['videoDuration'],1),'seconds',flush=True)
 return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path);ap.add_argument('--workspace',type=Path);ap.add_argument('--stills',action='store_true');ap.add_argument('--models',nargs='*');args=ap.parse_args();selected=json.loads(args.selection.read_text()) if args.selection else []
 if not selected:
  if not args.workspace:ap.error('Provide --selection or --workspace containing the original rollout archives')
  for m in json.loads((ROOT/'app/model-examples.json').read_text())['models']:
   selected.append(dict(name=m['name'],difficulty=m['difficulty'],task=m['task'],map=m['map'],steps=[{'manifest':str(args.workspace/s['sourceRelativePath']),'expectedSha256':s['manifestSha256']} for s in m['steps']]))
 if args.models:selected=[c for c in selected if c['name'] in args.models]
 models=[render(c,args.stills) for c in selected]
 if len(models)==8:
  data={'version':1,'selection':'Hand-reviewed, continuous three-decision excerpts from the 864 finalized provider-default episodes, chosen to illustrate behaviors in the aggregate profiles. These are qualitative examples, not matched tasks or estimates of average performance. Retained first/last frames limit available excerpts.','framePolicy':'Current input is the last image in the original model input sequence. It holds during inference. Saved action snapshots are evenly spaced through the recorded action interval because exact capture timestamps are unavailable. Short 0.12-second dissolves soften transitions. Collision counts appear at phase-report boundaries. No synthesized scene frames.','models':models}
  for dest in [ROOT/'app/model-examples.json',ROOT/'public/data/model-examples.json']:dest.write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':main()
