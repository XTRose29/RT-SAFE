"""Game-style overlays driven by original Task 19 decisions and phase reports."""
from pathlib import Path
import functools,math,sys
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'nyc/scripts'))
import replay_math as replay
W,H=960,1000
@functools.lru_cache(64)
def observation(path):return Image.open(ROOT/'public'/path).convert('RGB').resize((225,200),Image.Resampling.LANCZOS)
def command(step):
 a=step['action'];param=str(a.get('param',''))
 if a['type']=='move_to':return f'Move {step["commanded_distance_m"]:.0f} m  ·  waypoint {param}'
 if a['type']=='turn_around':return f'Turn {"right" if param.startswith("R") else "left"} {param[1:]}°'
 if a['type']=='wait':return f'Wait {param} s'
 return a['type'].replace('_',' ')
def events(model):
 result=[];start=model['steps'][0]['start_sim_seconds']
 for s in model['steps']:
  for phase,at in [('passive',s['start_sim_seconds']+s['inference_exposure_seconds']),('active',s['end_sim_seconds'])]:
   counts=s[phase+'_collisions'];n=sum(counts.values())
   if n:result.append({'at':(at-start)/6,'count':n,'phase':phase,'kinds':', '.join('pedestrian' if k=='human' else k for k,n in counts.items() if n)})
 return sorted(result,key=lambda e:e['at'])
def compose(frame,model,name,t,v,source="nyc"):
 duration=model['summary']['source_duration_seconds']/6;t=min(t,duration)
 state=replay.sample_agent(model,t*6);step=model['steps'][state['decision']-1];finished=state['finished'];counts=replay.counts_reported_by(model,t*6)
 if source=='recorded':
  # Preserve the complete horizontal field of view of the recorded camera.
  im=Image.new('RGB',(W,H),'#10243a');scaled=frame.resize((960,853),Image.Resampling.LANCZOS);im.paste(scaled,(0,73))
 else:im=frame.crop((442,0,1478,1080)).resize((W,H),Image.Resampling.LANCZOS)
 d=ImageDraw.Draw(im,'RGBA')
 for y in range(150):d.line((0,y,W,y),fill=(7,20,36,int(220*(1-y/175))))
 for y in range(770,H):d.line((0,y,W,y),fill=(7,20,36,int(160+70*(y-770)/230)))
 col='#93e0df' if name=='astra' else '#ffce8d';red='#ff6472';white='#ffffff';muted='#bdd0de'
 v.text(im,(28,24),'GPT-6 Astra' if name=='astra' else 'GPT-5.6 Sol',39,white,600)
 v.text(im,(30,77),'TASK 19  /  LOW REASONING  /  6×',18,col,600)
 v.text(im,(30,106),'GOAL: REACH THE FINAL SUBGOAL',16,white,500)
 v.text(im,(931,20),'COLLISIONS REPORTED',17,muted,600,'ra');v.text(im,(930,44),str(counts['total']).zfill(2),60,red if counts['total'] else white,600,'ra')
 if source=='nyc':
  v.rect(im,(28,133,257,371),'#122d48',8)
  im.paste(observation(step['input_image']),(30,165));v.text(im,(38,141),'INPUT · ORIGINAL FRAME',16,white,500)
  v.text(im,(30,382),f'Observation {step["source_decision"]}',17,white,500)
 else:
  v.rect(im,(28,133,524,174),'#122d48',7)
  label='FINAL RECORDED OUTPUT' if finished else 'ORIGINAL INPUT · OBSERVATION '+str(step['source_decision']) if state['phase']=='inference' else 'ORIGINAL ACTION SNAPSHOT'
  v.text(im,(40,144),label,18,white,600)
 phase='Segment complete' if finished else 'Thinking…' if state['phase']=='inference' else 'Acting'
 bw=v.font(30,600).getlength(phase)+38;bx=480-bw/2;by=685 if source=='recorded' else 454
 v.rect(im,(bx,by,bx+bw,by+56),'#142d4b',10,col,2);v.text(im,(bx+19,by+10),phase,30,white,600)
 if source=='nyc':v.line(im,[(480,510),(480,542)],col,3)
 elif state['phase']=='inference' and not finished:
  v.text(im,(480,749),'Input held during inference',18,white,500,'ma')
 relevant=[e for e in events(model) if e['at']<=t];last=relevant[-1] if relevant else None
 if last and not finished and t-last['at']<.9:
  v.rect(im,(5,5,W-6,H-6),None,6,red,7)
  alpha=round(90*(1-(t-last['at'])/.9));ImageDraw.Draw(im,'RGBA').rectangle((0,0,W,H),fill=(255,40,50,alpha//3))
  v.rect(im,(282,189 if source=='recorded' else 149,931,289 if source=='recorded' else 249),'#872b3c',10,red,2)
  v.text(im,(303,204 if source=='recorded' else 164),f'COLLISION REPORTED  +{last["count"]}',30,white,600)
  v.text(im,(305,250 if source=='recorded' else 210),'During inference' if last['phase']=='passive' else 'During action',23,'#ffcad0',500)
 v.text(im,(30,790),'MODEL OUTPUT',16,col,600)
 output='Route segment complete' if finished else command(step) if state['phase']=='action' else 'Choosing the next action…'
 v.text(im,(29,819),output,31,white,600)
 # Direction keys make commanded movement legible without inventing model reasoning.
 active=step['action']['type'] if state['phase']=='action' and not finished else ''
 for x,symbol,match in [(745,'turn',active=='turn_around'),(805,'move',active=='move_to'),(865,'wait',active=='wait')]:
  v.rect(im,(x,796,x+49,849),col if match else '#253d53',7)
  ink='#112b43' if match else '#c2d3df'
  if symbol=='move':v.arrow(im,(x+24,836),(x+24,808),ink,4)
  elif symbol=='turn':
   if str(step['action'].get('param','')).startswith('R'):
    v.line(im,[(x+14,836),(x+14,815),(x+34,815)],ink,4);v.arrow(im,(x+14,815),(x+37,815),ink,4)
   else:
    v.line(im,[(x+35,836),(x+35,815),(x+15,815)],ink,4);v.arrow(im,(x+35,815),(x+12,815),ink,4)
  else:
   v.line(im,[(x+18,810),(x+18,836)],ink,5);v.line(im,[(x+31,810),(x+31,836)],ink,5)
 v.text(im,(30,871),'ENVIRONMENT FEEDBACK',16,muted,600)
 if finished:feedback=f'{counts["active"]} while acting · {counts["passive"]} while thinking'
 elif last and t-last['at']<1.6:feedback=f'{last["kinds"].capitalize()} contact · '+('while thinking' if last['phase']=='passive' else 'while acting')
 else:feedback='No new collision report' if counts['total'] else 'No collision reported so far'
 v.text(im,(30,897),feedback,25,red if last and t-last['at']<1.6 and not finished else white,500)
 v.text(im,(30,943),f'Decision {state["decision"]:02} / {len(model["steps"])}',23,col,600)
 v.text(im,(930,943),f'{min(t*6,model["summary"]["source_duration_seconds"]):.1f} s simulation',23,white,500,'ra')
 v.rect(im,(30,980,930,985),'#34495c',2);v.rect(im,(30,980,30+900*t/duration,985),col,2)
 return im
