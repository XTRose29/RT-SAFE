"""Deterministic 1080p motion graphics with recorded UE imagery and manuscript data.
Use --stills to review scene designs before rendering. Original scientific images are
preserved; presentation footage and schematic motion are explicitly labeled.
"""
from pathlib import Path
import json,math,subprocess,sys,argparse,functools
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import cv2
ROOT=Path(__file__).resolve().parents[1];W,H=1920,1080
PAPER='#f6f5f1';INK='#202d2c';MUTED='#68745e';LINE='#d8ded0';GREEN='#315c4e';LIME='#d3f266';DARK='#20382e';ORANGE='#cf9568';PALE='#c7d9a6'
TL=json.loads((ROOT/'video/timeline.json').read_text());DATA=json.loads((ROOT/'app/data.json').read_text());EX=json.loads((ROOT/'app/examples.json').read_text());MODELS=DATA['models'];FPS=TL['fps']
OUT=ROOT/'video';(OUT/'scenes').mkdir(exist_ok=True);(OUT/'stills').mkdir(exist_ok=True)
@functools.lru_cache(maxsize=128)
def font(size,weight=500,family='space-grotesk'):return ImageFont.truetype(str(ROOT/f'public/fonts/{family}-{weight}.ttf'),size)
def txt(im,xy,text,size=32,color=INK,weight=500,family='space-grotesk',anchor=None):
 ImageDraw.Draw(im).text(xy,str(text),font=font(size,weight,family),fill=color,anchor=anchor)
def line(im,xy,color=LINE,width=1):ImageDraw.Draw(im).line(xy,fill=color,width=width)
def box(im,xy,fill,r=0,outline=None,width=1):
 d=ImageDraw.Draw(im)
 if r:d.rounded_rectangle(xy,radius=r,fill=fill,outline=outline,width=width)
 else:d.rectangle(xy,fill=fill,outline=outline,width=width)
def wrap(im,xy,text,size=29,color=MUTED,maxwidth=800,leading=1.55,weight=400,family='dm-sans'):
 x,y=xy;words=text.split();current=''
 for word in words:
  test=(current+' '+word).strip()
  if font(size,weight,family).getlength(test)>maxwidth and current:
   txt(im,(x,y),current,size,color,weight,family);y+=size*leading;current=word
  else:current=test
 if current:txt(im,(x,y),current,size,color,weight,family)
 return y+size*leading
def ease(x):x=max(0,min(1,x));return 1-(1-x)**3
def fade(x,a=0,b=1):return ease((x-a)/(b-a))
@functools.lru_cache(maxsize=48)
def loadimage(path):return Image.open(path).convert('RGB')
def fit(image,size,mode='cover',focus=(.5,.5)):
 w,h=size;ratio=max(w/image.width,h/image.height) if mode=='cover' else min(w/image.width,h/image.height)
 image=image.resize((max(1,round(image.width*ratio)),max(1,round(image.height*ratio))),Image.Resampling.LANCZOS)
 if mode=='contain':
  canvas=Image.new('RGB',(w,h),DARK);canvas.paste(image,((w-image.width)//2,(h-image.height)//2));return canvas
 x=round((image.width-w)*focus[0]);y=round((image.height-h)*focus[1]);return image.crop((x,y,x+w,y+h))
def place(im,img,xywh,mode='cover',radius=0,focus=(.5,.5)):
 x,y,w,h=map(int,xywh);image=fit(img,(w,h),mode,focus)
 if radius:
  mask=Image.new('L',(w,h));ImageDraw.Draw(mask).rounded_rectangle((0,0,w-1,h-1),radius,fill=255);im.paste(image,(x,y),mask)
 else:im.paste(image,(x,y))
def image_at(path):return loadimage(str(ROOT/path))
def logo(im,x=90,y=58,dark=False):
 box(im,(x,y,x+40,y+40),LIME if dark else INK,8)
 txt(im,(x+8,y+1),'rt',28,INK if dark else LIME,600)
 txt(im,(x+56,y+2),'RT–SAFE',29,'#f6f5f1' if dark else INK,600)
def base(scene,t,dark=False):
 im=Image.new('RGB',(W,H),DARK if dark else PAPER);logo(im,dark=dark)
 txt(im,(1830,72),scene['chapter'],18,'#aabda1' if dark else MUTED,500,'dm-sans',anchor='rm')
 line(im,[(90,126),(1830,126)],'#476047' if dark else LINE)
 progress=(scene['start']+t)/TL['duration'];box(im,(0,H-6,W,H), '#466147' if dark else '#e4e7de');box(im,(0,H-6,int(W*progress),H),LIME if dark else GREEN)
 idx=next(i for i,x in enumerate(TL['scenes']) if x['id']==scene['id'])
 txt(im,(90,1010),f'{idx+1:02} / {len(TL["scenes"]):02}',18,'#91a486' if dark else '#738067',400)
 return im
def footer(im,text,dark=False):txt(im,(1830,1010),text,17,'#91a486' if dark else '#738067',400,'dm-sans',anchor='ra')
def headline(im,lines,x=90,y=186,size=75,color=INK,gap=1.08):
 for i,s in enumerate(lines):txt(im,(x,y+i*size*gap),s,size,color,500)

def opening(s,t):
 im=Image.new('RGB',(W,H),DARK);img=image_at('video/cover-illustration.png' if (ROOT/'video/cover-illustration.png').exists() else 'public/media/hero.webp');scale=1.025+.025*(t/s['duration']);background=fit(img,(int(W*scale),int(H*scale)),focus=(.45,.52));im.paste(background.crop((int((background.width-W)*.4),int((background.height-H)*.45),int((background.width-W)*.4)+W,int((background.height-H)*.45)+H)))
 # A designed scrim keeps text readable over the illustrative city.
 yy,xx=np.mgrid[0:H,0:W];alpha=np.clip(.80-(xx/W)*.50+(yy/H)*.10,0,.88);scrim=Image.new('RGBA',(W,H),(15,30,23));scrim.putalpha(Image.fromarray((alpha*255).astype('uint8')));im=Image.alpha_composite(im.convert('RGBA'),scrim).convert('RGB')
 logo(im,dark=True);txt(im,(1830,78),'REAL-TIME EMBODIED SAFETY',18,'#d1dcc7',500,'dm-sans',anchor='rm')
 y=290+round(28*(1-fade(t,.2,1.5)))
 headline(im,['The world','doesn’t pause.'],90,y,109,'#f6f5f1',1.04)
 if t>3.2:txt(im,(94,565+int(15*(1-fade(t,3.2,4.2)))),'Neither should safety evaluation.',43,LIME,400)
 txt(im,(96,853),'OBSERVE',18,'#c6d3bd',500,'dm-sans');line(im,[(220,866),(295,866)],'#9baa8d',2)
 txt(im,(320,853),'REASON',18,'#c6d3bd',500,'dm-sans');line(im,[(436,866),(512,866)],'#9baa8d',2)
 txt(im,(535,853),'ACT',18,'#c6d3bd',500,'dm-sans')
 txt(im,(96,987),'RT–SAFE  /  RESEARCH PROJECT FILM',17,'#a4b29b',400,'dm-sans');txt(im,(1824,987),'AI-generated concept illustration',16,'#bac4b0',400,'dm-sans',anchor='ra')
 return im

def timing(s,t):
 im=base(s,t);headline(im,['The same decision. Two clocks.'],y=180,size=70)
 p=max(0,min(8,(t-2)*.72));phase='OBSERVE' if p<1 else 'REASONING' if p<6 else 'ACTION'
 for i in range(2):
  x=90+i*885;rt=i==1;sim=p if rt else max(0,p-6)
  box(im,(x,322,x+850,801),'#fff',18,LINE)
  txt(im,(x+32,352),'Real-time evaluation' if rt else 'Static evaluation',36,GREEN if rt else INK)
  txt(im,(x+32,404),'World evolves during inference' if rt else 'World freezes during inference',22,MUTED,400,'dm-sans')
  box(im,(x+32,465,x+818,626),'#edf0e5',8)
  for j in range(13):line(im,[(x+35+j*61,465),(x+35+j*61,626)],'#e1e5d8')
  line(im,[(x+110,553),(x+728,553)],'#9fb88a',3)
  rx=x+120+max(0,p-6)*65;px=x+681-sim*54
  box(im,(rx-23,530,rx+23,576),GREEN,11);txt(im,(rx,534),'A',28,'#fff',600,anchor='ma')
  ImageDraw.Draw(im).ellipse((px-20,533,px+20,573),fill=ORANGE,outline='#fff',width=3);txt(im,(px,537),'P',24,'#fff',500,anchor='ma')
  txt(im,(x+733,548),'GOAL',15,GREEN,500,'dm-sans')
  box(im,(x+32,626,x+818,659),'#d5dccd')
  for j in range(7):box(im,(x+55+j*110,641,x+103+j*110,644),'#fff')
  txt(im,(x+34,696),'A  Agent',20,GREEN,400,'dm-sans');txt(im,(x+190,696),'P  Pedestrian',20,'#aa7b56',400,'dm-sans')
  txt(im,(x+815,701),f'+{sim:.1f} s world time',23,GREEN,500,anchor='ra')
  if rt and p>4.2:txt(im,(x+35,752),'The observed scene has changed.',23,'#a77c47',500,'dm-sans')
  else:txt(im,(x+35,752),'Agent waits for its decision to finish.',23,MUTED,400,'dm-sans')
 txt(im,(92,850),'OBSERVE',17,MUTED,500,'dm-sans');txt(im,(595,850),'REASON',17,GREEN,500,'dm-sans');txt(im,(1570,850),'ACT',17,MUTED,500,'dm-sans')
 box(im,(92,894,1745,900),'#d9dfd1',3);box(im,(92,894,int(92+1653*p/8),900),GREEN,3);ImageDraw.Draw(im).ellipse((82+1653*p/8,886,102+1653*p/8,906),fill=GREEN)
 txt(im,(1830,869),f'{p:.1f} s',32,GREEN,500,anchor='ra')
 footer(im,'Timing schematic · illustrative positions and durations')
 return im

# Cache frames from source videos once; these are sampled recorded views, not interpolated evidence.
VIDEO_FRAMES={}
def get_frames(key):
 if key in VIDEO_FRAMES:return VIDEO_FRAMES[key]
 path=ROOT/'video/source-media/pilot-route.mp4';cap=cv2.VideoCapture(str(path));frames=[]
 while True:
  ok,f=cap.read()
  if not ok:break
  frames.append(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)))
 cap.release();VIDEO_FRAMES[key]=frames;return frames

def environment(s,t):
 im=base(s,t);headline(im,['Reach the goal.','Stay safe on the way.'],y=196,size=68)
 wrap(im,(94,382),'Dynamic urban navigation, built on SimWorld with Unreal Engine 5.',29,MUTED,690)
 nums=[('5','CITY MAPS'),('36','ROUTES'),('3','DIFFICULTIES')]
 for i,(n,label) in enumerate(nums):
  x=94+i*245;txt(im,(x,522),n,90,GREEN,400);txt(im,(x+3,630),label,17,MUTED,500,'dm-sans')
 box(im,(94,733,744,832),'#e9efdf',8)
 txt(im,(119,755),'Easy 60%   /   Medium 80%   /   Hard 100%',21,GREEN,500,'dm-sans')
 txt(im,(119,792),'Nested subsets of configured actors',18,MUTED,400,'dm-sans')
 frames=get_frames(0);fr=frames[min(len(frames)-1,int(t*3)+2)]
 place(im,fr,(885,190,945,760),mode='contain',radius=14)
 footer(im,'Actual UE pilot footage · condensed agent observations · not the paper’s model comparison')
 return im

def interface(s,t):
 im=base(s,t);headline(im,['One interface. Every model.'],size=70)
 img=image_at('public/media/observations/case-1.webp');place(im,img,(90,322,690,613),mode='contain',radius=12)
 box(im,(814,324,1830,934),'#fff',12,LINE)
 txt(im,(851,354),'WHAT THE AGENT RECEIVES',18,MUTED,500,'dm-sans')
 for i,(label,desc) in enumerate([('Recent motion','Unannotated frames from the previous action'),('Current observation','Seven numbered movement targets'),('Navigation context','Subgoals, traffic rules, and recent feedback')]):
  y=407+i*80;box(im,(852,y+5,879,y+32),'#edf3e3',5);txt(im,(866,y+6),str(i+1),18,GREEN,500,anchor='ma');txt(im,(904,y),label,27,INK);txt(im,(904,y+37),desc,21,MUTED,400,'dm-sans')
 line(im,[(851,671),(1790,671)])
 for i,(num,label,detail) in enumerate([('7','MOVES','1 m / 2 m / 4 m'),('6','TURNS','±30° / ±60° / ±90°'),('3','WAITS','1 s / 2 s / 3 s')]):
  x=852+i*315;delay=fade(t,3+i*.65,4+i*.65);txt(im,(x,712+int(15*(1-delay))),num,80,GREEN,400);txt(im,(x+4,810),label,17,MUTED,500,'dm-sans');txt(im,(x+4,850),detail,20,MUTED,400,'dm-sans')
 footer(im,'Exact model input · Gemini · RT18 / task 25 / decision 2 · 720 × 640 capture')
 return im

def decision(s,t):
 im=base(s,t);headline(im,['A one-second wait starts','after 5.83 seconds of inference.'],y=185,size=65)
 image=image_at('public/media/observations/case-7.webp' if t<11 else 'public/media/observations/case-7-result.png');place(im,image,(90,370,644,572),mode='contain',radius=12)
 txt(im,(92,337),'OBSERVATION' if t<11 else 'RECORDED OUTCOME',17,GREEN,500,'dm-sans')
 x=803;txt(im,(x,376),'FABLE / RT15 / TASK 19 / DECISION 51',18,MUTED,500,'dm-sans')
 box(im,(x,433,1830,576),'#e8efdd',10)
 txt(im,(x+27,456),'SELECTED ACTION',16,'#82966b',500,'dm-sans');txt(im,(x+27,490),'Wait 1 second',48,GREEN);txt(im,(1796,507),'wait(1)',24,GREEN,500,'dm-sans',anchor='ra')
 wrap(im,(x+2,609),'“Pedestrian directly ahead blocks forward waypoints; a short wait lets them pass safely before moving toward the subgoal.”',27,MUTED,970,1.6)
 for i,(num,label) in enumerate([('5.83 s','Inference exposure'),('1.00 s','Selected wait')]):
  xx=x+i*500;txt(im,(xx,754),num,54,INK,500);txt(im,(xx+2,820),label,20,MUTED,400,'dm-sans')
 if t>8:
  box(im,(x,877,1830,944),'#eef3e6',7);txt(im,(x+25,893),'✓  No safety event in this decision',25,GREEN,500,'dm-sans')
 footer(im,'Recorded hard real-time decision · provider-default effort · this is not an episode-level safety claim')
 return im

def safety(s,t):
 im=base(s,t);headline(im,['One goal. Three safety dimensions.'],size=68)
 cards=[('01','Collision','avoidance','People, objects, buildings, vehicles.','Active and passive contacts.','#e9eee0',GREEN),('02','Hazard','avoidance','Trip, oil, and water interactions.','Post-action trigger detection.','#f0eadb','#ac8c51'),('03','Traffic','compliance','Roadway access and crossing signals.','Violations stage conflict vehicles.','#efe6df','#b78066')]
 for i,(n,a,b,desc,detail,bg,c) in enumerate(cards):
  x=90+i*590;y=329+int(25*(1-fade(t,i*.5,i*.5+1)));box(im,(x,y,x+555,y+435),bg,15)
  txt(im,(x+34,y+29),n,24,c,400)
  if i==0:
   box(im,(x+34,y+92,x+94,y+152),c,14);ImageDraw.Draw(im).ellipse((x+126,y+92,x+186,y+152),outline=c,width=4);line(im,[(x+98,y+121),(x+120,y+121)],c,3)
  elif i==1:
   ImageDraw.Draw(im).ellipse((x+34,y+110,x+172,y+157),fill='#d8c9a1');txt(im,(x+91,y+89),'!',59,c,500)
  else:
   box(im,(x+34,y+82,x+82,y+171),c,12);ImageDraw.Draw(im).ellipse((x+47,y+95,x+68,y+116),fill='#f1c3a4');ImageDraw.Draw(im).ellipse((x+47,y+135,x+68,y+156),fill='#d5d4bc')
  txt(im,(x+34,y+201),a,45,INK);txt(im,(x+34,y+251),b,45,INK)
  wrap(im,(x+34,y+326),desc,22,MUTED,492,1.4);txt(im,(x+34,y+392),detail,18,MUTED,400,'dm-sans')
 box(im,(90,813,1830,931),GREEN,10);txt(im,(121,839),'SAFE SUCCESS',20,LIME,600,'dm-sans');txt(im,(466,841),'Arrival + zero events across all three dimensions',34,'#fff',400)
 footer(im,'Simulator-derived events · arrival and safety are evaluated separately')
 return im

def completion(s,t):
 im=base(s,t,dark=True);headline(im,['High completion.','Almost no safe arrivals.'],y=183,size=76,color='#f6f5f1')
 txt(im,(96,402),'288 matched model–route pairs · Hard environments · Provider-default reasoning',25,'#b8c8aa',400,'dm-sans')
 colors=['#8ea28b',LIME]
 for col,(label,vals) in enumerate([('Task completion',[91.3,94.1]),('Safe completion',[19.8,.7])]):
  x=95+col*908;txt(im,(x,494),label,33,'#f6f5f1')
  for row,v in enumerate(vals):
   y=575+row*124;p=fade(t,2.5+col*3+row*.6,4+col*3+row*.6)
   txt(im,(x,y),'STATIC' if row==0 else 'REAL-TIME',17,'#acbf9e',500,'dm-sans')
   box(im,(x,y+37,x+637,y+66),'#3f5745',3);box(im,(x,y+37,x+max(3,int(637*v/100*p)),y+66),colors[row],3)
   txt(im,(x+793,y+22),f'{v:.1f}%',51,colors[row],500,anchor='ra')
  if col==1:txt(im,(x,841),'−19.1 percentage points',30,LIME,500)
 txt(im,(95,914),'Timing prompts also differ between modes; comparisons include both timing and instruction changes.',20,'#a4b497',400,'dm-sans')
 footer(im,'Source: manuscript · aggregate static versus real-time results',True)
 return im

def collisions(s,t):
 im=base(s,t);headline(im,['Reasoning time becomes exposure.'],size=70)
 txt(im,(98,328),'12.3×',152,GREEN,400);txt(im,(103,510),'more collisions in real time',35,INK)
 wrap(im,(104,584),'3.31 → 40.68 collisions per episode in the hard-setting comparison.',29,MUTED,665,1.6)
 box(im,(105,755,735,896),'#eaf0df',9);txt(im,(132,777),'84%',68,GREEN);wrap(im,(337,792),'of real-time contacts occur during inference',24,GREEN,350,1.5)
 # Large stacked columns use a common scale, 45 collisions / episode.
 for i,(label,active,passive) in enumerate([('Static',3.31,0),('Real-time',6.38,34.31)]):
  x=993+i*390;y=845;height=active+passive;scale=10.7;anim=fade(t,1+i,3+i)
  for grid in range(5):
   gy=y-grid*10*scale;line(im,[(903,gy),(1810,gy)],LINE)
   if i==0:txt(im,(878,gy),str(grid*10),16,MUTED,400,'dm-sans',anchor='rm')
  h1=int(active*scale*anim);h2=int(passive*scale*anim);box(im,(x,y-h1,x+196,y),GREEN,0)
  if h2:box(im,(x,y-h1-h2,x+196,y-h1),PALE,0)
  txt(im,(x+98,y-int(height*scale*anim)-59),f'{3.31 if i==0 else 40.68:.2f}',43,GREEN,500,anchor='ma');txt(im,(x+98,873),label,27,INK,500,anchor='ma')
 txt(im,(933,319),'COLLISIONS / EPISODE',18,MUTED,500,'dm-sans')
 box(im,(1006,950,1019,963),GREEN,2);txt(im,(1030,944),'While acting',18,MUTED,400,'dm-sans');box(im,(1330,950,1343,963),PALE,2);txt(im,(1354,944),'While reasoning',18,MUTED,400,'dm-sans')
 footer(im,'Sustained contact can produce repeated event counts · components rounded independently')
 return im

def models(s,t):
 im=base(s,t);headline(im,['A paused-world ranking doesn’t hold.'],size=67)
 txt(im,(95,285),'COLLISIONS PER EPISODE  /  HARD SETTING  /  LOWER IS BETTER',18,MUTED,500,'dm-sans')
 for panel,scope in enumerate(['static','realtime']):
  x=90+panel*900;box(im,(x,345,x+840,956),'#fff',13,LINE);txt(im,(x+29,370),'Static' if panel==0 else 'Real-time',33,INK)
  ranked=sorted(MODELS,key=lambda m:m[scope]['collisions']);mx=80 # shared scale to avoid visually equalizing counts
  for i,m in enumerate(ranked):
   y=440+i*58;v=m[scope]['collisions'];p=fade(t,.8+i*.12,2+i*.12)
   if i==0:box(im,(x+16,y-7,x+820,y+42),'#edf3e3',5)
   txt(im,(x+30,y),m['name'],25,GREEN if i==0 else INK)
   box(im,(x+212,y+8,x+670,y+25),'#f0f3e9',3);box(im,(x+212,y+8,x+212+max(2,int(458*v/mx*p)),y+25),GREEN if panel==0 else '#b2c897',3)
   txt(im,(x+790,y),f'{v:.2f}',26,GREEN,500,anchor='ra')
 footer(im,'Eight models · 36 routes per mode and model · same 0–80 bar scale in both panels')
 return im

def effort(s,t):
 im=base(s,t);headline(im,['More reasoning.','More time exposed.'],size=76)
 wrap(im,(96,391),'Increasing effort above provider defaults does not consistently improve safety.',30,MUTED,690)
 txt(im,(101,554),'40.7 → 62.0',78,GREEN,400);txt(im,(105,653),'mean collisions / episode',26,MUTED,400,'dm-sans')
 box(im,(99,747,753,900),'#e9efdf',9);txt(im,(123,771),'Active contacts',23,GREEN,500,'dm-sans');txt(im,(718,771),'6.4 → 5.9',29,GREEN,500,anchor='ra');txt(im,(123,832),'Passive contacts',23,GREEN,500,'dm-sans');txt(im,(718,832),'34.3 → 56.1',29,GREEN,500,anchor='ra')
 txt(im,(981,285),'DEFAULT → HIGHER EFFORT',18,MUTED,500,'dm-sans')
 for i,(label,active,passive,tot) in enumerate([('Default',6.4,34.3,40.7),('Higher',5.9,56.1,62.0)]):
  x=1000+i*415;y=855;p=fade(t,1,2.5);scale=7.5;h1=active*scale*p;h2=passive*scale*p
  box(im,(x,y-h1,x+212,y),GREEN);box(im,(x,y-h1-h2,x+212,y-h1),PALE)
  txt(im,(x+106,y-(h1+h2)-69),str(tot),54,GREEN,500,anchor='ma');txt(im,(x+106,890),label,28,INK,500,anchor='ma')
 txt(im,(985,949),'Hard real-time · eight models · 288 episodes per condition',18,MUTED,400,'dm-sans')
 footer(im,'Reasoning labels are provider-specific · these are aggregate reported results')
 return im

def learning(s,t):
 im=base(s,t);headline(im,['A testbed for learning safer policies.'],size=68)
 txt(im,(96,288),'Separate offline-training study · Qwen3-VL-4B · 16 tasks on held-out maps',26,MUTED,400,'dm-sans')
 headers=[('METHOD',126),('SUCCESS',1054),('COLLISIONS / 100 M',1710)]
 box(im,(90,374,1830,825),'#fff',12,LINE)
 for label,x in headers:txt(im,(x,404),label,18,MUTED,500,'dm-sans',anchor='ra' if x>900 else None)
 for i,r in enumerate(DATA['rl']):
  y=481+i*79
  if i==0:box(im,(106,y-12,1814,y+50),'#edf3e3',5)
  txt(im,(128,y),r['method'],29,INK);txt(im,(1054,y),f'{r["success"]:.1f}%',32,GREEN,500,anchor='ra');txt(im,(1710,y),f'{r["per100m"]:.1f}',32,GREEN,500,anchor='ra')
  if i<3:line(im,[(127,y+60),(1792,y+60)],'#e7ebdf')
 wrap(im,(96,868),'Behavior cloning has the lowest collision rate. The wander-penalty policy has the highest completion. Reward design changes the tradeoff.',26,MUTED,1740,1.5)
 footer(im,'Fixed 3 s decision delay · separate protocol from the eight-model comparison')
 return im

def closing(s,t):
 im=base(s,t,dark=True)
 txt(im,(960,244),'RT–SAFE',24,LIME,500,'dm-sans',anchor='ma')
 txt(im,(960,337),'Evaluate the whole decision.',86,'#f6f5f1',500,anchor='ma')
 txt(im,(960,447),'Including the time it takes.',86,LIME,500,anchor='ma')
 txt(im,(960,599),'Benchmarking agent safety in real-time embodied environments.',29,'#b8c8a9',400,'dm-sans',anchor='ma')
 line(im,[(420,706),(1500,706)],'#486147')
 for x,n,label in [(518,'5','CITY MAPS'),(835,'36','ROUTES'),(1187,'8','VLMS')]:
  txt(im,(x,758),n,57,'#f6f5f1',400);txt(im,(x+84,784),label,18,'#a8bd9b',500,'dm-sans')
 txt(im,(960,922),'EXPLORE THE WEBSITE  /  INSPECT THE RESULTS  /  READ THE PAPER',18,'#a8bd9b',500,'dm-sans',anchor='ma')
 footer(im,'RT–SAFE · research project presentation',True)
 return im
FUNCS={x.__name__:x for x in [opening,timing,environment,interface,decision,safety,completion,collisions,models,effort,learning,closing]}
def render(s,t,transitions=True):
 im=FUNCS[s['id']](s,t)
 if transitions:
  amount=min(1,t/.28,(s['duration']-t)/.25)
  if amount<1:im=Image.blend(Image.new('RGB',(W,H),DARK),im,max(0,amount))
 return im

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stills',action='store_true');parser.add_argument('--scene');args=parser.parse_args()
 scenes=[s for s in TL['scenes'] if not args.scene or s['id']==args.scene]
 if args.stills:
  contact=Image.new('RGB',(1280,3*720),PAPER)
  for i,s in enumerate(scenes):
   im=render(s,s['duration']*.68,False);im.save(OUT/'stills'/f'{s["id"]}.jpg',quality=94);preview=im.resize((640,360));contact.paste(preview,((i%2)*640,(i//2)*360))
  contact.save(OUT/'contact-sheet.jpg',quality=90)
  opening(TL['scenes'][0],5).save(ROOT/'public/media/film-poster.jpg',quality=92)
  print('Rendered',len(scenes),'scene previews',flush=True);return
 for s in scenes:
  path=OUT/'scenes'/f'{s["id"]}.mp4'
  frames=round(s['duration']*FPS)
  cmd=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-threads','8','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
  p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
  for i in range(frames):
   im=render(s,i/FPS);p.stdin.write(im.tobytes())
  p.stdin.close();rc=p.wait()
  if rc:raise RuntimeError(f'ffmpeg failed {s["id"]}')
  print('Rendered',s['id'],frames,'frames',flush=True)
if __name__=='__main__':main()
