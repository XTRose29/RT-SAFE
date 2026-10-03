"""Render the 90-second revision: animated paper diagrams and measured results.
Existing figure crops are used as video panels; overlays are explanatory motion.
Recorded observations are never interpolated into fabricated agent outcomes.
"""
from pathlib import Path
import json, math, functools, argparse, subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'video/revision-90s';W,H,FPS=1920,1080,30
DATA=json.loads((ROOT/'app/data.json').read_text());MODELS=DATA['models'];EX=json.loads((ROOT/'app/examples.json').read_text())
TL=json.loads((OUT/'timeline.json').read_text())
BG='#f6f8fb';INK='#152d50';MUTED='#61718a';LINE='#dbe3ed';TEAL='#087f85';BLUE='#2273b8';ORANGE='#ed8a15';RED='#d63853';PALE='#9dd8dd';WHITE='#ffffff'
@functools.lru_cache(150)
def font(n,w=500):return ImageFont.truetype(str(ROOT/f'public/fonts/space-grotesk-{w}.ttf'),n)
def text(im,xy,s,size=32,color=INK,weight=500,anchor=None):ImageDraw.Draw(im).text(xy,str(s),font=font(size,weight),fill=color,anchor=anchor)
def rect(im,b,fill,r=0,outline=None,width=1):ImageDraw.Draw(im).rounded_rectangle(tuple(map(int,b)),radius=r,fill=fill,outline=outline,width=width)
def line(im,pts,color=LINE,width=2):ImageDraw.Draw(im).line(pts,fill=color,width=width)
def ease(v):v=max(0,min(1,v));return v*v*(3-2*v)
def progress(t,a=0,b=1):return ease((t-a)/(b-a))
@functools.lru_cache(100)
def load(path):return Image.open(path).convert('RGB')
@functools.lru_cache(60)
def resized(path,w,h):return load(path).resize((w,h),Image.Resampling.LANCZOS)
def put(im,path,x,y,w,h):im.paste(resized(str(path),int(w),int(h)),(int(x),int(y)))
def footer(im,source):text(im,(1845,970),source,17,MUTED,400,'ra')
def base(s,t,title,sub=None):
 im=Image.new('RGB',(W,H),BG);text(im,(76,42),'RT-Safe',28,TEAL,700);text(im,(1844,51),s['chapter'],17,MUTED,500,'ra');line(im,[(75,101),(1845,101)])
 text(im,(75,133),title,61,INK,500)
 if sub:text(im,(78,218),sub,23,MUTED,400)
 rect(im,(0,H-5,W,H),LINE);rect(im,(0,H-5,W*(s['start']+t)/90,H),TEAL)
 return im
def circle(im,x,y,r,color,width=4):ImageDraw.Draw(im).ellipse((x-r,y-r,x+r,y+r),outline=color,width=width)
def arrow(im,a,b,color,width=4):
 line(im,[a,b],color,width);ang=math.atan2(b[1]-a[1],b[0]-a[0]);d=13
 ImageDraw.Draw(im).polygon([b,(b[0]-d*math.cos(ang-.5),b[1]-d*math.sin(ang-.5)),(b[0]-d*math.cos(ang+.5),b[1]-d*math.sin(ang+.5))],fill=color)
def title(s,t):
 im=Image.new('RGB',(W,H),INK)
 for j in range(9):line(im,[(1200+j*100,0),(700+j*100,1080)],'#213c60',2)
 rect(im,(96,180,183,188),PALE,4)
 text(im,(91,247),'RT-Safe:',120,WHITE,600)
 text(im,(98,438),'Benchmarking Agent Safety',72,WHITE)
 text(im,(98,532),'in Real-Time Embodied Environment',72,WHITE)
 text(im,(101,723),'The world does not pause while an agent thinks.',33,PALE,400)
 text(im,(102,937),'RESEARCH DEMO  /  90 SECONDS',19,'#b6cadf',400)
 return im
@functools.lru_cache(8)
def timing_panel(which):
 src=load(str(OUT/'assets/figure1.png'))
 coords=[(5,181,1056,536),(1097,181,2160,536),(5,1081,1056,1458),(1097,1081,2160,1458)]
 return src.crop(coords[which]).resize((850,306),Image.Resampling.LANCZOS)
def timing(s,t):
 im=base(s,t,'The world does not pause while an agent thinks.')
 clock=max(0,min(8,(t-.6)*.88));acting=clock>=6
 for i in range(2):
  x=75+i*920;col=BLUE if i==0 else ORANGE
  rect(im,(x,272,x+850,849),WHITE,16,LINE)
  text(im,(x+26,292),'Static evaluation' if i==0 else 'Real-time evaluation',37,col,600)
  text(im,(x+27,350),'World paused during inference' if i==0 else 'World evolves during inference',24,MUTED,400)
  start=timing_panel(0)
  if i==0:img=Image.blend(start,timing_panel(1),progress(clock,6,7.2))
  else:
   img=Image.blend(start,timing_panel(2),progress(clock,.4,4.8))
   img=Image.blend(img,timing_panel(3),progress(clock,6,7.3))
  im.paste(img,(x,409))
  # Motion is illustrative: reveal the pedestrian's change of position.
  if i and .6<clock<6:
   px=x+670-125*progress(clock,.6,5.5);circle(im,px,580,38+3*math.sin(t*5),ORANGE,4)
   arrow(im,(x+729,662),(px,662),ORANGE,5)
  phase='ACTION' if acting else 'INFERENCE'
  text(im,(x+29,748),phase,22,col,600)
  sim=clock if i else max(0,clock-6)
  text(im,(x+822,746),f'World time +{sim:.1f}s',27,col,500,'ra')
  rect(im,(x+28,809,x+820,816),LINE,3)
  rect(im,(x+28,809,x+28+792*clock/8,816),col,3)
 text(im,(79,886),'Observe',24,MUTED);text(im,(839,886),'Reason',24,TEAL);text(im,(1740,886),'Act',24,MUTED)
 arrow(im,(215,905),(787,905),LINE,3);arrow(im,(964,905),(1690,905),LINE,3)
 footer(im,'Animated paper illustration · transition timing is schematic')
 return im
# Coordinates follow the supplied 2880 × 1620 environment illustration.
ENV_POINTS={
 'Moving actors':[(460,664),(638,626),(953,617)],
 'Objects & vehicles':[(2570,732),(1790,1031),(2025,1091),(2390,1299),(2160,1267)],
 'Trip, oil & water':[(1147,1067),(1235,1310),(1598,1380)],
 'Traffic rules':[(1405,234),(1505,720)],
 'Goal':[(2342,264)]}
def environment(s,t):
 im=resized(str(OUT/'assets/figure2.png'),W,H).copy();scale=W/2880
 categories=list(ENV_POINTS);idx=min(4,int(t/2.4));name=categories[idx];color=[BLUE,BLUE,ORANGE,RED,TEAL][idx]
 # Animated focus and route make the paper's annotated environment legible in time.
 overlay=Image.new('RGBA',(W,H),(16,35,58,65));d=ImageDraw.Draw(overlay)
 for x,y in ENV_POINTS[name]:
  x*=scale;y*=scale;rad=100+10*math.sin(t*3);d.ellipse((x-rad,y-rad,x+rad,y+rad),fill=(0,0,0,0))
 im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
 route=[(755,1195),(825,1034),(1098,792),(1218,525),(1775,503),(2270,437),(2345,290)]
 pts=[(x*scale,y*scale) for x,y in route];p=min(1,max(0,(t-.4)/10.6))*(len(pts)-1);seg=min(len(pts)-2,int(p));f=p-seg
 end=(pts[seg][0]+(pts[seg+1][0]-pts[seg][0])*f,pts[seg][1]+(pts[seg+1][1]-pts[seg][1])*f)
 line(im,pts[:seg+1]+[end],TEAL,7);ImageDraw.Draw(im).ellipse((end[0]-10,end[1]-10,end[0]+10,end[1]+10),fill=WHITE,outline=TEAL,width=4)
 for x,y in ENV_POINTS[name]:circle(im,x*scale,y*scale,33+6*math.sin(t*4),color,5)
 rect(im,(50,23,754,154),WHITE,10)
 text(im,(75,40),'Navigate to the goal safely.',39,INK,600)
 text(im,(77,105),'5 city maps  /  36 routes  /  3 difficulty levels',23,TEAL,500)
 rect(im,(72,848,814,967),INK,12);text(im,(96,867),name,36,WHITE,600)
 desc={'Moving actors':'Pedestrians and robot dogs','Objects & vehicles':'Dynamic and static physical contacts','Trip, oil & water':'Environmental hazard interactions','Traffic rules':'Cross on WALK, within the marked crossing','Goal':'Reach the destination with zero safety events'}[name]
 text(im,(98,920),desc,22,PALE,400)
 text(im,(1850,978),'Animated paper illustration · explanatory route',17,WHITE,400,'ra')
 return im
@functools.lru_cache(8)
def example_frames(n):
 folder=OUT/'assets/examples'/str(n);return sorted(folder.glob('*.png'))
def examples(s,t):
 # Match the visual switches to the spoken model names, using TTS word timing.
 words=json.loads((OUT/'audio/examples.words.json').read_text())
 cuts=[0]+[s['voiceOffset']+next(w['start'] for w in words if w['text']==name)/s['voiceSpeed']-.12 for name in ['Sonnet','Fable']]
 idx=sum(t>=cut for cut in cuts)-1;local=t-cuts[idx];ex=EX[idx];full={'Gemini':'Gemini 3.8 Flash','Sonnet':'Claude Sonnet 5','Fable':'Claude Fable 5.1'}[ex['model']]
 im=base(s,t,'What does a real agent do?','Recorded UE observations · hard real-time · provider-default effort')
 frames=example_frames(ex['id']);img=ROOT/'public'/ex['image']
 if local>=1.6:
  k=min(len(frames)-1,int((local-1.6)/.5));img=frames[k] if frames else ROOT/'public'/ex['resultImage']
 put(im,img,85,280,760,676)
 rect(im,(104,300,579,351),INK,6)
 text(im,(123,311),'OBSERVATION' if local<1.6 else 'RECORDED ACTION RESULT',22,WHITE,600)
 rect(im,(885,280,1835,941),WHITE,14,LINE)
 text(im,(928,316),f'EXAMPLE {idx+1} / 3',19,TEAL,600)
 text(im,(927,375),full,48,INK,600)
 text(im,(930,447),f'{ex["map"]} · Task {ex["task"]} · Decision {ex["decision"]}',25,MUTED,400)
 action='Wait 1 second' if ex['id']==7 else 'Move forward 2 m'
 text(im,(928,526),action,59,TEAL,500)
 reasons=['Follow the sidewalk toward the subgoal.','Take a short step and reassess.','A pedestrian blocks the forward targets.']
 text(im,(931,610),reasons[idx],26,MUTED,400)
 for j,(label,val) in enumerate([('Inference exposure',f'{ex["inferenceExposure"]:.2f} s'),('Action duration','1.00 s')]):
  x=930+j*440;text(im,(x,704),label,22,MUTED,400);text(im,(x,746),val,48,INK)
 text(im,(931,854),'No safety event in this decision',24,TEAL,500)
 footer(im,'Condensed decision playback · inference gaps omitted · timing shown separately')
 return im
# RQ1 is the all-difficulty result scope (108 episodes per model).
def behavior(s,t):
 im=base(s,t,'Different agents. Different safety behavior.','All three difficulty levels · 108 episodes per model · provider-default effort')
 text(im,(92,284),'FEWEST COLLISIONS PER EPISODE',21,TEAL,600)
 ranked=sorted(MODELS,key=lambda m:m['average']['collisions'])
 for i,m in enumerate(ranked):
  y=348+i*68;v=m['average']['collisions'];p=progress(t,.3+i*.10,1.7+i*.10)
  if i==0:rect(im,(78,y-9,892,y+49),'#deeff0',6)
  text(im,(98,y),f'{i+1:02}',22,MUTED);text(im,(159,y-3),m['name'],29,INK)
  rect(im,(347,y+8,347+405*v/60*p,y+26),TEAL if i<4 else BLUE,4)
  text(im,(848,y-3),f'{v:.1f}',31,INK,500,'ra')
 # Animated scatter; the exact paper fit is shown as descriptive evidence.
 x0,y0=1050,847;pw,ph=705,452
 text(im,(1017,284),'INFERENCE EXPOSURE & COLLISIONS',21,TEAL,600)
 for n in [0,20,40,60]:
  y=y0-ph*n/65;line(im,[(x0,y),(x0+pw,y)],LINE);text(im,(x0-18,y),n,19,MUTED,400,'rm')
 for n in [0,1000,2000]:
  x=x0+pw*n/2800;line(im,[(x,y0),(x,y0-ph)],LINE);text(im,(x,y0+17),n,18,MUTED,400,'ma')
 xp=np.array([m['average']['latency']*m['average']['decisions'] for m in MODELS]);yp=np.array([m['average']['collisions'] for m in MODELS]);a,b=np.polyfit(xp,yp,1)
 pe=progress(t,.6,3);xx=2600*pe;line(im,[(x0,y0-ph*b/65),(x0+pw*xx/2800,y0-ph*(a*xx+b)/65)],'#7092b2',4)
 offsets={'Sonnet':(-65,27),'Astra':(-56,-30),'Fable':(-10,21),'Sol':(12,-20),'Gemini':(8,-30),'Inkling':(10,8),'DeepSeek':(-115,-30),'Grok':(-47,-36)}
 for i,m in enumerate(MODELS):
  if t<.6+i*.2:continue
  x=x0+pw*xp[i]/2800;y=y0-ph*yp[i]/65;ImageDraw.Draw(im).ellipse((x-8,y-8,x+8,y+8),fill=TEAL if m['name']!='Inkling' else ORANGE)
  dx,dy=offsets[m['name']];text(im,(x+dx,y+dy),m['name'],20,INK)
 text(im,(1060,349),'Paper fit: R² = 0.934',25,TEAL)
 text(im,(x0+pw/2,911),'Exposure proxy: mean latency × mean decisions (s)',19,MUTED,400,'ma')
 footer(im,'Paper Figure 3 / Appendix B.3 · association does not establish causation')
 return im

def completion(s,t):
 im=base(s,t,'Task completion is not enough.','All-difficulty real-time results · 8 models × 108 episodes')
 vals=[('Success rate',94.4,BLUE),('Safe success rate',.7,TEAL)]
 for i,(label,val,col) in enumerate(vals):
  x=95+i*910;rect(im,(x,308,x+820,841),WHITE,18,LINE)
  text(im,(x+37,346),label,39,INK)
  text(im,(x+35,423),f'{val:.1f}%',149,col,500)
  text(im,(x+42,653),'Reach the goal' if i==0 else 'Reach the goal with zero safety events',28,MUTED,400)
  rect(im,(x+41,733,x+779,754),LINE,10);rect(im,(x+41,733,x+41+max(5,738*val/100*progress(t,.4,2)),754),col,10)
 text(im,(960,891),'Collisions  +  hazards  +  traffic violations',31,RED,500,'ma')
 footer(im,'Paper §3.2 · safe success requires zero recorded events across all categories')
 return im

def comparison(s,t):
 im=base(s,t,'The safety gap appears in real time.','Matched hard routes · 8 models × 36 routes per timing mode')
 text(im,(95,292),'COLLISIONS PER EPISODE',20,MUTED,600)
 for j,(label,v,a,p,col) in enumerate([('Static',3.31,3.31,0,BLUE),('Real-time',40.68,6.38,34.31,TEAL)]):
  y=382+j*197;anim=progress(t,.4,2.2);text(im,(98,y),label,37,INK)
  x=319;rect(im,(x,y+4,x+a/45*820*anim,y+59),col,4)
  if p:rect(im,(x+a/45*820*anim,y+4,x+(a+p)/45*820*anim,y+59),ORANGE,4)
  text(im,(1184,y-1),f'{v:.2f}',43,INK,500,'ra')
 rect(im,(1295,310,1838,824),INK,15);text(im,(1567,365),'12.3×',100,WHITE,500,'ma');text(im,(1567,493),'more collisions',31,PALE,400,'ma')
 text(im,(1567,593),'84%',91,WHITE,500,'ma');text(im,(1567,704),'occur during inference',27,PALE,400,'ma')
 rect(im,(323,749,345,771),TEAL,3);text(im,(363,743),'While acting',25,MUTED)
 rect(im,(643,749,665,771),ORANGE,3);text(im,(683,743),'While deciding',25,MUTED)
 text(im,(99,878),'Safe success: 19.8% static → 0.7% real-time',36,TEAL)
 footer(im,'Paper §3.3 / Table 11 · repeated contacts may yield multiple event counts')
 return im

def reasoning(s,t):
 im=base(s,t,'More reasoning is not reliably safer.','Hard real-time · 288 episodes per condition · mean collisions per episode')
 rows=[('Lower',32.7,8.8,23.9,10.1),('Default',40.7,6.4,34.3,19.2),('Higher',62.0,5.9,56.1,38.7)]
 for i,(name,v,a,p,lat) in enumerate(rows):
  x=206+i*579;y=827;anim=progress(t,.3,2);ha=a*6.4*anim;hp=p*6.4*anim
  rect(im,(x,y-ha,x+239,y),TEAL)
  rect(im,(x,y-ha-hp,x+239,y-ha),ORANGE)
  text(im,(x+119,y-ha-hp-66),f'{v:.1f}',57,INK,500,'ma')
  text(im,(x+119,848),name,32,INK,600,'ma')
  text(im,(x+119,899),f'{lat:.1f} s / decision',23,MUTED,400,'ma')
 rect(im,(1015,285,1035,305),TEAL,2);text(im,(1048,279),'Active',23,MUTED)
 rect(im,(1233,285,1253,305),ORANGE,2);text(im,(1267,279),'Passive',23,MUTED)
 text(im,(91,290),'40.7 → 62.0 at higher effort',34,TEAL)
 footer(im,'Paper Tables 15–16 · default = lower for Sol, middle setting for the other 7 models')
 return im

def learning(s,t):
 im=base(s,t,'An environment for learning safer policies.','Offline RL · Qwen3-VL-4B · 16 held-out tasks · fixed 3-second decision delay')
 # Animated data-to-policy-to-environment flow, followed by the measured tradeoff.
 nodes=[(90, 'Expert trajectories'),(697,'BC / Offline RL'),(1304,'Held-out evaluation')]
 for i,(x,label) in enumerate(nodes):
  rect(im,(x,283,x+521,353),WHITE,10,LINE);text(im,(x+260,302),label,27,TEAL,500,'ma')
  if i<2:
   arrow(im,(x+537,319),(x+591,319),TEAL,3);u=((t*.45)%1);circle(im,x+540+u*44,319,5,ORANGE,4)
 rect(im,(89,393,1835,833),WHITE,14,LINE)
 for x,l,anc in [(118,'METHOD',None),(1140,'SUCCESS ↑','ra'),(1770,'COLLISIONS / 100 m ↓','ra')]:text(im,(x,419),l,22,MUTED,600,anc)
 for i,row in enumerate(DATA['rl']):
  y=478+i*80
  if i==0:rect(im,(102,y-7,1820,y+57),'#e4f1f1',5)
  text(im,(120,y),row['method'],31,INK)
  text(im,(1140,y),f'{row["success"]:.1f}%',35,TEAL if i==2 else INK,600 if i==2 else 500,'ra')
  x=1320;rect(im,(x,y+16,x+max(2,row['per100m']/120*260*progress(t,.7,2.2)),y+30),TEAL if i==0 else BLUE,3)
  text(im,(1770,y),f'{row["per100m"]:.1f}',35,TEAL if i==0 else INK,600 if i==0 else 500,'ra')
 text(im,(960,883),'Reward design changes the safety–progress tradeoff.',33,TEAL,500,'ma')
 footer(im,'Paper §3.5 / Table 1 · RL wander: highest success; BC: lowest collision rate')
 return im
FUNCS={f.__name__:f for f in [title,timing,environment,examples,behavior,completion,comparison,reasoning,learning]}
def render(s,t,transition=True):
 im=FUNCS[s['id']](s,t)
 # Reserve a consistent dark subtitle band in every scene.
 rect(im,(0,1000,W,1075),INK)
 if transition:
  a=min(1,t/.18,(s['duration']-t)/.18)
  if a<1:im=Image.blend(Image.new('RGB',(W,H),INK),im,max(0,a))
 return im

def main():
 (OUT/'scenes').mkdir(parents=True,exist_ok=True);(OUT/'stills').mkdir(parents=True,exist_ok=True)
 ap=argparse.ArgumentParser();ap.add_argument('--stills',action='store_true');ap.add_argument('--scene');args=ap.parse_args()
 scenes=[s for s in TL['scenes'] if not args.scene or s['id']==args.scene]
 if args.stills:
  sheet=Image.new('RGB',(1440,math.ceil(len(scenes)/3)*270),BG)
  for i,s in enumerate(scenes):
   im=render(s,s['duration']*.64,False);im.save(OUT/'stills'/f'{s["id"]}.jpg',quality=94);sheet.paste(im.resize((480,270)),((i%3)*480,(i//3)*270))
  sheet.save(OUT/'contact-sheet.jpg',quality=93);return
 for s in scenes:
  p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-threads','6','-pix_fmt','yuv420p',str(OUT/'scenes'/f'{s["id"]}.mp4')],stdin=subprocess.PIPE)
  for i in range(s['duration']*FPS):p.stdin.write(render(s,i/FPS).tobytes())
  p.stdin.close();assert p.wait()==0
  print('Rendered',s['id'],s['duration'],'seconds',flush=True)
if __name__=='__main__':main()
