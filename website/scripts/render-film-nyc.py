"""90-second NYC edit: native scene footage and manuscript-backed result graphics."""
from pathlib import Path
import importlib.util, json, math
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('film90',ROOT/'scripts/render-film-90s.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
v.OUT=ROOT/'video/nyc-90s';v.TL=json.loads((v.OUT/'timeline.json').read_text())
MEDIA=ROOT/'public/media/nyc'
clips={}
LOGOS={'Astra':'openai','Sol':'openai','Fable':'claude','Sonnet':'claude','Gemini':'gemini','DeepSeek':'deepseek','Grok':'grok','Inkling':'thinking-machines'}
logo_cache={}
def logo(im,name,x,y,size=42):
    key=(name,size)
    if key not in logo_cache:
        logo_cache[key]=Image.open(ROOT/'public/media/logos'/f'{LOGOS[name]}.png').convert('RGBA').resize((size,size),Image.Resampling.LANCZOS)
    im.paste(logo_cache[key],(int(x),int(y)),logo_cache[key])
class Clip:
    def __init__(self,path):
        self.cap=cv2.VideoCapture(str(path))
        if not self.cap.isOpened():raise RuntimeError('Missing video '+str(path))
        self.fps=self.cap.get(cv2.CAP_PROP_FPS);self.count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT));self.index=-1;self.image=None
    def frame(self,t):
        target=min(self.count-1,max(0,int(t*self.fps)))
        if target==self.index:return self.image.copy()
        if target!=self.index+1:self.cap.set(cv2.CAP_PROP_POS_FRAMES,target)
        ok,frame=self.cap.read()
        if not ok:raise RuntimeError('Video decoding failed at frame '+str(target))
        self.index=target;self.image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));return self.image.copy()
def footage(name,t,size=(1920,1080)):
    if name not in clips:clips[name]=Clip((ROOT/'public/media'/name[1:] if name.startswith('/') else MEDIA/name).with_suffix('.mp4'))
    return clips[name].frame(t).resize(size,Image.Resampling.LANCZOS)
def shade(im,top=0,bottom=0):
    alpha=np.linspace(top,bottom,im.height).astype(np.uint8)
    layer=np.zeros((im.height,im.width,4),dtype=np.uint8);layer[:,:,:3]=(9,24,43);layer[:,:,3]=alpha[:,None]
    return Image.alpha_composite(im.convert('RGBA'),Image.fromarray(layer)).convert('RGB')
def title(s,t):
    im=shade(footage('hero',t),155,210)
    v.rect(im,(98,156,179,164),v.PALE,4)
    v.text(im,(91,226),'RT-Safe:',120,v.WHITE,600)
    v.text(im,(98,419),'Benchmarking Agent Safety',72,v.WHITE)
    v.text(im,(98,511),'in Real-Time Embodied Environment',72,v.WHITE)
    v.text(im,(100,711),'The world does not pause while an agent thinks.',34,v.PALE,400)
    v.text(im,(101,935),'MADISON SQUARE PARK, NYC  /  UNREAL ENGINE',19,'#c5d8e6',400)
    return im
timing_meta=None
timing_presentation=None
def opening_static(width=1920):
    global timing_meta,timing_presentation
    if timing_presentation is None:
        spec=importlib.util.spec_from_file_location('timing_presentation',ROOT/'scripts/timing-presentation.py')
        timing_presentation=importlib.util.module_from_spec(spec);spec.loader.exec_module(timing_presentation)
        timing_meta=json.loads((ROOT/'public/data/nyc/timing-scene.json').read_text())
    frame=v.load(str(MEDIA/'timing-static-wide.webp'))
    return timing_presentation.compose(frame,timing_meta,0,0,'static',v,width=width)

def timing(s,t):
    im=Image.new('RGB',(1920,1080),v.INK)
    if t<2.8:
        im.paste(opening_static(),(0,0))
    elif t<4.8:
        im.paste(shade(opening_static(),160,190),(0,0))
        v.text(im,(960,350),'But the real world',88,v.WHITE,600,'ma')
        v.text(im,(960,466),'does not stop.',108,'#ffca7a',600,'ma')
        v.text(im,(960,655),'People keep moving. Risk keeps changing.',32,v.WHITE,500,'ma')
    elif t<7.0:
        im=shade(footage('hero',t-4.8),190,210)
        v.text(im,(960,273),'WE INTRODUCE',36,v.PALE,600,'ma')
        v.text(im,(960,375),'RT-Safe',118,v.WHITE,600,'ma')
        v.text(im,(960,552),'An embodied environment that keeps evolving',43,v.WHITE,500,'ma')
        v.text(im,(960,624),'while the agent thinks.',49,'#ffca7a',600,'ma')
    elif t<7.7:
        p=v.ease((t-7.0)/.7);width=round(1920-960*p)
        im.paste(footage('timing-realtime',0,(960,1000)),(960,0))
        im.paste(opening_static(width),(0,0))
        v.line(im,[(width-1,0),(width-1,1000)],v.WHITE,3)
    else:
        source=min(10.99,(t-7.7)*11/7.3)
        for i,name in enumerate(('timing-static','timing-realtime')):
            im.paste(footage(name,source,(960,1000)),(i*960,0))
        v.line(im,[(959,0),(959,1000)],v.WHITE,3)
    return im
def environment(s,t):
    if t<2.4:
        im=shade(footage('hero',t+2),195,215)
        v.text(im,(960,253),'OUR TASK & ENVIRONMENT',32,v.PALE,600,'ma')
        v.text(im,(960,368),'Inside the RT-Safe environment',74,v.WHITE,600,'ma')
        v.text(im,(960,509),'Navigate to the goal safely.',38,v.WHITE,500,'ma')
        for x,label,col in [(415,'Collisions','#88c8ff'),(960,'Hazards','#ffca7a'),(1505,'Traffic-rule violations','#ff9bad')]:
            v.rect(im,(x-235,671,x+235,754),'#18334e',12,col,2)
            v.text(im,(x,695),label,29,col,600,'ma')
        return im
    return footage('environment-tour',t)
def examples(s,t):
    if t<2:
        im=shade(footage('hero',t+4),195,220)
        v.text(im,(960,310),'FRONTIER AGENTS  /  REAL-TIME SAFETY',25,v.PALE,600,'ma')
        v.text(im,(960,425),'How do frontier agents perform?',78,v.WHITE,600,'ma')
        v.text(im,(960,567),'Same route. Different decisions.',38,v.WHITE,500,'ma')
        v.text(im,(960,732),'GPT-6 Astra    /    GPT-5.6 Sol',29,v.PALE,500,'ma')
        return im
    t-=2
    # Two chronological excerpts, followed by the full-segment outcomes.
    source=t if t<5 else 34+(t-5) if t<10 else 44.05
    im=Image.new('RGB',(1920,1080),v.INK)
    for i,name in enumerate(('/recorded/astra','/recorded/sol')):im.paste(footage(name,source,(960,1000)),(i*960,0))
    v.line(im,[(959,0),(959,1000)],v.WHITE,3)
    if 5<=t<7 or t>=10:
        label='LATER IN THE SAME RUN' if t<10 else 'FULL SEGMENT OUTCOMES'
        v.rect(im,(701,107,1219,149),v.INK,8,v.PALE)
        v.text(im,(960,117),label,20,v.WHITE,600,'ma')
    return im
PROFILES={p['name']:p['values'] for p in json.loads((ROOT/'public/data/behavior.json').read_text())['profiles']}
def animated_radar(im,name,col,x,t):
    cx,cy=x+225,535;radius=128
    angles=[-math.pi/2+i*math.pi/3 for i in range(6)]
    labels=[['Fewer','collisions'],['Quicker','decisions'],['Fewer','decisions'],['Longer commanded','moves'],['More','waiting'],['More','turning']]
    for r in [.25,.5,.75,1]:v.circle(im,cx,cy,radius*r,'#dce4ec',1)
    for i,a in enumerate(angles):
        end=(cx+radius*math.cos(a),cy+radius*math.sin(a))
        v.line(im,[(cx,cy),end],'#dce4ec',1)
        lx=cx+181*math.cos(a);ly=cy+172*math.sin(a)
        for j,label in enumerate(labels[i]):v.text(im,(lx,ly-14+j*22),label,18,'#34495e',500,'ma')
    progress=v.ease(t/1.15)
    points=[(cx+radius*value*progress*math.cos(a),cy+radius*value*progress*math.sin(a)) for value,a in zip(PROFILES[name],angles)]
    rgb=tuple(int(col[i:i+2],16) for i in (1,3,5))
    d=ImageDraw.Draw(im,'RGBA');d.polygon(points,fill=rgb+(40,))
    v.line(im,points+[points[0]],col,3)
    for px,py in points:v.circle(im,px,py,3.5,col,2)
    if t>1.25:
        axis={'Inkling':3,'Grok':0,'Astra':1,'Fable':4,'Sonnet':0,'Sol':5,'Gemini':1,'DeepSeek':0}[name]
        px,py=points[axis];pulse=5+4*(.5+.5*math.sin((t-1.25)*4))
        v.circle(im,px,py,pulse,col,2)
    v.rect(im,(x+30,863,x+420,867),'#edf1f5',2)
    v.rect(im,(x+30,863,x+30+390*min(1,t/4.2),867),col,2)

def behavior(s,t):
    im=Image.new('RGB',(1920,1080),v.BG)
    v.text(im,(62,28),'REAL-TIME BENCHMARK  /  PROVIDER-DEFAULT REASONING',20,v.TEAL,600)
    if t<5.2:
        v.text(im,(60,76),'Eight models. Completion and safety.',60,v.INK,600)
        v.text(im,(64,158),'All three difficulties · 108 episodes per model · ranked by collisions per episode',25,v.MUTED,400)
        cols=[(620,'SUCCESS ↑'),(905,'SAFE SUCCESS ↑'),(1240,'COLLISIONS / EP. ↓'),(1545,'LATENCY (s) ↓'),(1830,'DECISIONS / EP.')]
        v.text(im,(77,224),'MODEL',21,v.MUTED,600)
        for x,label in cols:v.text(im,(x,224),label,21,v.MUTED,600,'ra')
        for i,m in enumerate(sorted(v.MODELS,key=lambda m:m['average']['collisions'])):
            progress=v.ease((t-.08*i)/1.05)
            if progress<=0:continue
            y=279+i*75+round(18*(1-progress));a=m['average'];col=v.TEAL if i==0 else v.INK
            v.rect(im,(60,y-7,1860,y+58),'#e0f0ee' if i==0 else v.WHITE,8)
            v.text(im,(80,y+6),f'{i+1:02}',23,v.MUTED,500);logo(im,m['name'],135,y+1,38);v.text(im,(190,y),m['name'],35,col,600)
            # A dedicated collision band makes the ranking metric visible at a glance.
            v.rect(im,(989,y-7,1270,y+58),'#e0f0ee' if i==0 else '#fff0e7',5)
            v.rect(im,(1005,y+39,1005+235*a['collisions']/60*progress,y+47),v.TEAL if i==0 else '#cf603a',3)
            vals=[f'{a["success"]*progress:.1f}%',f'{a["safeSuccess"]*progress:.1f}%',f'{a["collisions"]*progress:.1f}',f'{a["latency"]*progress:.1f}',f'{a["decisions"]*progress:.1f}']
            for j,((x,_),value) in enumerate(zip(cols,vals)):
                v.text(im,(x,y),value,35,('#087d88' if i==0 else '#ae4527') if j==2 else v.RED if j==1 and a['safeSuccess']==0 else col,600 if j==2 else 500,'ra')
        focus=0 if t<2.1 else 1 if t<3.65 else 2
        bounds=[(467,643),(734,935),(989,1270)][focus]
        if t>1.45:v.rect(im,(bounds[0],211,bounds[1],874),None,8,v.TEAL if focus!=2 else '#cf603a',3)
        takeaway=['High task completion across all eight models.','Every model: safe success below 4%.','Sonnet: fewest collisions. Grok: about 3× as many.'][focus]
        v.text(im,(65,909),takeaway,31,v.TEAL,500)
        v.footer(im,'Paper Table 5 / Appendix B.3 · means across easy, medium, and hard conditions')
    else:
        v.text(im,(60,76),'How do their behaviors differ?',60,v.INK,600)
        v.text(im,(64,158),'Same six axes. Different choices about speed, movement, turning, and waiting.',25,v.MUTED,400)
        cards=[('Inkling','#518b37',['Longer moves.','Fewer decisions.'],'26.0 s / decision'),('Grok','#c23b52',['Slow responses.','Most collisions.'],'59.5 collisions / ep.'),('Astra','#16878a',['Less turning and waiting','than Fable.'],'51.7 decisions / ep.'),('Fable','#9557b4',['More turning and waiting.','Slightly fewer collisions.'],'20.9 collisions / ep.')]
        if t>=9.6:
            cards=[('Sonnet','#cb6b9c',['Fastest responses.','Fewest collisions.'],'19.6 collisions / ep.'),('Sol','#357abb',['Most decisions.','Most turning and waiting.'],'69.1 decisions / ep.'),('Gemini','#b88b18',['Shorter commanded moves.','Frequent turning.'],'11.2 s / decision'),('DeepSeek','#ba6f3e',['Slow responses.','High collision count.'],'51.4 collisions / ep.')]
        v.text(im,(1852,34),'1–4 / 8' if t<9.6 else '5–8 / 8',22,v.MUTED,500,'ra')
        for i,(name,col,lines,stat) in enumerate(cards):
            x=50+i*470;v.rect(im,(x,225,x+450,878),v.WHITE,14,v.LINE)
            logo(im,name,x+225-v.font(38,600).getlength(name)/2-37,250,38)
            v.text(im,(x+245,249),name,38,col,600,'ma')
            local=t-(5.2 if t<9.6 else 9.6)
            animated_radar(im,name,col,x,max(0,local-i*.12))
            for j,line in enumerate(lines):v.text(im,(x+225,758+j*31),line,23,v.INK,500,'ma')
            v.text(im,(x+225,832),stat,22,col,600,'ma')
        v.text(im,(65,919),'Radar area is a behavior profile, not an overall safety score.',28,v.MUTED,400)
        v.footer(im,'Paper Figures 3 & 7 · each axis normalized across all eight models · commanded move length, not realized progress')
    return im
def learning(s,t):
    im=v.base(s,t,'An environment for learning safer behavior.','Offline RL · Qwen3-VL-4B · 16 held-out tasks · fixed 3-second decision delay')
    im.paste(footage('hero',t%8,(710,399)),(75,306))
    v.rect(im,(75,705,785,827),v.INK,0)
    v.text(im,(102,727),'Observe → act → receive reward',28,v.WHITE,500)
    v.text(im,(102,776),'NYC environment visualization',21,v.PALE,400)
    x=843
    for px,label,anchor in [(x+22,'METHOD',None),(1500,'SUCCESS ↑','ra'),(1815,'COLL. / 100 m ↓','ra')]:
        v.text(im,(px,294),label,19,v.MUTED,600,anchor)
    for i,row in enumerate(v.DATA['rl']):
        y=369+i*105
        v.rect(im,(x,y-9,1844,y+75),'#e4f1f1' if i==0 else v.WHITE,8,v.LINE)
        v.text(im,(x+21,y+12),row['method'],26,v.INK,500)
        v.text(im,(1500,y+8),f'{row["success"]:.1f}%',35,v.TEAL if i==2 else v.INK,600,'ra')
        v.text(im,(1815,y+8),f'{row["per100m"]:.1f}',35,v.TEAL if i==0 else v.INK,600,'ra')
    v.text(im,(960,887),'Learn to reach the goal. Learn to arrive safely.',37,v.TEAL,500,'ma')
    v.footer(im,'Paper §3.5 / Table 1 · source evaluation results; NYC footage is an environment visualization')
    return im
v.FUNCS.update({f.__name__:f for f in (title,timing,environment,examples,behavior,learning)})
if __name__=='__main__':v.main()
