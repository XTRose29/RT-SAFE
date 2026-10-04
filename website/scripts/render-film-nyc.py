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
    if name not in clips:clips[name]=Clip(MEDIA/(name+'.mp4'))
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
def timing(s,t):
    im=v.base(s,t,'The world does not pause while an agent thinks.')
    for i,name in enumerate(('timing-static','timing-realtime')):
        x=65+i*920;col=v.BLUE if i==0 else v.ORANGE
        v.rect(im,(x,270,x+870,928),v.WHITE,14,v.LINE)
        v.text(im,(x+22,293),'Static evaluation' if i==0 else 'Real-time evaluation',36,col,600)
        im.paste(footage(name,t,(870,489)),(x,362))
        v.text(im,(x+23,874),'INFERENCE' if t<7 else 'ACTION' if t<10 else 'OUTCOME',20,col,600)
        clock=max(0,t-7) if i==0 else t
        v.text(im,(x+847,873),f'World time +{clock:.1f} s',25,col,500,'ra')
    v.footer(im,'Native NYC scene · same initial state and commanded move · illustrative encounter')
    return im
def environment(s,t):
    return footage('environment-tour',t)
def examples(s,t):
    im=v.base(s,t,'Same route. Different decisions.', 'Full-segment outcomes · original recorded run' if t>=10.8 else 'GPT-6 Astra and GPT-5.6 Sol · recorded RT15 Task 19 · low reasoning effort')
    for i,name in enumerate(('astra','sol')):
        x=65+i*920;col=v.TEAL if i==0 else v.BLUE
        v.rect(im,(x,274,x+870,927),v.WHITE,12,v.LINE)
        v.text(im,(x+22,291),'GPT-6 Astra' if i==0 else 'GPT-5.6 Sol',36,col,600)
        im.paste(footage(name,t if t<10.8 else 44.05,(870,489)),(x,355))
        if t<10.8:
            v.text(im,(x+22,874),'6× recorded simulation time',24,v.MUTED,400)
        else:
            v.text(im,(x+22,872),'1 collision · 12 decisions' if i==0 else '17 collisions · 33 decisions',31,col,600)
            v.text(im,(x+843,917),'FULL SEGMENT TOTAL',16,v.MUTED,400,'ra')
    v.footer(im,'NYC reconstruction · original agent poses and timing · counts from the full recorded segment')
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
v.FUNCS.update({f.__name__:f for f in (title,timing,environment,examples,learning)})
if __name__=='__main__':v.main()
