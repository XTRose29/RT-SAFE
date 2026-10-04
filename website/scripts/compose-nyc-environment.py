"""Animate route guidance and hazard labels over the native NYC environment shot."""
from pathlib import Path
import argparse,importlib.util,json,math
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('media',ROOT/'scripts/build-nyc-media.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);v=m.v
def project(point,loc,rotation,fov):
    pitch,yaw=map(math.radians,rotation[:2])
    forward=np.array([math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch)])
    right=np.array([-math.sin(yaw),math.cos(yaw),0]);up=np.cross(forward,right)
    delta=np.array(point)-np.array(loc);depth=float(delta@forward)
    if depth<1:return None
    focal=960/math.tan(math.radians(fov/2))
    return (960+float(delta@right)/depth*focal,540-float(delta@up)/depth*focal)
LABELS={'goal':'Goal','pedestrians':'Moving pedestrians','dog':'Robot dog','scooter':'Movable objects','box':'Movable objects',
    'oil':'Oil','water':'Water','trip':'Trip hazard','crossing':'Marked crossing','signal':'Traffic signal'}
OFFSETS={'goal':(-80,-90),'pedestrians':(90,-120),'dog':(110,40),'scooter':(-240,45),
    'box':(95,-85),'oil':(-240,-60),'water':(85,30),'trip':(100,-90),'crossing':(-200,65),'signal':(75,-100)}
GROUPS=[('Reach the goal safely.','A planned route through a world that keeps moving.',['goal'],v.TEAL),
    ('Avoid collisions.','People, robots, vehicles, and physical objects.',['pedestrians','dog','box'],v.BLUE),
    ('Watch for hazards.','Trip, oil, water, and other environmental hazards.',['oil','water','trip'],v.ORANGE),
    ('Follow traffic rules.','Use the marked crossing and obey the signal.',['crossing','signal'],v.RED),
    ('Measure safe completion.','Reach the destination with zero recorded safety events.',['goal'],v.TEAL)]
def overlay(im,meta,index,t):
    _,loc,rotation=meta['camera_poses'][index]
    def screen(p):return project(p,loc,rotation,meta['horizontal_fov'])
    points=[screen(p) for p in meta['route']]
    # Flowing dashes indicate a planned route, not a measured agent trajectory.
    for pa,pb in zip(points,points[1:]):
        if pa is None or pb is None:continue
        length=math.dist(pa,pb)
        for d in np.arange(-(t*34)%32-32,length,32):
            a=max(0,d);b=min(length,d+18)
            if b<a:continue
            v.line(im,[(pa[0]+(pb[0]-pa[0])*a/length,pa[1]+(pb[1]-pa[1])*a/length),
                       (pa[0]+(pb[0]-pa[0])*b/length,pa[1]+(pb[1]-pa[1])*b/length)],'#39c6c1',6)
    v.rect(im,(48,31,928,158),v.INK,10)
    v.text(im,(75,48),'Navigate to the goal safely.',43,v.WHITE,600)
    v.text(im,(78,112),'RT-Safe  /  NYC environment demonstration',22,v.PALE,400)
    title,desc,anchors,color=GROUPS[min(4,int(t/2.4))]
    for key in anchors:
        point=meta['dynamic_anchors'][key][index] if key in meta.get('dynamic_anchors',{}) else meta['anchors'][key]
        xy=screen(point)
        if xy is None:continue
        x,y=xy
        if not (12<x<1908 and 180<y<870):continue
        v.circle(im,x,y,21+5*math.sin(t*4),color,4)
        dx,dy=OFFSETS[key];label=LABELS[key];width=v.font(26,600).getlength(label)+36
        lx=min(1880-width,max(40,x+dx));ly=min(784,max(185,y+dy))
        v.line(im,[(x,y),(lx+width/2,ly+25)],color,3)
        v.rect(im,(lx,ly,lx+width,ly+51),v.WHITE,5,color,2)
        v.text(im,(lx+18,ly+9),label,26,color,600)
    v.rect(im,(48,848,1008,979),v.INK,12)
    v.text(im,(77,864),title,38,v.WHITE,600)
    v.text(im,(79,923),desc,25,v.PALE,400)
    v.text(im,(1855,962),'Illustrative planned route · native Unreal rendering',17,v.INK,500,'ra')
    v.rect(im,(0,1000,1920,1080),v.INK)
    return im
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args()
    stem='environment-samples' if args.preview else 'environment';folder=m.NYC/'renders'/stem
    meta=json.loads((folder/'scene.json').read_text());frames=sorted(folder.glob(stem+'_*.png'))
    assert len(frames)==len(meta['times']),(len(frames),len(meta['times']))
    if args.preview:
        sheet=Image.new('RGB',(1440,810),v.BG)
        for i,p in enumerate(frames):
            im=overlay(Image.open(p).convert('RGB'),meta,i,meta['times'][i]);im.save(folder/f'annotated-{i}.jpg',quality=95)
            sheet.paste(im.resize((480,270)),((i%3)*480,(i//3)*270))
        sheet.save(folder/'contact-sheet.jpg',quality=94);return
    proc=m.writer(m.MEDIA/'environment-tour.mp4',crf=19)
    for i,p in enumerate(frames):
        im=overlay(Image.open(p).convert('RGB'),meta,i,meta['times'][i]);proc.stdin.write(im.tobytes())
        if i==0:im.save(m.MEDIA/'environment.webp',quality=93)
    m.finish(proc)
    def ts(t):return f'00:00:{t:06.3f}'
    captions=['WEBVTT','']
    for i,(title,desc,_,_) in enumerate(GROUPS):captions.extend([f'{ts(i*2.4)} --> {ts((i+1)*2.4)}',title+' '+desc,''])
    (m.MEDIA/'environment-tour.vtt').write_text('\n'.join(captions))
    print('Exported 12-second annotated environment tour',flush=True)
if __name__=='__main__':main()
