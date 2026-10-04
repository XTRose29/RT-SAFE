"""Portrait framing and in-scene labels for the native timing comparison."""
import math
from PIL import Image,ImageDraw

WIDTH,HEIGHT=960,1000
CROP_LEFT=442
CROP_WIDTH=1036

def project(point,meta):
    pitch,yaw=map(math.radians,meta['camera_rotation'][:2])
    forward=(math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch))
    right=(-math.sin(yaw),math.cos(yaw),0)
    up=(-math.sin(pitch)*math.cos(yaw),-math.sin(pitch)*math.sin(yaw),math.cos(pitch))
    delta=[p-c for p,c in zip(point,meta['camera_location'])]
    depth=sum(a*b for a,b in zip(delta,forward));focal=960/math.tan(math.radians(meta['horizontal_fov']/2))
    x=960+sum(a*b for a,b in zip(delta,right))/depth*focal
    y=540-sum(a*b for a,b in zip(delta,up))/depth*focal
    return ((x-CROP_LEFT)*WIDTH/CROP_WIDTH,y*HEIGHT/1080)

def compose(im,meta,index,t,mode,v):
    im=im.crop((CROP_LEFT,0,CROP_LEFT+CROP_WIDTH,1080)).resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS)
    d=ImageDraw.Draw(im,'RGBA')
    for y in range(270):d.line((0,y,WIDTH,y),fill=(8,23,38,int(190*(1-y/270))))
    for y in range(850,HEIGHT):d.line((0,y,WIDTH,y),fill=(8,23,38,int(170*(y-850)/150)))
    color='#a9e8ed' if mode=='static' else '#ffca7a'
    v.text(im,(30,24),'CONVENTIONAL BENCHMARKS' if mode=='static' else 'RT-SAFE  /  REAL-WORLD TIMING',23,color,600)
    v.text(im,(28,72),'Static evaluation' if mode=='static' else 'Real-time evaluation',49,v.WHITE,600)
    status='WORLD PAUSED' if mode=='static' and t<7 else 'WORLD KEEPS MOVING'
    v.text(im,(31,145),status,35,color,600)
    v.text(im,(31,196),'During agent inference' if t<7 else 'During agent action',27,v.WHITE,500)
    pose=meta['robot_poses'][index][1]
    if t<7:
        # Project the approaching pedestrian's real positions to explain world motion.
        hp=meta['human_poses'][index][1]
        foot=project(hp,meta)
        d=ImageDraw.Draw(im,'RGBA')
        d.ellipse((foot[0]-35,foot[1]-12,foot[0]+35,foot[1]+12),outline=color,width=3)
        if mode=='realtime':
            origin=project(meta['human_poses'][0][1],meta)
            if math.dist(origin,foot)>24:v.arrow(im,origin,foot,color,5)
        else:
            v.rect(im,(foot[0]+48,foot[1]-68,foot[0]+59,foot[1]-39),color,2)
            v.rect(im,(foot[0]+65,foot[1]-68,foot[0]+76,foot[1]-39),color,2)
    head=project((pose[0],pose[1],pose[2]+190),meta)
    label='Thinking…' if t<7 else 'Acting' if t<10 else 'Action complete'
    w=int(v.font(32,600).getlength(label)+44);x=max(18,min(WIDTH-w-18,head[0]-w/2));y=max(165,head[1]-83)
    v.line(im,[(head[0],y+59),(head[0],head[1]-5)],color,3)
    v.rect(im,(x,y,x+w,y+59),v.INK,12,color,2)
    v.text(im,(x+22,y+9),label,32,v.WHITE,600)
    if mode=='realtime' and 8.65<t<9.75:
        center=project((pose[0],pose[1],pose[2]+95),meta)
        v.circle(im,center[0]+18,center[1],78,color,4)
    clock=max(0,t-7) if mode=='static' else t
    if t>=8.65:
        outcome='Path remains clear' if mode=='static' else 'Collision risk'
        v.text(im,(31,869),outcome,33,color,600)
    else:
        v.text(im,(31,869),'World frozen while the agent thinks.' if mode=='static' else 'Pedestrians move while the agent thinks.',30,v.WHITE,500)
    v.text(im,(31,925),f'World time +{clock:.1f} s',25,v.WHITE,500)
    v.text(im,(WIDTH-28,971),'Illustrative NYC encounter',15,'#d4e2e5',400,'ra')
    return im
