"""Portrait framing and in-scene labels for the native timing comparison."""
import math
from PIL import Image,ImageDraw

WIDTH,HEIGHT=960,1000
CROP_LEFT=442
CROP_WIDTH=1036

def crop_box(width):
    cw=min(1920,1080*width/HEIGHT);ch=min(1080,1920*HEIGHT/width)
    return ((1920-cw)/2,(1080-ch)/2,(1920+cw)/2,(1080+ch)/2)

def project(point,meta,width=WIDTH):
    pitch,yaw=map(math.radians,meta['camera_rotation'][:2])
    forward=(math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch))
    right=(-math.sin(yaw),math.cos(yaw),0)
    up=(-math.sin(pitch)*math.cos(yaw),-math.sin(pitch)*math.sin(yaw),math.cos(pitch))
    delta=[p-c for p,c in zip(point,meta['camera_location'])]
    depth=sum(a*b for a,b in zip(delta,forward));focal=960/math.tan(math.radians(meta['horizontal_fov']/2))
    x=960+sum(a*b for a,b in zip(delta,right))/depth*focal
    y=540-sum(a*b for a,b in zip(delta,up))/depth*focal
    left,top,right,bottom=crop_box(width)
    return ((x-left)*width/(right-left),(y-top)*HEIGHT/(bottom-top))

def compose(im,meta,index,t,mode,v,width=WIDTH):
    panel_width=int(width)
    im=im.crop(crop_box(panel_width)).resize((panel_width,HEIGHT),Image.Resampling.LANCZOS)
    d=ImageDraw.Draw(im,'RGBA')
    for y in range(270):d.line((0,y,panel_width,y),fill=(8,23,38,int(190*(1-y/270))))
    for y in range(850,HEIGHT):d.line((0,y,panel_width,y),fill=(8,23,38,int(170*(y-850)/150)))
    color='#a9e8ed' if mode=='static' else '#ffca7a'
    v.text(im,(30,30),'CONVENTIONAL BENCHMARKS' if mode=='static' else 'INTRODUCING RT-SAFE',35,color,600)
    v.text(im,(28,88),'Static evaluation' if mode=='static' else 'Real-time evaluation',49,v.WHITE,600)
    status='WORLD PAUSED' if mode=='static' and t<7 else 'WORLD KEEPS MOVING'
    v.text(im,(31,160),status,35,color,600)
    v.text(im,(31,208),'During agent inference' if t<7 else 'During agent action',27,v.WHITE,500)
    pose=meta['robot_poses'][index][1]
    if t<7:
        # Project the approaching pedestrian's real positions to explain world motion.
        hp=meta['human_poses'][index][1]
        foot=project(hp,meta,panel_width)
        d=ImageDraw.Draw(im,'RGBA')
        d.ellipse((foot[0]-35,foot[1]-12,foot[0]+35,foot[1]+12),outline=color,width=3)
        if mode=='realtime':
            origin=project(meta['human_poses'][0][1],meta,panel_width)
            if math.dist(origin,foot)>24:v.arrow(im,origin,foot,color,5)
        else:
            v.rect(im,(foot[0]+48,foot[1]-68,foot[0]+59,foot[1]-39),color,2)
            v.rect(im,(foot[0]+65,foot[1]-68,foot[0]+76,foot[1]-39),color,2)
    head=project((pose[0],pose[1],pose[2]+190),meta,panel_width)
    contact=mode=='realtime' and t>=meta.get('contact_time',99)
    label='Contact · stopped' if contact else 'Thinking…' if t<7 else 'Acting' if t<10 else 'Action complete'
    w=int(v.font(32,600).getlength(label)+44);x=max(18,min(panel_width-w-18,head[0]-w/2));y=max(165,head[1]-83)
    v.line(im,[(head[0],y+59),(head[0],head[1]-5)],color,3)
    v.rect(im,(x,y,x+w,y+59),v.INK,12,color,2)
    v.text(im,(x+22,y+9),label,32,v.WHITE,600)
    if contact:
        center=project((pose[0],pose[1],pose[2]+95),meta,panel_width)
        v.circle(im,center[0]+18,center[1],78+8*math.sin(t*12),'#ff6472',5)
        v.rect(im,(18,18,panel_width-18,HEIGHT-8),None,10,'#ff6472',6)
    clock=max(0,t-7) if mode=='static' else t
    if t>=meta.get('contact_time',8.95):
        outcome='Path remains clear' if mode=='static' else 'Collision · movement interrupted'
        if contact:
            w=min(panel_width-25,50+v.font(33,600).getlength(outcome))
            v.rect(im,(23,857,w,915),v.INK,9,'#ff6472',2)
        v.text(im,(31,869),outcome,33,'#ffb2ba' if contact else color,600)
    else:
        description=('World frozen while the agent thinks.' if mode=='static' else 'Pedestrians move while the agent thinks.') if t<7 else 'The world advances during the move.'
        v.text(im,(31,869),description,30,v.WHITE,500)
    v.text(im,(31,925),f'World time +{clock:.1f} s',25,v.WHITE,500)
    v.text(im,(panel_width-28,971),'Illustrative encounter · Madison Square Park',15,'#d4e2e5',400,'ra')
    return im
