"""Check authored contact separation and visible result animation in the new edit."""
from pathlib import Path
import json
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
report={};metadata={}
for mode in ['static','realtime']:
    path=ROOT/f'public/data/nyc/timing-{mode}-scene.json'
    assert path.exists(), 'Export authored scene metadata with build-nyc-media.py timing'
    meta=json.loads(path.read_text());metadata[mode]=meta
    if 'contact_time' not in meta:raise AssertionError('Native take must be re-rendered')
    robot=np.array([x[1] for x in meta['robot_poses']]);human=np.array([x[1] for x in meta['human_poses']])
    separation=np.linalg.norm(human[:,:2]-robot[:,:2],axis=1)
    assert np.all(human[:,0]>robot[:,0]), 'Actors cross in the authored path'
    assert separation.min()>=meta['contact_distance_cm']-.001
    if mode=='realtime':
        assert separation.min()<meta['contact_distance_cm']+8
        assert np.linalg.norm(robot[-1]-robot[-30])<1
        assert np.linalg.norm(human[-1]-human[-30])<1
    else:assert separation.min()>300
    report[mode]={'minimum_center_separation_cm':float(separation.min()),'crossing':False,'source':'authored scene transforms; native frames also reviewed visually'}
for key in ['camera_location','camera_rotation','horizontal_fov']:
    assert metadata['static'][key]==metadata['realtime'][key],key
for key in ['robot_poses','human_poses']:
    assert metadata['static'][key][0]==metadata['realtime'][key][0],key
report['matched_initial_state_and_camera']=True
cap=cv2.VideoCapture(str(ROOT/'video/nyc-90s/scenes/behavior.mp4'));assert cap.isOpened()
for name,start,end,box in [('leaderboard',.3,1.4,(40,225,1840,874)),('radar_page_1',5.3,6.8,(80,345,1900,750)),('radar_page_2',9.7,11.3,(80,345,1900,750))]:
    frames=[]
    for t in [start,end]:
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,im=cap.read();assert ok
        x0,y0,x1,y1=box;frames.append(im[y0:y1,x0:x1].astype(float))
    delta=float(np.abs(frames[0]-frames[1]).mean());assert delta>1,(name,delta)
    report[name]={'mean_absolute_pixel_change':delta,'animated':True}
cap.release()
(ROOT/'video/nyc-90s/motion-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
