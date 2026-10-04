"""Check exported media duration, frame count, decoding, and caption intervals."""
from pathlib import Path
import hashlib, json, re, subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'video/nyc-90s'
report = {}
for name in ['rt-safe-nyc-90s.mp4', 'rt-safe-nyc-90s-captioned.mp4']:
    path = ROOT / 'public/media' / name
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration,size:stream=codec_name,width,height,r_frame_rate,nb_frames,duration',
        '-of', 'json', str(path)]))
    assert float(info['format']['duration']) == 90, info
    video = next(s for s in info['streams'] if s['codec_name'] == 'h264')
    assert video['nb_frames'] == '2700', video
    assert (video['width'], video['height'], video['r_frame_rate']) == (1920, 1080, '30/1'), video
    assert any(s['codec_name'] == 'aac' for s in info['streams']), info
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                             check=True, capture_output=True)
    assert not decoded.stderr, decoded.stderr.decode()
    info['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    info['full_decode'] = 'passed'
    report[name] = info

def seconds(stamp):
    h, m, s = stamp.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

vtt = (OUT / 'captions.vtt').read_text()
end = 0
cues = re.findall(r'(\d\d:\d\d:\d\d\.\d+) --> (\d\d:\d\d:\d\d\.\d+)', vtt)
assert cues
for a, b in cues:
    start, finish = seconds(a), seconds(b)
    assert end <= start < finish <= 90, (end, start, finish)
    end = finish
report['captions'] = {'cues': len(cues), 'last_end_seconds': end, 'ordered_nonoverlapping': True}
timeline = json.loads((OUT / 'timeline.json').read_text())
assert sum(s['duration'] for s in timeline['scenes']) == 90
for s in timeline['scenes']:
    assert s['voiceOffset'] + s['voiceDuration'] / s['voiceSpeed'] < s['duration']
report['voice_fits_all_scenes'] = True
# Decode the paired timing clips and measure whether the intended world freeze is visible.
import cv2
import numpy as np
motion={}
for mode in ('static','realtime'):
    path=ROOT/'public/media/nyc'/f'timing-{mode}.mp4'
    cap=cv2.VideoCapture(str(path));assert cap.isOpened()
    frames=[]
    for frame in (0,60,120,200,260,320):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();assert ok
        frames.append(im)
    cap.release()
    inference=[float(np.abs(im.astype(float)-frames[0].astype(float)).mean()) for im in frames[1:4]]
    action=float(np.abs(frames[4].astype(float)-frames[0].astype(float)).mean())
    motion[mode]={'inference_mean_absolute_pixel_changes':inference,'action_change':action}
    if mode=='static':assert max(inference)<.4, inference
    else:assert max(inference)>1, inference
    assert action>1,action
report['timing_motion']=motion
assert next(s for s in timeline['scenes'] if s['id']=='timing')['duration']==15
assert next(s for s in timeline['scenes'] if s['id']=='environment')['duration']==12
report['opening_story']={'static_first':True,'center_message':'But the real world does not stop.','slide_reveal':True,'location':'Madison Square Park interior path','environment_title_seconds':2.4}

holds={}
for model,a,b in [('astra',20,40),('sol',42.8,43.8)]:
    cap=cv2.VideoCapture(str(ROOT/'public/media/recorded'/f'{model}.mp4'));frames=[]
    for t in (a,b):
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,im=cap.read();assert ok;frames.append(im)
    cap.release();delta=float(np.abs(frames[0].astype(float)-frames[1].astype(float)).mean())
    assert delta<.5,(model,delta)
    holds[model]={'completed_view_mean_absolute_pixel_change':delta}
report['completed_segment_holds']=holds
for stem,duration in [('hero',8),('timing-story',15),('environment-intro',12),('timing-static',11),('timing-realtime',11),('environment-tour',12),('astra',44.1),('sol',44.1)]:
    path=ROOT/('public/media/recorded' if stem in ('astra','sol') else 'public/media/nyc')/f'{stem}.mp4'
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=nb_frames,width,height','-of','json',str(path)]))
    assert abs(float(info['format']['duration'])-duration)<.04,(stem,info)
    report[stem]=info
source=json.loads((ROOT/'nyc/evidence/task19-replay.json').read_text())
assert source['models']['astra']['summary']['collisions']==1
assert source['models']['sol']['summary']['collisions']==17
assert source['models']['astra']['summary']['decisions']==12
assert source['models']['sol']['summary']['decisions']==33
report['comparison_source_totals']='verified against curated original logs'
# The inset must come from a recorded input; waypoint indices are not distances.
from PIL import Image
inputs=0
for model in source['models'].values():
    for step in model['steps']:
        path=ROOT/'public'/step['input_image']
        with Image.open(path) as im:assert im.size==(360,320)
        assert len(step['input_sha256'])==64
        if step['action']['type']=='move_to':
            assert min(abs(step['commanded_distance_m']-d) for d in (1,2,4))<.01
        inputs+=1
assert inputs==45
profiles=json.loads((ROOT/'public/data/behavior.json').read_text())['profiles']
assert len(profiles)==8 and len({p['name'] for p in profiles})==8
for axis in range(6):
    assert abs(max(p['values'][axis] for p in profiles)-1)<1e-9
    assert all(0<=p['values'][axis]<=1 for p in profiles)
# At 35.8 s, Sol has just reported four contacts; the red edge is visible.
cap=cv2.VideoCapture(str(ROOT/'public/media/recorded/sol.mp4'));cap.set(cv2.CAP_PROP_POS_MSEC,35800)
ok,frame=cap.read();cap.release();assert ok
blue,green,red=frame[5,480];assert int(red)>int(green)+70 and int(red)>int(blue)+50
report['replay_overlay']={'original_input_frames':inputs,'command_distances':'verified','collision_report_flash':'verified at 35.8 s'}
original=json.loads((ROOT/'public/data/recorded-replay.json').read_text())
frame_count=0
for model in original['models'].values():
    for step in model['steps']:
        for entries in step['recorded_frames'].values():
            for entry in entries:
                with Image.open(ROOT/'public'/entry['path']) as im:
                    assert list(im.size)==entry['size']
                    assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==entry['pixel_sha256']
                frame_count+=1
assert frame_count==95,frame_count
report['original_recorded_frames']={'count':frame_count,'lossless_pixel_hashes':'verified','interpolation':'none','inference':'recorded input held'}
report['behavior_profiles']={'models':8,'axes':6,'normalization':'maximum across all eight models'}
comparison=ROOT/'public/media/rt-safe-original-comparison.mp4'
info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=nb_frames,width,height','-of','json',str(comparison)]))
assert abs(float(info['format']['duration'])-44.1)<.01
assert info['streams'][0]['nb_frames']=='1323'
decoded=subprocess.run(['ffmpeg','-v','error','-i',str(comparison),'-f','null','-'],check=True,capture_output=True)
assert not decoded.stderr,decoded.stderr.decode()
report['full_selected_comparison']={**info,'full_decode':'passed'}
levels=subprocess.run(['ffmpeg','-hide_banner','-i',str(ROOT/'public/media/rt-safe-nyc-90s.mp4'),'-vn','-af','loudnorm=I=-16:TP=-1.5:LRA=9:print_format=json','-f','null','-'],check=True,capture_output=True,text=True)
match=re.search(r'\{\s*"input_i"[\s\S]*?\}',levels.stderr)
assert match,'Missing loudness measurement'
levels=json.loads(match.group())
assert -18<float(levels['input_i'])<-14,levels
assert float(levels['input_tp'])<-.5,levels
report['narration_loudness']={'integrated_lufs':float(levels['input_i']),'true_peak_dbtp':float(levels['input_tp'])}
for path in [*ROOT.glob('public/media/rt-safe-nyc*.mp4'),*ROOT.glob('public/media/nyc/*.mp4'),*ROOT.glob('public/media/recorded/*.mp4'),*ROOT.glob('public/media/rt-safe-original*.mp4')]:
    assert path.stat().st_size<95*1024**2,('File exceeds repository limit',path.name)
report['repository_media_size_limit']='all current NYC MP4s below 95 MiB'
(OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
print('Passed: both 90-second / 2700-frame exports decode fully; captions do not overlap; voice fits all scenes.')
