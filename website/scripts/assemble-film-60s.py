"""Re-cut the approved silent NYC master to exactly 60 seconds, excluding RL.

python scripts/assemble-film-60s.py --source /path/to/approved-90s-master.mp4
Source hash is pinned in video/nyc-60s/timeline.json. Intermediate files stay
outside the public assets. The four recorded-agent cuts retain their speed.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, tempfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'video/nyc-60s';MEDIA=ROOT/'public/media'
def run(args):subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*args],check=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);args=ap.parse_args();tl=json.loads((OUT/'timeline.json').read_text())
 assert hashlib.sha256(args.source.read_bytes()).hexdigest()==tl['sourceSha256'],'Unexpected source edit'
 with tempfile.TemporaryDirectory(prefix='rt-safe-60s-') as work:
  parts=[]
  for scene in tl['scenes']:
   for i,r in enumerate(scene['ranges']):
    a,b,n=(r[k] for k in ('sourceStartFrame','sourceEndFrame','outputFrames'))
    if scene['id']=='examples':assert n==b-a,'Never retime recorded HUD footage'
    target=Path(work)/f'{scene["id"]}-{i}.mp4';parts.append(target)
    vf=f'trim=start_frame={a}:end_frame={b},setpts=(PTS-STARTPTS)*{n}/{b-a},fps=30,tpad=stop_mode=clone:stop_duration=1,trim=end_frame={n},setpts=N/(30*TB)'
    run(['-i',str(args.source),'-map','0:v:0','-vf',vf,'-an','-sn','-c:v','libx264','-preset','slow','-crf','19','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(target)])
   print('Re-cut',scene['id'],scene['duration'],'s',flush=True)
  concat=Path(work)/'concat.txt';concat.write_text(''.join(f"file '{p}'\n" for p in parts))
  target=Path(work)/'rt-safe-video.mp4'
  run(['-f','concat','-safe','0','-i',str(concat),'-map','0:v:0','-an','-sn','-c:v','copy','-movflags','+faststart',str(target)])
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)]))
  assert len(probe['streams'])==1 and probe['streams'][0]['codec_type']=='video'
  assert (probe['streams'][0]['width'],probe['streams'][0]['height'],probe['streams'][0]['r_frame_rate'])==(1920,1080,'30/1')
  assert int(probe['streams'][0]['nb_frames'])==1800 and abs(float(probe['format']['duration'])-60)<.001
  for name in ('rt-safe-video.mp4','rt-safe-nyc-90s.mp4','rt-safe-nyc-90s-captioned.mp4'):shutil.copy2(target,MEDIA/name)
  (OUT/'validation.json').write_text(json.dumps({'duration':60,'frames':1800,'fps':30,'size':[1920,1080],'audioStreams':0,'subtitleStreams':0,'rlIncluded':False,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'compatibilityAliases':['rt-safe-nyc-90s.mp4','rt-safe-nyc-90s-captioned.mp4']},indent=2)+'\n')
 (OUT/'chapters.txt').write_text('\n'.join(f'{int(s["start"])//60:02}:{int(s["start"])%60:02} {s["chapter"]}' for s in tl['scenes'])+'\n')
 print('Published 1,800 frames: 60 seconds, silent, no RL.',flush=True)
if __name__=='__main__':main()
