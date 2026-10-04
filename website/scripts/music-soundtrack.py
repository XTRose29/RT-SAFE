"""Compose a quiet original instrumental and replace narration without re-encoding video."""
from pathlib import Path
import argparse, hashlib, json, subprocess, tempfile, wave
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'video/nyc-90s'
TRACK=OUT/'audio/soft-instrumental.flac'
SR=48000
DURATION=90

def run(args,**kw):
    return subprocess.run(args,check=True,**kw)

def generate():
    TRACK.parent.mkdir(parents=True,exist_ok=True)
    mix=np.zeros((SR*DURATION,2),dtype=np.float32)
    beat=60/64
    def add(midi,start,length,volume,pan=0,kind='keys'):
        count=min(int(length*SR),len(mix)-int(start*SR))
        if count<=0:return
        t=np.arange(count,dtype=np.float32)/SR
        hz=440*2**((midi-69)/12)
        if kind=='pad':
            wave_=sum(np.sin(2*np.pi*hz*ratio*t+phase)*amp for ratio,phase,amp in [(1,0,.65),(1.0018,.7,.2),(.9984,1.3,.15),(2,.3,.06)])
            env=np.minimum(t/1.6,1)*np.minimum(np.maximum(length-t,0)/2.2,1)
        else:
            wave_=np.sin(2*np.pi*hz*t)*np.exp(-t/2.5)
            wave_+=.22*np.sin(2*np.pi*hz*2*t+.2)*np.exp(-t/.9)
            wave_+=.07*np.sin(2*np.pi*hz*3*t)*np.exp(-t/.45)
            env=(1-np.exp(-t/.022))*np.minimum(np.maximum(length-t,0)/.45,1)
        sig=wave_*env*volume
        offset=int(start*SR)
        mix[offset:offset+count,0]+=sig*np.sqrt((1-pan)/2)
        mix[offset:offset+count,1]+=sig*np.sqrt((1+pan)/2)
    # D major: warm ninth/seventh voicings. Two bars per chord, 24 bars total.
    progression=[(38,[57,61,64,66]),(35,[54,57,62,66]),(31,[55,59,62,66]),(33,[57,59,64,69])]*2
    progression += [(35,[54,57,62,66]),(31,[55,59,62,66]),(33,[57,59,64,69]),(38,[57,61,64,66])]
    for i,(bass,notes) in enumerate(progression):
        start=i*8*beat
        for j,note in enumerate(notes):add(note,start,9.3,.025,(-.45+j*.3),'pad')
        add(bass,start,7.2,.055,0,'keys')
        # Sparse, rounded electric-piano arpeggio, leaving space between phrases.
        for j,offset in enumerate([.5,2,4.5,6]):
            add(notes[[0,2,1,3][j]],start+offset*beat,4.1,.071 if j%2==0 else .054,[-.22,.2,-.1,.25][j])
    melody=[(4,74),(7,73),(12,69),(15,66),(20,71),(23,69),(28,73),(31,76),(36,74),(39,73),(44,69),(47,66),(52,71),(55,74),(60,73),(63,69),(68,71),(71,69),(76,73),(79,76),(84,74)]
    for start,note in melody:add(note,start,4.8,.054,.08)
    # A diffuse stereo tail; no percussion, samples, or voice recordings.
    dry=mix.copy()
    for delay,level in [(.17,.12),(.29,.10),(.43,.09),(.61,.075),(.83,.05),(1.13,.04)]:
        shift=int(delay*SR);mix[shift:]+=dry[:-shift,::-1]*level
    fadein=np.minimum(np.arange(len(mix))/(SR*2.5),1)
    fadeout=np.minimum((len(mix)-1-np.arange(len(mix)))/(SR*4.5),1)
    mix*= (fadein*fadeout)[:,None]
    peak=float(np.max(np.abs(mix)));mix*=.72/max(peak,.001)
    with tempfile.TemporaryDirectory() as tmp:
        raw=Path(tmp)/'instrumental.wav'
        with wave.open(str(raw),'wb') as f:
            f.setnchannels(2);f.setsampwidth(2);f.setframerate(SR)
            f.writeframes((mix*32767).astype('<i2').tobytes())
        run(['ffmpeg','-v','error','-y','-i',str(raw),'-af','loudnorm=I=-25:TP=-5:LRA=7','-ar',str(SR),'-c:a','flac',str(TRACK)])
    print('Composed 90-second soft instrumental',flush=True)
    return TRACK

def music_filter(duration,fade=True):
    return f'afade=t=in:d=0.8,afade=t=out:st={duration-1.2}:d=1.2' if fade else 'anull'

def video_hash(path):
    return subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-c:v','copy','-f','hash','-hash','sha256','-'],text=True).strip()

def replace_audio(path,start,duration):
    before=video_hash(path)
    temp=path.with_name(path.stem+'.music-tmp.mp4')
    run(['ffmpeg','-v','error','-y','-i',str(path),'-ss',str(start),'-i',str(TRACK),
         '-map','0:v:0','-map','1:a:0','-c:v','copy','-af',music_filter(duration,duration<90),
         '-c:a','aac','-b:a','192k','-ar',str(SR),'-t',str(duration),'-movflags','+faststart',str(temp)])
    after=video_hash(temp);assert before==after,(path,'Video stream changed')
    expected=subprocess.check_output(['ffmpeg','-v','error','-ss',str(start),'-i',str(TRACK),'-af',music_filter(duration,duration<90),'-c:a','aac','-b:a','192k','-ar',str(SR),'-t',str(duration),'-f','hash','-hash','sha256','-'],text=True).strip()
    actual=subprocess.check_output(['ffmpeg','-v','error','-i',str(temp),'-map','0:a:0','-c:a','copy','-f','hash','-hash','sha256','-'],text=True).strip()
    assert actual==expected,(path,'Audio does not match instrumental')
    temp.replace(path)
    return {'audio_matches_instrumental_encode':True,'audio_stream_sha256':actual.split('=')[-1],'video_stream_sha256':after.split('=')[-1],'video_unchanged':True,'duration_seconds':duration,'audio':'original soft instrumental; no voice','music_start_seconds':start}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    generate()
    if args.apply:
        results={}
        for relative,start,duration in [('rt-safe-nyc-90s.mp4',0,90),('rt-safe-nyc-90s-captioned.mp4',0,90),('nyc/timing-story.mp4',3,15),('nyc/environment-intro.mp4',18,12)]:
            results[relative]=replace_audio(ROOT/'public/media'/relative,start,duration)
            print('Replaced narration:',relative,flush=True)
        (OUT/'soundtrack-validation.json').write_text(json.dumps({'track_sha256':hashlib.sha256(TRACK.read_bytes()).hexdigest(),'exports':results},indent=2)+'\n')
