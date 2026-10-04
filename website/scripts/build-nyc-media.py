"""Encode native MRQ frames and assemble the synchronized recorded comparison."""
from pathlib import Path
import argparse,importlib.util,json,math,subprocess,sys
import cv2
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];NYC=ROOT/'nyc';MEDIA=ROOT/'public/media/nyc'
sys.path.insert(0,str(NYC/'scripts'))
import replay_math as replay
spec=importlib.util.spec_from_file_location('film',ROOT/'scripts/render-film-nyc.py')
film=importlib.util.module_from_spec(spec);spec.loader.exec_module(film)
v=film.v
def writer(path,width=1920,height=1080,crf=20):
    path.parent.mkdir(parents=True,exist_ok=True)
    return subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf',str(crf),'-threads','6',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
def finish(process):
    process.stdin.close()
    if process.wait()!=0:raise RuntimeError('ffmpeg export failed')
def encode(folder,stem,dest):
    frames=sorted(folder.glob(stem+'_*.png'))
    if not frames:raise RuntimeError('No native frames in '+str(folder))
    first=int(frames[0].stem.rsplit('_',1)[1]);nums=[int(p.stem.rsplit('_',1)[1]) for p in frames]
    assert nums==list(range(first,first+len(frames))), 'Missing MRQ frames'
    dest.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ffmpeg','-v','error','-y','-framerate','30','-start_number',str(first),'-i',str(folder/(stem+'_%04d.png')),
        '-frames:v',str(len(frames)),'-an','-c:v','libx264','-preset','fast','-crf','17','-threads','6','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(dest)],check=True)
    return len(frames)
def hero():
    master=NYC/'encoded/hero-master.mp4'
    encode(NYC/'renders/hero','hero',master)
    web=NYC/'encoded/hero-web.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(master),'-vf','scale=1920:1080:flags=lanczos',
        '-an','-c:v','libx264','-preset','slow','-crf','23','-maxrate','8M','-bufsize','16M','-threads','6',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(web)],check=True)
    import shutil
    shutil.copy2(web,MEDIA/'hero.mp4')
    print('Exported web aerial loop; high-quality master retained locally',flush=True)
def timing():
    spec=importlib.util.spec_from_file_location('timing_presentation',ROOT/'scripts/timing-presentation.py')
    presentation=importlib.util.module_from_spec(spec);spec.loader.exec_module(presentation)
    for name in ('static','realtime'):
        stem='timing-close-'+name;raw=NYC/'encoded'/f'{stem}.mp4'
        encode(NYC/'renders'/stem,stem,raw)
        meta=json.loads((NYC/'renders'/stem/'scene.json').read_text())
        clip=film.Clip(raw);process=writer(MEDIA/f'timing-{name}.mp4',width=960,height=1000,crf=19)
        for f in range(330):
            t=f/30;source=0 if name=='static' and t<7 else f
            im=presentation.compose(clip.frame(source/30),meta,source,t,name,v)
            process.stdin.write(im.tobytes())
            if f==0:im.save(MEDIA/f'timing-{name}.webp',quality=93)
        finish(process);print('Exported closer sidewalk timing scene:',name,flush=True)
def comparison():
    spec=importlib.util.spec_from_file_location('replay_presentation',ROOT/'scripts/replay-presentation.py')
    hud=importlib.util.module_from_spec(spec);spec.loader.exec_module(hud)
    data=json.loads((NYC/'evidence/task19-replay.json').read_text())
    for name,model in data['models'].items():
        raw=NYC/'encoded'/f'{name}.mp4'
        if not raw.exists():encode(NYC/'renders/task19'/name,name,raw)
        clip=film.Clip(raw);process=writer(MEDIA/f'{name}.mp4',width=960,height=1000,crf=19)
        end=model['summary']['source_duration_seconds']/6
        for frame in range(1323):
            t=frame/30
            im=hud.compose(clip.frame(min(t,end)),model,name,t,v)
            process.stdin.write(im.tobytes())
            if frame==0:im.save(MEDIA/f'{name}.webp',quality=93)
        finish(process);print('Exported game-style replay:',name,flush=True)
    clips=[film.Clip(MEDIA/f'{n}.mp4') for n in ('astra','sol')]
    process=writer(ROOT/'public/media/rt-safe-nyc-comparison.mp4',crf=19)
    for frame in range(1323):
        t=frame/30;im=Image.new('RGB',(1920,1080),v.INK)
        for i,clip in enumerate(clips):im.paste(clip.frame(t),(i*960,0))
        v.line(im,[(959,0),(959,1000)],v.WHITE,3)
        v.text(im,(30,1016),'NYC reconstruction · original observations and recorded agent behavior · RT15 / Task 19 / 6×',24,v.WHITE,500)
        v.text(im,(30,1051),'Collision alerts mark report times in the source logs; contacts are not newly measured in NYC.',18,v.PALE,400)
        process.stdin.write(im.tobytes())
        if frame==0:im.save(MEDIA/'comparison-poster.webp',quality=93)
    finish(process)
    print('Exported complete 44.1-second comparison',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['timing','comparison','hero']);args=parser.parse_args()
    MEDIA.mkdir(parents=True,exist_ok=True)
    globals()[args.mode]()
