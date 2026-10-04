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
    data=json.loads((NYC/'evidence/task19-replay.json').read_text())
    for name,model in data['models'].items():
        raw=NYC/'encoded'/f'{name}.mp4';encode(NYC/'renders/task19'/name,name,raw)
        clip=film.Clip(raw);process=writer(MEDIA/f'{name}.mp4')
        for frame in range(1323):
            t=frame/30;state=replay.sample_agent(model,t*6);counts=replay.counts_reported_by(model,t*6)
            im=clip.frame(min(t,math.ceil(model['summary']['source_duration_seconds']*5)/30));v.rect(im,(0,993,1920,1080),v.INK)
            phase='Segment complete' if state['finished'] else 'Inference' if state['phase']=='inference' else 'Action'
            v.text(im,(34,1011),phase,40,v.WHITE,600)
            v.text(im,(690,1011),f'Decision {state["decision"]:02} / {model["summary"]["decisions"]}',40,v.PALE,500)
            v.text(im,(1885,1011),f'Reported collisions: {counts["total"]}',40,v.WHITE,500,'ra')
            process.stdin.write(im.tobytes())
            if frame==0:im.save(MEDIA/f'{name}.webp',quality=91)
        finish(process);print('Exported',name,flush=True)
    clips=[film.Clip(MEDIA/f'{n}.mp4') for n in ('astra','sol')]
    process=writer(ROOT/'public/media/rt-safe-nyc-comparison.mp4',crf=19)
    for frame in range(1323):
        t=frame/30;im=Image.new('RGB',(1920,1080),v.BG)
        v.text(im,(65,40),'RT-Safe',28,v.TEAL,700)
        v.text(im,(1855,46),'NYC RECONSTRUCTION  /  ORIGINAL RECORDED BEHAVIOR',19,v.MUTED,500,'ra')
        v.line(im,[(65,96),(1855,96)])
        v.text(im,(65,133),'Same route. Different decisions.',62,v.INK,500)
        v.text(im,(68,218),'RT15 · Task 19 · final subgoal · easy difficulty · low reasoning · 6× playback',26,v.MUTED,400)
        for i,(name,model) in enumerate(data['models'].items()):
            x=65+i*920;col=v.TEAL if i==0 else v.BLUE
            v.rect(im,(x,278,x+870,929),v.WHITE,12,v.LINE)
            v.text(im,(x+23,297),'GPT-6 Astra' if i==0 else 'GPT-5.6 Sol',37,col,600)
            im.paste(clips[i].frame(t).resize((870,489),Image.Resampling.LANCZOS),(x,358))
            v.text(im,(x+23,869),f'{model["summary"]["collisions"]} collision'+('' if i==0 else 's')+f' · {model["summary"]["decisions"]} decisions',32,col,600)
            v.text(im,(x+847,916),'FULL SELECTED SEGMENT',16,v.MUTED,400,'ra')
        v.text(im,(68,966),'Original agent poses and simulation clock; surrounding patrols reconstructed in the NYC scene.',21,v.MUTED,400)
        v.text(im,(68,1003),'Reported collision totals come from the source logs. This visualization is not a new NYC evaluation.',20,v.MUTED,400)
        process.stdin.write(im.tobytes())
        if frame==0:im.save(MEDIA/'comparison-poster.webp',quality=93)
    finish(process)
    (ROOT/'public/media/rt-safe-nyc-comparison.vtt').write_text('''WEBVTT

00:00:00.000 --> 00:00:11.437
GPT-6 Astra and GPT-5.6 Sol follow their recorded Task 19 final-subgoal trajectories, reconstructed in NYC at 6× speed.

00:00:11.437 --> 00:00:26.000
Astra has completed the segment: 12 decisions, 1 recorded collision. Sol is still navigating.

00:00:26.000 --> 00:00:42.085
The agent pauses during inference while surrounding patrols continue moving. Source collision counts appear as they are reported.

00:00:42.085 --> 00:00:44.100
Sol completes the segment: 33 decisions, 17 recorded collisions. These are original RT15 measurements, not a new evaluation on NYC.
''')
    print('Exported complete 44.1-second comparison',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['timing','comparison','hero']);args=parser.parse_args()
    MEDIA.mkdir(parents=True,exist_ok=True)
    globals()[args.mode]()
