"""Assemble narrated scenes, normalize audio, and export a captioned conference cut."""
from pathlib import Path
import json,subprocess,re
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'video';TL=json.loads((OUT/'timeline.json').read_text())
def run(args):subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*args],check=True)
def concatenate(paths,output):
 f=OUT/(output.stem+'-concat.txt');f.write_text(''.join("file '"+str(p).replace("'","'\\''")+"'\n" for p in paths))
 run(['-f','concat','-safe','0','-i',str(f),'-c','copy','-movflags','+faststart',str(output)]);f.unlink()
paths=[]
for s in TL['scenes']:
 out=OUT/'scenes'/(s['id']+'-voiced.mp4');delay=round(s['voiceOffset']*1000)
 run(['-i',str(OUT/'scenes'/(s['id']+'.mp4')),'-i',str(OUT/'audio'/(s['id']+'.mp3')),'-filter_complex',f'[1:a]adelay={delay}|{delay},apad[a]','-map','0:v','-map','[a]','-t',str(s['duration']),'-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-ac','2',str(out)])
 paths.append(out)
raw=OUT/'rt-safe-master-raw.mp4';concatenate(paths,raw)
master=OUT/'rt-safe-demo-1080p.mp4'
# One consistent narration level. No stock music is needed for clear academic narration.
run(['-i',str(raw),'-af','loudnorm=I=-16:TP=-1.5:LRA=9','-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart',str(master)])
raw.unlink()
# Web copy keeps 1080p text sharp and is ready for GitHub's file-size limit.
run(['-i',str(master),'-c:v','libx264','-preset','slow','-threads','8','-crf','23','-c:a','aac','-b:a','128k','-movflags','+faststart',str(ROOT/'public/media/rt-safe-demo.mp4')])
# A one-minute highlights cut: problem, completion gap, collision exposure, closing.
selected=['opening','completion','collisions','closing'];teaser=OUT/'rt-safe-highlights.mp4';concatenate([OUT/'scenes'/(s+'-voiced.mp4') for s in selected],teaser)
run(['-i',str(teaser),'-af','loudnorm=I=-16:TP=-1.5:LRA=9','-c:v','libx264','-preset','fast','-threads','8','-crf','23','-c:a','aac','-b:a','128k','-movflags','+faststart',str(ROOT/'public/media/rt-safe-highlights.mp4')])
# Export ASS at native resolution to make a dependable burned-caption edition.
vtt=(ROOT/'public/media/rt-safe-demo.vtt').read_text()
def ass_stamp(ts):
 h,m,sec=ts.split(':');return f'{int(h)}:{m}:{float(sec):05.2f}'
ass='''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DM Sans,28,&H00FFFFFF,&H00FFFFFF,&H001A2B22,&H001A2B22,0,0,0,0,100,100,0,0,3,7,0,2,130,130,13,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
for block in vtt.split('\n\n'):
 if '-->' not in block:continue
 lines=block.strip().splitlines();a,b=lines[0].split(' --> ');text=' '.join(lines[1:]).replace('{','').replace('}','')
 ass+=f'Dialogue: 0,{ass_stamp(a)},{ass_stamp(b)},Default,,0,0,0,,{text}\n'
ass_path=OUT/'captions.ass';ass_path.write_text(ass)
run(['-i',str(master),'-vf',f'ass={ass_path}:fontsdir={ROOT/"public/fonts"}','-c:v','libx264','-preset','fast','-threads','8','-crf','20','-c:a','copy','-movflags','+faststart',str(OUT/'rt-safe-conference-captioned.mp4')])
# Shift VTT intervals into highlight time.
blocks=vtt.split('\n\n');high=['WEBVTT',''];cursor=0
for name in selected:
 s=next(x for x in TL['scenes'] if x['id']==name)
 def seconds(st):
  h,m,v=st.split(':');return int(h)*3600+int(m)*60+float(v)
 def stamp(v):
  ms=round(v*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}.{ms%1000:03}'
 for block in blocks:
  if '-->' not in block:continue
  lines=block.splitlines();a,b=lines[0].split(' --> ');a=seconds(a);b=seconds(b)
  if s['start']<=a<s['start']+s['duration']:high += [f'{stamp(a-s["start"]+cursor)} --> {stamp(b-s["start"]+cursor)}',' '.join(lines[1:]),'']
 cursor+=s['duration']
(ROOT/'public/media/rt-safe-highlights.vtt').write_text('\n'.join(high))
# Chapter markers travel with the edit sources.
(OUT/'chapters.txt').write_text('\n'.join(f'{int(s["start"])//60:02}:{int(s["start"])%60:02} {s["chapter"]}' for s in TL['scenes'])+'\n')
import shutil
shutil.copy2(OUT/'rt-safe-conference-captioned.mp4',ROOT/'public/media/rt-safe-conference-captioned.mp4')
print('Exported full film, web MP4, highlights, and conference captions.',flush=True)
