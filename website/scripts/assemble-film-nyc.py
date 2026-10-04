"""Export a frame-exact 90-second film, with optional burned-in captions."""
from pathlib import Path
import json, shutil, subprocess, importlib.util

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'video/nyc-90s'
MEDIA = ROOT / 'public/media'
TL = json.loads((OUT / 'timeline.json').read_text())

def run(args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *args], check=True)

def concat_file(paths, name):
    path = OUT / name
    path.write_text(''.join("file '" + str(p).replace("'", "'\\''") + "'\n" for p in paths))
    return path

video_list = concat_file([OUT / 'scenes' / (s['id'] + '.mp4') for s in TL['scenes']], 'video-concat.txt')
master = MEDIA / 'rt-safe-nyc-90s.mp4'
run(['-f', 'concat', '-safe', '0', '-i', str(video_list),
     '-map', '0:v:0', '-an', '-c:v', 'copy',
     '-t', '90', '-movflags', '+faststart', str(master)])
shutil.copy2(master, MEDIA / 'rt-safe-video.mp4')
print('Exported silent master and canonical rt-safe-video.mp4', flush=True)

def ass_stamp(ts):
    h, m, sec = ts.split(':')
    return f'{int(h)}:{m}:{float(sec):05.2f}'

ass = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Space Grotesk,32,&H00FFFFFF,&H00FFFFFF,&H00502D15,&H00502D15,0,0,0,0,100,100,0,0,1,0,0,2,90,90,25,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
for block in (OUT / 'captions.vtt').read_text().split('\n\n'):
    if '-->' not in block:
        continue
    lines = block.strip().splitlines()
    start, end = lines[0].split(' --> ')
    caption = ' '.join(lines[1:]).replace('{', '').replace('}', '')
    ass += f'Dialogue: 0,{ass_stamp(start)},{ass_stamp(end)},Default,,0,0,0,,{caption}\n'
ass_path = OUT / 'captions.ass'
ass_path.write_text(ass)
run(['-i', str(master), '-vf', f'ass={ass_path}:fontsdir={ROOT / "public/fonts"}',
     '-c:v', 'libx264', '-preset', 'fast', '-threads', '8', '-crf', '19',
     '-an', '-t', '90', '-movflags', '+faststart',
     str(MEDIA / 'rt-safe-nyc-90s-captioned.mp4')])
shutil.copy2(OUT / 'captions.vtt', MEDIA / 'rt-safe-nyc-90s.vtt')
shutil.copy2(OUT / 'captions.srt', MEDIA / 'rt-safe-nyc-90s.srt')
(OUT / 'chapters.txt').write_text('\n'.join(
    f'{s["start"] // 60:02}:{s["start"] % 60:02} {s["chapter"]}' for s in TL['scenes']) + '\n')
print('Exported captioned edition and VTT/SRT captions', flush=True)
