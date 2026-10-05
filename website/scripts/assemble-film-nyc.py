"""Export a frame-exact 90-second silent film without subtitle tracks."""
from pathlib import Path
import json, shutil, subprocess

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

# Preserve the old download URL, now pointing to the same subtitle-free film.
shutil.copy2(master, MEDIA / 'rt-safe-nyc-90s-captioned.mp4')
(OUT / 'chapters.txt').write_text('\n'.join(
    f'{s["start"] // 60:02}:{s["start"] % 60:02} {s["chapter"]}' for s in TL['scenes']) + '\n')
print('Exported subtitle-free compatibility alias', flush=True)
