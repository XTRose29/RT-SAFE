#!/usr/bin/env python3
"""Package six social posts, approved silent clips, and original paper figures.

Requires ffmpeg, ffprobe and pymupdf. No generated or redrawn research figures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=ROOT/'runtime/film-60s/source-90s.mp4')
parser.add_argument('--paper-source', type=Path, help='Optional latest paper-source ZIP; figures are copied byte-for-byte')
parser.add_argument('--output', type=Path, default=ROOT/'runtime/rt-safe-social-kit')
args = parser.parse_args()
posts = json.loads((ROOT/'website/video/social-kit/posts.json').read_text())
recipe = json.loads((ROOT/'website/video/nyc-60s/timeline.json').read_text())
assert hashlib.sha256(args.source.read_bytes()).hexdigest() == recipe['sourceSha256'], 'Unexpected source film'
args.output.mkdir(parents=True, exist_ok=True)
paper = ROOT/'website/public/media/rt-safe-paper.pdf'
doc = pymupdf.open(paper)
manifest = {'paperSha256': hashlib.sha256(paper.read_bytes()).hexdigest(), 'assets': []}
source_zip = zipfile.ZipFile(args.paper_source) if args.paper_source else None
if source_zip:
    manifest['paperSourceArchive'] = args.paper_source.name
    manifest['paperSourceSha256'] = hashlib.sha256(args.paper_source.read_bytes()).hexdigest()
for i, post in enumerate(posts):
    folder = args.output/post['folder']
    folder.mkdir(exist_ok=True)
    asset = folder/post['attachment']
    if i == 0:
        shutil.copyfile(ROOT/'website/public/media/rt-safe-video.mp4', asset)
    elif i in (1, 2):
        start, duration = (3, 15) if i == 1 else (18, 12)
        subprocess.run(['ffmpeg','-y','-v','error','-ss',str(start),'-i',str(args.source),
                        '-t',str(duration),'-map','0:v:0','-an','-sn','-dn',
                        '-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p',
                        '-r','30','-movflags','+faststart',str(asset)],check=True)
    else:
        if source_zip:
            name = ('figure/rq1_main.png', 'figure/rq2.png', 'figure/rq3.png')[i-3]
            asset.write_bytes(source_zip.read(name))
            pix = pymupdf.Pixmap(str(asset))
        else:
            page = doc[i+2]
            assert len(page.get_images()) == 1
            pix = pymupdf.Pixmap(doc, page.get_images()[0][0])
            pix.save(asset)
    (folder/'post.txt').write_text(post['text']+'\n')
    (folder/'alt-text.txt').write_text(post['alt']+'\n')
    (folder/'source.txt').write_text(post['source']+'\n')
    entry = {'file':str(asset.relative_to(args.output)), 'sha256':hashlib.sha256(asset.read_bytes()).hexdigest(), 'source':post['source']}
    if i < 3:
        info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(asset)]))
        assert len(info['streams']) == 1 and info['streams'][0]['codec_type'] == 'video'
        duration = float(info['format']['duration'])
        assert duration == (60,15,12)[i]
        assert (info['streams'][0]['width'],info['streams'][0]['height']) == (1920,1080)
        entry.update(durationSeconds=duration, width=1920,height=1080,audioStreams=0,subtitleStreams=0)
    else:
        entry.update(width=pix.width,height=pix.height,paperPage=i+3,paperFigure=i)
    manifest['assets'].append(entry)
(args.output/'all-posts.txt').write_text('\n\n'.join(
    f"{p['title']}\nAttach: {p['folder']}/{p['attachment']}\n\n{p['text']}" for p in posts))
(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(args.output/'README.txt').write_text('''RT-SAFE social post kit

Six posts, ordered like the paper: announcement, motivation, environment, RQ1, RQ2, RQ3.
Each numbered folder contains one attachment, ready-to-copy post.txt, alt-text.txt, and source.txt.
Use the main announcement as the first post and publish 2–6 as replies, or adapt them as separate posts.

01-main: full 60-second demo
02-problem: 15-second illustrative static/real-time comparison
03-environment: 12-second NYC task and environment demonstration
04-model-results: original paper Figure 3
05-static-vs-realtime: original paper Figure 4
06-reasoning: original paper Figure 5

All videos are 1920 x 1080 H.264 at 30 fps. They have no audio or subtitle tracks.
Existing on-screen explanatory labels remain. No RL footage is included.
The short clips preserve the pacing of the approved original film before its 60-second recut.
The PNGs are original paper figures at native resolution. When a paper-source ZIP is supplied, its figure files are copied byte-for-byte. See manifest.json for source hashes.
Upload the video/image file to the post rather than a screenshot of its player.

Website and demo: https://xtrose29.github.io/RT-SAFE/
Code: https://github.com/XTRose29/RT-SAFE
Paper: https://xtrose29.github.io/RT-SAFE/media/rt-safe-paper.pdf
SimWorld: https://simworld.org/

Reading the results:
- Post 1: hard environments, provider-default reasoning (94.1% success).
- Post 4: all three difficulties, provider-default reasoning (94.4% success).
- Both settings round to 0.7% safe success.
- Figure 3's exposure measure is a proxy and its relationship is correlational.
- Static/real-time prompts include mode-specific timing instructions.
- Figure 5 plots lower/medium/higher settings. The accompanying 40.7 to 62.0 comparison is provider-default to higher, which is not identical to medium to higher for every model.
- NYC clips illustrate the environment; they do not replace the five evaluation maps or constitute new quantitative evidence.

Rebuild from the repository using:
python website/scripts/build-social-kit.py --source /path/to/approved-source-90s.mp4 --paper-source /path/to/paper-source.zip --output /path/to/rt-safe-social-kit
The source revision and hash are recorded in website/video/nyc-60s/timeline.json.
''')
archive = args.output.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for file in sorted(args.output.rglob('*')):
        if file.is_file(): z.write(file,Path(args.output.name)/file.relative_to(args.output))
print(json.dumps({'folder':str(args.output),'zip':str(archive),'bytes':archive.stat().st_size,'assets':manifest['assets']},indent=2))
