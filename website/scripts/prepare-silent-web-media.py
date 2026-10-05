"""Remove audio/subtitle streams; keep legacy captioned URLs as plain-video aliases."""
from pathlib import Path
import json, shutil, subprocess
ROOT=Path(__file__).resolve().parents[1]
MEDIA=ROOT/'public/media'
aliases={
 'rt-safe-nyc-90s-captioned.mp4':'rt-safe-nyc-90s.mp4',
 'rt-safe-demo-90s-captioned.mp4':'rt-safe-demo-90s.mp4',
 'rt-safe-conference-captioned.mp4':'rt-safe-demo.mp4',
}
report={}
for path in sorted(MEDIA.rglob('*.mp4')):
 if path.name in aliases:continue
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type','-of','json',str(path)]))
 if any(s['codec_type']!='video' for s in info['streams']):
  temporary=path.with_name(path.stem+'-silent-tmp.mp4')
  subprocess.run(['ffmpeg','-v','error','-y','-i',str(path),'-map','0:v:0','-c:v','copy','-an','-sn','-movflags','+faststart',str(temporary)],check=True)
  temporary.replace(path)
 report[str(path.relative_to(MEDIA))]='video only'
for alias,source in aliases.items():
 shutil.copy2(MEDIA/source,MEDIA/alias);report[alias]='uncaptioned alias of '+source
(ROOT/'video/nyc-90s/web-media-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('Prepared',len(report),'silent, subtitle-free website videos')
