"""Check exported media duration, frame count, decoding, and caption intervals."""
from pathlib import Path
import hashlib, json, re, subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'video/revision-90s'
report = {}
for name in ['rt-safe-demo-90s.mp4', 'rt-safe-demo-90s-captioned.mp4']:
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
(OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
print('Passed: both 90-second / 2700-frame exports decode fully; captions do not overlap; voice fits all scenes.')
