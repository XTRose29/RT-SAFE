"""Build the NYC narration with the recorded Astra/Sol case and behavior profiles."""
import importlib.util, asyncio, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('narration90',ROOT/'scripts/narrate-film-90s.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
n.OUT=ROOT/'video/nyc-90s';n.AUDIO=n.OUT/'audio';n.AUDIO.mkdir(parents=True,exist_ok=True)
# Changed scripts invalidate their own cached voice and word timings.
overrides={
 'examples':(16,'HOW DO FRONTIER AGENTS PERFORM?', 'How do frontier agents perform? On the same recorded route, Astra makes twelve decisions with one collision. Sol makes thirty-three decisions with seventeen collisions. Recorded behavior, reconstructed in NYC.'),
 'behavior':(14,'EIGHT MODELS: PERFORMANCE AND BEHAVIOR', 'Sonnet records the fewest collisions; Grok, the most. Inkling favors longer moves. Sol turns and waits most. Response time and action choices both shape behavior.'),
 'learning':(6,'AN ENVIRONMENT FOR LEARNING', 'R T Safe supports reinforcement learning. Reward design shapes safety.'),
}
n.SCENES=[(name,*overrides[name]) if name in overrides else (name,duration,chapter,script) for name,duration,chapter,script in n.SCENES]
for name,_,_,script in n.SCENES:
 cache=n.AUDIO/(name+'.script.txt')
 if name in overrides:
  if not cache.exists() or cache.read_text()!=script:
   for suffix in ('.mp3','.words.json'):(n.AUDIO/(name+suffix)).unlink(missing_ok=True)
 else:
  for suffix in ('.mp3','.words.json'):
   shutil.copy2(ROOT/'video/revision-90s/audio'/f'{name}{suffix}',n.AUDIO/f'{name}{suffix}')
asyncio.run(n.main())
for name,_,_,script in n.SCENES:(n.AUDIO/(name+'.script.txt')).write_text(script)

# Keep short sentence endings with their preceding phrase for readable captions.
def seconds(stamp):
    h,m,s=stamp.split(':');return int(h)*3600+int(m)*60+float(s)
cues=[]
for block in (n.OUT/'captions.vtt').read_text().split('\n\n'):
    if '-->' not in block:continue
    lines=block.strip().splitlines();start,end=lines[0].split(' --> ');caption=' '.join(lines[1:])
    if (cues and len(caption.split())<=3 and not cues[-1][2].endswith(('.', '?', '!'))
        and seconds(start)-seconds(cues[-1][1])<.2 and len(cues[-1][2])+len(caption)<105):
        cues[-1][1]=end;cues[-1][2]+=' '+caption
    else:cues.append([start,end,caption])
(n.OUT/'captions.vtt').write_text('WEBVTT\n\n'+'\n\n'.join(f'{a} --> {b}\n{c}' for a,b,c in cues)+'\n')
(n.OUT/'captions.srt').write_text('\n\n'.join(f'{i+1}\n{a.replace(".",",")} --> {b.replace(".",",")}\n{c}' for i,(a,b,c) in enumerate(cues))+'\n')
