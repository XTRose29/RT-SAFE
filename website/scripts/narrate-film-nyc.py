"""Build the NYC narration with the recorded Astra/Sol case and behavior profiles."""
import importlib.util, asyncio, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('narration90',ROOT/'scripts/narrate-film-90s.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
n.OUT=ROOT/'video/nyc-90s';n.AUDIO=n.OUT/'audio';n.AUDIO.mkdir(parents=True,exist_ok=True)
n.SCENES=[(name,duration,'ASTRA / SOL: SAME RECORDED ROUTE' if name=='examples' else chapter,
    'On the same recorded route, Astra finishes in twelve decisions with one collision. Sol takes thirty-three decisions with seventeen collisions. This NYC reconstruction uses recorded agent positions and simulation timing.' if name=='examples' else script)
    for name,duration,chapter,script in n.SCENES]
behavior_script='Sonnet has the fewest collisions, but safe success remains rare. Inkling favors longer moves. Fable turns and waits more than Astra.'
n.SCENES=[(name,duration,chapter,behavior_script if name=='behavior' else script) for name,duration,chapter,script in n.SCENES]
cache=n.AUDIO/'behavior.script.txt'
if not cache.exists() or cache.read_text()!=behavior_script:
    for suffix in ('.mp3','.words.json'):(n.AUDIO/('behavior'+suffix)).unlink(missing_ok=True)
for name,_,_,_ in n.SCENES:
    if name in ('examples','behavior'):continue
    for suffix in ('.mp3','.words.json'):
        shutil.copy2(ROOT/'video/revision-90s/audio'/f'{name}{suffix}',n.AUDIO/f'{name}{suffix}')
asyncio.run(n.main())
cache.write_text(behavior_script)

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
