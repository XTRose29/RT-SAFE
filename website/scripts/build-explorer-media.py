"""Package native square captures as browser cubemap faces and viewpoint metadata."""
from pathlib import Path
import json
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'nyc/renders/explorer'
out=ROOT/'public/media/nyc/explorer';out.mkdir(parents=True,exist_ok=True)
metadata=json.loads((source/'scene.json').read_text())
labels={'approach':'The approach','hazards':'Along the path','crossing':'The crossing','park':'In the park'}
for loc in metadata['locations']:
 loc['label']=labels[loc['id']]
 for face in metadata['faces']:
  path=source/loc['id']/f'{loc["id"]}_{face["frame"]:04d}.png'
  im=Image.open(path).convert('RGB');assert im.size==(1536,1536)
  im.save(out/f'{loc["id"]}-{face["id"]}.webp',quality=91,method=6)
(ROOT/'public/data/nyc/explorer.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Packaged',len(metadata['locations']),'native viewpoints')
