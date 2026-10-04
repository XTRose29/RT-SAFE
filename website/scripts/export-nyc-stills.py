"""Export the inspected native MRQ camera views and project share image."""
from pathlib import Path
import importlib.util
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'public/media/nyc';OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('film',ROOT/'scripts/render-film-nyc.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f);v=f.v
shots={'camera-skyline':'quality-v2/city_detail/city_detail_0001.png',
       'camera-park':'survey-v1/park/park_0001.png',
       'camera-high_follow':'quality-v2/follow_raster/follow_raster_0001.png',
       'camera-intersection':'quality-v2/crossing/crossing_0001.png'}
for name,source in shots.items():
 im=Image.open(ROOT/'nyc/previews'/source).convert('RGB');im.thumbnail((1920,1080));im.save(OUT/(name+'.webp'),quality=93)
for name,source in [('hero','camera-park'),('high_follow','camera-high_follow')]:
 (OUT/(name+'.webp')).write_bytes((OUT/(source+'.webp')).read_bytes())
poster=f.shade(Image.open(OUT/'hero.webp').convert('RGB'),125,200)
v.text(poster,(95,178),'RT-Safe',114,v.WHITE,600)
v.text(poster,(102,374),'Benchmarking Agent Safety',62,v.WHITE,500)
v.text(poster,(102,467),'in Real-Time Embodied Environment',62,v.WHITE,500)
v.text(poster,(104,693),'The world does not pause while an agent thinks.',32,v.PALE,400)
v.text(poster,(105,913),'NYC  /  NATIVE UNREAL ENGINE RENDERING  /  90 SECONDS',21,v.PALE,500)
poster.save(OUT/'film-poster.webp',quality=94)
og=Image.open(OUT/'hero.webp').convert('RGB').resize((1200,675),Image.Resampling.LANCZOS).crop((0,22,1200,652))
og=f.shade(og,135,200)
v.text(og,(63,72),'RT-Safe',92,v.WHITE,600)
v.text(og,(70,254),'The world does not pause',46,v.WHITE,500)
v.text(og,(70,317),'while an agent thinks.',46,v.WHITE,500)
v.text(og,(73,523),'Benchmarking safety in real-time embodied environments',23,v.PALE,400)
og.save(ROOT/'public/og.png',optimize=True)
print('Exported four camera views, film poster, and NYC share image')
