"""Create an isolated local rendering project from a separately provisioned NYC runtime."""
from pathlib import Path
import argparse,json,os,re,secrets,shutil
ap=argparse.ArgumentParser()
ap.add_argument('--source',type=Path,required=True,help='Existing Madison Square Park Unreal project directory')
ap.add_argument('--engine',type=Path,help='Compatible UnrealEngine directory; inferred from the source project when omitted')
ap.add_argument('--rpc-port',type=int,default=47834)
args=ap.parse_args();source=args.source.resolve();root=Path(__file__).resolve().parents[1];dest=root/'ue'
map_name='madison_square_park_1km'
for relative in ['SimWorld.uproject',f'Content/_batch/{map_name}.umap','Binaries/Linux/SimWorldEditor']:
 if not (source/relative).exists():raise SystemExit('Required runtime file missing: '+str(source/relative))
engine=(args.engine or source.parent/'UnrealEngine-5.8.0-preview-1').resolve()
if not (engine/'Engine/Binaries/Linux').exists():raise SystemExit('Provide --engine for the compatible Unreal runtime')
if dest.resolve()==source:raise SystemExit('The working project must be separate from the source project')
if dest.exists() and not (root/'project-source.json').exists():raise SystemExit('Working ue/ directory already exists; refusing to overwrite an untracked project')
def link(target,path):
 if path.is_symlink():
  if path.resolve()!=target.resolve():raise SystemExit('Conflicting existing link: '+str(path))
 elif path.exists():raise SystemExit('Conflicting existing path: '+str(path))
 else:path.symlink_to(target,target_is_directory=target.is_dir())
def copy_once(src,target):
 if not target.exists():shutil.copy2(src,target)
dest.mkdir(parents=True,exist_ok=True)
copy_once(source/'SimWorld.uproject',dest/'SimWorld.uproject')
if not (dest/'Config').exists():shutil.copytree(source/'Config',dest/'Config')
for name in ['Binaries','Plugins','Source']:
 if (source/name).exists():link(source/name,dest/name)
content=dest/'Content';content.mkdir(exist_ok=True);(content/'_batch').mkdir(exist_ok=True);(content/'RTSafeNYC').mkdir(exist_ok=True)
for item in (source/'Content').iterdir():
 if item.name not in ('_batch','RTSafeNYC'):link(item,content/item.name)
for suffix in ['.umap','_BuiltData.uasset']:
 item=source/'Content/_batch'/(map_name+suffix)
 if item.exists():copy_once(item,content/'_batch'/item.name)
link(engine,root/'UnrealEngine-5.8.0-preview-1')
config=dest/'Config/DefaultEngine.ini';text=config.read_text()
if 'r.AllowOcclusionQueries=0' not in text:config.write_text(text+'\n[SystemSettings]\nr.AllowOcclusionQueries=0\n')
(root/'runtime').mkdir(exist_ok=True)
yaml=(root/'config/spear-template.yaml').read_text()
yaml=re.sub(r'RPC_SERVER_PORT: \d+',f'RPC_SERVER_PORT: {args.rpc_port}',yaml)
yaml=re.sub(r'SHARED_MEMORY_INITIAL_UNIQUE_ID: \d+',f'SHARED_MEMORY_INITIAL_UNIQUE_ID: {secrets.randbelow(10000000)+1000000}',yaml)
(root/'runtime/spear.yaml').write_text(yaml)
(root/'project-source.json').write_text(json.dumps({'source_project':str(source),'engine':str(engine),'map':map_name,'rpc_port':args.rpc_port},indent=2)+'\n')
print('Isolated project prepared at',dest)
