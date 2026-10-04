"""Launch only this isolated MRQ session; record its PID for precise cleanup."""
import os,ctypes,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
# Permit same-user diagnostics on this rendering process (Linux Yama).
if os.environ.get('RTSAFE_DEBUG_PTRACE')=='1':
    ctypes.CDLL(None).prctl(0x59616d61,ctypes.c_ulong(-1).value,0,0,0)
(root/'runtime/editor.pid').write_text(str(os.getpid()))
binary=str(root/'ue/Binaries/Linux/SimWorldEditor')
args=[binary,str(root/'ue/SimWorld.uproject'),'/Game/_batch/madison_square_park_1km',
 '-sp-config-file='+str(root/'runtime/spear.yaml'),'-RenderOffScreen','-norhithread','-corelimit=8','-graphicsadapter='+os.environ.get('RTSAFE_GPU','2'),
 '-nosplash','-nop4','-nosound','-unattended','-nozenautolaunch','-NoLiveCoding','-NoHotReloadFromIDE',
 '-stdout','-FullStdOutLogOutput','-abslog='+str(root/f'runtime/editor-{os.getpid()}.log'),
 '-DDC=NoZenLocalFallback','-LocalDataCachePath='+os.environ.get('RTSAFE_NYC_DDC',str(Path(binary).resolve().parents[2]/'DerivedDataCache')),
 '-NoDDCCleanup','-AssetGatherAll=false','-AssetGatherSync=true','-NoDependsGathering',
 '-ExecCmds=r.Nanite 0,r.Shadow.Virtual.Enable 0,r.DynamicGlobalIlluminationMethod 0,r.ReflectionMethod 0,py '+str(root/'scripts/editor_bridge.py')]
os.execv(binary,args)
