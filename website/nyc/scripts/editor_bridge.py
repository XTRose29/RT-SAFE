"""Keep a warmed editor available for this project's local render commands."""
import builtins, json, os, time, traceback
import unreal
import faulthandler

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
COMMAND = ROOT + '/runtime/command.json'
STATUS = ROOT + '/runtime/editor-status.json'
def write_status(payload):
    with open(STATUS+'.tmp','w') as f:json.dump(payload,f)
    os.replace(STATUS+'.tmp',STATUS)
if not getattr(builtins, '_rtsafe_nyc_bridge', None):
    builtins._rtsafe_nyc_trace=open(ROOT+'/runtime/python-stacks.log','w')
    faulthandler.enable(file=builtins._rtsafe_nyc_trace)
    faulthandler.dump_traceback_later(180,repeat=True,file=builtins._rtsafe_nyc_trace)
    state = {'last': None, 'last_poll': 0, 'namespaces': [], 'started':time.monotonic()}
    builtins._rtsafe_nyc_bridge = state
    def tick(delta):
        now = time.monotonic()
        if now - state['last_poll'] < 1:
            return
        state['last_poll'] = now
        try:
            render_state=getattr(builtins,'_rtsafe_nyc_render_state',{})
            if render_state.get('active') is not None:
                write_status({'ready':False,'rendering':True,'pid':os.getpid(),'updated':time.time(),'last_command':state['last']})
                return
            world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
            if world is None:return
            status = {'ready': True, 'pid': os.getpid(), 'world': world.get_path_name(), 'updated': time.time(), 'last_command': state['last']}
            write_status(status)
            if now-state['started']<20:return
            if not os.path.exists(COMMAND): return
            with open(COMMAND) as f: cmd = json.load(f)
            if cmd['id'] == state['last']: return
            state['last'] = cmd['id']
            os.environ.update({str(k): str(v) for k,v in cmd.get('env', {}).items()})
            ns = {'__name__': '__main__', '__file__': cmd['script']}
            state['namespaces'].append(ns)
            unreal.log_warning('RTSAFE_NYC executing ' + cmd['script'])
            with open(cmd['script']) as f: code = f.read()
            exec(compile(code, cmd['script'], 'exec'), ns)
        except Exception:
            error = traceback.format_exc()
            unreal.log_error('RTSAFE_NYC ' + error)
            with open(ROOT + '/runtime/command-error.txt','w') as f: f.write(error)
    state['handle'] = unreal.register_slate_post_tick_callback(tick)
    unreal.log_warning('RTSAFE_NYC editor bridge ready')
