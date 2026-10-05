"""Executed by the owned Unreal editor, after startup has loaded its world."""

import json
import os
from pathlib import Path
import sys
import time
import traceback
import unreal

_started = time.monotonic()
_handle = None


def _initialize(delta):
    global _handle
    if time.monotonic() - _started < 10:
        return
    unreal.unregister_slate_post_tick_callback(_handle)
    destination = Path(os.environ["RTSAFE_BOOTSTRAP_STATUS"])
    try:
        sys.path.insert(0, os.environ["RTSAFE_SPEAR_PYTHON"])
        import spear

        result = {
            "ok": True,
            "spear_version": spear.__version__,
            "engine_version": unreal.SystemLibrary.get_engine_version(),
            "pid": os.getpid(),
        }
    except Exception:
        result = {"ok": False, "error": traceback.format_exc(), "pid": os.getpid()}
    destination.write_text(json.dumps(result, indent=2))


_handle = unreal.register_slate_post_tick_callback(_initialize)
