"""Regression: the upstream launcher exits before its engine child."""
import json
import socket
import subprocess
import sys
import time
import platform
from pathlib import Path

import pytest

from tools.runtime_process import stop_process_group


@pytest.mark.parametrize("zombie_leader", [False, True])
def test_cleanup_stops_child_even_after_launcher_exits(tmp_path, zombie_leader):
    if zombie_leader and platform.machine() != "x86_64":
        pytest.skip("Linux x86-64 SYS_exit fixture")
    ready = tmp_path / "ready.json"
    child = "\n".join([
        "import json, os, signal, socket, time",
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
        "s = socket.socket()",
        "s.bind(('127.0.0.1', 0))",
        "s.listen()",
        f"open({str(ready) + '.tmp'!r}, 'w').write(json.dumps({{'port': s.getsockname()[1], 'pid': os.getpid()}}))",
        f"os.replace({str(ready) + '.tmp'!r}, {str(ready)!r})",
        "time.sleep(60)",
    ])
    if zombie_leader:
        # Exit only the main OS thread, leaving a worker holding the socket.
        # /proc/<pid>/stat now says Z even though the process is not finished.
        child = ("import ctypes, threading, time, signal\n"
                 "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                 "def worker():\n" + "\n".join("    " + line for line in child.splitlines()
                     if "signal.signal" not in line) + "\n"
                 "threading.Thread(target=worker).start()\n"
                 "time.sleep(0.1)\nctypes.CDLL(None).syscall(60, 0)\n")
    parent = f"import subprocess; subprocess.Popen([{sys.executable!r}, '-c', {child!r}])"
    process = subprocess.Popen([sys.executable, "-c", parent], start_new_session=True)
    try:
        process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        child_info = json.loads(ready.read_text())
        port = child_info["port"]
        if zombie_leader:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = Path(f"/proc/{child_info['pid']}/stat").read_text().rsplit(")", 1)[1].split()[0]
                if state == "Z":
                    break
                time.sleep(0.02)
            assert state == "Z"
        stop_process_group(process, grace_seconds=0.1)
        # A live descendant would still own the listener, even though the
        # launcher already exited successfully before cleanup was called.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    finally:
        stop_process_group(process, grace_seconds=0.1)
