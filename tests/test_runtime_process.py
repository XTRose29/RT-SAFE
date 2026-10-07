"""Regression: the upstream launcher exits before its engine child."""
import json
import socket
import subprocess
import sys
import time

from tools.runtime_process import stop_process_group


def test_cleanup_stops_child_even_after_launcher_exits(tmp_path):
    ready = tmp_path / "ready.json"
    child = "\n".join([
        "import json, os, signal, socket, time",
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
        "s = socket.socket()",
        "s.bind(('127.0.0.1', 0))",
        "s.listen()",
        f"open({str(ready)!r}, 'w').write(json.dumps({{'port': s.getsockname()[1]}}))",
        "time.sleep(60)",
    ])
    parent = f"import subprocess; subprocess.Popen([{sys.executable!r}, '-c', {child!r}])"
    process = subprocess.Popen([sys.executable, "-c", parent], start_new_session=True)
    try:
        process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        port = json.loads(ready.read_text())["port"]
        stop_process_group(process, grace_seconds=0.1)
        # A live descendant would still own the listener, even though the
        # launcher already exited successfully before cleanup was called.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    finally:
        stop_process_group(process, grace_seconds=0.1)
