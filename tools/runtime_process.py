"""Lifecycle helpers for engine launchers started in their own process group."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time


def _running_group(pgid: int) -> bool:
    # A process leader can be a zombie while its worker threads still hold
    # sockets/GPU resources. Check those threads before treating it as dead.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid:
                continue
            if fields[0] not in {"Z", "X"}:
                return True
            for task in (entry / "task").iterdir():
                try:
                    state = (task / "stat").read_text().rsplit(")", 1)[1].split()[0]
                    if state not in {"Z", "X"}:
                        return True
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return False


def stop_process_group(process: subprocess.Popen, *, grace_seconds: float = 30) -> None:
    """Stop only a process launched with ``start_new_session=True`` and its children.

    Waiting for just the launcher is insufficient: the official shell wrapper
    can exit while its engine still owns the UnrealCV port.
    """
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    deadline = time.monotonic() + grace_seconds
    while _running_group(pgid) and time.monotonic() < deadline:
        process.poll()  # Reap the direct child, even while descendants remain.
        time.sleep(0.05)
    if _running_group(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5
        while _running_group(pgid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.05)
        if _running_group(pgid):
            raise RuntimeError(f"Owned process group {pgid} did not stop")
    process.wait(timeout=5)
