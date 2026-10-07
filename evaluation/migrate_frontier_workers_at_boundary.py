#!/usr/bin/env python3
"""Move selected rollout workers only between completed measured cells."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def append_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": now(), **event}, sort_keys=True) + "\n")


def proc_argv(pid: int) -> list[str]:
    try:
        return [
            item.decode(errors="replace")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if item
        ]
    except OSError:
        return []


def worker_pid(suite: Path, worker_id: int) -> int | None:
    suite_text = str(suite.resolve())
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        argv = proc_argv(int(entry.name))
        if (
            any(Path(item).name == "run_openai_frontier_matrix.py" for item in argv)
            and "--mode" in argv
            and argv[argv.index("--mode") + 1 : argv.index("--mode") + 2]
            == ["worker"]
            and "--suite-dir" in argv
            and suite_text in argv
            and "--worker-id" in argv
        ):
            try:
                if int(argv[argv.index("--worker-id") + 1]) == worker_id:
                    return int(entry.name)
            except (ValueError, IndexError):
                continue
    return None


def cell_status_path(suite: Path, status: dict) -> Path | None:
    try:
        condition = str(status["condition"])
        task_id = int(status["task_id"])
    except (KeyError, TypeError, ValueError):
        return None
    matches = list(
        (suite / "cells" / condition).glob(
            f"*/task_{task_id:03d}/status.json"
        )
    )
    return matches[0] if len(matches) == 1 else None


def cell_key(status: dict) -> tuple[int, str, int] | None:
    try:
        return (
            int(status["task_id"]),
            str(status["condition"]),
            int(status["attempt"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def owned_ue_groups(worker_dir: Path, old_gpu: int) -> set[int]:
    marker = str(worker_dir.resolve())
    adapter = f"-graphicsadapter={old_gpu}"
    groups: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        argv = proc_argv(pid)
        joined = " ".join(argv)
        if marker not in joined or adapter not in argv:
            continue
        if not any(
            Path(item).name in {"SimWorld.sh", "SimWorld"} for item in argv
        ):
            continue
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    return groups


def terminate_group(group: int, sig: signal.Signals) -> None:
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def migrate(
    suite: Path,
    worker_id: int,
    old_gpu: int,
    expected_gpu: int,
    events: Path,
) -> None:
    worker_dir = suite / "workers" / f"worker_{worker_id:02d}"
    maintenance = worker_dir / ".maintenance.lock"
    maintenance.write_text(
        json.dumps({"time": now(), "worker_id": worker_id}) + "\n",
        encoding="utf-8",
    )
    pid = worker_pid(suite, worker_id)
    if pid is None:
        append_event(events, {
            "event": "worker_already_absent", "worker_id": worker_id,
            "old_gpu": old_gpu, "expected_gpu": expected_gpu,
        })
        maintenance.unlink(missing_ok=True)
        return
    try:
        ue_groups = owned_ue_groups(worker_dir, old_gpu)
        append_event(events, {
            "event": "boundary_migration_started", "worker_id": worker_id,
            "pid": pid, "old_gpu": old_gpu, "expected_gpu": expected_gpu,
            "ue_groups": sorted(ue_groups),
        })
        terminate_group(pid, signal.SIGTERM)
        for group in ue_groups:
            terminate_group(group, signal.SIGTERM)
        deadline = time.monotonic() + 30.0
        while worker_pid(suite, worker_id) == pid and time.monotonic() < deadline:
            time.sleep(0.25)
        if worker_pid(suite, worker_id) == pid:
            terminate_group(pid, signal.SIGKILL)
        # The maintenance lock prevents the watchdog from starting a same-GPU
        # replacement while stale owned renderer groups are being removed.
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            stale = owned_ue_groups(worker_dir, old_gpu)
            if not stale:
                break
            for group in stale:
                terminate_group(group, signal.SIGTERM)
            time.sleep(0.5)
        for group in owned_ue_groups(worker_dir, old_gpu):
            terminate_group(group, signal.SIGKILL)
        append_event(events, {
            "event": "boundary_migration_finished", "worker_id": worker_id,
            "old_gpu": old_gpu, "expected_gpu": expected_gpu,
        })
    finally:
        maintenance.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument(
        "--targets", nargs="+", required=True,
        help="worker:expected_gpu pairs, for example 0:5 4:4 5:6",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--force-restart", action="store_true",
        help="restart at the boundary even when the GPU assignment is unchanged",
    )
    args = parser.parse_args()
    suite = args.suite.resolve()
    targets = {
        int(item.split(":", 1)[0]): int(item.split(":", 1)[1])
        for item in args.targets
    }
    events = suite.parent / "boundary_migration.jsonl"
    pending: dict[int, dict] = {}
    for worker_id, expected_gpu in targets.items():
        status = load_json(
            suite / "workers" / f"worker_{worker_id:02d}" / "status.json"
        )
        key = cell_key(status)
        if key is None:
            raise RuntimeError(f"worker {worker_id} has no active cell status")
        pending[worker_id] = {
            "key": key,
            "old_gpu": int(status["gpu"]),
            "expected_gpu": expected_gpu,
        }
    append_event(events, {
        "event": "boundary_watcher_started", "pid": os.getpid(),
        "targets": pending,
    })

    while pending:
        for worker_id in list(pending):
            item = pending[worker_id]
            status_path = (
                suite / "workers" / f"worker_{worker_id:02d}" / "status.json"
            )
            status = load_json(status_path)
            if (
                not args.force_restart
                and int(status.get("gpu", item["old_gpu"]))
                == item["expected_gpu"]
            ):
                append_event(events, {
                    "event": "worker_on_expected_gpu", "worker_id": worker_id,
                    "gpu": item["expected_gpu"],
                })
                pending.pop(worker_id)
                continue
            current_key = cell_key(status)
            if current_key != tuple(item["key"]):
                # The boundary was missed; preserve the new measured cell and
                # wait for its own completion instead of interrupting it.
                if current_key is not None:
                    item["key"] = current_key
                    item["old_gpu"] = int(status["gpu"])
                    append_event(events, {
                        "event": "advanced_cell_preserved",
                        "worker_id": worker_id, "cell": current_key,
                    })
                continue
            cell_path = cell_status_path(suite, status)
            cell_status = load_json(cell_path) if cell_path else {}
            if cell_status.get("state") not in {"completed", "needs_attention"}:
                continue
            # Re-read immediately before signalling to close the status race.
            if cell_key(load_json(status_path)) != tuple(item["key"]):
                continue
            migrate(
                suite, worker_id, int(item["old_gpu"]),
                int(item["expected_gpu"]), events,
            )
            pending.pop(worker_id)
        time.sleep(args.interval)
    append_event(events, {"event": "boundary_watcher_finished"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
