#!/usr/bin/env python3
"""Keep six resume-safe GPT-6 Astra campaign workers alive until completion."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "evaluation/run_openai_frontier_matrix.py"
MODEL = "gpt-6-astra"
EFFORTS = "low,high"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": now(), **event}, sort_keys=True) + "\n")


def live_worker_ids(suite: Path) -> set[int]:
    result: set[int] = set()
    suite_text = str(suite)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        decoded = [item.decode(errors="replace") for item in args if item]
        if (
            "--mode" not in decoded
            or "worker" not in decoded
            or "--suite-dir" not in decoded
            or suite_text not in decoded
            or "--worker-id" not in decoded
        ):
            continue
        try:
            result.add(int(decoded[decoded.index("--worker-id") + 1]))
        except (ValueError, IndexError):
            continue
    return result


def completion(suite: Path) -> tuple[int, int]:
    manifest = json.loads((suite / "suite_manifest.json").read_text())
    expected = int(manifest["rollout_count"])
    completed = 0
    for path in (suite / "cells").glob("**/status.json"):
        try:
            if json.loads(path.read_text()).get("state") == "completed":
                completed += 1
        except (OSError, json.JSONDecodeError):
            continue
    return completed, expected


def worker_command(args: argparse.Namespace, worker_id: int) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--mode", "worker",
        "--suite-dir", str(args.suite_dir),
        "--worker-id", str(worker_id),
        "--worker-count", str(args.workers),
        "--gpus", *map(str, args.gpus),
        "--ports", *map(str, args.ports),
        "--key-file", str(args.key_file),
        "--ue-launcher", str(args.ue_launcher),
        "--max-attempts", "3",
        "--request-timeout", "300",
        "--cell-timeout", "14400",
        "--ue-ready-timeout", "180",
        "--ue-settle-seconds", "30",
        "--ue-start-stagger-seconds", "30",
        "--process-shutdown-timeout", "20",
        "--resume",
        "--record-png-compress-level", "0",
        "--record-images-first-steps", "0",
        "--record-images-last-steps", "0",
        "--record-output-images",
        "--dynamic-queue",
        "--ue-no-rhi-thread",
        "--ue-port-stride", str(args.port_stride),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gpus", nargs="+", type=int, default=[7, 6, 5, 4, 2, 3])
    parser.add_argument(
        "--ports", nargs="+", type=int,
        default=[12012, 12013, 12014, 12015, 12016, 12017],
    )
    parser.add_argument("--port-stride", type=int, default=100)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--key-file", type=Path, default=Path(".secrets/openai_api.txt"))
    parser.add_argument(
        "--ue-launcher", type=Path,
        default=Path(os.environ.get("SIMWORLD_UE_LAUNCHER", str(REPO_ROOT / "runtime/SimWorld.sh"))),
    )
    args = parser.parse_args()
    args.suite_dir = args.suite_dir.resolve()
    if len(args.gpus) < args.workers or len(args.ports) < args.workers:
        raise ValueError("one GPU and base port are required per worker")

    events = args.suite_dir / "watchdog.jsonl"
    output = args.suite_dir / "watchdog_workers.log"
    environment = os.environ.copy()
    environment.update({
        "SIMWORLD_CAMPAIGN_MODEL": MODEL,
        "SIMWORLD_CAMPAIGN_PROVIDER": "openai",
        "SIMWORLD_CAMPAIGN_API_MODE": "responses",
        "SIMWORLD_CAMPAIGN_INPUT_PRICE": "10",
        "SIMWORLD_CAMPAIGN_OUTPUT_PRICE": "50",
        "SIMWORLD_CAMPAIGN_EFFORTS": EFFORTS,
    })
    append_event(events, {"event": "watchdog_started", "pid": os.getpid()})
    children: list[subprocess.Popen] = []
    with output.open("ab", buffering=0) as stream:
        while True:
            completed, expected = completion(args.suite_dir)
            if completed >= expected:
                append_event(events, {
                    "event": "suite_complete", "completed": completed,
                    "expected": expected,
                })
                return 0
            live = live_worker_ids(args.suite_dir)
            for worker_id in range(args.workers):
                if worker_id in live:
                    continue
                process = subprocess.Popen(
                    worker_command(args, worker_id),
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                children.append(process)
                append_event(events, {
                    "event": "worker_started", "worker_id": worker_id,
                    "pid": process.pid, "completed": completed,
                    "expected": expected,
                })
                # Preserve the benchmark's global UE startup spacing even if
                # several legacy workers disappear in the same check.
                time.sleep(args.interval)
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
