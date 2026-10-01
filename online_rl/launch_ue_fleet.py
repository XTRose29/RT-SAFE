#!/usr/bin/env python3
"""Launch a local fleet of packaged SimWorld UnrealCV servers for VAGEN."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


def build_command(
    launcher: str | Path,
    *,
    map_name: str,
    gpu: int,
    port: int,
    width: int,
    height: int,
    fps: int,
    internal_log: str | Path,
) -> list[str]:
    return [
        str(Path(launcher).expanduser().resolve()),
        "-RenderoffScreen",
        map_name,
        f"-graphicsadapter={gpu}",
        "-cvport",
        str(port),
        f"-ResX={width}",
        f"-ResY={height}",
        f"-FPSMAX={fps}",
        "-noraytracing",
        f"-abslog={Path(internal_log).expanduser().resolve()}",
        "-norhithread",
    ]


def tcp_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stop_processes(processes: list[subprocess.Popen], grace_s: float = 20.0) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and any(p.poll() is None for p in processes):
        time.sleep(0.1)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--map", dest="map_name", default="RT10")
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--ports", nargs="+", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.gpus) != len(args.ports):
        parser.error("--gpus and --ports must contain the same number of values")
    if len(set(args.ports)) != len(args.ports):
        parser.error("--ports must be unique")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    launcher = Path(args.launcher).expanduser().resolve()
    if not launcher.is_file():
        raise FileNotFoundError(f"UE launcher does not exist: {launcher}")
    output = Path(args.output).expanduser().resolve()
    logs = Path(args.logs).expanduser().resolve()
    rows = []
    commands = []
    for index, (gpu, port) in enumerate(zip(args.gpus, args.ports)):
        instance_id = f"{args.map_name.lower()}-{index:02d}"
        internal_log = logs / f"{instance_id}_internal.log"
        command = build_command(
            launcher,
            map_name=args.map_name,
            gpu=gpu,
            port=port,
            width=args.width,
            height=args.height,
            fps=args.fps,
            internal_log=internal_log,
        )
        commands.append(command)
        rows.append(
            {
                "id": instance_id,
                "host": args.host,
                "port": port,
                "map_name": args.map_name,
                "gpu": gpu,
            }
        )
    if args.dry_run:
        print(json.dumps({"instances": rows, "commands": commands}, indent=2))
        return 0

    occupied = [port for port in args.ports if tcp_open(args.host, port)]
    if occupied:
        raise RuntimeError(f"refusing to use occupied UnrealCV ports: {occupied}")
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen] = []
    streams = []
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        for row, command in zip(rows, commands):
            console_log = logs / f"{row['id']}.log"
            stream = console_log.open("ab", buffering=0)
            streams.append(stream)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=launcher.parent,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        started = time.monotonic()
        pending = set(range(len(processes)))
        while pending and not stopping:
            for index in tuple(pending):
                process = processes[index]
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{rows[index]['id']} exited during startup; "
                        f"inspect {logs / (rows[index]['id'] + '.log')}"
                    )
                if tcp_open(args.host, rows[index]["port"]):
                    pending.remove(index)
            if time.monotonic() - started >= args.ready_timeout:
                raise TimeoutError(
                    f"UE fleet did not become ready within {args.ready_timeout:g}s; "
                    f"pending={[rows[index]['id'] for index in sorted(pending)]}"
                )
            if pending:
                time.sleep(1.0)
        if stopping:
            return 130
        write_json_atomic(
            output,
            {
                "status": "ready",
                "created_at_unix": time.time(),
                "instances": rows,
            },
        )
        print(f"UE fleet ready: {len(rows)} instances; endpoints={output}", flush=True)
        while not stopping:
            exited = [
                rows[index]["id"]
                for index, process in enumerate(processes)
                if process.poll() is not None
            ]
            if exited:
                raise RuntimeError(f"UE fleet member(s) exited: {exited}")
            time.sleep(1.0)
        return 0
    finally:
        stop_processes(processes)
        for stream in streams:
            stream.close()
        if output.exists():
            write_json_atomic(
                output,
                {"status": "stopped", "stopped_at_unix": time.time(), "instances": []},
            )


if __name__ == "__main__":
    raise SystemExit(main())
