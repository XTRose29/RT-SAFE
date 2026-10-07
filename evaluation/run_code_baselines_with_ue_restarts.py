#!/usr/bin/env python3
"""Run code baselines with a fresh packaged UE process per condition."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UE_EXE = Path(os.environ.get("SIMWORLD_UE_LAUNCHER", str(REPO_ROOT / "runtime/SimWorld.sh")))


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def tcp_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_port(port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if tcp_open("127.0.0.1", port):
            return
        time.sleep(1)
    raise TimeoutError(f"UE did not open UnrealCV port {port} within {timeout_s}s")


def start_ue(
    log_path: Path, graphics_adapter: int, fpsmax: int, resx: int, resy: int
) -> subprocess.Popen:
    cmd = [
        str(UE_EXE),
        "-RenderoffScreen",
        "RT10",
        f"-graphicsadapter={graphics_adapter}",
        f"-ResX={resx}",
        f"-ResY={resy}",
        f"-FPSMAX={fpsmax}",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Starting UE: {' '.join(cmd)}")
    out = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd=UE_EXE.parent,
        stdout=out,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def stop_process(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    log(f"Stopping {name} pid={proc.pid}")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 20
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def cell_status(args: argparse.Namespace, baseline: str, env_mode: str) -> dict:
    status_path = (
        REPO_ROOT / "results" / args.suite_name / "runs" /
        f"{baseline}__{env_mode}__sync__deasy__code" / "round_01" /
        "run_status.json"
    )
    if not status_path.exists():
        return {}
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unreadable", "error": str(exc)}
    result_path = status.get("result_path")
    if status.get("status") == "completed" and result_path:
        try:
            result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        except Exception as exc:
            status["status"] = "invalid_result"
            status["error"] = f"Could not read result JSON: {exc}"
            return status
        if (
            not result.get("success") and
            int(result.get("decision_count") or 0) <= 1 and
            int(result.get("final_step") or 0) == 0
        ):
            status["status"] = "invalid_result"
            status["error"] = "Run ended before completing the first action; treating as UE/socket crash."
    return status


def run_cell_once(args: argparse.Namespace, baseline: str, env_mode: str, log_dir: Path, attempt: int) -> int:
    suffix = f"{baseline}_{env_mode}_attempt{attempt:02d}"
    ue_log = log_dir / f"ue_{suffix}.log"
    runner_log = log_dir / f"runner_{suffix}.log"
    ue_proc = None
    try:
        ue_proc = start_ue(
            ue_log, args.graphics_adapter, args.fpsmax, args.resx, args.resy
        )
        wait_for_port(args.ue_port, args.ue_ready_timeout_s)
        log(f"UE ready on port {args.ue_port}; log={ue_log}")
        cmd = [
            sys.executable,
            "evaluation/run_code_baselines.py",
            "--suite-name",
            args.suite_name,
            "--baselines",
            baseline,
            "--env-modes",
            env_mode,
            "--time-modes",
            "sync",
            "--difficulties",
            "easy",
            "--task-file",
            args.task_file,
            "--task-index",
            str(args.task_index),
            "--seed",
            str(args.seed),
            "--max-steps",
            str(args.max_steps),
            "--internal-latency-seconds",
            str(args.internal_latency_seconds),
            "--max-rounds",
            "1",
            "--ue-port",
            str(args.ue_port),
            "--inter-run-sleep",
            "1",
        ]
        if args.load_all_unsafe_triggers:
            cmd.append("--load-all-unsafe-triggers")
        if args.uniform_greedy_movement:
            cmd.append("--uniform-greedy-movement")
        log(f"Running {baseline}/{env_mode} attempt {attempt}; log={runner_log}")
        with runner_log.open("w", encoding="utf-8") as out:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=args.runner_timeout_s,
            )
        log(f"{baseline}/{env_mode} exited with code {proc.returncode}")
        return proc.returncode
    except subprocess.TimeoutExpired:
        log(f"{baseline}/{env_mode} timed out after {args.runner_timeout_s}s")
        return 124
    finally:
        stop_process(ue_proc, "UE")
        time.sleep(args.after_ue_sleep_s)


def run_cell(args: argparse.Namespace, baseline: str, env_mode: str, log_dir: Path) -> int:
    last_return_code = 1
    for attempt in range(1, args.max_attempts + 1):
        last_return_code = run_cell_once(args, baseline, env_mode, log_dir, attempt)
        status = cell_status(args, baseline, env_mode)
        if status.get("status") == "completed" and status.get("result_path"):
            log(f"{baseline}/{env_mode} completed on attempt {attempt}")
            return 0
        log(f"{baseline}/{env_mode} did not complete on attempt {attempt}: {status}")
    return last_return_code or 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-name", default="code_baselines_greedy_safety_static_dynamic_easy_20260704_run")
    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=243)
    parser.add_argument("--internal-latency-seconds", type=float, default=0.0)
    parser.add_argument("--ue-port", type=int, default=9007)
    parser.add_argument("--graphics-adapter", type=int, default=2)
    parser.add_argument("--fpsmax", type=int, default=15)
    parser.add_argument("--resx", type=int, default=900)
    parser.add_argument("--resy", type=int, default=800)
    parser.add_argument("--load-all-unsafe-triggers", action="store_true")
    parser.add_argument("--uniform-greedy-movement", action="store_true")
    parser.add_argument("--ue-ready-timeout-s", type=float, default=90)
    parser.add_argument("--runner-timeout-s", type=float, default=1800)
    parser.add_argument("--after-ue-sleep-s", type=float, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--baselines", nargs="+", default=["greedy", "safety"], choices=["greedy", "safety"])
    parser.add_argument("--env-modes", nargs="+", default=["realtime", "static"], choices=["realtime", "static"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_dir = REPO_ROOT / "results" / args.suite_name / "detached_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for baseline in args.baselines:
        for env_mode in args.env_modes:
            failures += int(run_cell(args, baseline, env_mode, log_dir) != 0)
    log(f"done; failures={failures}; summary={REPO_ROOT / 'results' / args.suite_name / 'summary.jsonl'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
