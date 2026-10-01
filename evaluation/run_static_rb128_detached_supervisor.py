#!/usr/bin/env python3
"""Detached supervisor for the static Qwen3-VL rb128 run.

Runs one new round at a time with a fresh UE process, while reusing the local
Qwen thinking server on port 30001. Intended to be launched inside tmux so the
experiment keeps running after the remote session is closed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(".venv/bin/python")
RUNNER = REPO_ROOT / "evaluation" / "run_qwen3vl8b_setting_sweep.py"
UE_EXE = Path(os.environ.get("SIMWORLD_UE_LAUNCHER", str(REPO_ROOT / "runtime/SimWorld.sh")))

SUITE_NAME = "qwen3vl_thinking_rb128_shortprompt_static_easy_20260704"
SETTING_ID = "reasoning__latency_aware__static__sync__deasy__video_history__rb128__think_shorter"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def tcp_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_qwen(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    models_url = url.rstrip("/") + "/models"
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=5) as response:
                if response.status == 200:
                    log(f"Qwen server ready: {models_url}")
                    return
                last_error = f"HTTP {response.status}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"Qwen server did not become ready at {models_url}: {last_error}")


def start_ue(log_path: Path, gpu: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    cmd = [
        str(UE_EXE),
        "-RenderoffScreen",
        "RT10",
        f"-graphicsadapter={gpu}",
        "-ResX=1280",
        "-ResY=720",
        "-FPSMAX=15",
    ]
    log(f"Starting UE: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def wait_for_ue(log_path: Path, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    markers = ("Attach a UnrealcvSensor to the pawn", "Enabling input")
    while time.time() < deadline:
        text = ""
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
        if all(marker in text for marker in markers) and tcp_open("127.0.0.1", port):
            log(f"UE ready on port {port}; log={log_path}")
            return
        time.sleep(5)
    raise TimeoutError(f"UE did not become ready on port {port}; see {log_path}")


def stop_process_group(proc: subprocess.Popen | None, grace_s: float = 20.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            continue


def result_path_for_round(suite_dir: Path, round_id: int) -> Path:
    return (
        suite_dir
        / "runs"
        / SETTING_ID
        / f"round_{round_id:02d}"
        / f"task_1_{SETTING_ID}__round{round_id:02d}.json"
    )


def run_status_path_for_round(suite_dir: Path, round_id: int) -> Path:
    return suite_dir / "runs" / SETTING_ID / f"round_{round_id:02d}" / "run_status.json"


def count_successes(suite_dir: Path) -> int:
    successes = 0
    setting_dir = suite_dir / "runs" / SETTING_ID
    for result_path in setting_dir.glob("round_*/task_*.json"):
        data = load_json(result_path)
        if data.get("success") is True:
            successes += 1
    return successes


def next_unfinished_round(suite_dir: Path, max_rounds: int) -> int | None:
    for round_id in range(1, max_rounds + 1):
        if result_path_for_round(suite_dir, round_id).exists():
            continue
        status = load_json(run_status_path_for_round(suite_dir, round_id))
        if status.get("status") in {"completed", "error"}:
            continue
        return round_id
    return None


def mark_round_error(suite_dir: Path, round_id: int, error: str) -> None:
    condition_id = f"{SETTING_ID}__round{round_id:02d}"
    record = {
        "condition_id": condition_id,
        "setting_id": SETTING_ID,
        "round": round_id,
        "status": "error",
        "finished_at": now(),
        "success": False,
        "failed": None,
        "error": error,
    }
    write_json(run_status_path_for_round(suite_dir, round_id), record)
    append_jsonl(suite_dir / "summary.jsonl", record)


def newest_mtime(paths: list[Path]) -> float:
    mtimes = []
    for path in paths:
        try:
            mtimes.append(path.stat().st_mtime)
        except FileNotFoundError:
            continue
    return max(mtimes) if mtimes else time.time()


def run_one_round(args: argparse.Namespace, suite_dir: Path, round_id: int, log_path: Path) -> int:
    runner_target_successes = min(args.target_successes, round_id)
    cmd = [
        str(PYTHON),
        str(RUNNER.relative_to(REPO_ROOT)),
        "--suite-name",
        SUITE_NAME,
        "--model",
        "qwen3-vl-8b-thinking",
        "--baselines",
        "reasoning",
        "--prompt-variants",
        "latency_aware",
        "--env-modes",
        args.env_mode,
        "--time-modes",
        "sync",
        "--difficulties",
        "easy",
        "--action-frame-modes",
        "video_history",
        "--task-file",
        "data/map1_10roads/tasks.json",
        "--task-index",
        "1",
        "--seed",
        "0",
        "--max-steps",
        "243",
        "--target-successes",
        str(runner_target_successes),
        "--max-rounds",
        str(round_id),
        "--reasoning-budget-tokens",
        str(args.reasoning_budget_tokens),
        "--reasoning-budget-prompt-modes",
        "think_shorter",
        "--reasoning-budget-answer-tokens",
        "512",
        "--inter-run-sleep",
        "1",
        "--resume",
    ]
    if args.reasoning_budget_prompt_text:
        cmd.extend(["--reasoning-budget-prompt-text", args.reasoning_budget_prompt_text])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log(f"Running through max_rounds={round_id}; log={log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch()
    with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        watch_paths = [
            log_path,
            run_status_path_for_round(suite_dir, round_id),
            result_path_for_round(suite_dir, round_id),
        ]
        started_at = time.time()
        last_change_at = started_at
        while True:
            code = proc.poll()
            if code is not None:
                return code
            now_s = time.time()
            if now_s - started_at > args.round_timeout_s:
                log(f"Round {round_id} timed out after {args.round_timeout_s}s")
                stop_process_group(proc, grace_s=15.0)
                mark_round_error(suite_dir, round_id, f"supervisor timeout after {args.round_timeout_s}s")
                return 124
            newest = newest_mtime(watch_paths)
            if newest > last_change_at + 1e-6:
                last_change_at = newest
            elif now_s - last_change_at > args.round_stall_timeout_s:
                log(
                    f"Round {round_id} stalled for {args.round_stall_timeout_s}s "
                    f"without log/status/result updates"
                )
                stop_process_group(proc, grace_s=15.0)
                mark_round_error(
                    suite_dir,
                    round_id,
                    f"supervisor stall timeout after {args.round_stall_timeout_s}s without updates",
                )
                return 125
            time.sleep(args.monitor_interval_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-name", default=SUITE_NAME)
    parser.add_argument("--setting-id", default=SETTING_ID)
    parser.add_argument("--reasoning-budget-tokens", type=int, default=128)
    parser.add_argument("--reasoning-budget-prompt-text", default=None)
    parser.add_argument("--target-successes", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--qwen-url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--env-mode", choices=["static", "realtime"], default="static")
    parser.add_argument("--qwen-ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--ue-port", type=int, default=9006)
    parser.add_argument("--ue-gpu", default="7")
    parser.add_argument("--ue-ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--round-timeout-s", type=float, default=21600.0)
    parser.add_argument("--round-stall-timeout-s", type=float, default=1800.0)
    parser.add_argument("--monitor-interval-s", type=float, default=30.0)
    parser.add_argument("--between-round-sleep-s", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    global SUITE_NAME, SETTING_ID
    args = parse_args()
    SUITE_NAME = args.suite_name
    SETTING_ID = args.setting_id
    suite_dir = REPO_ROOT / "results" / SUITE_NAME
    log_dir = suite_dir / "detached_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    wait_for_qwen(args.qwen_url, args.qwen_ready_timeout_s)

    while True:
        successes = count_successes(suite_dir)
        if successes >= args.target_successes:
            log(f"Target successes reached: {successes}/{args.target_successes}")
            return 0

        round_id = next_unfinished_round(suite_dir, args.max_rounds)
        if round_id is None:
            log(f"No unfinished rounds remain; successes={successes}/{args.target_successes}")
            return 0 if successes >= args.target_successes else 2

        ue_log = log_dir / f"ue_round_{round_id:02d}.log"
        runner_log = log_dir / f"runner_round_{round_id:02d}.log"
        ue_proc = None
        try:
            ue_proc = start_ue(ue_log, args.ue_gpu)
            wait_for_ue(ue_log, args.ue_port, args.ue_ready_timeout_s)
            code = run_one_round(args, suite_dir, round_id, runner_log)
            log(f"Round {round_id} runner exited with code {code}")
            if code != 0 and not result_path_for_round(suite_dir, round_id).exists():
                status = load_json(run_status_path_for_round(suite_dir, round_id))
                if status.get("status") not in {"completed", "error"}:
                    mark_round_error(suite_dir, round_id, f"runner exited with code {code}")
        except Exception as exc:
            log(f"Round {round_id} supervisor error: {exc}")
            if not result_path_for_round(suite_dir, round_id).exists():
                mark_round_error(suite_dir, round_id, f"supervisor error: {exc}")
        finally:
            stop_process_group(ue_proc)
            time.sleep(args.between_round_sleep_s)


if __name__ == "__main__":
    raise SystemExit(main())
