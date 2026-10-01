#!/usr/bin/env python3
"""Run prompt/budget conditions one at a time, restarting UE between cells."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "evaluation" / "run_static_reasoning_prompt_budget_sweep.py"
PYTHON = Path(".venv/bin/python")
UE_EXE = Path(os.environ.get("SIMWORLD_UE_LAUNCHER", str(REPO_ROOT / "runtime/SimWorld.sh")))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_budget_name(spec: str) -> str:
    return spec.split(":", 1)[0]


def condition_done(summary_path: Path, env_mode: str, model: str, budget: str, prompt: str) -> bool:
    budget_name = parse_budget_name(budget)
    for row in load_jsonl(summary_path):
        if (
            row.get("env_mode", "static") == env_mode
            and row.get("model") == model
            and row.get("budget_name") == budget_name
            and row.get("prompt_variant") == prompt
            and row.get("status") == "completed"
            and row.get("success") is True
        ):
            return True
    return False


def start_ue(args: argparse.Namespace, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.ue_gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    cmd = [
        str(UE_EXE),
        "-RenderoffScreen",
        "RT10",
        f"-graphicsadapter={args.ue_gpu}",
        "-ResX=1280",
        "-ResY=720",
        "-FPSMAX=15",
    ]
    return subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def wait_for_ue(log_path: Path, timeout: float) -> None:
    start = time.time()
    ready_markers = ("Attach a UnrealcvSensor to the pawn", "Enabling input")
    while time.time() - start < timeout:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if all(marker in text for marker in ready_markers):
                return
        time.sleep(2.0)
    raise TimeoutError(f"UE did not become ready within {timeout}s; see {log_path}")


def stop_process_group(proc: subprocess.Popen, grace: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run_condition(args: argparse.Namespace, env_mode: str, model: str, budget: str, prompt: str) -> int:
    cmd = [
        str(PYTHON),
        str(RUNNER.relative_to(REPO_ROOT)),
        "--suite-name",
        args.suite_name,
        "--env-modes",
        env_mode,
        "--models",
        model,
        "--budgets",
        budget,
        "--prompt-variants",
        prompt,
        "--target-successes",
        str(args.target_successes),
        "--target-collision-count",
        str(args.target_collision_count),
        "--max-rounds",
        str(args.max_rounds),
        "--max-steps",
        str(args.max_steps),
        "--url",
        args.url,
        "--ue-port",
        str(args.ue_port),
        "--use-tick",
        "false",
        "--set-game-speed",
        "false",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, start_new_session=True)
    try:
        return proc.wait(timeout=args.condition_timeout)
    except subprocess.TimeoutExpired:
        print(
            f"[ue-restart-sweep] condition timed out after {args.condition_timeout}s: "
            f"{env_mode} {model} {budget} {prompt}",
            flush=True,
        )
        stop_process_group(proc, grace=10.0)
        return 124


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--env-modes", nargs="+", default=["static", "realtime"])
    parser.add_argument("--models", nargs="+", default=["qwen3-vl-8b"])
    parser.add_argument("--budgets", nargs="+", default=["low:128", "medium:256", "high:512"])
    parser.add_argument(
        "--prompt-variants",
        nargs="+",
        default=["basic", "guarded_recovery", "strict_barrier"],
    )
    parser.add_argument("--target-successes", type=int, default=1)
    parser.add_argument("--target-collision-count", type=int, default=999)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--ue-port", type=int, default=9006)
    parser.add_argument("--ue-gpu", default="7")
    parser.add_argument("--ue-ready-timeout", type=float, default=120.0)
    parser.add_argument("--between-condition-sleep", type=float, default=8.0)
    parser.add_argument("--condition-timeout", type=float, default=1800.0)
    parser.add_argument("--max-condition-retries", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_dir = REPO_ROOT / "results" / args.suite_name
    summary_path = suite_dir / "summary.jsonl"
    failures_path = suite_dir / "condition_failures.jsonl"
    log_dir = suite_dir / "ue_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for env_mode in args.env_modes:
        for model in args.models:
            for budget in args.budgets:
                for prompt in args.prompt_variants:
                    if condition_done(summary_path, env_mode, model, budget, prompt):
                        print(
                            f"[ue-restart-sweep] skip completed {env_mode} {model} {budget} {prompt}",
                            flush=True,
                        )
                        continue

                    condition_succeeded = False
                    last_code = None
                    for retry in range(1, args.max_condition_retries + 1):
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        log_path = log_dir / (
                            f"{stamp}_{env_mode}_{model}_{parse_budget_name(budget)}_{prompt}"
                            f"_try{retry}.log"
                        )
                        print(
                            f"[ue-restart-sweep] start UE for {env_mode} {model} {budget} {prompt} "
                            f"(try {retry}/{args.max_condition_retries})",
                            flush=True,
                        )
                        ue_proc = start_ue(args, log_path)
                        try:
                            wait_for_ue(log_path, args.ue_ready_timeout)
                            print(f"[ue-restart-sweep] UE ready: {log_path}", flush=True)
                            code = run_condition(args, env_mode, model, budget, prompt)
                            last_code = code
                            if code == 0 and condition_done(summary_path, env_mode, model, budget, prompt):
                                condition_succeeded = True
                                break
                            print(
                                f"[ue-restart-sweep] condition failed with code {code}: "
                                f"{env_mode} {model} {budget} {prompt}",
                                flush=True,
                            )
                        except Exception as exc:
                            last_code = 1
                            print(
                                f"[ue-restart-sweep] condition exception: {exc}: "
                                f"{env_mode} {model} {budget} {prompt}",
                                flush=True,
                            )
                        finally:
                            print("[ue-restart-sweep] stopping UE", flush=True)
                            stop_process_group(ue_proc)
                            time.sleep(args.between_condition_sleep)

                        if condition_done(summary_path, env_mode, model, budget, prompt):
                            condition_succeeded = True
                            break

                    if not condition_succeeded:
                        append_jsonl(
                            failures_path,
                            {
                                "finished_at": datetime.now().isoformat(),
                                "env_mode": env_mode,
                                "model": model,
                                "budget": budget,
                                "budget_name": parse_budget_name(budget),
                                "prompt_variant": prompt,
                                "attempts": args.max_condition_retries,
                                "last_code": last_code,
                            },
                        )
                        print(
                            f"[ue-restart-sweep] giving up after {args.max_condition_retries} tries: "
                            f"{env_mode} {model} {budget} {prompt}",
                            flush=True,
                        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
