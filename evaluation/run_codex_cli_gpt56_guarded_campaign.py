#!/usr/bin/env python3
"""Run the GPT-5.6 Codex CLI benchmark arms behind an account-usage guard.

The campaign runs one model at a time in the requested order.  Each model uses
multiple isolated UE workers.  The driver polls Codex's read-only account rate
limit endpoint and interrupts the active coordinator before starting more work
when the configured minimum remaining allowance is reached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_RUNNER = REPO_ROOT / "evaluation" / "run_openai_frontier_matrix.py"
DEFAULT_RESULTS_ROOT = Path(
    "results/"
    "codex_cli_gpt56_easy_seed0_lmh_20260912"
)
DEFAULT_MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
DEFAULT_EFFORTS = ("low", "medium", "high")
SOURCE_FILES = (
    "SimWorld/simworld/communicator/unrealcv.py",
    "base/rt_agent.py",
    "base/world_manager.py",
    "evaluation/run_openai_benchmark.py",
    "evaluation/run_openai_frontier_matrix.py",
    "evaluation/run_codex_cli_gpt56_guarded_campaign.py",
    "llm/codex_cli_backend.py",
    "llm/prompt.py",
    "llm/rt_llm.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "_")


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    ).stdout.strip()


def source_snapshot() -> dict[str, Any]:
    files: dict[str, str | None] = {}
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        files[relative] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else None
        )
    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "dirty_worktree_entries": git_output(
            "status", "--short", "--untracked-files=all"
        ).splitlines(),
        "source_sha256": files,
    }


def read_codex_rate_limits(timeout: float = 20.0) -> dict[str, Any]:
    """Read the signed-in ChatGPT Codex limit without making a model call."""
    process = subprocess.Popen(
        ["codex", "app-server"],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    messages = (
        {
            "method": "initialize",
            "id": 0,
            "params": {
                "clientInfo": {
                    "name": "simworld_usage_guard",
                    "title": "SimWorld Usage Guard",
                    "version": "1.0.0",
                }
            },
        },
        {"method": "initialized", "params": {}},
        {"method": "account/rateLimits/read", "id": 6},
    )
    try:
        for message in messages:
            process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        process.stdin.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                [process.stdout], [], [], max(0.0, deadline - time.monotonic())
            )
            if not ready:
                break
            line = process.stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") != 6:
                continue
            result = message.get("result") or {}
            limits = result.get("rateLimits") or {}
            primary = limits.get("primary") or {}
            used = float(primary["usedPercent"])
            return {
                "checked_at": utc_now(),
                "used_percent": used,
                "remaining_percent": max(0.0, 100.0 - used),
                "window_duration_minutes": primary.get("windowDurationMins"),
                "resets_at": primary.get("resetsAt"),
                "plan_type": limits.get("planType"),
                "purchased_credit_balance": (
                    (limits.get("credits") or {}).get("balance")
                ),
                "banked_full_resets": int(
                    (result.get("rateLimitResetCredits") or {}).get(
                        "availableCount", 0
                    )
                    or 0
                ),
            }
        raise RuntimeError("Codex app-server did not return account rate limits")
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def stop_process_group(process: subprocess.Popen[Any], *, grace: float = 180.0) -> int:
    if process.poll() is not None:
        return int(process.returncode or 0)
    os.killpg(process.pid, signal.SIGINT)
    try:
        return process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return process.wait(timeout=30)


def coordinator_command(args: argparse.Namespace, model: str) -> list[str]:
    suite = args.results_root / model_slug(model)
    return [
        sys.executable,
        str(MATRIX_RUNNER),
        "--mode",
        "coordinator",
        "--suite-dir",
        str(suite),
        "--workers",
        str(args.workers),
        "--worker-count",
        str(args.workers),
        "--gpus",
        *map(str, args.gpus),
        "--ports",
        *map(str, args.ports),
        "--max-attempts",
        str(args.max_attempts),
        "--request-timeout",
        str(args.request_timeout),
        "--unrealcv-request-timeout",
        str(args.unrealcv_request_timeout),
        "--cell-timeout",
        str(args.cell_timeout),
        "--ue-ready-timeout",
        str(args.ue_ready_timeout),
        "--ue-settle-seconds",
        str(args.ue_settle_seconds),
        "--ue-start-stagger-seconds",
        str(args.ue_start_stagger_seconds),
        "--process-shutdown-timeout",
        str(args.process_shutdown_timeout),
        "--status-interval",
        str(args.status_interval),
        "--ue-port-stride",
        str(args.ue_port_stride),
        "--record-png-compress-level",
        str(args.record_png_compress_level),
        "--record-images-first-steps",
        str(args.record_images_first_steps),
        "--record-images-last-steps",
        str(args.record_images_last_steps),
        "--dynamic-queue",
        "--fast-simulation",
        "--ue-no-rhi-thread",
        "--no-record-output-images",
        "--resume",
    ]


def model_environment(model: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "SIMWORLD_CAMPAIGN_MODEL": model,
            "SIMWORLD_CAMPAIGN_PROVIDER": "codex-cli",
            "SIMWORLD_CAMPAIGN_API_MODE": "codex_cli",
            "SIMWORLD_CAMPAIGN_DIFFICULTIES": "easy",
            "SIMWORLD_CAMPAIGN_ENV_MODES": "realtime,static",
            "SIMWORLD_CAMPAIGN_EFFORTS": ",".join(DEFAULT_EFFORTS),
            "SIMWORLD_CAMPAIGN_INPUT_PRICE": "0",
            "SIMWORLD_CAMPAIGN_OUTPUT_PRICE": "0",
        }
    )
    return environment


def run_model(args: argparse.Namespace, model: str, events: Path) -> int:
    model_dir = args.results_root / model_slug(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    command = coordinator_command(args, model)
    write_json(model_dir / "guarded_command.json", command)
    log_path = model_dir / "guarded_coordinator.log"
    append_jsonl(events, {"time": utc_now(), "event": "model_start", "model": model})
    consecutive_guard_errors = 0
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=model_environment(model),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                time.sleep(args.usage_poll_seconds)
                try:
                    limits = read_codex_rate_limits(args.usage_query_timeout)
                    consecutive_guard_errors = 0
                    append_jsonl(
                        events,
                        {
                            "time": utc_now(),
                            "event": "usage_check",
                            "model": model,
                        }
                        | limits,
                    )
                    write_json(args.results_root / "latest_usage.json", limits)
                    if limits["remaining_percent"] <= args.minimum_remaining_percent:
                        append_jsonl(
                            events,
                            {
                                "time": utc_now(),
                                "event": "usage_guard_stop",
                                "model": model,
                                "minimum_remaining_percent": args.minimum_remaining_percent,
                            }
                            | limits,
                        )
                        stop_process_group(process)
                        return 75
                except Exception as exc:
                    consecutive_guard_errors += 1
                    append_jsonl(
                        events,
                        {
                            "time": utc_now(),
                            "event": "usage_guard_error",
                            "model": model,
                            "consecutive_errors": consecutive_guard_errors,
                            "error": str(exc),
                        },
                    )
                    if consecutive_guard_errors >= args.maximum_guard_errors:
                        append_jsonl(
                            events,
                            {
                                "time": utc_now(),
                                "event": "usage_guard_unavailable_stop",
                                "model": model,
                            },
                        )
                        stop_process_group(process)
                        return 76
        except KeyboardInterrupt:
            stop_process_group(process)
            return 130
        return int(process.returncode or 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 4, 5])
    parser.add_argument("--ports", nargs="+", type=int, default=[12000, 12100, 12200, 12300])
    parser.add_argument("--minimum-remaining-percent", type=float, default=20.0)
    parser.add_argument("--usage-poll-seconds", type=float, default=30.0)
    parser.add_argument("--usage-query-timeout", type=float, default=20.0)
    parser.add_argument("--maximum-guard-errors", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--unrealcv-request-timeout", type=float, default=120.0)
    parser.add_argument("--cell-timeout", type=float, default=14400.0)
    parser.add_argument("--ue-ready-timeout", type=float, default=300.0)
    parser.add_argument("--ue-settle-seconds", type=float, default=30.0)
    parser.add_argument("--ue-start-stagger-seconds", type=float, default=30.0)
    parser.add_argument("--process-shutdown-timeout", type=float, default=20.0)
    parser.add_argument("--status-interval", type=float, default=30.0)
    parser.add_argument("--ue-port-stride", type=int, default=1)
    parser.add_argument("--record-png-compress-level", type=int, default=3)
    parser.add_argument("--record-images-first-steps", type=int, default=3)
    parser.add_argument("--record-images-last-steps", type=int, default=3)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if len(args.gpus) < args.workers or len(args.ports) < args.workers:
        parser.error("one GPU and port are required per worker")
    if not 0 < args.minimum_remaining_percent < 100:
        parser.error("--minimum-remaining-percent must be between 0 and 100")
    if args.usage_poll_seconds <= 0:
        parser.error("--usage-poll-seconds must be positive")
    unknown = set(args.models) - set(DEFAULT_MODELS)
    if unknown:
        parser.error("unsupported models: " + ", ".join(sorted(unknown)))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.results_root = args.results_root.resolve()
    args.results_root.mkdir(parents=True, exist_ok=True)
    events = args.results_root / "campaign_events.jsonl"
    initial_limits = read_codex_rate_limits(args.usage_query_timeout)
    campaign = {
        "schema_version": 1,
        "created_at": utc_now(),
        "models_in_order": args.models,
        "difficulty": "easy",
        "environment_modes": ["realtime", "static"],
        "reasoning_efforts": list(DEFAULT_EFFORTS),
        "tasks_per_condition": 36,
        "rollouts_per_model": 216,
        "seed": 0,
        "workers": args.workers,
        "gpus": args.gpus[: args.workers],
        "ports": args.ports[: args.workers],
        "minimum_remaining_percent": args.minimum_remaining_percent,
        "usage_poll_seconds": args.usage_poll_seconds,
        "automatically_consume_banked_resets": False,
        "initial_rate_limits": initial_limits,
        "source": source_snapshot(),
    }
    write_json(args.results_root / "campaign_manifest.json", campaign)
    append_jsonl(events, {"time": utc_now(), "event": "campaign_start"} | initial_limits)
    if initial_limits["remaining_percent"] <= args.minimum_remaining_percent:
        append_jsonl(events, {"time": utc_now(), "event": "initial_usage_guard_stop"})
        return 75

    for model in args.models:
        limits = read_codex_rate_limits(args.usage_query_timeout)
        write_json(args.results_root / "latest_usage.json", limits)
        if limits["remaining_percent"] <= args.minimum_remaining_percent:
            append_jsonl(
                events,
                {"time": utc_now(), "event": "between_models_usage_guard_stop", "model": model}
                | limits,
            )
            return 75
        code = run_model(args, model, events)
        append_jsonl(
            events,
            {
                "time": utc_now(),
                "event": "model_exit",
                "model": model,
                "return_code": code,
            },
        )
        if code != 0:
            return code

    final_limits = read_codex_rate_limits(args.usage_query_timeout)
    write_json(args.results_root / "latest_usage.json", final_limits)
    append_jsonl(events, {"time": utc_now(), "event": "campaign_complete"} | final_limits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
