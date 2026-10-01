#!/usr/bin/env python3
"""Run exactly one separately labeled GPT-5.6 Sol Codex CLI pilot rollout."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.run_qwen3vl8b_all_maps import UEServer


MODEL = "gpt-5.6-sol"
TASK_ID = 0
TASK_INDEX = 0
TASK_FILE = "data/map1_10roads/tasks.json"
UE_MAP = "RT10"
DEFAULT_UE_LAUNCHER = Path(
    "runtime/SimWorld.sh"
)
DEFAULT_OUTPUT = Path(
    "results/"
    "codex_cli_gpt5.6_sol_task0_easy_realtime_low_seed0_pilot_20260911"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ue-launcher", type=Path, default=DEFAULT_UE_LAUNCHER)
    parser.add_argument("--ue-gpu", type=int, default=2)
    parser.add_argument("--ue-port", type=int, default=9047)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="low")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--rollout-timeout", type=float, default=3600.0)
    parser.add_argument("--ue-ready-timeout", type=float, default=240.0)
    parser.add_argument("--ue-settle-seconds", type=float, default=8.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        print(
            f"Refusing to mix a pilot with existing artifacts: {output}",
            file=sys.stderr,
        )
        return 2
    output.mkdir(parents=True, exist_ok=True)
    attempt = output / "attempt_001"
    attempt.mkdir(parents=True, exist_ok=True)
    ue_log = output / "ue" / "rt10.log"
    ue_log.parent.mkdir(parents=True, exist_ok=True)
    rollout_log = attempt / "rollout.log"

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "experiment": "codex_cli_transport_pilot",
        "model": MODEL,
        "provider": "codex-cli",
        "api_mode": "codex_cli",
        "reasoning_effort": args.reasoning_effort,
        "task_id": TASK_ID,
        "task_index": TASK_INDEX,
        "task_file": TASK_FILE,
        "difficulty": "easy",
        "environment_mode": "realtime",
        "seed": 0,
        "prompt_style": "instructional",
        "traffic_policy": "visual_only",
        "observation_resolution": [720, 640],
        "temporal_images": "all approximately 0.5s action frames then current frame",
        "uniform_waypoint_movement": True,
        "max_output_tokens": None,
        "full_experiment_authorized": False,
        "inference_surface": "chatgpt_codex_cli",
        "benchmark_protocol_comparable_with_responses_api": True,
        "strict_model_transport_comparable_with_responses_api": False,
        "comparison_note": (
            "The benchmark prompt, image sequence, world settings, and scoring "
            "can match Responses API runs, but codex exec adds a product-level "
            "agent/runtime instruction layer. Analyze the access surfaces as "
            "separate experimental arms."
        ),
        "git_commit": commit,
        "dirty_worktree_entries": dirty,
    }
    write_json(output / "pilot_manifest.json", manifest)

    ue_args = SimpleNamespace(
        ue_launcher=str(args.ue_launcher),
        ue_gpu=args.ue_gpu,
        ue_port=args.ue_port,
        ue_host="127.0.0.1",
        ue_width=1280,
        ue_height=720,
        ue_fps=30,
        ue_no_rhi_thread=True,
        ue_ready_timeout=args.ue_ready_timeout,
        ue_settle_seconds=args.ue_settle_seconds,
        process_shutdown_timeout=20.0,
    )
    ue = UEServer(ue_args, ue_log)
    process: subprocess.Popen | None = None
    started_at = utc_now()
    try:
        ue.start(UE_MAP)
        write_json(
            output / "ue" / "startup.json",
            {
                "started_at": started_at,
                "startup_seconds": ue.startup_seconds,
                "renderer": ue.renderer_info,
                "gpu": args.ue_gpu,
                "port": args.ue_port,
            },
        )
        command = [
            sys.executable,
            str(REPO_ROOT / "evaluation/run_openai_benchmark.py"),
            "--model",
            MODEL,
            "--provider",
            "codex-cli",
            "--api-mode",
            "codex_cli",
            "--reasoning-effort",
            args.reasoning_effort,
            "--max-output-tokens",
            "none",
            "--task-file",
            TASK_FILE,
            "--task-index",
            str(TASK_INDEX),
            "--difficulty",
            "easy",
            "--seed",
            "0",
            "--max-steps",
            "-1",
            "--ue-port",
            str(args.ue_port),
            "--prompt-style",
            "instructional",
            "--traffic-policy",
            "visual_only",
            "--static-signal-vehicles",
            "--action-frame-mode",
            "video_history",
            "--results-dir",
            str(attempt),
            "--request-timeout",
            str(args.request_timeout),
            "--record-png-compress-level",
            "3",
            "--record-images-first-steps",
            "0",
            "--record-images-last-steps",
            "0",
            "--no-record-output-images",
            "--no-record-demo-images",
        ]
        write_json(output / "resolved_command.json", command)
        environment = os.environ.copy()
        environment.update(
            {
                "SIMWORLD_UNREALCV_REQUEST_TIMEOUT_SECONDS": "120",
                "SIMWORLD_UNREALCV_RECONNECT_RETRIES": "0",
                "SIMWORLD_OBSERVATION_WIDTH": "720",
                "SIMWORLD_OBSERVATION_HEIGHT": "640",
                "SIMWORLD_AGENT_CAMERA_FOV_DEG": "100",
                "SIMWORLD_FIRST_PERSON_CAMERA_PITCH_DEG": "-25",
                "SIMWORLD_CONCURRENT_REALTIME_INFERENCE": "1",
                "SIMWORLD_UNIFORM_GREEDY_MOVEMENT": "1",
                "SIMWORLD_SYNC_SETTLE_SECONDS": "0.02",
                "SIMWORLD_TICK_INTERVAL_SETTLE_SECONDS": "0",
                "SIMWORLD_TICK_COMPLETION_SETTLE_SECONDS": "0.05",
            }
        )
        with rollout_log.open("ab", buffering=0) as stream:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=args.rollout_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                write_json(
                    output / "pilot_summary.json",
                    {"completed_at": utc_now(), "valid_terminal_rollout": False, "error": "rollout timeout"},
                )
                return 2

        result_files = sorted(attempt.glob("task_*.json"))
        result = (
            json.loads(result_files[0].read_text(encoding="utf-8"))
            if len(result_files) == 1
            else None
        )
        issues: list[str] = []
        if result is None:
            issues.append(f"expected one result JSON; found {len(result_files)}")
        else:
            expected = {
                "task_id": TASK_ID,
                "model": MODEL,
                "provider": "codex-cli",
                "api_mode": "codex_cli",
                "reasoning_effort": args.reasoning_effort,
            }
            for key, value in expected.items():
                if result.get(key) != value:
                    issues.append(f"{key}={result.get(key)!r}; expected {value!r}")
            if result.get("rollout_error"):
                issues.append(f"rollout_error={result['rollout_error']}")
            if int(result.get("decision_count") or 0) <= 0:
                issues.append("no model decisions were recorded")
            if int(result.get("total_tokens") or 0) <= 0:
                issues.append("no Codex CLI usage was recorded")
        summary = {
            "completed_at": utc_now(),
            "runner_return_code": return_code,
            "valid_terminal_rollout": not issues,
            "issues": issues,
            "result_path": str(result_files[0]) if len(result_files) == 1 else None,
            "behavioral_success": result.get("success") if result else None,
            "termination_reason": result.get("termination_reason") if result else None,
            "decision_count": result.get("decision_count") if result else None,
            "total_tokens": result.get("total_tokens") if result else None,
            "cached_input_tokens": result.get("total_cached_prompt_tokens") if result else None,
            "reasoning_tokens": result.get("total_reasoning_tokens") if result else None,
        }
        write_json(output / "pilot_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if not issues else 2
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=20)
        ue.stop()


if __name__ == "__main__":
    raise SystemExit(main())
