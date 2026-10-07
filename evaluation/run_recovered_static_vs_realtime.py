#!/usr/bin/env python3
"""Compare static-thinking vs realtime-thinking for the recovered June 24 setting.

The comparison keeps the model, task, difficulty, prompt recovery, use_tick,
token_based, and action-frame settings fixed. The only intended env-mode
difference is:
- static: realtime_thinking=False, env pauses while VLM thinks
- realtime: realtime_thinking=True, env advances while VLM thinks

Each condition retries until it gets --target-successes successful task runs or
reaches --max-rounds attempts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.run_recovered_20260624_011915 import (  # noqa: E402
    patch_runtime,
    read_json,
    result_in,
    summarize,
    write_json,
)


ENV_MODES = {
    "static": False,
    "realtime": True,
}


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_attempt(
    args: argparse.Namespace,
    suite_dir: Path,
    env_mode: str,
    round_id: int,
) -> dict[str, Any]:
    from base.rt_communicator import RTCommunicator
    from base.rt_unrealcv import RTUnrealCV
    from manager.world_manager import WorldManager

    realtime_thinking = ENV_MODES[env_mode]
    condition_id = f"{env_mode}_round_{round_id:02d}"
    run_dir = suite_dir / "runs" / env_mode / f"round_{round_id:02d}"
    config_path = suite_dir / "configs" / f"{condition_id}.agents.json"
    agent_config = {
        "llm": [
            {
                "model": args.model,
                "provider": args.provider,
                "url": args.url,
                "reasoning": True,
            }
        ]
    }
    write_json(config_path, agent_config)

    condition = {
        "condition_id": condition_id,
        "env_mode": env_mode,
        "round": round_id,
        "source_result": args.source_result,
        "task_index": args.task_index,
        "task_file": args.task_file,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "model": args.model,
        "provider": args.provider,
        "url": args.url,
        "reasoning_enabled": True,
        "llm_extra_body": None,
        "prompt_style": "instructional",
        "prompt_recovery": "20260624_011915_safety_preview_visible",
        "termination_recovery": "20260624_max_steps_success_or_failed_only",
        "token_based": False,
        "use_tick": True,
        "realtime_thinking": realtime_thinking,
        "use_action_frames": True,
        "record_per_step": True,
        "max_steps": args.max_steps,
    }
    write_json(run_dir / "condition.json", condition)
    write_json(
        run_dir / "run_status.json",
        {
            "condition_id": condition_id,
            "env_mode": env_mode,
            "round": round_id,
            "status": "running",
            "started_at": datetime.now().isoformat(),
        },
    )

    communicator = None
    world_manager = None
    try:
        communicator = RTCommunicator(RTUnrealCV(port=args.ue_port, ip=args.ue_ip))
        world_manager = WorldManager(
            communicator=communicator,
            agent_path=str(config_path.relative_to(REPO_ROOT)),
            task_file_path=args.task_file,
            seed=args.seed,
            control_mode="llm",
            token_based=False,
            use_tick=True,
            realtime_thinking=realtime_thinking,
            record_per_step=True,
            use_action_frames=True,
            prompt_style="instructional",
            max_steps=args.max_steps,
            results_dir=str(run_dir.relative_to(REPO_ROOT)),
        )
        world_manager._batch_suffix = f"recovered_static_vs_realtime_{condition_id}"
        world_manager.run_single_task(args.task_index, difficulty=args.difficulty)
        result_path = result_in(run_dir)
        record = {
            "condition_id": condition_id,
            "env_mode": env_mode,
            "round": round_id,
            "status": "completed",
            "finished_at": datetime.now().isoformat(),
            "result_path": str(result_path) if result_path else None,
            **summarize(result_path),
        }
    except Exception as exc:
        record = {
            "condition_id": condition_id,
            "env_mode": env_mode,
            "round": round_id,
            "status": "error",
            "finished_at": datetime.now().isoformat(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if world_manager is not None:
            try:
                world_manager.cleanup()
            except Exception as exc:
                record["cleanup_error"] = str(exc)
        elif communicator is not None:
            try:
                communicator.clear_agents()
            except Exception:
                pass

    write_json(run_dir / "run_status.json", record)
    return record


def aggregate(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in args.env_modes}
    for record in records:
        by_mode.setdefault(record["env_mode"], []).append(record)

    summary: dict[str, Any] = {
        "updated_at": datetime.now().isoformat(),
        "target_successes": args.target_successes,
        "max_rounds": args.max_rounds,
        "modes": {},
    }
    for mode, rows in by_mode.items():
        completed = [r for r in rows if r.get("status") == "completed"]
        successes = [r for r in completed if r.get("success") is True]
        first_success = successes[0] if successes else None
        summary["modes"][mode] = {
            "attempts": len(rows),
            "completed": len(completed),
            "successes": len(successes),
            "success_rate_completed": (len(successes) / len(completed)) if completed else None,
            "first_success_round": first_success.get("round") if first_success else None,
            "selected_success_result_path": first_success.get("result_path") if first_success else None,
            "selected_success_metrics": {
                key: first_success.get(key) if first_success else None
                for key in [
                    "final_step",
                    "decision_count",
                    "collision_count",
                    "passive_collision_count",
                    "red_light_violations_count",
                    "fall_count",
                    "time_cost",
                    "sim_time",
                    "avg_response_time",
                    "avg_completion_tokens",
                    "avg_reasoning_tokens",
                ]
            },
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", default="results/20260624_011915/task_1.json")
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--env-modes", nargs="+", default=["static", "realtime"], choices=sorted(ENV_MODES))
    parser.add_argument("--target-successes", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--provider", default="self-hosted")
    parser.add_argument("--url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--ue-ip", default="127.0.0.1")
    parser.add_argument("--ue-port", type=int, default=9006)
    parser.add_argument("--inter-run-sleep", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patch_runtime()

    source = read_json(REPO_ROOT / args.source_result)
    suite_name = args.suite_name or f"static_vs_realtime_recovered_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir = REPO_ROOT / "results" / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "suite_dir": str(suite_dir),
        "source_result": args.source_result,
        "source_result_metadata": {
            "timestamp": source.get("timestamp"),
            "success": source.get("success"),
            "model": source.get("model"),
            "reasoning_enabled": source.get("reasoning_enabled"),
            "difficulty": source.get("difficulty"),
            "prompt_style": source.get("prompt_style"),
            "token_based": source.get("token_based"),
            "use_tick": source.get("use_tick"),
            "realtime_thinking": source.get("realtime_thinking"),
            "use_action_frames": source.get("use_action_frames"),
            "max_steps": source.get("max_steps"),
        },
        "fixed_settings": {
            "prompt": "June 24 safety/progress preview visible in VLM user prompt",
            "termination": "success, explicit failure, or max_steps only",
            "llm_extra_body": None,
            "token_based": False,
            "use_tick": True,
            "use_action_frames": True,
            "record_per_step": True,
            "visual_proxy_note": "Original 30006 proxy unavailable; using Qwen server directly at 30001.",
        },
        "varied_axis": {
            "static": "realtime_thinking=False; env pauses while VLM thinks.",
            "realtime": "realtime_thinking=True; env advances while VLM thinks.",
        },
        "target_successes": args.target_successes,
        "max_rounds": args.max_rounds,
    }
    write_json(suite_dir / "experiment_manifest.json", manifest)

    summary_path = suite_dir / "summary.jsonl"
    records: list[dict[str, Any]] = load_jsonl(summary_path)
    print(f"[static-vs-realtime] suite={suite_dir}", flush=True)
    for env_mode in args.env_modes:
        existing = [r for r in records if r.get("env_mode") == env_mode]
        successes = sum(1 for r in existing if r.get("status") == "completed" and r.get("success") is True)
        attempts = len(existing)
        print(f"[static-vs-realtime] env_mode={env_mode}", flush=True)
        if successes >= args.target_successes:
            print(
                "[static-vs-realtime] "
                + json.dumps(
                    {
                        "env_mode": env_mode,
                        "status": "skipped_existing_success",
                        "attempts": attempts,
                        "successes": successes,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        for round_id in range(attempts + 1, args.max_rounds + 1):
            record = run_attempt(args, suite_dir, env_mode, round_id)
            records.append(record)
            append_jsonl(summary_path, record)
            print(
                "[static-vs-realtime] "
                + json.dumps(
                    {
                        "env_mode": env_mode,
                        "round": round_id,
                        "status": record.get("status"),
                        "success": record.get("success"),
                        "failed": record.get("failed"),
                        "final_step": record.get("final_step"),
                        "decision_count": record.get("decision_count"),
                        "collisions": record.get("collision_count"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if record.get("success") is True:
                successes += 1
                if successes >= args.target_successes:
                    break
            if round_id < args.max_rounds and args.inter_run_sleep > 0:
                time.sleep(args.inter_run_sleep)

    agg = aggregate(records, args)
    write_json(suite_dir / "aggregate_summary.json", agg)
    print(f"[static-vs-realtime] aggregate={json.dumps(agg, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
