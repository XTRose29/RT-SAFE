#!/usr/bin/env python3
"""
Run deterministic code baselines for the realtime navigation task.

The policies do not call a VLM:
- greedy: turn/move directly toward the active route subgoal
- safety: take greedy progress when collision-safe, otherwise choose a
  collision-safe progress/hold fallback from the internal evaluator

Each baseline decision has zero internal latency. Move duration is derived from
the selected waypoint's metric distance and the agent's constant walking speed,
so the rendered 1m/2m/4m candidates remain physically consistent.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.run_experiment_matrix import (  # noqa: E402
    ENV_MODES,
    TIME_MODES,
    append_jsonl,
    existing_result,
    sanitize_id,
    summarize_result,
    write_json,
)


CODE_BASELINES = {
    "greedy": {
        "name": "greedy",
        "control_mode": "baseline_greedy",
        "description": "Deterministic geometric policy that walks toward the active route subgoal.",
    },
    "safety": {
        "name": "safety",
        "control_mode": "baseline_safety",
        "description": "Deterministic safety-first policy using the internal collision preview each step.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=243)
    parser.add_argument("--internal-latency-seconds", type=float, default=0.0)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--baselines", nargs="+", default=["greedy", "safety"], choices=sorted(CODE_BASELINES))
    parser.add_argument("--env-modes", nargs="+", default=["realtime", "static"], choices=sorted(ENV_MODES))
    parser.add_argument("--time-modes", nargs="+", default=["sync"], choices=sorted(TIME_MODES))
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easy"],
        choices=["easy", "medium", "default", "level0", "level1", "level2", "level3", "level4"],
    )
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--ue-ip", default="127.0.0.1")
    parser.add_argument("--ue-port", type=int, default=9006)
    parser.add_argument("--inter-run-sleep", type=float, default=20.0)
    parser.add_argument("--record-per-step", action="store_true", default=True)
    parser.add_argument("--no-record-per-step", dest="record_per_step", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--load-all-unsafe-triggers",
        action="store_true",
        help="Keep the requested difficulty label but activate 100%% of every enabled hazard category.",
    )
    parser.add_argument(
        "--uniform-greedy-movement",
        action="store_true",
        help="Interpolate each move at constant speed so execution reaches the rendered metric waypoint.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Do not destroy spawned UE actors after a run.",
    )
    return parser.parse_args()


def baseline_llm_stub(baseline_name: str) -> dict[str, Any]:
    return {
        "model": f"code-baseline-{baseline_name}",
        "provider": "self-hosted",
        "url": "http://127.0.0.1:30001/v1",
        "reasoning": False,
        "extra_body": {"code_baseline": True},
    }


def build_conditions(args: argparse.Namespace, suite_dir: Path) -> list[dict[str, Any]]:
    conditions = []
    for baseline_name in args.baselines:
        baseline = CODE_BASELINES[baseline_name]
        for env_mode in args.env_modes:
            for time_mode in args.time_modes:
                for difficulty in args.difficulties:
                    setting_id = sanitize_id(f"{baseline_name}__{env_mode}__{time_mode}__d{difficulty}__code")
                    for round_id in range(1, args.max_rounds + 1):
                        condition_id = sanitize_id(f"{setting_id}__round{round_id:02d}")
                        run_dir = suite_dir / "runs" / setting_id / f"round_{round_id:02d}"
                        agent_config_path = suite_dir / "configs" / f"{condition_id}.agents.json"
                        condition = {
                            "condition_id": condition_id,
                            "setting_id": setting_id,
                            "round": round_id,
                            "task_index": args.task_index,
                            "task_file": args.task_file,
                            "seed": args.seed,
                            "max_steps": args.max_steps,
                            "difficulty": difficulty,
                            "baseline": copy.deepcopy(baseline),
                            "baseline_policy": baseline_name,
                            "control_mode": baseline["control_mode"],
                            "env_mode": env_mode,
                            "realtime_thinking": ENV_MODES[env_mode],
                            "time_mode": time_mode,
                            "use_tick": TIME_MODES[time_mode],
                            "token_based": False,
                            "record_per_step": args.record_per_step,
                            "use_action_frames": False,
                            "prompt_style": "code_baseline",
                            "fixed_action_seconds": None,
                            "move_timing": "distance_based_constant_speed",
                            "internal_latency_seconds": float(args.internal_latency_seconds),
                            "load_all_unsafe_triggers": bool(args.load_all_unsafe_triggers),
                            "uniform_greedy_movement": bool(args.uniform_greedy_movement),
                            "agent_config_path": os.path.relpath(agent_config_path, REPO_ROOT),
                            "run_dir": os.path.relpath(run_dir, REPO_ROOT),
                            "llm": baseline_llm_stub(baseline_name),
                        }
                        conditions.append(condition)
    return conditions


def write_manifest(suite_dir: Path, conditions: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "suite_dir": str(suite_dir),
        "reference_suite": (
            "results/building_collision_feedback_simple_tl_prompt_qwen3vl8b_dynamic_easy_"
            "20260630_gate_rerun_fixed_tlcheck_20260630"
        ),
        "task_file": args.task_file,
        "task_index": args.task_index,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "record_per_step": args.record_per_step,
        "token_based": False,
        "internal_latency_seconds": float(args.internal_latency_seconds),
        "fixed_action_seconds": None,
        "move_timing": "distance_based_constant_speed",
        "load_all_unsafe_triggers": bool(args.load_all_unsafe_triggers),
        "uniform_greedy_movement": bool(args.uniform_greedy_movement),
        "condition_count": len(conditions),
        "mappings": {
            "env_mode": {
                "static": "realtime_thinking=False; no thinking-time advance.",
                "realtime": "realtime_thinking=True; baseline still has zero decision latency.",
            },
            "time_mode": {
                "sync": "use_tick=True; tick-based synchronous stepping.",
                "async": "use_tick=False; wall-clock asynchronous stepping.",
            },
            "baselines": CODE_BASELINES,
        },
        "conditions": conditions,
    }
    manifest_path = suite_dir / "experiment_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def existing_terminal_status(run_dir: Path) -> dict[str, Any] | None:
    status_path = run_dir / "run_status.json"
    if not status_path.exists():
        return None
    try:
        with status_path.open("r", encoding="utf-8") as f:
            status = json.load(f)
    except Exception:
        return None
    if status.get("status") in {"completed", "error"}:
        return status
    return None


def run_condition(condition: dict[str, Any], summary_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    from base.rt_communicator import RTCommunicator
    from base.rt_unrealcv import RTUnrealCV
    from manager.world_manager import WorldManager

    run_dir = (REPO_ROOT / condition["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        result_path = existing_result(run_dir)
        if result_path:
            record = {
                "condition_id": condition["condition_id"],
                "setting_id": condition["setting_id"],
                "round": condition["round"],
                "baseline_policy": condition["baseline_policy"],
                "status": "skipped_existing",
                "result_path": str(result_path),
                **summarize_result(result_path),
            }
            append_jsonl(summary_path, record)
            return record
        terminal_status = existing_terminal_status(run_dir)
        if terminal_status:
            record = {
                "condition_id": condition["condition_id"],
                "setting_id": condition["setting_id"],
                "round": condition["round"],
                "baseline_policy": condition["baseline_policy"],
                "status": f"skipped_existing_{terminal_status.get('status')}",
                "success": False,
                "error": terminal_status.get("error"),
            }
            append_jsonl(summary_path, record)
            return record

    status = {
        "condition_id": condition["condition_id"],
        "setting_id": condition["setting_id"],
        "round": condition["round"],
        "baseline_policy": condition["baseline_policy"],
        "status": "running",
        "started_at": datetime.now().isoformat(),
    }
    write_json(run_dir / "run_status.json", status)

    communicator = None
    world_manager = None
    try:
        os.environ["CODE_BASELINE_THINKING_SECONDS"] = str(condition["internal_latency_seconds"])
        os.environ["SIMWORLD_UNIFORM_GREEDY_MOVEMENT"] = (
            "1" if condition["uniform_greedy_movement"] else "0"
        )
        communicator = RTCommunicator(RTUnrealCV(port=args.ue_port, ip=args.ue_ip))
        world_manager = WorldManager(
            communicator=communicator,
            agent_path=condition["agent_config_path"],
            task_file_path=condition["task_file"],
            seed=condition["seed"],
            control_mode=condition["control_mode"],
            token_based=condition["token_based"],
            use_tick=condition["use_tick"],
            realtime_thinking=condition["realtime_thinking"],
            record_per_step=condition["record_per_step"],
            use_action_frames=condition["use_action_frames"],
            prompt_style=condition["prompt_style"],
            max_steps=condition["max_steps"],
            results_dir=condition["run_dir"],
            load_all_unsafe_triggers=condition["load_all_unsafe_triggers"],
        )
        world_manager._batch_suffix = condition["condition_id"]
        world_manager.run_single_task(condition["task_index"], difficulty=condition["difficulty"])
        result_path = existing_result(run_dir)
        if result_path is None:
            raise RuntimeError("The run finished without writing a result file")
        result = json.loads(result_path.read_text())
        if result.get("rollout_error"):
            raise RuntimeError(f"The run recorded a rollout error: {result['rollout_error']}")
        record = {
            "condition_id": condition["condition_id"],
            "setting_id": condition["setting_id"],
            "round": condition["round"],
            "baseline_policy": condition["baseline_policy"],
            "control_mode": condition["control_mode"],
            "env_mode": condition["env_mode"],
            "time_mode": condition["time_mode"],
            "fixed_action_seconds": condition["fixed_action_seconds"],
            "internal_latency_seconds": condition["internal_latency_seconds"],
            "status": "completed",
            "finished_at": datetime.now().isoformat(),
            "result_path": str(result_path) if result_path else None,
            **summarize_result(result_path),
        }
        append_jsonl(summary_path, record)
        write_json(run_dir / "run_status.json", record)
        return record
    except Exception as exc:
        record = {
            "condition_id": condition["condition_id"],
            "setting_id": condition["setting_id"],
            "round": condition["round"],
            "baseline_policy": condition["baseline_policy"],
            "control_mode": condition["control_mode"],
            "env_mode": condition["env_mode"],
            "time_mode": condition["time_mode"],
            "status": "error",
            "finished_at": datetime.now().isoformat(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        append_jsonl(summary_path, record)
        write_json(run_dir / "run_status.json", record)
        return record
    finally:
        if args.skip_cleanup:
            if communicator is not None:
                try:
                    communicator.unrealcv.disconnect()
                except Exception:
                    pass
        elif world_manager is not None:
            world_manager.cleanup()
        elif communicator is not None:
            try:
                communicator.clear_agents()
            except Exception:
                pass


def write_condition_files(conditions: list[dict[str, Any]]) -> None:
    for condition in conditions:
        write_json(REPO_ROOT / condition["agent_config_path"], {"llm": [condition["llm"]]})
        write_json(REPO_ROOT / condition["run_dir"] / "condition.json", condition)


def main() -> int:
    args = parse_args()
    suite_name = args.suite_name or f"code_baselines_greedy_safety_static_dynamic_easy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir = REPO_ROOT / args.output_root / sanitize_id(suite_name)
    suite_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_conditions(args, suite_dir)
    write_condition_files(conditions)
    manifest_path = write_manifest(suite_dir, conditions, args)
    summary_path = suite_dir / "summary.jsonl"
    print(f"[code-baselines] conditions={len(conditions)} manifest={manifest_path}", flush=True)

    if args.plan_only:
        print("[code-baselines] plan-only; no UE run started", flush=True)
        return 0

    failed = False
    for index, condition in enumerate(conditions, start=1):
        print(f"[code-baselines] {index}/{len(conditions)} {condition['condition_id']}", flush=True)
        record = run_condition(condition, summary_path, args)
        if record.get("status") not in {"completed", "skipped_existing"}:
            failed = True
        if index < len(conditions) and args.inter_run_sleep > 0:
            time.sleep(args.inter_run_sleep)

    print(f"[code-baselines] done summary={summary_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
