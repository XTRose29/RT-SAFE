#!/usr/bin/env python3
"""
Run a single-task VLM experiment matrix with explicit condition folders.

This driver keeps the task and base VLM fixed while varying:
- VLM reasoning mode and reasoning length/budget labels
- Prompt style
- Static vs realtime environment during thinking
- Synchronous vs asynchronous environment stepping
- Difficulty level

Results are written under SimWorld-RealTime/results/<suite_name> by default.
Use --plan-only to create the manifest/config files without launching UE or the VLM.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PROMPT_VARIANTS = {
    "naive": "naive",
    "latency_aware": "instructional",
    "future_state": "future_state",
    "adaptive": "adaptive",
}

ENV_MODES = {
    "static": False,
    "realtime": True,
}

TIME_MODES = {
    "sync": True,
    "async": False,
}


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    reasoning: bool
    reasoning_effort: str | None = None
    max_tokens: int | None = None
    description: str = ""


BASELINES = {
    "non_reasoning": BaselineSpec(
        name="non_reasoning",
        reasoning=False,
        reasoning_effort=None,
        max_tokens=None,
        description="Same VLM with reasoning disabled.",
    ),
    "reasoning": BaselineSpec(
        name="reasoning",
        reasoning=True,
        reasoning_effort=None,
        max_tokens=None,
        description="Same VLM with provider default reasoning behavior.",
    ),
    "reasoning_minimal": BaselineSpec(
        name="reasoning_minimal",
        reasoning=True,
        reasoning_effort="minimal",
        max_tokens=64,
        description="Reasoning enabled with a minimal effort/token cap.",
    ),
    "reasoning_low": BaselineSpec(
        name="reasoning_low",
        reasoning=True,
        reasoning_effort="low",
        max_tokens=128,
        description="Reasoning enabled with a low effort/token cap.",
    ),
    "reasoning_medium": BaselineSpec(
        name="reasoning_medium",
        reasoning=True,
        reasoning_effort="medium",
        max_tokens=256,
        description="Reasoning enabled with a medium effort/token cap.",
    ),
    "reasoning_high": BaselineSpec(
        name="reasoning_high",
        reasoning=True,
        reasoning_effort="high",
        max_tokens=512,
        description="Reasoning enabled with a high effort/token cap.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a same-task VLM experiment matrix and save results in a new results subfolder."
    )
    parser.add_argument("--source-result", default="results/20260624_194515/task_1.json")
    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--agent-config", default="data/agents.json")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument(
        "--prefer-agent-model",
        action="store_true",
        help="Use --agent-config model instead of the source result model when --model is omitted.",
    )
    parser.add_argument(
        "--qwen-thinking-extra-body",
        choices=["auto", "off"],
        default="auto",
        help="In auto mode, add chat_template_kwargs.enable_thinking for Qwen3 self-hosted configs.",
    )

    parser.add_argument("--baselines", nargs="+", default=list(BASELINES), choices=sorted(BASELINES))
    parser.add_argument("--prompt-variants", nargs="+", default=["naive", "latency_aware", "future_state"], choices=sorted(PROMPT_VARIANTS))
    parser.add_argument("--env-modes", nargs="+", default=["static", "realtime"], choices=sorted(ENV_MODES))
    parser.add_argument("--time-modes", nargs="+", default=["sync", "async"], choices=sorted(TIME_MODES))
    parser.add_argument("--difficulties", nargs="+", default=["easy", "medium", "default"], choices=["easy", "medium", "default"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of conditions to run or plan.")

    parser.add_argument("--token-based", dest="token_based", action="store_true")
    parser.add_argument("--no-token-based", dest="token_based", action="store_false")
    parser.set_defaults(token_based=None)
    parser.add_argument("--record-per-step", dest="record_per_step", action="store_true")
    parser.add_argument("--no-record-per-step", dest="record_per_step", action="store_false")
    parser.set_defaults(record_per_step=None)
    parser.add_argument("--use-action-frames", dest="use_action_frames", action="store_true")
    parser.add_argument("--no-action-frames", dest="use_action_frames", action="store_false")
    parser.set_defaults(use_action_frames=None)

    parser.add_argument("--plan-only", action="store_true", help="Write manifest/configs only; do not connect to UE.")
    parser.add_argument("--resume", action="store_true", help="Skip run folders that already contain task_*.json.")
    parser.add_argument("--inter-run-sleep", type=float, default=20.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def sanitize_id(value: str) -> str:
    value = value.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def first_llm_config(agent_config_path: Path) -> dict[str, Any]:
    data = read_json(agent_config_path)
    llm = data.get("llm", data)
    if isinstance(llm, list):
        if not llm:
            raise ValueError(f"No llm configs found in {agent_config_path}")
        return copy.deepcopy(llm[0])
    return copy.deepcopy(llm)


def infer_task_index(source_result: dict[str, Any] | None) -> int:
    if not source_result:
        return 1
    task_id = source_result.get("task_id", 1)
    try:
        return int(task_id)
    except (TypeError, ValueError):
        return 1


def infer_record_per_step(source_path: Path, source_result: dict[str, Any] | None) -> bool:
    if not source_result:
        return False
    if "record_per_step" in source_result:
        return bool(source_result["record_per_step"])
    task_id = source_result.get("task_id", "*")
    return any(source_path.parent.glob(f"task_{task_id}_*_steps"))


def resolve_base_config(args: argparse.Namespace, source_result: dict[str, Any] | None) -> dict[str, Any]:
    cfg = first_llm_config(REPO_ROOT / args.agent_config)

    if args.model:
        cfg["model"] = args.model
    elif source_result and source_result.get("model") and not args.prefer_agent_model:
        cfg["model"] = source_result["model"]

    if args.provider:
        cfg["provider"] = args.provider
    if args.url:
        cfg["url"] = args.url

    return cfg


def add_qwen_thinking_extra_body(cfg: dict[str, Any], reasoning: bool) -> None:
    model = str(cfg.get("model", "")).lower()
    provider = str(cfg.get("provider", "")).lower()
    if "qwen3" not in model or provider not in {"self-hosted", "dashscope"}:
        return
    extra_body = copy.deepcopy(cfg.get("extra_body") or {})
    chat_template_kwargs = copy.deepcopy(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs["enable_thinking"] = bool(reasoning)
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    cfg["extra_body"] = extra_body


def llm_config_for_baseline(
    base_config: dict[str, Any],
    baseline: BaselineSpec,
    qwen_extra_body_mode: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg["reasoning"] = baseline.reasoning
    cfg["reasoning_effort"] = baseline.reasoning_effort
    cfg["max_tokens"] = baseline.max_tokens

    if baseline.reasoning_effort is None:
        cfg.pop("reasoning_effort", None)
    if baseline.max_tokens is None:
        cfg.pop("max_tokens", None)
    if qwen_extra_body_mode == "auto":
        add_qwen_thinking_extra_body(cfg, baseline.reasoning)

    return cfg


def build_conditions(args: argparse.Namespace, suite_dir: Path) -> list[dict[str, Any]]:
    source_path = REPO_ROOT / args.source_result
    source_result = read_json(source_path) if source_path.exists() else None
    base_config = resolve_base_config(args, source_result)

    task_index = args.task_index if args.task_index is not None else infer_task_index(source_result)
    seed = args.seed if args.seed is not None else int(source_result.get("seed", 0) if source_result else 0)
    max_steps = args.max_steps if args.max_steps is not None else int(source_result.get("max_steps", -1) if source_result else -1)
    token_based = args.token_based if args.token_based is not None else bool(source_result.get("token_based", False) if source_result else False)
    record_per_step = args.record_per_step if args.record_per_step is not None else infer_record_per_step(source_path, source_result)
    use_action_frames = args.use_action_frames if args.use_action_frames is not None else bool(source_result.get("use_action_frames", False) if source_result else False)

    conditions = []
    product = itertools.product(
        args.baselines,
        args.prompt_variants,
        args.env_modes,
        args.time_modes,
        args.difficulties,
        range(1, args.repeats + 1),
    )

    for baseline_name, prompt_variant, env_mode, time_mode, difficulty, repeat in product:
        baseline = BASELINES[baseline_name]
        llm_config = llm_config_for_baseline(base_config, baseline, args.qwen_thinking_extra_body)
        prompt_style = PROMPT_VARIANTS[prompt_variant]
        condition_id = sanitize_id(
            f"{baseline_name}__{prompt_variant}__{env_mode}__{time_mode}__d{difficulty}__r{repeat:02d}"
        )
        run_dir = suite_dir / "runs" / condition_id
        agent_config_path = suite_dir / "configs" / f"{condition_id}.agents.json"
        conditions.append(
            {
                "condition_id": condition_id,
                "task_index": task_index,
                "task_file": args.task_file,
                "seed": seed,
                "max_steps": max_steps,
                "difficulty": difficulty,
                "repeat": repeat,
                "baseline": asdict(baseline),
                "prompt_variant": prompt_variant,
                "prompt_style": prompt_style,
                "env_mode": env_mode,
                "realtime_thinking": ENV_MODES[env_mode],
                "time_mode": time_mode,
                "use_tick": TIME_MODES[time_mode],
                "token_based": token_based,
                "record_per_step": record_per_step,
                "use_action_frames": use_action_frames,
                "agent_config_path": str(agent_config_path),
                "run_dir": str(run_dir),
                "llm": llm_config,
            }
        )
        if args.limit is not None and len(conditions) >= args.limit:
            break

    return conditions


def write_condition_files(suite_dir: Path, conditions: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    created_at = datetime.now().isoformat()
    for condition in conditions:
        agent_config_path = REPO_ROOT / condition["agent_config_path"]
        write_json(agent_config_path, {"llm": [condition["llm"]]})
        write_json(REPO_ROOT / condition["run_dir"] / "condition.json", condition)

    manifest = {
        "created_at": created_at,
        "suite_dir": str(suite_dir),
        "source_result": args.source_result,
        "task_file": args.task_file,
        "agent_config": args.agent_config,
        "condition_count": len(conditions),
        "mappings": {
            "env_mode": {
                "static": "realtime_thinking=False; environment does not advance while VLM thinks.",
                "realtime": "realtime_thinking=True; environment advances while VLM thinks.",
            },
            "time_mode": {
                "sync": "use_tick=True; tick-based synchronous stepping.",
                "async": "use_tick=False; wall-clock asynchronous stepping.",
            },
            "prompt_variant": PROMPT_VARIANTS,
            "baselines": {name: asdict(spec) for name, spec in BASELINES.items()},
        },
        "conditions": conditions,
    }
    manifest_path = suite_dir / "experiment_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def existing_result(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("task_*.json"))
    return matches[0] if matches else None


def summarize_result(result_path: Path | None) -> dict[str, Any]:
    if result_path is None or not result_path.exists():
        return {}
    data = read_json(result_path)
    keys = [
        "success",
        "failed",
        "difficulty",
        "model",
        "reasoning_enabled",
        "reasoning_effort",
        "max_llm_tokens",
        "prompt_style",
        "use_tick",
        "realtime_thinking",
        "final_step",
        "decision_count",
        "collision_count",
        "passive_collision_count",
        "collision_type_counts",
        "passive_collision_type_counts",
        "building_collision_count",
        "human_collision_count",
        "object_collision_count",
        "building_collision_recovery_count",
        "consecutive_same_building_action_count",
        "consecutive_building_collision_count",
        "red_light_violations_count",
        "fall_count",
        "time_cost",
        "sim_time",
        "avg_response_time",
        "avg_completion_tokens",
        "avg_reasoning_tokens",
    ]
    return {key: data.get(key) for key in keys}


def run_condition(condition: dict[str, Any], summary_path: Path, args: argparse.Namespace) -> None:
    from base.rt_communicator import RTCommunicator
    from manager.world_manager import WorldManager

    run_dir = REPO_ROOT / condition["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        result_path = existing_result(run_dir)
        if result_path:
            append_jsonl(
                summary_path,
                {
                    "condition_id": condition["condition_id"],
                    "status": "skipped_existing",
                    "result_path": str(result_path),
                    **summarize_result(result_path),
                },
            )
            return

    status = {
        "condition_id": condition["condition_id"],
        "status": "running",
        "started_at": datetime.now().isoformat(),
    }
    write_json(run_dir / "run_status.json", status)

    communicator = None
    world_manager = None
    try:
        communicator = RTCommunicator()
        world_manager = WorldManager(
            communicator=communicator,
            agent_path=condition["agent_config_path"],
            task_file_path=condition["task_file"],
            seed=condition["seed"],
            control_mode="llm",
            token_based=condition["token_based"],
            use_tick=condition["use_tick"],
            realtime_thinking=condition["realtime_thinking"],
            record_per_step=condition["record_per_step"],
            use_action_frames=condition["use_action_frames"],
            prompt_style=condition["prompt_style"],
            max_steps=condition["max_steps"],
            results_dir=condition["run_dir"],
        )
        world_manager._batch_suffix = condition["condition_id"]
        world_manager.run_single_task(condition["task_index"], difficulty=condition["difficulty"])
        result_path = existing_result(run_dir)
        record = {
            "condition_id": condition["condition_id"],
            "status": "completed",
            "finished_at": datetime.now().isoformat(),
            "result_path": str(result_path) if result_path else None,
            **summarize_result(result_path),
        }
        append_jsonl(summary_path, record)
        write_json(run_dir / "run_status.json", record)
    except Exception as exc:
        record = {
            "condition_id": condition["condition_id"],
            "status": "error",
            "finished_at": datetime.now().isoformat(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        append_jsonl(summary_path, record)
        write_json(run_dir / "run_status.json", record)
    finally:
        if world_manager is not None:
            world_manager.cleanup()
        elif communicator is not None:
            try:
                communicator.clear_agents()
            except Exception:
                pass


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    suite_name = args.suite_name or f"experiment_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir = REPO_ROOT / args.output_root / sanitize_id(suite_name)
    suite_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_conditions(args, suite_dir)
    manifest_path = write_condition_files(suite_dir, conditions, args)
    summary_path = suite_dir / "run_summary.jsonl"

    print(f"Wrote experiment manifest: {manifest_path}")
    print(f"Conditions: {len(conditions)}")
    if args.plan_only:
        print("Plan-only mode; no UE/VLM run started.")
        return 0

    for idx, condition in enumerate(conditions, start=1):
        print(f"[{idx}/{len(conditions)}] Running {condition['condition_id']}")
        run_condition(condition, summary_path, args)
        if idx < len(conditions) and args.inter_run_sleep > 0:
            time.sleep(args.inter_run_sleep)

    print(f"Wrote run summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
