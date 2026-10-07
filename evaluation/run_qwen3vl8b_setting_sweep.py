#!/usr/bin/env python3
"""
Run the current SimWorld-RealTime VLM settings with Qwen3-VL-8B.

This is a thin orchestration layer over evaluation/run_experiment_matrix.py that
adds per-setting stopping rules:
- retry each setting for at most --max-rounds rounds
- stop a setting early after --target-successes successful runs

Every run uses record_per_step=True so each step directory contains the system
prompt, user prompt, input images, full VLM response, parsed reasoning/action,
feedback, and collision details.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import types
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_simworld = types.ModuleType("simworld")
_simworld.__path__ = [str(REPO_ROOT / "SimWorld" / "simworld")]
sys.modules.setdefault("simworld", _simworld)

from evaluation.run_experiment_matrix import (  # noqa: E402
    BASELINES,
    ENV_MODES,
    PROMPT_VARIANTS,
    TIME_MODES,
    add_qwen_thinking_extra_body,
    append_jsonl,
    existing_result,
    sanitize_id,
    summarize_result,
    write_json,
)


ACTION_FRAME_MODES = {
    "image_history": False,
    "video_history": True,
}

REASONING_BUDGET_PROMPT_MODES = {
    "original": None,
    "think_shorter": (
        "Reasoning budget instruction: think briefly about the safety and progress facts needed "
        "for this action decision. Keep the hidden thinking concise, then provide the required "
        "final Reasoning, Action, and Param lines."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--suite-name", default=None)

    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--provider", default="self-hosted")
    parser.add_argument("--url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--ue-ip", default="127.0.0.1")
    parser.add_argument("--ue-port", type=int, default=9006)
    parser.add_argument(
        "--qwen-thinking-extra-body",
        choices=["auto", "off"],
        default="auto",
        help="In auto mode, add chat_template_kwargs.enable_thinking for Qwen3 reasoning baselines.",
    )

    parser.add_argument("--baselines", nargs="+", default=list(BASELINES), choices=sorted(BASELINES))
    parser.add_argument("--prompt-variants", nargs="+", default=sorted(PROMPT_VARIANTS), choices=sorted(PROMPT_VARIANTS))
    parser.add_argument("--env-modes", nargs="+", default=sorted(ENV_MODES), choices=sorted(ENV_MODES))
    parser.add_argument("--time-modes", nargs="+", default=sorted(TIME_MODES), choices=sorted(TIME_MODES))
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easy", "medium", "default"],
        choices=["easy", "medium", "default", "level0", "level1", "level2", "level3", "level4"],
    )
    parser.add_argument("--action-frame-modes", nargs="+", default=sorted(ACTION_FRAME_MODES), choices=sorted(ACTION_FRAME_MODES))

    parser.add_argument("--target-successes", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument(
        "--reasoning-budget-tokens",
        nargs="+",
        type=int,
        default=None,
        help="Hard thinking-token budgets. The local Qwen server force-closes </think> after this budget and then generates the final answer.",
    )
    parser.add_argument(
        "--reasoning-budget-answer-tokens",
        type=int,
        default=512,
        help="Final-answer token budget after forced </think> for reasoning-budget runs.",
    )
    parser.add_argument(
        "--reasoning-budget-prompt-modes",
        nargs="+",
        choices=sorted(REASONING_BUDGET_PROMPT_MODES),
        default=["original"],
        help="original keeps the prompt unchanged; think_shorter appends a concise-thinking instruction and still enforces the hard budget.",
    )
    parser.add_argument(
        "--reasoning-budget-prompt-text",
        default=None,
        help="Optional custom prompt suffix for --reasoning-budget-prompt-modes think_shorter.",
    )
    parser.add_argument("--token-based", action="store_true", default=False)
    parser.add_argument("--inter-run-sleep", type=float, default=20.0)
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Do not destroy spawned UE actors after a run. Use this when running one round per fresh UE server.",
    )
    parser.add_argument("--limit-settings", type=int, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--coverage-preset",
        choices=["none", "requested_axes"],
        default="none",
        help="Use a compact hand-picked suite that covers the requested VLM/env axes without the full factorial.",
    )
    return parser.parse_args()


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


def run_condition_with_unrealcv(condition: dict[str, Any], summary_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    from base.rt_communicator import RTCommunicator
    from base.rt_unrealcv import RTUnrealCV
    from manager.world_manager import WorldManager
    import time
    import traceback

    run_dir = REPO_ROOT / condition["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        result_path = existing_result(run_dir)
        if result_path:
            record = {
                "condition_id": condition["condition_id"],
                "setting_id": condition.get("setting_id"),
                "round": condition.get("round"),
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
                "setting_id": condition.get("setting_id"),
                "round": condition.get("round"),
                "status": f"skipped_existing_{terminal_status.get('status')}",
                "success": False,
                "failed": terminal_status.get("failed"),
                "error": terminal_status.get("error"),
            }
            append_jsonl(summary_path, record)
            return record

    status = {
        "condition_id": condition["condition_id"],
        "setting_id": condition.get("setting_id"),
        "round": condition.get("round"),
        "status": "running",
        "started_at": datetime.now().isoformat(),
    }
    write_json(run_dir / "run_status.json", status)

    communicator = None
    world_manager = None
    try:
        communicator = RTCommunicator(RTUnrealCV(port=args.ue_port, ip=args.ue_ip))
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
            "setting_id": condition.get("setting_id"),
            "round": condition.get("round"),
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
            "setting_id": condition.get("setting_id"),
            "round": condition.get("round"),
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
        time.sleep(args.inter_run_sleep)


def llm_config_for_baseline(args: argparse.Namespace, baseline_name: str) -> dict[str, Any]:
    baseline = BASELINES[baseline_name]
    cfg: dict[str, Any] = {
        "model": args.model,
        "provider": args.provider,
        "url": args.url,
        "reasoning": baseline.reasoning,
        "reasoning_effort": baseline.reasoning_effort,
        "max_tokens": baseline.max_tokens,
    }
    if baseline.reasoning_effort is None:
        cfg.pop("reasoning_effort")
    if baseline.max_tokens is None:
        cfg.pop("max_tokens")
    if args.llm_max_tokens is not None:
        cfg["max_tokens"] = args.llm_max_tokens
    if args.qwen_thinking_extra_body == "auto":
        add_qwen_thinking_extra_body(cfg, baseline.reasoning)
    return cfg


def apply_reasoning_budget_to_llm_config(args: argparse.Namespace, setting: dict[str, Any]) -> None:
    budget_tokens = setting.get("reasoning_budget_tokens")
    if budget_tokens is None:
        return

    llm_cfg = setting["llm"]
    extra_body = copy.deepcopy(llm_cfg.get("extra_body") or {})
    extra_body["reasoning_budget_mode"] = "force_close"
    extra_body["reasoning_budget_tokens"] = int(budget_tokens)
    extra_body["reasoning_budget_answer_tokens"] = int(args.reasoning_budget_answer_tokens)
    extra_body["reasoning_budget_force_close"] = True
    llm_cfg["extra_body"] = extra_body
    llm_cfg["max_tokens"] = int(budget_tokens) + int(args.reasoning_budget_answer_tokens) + 16

    if setting.get("reasoning_budget_prompt_mode") == "think_shorter":
        llm_cfg["prompt_suffix"] = args.reasoning_budget_prompt_text or REASONING_BUDGET_PROMPT_MODES["think_shorter"]


def make_setting_id(parts: dict[str, str]) -> str:
    fields = [
        parts["baseline"],
        parts["prompt_variant"],
        parts["env_mode"],
        parts["time_mode"],
        f"d{parts['difficulty']}",
        parts["action_frame_mode"],
    ]
    if parts.get("reasoning_budget_name"):
        fields.extend([parts["reasoning_budget_name"], parts.get("reasoning_budget_prompt_mode", "original")])
    return sanitize_id("__".join(fields))


REQUESTED_AXES_PRESET = [
    {
        "baseline": "non_reasoning",
        "prompt_variant": "naive",
        "env_mode": "static",
        "time_mode": "sync",
        "difficulty": "easy",
        "action_frame_mode": "image_history",
        "coverage": "VLM baseline / non-reasoning / static-sync / easy / image-history",
    },
    {
        "baseline": "reasoning_minimal",
        "prompt_variant": "latency_aware",
        "env_mode": "static",
        "time_mode": "sync",
        "difficulty": "easy",
        "action_frame_mode": "video_history",
        "coverage": "reasoning budget minimal / latency-aware prompt / video history",
    },
    {
        "baseline": "reasoning_low",
        "prompt_variant": "latency_aware",
        "env_mode": "static",
        "time_mode": "sync",
        "difficulty": "easy",
        "action_frame_mode": "video_history",
        "coverage": "reasoning budget low / latency-aware prompt / video history",
    },
    {
        "baseline": "reasoning_medium",
        "prompt_variant": "latency_aware",
        "env_mode": "static",
        "time_mode": "sync",
        "difficulty": "easy",
        "action_frame_mode": "video_history",
        "coverage": "reasoning budget medium / latency-aware prompt / video history",
    },
    {
        "baseline": "reasoning_high",
        "prompt_variant": "latency_aware",
        "env_mode": "static",
        "time_mode": "sync",
        "difficulty": "easy",
        "action_frame_mode": "video_history",
        "coverage": "reasoning budget high / latency-aware prompt / video history",
    },
    {
        "baseline": "reasoning_low",
        "prompt_variant": "future_state",
        "env_mode": "static",
        "time_mode": "sync",
        "difficulty": "easy",
        "action_frame_mode": "video_history",
        "coverage": "future-state hint/instruction prompt",
    },
    {
        "baseline": "reasoning_low",
        "prompt_variant": "latency_aware",
        "env_mode": "realtime",
        "time_mode": "sync",
        "difficulty": "easy",
        "action_frame_mode": "video_history",
        "coverage": "realtime env: environment advances while VLM thinks",
    },
    {
        "baseline": "reasoning_low",
        "prompt_variant": "latency_aware",
        "env_mode": "static",
        "time_mode": "async",
        "difficulty": "easy",
        "action_frame_mode": "video_history",
        "coverage": "asynchronous env stepping",
    },
    {
        "baseline": "reasoning_low",
        "prompt_variant": "latency_aware",
        "env_mode": "static",
        "time_mode": "sync",
        "difficulty": "medium",
        "action_frame_mode": "video_history",
        "coverage": "difficulty medium",
    },
    {
        "baseline": "reasoning_low",
        "prompt_variant": "latency_aware",
        "env_mode": "static",
        "time_mode": "sync",
        "difficulty": "default",
        "action_frame_mode": "video_history",
        "coverage": "difficulty default",
    },
]


def setting_from_parts(args: argparse.Namespace, parts: dict[str, str]) -> dict[str, Any]:
    setting = {
        **parts,
        "setting_id": make_setting_id(parts),
        "baseline_spec": asdict(BASELINES[parts["baseline"]]),
        "llm": llm_config_for_baseline(args, parts["baseline"]),
        "prompt_style": PROMPT_VARIANTS[parts["prompt_variant"]],
        "realtime_thinking": ENV_MODES[parts["env_mode"]],
        "use_tick": TIME_MODES[parts["time_mode"]],
        "use_action_frames": ACTION_FRAME_MODES[parts["action_frame_mode"]],
    }
    if parts.get("reasoning_budget_tokens") is not None:
        setting["reasoning_budget_name"] = parts["reasoning_budget_name"]
        setting["reasoning_budget_tokens"] = parts["reasoning_budget_tokens"]
        setting["reasoning_budget_answer_tokens"] = args.reasoning_budget_answer_tokens
        setting["reasoning_budget_prompt_mode"] = parts.get("reasoning_budget_prompt_mode", "original")
        setting["reasoning_budget_prompt_suffix"] = (
            args.reasoning_budget_prompt_text
            if setting["reasoning_budget_prompt_mode"] == "think_shorter" and args.reasoning_budget_prompt_text
            else REASONING_BUDGET_PROMPT_MODES.get(setting["reasoning_budget_prompt_mode"])
        )
        apply_reasoning_budget_to_llm_config(args, setting)
    if "coverage" in parts:
        setting["coverage"] = parts["coverage"]
    return setting


def reasoning_budget_axis(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.reasoning_budget_tokens:
        return [
            {
                "reasoning_budget_name": None,
                "reasoning_budget_tokens": None,
                "reasoning_budget_prompt_mode": "original",
            }
        ]

    axis = []
    for budget_tokens in args.reasoning_budget_tokens:
        for prompt_mode in args.reasoning_budget_prompt_modes:
            axis.append(
                {
                    "reasoning_budget_name": f"rb{budget_tokens}",
                    "reasoning_budget_tokens": budget_tokens,
                    "reasoning_budget_prompt_mode": prompt_mode,
                }
            )
    return axis


def build_settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.coverage_preset == "requested_axes":
        settings = [
            setting_from_parts(args, {**spec, **budget_parts})
            for spec in REQUESTED_AXES_PRESET
            for budget_parts in reasoning_budget_axis(args)
        ]
        if args.limit_settings is not None:
            return settings[: args.limit_settings]
        return settings

    settings = []
    product = itertools.product(
        args.baselines,
        args.prompt_variants,
        args.env_modes,
        args.time_modes,
        args.difficulties,
        args.action_frame_modes,
        reasoning_budget_axis(args),
    )
    for baseline_name, prompt_variant, env_mode, time_mode, difficulty, action_frame_mode, budget_parts in product:
        parts = {
            "baseline": baseline_name,
            "prompt_variant": prompt_variant,
            "env_mode": env_mode,
            "time_mode": time_mode,
            "difficulty": difficulty,
            "action_frame_mode": action_frame_mode,
            **budget_parts,
        }
        setting = setting_from_parts(args, parts)
        settings.append(setting)
        if args.limit_settings is not None and len(settings) >= args.limit_settings:
            break
    return settings


def condition_for_round(args: argparse.Namespace, suite_dir: Path, setting: dict[str, Any], round_id: int) -> dict[str, Any]:
    condition_id = sanitize_id(f"{setting['setting_id']}__round{round_id:02d}")
    run_dir = suite_dir / "runs" / setting["setting_id"] / f"round_{round_id:02d}"
    agent_config_path = suite_dir / "configs" / f"{condition_id}.agents.json"
    return {
        "condition_id": condition_id,
        "setting_id": setting["setting_id"],
        "round": round_id,
        "task_index": args.task_index,
        "task_file": args.task_file,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "difficulty": setting["difficulty"],
        "repeat": round_id,
        "baseline": copy.deepcopy(setting["baseline_spec"]),
        "prompt_variant": setting["prompt_variant"],
        "prompt_style": setting["prompt_style"],
        "env_mode": setting["env_mode"],
        "realtime_thinking": setting["realtime_thinking"],
        "time_mode": setting["time_mode"],
        "use_tick": setting["use_tick"],
        "token_based": args.token_based,
        "record_per_step": True,
        "action_frame_mode": setting["action_frame_mode"],
        "use_action_frames": setting["use_action_frames"],
        "reasoning_budget_name": setting.get("reasoning_budget_name"),
        "reasoning_budget_tokens": setting.get("reasoning_budget_tokens"),
        "reasoning_budget_answer_tokens": setting.get("reasoning_budget_answer_tokens"),
        "reasoning_budget_prompt_mode": setting.get("reasoning_budget_prompt_mode"),
        "reasoning_budget_prompt_suffix": setting.get("reasoning_budget_prompt_suffix"),
        "agent_config_path": str(agent_config_path.relative_to(REPO_ROOT)),
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "llm": copy.deepcopy(setting["llm"]),
    }


def write_manifest(suite_dir: Path, settings: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "suite_dir": str(suite_dir),
        "task_file": args.task_file,
        "task_index": args.task_index,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "target_successes": args.target_successes,
        "max_rounds": args.max_rounds,
        "model": args.model,
        "provider": args.provider,
        "url": args.url,
        "token_based": args.token_based,
        "record_per_step": True,
        "settings_count": len(settings),
        "reasoning_budget_axis": {
            "tokens": args.reasoning_budget_tokens,
            "answer_tokens": args.reasoning_budget_answer_tokens,
            "prompt_modes": args.reasoning_budget_prompt_modes,
            "prompt_text_override": args.reasoning_budget_prompt_text,
            "mechanism": "local Qwen server generates up to reasoning_budget_tokens, force-appends </think> if needed, then generates final answer.",
        },
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
            "action_frame_mode": ACTION_FRAME_MODES,
            "baselines": {name: asdict(spec) for name, spec in BASELINES.items()},
            "reasoning_budget_prompt_mode": REASONING_BUDGET_PROMPT_MODES,
        },
        "settings": settings,
    }
    manifest_path = suite_dir / "experiment_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def setting_success_count(setting_dir: Path) -> int:
    successes = 0
    for result_path in setting_dir.glob("round_*/task_*.json"):
        try:
            with result_path.open("r", encoding="utf-8") as f:
                if json.load(f).get("success", False):
                    successes += 1
        except Exception:
            continue
    return successes


def run_setting(args: argparse.Namespace, suite_dir: Path, setting: dict[str, Any], summary_path: Path) -> None:
    setting_dir = suite_dir / "runs" / setting["setting_id"]
    setting_dir.mkdir(parents=True, exist_ok=True)
    successes = setting_success_count(setting_dir) if args.resume else 0

    for round_id in range(1, args.max_rounds + 1):
        if successes >= args.target_successes:
            append_jsonl(
                summary_path,
                {
                    "setting_id": setting["setting_id"],
                    "status": "target_successes_reached",
                    "successes": successes,
                    "target_successes": args.target_successes,
                    "round": round_id - 1,
                },
            )
            break

        condition = condition_for_round(args, suite_dir, setting, round_id)
        write_json(REPO_ROOT / condition["agent_config_path"], {"llm": [condition["llm"]]})
        write_json(REPO_ROOT / condition["run_dir"] / "condition.json", condition)

        if args.plan_only:
            continue

        result_summary = run_condition_with_unrealcv(condition, summary_path, args)
        if result_summary.get("success") and result_summary.get("status") != "skipped_existing":
            successes += 1

    write_json(
        setting_dir / "setting_status.json",
        {
            "setting_id": setting["setting_id"],
            "successes": successes,
            "target_successes": args.target_successes,
            "max_rounds": args.max_rounds,
            "completed_at": datetime.now().isoformat(),
        },
    )


def main() -> int:
    args = parse_args()
    if args.target_successes < 1:
        raise ValueError("--target-successes must be >= 1")
    if args.max_rounds < args.target_successes:
        raise ValueError("--max-rounds must be >= --target-successes")

    suite_name = args.suite_name or f"qwen3vl8b_setting_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir = REPO_ROOT / args.output_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    settings = build_settings(args)
    manifest_path = write_manifest(suite_dir, settings, args)
    summary_path = suite_dir / "summary.jsonl"
    print(f"[sweep] settings={len(settings)} manifest={manifest_path}", flush=True)

    for index, setting in enumerate(settings, start=1):
        print(f"[sweep] setting {index}/{len(settings)}: {setting['setting_id']}", flush=True)
        run_setting(args, suite_dir, setting, summary_path)

    print(f"[sweep] done summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
