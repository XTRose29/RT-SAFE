#!/usr/bin/env python3
"""Reasoning-budget and prompt sweep using the recovered June 24 prompt.

Fixed setting:
- task 1, seed 0, easy difficulty by default
- recovered 20260624_011915 basic safety/progress-preview prompt
- old recovered termination: success, explicit failure, or max_steps

Varied axes:
- model
- reasoning effort / max-token budget
- prompt variant: basic plus collision-avoidance variants from run_recovered_prompt_sweep
- environment mode: static vs realtime thinking
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
    OLD_USER_PROMPT,
    old_system_prompt,
    read_json,
    result_in,
    summarize,
    write_json,
)
from evaluation.run_recovered_prompt_sweep import (  # noqa: E402
    PROMPT_VARIANTS,
    build_system_prompt,
    build_user_prompt,
    score_record,
)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_budget(spec: str) -> dict[str, Any]:
    """Parse effort:max_tokens or name:effort:max_tokens."""

    parts = spec.split(":")
    if len(parts) == 2:
        effort, max_tokens = parts
        name = effort
    elif len(parts) == 3:
        name, effort, max_tokens = parts
    else:
        raise ValueError(f"Invalid budget spec {spec!r}; use effort:max_tokens or name:effort:max_tokens")
    return {
        "name": name,
        "reasoning_effort": None if effort in {"none", "default", ""} else effort,
        "max_tokens": None if max_tokens in {"none", "default", ""} else int(max_tokens),
    }


ENV_MODES = {
    "static": False,
    "realtime": True,
}


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def variant_system_prompt(realtime_thinking: bool, variant: str) -> str:
    if variant == "basic":
        return old_system_prompt(realtime_thinking)
    return build_system_prompt(realtime_thinking, variant)


def variant_user_prompt(variant: str) -> str:
    if variant == "basic":
        return OLD_USER_PROMPT
    return build_user_prompt(OLD_USER_PROMPT, variant)


def add_qwen_thinking_extra_body(llm_config: dict[str, Any], reasoning: bool) -> None:
    model = str(llm_config.get("model", "")).lower()
    provider = llm_config.get("provider")
    if "qwen3" not in model or provider not in {"self-hosted", "dashscope"}:
        return
    extra_body = copy.deepcopy(llm_config.get("extra_body") or {})
    chat_template_kwargs = copy.deepcopy(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs["enable_thinking"] = bool(reasoning)
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    llm_config["extra_body"] = extra_body


def patch_runtime_for_variant(variant: str) -> None:
    import base.rt_agent as rt_agent_mod
    import manager.world_manager as world_manager_mod
    from llm.prompt import get_image_description

    def recovered_static_plan(self, observation, mode):
        self.decision_count += 1

        if mode == "human":
            action, time_cost = self.get_human_action(observation)
            return action, time_cost, 0, None, None

        if mode != "llm":
            return None, 0, 0, None, None

        relative_distance = self.position.distance(self.current_destination)
        current_yaw_rad = math.radians(self.yaw)
        dx = self.current_destination.x - self.position.x
        dy = self.current_destination.y - self.position.y
        target_yaw_rad = math.atan2(dy, dx)
        relative_angle = math.degrees(target_yaw_rad - current_yaw_rad)
        if relative_angle > 180:
            relative_angle -= 360
        elif relative_angle < -180:
            relative_angle += 360
        relative_angle = -relative_angle

        expected_latency = self._estimate_reasoning_latency()
        action_safety_preview = self.evaluator.build_action_safety_context(
            observation["waypoints"],
            expected_latency=expected_latency,
        )
        action_safety_context = self.evaluator.format_action_safety_context(
            action_safety_preview
        )
        safe_trajectory_plan = self.evaluator.build_safe_trajectory_plan(
            expected_latency=expected_latency
        )
        safe_trajectory_context = self.evaluator.format_safe_trajectory_plan(
            safe_trajectory_plan
        )
        safe_trajectory_record = dict(safe_trajectory_plan)
        safe_trajectory_record["step_num"] = self.step_num
        self.safe_trajectory_history.append(safe_trajectory_record)

        user_prompt = variant_user_prompt(variant).format(
            step_num=self.step_num,
            current_position=self.position,
            speed=self.speed,
            direction=self.direction,
            subgoal=self.current_destination,
            relative_distance=relative_distance,
            relative_angle=relative_angle,
            required_time=self.required_time,
            time_spent=self.sim_time_elapsed,
            history="\n".join(self.text_history),
            image_description=get_image_description(
                self.use_action_frames, is_first_step=(self.step_num == 0)
            ),
            action_safety_context=action_safety_context,
        )
        system_prompt = variant_system_prompt(self.realtime_thinking, variant)

        if self.use_action_frames and self.action_frames:
            images = [img for img in self.action_frames if img is not None]
            images.append(observation["ego_view"])
        elif self.use_action_frames:
            images = [observation["ego_view"]]
        else:
            images = [img for img in self.image_history if img is not None]
            images.append(observation["ego_view"])

        input_images_for_record = [img for img in images if img is not None]
        (
            action,
            time_cost,
            char_count,
            full_response,
            total_tokens,
            completion_tokens,
            reasoning_tokens,
        ) = self.llm.generate_response_openai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
        )
        self.total_char_count += char_count
        if total_tokens is not None:
            self.total_token_count += total_tokens
        if completion_tokens is not None:
            self.total_completion_tokens += completion_tokens
        if reasoning_tokens is not None:
            self.total_reasoning_tokens += reasoning_tokens

        prompt_data = {
            "decision_index": self.decision_count,
            "prompt_variant": variant,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "full_response": full_response,
            "input_images": input_images_for_record,
            "action_safety_preview": action_safety_preview,
            "internal_action_safety_context": action_safety_context,
            "safe_trajectory_plan": safe_trajectory_plan,
            "safe_trajectory_context": safe_trajectory_context,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "char_count": char_count,
            "response_time": time_cost,
        }

        if action is None:
            self.parse_error_count += 1
            return None, time_cost, char_count, completion_tokens, prompt_data
        return action, time_cost, char_count, completion_tokens, prompt_data

    def recovered_run(self):
        try:
            while self.agent.step_num < self.max_steps:
                if hasattr(self.agent, "success") and self.agent.success:
                    if self.control_mode == "human" and hasattr(self.agent, "human_interface"):
                        self.agent.human_interface.task_completed()
                    break
                if getattr(self.agent, "failed", False):
                    break
                self.agent.step(mode=self.control_mode)
        except Exception as exc:
            self.logger.error(f"Error during agent step: {exc}")
            self.logger.error(traceback.format_exc())
        finally:
            self.save_evaluation_data()

    rt_agent_mod.RTAgent.plan = recovered_static_plan
    world_manager_mod.WorldManager.run = recovered_run


def condition_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("env_mode", "static")),
        str(record.get("model")),
        str(record.get("budget_name")),
        str(record.get("prompt_variant")),
    )


def run_attempt(
    args: argparse.Namespace,
    suite_dir: Path,
    model: str,
    budget: dict[str, Any],
    variant: str,
    env_mode: str,
    round_id: int,
) -> dict[str, Any]:
    from base.rt_communicator import RTCommunicator
    from base.rt_unrealcv import RTUnrealCV
    from manager.world_manager import WorldManager

    patch_runtime_for_variant(variant)
    if not args.set_game_speed:
        RTCommunicator.set_game_speed = lambda self, scale: None

    realtime_thinking = ENV_MODES[env_mode]
    safe_model = model.replace("/", "_")
    condition_id = f"{env_mode}__{safe_model}__{budget['name']}__{variant}__round_{round_id:02d}"
    run_dir = suite_dir / "runs" / env_mode / safe_model / budget["name"] / variant / f"round_{round_id:02d}"
    config_path = suite_dir / "configs" / f"{condition_id}.agents.json"
    llm_config = {
        "model": model,
        "provider": args.provider,
        "url": args.url,
        "reasoning": True,
    }
    if budget["reasoning_effort"] is not None:
        llm_config["reasoning_effort"] = budget["reasoning_effort"]
    if budget["max_tokens"] is not None:
        llm_config["max_tokens"] = budget["max_tokens"]
    if args.qwen_thinking_extra_body == "auto":
        add_qwen_thinking_extra_body(llm_config, reasoning=True)
    agent_config = {"llm": [llm_config]}
    write_json(config_path, agent_config)

    condition = {
        "condition_id": condition_id,
        "env_mode": env_mode,
        "model": model,
        "budget_name": budget["name"],
        "reasoning_effort": budget["reasoning_effort"],
        "max_llm_tokens": budget["max_tokens"],
        "prompt_variant": variant,
        "round": round_id,
        "source_result": args.source_result,
        "task_index": args.task_index,
        "task_file": args.task_file,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "provider": args.provider,
        "url": args.url,
        "reasoning_enabled": True,
        "llm_extra_body": llm_config.get("extra_body"),
        "prompt_recovery": "20260624_011915_safety_preview_visible",
        "prompt_variant_description": None if variant == "basic" else PROMPT_VARIANTS[variant],
        "termination_recovery": "20260624_max_steps_success_or_failed_only",
        "token_based": False,
        "use_tick": args.use_tick,
        "set_game_speed": args.set_game_speed,
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
            use_tick=args.use_tick,
            realtime_thinking=realtime_thinking,
            record_per_step=True,
            use_action_frames=True,
            prompt_style="instructional",
            max_steps=args.max_steps,
            results_dir=str(run_dir.relative_to(REPO_ROOT)),
        )
        world_manager._batch_suffix = f"reasoning_prompt_budget_{condition_id}"
        world_manager.run_single_task(args.task_index, difficulty=args.difficulty)
        result_path = result_in(run_dir)
        record = {
            "condition_id": condition_id,
            "env_mode": env_mode,
            "model": model,
            "budget_name": budget["name"],
            "reasoning_effort": budget["reasoning_effort"],
            "max_llm_tokens": budget["max_tokens"],
            "prompt_variant": variant,
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
            "model": model,
            "budget_name": budget["name"],
            "reasoning_effort": budget["reasoning_effort"],
            "max_llm_tokens": budget["max_tokens"],
            "prompt_variant": variant,
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
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(condition_key(record), []).append(record)

    conditions: dict[str, Any] = {}
    best_success: dict[str, Any] | None = None
    for key, rows in sorted(groups.items()):
        completed = [r for r in rows if r.get("status") == "completed"]
        successes = [r for r in completed if r.get("success") is True]
        selected = min(successes, key=score_record) if successes else None
        if selected is not None and (best_success is None or score_record(selected) < score_record(best_success)):
            best_success = selected
        label = "__".join(key)
        conditions[label] = {
            "env_mode": key[0],
            "model": key[1],
            "budget_name": key[2],
            "prompt_variant": key[3],
            "attempts": len(rows),
            "completed": len(completed),
            "successes": len(successes),
            "success_rate_completed": (len(successes) / len(completed)) if completed else None,
            "selected_success_round": selected.get("round") if selected else None,
            "selected_success_result_path": selected.get("result_path") if selected else None,
            "selected_success_metrics": {
                metric: selected.get(metric) if selected else None
                for metric in [
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

    return {
        "updated_at": datetime.now().isoformat(),
        "env_modes": args.env_modes,
        "target_successes": args.target_successes,
        "target_collision_count": args.target_collision_count,
        "max_rounds": args.max_rounds,
        "conditions": conditions,
        "best_success": {
            "model": best_success.get("model") if best_success else None,
            "budget_name": best_success.get("budget_name") if best_success else None,
            "prompt_variant": best_success.get("prompt_variant") if best_success else None,
            "round": best_success.get("round") if best_success else None,
            "result_path": best_success.get("result_path") if best_success else None,
            "metrics": {
                metric: best_success.get(metric) if best_success else None
                for metric in [
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
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", default="results/20260624_011915/task_1.json")
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen3-VL-8B-Thinking"])
    parser.add_argument("--budgets", nargs="+", default=["low:128", "medium:256", "high:512"])
    parser.add_argument("--env-modes", nargs="+", default=["static"], choices=sorted(ENV_MODES))
    parser.add_argument(
        "--prompt-variants",
        nargs="+",
        default=["basic", "future_clearance", "guarded_recovery"],
        choices=["basic", *sorted(PROMPT_VARIANTS)],
    )
    parser.add_argument("--target-successes", type=int, default=1)
    parser.add_argument("--target-collision-count", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--provider", default="self-hosted")
    parser.add_argument("--url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--qwen-thinking-extra-body", default="auto", choices=["auto", "off"])
    parser.add_argument("--use-tick", type=parse_bool, default=True)
    parser.add_argument("--set-game-speed", type=parse_bool, default=True)
    parser.add_argument("--ue-ip", default="127.0.0.1")
    parser.add_argument("--ue-port", type=int, default=9006)
    parser.add_argument("--inter-run-sleep", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    budgets = [parse_budget(spec) for spec in args.budgets]

    source = read_json(REPO_ROOT / args.source_result)
    suite_name = args.suite_name or f"static_reasoning_prompt_budget_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            "collision_count": source.get("collision_count"),
            "passive_collision_count": source.get("passive_collision_count"),
            "red_light_violations_count": source.get("red_light_violations_count"),
        },
        "fixed_settings": {
            "task_index": args.task_index,
            "seed": args.seed,
            "difficulty": args.difficulty,
            "provider": args.provider,
            "url": args.url,
            "prompt_base": "June 24 safety/progress preview visible in VLM user prompt",
            "termination": "success, explicit failure, or max_steps only",
            "token_based": False,
            "use_tick": args.use_tick,
            "set_game_speed": args.set_game_speed,
            "env_modes": args.env_modes,
            "use_action_frames": True,
            "record_per_step": True,
            "qwen_thinking_extra_body": args.qwen_thinking_extra_body,
        },
        "models": args.models,
        "budgets": budgets,
        "prompt_variants": {
            name: None if name == "basic" else PROMPT_VARIANTS[name]
            for name in args.prompt_variants
        },
        "target_successes": args.target_successes,
        "target_collision_count": args.target_collision_count,
        "max_rounds": args.max_rounds,
    }
    write_json(suite_dir / "experiment_manifest.json", manifest)

    summary_path = suite_dir / "summary.jsonl"
    records = load_jsonl(summary_path)
    print(f"[reasoning-prompt-budget-sweep] suite={suite_dir}", flush=True)
    for env_mode in args.env_modes:
        for model in args.models:
            for budget in budgets:
                for variant in args.prompt_variants:
                    existing = [
                        r
                        for r in records
                        if r.get("env_mode", "static") == env_mode
                        and r.get("model") == model
                        and r.get("budget_name") == budget["name"]
                        and r.get("prompt_variant") == variant
                    ]
                    successes = [
                        r
                        for r in existing
                        if r.get("status") == "completed" and r.get("success") is True
                    ]
                    print(
                        "[reasoning-prompt-budget-sweep] condition="
                        + json.dumps(
                            {
                                "env_mode": env_mode,
                                "model": model,
                                "budget": budget,
                                "prompt_variant": variant,
                                "existing_attempts": len(existing),
                                "existing_successes": len(successes),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if (
                        len(successes) >= args.target_successes
                        and min(score_record(r) for r in successes)[0] <= args.target_collision_count
                    ):
                        continue

                    for round_id in range(len(existing) + 1, args.max_rounds + 1):
                        record = run_attempt(args, suite_dir, model, budget, variant, env_mode, round_id)
                        records.append(record)
                        append_jsonl(summary_path, record)
                        print(
                            "[reasoning-prompt-budget-sweep] "
                            + json.dumps(
                                {
                                    "condition_id": record.get("condition_id"),
                                    "env_mode": record.get("env_mode"),
                                    "status": record.get("status"),
                                    "success": record.get("success"),
                                    "failed": record.get("failed"),
                                    "stuck": record.get("stuck"),
                                    "final_step": record.get("final_step"),
                                    "decision_count": record.get("decision_count"),
                                    "collisions": record.get("collision_count"),
                                    "passive_collisions": record.get("passive_collision_count"),
                                    "red_lights": record.get("red_light_violations_count"),
                                    "avg_response_time": record.get("avg_response_time"),
                                    "avg_completion_tokens": record.get("avg_completion_tokens"),
                                    "avg_reasoning_tokens": record.get("avg_reasoning_tokens"),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        successes = [
                            r
                            for r in records
                            if r.get("env_mode", "static") == env_mode
                            and r.get("model") == model
                            and r.get("budget_name") == budget["name"]
                            and r.get("prompt_variant") == variant
                            and r.get("status") == "completed"
                            and r.get("success") is True
                        ]
                        if (
                            len(successes) >= args.target_successes
                            and min(score_record(r) for r in successes)[0] <= args.target_collision_count
                        ):
                            break
                        if round_id < args.max_rounds and args.inter_run_sleep > 0:
                            time.sleep(args.inter_run_sleep)

    agg = aggregate(records, args)
    write_json(suite_dir / "aggregate_summary.json", agg)
    print(f"[static-reasoning-sweep] aggregate={json.dumps(agg, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
