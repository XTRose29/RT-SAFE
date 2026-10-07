#!/usr/bin/env python3
"""Prompt sweep for the recovered 2026-06-24 successful Qwen3-VL-8B setting.

Fixed setting:
- task 1, seed 0, easy difficulty, max_steps=120
- qwen3-vl-8b through the local OpenAI-compatible server
- recovered June 24 safety/progress preview in the user prompt
- recovered old termination: success, explicit failure, or max_steps only
- realtime_thinking=True, token_based=False, use_tick=True

Varied axis:
- prompt variant, with collision-avoidance emphasis.
"""

from __future__ import annotations

import argparse
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


PROMPT_VARIANTS: dict[str, dict[str, str]] = {
    "strict_barrier": {
        "system_suffix": """
Collision-avoidance variant: strict barrier.
- Treat any candidate with safety=0.00, collision risk, or route_corridor violation as forbidden unless every wait/turn/move option is also unsafe.
- Prefer a shorter waypoint with lower progress over a long waypoint if it improves dynamic-obstacle clearance.
- If the best progress action is blocked by a pedestrian or movable object, choose wait 1 second or a small turn to increase clearance before moving.
- Do not justify moving through a 0cm or near-zero-clearance gap for progress.
""",
        "user_suffix": """
Extra collision-avoidance rule for this run:
Do not force progress through pedestrians or moving objects. If the preview marks the highest-progress waypoint unsafe, take the safest barrier-feasible shorter waypoint. If none is barrier-feasible, prefer wait:1 or a small turn over a risky move unless waiting is explicitly worse.
""",
    },
    "future_clearance": {
        "system_suffix": """
Collision-avoidance variant: future clearance.
- Predict obstacle positions after reasoning latency plus action duration.
- Reject a waypoint if a pedestrian, dog, ball, bottle, box, scooter, vehicle, or falling object can enter the swept path before the movement completes.
- Prefer actions with extra future clearance even if they add one or two extra steps to the route.
- At crosswalks, wait for a clearly legal and open crossing instead of cutting through traffic or crowds.
""",
        "user_suffix": """
Extra collision-avoidance rule for this run:
Before choosing a waypoint, imagine the scene after your response latency and after the action finishes. Avoid tight gaps; choose the action that keeps the largest future clearance while still eventually reaching the subgoal.
""",
    },
    "short_step_cautious": {
        "system_suffix": """
Collision-avoidance variant: short-step cautious.
- In crowded scenes, prefer 100cm or 200cm moves over 400cm moves because shorter actions reduce exposure and permit replanning.
- Use wait:1 only to let a visible moving obstacle clear; avoid wait:2/3 unless all short moves are unsafe.
- Use turn_around only when it improves visibility/alignment or recovers from an obstacle/building conflict.
- Continue making progress when a short, barrier-feasible action exists.
""",
        "user_suffix": """
Extra collision-avoidance rule for this run:
Bias toward short safe moves in crowded areas. A 100cm/200cm barrier-feasible waypoint is better than a 400cm tight gap. Keep reaching the destination, but spend extra decisions to avoid contact.
""",
    },
    "guarded_recovery": {
        "system_suffix": """
Collision-avoidance variant: guarded recovery.
- Hard rule: never choose a move_to action with safety=0.00, 0cm/near-zero clearance, or an obstacle already in the swept path.
- If no barrier-feasible progressing move exists, choose wait:1 or a small turn only once or twice to let the scene change.
- If recent history shows two or more non-progress actions, recover by choosing the shortest move_to waypoint with positive safety/clearance, even if progress is modest.
- When recovering, rank by positive clearance first, then route-corridor legality, then progress; do not choose the highest-progress zero-safety move.
- The end goal still matters: keep taking positive-clearance short moves whenever they exist.
""",
        "user_suffix": """
Extra collision-avoidance rule for this run:
Safety=0.00 or 0cm clearance means forbidden, not "least risky." If all progress moves are forbidden, wait:1 or turn briefly. If you have already waited/turned recently, pick the shortest legal waypoint with positive clearance and any progress rather than spinning or forcing a zero-safety gap.
""",
    },
    "full_visible_trace": {
        "system_suffix": """
Reasoning-budget variant: full visible safety trace.
- Use the available reasoning budget to write a complete but action-focused visible trace before the final action.
- Explicitly evaluate traffic-light legality, sidewalk/crosswalk alignment, visible obstacles, safety-preview barriers, temporal exposure, and progress.
- For each plausible move_to candidate, state whether it is allowed or rejected and why.
- Finish with the exact required Action and Param lines so the evaluator can parse the decision.
""",
        "user_suffix": """
Extra reasoning-budget instruction for this run:
Use a fuller visible decision trace than the basic setting. Identify obstacles and legal constraints, compare the safest progressing waypoint against wait/turn alternatives, then output the final action in the required format.
""",
    },
    "answer_first_trace": {
        "system_suffix": """
Reasoning-budget variant: answer-first trace.
- Put the parseable decision first using exactly the required Reasoning, Action, and Param lines.
- After the Param line, add a short Trace section that records the safety comparison and obstacle reasoning.
- Keep the first three lines compact so the action can execute even with long reasoning budgets.
""",
        "user_suffix": """
Extra reasoning-budget instruction for this run:
Start with the final decision in exactly three lines: Reasoning, Action, Param. After Param, add a Trace section comparing obstacle risk, traffic-light/crosswalk legality, safety-preview barriers, temporal exposure, and progress.
""",
    },
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


def build_user_prompt(base_prompt: str, variant: str) -> str:
    suffix = PROMPT_VARIANTS[variant]["user_suffix"].strip()
    return base_prompt.replace(
        "Output your decision in the required format above.",
        f"{suffix}\n\nOutput your decision in the required format above.",
    )


def build_system_prompt(realtime_thinking: bool, variant: str) -> str:
    prompt = old_system_prompt(realtime_thinking)
    suffix = PROMPT_VARIANTS[variant]["system_suffix"].strip()
    return prompt.replace(
        "Reasoning must be concise and focus on per-obstacle safety, temporal exposure, and progress.",
        f"{suffix}\n\nReasoning must be concise and focus on per-obstacle safety, temporal exposure, and progress.",
    )


def patch_runtime_for_variant(variant: str) -> None:
    import base.rt_agent as rt_agent_mod
    import manager.world_manager as world_manager_mod
    from llm.prompt import get_image_description

    def recovered_variant_plan(self, observation, mode):
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

        user_prompt = build_user_prompt(OLD_USER_PROMPT, variant).format(
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
        self.logger.debug(f"Step {self.step_num}, User prompt: {user_prompt}")

        system_prompt = build_system_prompt(self.realtime_thinking, variant)

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

    rt_agent_mod.RTAgent.plan = recovered_variant_plan
    world_manager_mod.WorldManager.run = recovered_run


def score_record(record: dict[str, Any]) -> tuple[int, int, float, int]:
    return (
        int(record.get("collision_count") or 0),
        int(record.get("passive_collision_count") or 0),
        float(record.get("sim_time") or 0.0),
        int(record.get("final_step") or 10**9),
    )


def run_attempt(
    args: argparse.Namespace,
    suite_dir: Path,
    variant: str,
    round_id: int,
) -> dict[str, Any]:
    from base.rt_communicator import RTCommunicator
    from base.rt_unrealcv import RTUnrealCV
    from manager.world_manager import WorldManager

    patch_runtime_for_variant(variant)

    condition_id = f"{variant}_round_{round_id:02d}"
    run_dir = suite_dir / "runs" / variant / f"round_{round_id:02d}"
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
        "prompt_variant": variant,
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
        "prompt_recovery": "20260624_011915_safety_preview_visible",
        "prompt_variant_description": PROMPT_VARIANTS[variant],
        "termination_recovery": "20260624_max_steps_success_or_failed_only",
        "token_based": False,
        "use_tick": True,
        "realtime_thinking": True,
        "use_action_frames": True,
        "record_per_step": True,
        "max_steps": args.max_steps,
    }
    write_json(run_dir / "condition.json", condition)
    write_json(
        run_dir / "run_status.json",
        {
            "condition_id": condition_id,
            "prompt_variant": variant,
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
            realtime_thinking=True,
            record_per_step=True,
            use_action_frames=True,
            prompt_style="instructional",
            max_steps=args.max_steps,
            results_dir=str(run_dir.relative_to(REPO_ROOT)),
        )
        world_manager._batch_suffix = f"recovered_prompt_sweep_{condition_id}"
        world_manager.run_single_task(args.task_index, difficulty=args.difficulty)
        result_path = result_in(run_dir)
        record = {
            "condition_id": condition_id,
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
    by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in args.prompt_variants}
    for record in records:
        by_variant.setdefault(record["prompt_variant"], []).append(record)

    variants: dict[str, Any] = {}
    best_success: dict[str, Any] | None = None
    for variant, rows in by_variant.items():
        completed = [r for r in rows if r.get("status") == "completed"]
        successes = [r for r in completed if r.get("success") is True]
        selected = min(successes, key=score_record) if successes else None
        if selected is not None and (best_success is None or score_record(selected) < score_record(best_success)):
            best_success = selected
        variants[variant] = {
            "attempts": len(rows),
            "completed": len(completed),
            "successes": len(successes),
            "success_rate_completed": (len(successes) / len(completed)) if completed else None,
            "selected_success_round": selected.get("round") if selected else None,
            "selected_success_result_path": selected.get("result_path") if selected else None,
            "selected_success_metrics": {
                key: selected.get(key) if selected else None
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

    return {
        "updated_at": datetime.now().isoformat(),
        "target_successes": args.target_successes,
        "max_rounds": args.max_rounds,
        "target_collision_count": args.target_collision_count,
        "variants": variants,
        "best_success": {
            "prompt_variant": best_success.get("prompt_variant") if best_success else None,
            "round": best_success.get("round") if best_success else None,
            "result_path": best_success.get("result_path") if best_success else None,
            "metrics": {
                key: best_success.get(key) if best_success else None
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
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", default="results/20260624_011915/task_1.json")
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--prompt-variants", nargs="+", default=list(PROMPT_VARIANTS), choices=sorted(PROMPT_VARIANTS))
    parser.add_argument("--target-successes", type=int, default=1)
    parser.add_argument("--target-collision-count", type=int, default=0)
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

    source = read_json(REPO_ROOT / args.source_result)
    suite_name = args.suite_name or f"recovered_prompt_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            "model": args.model,
            "provider": args.provider,
            "url": args.url,
            "prompt_base": "June 24 safety/progress preview visible in VLM user prompt",
            "termination": "success, explicit failure, or max_steps only",
            "llm_extra_body": None,
            "token_based": False,
            "use_tick": True,
            "realtime_thinking": True,
            "use_action_frames": True,
            "record_per_step": True,
        },
        "prompt_variants": {name: PROMPT_VARIANTS[name] for name in args.prompt_variants},
        "target_successes": args.target_successes,
        "target_collision_count": args.target_collision_count,
        "max_rounds": args.max_rounds,
    }
    write_json(suite_dir / "experiment_manifest.json", manifest)

    summary_path = suite_dir / "summary.jsonl"
    records = load_jsonl(summary_path)
    print(f"[prompt-sweep] suite={suite_dir}", flush=True)
    for variant in args.prompt_variants:
        existing = [r for r in records if r.get("prompt_variant") == variant]
        successes = [r for r in existing if r.get("status") == "completed" and r.get("success") is True]
        print(f"[prompt-sweep] variant={variant}", flush=True)
        if len(successes) >= args.target_successes and min(score_record(r) for r in successes)[0] <= args.target_collision_count:
            print(
                "[prompt-sweep] "
                + json.dumps(
                    {
                        "prompt_variant": variant,
                        "status": "skipped_existing_target_met",
                        "attempts": len(existing),
                        "successes": len(successes),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue

        for round_id in range(len(existing) + 1, args.max_rounds + 1):
            record = run_attempt(args, suite_dir, variant, round_id)
            records.append(record)
            append_jsonl(summary_path, record)
            print(
                "[prompt-sweep] "
                + json.dumps(
                    {
                        "prompt_variant": variant,
                        "round": round_id,
                        "status": record.get("status"),
                        "success": record.get("success"),
                        "failed": record.get("failed"),
                        "final_step": record.get("final_step"),
                        "decision_count": record.get("decision_count"),
                        "collisions": record.get("collision_count"),
                        "passive_collisions": record.get("passive_collision_count"),
                        "red_lights": record.get("red_light_violations_count"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

            successes = [
                r
                for r in records
                if r.get("prompt_variant") == variant
                and r.get("status") == "completed"
                and r.get("success") is True
            ]
            if len(successes) >= args.target_successes and min(score_record(r) for r in successes)[0] <= args.target_collision_count:
                break
            if round_id < args.max_rounds and args.inter_run_sleep > 0:
                time.sleep(args.inter_run_sleep)

    agg = aggregate(records, args)
    write_json(suite_dir / "aggregate_summary.json", agg)
    print(f"[prompt-sweep] aggregate={json.dumps(agg, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
