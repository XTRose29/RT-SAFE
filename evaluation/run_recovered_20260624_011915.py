#!/usr/bin/env python3
"""Rerun the 2026-06-24 01:19:15 Qwen3-VL-8B safety-prompt setting.

This runner intentionally patches the current process to recover the old
experiment behavior without reverting the working tree:
- old June 24 safety-preview prompt: preview text is shown to the VLM
- old termination loop: stop only on success, explicit failure, or max_steps
- same task/env/model metadata as results/20260624_011915/task_1.json
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


OLD_USER_PROMPT = """This is step {step_num} of your delivery task. 
You are now at {current_position}, with speed {speed} cm/s and direction {direction}. 
Your current subgoal is {subgoal}. 
Distance to the next subgoal: {relative_distance:.2f} cm. 
Relative angle to the next subgoal: {relative_angle:.2f} degrees 
(negative means the subgoal is to your right, positive means to your left). 
Time spent so far: {time_spent}.

You have the following history of actions, feedback, and observations:
{history}

{image_description}

{action_safety_context}

Based on all the information above, decide the next action.
Remember that longer movements and waiting increase exposure time to moving obstacles.
Use visual perception as the primary basis, but use the safety preview to verify whether a barrier-feasible path action exists. A route_corridor entry means that waypoint leaves the planned sidewalk/crosswalk corridor and should be treated as unsafe unless it is correcting an already off-route position.
Choose progress among actions that pass the safety barrier; if none pass, choose the action that best increases clearance while preserving progress, including turn/wait only when staying still is safer than moving.

Output your decision in the required format above.
"""


def old_system_prompt(realtime_thinking: bool = True) -> str:
    base_intro = (
        "You are a delivery agent in a dynamic city environment (unit: centimeter). "
        "Your goal is to deliver a package to the final destination within the time limit "
        "by reaching subgoals sequentially. The environment contains static and moving obstacles "
        "(e.g., pedestrians, robots, balls). You must navigate using sidewalks and crosswalks, "
        "follow traffic lights, and avoid collisions."
    )
    realtime_reminder = (
        " Time is critical. The environment continues to evolve during both your reasoning "
        "and action execution. Longer reasoning or longer movement increases uncertainty "
        "and exposure to moving obstacles. You must balance safety, progress toward the subgoal, "
        "and temporal risk. Prefer shorter and safer actions when the environment is highly dynamic."
    )
    closing = (
        " At each step, choose an action that is safe, efficient, and temporally robust. "
        "Use the provided online safety/progress preview as a barrier-style action filter, "
        "while using the image to catch visible risks the location data may miss."
    )
    intro = base_intro + (realtime_reminder if realtime_thinking else "") + closing

    return f"""{intro}

Your task is to reach the current subgoal (the next waypoint on your path).
You will be given the relative distance and angle to the next subgoal.

Before choosing an action, briefly consider:
1. Per-obstacle safety: estimate risk for each visible obstacle and use the worst obstacle as the barrier value for each candidate action.
2. Exposure time: include expected reasoning latency plus action duration; longer moves increase uncertainty and risk.
3. Safe path existence: barrier-feasible means clear of obstacles and still inside the planned sidewalk/crosswalk route corridor. If no barrier-feasible progressing action exists, compare the best cautious progress action with turn/wait; do not freeze if staying still is also unsafe.
4. Progress toward the subgoal among actions that pass the safety barrier, or the least risky cautious progress action when all options are imperfect.
5. Compare your final choice with the provided best safe-progress action and explain any mismatch in one short clause.

Action space (exactly one action per step):

1. move_to - Move to one of 7 waypoints.
Red points (1-7) in the image:
- Point 1: 100cm / 0°
- Points 2-4: 200cm (0° / ±45°)
- Points 5-7: 400cm (0° / ±45°)
Duration depends on the distance and your speed.
Param: 1, 2, 3, 4, 5, 6, or 7

2. turn_around - Turn left or right by 30, 60, or 90 degrees.
Duration is approximately 1 second.
Param: L30, L60, L90, R30, R60, or R90

3. wait - Wait in place.
Duration equals the parameter value (1, 2, or 3 seconds).
Param: 1, 2, or 3

Reasoning must be concise and focus on per-obstacle safety, temporal exposure, and progress.

You must respond with this exact format:
Reasoning: [brief explanation]
Action: [move_to / turn_around / wait]
Param: [required value]
"""


def patch_runtime() -> None:
    import base.rt_agent as rt_agent_mod
    import manager.world_manager as world_manager_mod
    from llm.prompt import (
        get_image_description,
        get_system_prompt_adaptive,
        get_system_prompt_future_state,
        get_system_prompt_naive,
    )

    def recovered_plan(self, observation, mode):
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

        user_prompt = OLD_USER_PROMPT.format(
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

        if self.prompt_style == "naive":
            system_prompt = get_system_prompt_naive(self.realtime_thinking)
        elif self.prompt_style == "adaptive":
            system_prompt = get_system_prompt_adaptive(self.realtime_thinking)
        elif self.prompt_style == "future_state":
            system_prompt = get_system_prompt_future_state(self.realtime_thinking)
        else:
            system_prompt = old_system_prompt(self.realtime_thinking)

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

    rt_agent_mod.RTAgent.plan = recovered_plan
    world_manager_mod.WorldManager.run = recovered_run


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def result_in(run_dir: Path) -> Path | None:
    paths = sorted(run_dir.glob("task_*.json"))
    return paths[-1] if paths else None


def summarize(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = read_json(path)
    keys = [
        "success",
        "failed",
        "stuck",
        "stuck_reason",
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
        "llm_extra_body",
    ]
    return {key: data.get(key) for key in keys}


def run_round(args: argparse.Namespace, suite_dir: Path, round_id: int) -> dict[str, Any]:
    from base.rt_communicator import RTCommunicator
    from base.rt_unrealcv import RTUnrealCV
    from manager.world_manager import WorldManager

    run_id = f"round_{round_id:02d}"
    run_dir = suite_dir / "runs" / run_id
    config_path = suite_dir / "configs" / f"{run_id}.agents.json"
    llm_config = {
        "model": args.model,
        "provider": args.provider,
        "url": args.url,
        "reasoning": args.reasoning,
    }
    if args.reasoning_effort is not None:
        llm_config["reasoning_effort"] = args.reasoning_effort
    if args.max_tokens is not None:
        llm_config["max_tokens"] = args.max_tokens
    agent_config = {"llm": [llm_config]}
    write_json(config_path, agent_config)

    condition = {
        "round": round_id,
        "source_result": args.source_result,
        "task_index": args.task_index,
        "task_file": args.task_file,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "model": args.model,
        "provider": args.provider,
        "url": args.url,
        "condition_name": args.condition_name,
        "reasoning_enabled": args.reasoning,
        "reasoning_effort": args.reasoning_effort,
        "max_llm_tokens": args.max_tokens,
        "llm_extra_body": None,
        "prompt_style": "instructional",
        "prompt_recovery": "20260624_011915_safety_preview_visible",
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
        {"round": round_id, "status": "running", "started_at": datetime.now().isoformat()},
    )

    communicator = None
    world_manager = None
    try:
        print(f"[recovered] {run_id}: connecting UE {args.ue_ip}:{args.ue_port}", flush=True)
        communicator = RTCommunicator(RTUnrealCV(port=args.ue_port, ip=args.ue_ip))
        print(f"[recovered] {run_id}: creating WorldManager", flush=True)
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
        world_manager._batch_suffix = f"recovered_20260624_011915_{run_id}"
        print(f"[recovered] {run_id}: running task {args.task_index} ({args.difficulty})", flush=True)
        world_manager.run_single_task(args.task_index, difficulty=args.difficulty)
        print(f"[recovered] {run_id}: task run returned", flush=True)
        result_path = result_in(run_dir)
        record = {
            "round": round_id,
            "status": "completed",
            "finished_at": datetime.now().isoformat(),
            "result_path": str(result_path) if result_path else None,
            **summarize(result_path),
        }
    except Exception as exc:
        record = {
            "round": round_id,
            "status": "error",
            "finished_at": datetime.now().isoformat(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if world_manager is not None:
            world_manager.cleanup()
        elif communicator is not None:
            try:
                communicator.clear_agents()
            except Exception:
                pass

    write_json(run_dir / "run_status.json", record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", default="results/20260624_011915/task_1.json")
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--provider", default="self-hosted")
    parser.add_argument("--url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--reasoning", dest="reasoning", action="store_true")
    parser.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.set_defaults(reasoning=True)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--condition-name", default=None)
    parser.add_argument("--stop-after-successes", type=int, default=None)
    parser.add_argument("--ue-ip", default="127.0.0.1")
    parser.add_argument("--ue-port", type=int, default=9006)
    parser.add_argument("--inter-run-sleep", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patch_runtime()

    source = read_json(REPO_ROOT / args.source_result)
    suite_name = args.suite_name or f"recovered_20260624_011915_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        "recovered_behavior": {
            "prompt": "June 24 safety/progress preview visible in VLM user prompt",
            "termination": "success, explicit failure, or max_steps only",
            "llm_extra_body": None,
            "visual_proxy_note": "Original 30006 proxy was unavailable; using same underlying Qwen server at 30001.",
        },
        "rounds": args.rounds,
        "condition_name": args.condition_name,
        "reasoning_enabled": args.reasoning,
        "reasoning_effort": args.reasoning_effort,
        "max_llm_tokens": args.max_tokens,
        "stop_after_successes": args.stop_after_successes,
    }
    write_json(suite_dir / "experiment_manifest.json", manifest)

    summary_path = suite_dir / "summary.jsonl"
    print(f"[recovered] suite={suite_dir}", flush=True)
    for round_id in range(1, args.rounds + 1):
        print(f"[recovered] round {round_id}/{args.rounds}", flush=True)
        record = run_round(args, suite_dir, round_id)
        append_jsonl(summary_path, record)
        print(
            "[recovered] "
            + json.dumps(
                {
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
        if args.inter_run_sleep > 0 and round_id < args.rounds:
            time.sleep(args.inter_run_sleep)
        if (
            args.stop_after_successes is not None
            and args.stop_after_successes > 0
            and len([r for r in [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()] if r.get("success") is True]) >= args.stop_after_successes
        ):
            print(f"[recovered] stop_after_successes reached: {args.stop_after_successes}", flush=True)
            break

    records = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = [r for r in records if r.get("status") == "completed"]
    successes = [r for r in completed if r.get("success") is True]
    aggregate = {
        "completed_rounds": len(completed),
        "successes": len(successes),
        "success_rate": (len(successes) / len(completed)) if completed else None,
        "updated_at": datetime.now().isoformat(),
    }
    write_json(suite_dir / "aggregate_summary.json", aggregate)
    print(f"[recovered] aggregate={json.dumps(aggregate, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
