#!/usr/bin/env python3
"""Run the June 21 prompt with a static-thinking Qwen3-VL Thinking model."""

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


JUNE21_USER_PROMPT = """This is step {step_num} of your delivery task. 
You are now at {current_position}, with speed {speed} cm/s and direction {direction}. 
Your current subgoal is {subgoal}. 
Distance to the next subgoal: {relative_distance:.2f} cm. 
Relative angle to the next subgoal: {relative_angle:.2f} degrees 
(negative means the subgoal is to your right, positive means to your left). 
Time spent so far: {time_spent}.

You have the following history of actions, feedback, and observations:
{history}

{image_description}

Based on all the information above, decide the next action.
Remember that longer movements and waiting increase exposure time to moving obstacles.
Use visual perception as the primary basis and evaluate safety, temporal risk, and efficiency.

Output your decision in the required format above.
"""


def june21_system_prompt(_realtime_thinking: bool = True) -> str:
    return """You are a delivery agent in a dynamic city environment (unit: centimeter). Your goal is to deliver a package to the final destination within the time limit by reaching subgoals sequentially. The environment contains static and moving obstacles (e.g., pedestrians, robots, balls). You must navigate using sidewalks and crosswalks, follow traffic lights, and avoid collisions. Time is critical. The environment continues to evolve during both your reasoning and action execution. Longer reasoning or longer movement increases uncertainty and exposure to moving obstacles. You must balance safety, progress toward the subgoal, and temporal risk. Prefer shorter and safer actions when the environment is highly dynamic. At each step, choose an action that is safe, efficient, and temporally robust.

Your task is to reach the current subgoal (the next waypoint on your path).
You will be given the relative distance and angle to the next subgoal.

Before choosing an action, briefly consider:
1. Immediate safety (obstacles, traffic lights, moving agents).
2. Exposure time of the action (longer moves increase risk).
3. Expected environment change during execution.
4. Progress toward the subgoal.

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

Reasoning must be concise and focus on safety, temporal exposure, and progress.

You must respond with this exact format:
Reasoning: [brief explanation]
Action: [move_to / turn_around / wait]
Param: [required value]
"""


def patch_runtime(recover_old_termination: bool) -> None:
    import base.rt_agent as rt_agent_mod
    import llm.prompt as prompt_mod

    prompt_mod.USER_PROMPT = JUNE21_USER_PROMPT
    rt_agent_mod.USER_PROMPT = JUNE21_USER_PROMPT
    prompt_mod.get_system_prompt = june21_system_prompt
    rt_agent_mod.get_system_prompt = june21_system_prompt

    if not recover_old_termination:
        return

    import manager.world_manager as world_manager_mod

    def june21_run(self):
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

    world_manager_mod.WorldManager.run = june21_run


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def result_in(run_dir: Path) -> Path | None:
    paths = sorted(run_dir.glob("task_*.json"))
    return paths[-1] if paths else None


def summarize(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = read_json(path)
    keys = [
        "model",
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
        "reasoning_enabled",
        "reasoning_effort",
        "max_llm_tokens",
        "llm_extra_body",
        "realtime_thinking",
        "token_based",
        "use_tick",
        "prompt_style",
        "use_action_frames",
        "max_steps",
        "difficulty",
    ]
    return {key: data.get(key) for key in keys}


def run_once(args: argparse.Namespace, suite_dir: Path) -> dict[str, Any]:
    from base.rt_communicator import RTCommunicator
    from base.rt_unrealcv import RTUnrealCV
    from manager.world_manager import WorldManager

    run_dir = suite_dir / "runs" / "round_01"
    config_path = suite_dir / "configs" / "round_01.agents.json"
    llm_config: dict[str, Any] = {
        "model": args.model,
        "provider": args.provider,
        "url": args.url,
        "reasoning": True,
    }
    if args.reasoning_effort:
        llm_config["reasoning_effort"] = args.reasoning_effort
    if args.max_tokens is not None:
        llm_config["max_tokens"] = args.max_tokens
    write_json(config_path, {"llm": [llm_config]})

    condition = {
        "source_result": args.source_result,
        "prompt_recovery": "20260621_182322_exact_system_and_user_prompt",
        "prompt_note": "Exact June 21 prompt text is used even though env thinking is static.",
        "termination_recovery": "20260621_max_steps_success_or_failed_only"
        if args.recover_old_termination
        else "current_world_manager",
        "task_index": args.task_index,
        "task_file": args.task_file,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "model": args.model,
        "provider": args.provider,
        "url": args.url,
        "reasoning_enabled": True,
        "reasoning_effort": args.reasoning_effort,
        "max_llm_tokens": args.max_tokens,
        "token_based": False,
        "use_tick": True,
        "realtime_thinking": False,
        "use_action_frames": True,
        "record_per_step": True,
        "prompt_style": "instructional",
        "max_steps": args.max_steps,
    }
    write_json(run_dir / "condition.json", condition)
    write_json(
        run_dir / "run_status.json",
        {"status": "running", "started_at": datetime.now().isoformat()},
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
            realtime_thinking=False,
            record_per_step=True,
            use_action_frames=True,
            prompt_style="instructional",
            max_steps=args.max_steps,
            results_dir=str(run_dir.relative_to(REPO_ROOT)),
        )
        world_manager._batch_suffix = "june21_prompt_static_thinking"
        world_manager.run_single_task(args.task_index, difficulty=args.difficulty)
        result_path = result_in(run_dir)
        record = {
            "status": "completed",
            "finished_at": datetime.now().isoformat(),
            "result_path": str(result_path) if result_path else None,
            **summarize(result_path),
        }
    except Exception as exc:
        record = {
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
    with (suite_dir / "summary.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", default="results/20260621_182322/task_1.json")
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--provider", default="self-hosted")
    parser.add_argument("--url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--ue-ip", default="127.0.0.1")
    parser.add_argument("--ue-port", type=int, default=9006)
    parser.add_argument("--recover-old-termination", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patch_runtime(args.recover_old_termination)

    suite_name = args.suite_name or f"qwen3vl8b_thinking_june21_prompt_static_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir = REPO_ROOT / "results" / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    source = read_json(REPO_ROOT / args.source_result)
    write_json(
        suite_dir / "experiment_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "suite_dir": str(suite_dir),
            "source_result": args.source_result,
            "source_result_metadata": {
                "model": source.get("model"),
                "success": source.get("success"),
                "difficulty": source.get("difficulty"),
                "realtime_thinking": source.get("realtime_thinking"),
                "prompt_style": source.get("prompt_style"),
                "use_action_frames": source.get("use_action_frames"),
                "max_steps": source.get("max_steps"),
            },
            "changed_for_this_run": {
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "max_tokens": args.max_tokens,
                "realtime_thinking": False,
            },
        },
    )
    print(f"[june21-static-thinking] suite={suite_dir}", flush=True)
    record = run_once(args, suite_dir)
    print(f"[june21-static-thinking] {json.dumps(record, ensure_ascii=False)}", flush=True)
    time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
