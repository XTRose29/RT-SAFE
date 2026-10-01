#!/usr/bin/env python3
"""Run one real VAGEN-adapter decision against an already-running UE fleet."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from online_rl.vagen_env import SimWorldRealTimeGymEnv


REPO = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoints",
        default=os.environ.get("SIMWORLD_RL_UE_ENDPOINTS"),
        help="fleet endpoint JSON (defaults to SIMWORLD_RL_UE_ENDPOINTS)",
    )
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--fixed-latency", type=float, default=1.0)
    parser.add_argument("--action", default="Action: wait\nParam: 1")
    return parser.parse_args(argv)


def observation_summary(observation: dict[str, Any]) -> dict[str, Any]:
    images = observation.get("multi_modal_input", {}).get("<image>", [])
    return {
        "prompt_chars": len(observation.get("obs_str", "")),
        "image_placeholders": observation.get("obs_str", "").count("<image>"),
        "images": len(images),
        "image_types": [type(image).__name__ for image in images],
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.endpoints:
        raise RuntimeError("pass --endpoints or set SIMWORLD_RL_UE_ENDPOINTS")
    config = {
        "backend": "unreal",
        "ue_endpoints": args.endpoints,
        "repo_root": str(REPO),
        "map_name": "RT10",
        "task_file": "data/map1_10roads/tasks.json",
        "task_index": args.task_index,
        "difficulty": args.difficulty,
        "max_steps": 2,
        "max_turns": 1,
        "prompt_style": "naive",
        "traffic_policy": "visual_only",
        "realtime_thinking": True,
        "latency_mode": "fixed",
        "fixed_latency_seconds": args.fixed_latency,
        "max_images": 2,
        "observation_width": 720,
        "observation_height": 640,
    }
    env = SimWorldRealTimeGymEnv(config)
    started = time.monotonic()
    try:
        reset_started = time.monotonic()
        observation, reset_info = await env.reset(args.seed)
        reset_seconds = time.monotonic() - reset_started
        system_prompt = await env.system_prompt()
        step_started = time.monotonic()
        next_observation, reward, done, info = await env.step(args.action)
        step_seconds = time.monotonic() - step_started
        return {
            "ok": True,
            "endpoint": {
                key: reset_info.get(key)
                for key in ("endpoint_id", "ue_host", "ue_port", "map_name")
            },
            "task": {
                key: reset_info.get(key)
                for key in ("task_id", "task_index", "difficulty")
            },
            "system_prompt_chars": len(system_prompt["obs_str"]),
            "initial_observation": observation_summary(observation),
            "action": args.action,
            "reward": reward,
            "reward_terms": info["reward_terms"],
            "done": done,
            "termination": info["termination"],
            "state": info["benchmark_state"],
            "next_observation": observation_summary(next_observation),
            "reset_wall_seconds": reset_seconds,
            "step_wall_seconds": step_seconds,
            "total_wall_seconds": time.monotonic() - started,
        }
    finally:
        await env.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = asyncio.run(run(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
