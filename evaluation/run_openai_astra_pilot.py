#!/usr/bin/env python3
"""Run the paired GPT-6 Astra five-map SimWorld safety pilot.

The pilot holds the prior GPT-5.6 Luna instructional protocol fixed and runs
one canonical task from each map in easy/hard x realtime/static conditions.
The identical 20 task-condition cells are evaluated at low and high reasoning,
for 40 paired rollouts total. Every step retains its input image, post-action
image, model decision, timing, token usage, and safety telemetry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from evaluation import run_openai_luna_all36 as campaign


MODEL = "gpt-6-astra"
INPUT_PRICE = 10.00
OUTPUT_PRICE = 50.00
TASK_IDS = (0, 8, 16, 24, 32)
OBSERVATION_WIDTH = 1080
OBSERVATION_HEIGHT = 960
UE_WIDTH = 1920
UE_HEIGHT = 1080
DEFAULT_SUITE = Path(
    "results/"
    "openai_gpt6_astra_5map_paired_low_high_1080x960_20260907"
)

_original_all_tasks = campaign.all_tasks
_original_planned_manifest = campaign.planned_manifest
_original_runner_command = campaign.runner_command


def all_conditions() -> list[campaign.Condition]:
    """Return the eight paired timing/difficulty/reasoning conditions."""
    result: list[campaign.Condition] = []
    for difficulty in ("easy", "hard"):
        for env_mode in ("realtime", "static"):
            for effort in ("low", "high"):
                result.append(
                    campaign.Condition(
                        len(result),
                        difficulty,
                        env_mode,
                        "instructional",
                        "instructional",
                        effort,
                    )
                )
    return result


def all_tasks(selected: set[int] | None = None) -> list[campaign.TaskSpec]:
    """Select one stable task per map, optionally narrowed by task ID."""
    requested = set(TASK_IDS) if selected is None else set(selected)
    unknown = requested - set(TASK_IDS)
    if unknown:
        raise ValueError(
            f"GPT-6 pilot only permits canonical task IDs {list(TASK_IDS)}; "
            f"received {sorted(unknown)}"
        )
    return _original_all_tasks(requested)


def planned_manifest() -> dict[str, Any]:
    """Annotate the inherited Luna campaign plan for this GPT-6 pilot."""
    manifest = _original_planned_manifest()
    manifest.update(
        {
            "schema_version": 2,
            "experiment": "gpt6_astra_5map_paired_low_high",
            "model": MODEL,
            "reasoning_arms": {
                "low": "explicit reasoning.effort=low",
                "high": "explicit reasoning.effort=high",
            },
            "task_selection": {
                "policy": "one canonical task per map",
                "task_ids": list(TASK_IDS),
            },
            "artifact_policy": {
                "record_per_step": True,
                "retain_all_input_images": True,
                "retain_all_post_action_images": True,
                "retain_decision_manifest": True,
            },
            "planning": {
                "basis": (
                    "matched GPT-5.6 Luna low-effort task IDs 0,8,16,24,32 "
                    "plus a conservative GPT-6 high-effort allowance"
                ),
                "historical_low_arm": {
                    "rollouts": 20,
                    "decisions": 934,
                    "input_tokens": 1_568_785,
                    "billed_output_tokens": 161_468,
                    "gpt6_standard_price_estimate_usd": 23.76125,
                },
                "expected_scenario": {
                    "decisions_per_arm": 1_000,
                    "input_tokens_per_decision": 2_000,
                    "low_billed_output_tokens_per_decision": 256,
                    "high_billed_output_tokens_per_decision": 1_024,
                    "standard_price_estimate_usd": 104.0,
                    "all_input_at_cache_write_rate_estimate_usd": 114.0,
                },
                "route_cap_scenario": {
                    "decisions_per_arm": 5_172,
                    "standard_price_estimate_usd": 537.888,
                    "all_input_at_cache_write_rate_estimate_usd": 589.608,
                },
                "not_a_hard_ceiling": (
                    "The inherited Luna protocol intentionally leaves "
                    "max_output_tokens unset, so reasoning usage can exceed "
                    "the planning allowance."
                ),
            },
        }
    )
    return manifest


def runner_command(
    args: Any,
    task: campaign.TaskSpec,
    condition: campaign.Condition,
    attempt: Path,
    port: int,
) -> list[str]:
    """Forward the full-image audit policy to every isolated rollout."""
    command = _original_runner_command(args, task, condition, attempt, port)
    command.extend(
        [
            "--record-png-compress-level",
            str(args.record_png_compress_level),
            "--record-images-first-steps",
            "0",
            "--record-images-last-steps",
            "0",
            "--record-output-images",
        ]
    )
    return command


def configure_campaign() -> None:
    """Patch the generic, proven Luna orchestrator with pilot constants."""
    campaign.MODEL = MODEL
    campaign.INPUT_PRICE = INPUT_PRICE
    campaign.OUTPUT_PRICE = OUTPUT_PRICE
    campaign.OBSERVATION_WIDTH = OBSERVATION_WIDTH
    campaign.OBSERVATION_HEIGHT = OBSERVATION_HEIGHT
    campaign.UE_WIDTH = UE_WIDTH
    campaign.UE_HEIGHT = UE_HEIGHT
    campaign.all_conditions = all_conditions
    campaign.all_tasks = all_tasks
    campaign.planned_manifest = planned_manifest
    campaign.runner_command = runner_command
    # coordinator_main uses its module's __file__ for worker subprocesses.
    campaign.__file__ = __file__


def normalized_argv(argv: list[str]) -> list[str]:
    """Apply pilot-safe defaults while preserving explicit operator choices."""
    result = list(argv)
    if "--suite-dir" not in result:
        result = ["--suite-dir", str(DEFAULT_SUITE), *result]
    if "--record-images-first-steps" not in result:
        result.extend(["--record-images-first-steps", "0"])
    if "--record-images-last-steps" not in result:
        result.extend(["--record-images-last-steps", "0"])
    return result


def main(argv: list[str] | None = None) -> int:
    configure_campaign()
    args = campaign.parse_args(normalized_argv(list(argv or [])))
    if args.mode == "plan":
        print(json.dumps(campaign.ensure_suite(args.suite_dir.resolve()), indent=2))
        return 0
    if args.mode == "summary":
        print(
            json.dumps(
                campaign.write_summary(
                    args.suite_dir.resolve(), validate_completed=True
                ),
                indent=2,
            )
        )
        return 0
    return (
        campaign.worker_main(args)
        if args.mode == "worker"
        else campaign.coordinator_main(args)
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
