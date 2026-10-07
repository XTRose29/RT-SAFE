#!/usr/bin/env python3
"""Run paired GPT-5.6 Sol low/high rollouts on easy realtime Task 0.

This narrow campaign intentionally matches the GPT-6 Astra Task 0 protocol:
seed 0, instructional prompt, visual-only traffic policy, image history,
1080x960 observations, and complete per-step input/post-action image capture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from evaluation import run_openai_luna_all36 as campaign


MODEL = "gpt-5.6-sol"
INPUT_PRICE = 4.00
OUTPUT_PRICE = 20.00
TASK_ID = 0
OBSERVATION_WIDTH = 1080
OBSERVATION_HEIGHT = 960
UE_WIDTH = 1920
UE_HEIGHT = 1080
DEFAULT_SUITE = Path(
    "results/"
    "openai_gpt5.6_sol_task0_easy_realtime_low_high_20260907_run01"
)

_original_all_tasks = campaign.all_tasks
_original_planned_manifest = campaign.planned_manifest
_original_runner_command = campaign.runner_command


def all_conditions() -> list[campaign.Condition]:
    """Return the paired easy realtime low/high conditions."""
    return [
        campaign.Condition(
            ordinal,
            "easy",
            "realtime",
            "instructional",
            "instructional",
            effort,
        )
        for ordinal, effort in enumerate(("low", "high"))
    ]


def all_tasks(selected: set[int] | None = None) -> list[campaign.TaskSpec]:
    """Select Task 0 and reject accidental expansion to another task."""
    requested = {TASK_ID} if selected is None else set(selected)
    if requested != {TASK_ID}:
        raise ValueError(f"GPT-5.6 Sol pilot only permits Task {TASK_ID}")
    return _original_all_tasks(requested)


def planned_manifest() -> dict[str, Any]:
    """Describe the paired Sol experiment and full image audit policy."""
    manifest = _original_planned_manifest()
    manifest.update(
        {
            "schema_version": 2,
            "experiment": "gpt5.6_sol_task0_easy_realtime_low_high",
            "model": MODEL,
            "reasoning_arms": {
                "low": "explicit reasoning.effort=low",
                "high": "explicit reasoning.effort=high",
            },
            "task_selection": {
                "policy": "Task 0 only, matched to GPT-6 Astra rollout",
                "task_ids": [TASK_ID],
            },
            "artifact_policy": {
                "record_per_step": True,
                "retain_all_input_images": True,
                "retain_all_post_action_images": True,
                "retain_decision_manifest": True,
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
    """Forward complete step-image retention to the rollout process."""
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
    """Configure the proven campaign coordinator for this paired run."""
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
    campaign.__file__ = __file__


def normalized_argv(argv: list[str]) -> list[str]:
    """Apply full-image defaults while preserving explicit operator choices."""
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
