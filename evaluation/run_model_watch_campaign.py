#!/usr/bin/env python3
"""Run the existing RT-SAFE campaign with a frozen model-watch job.

RTSAFE_WATCH_JOB points to a private job.json. Workers inherit this setting.
Planning does not load credentials or start Unreal.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def configure(job):
    from evaluation import run_openai_frontier_matrix as campaign

    campaign.MODEL = job["model"]
    pool = job.get("provider") == "account-pool"
    campaign.PROVIDER = job.get("provider", "openrouter")
    campaign.API_MODE = "chat_completions"
    campaign.SUBSCRIPTION_CLI = False
    campaign.INFERENCE_SURFACE = "account_pool_gateway" if pool else "openrouter_api"
    campaign.BILLING_MODE = "account_pool_subscription" if pool else "api_usage_based"
    campaign.INPUT_PRICE = 0 if pool else job["pricing_per_M"]["input"]
    campaign.OUTPUT_PRICE = 0 if pool else job["pricing_per_M"]["output"]
    campaign.DIFFICULTIES = (job["selection"]["difficulty"],)
    campaign.ENV_MODES = (job["selection"]["timing"],)
    campaign.EFFORTS = tuple(job["efforts"])
    original_tasks = campaign.all_tasks
    original_command = campaign.runner_command
    original_manifest = campaign.planned_manifest
    original_summary = campaign.build_summary
    wanted = {(t["map"], t["task"]): t for t in job["selection"]["tasks"]}

    def tasks(selected=None):
        result = [t for t in original_tasks(selected) if (t.map_name, t.task_id) in wanted]
        if selected is None and len(result) != len(wanted):
            raise ValueError("Frozen task selection differs from this checkout")
        for t in result:
            if abs(t.route_length_m - wanted[t.map_name, t.task_id]["length_m"]) >= .001:
                raise ValueError(f"Frozen route geometry changed: {t.map_name}/{t.task_id}")
        return result

    def command(args, task, condition, attempt, port):
        result = original_command(args, task, condition, attempt, port)
        if pool:
            return result + ["--base-url", job["base_url"]]
        return result + [
            "--input-price-per-million", str(campaign.INPUT_PRICE),
            "--output-price-per-million", str(campaign.OUTPUT_PRICE),
        ]

    def manifest():
        result = original_manifest()
        if pool:
            result.update(base_url=job["base_url"], allow_paid=False,
                          strict_model_transport_comparable_across_surfaces=False,
                          comparison_note="Account gateway pilot with explicit model routes; report separately from historical transports and execution protocols.")
            result["planning"].update(all_caps_estimate_usd=None,
                                      input_price_usd_per_million=None, output_price_usd_per_million=None)
        return result

    def summary(*args, **kwargs):
        result = original_summary(*args, **kwargs)
        if pool:
            result["estimated_api_cost_usd"] = None
            result["reported_api_cost_usd"] = None
            for row in result["conditions"]:
                row["estimated_api_cost_usd"] = None
                row["reported_api_cost_usd"] = None
                row["reasoning_tokens"] = None
                row["visible_output_tokens"] = None
            result["token_accounting_note"] = "Input/output totals are provider-reported; the pool may omit reasoning-token detail."
        return result

    campaign.all_tasks = tasks
    campaign.runner_command = command
    campaign.planned_manifest = manifest
    campaign.build_summary = summary
    # The coordinator launches this adapter for every worker as well.
    campaign.__file__ = __file__
    return campaign


def main(argv=None):
    job = json.loads(Path(os.environ["RTSAFE_WATCH_JOB"]).read_text())
    campaign = configure(job)
    args = campaign.parse_args(argv)
    if args.mode == "plan":
        print(json.dumps(campaign.planned_manifest(), indent=2))
        return 0
    if args.mode == "summary":
        summary = campaign.write_summary(args.suite_dir.resolve(), validate_completed=True)
        print(json.dumps(summary, indent=2))
        return 0
    if args.mode == "worker":
        return campaign.worker_main(args)
    if args.mode == "coordinator":
        return campaign.coordinator_main(args)
    raise ValueError("Start this campaign through tools/model_watch.py start JOB")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
