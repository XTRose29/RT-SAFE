#!/usr/bin/env python3
"""Summarize a Claude Code subscription campaign directory (read-only).

Reports per arm: cells, outcomes, decisions, token usage (input, output,
thinking, cache), Claude Code's API-list-price equivalent (not a bill),
latency, image encodings, and the plan-usage change recorded by the
campaign guard before and after the run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_arm(suite: Path) -> dict[str, Any]:
    cells = outcomes = 0
    states: Counter[str] = Counter()
    results = []
    for status_path in sorted(suite.glob("cells/*/*/task_*/status.json")):
        cells += 1
        status = read_json(status_path)
        states[status.get("state", "unknown")] += 1
        if status.get("state") == "completed":
            results.append(read_json(Path(status["result_path"])))
    decisions = []
    for attempt in sorted(suite.glob("cells/*/*/task_*/attempt_*")):
        for manifest_path in attempt.glob("**/*_manifest.json"):
            metrics = read_json(manifest_path).get("metrics") or {}
            claude = (metrics.get("provider_usage") or {}).get("claude_code") or {}
            decisions.append((metrics, claude))
    walls = [c["latency"]["process_wall_seconds"] for _, c in decisions if c.get("latency")]
    apis = [c["latency"]["duration_api_ms"] / 1000 for _, c in decisions
            if (c.get("latency") or {}).get("duration_api_ms") is not None]
    encodes = [c["latency"]["image_encode_seconds"] for _, c in decisions if c.get("latency")]

    def total(key: str) -> int:
        return sum(int(m.get(key) or 0) for m, _ in decisions)

    outcomes = Counter("success" if r.get("success") else "failure" for r in results)
    return {
        "suite": str(suite),
        "cells": cells,
        "cell_states": dict(states),
        "outcomes": dict(outcomes),
        "decisions": len(decisions),
        "decisions_per_completed_rollout": [int(r.get("decision_count") or 0) for r in results],
        "prompt_tokens": total("prompt_tokens"),
        "output_tokens": total("completion_tokens"),
        "thinking_tokens": total("reasoning_tokens"),
        "cache_read_tokens": sum(int((m.get("provider_usage") or {}).get("cached_tokens") or 0) for m, _ in decisions),
        "api_list_price_equivalent_usd": round(sum(
            float(c.get("api_list_price_equivalent_usd") or 0) for _, c in decisions), 4),
        "served_models": dict(Counter(model for _, c in decisions for model in c.get("served_models") or [])),
        "effective_efforts": dict(Counter(c.get("effective_effort") for _, c in decisions)),
        "image_encodings": dict(Counter(i.get("encoding") for _, c in decisions for i in c.get("images") or [])),
        "rate_limit_statuses": dict(Counter((c.get("rate_limit") or {}).get("status") for _, c in decisions)),
        "api_retries": sum(len(c.get("api_retries") or []) for _, c in decisions),
        "latency_seconds": {
            "process_wall_mean": round(fmean(walls), 2) if walls else None,
            "process_wall_median": round(median(walls), 2) if walls else None,
            "api_mean": round(fmean(apis), 2) if apis else None,
            "image_encode_mean": round(fmean(encodes), 3) if encodes else None,
        },
        "stop_file": read_json(suite / "CLAUDE_CODE_STOP.json") if (suite / "CLAUDE_CODE_STOP.json").exists() else None,
    }


def usage_delta(root: Path) -> dict[str, Any] | None:
    befores = [read_json(path) for path in root.glob("usage_before*.json")]
    if not befores:
        return None
    first = min(befores, key=lambda usage: usage["checked_at"])
    afters = [read_json(path) for path in [*root.glob("usage_after*.json"), root / "latest_usage.json"]
              if path.exists()]
    last = max(afters, key=lambda usage: usage["checked_at"]) if afters else None

    def windows(usage: dict[str, Any]) -> dict[str, float]:
        values = {"five_hour": usage["five_hour"]["utilization"], "seven_day": usage["seven_day"]["utilization"]}
        values |= {f"scoped:{e['display_name']}": e["utilization"] for e in usage.get("model_scoped") or []}
        return values

    result = {"before": windows(first), "before_checked_at": first.get("checked_at")}
    if last:
        result |= {
            "after": windows(last),
            "after_checked_at": last.get("checked_at"),
            "delta_percentage_points": {
                key: round(value - windows(first).get(key, 0.0), 2) for key, value in windows(last).items()
            },
            "usage_credit_minor_units_spent": (
                last["usage_credits"]["used_minor"] - first["usage_credits"]["used_minor"]
            ),
            "note": "Plan windows are shared with every other Claude session on the account.",
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    args = parser.parse_args()
    root = args.results_root.resolve()
    summary = {
        "results_root": str(root),
        "arms": {suite.name: summarize_arm(suite) for suite in sorted(root.iterdir())
                 if suite.is_dir() and (suite / "cells").is_dir()},
        "plan_usage": usage_delta(root),
    }
    (root / "campaign_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
