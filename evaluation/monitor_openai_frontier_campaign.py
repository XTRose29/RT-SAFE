#!/usr/bin/env python3
"""Append compact health snapshots for a running frontier rollout campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def process_count(pattern: str) -> int:
    result = subprocess.run(
        ["pgrep", "-u", str(os.getuid()), "-f", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def live_process_counts(campaign_root: Path) -> dict[str, int]:
    """Count campaign processes without matching this monitor or zombies."""
    root_text = str(campaign_root.resolve())
    workers = 0
    renderers = 0
    watchdogs = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            state = raw[raw.rfind(")") + 2 :].split()[0]
            command = " ".join(
                item.decode(errors="replace")
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            )
        except (OSError, IndexError):
            continue
        if not command or state == "Z":
            continue
        if (
            "run_openai_frontier_matrix.py" in command
            and "--mode worker" in command
            and root_text in command
        ):
            workers += 1
        if (
            "SimWorld/Binaries/Linux/SimWorld SimWorld" in command
            and "-RenderoffScreen" in command
            and root_text in command
        ):
            renderers += 1
        if (
            "watchdog_openai_frontier_campaign.py" in command
            and root_text in command
        ):
            watchdogs += 1
    return {
        "worker_processes": workers,
        "active_ue_processes": renderers,
        "watchdog_processes": watchdogs,
    }


def live_estimated_cost(model_dir: Path, manifest: dict) -> float:
    """Price validated completed cells without relying on a stale summary."""
    planning = manifest.get("planning") or {}
    input_price = float(planning.get("input_price_usd_per_million") or 0)
    output_price = float(planning.get("output_price_usd_per_million") or 0)
    prompt_tokens = 0
    completion_tokens = 0
    for status_path in (model_dir / "cells").glob("**/status.json"):
        status = load_json(status_path)
        if status.get("state") != "completed" or not status.get("result_path"):
            continue
        result_path = Path(str(status["result_path"]))
        if not result_path.is_absolute():
            result_path = status_path.parent / result_path
        result = load_json(result_path)
        prompt_tokens += int(result.get("total_prompt_tokens") or 0)
        completion_tokens += int(result.get("total_completion_tokens") or 0)
    return round(
        (prompt_tokens * input_price + completion_tokens * output_price)
        / 1_000_000,
        6,
    )


def snapshot(root: Path) -> dict:
    models = {}
    total_completed = 0
    total_expected = 0
    total_failures = 0
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest = load_json(model_dir / "suite_manifest.json")
        if not manifest:
            continue
        expected = int(manifest.get("rollout_count") or 0)
        states = Counter()
        for status_path in (model_dir / "cells").glob("**/status.json"):
            state = str(load_json(status_path).get("state") or "unreadable")
            states[state] += 1
        attempts = 0
        failed_attempts = 0
        for event_path in (model_dir / "workers").glob("*/events.jsonl"):
            try:
                lines = event_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") == "attempt_finished":
                    attempts += 1
                    if event.get("validation_issues"):
                        failed_attempts += 1
        completed = states.get("completed", 0)
        total_completed += completed
        total_expected += expected
        total_failures += failed_attempts
        models[manifest.get("model", model_dir.name)] = {
            "completed": completed,
            "expected": expected,
            "states": dict(states),
            "attempts_finished": attempts,
            "failed_attempts": failed_attempts,
            "estimated_cost_usd": live_estimated_cost(model_dir, manifest),
        }
    usage = subprocess.run(
        ["du", "-sb", str(root)],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.split()
    return {
        "time": now(),
        "campaign_root": str(root),
        "models": models,
        "completed": total_completed,
        "expected_started_phases": total_expected,
        "failed_attempts": total_failures,
        "step_manifests": sum(1 for _ in root.glob("**/step_*_manifest.json")),
        "png_files": sum(1 for _ in root.glob("**/*.png")),
        "disk_bytes": int(usage[0]) if usage else None,
        "campaign_processes": process_count("run_openai_frontier_matrix.py"),
        "ue_processes": process_count(
            "SimWorld-Base20260313/Linux-runtime/SimWorld"
        ),
        **live_process_counts(root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=8 * 60 * 60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "monitor.jsonl"
    deadline = time.monotonic() + args.duration
    while True:
        value = snapshot(root)
        line = json.dumps(value, ensure_ascii=False)
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        if args.once or time.monotonic() >= deadline:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
