#!/usr/bin/env python3
"""Merge validated rollout video indexes without copying large video files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("indexes", nargs="+", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    by_task: dict[tuple[int, int], dict] = {}
    failures = []
    examples: dict[str, list[dict]] = {}
    source_indexes = []
    for path in args.indexes:
        resolved = path.resolve()
        source_indexes.append(str(resolved))
        index = json.loads(resolved.read_text(encoding="utf-8"))
        failures.extend(index.get("failures") or [])
        for rollout in index.get("rollouts") or []:
            key = (int(rollout["task_index"]), int(rollout["task_id"]))
            if key in by_task:
                raise ValueError(f"duplicate rollout for task key {key}")
            by_task[key] = rollout
        for category, rows in (
            index.get("unsafe_trigger_video_examples") or {}
        ).items():
            examples.setdefault(category, []).extend(rows)

    rollouts = [by_task[key] for key in sorted(by_task)]
    output = {
        "suite_dir": str(output_dir),
        "source_indexes": source_indexes,
        "completed_video_count": len(rollouts),
        "failures": failures,
        "all_alignment_checks_passed": bool(rollouts)
        and not failures
        and all(item.get("alignment_passed") for item in rollouts),
        "rollouts": rollouts,
        "unsafe_trigger_video_examples": examples,
    }
    output_path = output_dir / "video_index.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "rollouts": len(rollouts),
        "passed": output["all_alignment_checks_passed"],
    }))
    return 0 if output["all_alignment_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
