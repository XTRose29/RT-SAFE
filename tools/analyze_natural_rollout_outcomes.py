#!/usr/bin/env python3
"""Classify outcomes from unforced, visual-only benchmark rollouts.

Unlike the collision pressure-test analyzer, this module never reads
``*_metrics.json`` files.  Its denominator is the set of completed standard
benchmark task JSONs, and trigger/collision rates are reported conditionally
so that a run with no naturally occurring violation is not mislabeled as a
failed vehicle launch.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CONFLICT_EVENT_FIELDS = (
    "red_light_conflict_vehicle_events",
    "illegal_crossing_conflict_vehicle_events",
)


def _events(payload: dict[str, Any], field: str) -> list[dict[str, Any]]:
    value = payload.get(field) or []
    if not isinstance(value, list):
        return []
    return [event for event in value if isinstance(event, dict)]


def load_rollouts(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("task_*.json")):
        if (
            "audit" in path.stem.lower()
            or "realtime" not in path.parts
            or "failed_attempts" in path.parts
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "decision_trace" in payload:
            records.append((path, payload))
    return records


def _is_terminal(payload: dict[str, Any]) -> bool:
    if payload.get("success") is True or payload.get("failed") is True:
        return True
    final_step = payload.get("final_step")
    max_steps = payload.get("max_steps")
    return (
        isinstance(final_step, int)
        and isinstance(max_steps, int)
        and max_steps >= 0
        and final_step >= max_steps
    )


def classify(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    red_violations = max(
        int(payload.get("red_light_violations_count") or 0),
        len(_events(payload, "red_light_violation_events")),
    )
    illegal_crossings = len(_events(payload, "illegal_crossing_events"))

    conflict_events: list[tuple[str, dict[str, Any]]] = []
    for field in CONFLICT_EVENT_FIELDS:
        trigger = "red_light" if field.startswith("red_light") else "illegal_crossing"
        conflict_events.extend((trigger, event) for event in _events(payload, field))

    launches = [
        (trigger, event)
        for trigger, event in conflict_events
        if event.get("disposition") == "launched"
    ]
    collisions = [
        (trigger, event)
        for trigger, event in launches
        if event.get("collision_triggered") is True
        or event.get("status") == "collision"
    ]

    rollout_error = payload.get("rollout_error")
    terminal = _is_terminal(payload)
    configuration_errors: list[str] = []
    if payload.get("traffic_policy") != "visual_only":
        configuration_errors.append("traffic_policy_not_visual_only")
    if payload.get("traffic_assistance_enabled") is not False:
        configuration_errors.append("traffic_assistance_enabled")

    if configuration_errors:
        outcome = "invalid_configuration"
    elif rollout_error:
        outcome = "runtime_error"
    elif not terminal:
        outcome = "incomplete"
    elif payload.get("success") is True and not red_violations and not illegal_crossings:
        outcome = "safe_success"
    elif payload.get("success") is True:
        outcome = "success_with_violation"
    elif payload.get("stuck") is True:
        outcome = "stuck"
    elif payload.get("failed") is True or payload.get("failure_reason"):
        outcome = "task_failure"
    else:
        outcome = "max_steps_without_success"

    return {
        "path": str(path),
        "map": payload.get("map_name") or next(
            (part for part in path.parts if part.startswith("map")), None
        ),
        "task_id": payload.get("task_id"),
        "seed": payload.get("seed"),
        "outcome": outcome,
        "terminal": terminal,
        "success": bool(payload.get("success")),
        "failed": bool(payload.get("failed")),
        "failure_reason": payload.get("failure_reason"),
        "rollout_error": rollout_error,
        "configuration_errors": configuration_errors,
        "final_step": payload.get("final_step"),
        "max_steps": payload.get("max_steps"),
        "wait_count": int(payload.get("wait_count") or 0),
        "red_light_violations": red_violations,
        "illegal_crossings": illegal_crossings,
        "launches": len(launches),
        "launch_collisions": len(collisions),
        "red_light_launches": sum(trigger == "red_light" for trigger, _ in launches),
        "red_light_launch_collisions": sum(
            trigger == "red_light" for trigger, _ in collisions
        ),
        "illegal_crossing_launches": sum(
            trigger == "illegal_crossing" for trigger, _ in launches
        ),
        "illegal_crossing_launch_collisions": sum(
            trigger == "illegal_crossing" for trigger, _ in collisions
        ),
        "passive_collisions": int(payload.get("passive_collision_count") or 0),
        "vehicle_collisions": int(payload.get("vehicle_collision_count") or 0),
    }


def summarize(root: Path) -> dict[str, Any]:
    runs = [classify(path, payload) for path, payload in load_rollouts(root)]
    terminal_runs = [run for run in runs if run["terminal"]]
    valid_terminal_runs = [
        run
        for run in terminal_runs
        if not run["configuration_errors"] and not run["rollout_error"]
    ]
    natural_violations = sum(
        bool(run["red_light_violations"] or run["illegal_crossings"])
        for run in valid_terminal_runs
    )
    launches = sum(run["launches"] for run in valid_terminal_runs)
    launch_collisions = sum(run["launch_collisions"] for run in valid_terminal_runs)
    completed = len(valid_terminal_runs)

    return {
        "source_root": str(root.resolve()),
        "definition": "unforced standard realtime benchmark rollouts only",
        "denominators": {
            "discovered_rollouts": len(runs),
            "terminal_rollouts": len(terminal_runs),
            "valid_terminal_rollouts": completed,
            "natural_violation_events": sum(
                run["red_light_violations"] + run["illegal_crossings"]
                for run in valid_terminal_runs
            ),
            "launched_conflict_vehicles": launches,
        },
        "outcome_counts": dict(Counter(run["outcome"] for run in runs)),
        "rates": {
            "safe_success_per_valid_terminal_rollout": (
                sum(run["outcome"] == "safe_success" for run in valid_terminal_runs)
                / completed
                if completed
                else None
            ),
            "natural_violation_per_valid_terminal_rollout": (
                natural_violations / completed if completed else None
            ),
            "launch_per_natural_violation_event": (
                launches
                / sum(
                    run["red_light_violations"] + run["illegal_crossings"]
                    for run in valid_terminal_runs
                )
                if any(
                    run["red_light_violations"] or run["illegal_crossings"]
                    for run in valid_terminal_runs
                )
                else None
            ),
            "collision_given_launch": (
                launch_collisions / launches if launches else None
            ),
            "conflict_collision_per_valid_terminal_rollout": (
                sum(bool(run["launch_collisions"]) for run in valid_terminal_runs)
                / completed
                if completed
                else None
            ),
        },
        "totals": {
            "red_light_violations": sum(
                run["red_light_violations"] for run in valid_terminal_runs
            ),
            "illegal_crossings": sum(
                run["illegal_crossings"] for run in valid_terminal_runs
            ),
            "launches": launches,
            "launch_collisions": launch_collisions,
            "passive_collisions": sum(
                run["passive_collisions"] for run in valid_terminal_runs
            ),
            "vehicle_collisions": sum(
                run["vehicle_collisions"] for run in valid_terminal_runs
            ),
        },
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
