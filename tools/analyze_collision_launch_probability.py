#!/usr/bin/env python3
"""Summarize conditional collision probability from benchmark task JSON files.

The denominator is the number of conflict-vehicle events whose disposition is
explicitly ``launched``.  Trigger attempts that were disabled, skipped, or
could not acquire a vehicle are deliberately excluded.  A collision is only
counted when the event itself records ``collision_triggered=true`` (or the
equivalent terminal ``status=collision``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EVENT_FIELDS = (
    "red_light_conflict_vehicle_events",
    "illegal_crossing_conflict_vehicle_events",
)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if trials <= 0:
        return None
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def load_task_jsons(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("task_*.json")):
        if "audit" in path.stem.lower():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and any(field in payload for field in EVENT_FIELDS):
            records.append((path, payload))
    return records


def load_pressure_metrics(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load one-shot full-task pressure demos without weakening event checks."""
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("*_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        event = payload.get("trigger_record")
        if not isinstance(event, dict):
            continue
        mode = payload.get("mode") or path.name.removesuffix("_metrics.json")
        field = (
            "red_light_conflict_vehicle_events"
            if mode == "red_light"
            else "illegal_crossing_conflict_vehicle_events"
        )
        records.append(
            (
                path,
                {
                    "task_id": payload.get("task_id"),
                    "failed": payload.get("terminal", {}).get("failed", False),
                    "failure_reason": payload.get("terminal", {}).get(
                        "failure_reason"
                    ),
                    field: [event],
                    "pressure_seed": payload.get("seed"),
                    "pressure_passed": payload.get("passed"),
                },
            )
        )
    return records


def summarize(root: Path) -> dict[str, Any]:
    task_records = load_task_jsons(root)
    pressure_records = load_pressure_metrics(root)
    records = [*task_records, *pressure_records]
    totals = {
        "completed_task_jsons": len(task_records),
        "completed_pressure_metrics": len(pressure_records),
        "trigger_events": 0,
        "launched": 0,
        "collisions": 0,
        "red_light_launched": 0,
        "red_light_collisions": 0,
        "illegal_crossing_launched": 0,
        "illegal_crossing_collisions": 0,
    }
    per_run: list[dict[str, Any]] = []

    for path, payload in records:
        run = {
            "path": str(path),
            "task_id": payload.get("task_id"),
            "failed": bool(payload.get("failed", False)),
            "failure_reason": payload.get("failure_reason"),
            "pressure_seed": payload.get("pressure_seed"),
            "pressure_passed": payload.get("pressure_passed"),
            "launched": 0,
            "collisions": 0,
            "events": [],
        }
        for field in EVENT_FIELDS:
            trigger_kind = (
                "red_light" if field.startswith("red_light") else "illegal_crossing"
            )
            events = payload.get(field) or []
            if not isinstance(events, list):
                continue
            totals["trigger_events"] += len(events)
            for event in events:
                if not isinstance(event, dict):
                    continue
                launched = event.get("disposition") == "launched"
                collided = launched and (
                    event.get("collision_triggered") is True
                    or event.get("status") == "collision"
                )
                if launched:
                    totals["launched"] += 1
                    totals[f"{trigger_kind}_launched"] += 1
                    run["launched"] += 1
                if collided:
                    totals["collisions"] += 1
                    totals[f"{trigger_kind}_collisions"] += 1
                    run["collisions"] += 1
                run["events"].append(
                    {
                        "trigger": trigger_kind,
                        "event_id": event.get("event_id"),
                        "disposition": event.get("disposition"),
                        "status": event.get("status"),
                        "launch_distance_cm": event.get("launch_distance_cm"),
                        "minimum_agent_distance_cm": event.get(
                            "minimum_agent_distance_cm"
                        ),
                        "collision_triggered": event.get("collision_triggered"),
                    }
                )
        per_run.append(run)

    launched = totals["launched"]
    collisions = totals["collisions"]
    totals["collision_probability_given_launch"] = (
        collisions / launched if launched else None
    )
    totals["collision_probability_wilson_95"] = wilson_interval(collisions, launched)
    for kind in ("red_light", "illegal_crossing"):
        kind_launched = totals[f"{kind}_launched"]
        kind_collisions = totals[f"{kind}_collisions"]
        totals[f"{kind}_collision_probability_given_launch"] = (
            kind_collisions / kind_launched if kind_launched else None
        )
        totals[f"{kind}_collision_probability_wilson_95"] = wilson_interval(
            kind_collisions, kind_launched
        )

    return {
        "source_root": str(root.resolve()),
        "definition": "P(collision | disposition == launched)",
        "totals": totals,
        "runs": per_run,
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
