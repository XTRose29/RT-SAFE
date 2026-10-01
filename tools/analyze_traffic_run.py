#!/usr/bin/env python3
"""Summarize traffic-light alignment from per-step benchmark sidecars."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("steps_dir", type=Path)
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument(
        "--crosswalk-half-width-cm",
        type=float,
        default=300.0,
        help="painted route-crosswalk half width used for trajectory QA",
    )
    args = parser.parse_args()

    rows = []
    for step_dir in sorted(args.steps_dir.glob("step_*")):
        sidecars = list(step_dir.glob("*_traffic_light_snapshot.json"))
        if not sidecars:
            continue
        with sidecars[0].open(encoding="utf-8") as handle:
            payload = json.load(handle)
        # model_input is the state captured immediately before the rendered
        # frame supplied to the VLM.  That is the authoritative visual/prompt
        # alignment sample; action_execution is a later safety re-check.
        snapshot = payload.get("model_input") or payload
        signals = snapshot.get("signal_states") or []
        rows.append(
            {
                "step": step_dir.name,
                "pedestrian_state": snapshot.get("pedestrian_state"),
                "permission": snapshot.get("route_crossing_permission"),
                "on_crosswalk": bool(snapshot.get("agent_on_crosswalk")),
                "crossing_in_progress": bool(
                    snapshot.get(
                        "crossing_in_progress",
                        snapshot.get("agent_on_crosswalk"),
                    )
                ),
                "lateral_distance_cm": snapshot.get(
                    "crosswalk_lateral_distance_cm",
                    snapshot.get("distance_cm"),
                ),
                "occupied": bool(snapshot.get("pedestrian_occupied")),
                "legacy_hold": bool(snapshot.get("legacy_occupancy_hold")),
                "any_vehicle_green": any(
                    bool(signal.get("vehicle_green"))
                    or signal.get("vehicle_state") == "GREEN"
                    for signal in signals
                ),
                "all_pedestrian_walk": bool(signals)
                and all(
                    bool(signal.get("pedestrian_walk"))
                    or signal.get("pedestrian_state") == "WALK"
                    for signal in signals
                ),
            }
        )

    relevant = [
        row
        for row in rows
        if row["crossing_in_progress"]
        or row["legacy_hold"]
        or row["occupied"]
    ]
    print(json.dumps(rows if args.show_all else (relevant or rows[-12:]), indent=2))

    violations = [
        row
        for row in rows
        if row["crossing_in_progress"]
        and (
            not row["on_crosswalk"]
            or row["lateral_distance_cm"] is None
            or row["lateral_distance_cm"] > args.crosswalk_half_width_cm
            or row["pedestrian_state"] != "WALK"
            or row["any_vehicle_green"]
            # Native UE controllers expose pedestrian_occupied directly.
            # legacy_occupancy_hold is only the compatibility fallback for
            # older packaged maps and must not be required on native runs.
            or not (row["occupied"] or row["legacy_hold"])
        )
    ]
    print(
        json.dumps(
            {
                "recorded_steps": len(rows),
                "crosswalk_steps": sum(
                    row["crossing_in_progress"] for row in rows
                ),
                "alignment_violations": violations,
            },
            indent=2,
        )
    )
    raise SystemExit(1 if violations else 0)


if __name__ == "__main__":
    main()
