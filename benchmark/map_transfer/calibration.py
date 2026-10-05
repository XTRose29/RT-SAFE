"""Measure all 16 actions in native physics on an unobstructed pilot route."""

from __future__ import annotations
import json
import math
from pathlib import Path
from .contract import Action, WAYPOINTS, TURN_ANGLES
from .runner import NativeEpisode


def run_calibration(client, manifest: dict, output: Path, task_id="nyc-park-clear"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Calibration output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    def measure(episode, action):
        before = client.call("status")
        after = episode.step(action)
        position_error = math.dist(before["position_cm"][:2], after["position_cm"][:2])
        delta_yaw = (after["yaw_deg"] - before["yaw_deg"] + 180) % 360 - 180
        seconds = after["simulation_time"] - before["simulation_time"]
        row = {
            "action": action.as_dict(),
            "simulation_seconds": seconds,
            "horizontal_displacement_cm": position_error,
            "yaw_change_deg": delta_yaw,
            "finish_reason": after["last_command"]["finish_reason"],
            "native_events": after["counts"],
        }
        rows.append(row)
        (output / "measurements.json").write_text(json.dumps(rows, indent=2))
        assert not after["errors"], after["errors"]
        assert not any(after["counts"].values()), row
        if action.action_type == "move_to":
            distance, yaw = WAYPOINTS[action.action_param]
            assert distance - 35 <= position_error <= distance + 10, row
            assert abs((delta_yaw - yaw + 180) % 360 - 180) <= 1, row
            assert row["finish_reason"] == "target_reached", row
        elif action.action_type == "turn_around":
            yaw = TURN_ANGLES[action.action_param]
            expected = manifest["agent"].get("turn_duration_s", 1.0)
            assert abs(seconds - expected) <= 0.1, row
            assert abs((delta_yaw - yaw + 180) % 360 - 180) <= 1, row
            assert position_error <= 2, row
            assert row["finish_reason"] == "turn_complete", row
        else:
            assert abs(seconds - int(action.action_param)) <= 0.1, row
            assert position_error <= 2, row
            assert row["finish_reason"] == "wait_complete", row
        print(json.dumps(row), flush=True)

    for waypoint in WAYPOINTS:
        with NativeEpisode(
            client,
            manifest,
            task_id,
            output / ("move-" + waypoint),
            mode="static",
            reload_source=True,
        ) as episode:
            episode.observe(0)
            measure(episode, Action("move_to", waypoint))
            episode.observe(1)
            episode.finish("calibration_complete")
    with NativeEpisode(
        client,
        manifest,
        task_id,
        output / "turns-and-waits",
        mode="static",
        reload_source=True,
    ) as episode:
        episode.observe(0)
        for turn in TURN_ANGLES:
            measure(episode, Action("turn_around", turn))
        for wait in ("1", "2", "3"):
            measure(episode, Action("wait", wait))
        episode.observe(1)
        episode.finish("calibration_complete")
    report = {
        "passed": True,
        "actions": len(rows),
        "task_id": task_id,
        "measurements": rows,
        "tolerances": {
            "move_shortfall_cm": 35,
            "move_overshoot_cm": 10,
            "yaw_deg": 1,
            "stationary_drift_cm": 2,
            "turn_wait_timing_s": 0.1,
        },
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    return report
