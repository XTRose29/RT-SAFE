"""Check saved native-run evidence without requiring Unreal or credentials."""

from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .contract import EVENT_TYPES


def verify_run(directory: str | Path) -> dict:
    root = Path(directory)

    def read(name):
        return json.loads((root / name).read_text())

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    summary = read("summary.json")
    state = read("final-state.json")
    provenance = read("provenance.json")
    require(
        summary.get("pilot") is True and provenance.get("pilot") is True,
        "Run must remain labeled as a pilot",
    )
    for key in ("task_id", "mode"):
        require(summary[key] == provenance[key], f"Mismatched {key}")
    counts = Counter()
    last_time = -1.0
    for index, event in enumerate(state["events"]):
        require(event["index"] == index, "Event indices must be contiguous")
        require(event["type"] in EVENT_TYPES, "Unknown event type")
        require(
            event["phase"] in ("action", "inference"), "Event outside active episode"
        )
        timestamp = event["simulation_time"]
        require(
            math.isfinite(timestamp)
            and last_time <= timestamp <= state["simulation_time"] + 0.1,
            "Event timestamp outside episode or out of order",
        )
        last_time = timestamp
        if event["type"] == "collision":
            require(
                event["evidence"].get("source") == "unreal_blocking_hit",
                "Collision lacks native blocking-hit evidence",
            )
            require(
                event["evidence"].get("penetrating") is False,
                "Collision is an initial penetration",
            )
        counts[event["type"]] += 1
    expected = {kind: counts[kind] for kind in sorted(EVENT_TYPES)}
    require(
        summary["safety_events"] == expected == state["counts"],
        "Summary counts disagree with native event log",
    )
    passive = sum(
        e["type"] == "collision" and e["phase"] == "inference" for e in state["events"]
    )
    require(summary["passive_collisions"] == passive, "Passive count mismatch")
    require(
        summary["active_collisions"] == counts["collision"] - passive,
        "Active count mismatch",
    )
    require(summary["success"] == state["goal_reached"], "Arrival mismatch")
    require(
        summary["safe_success"] == bool(state["goal_reached"] and not state["events"]),
        "Safe success disagrees with arrival and events",
    )
    require(summary["runtime_errors"] == state["errors"], "Runtime error mismatch")
    require(
        summary["terminal_reason"] == state["terminal_reason"],
        "Terminal reason mismatch",
    )
    decisions = root / "decisions.jsonl"
    records = (
        [
            json.loads(line)
            for line in decisions.read_text().splitlines()
            if line.strip()
        ]
        if decisions.exists()
        else []
    )
    require(len(records) == summary["decisions"], "Decision count mismatch")
    require(
        [r["index"] for r in records] == list(range(len(records))),
        "Decision indices must be contiguous",
    )
    trajectory = read("trajectory.json")["samples"]
    require(bool(trajectory), "Missing trajectory samples")
    last_time = -1.0
    for sample in trajectory:
        timestamp = sample["simulation_time"]
        require(
            math.isfinite(timestamp)
            and last_time <= timestamp <= state["simulation_time"] + 0.1,
            "Trajectory timestamp outside episode or out of order",
        )
        last_time = timestamp
        require(
            all(math.isfinite(v) for v in sample["agent_position_cm"]),
            "Non-finite trajectory position",
        )
    observations = sorted(root.glob("observation-[0-9]*.json"))
    for path in observations:
        metadata = json.loads(path.read_text())
        stem = path.stem
        for suffix, key in (
            ("-raw.jpg", "rgb_sha256"),
            (".png", "policy_image_sha256"),
        ):
            actual = hashlib.sha256((root / (stem + suffix)).read_bytes()).hexdigest()
            require(
                actual == metadata[key],
                f"Observation checksum mismatch: {stem + suffix}",
            )
    resolved = root / "manifest.json"
    if resolved.exists():
        value = json.loads(resolved.read_text())
        actual = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        require(
            actual == provenance["manifest_sha256"],
            "Resolved manifest checksum mismatch",
        )
    return {
        "verified": True,
        "task_id": summary["task_id"],
        "mode": summary["mode"],
        "events": len(state["events"]),
        "decisions": len(records),
        "observations": len(observations),
        "trajectory_samples": len(trajectory),
        "resolved_manifest_present": resolved.exists(),
        "runtime_errors": len(state["errors"]),
    }
