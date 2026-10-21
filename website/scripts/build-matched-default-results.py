"""Publish a complete, matched 20-task default-reasoning comparison.

Raw records, the selection manifest and the source audit remain local. The
historical campaign keeps its original scoring; no failures are dropped.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean, median

SITE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("subset", SITE / "scripts/build-subset-results.py")
subset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subset)


def validate_selection(manifest):
    tasks = manifest["tasks"]
    assert len(tasks) == 20 and len({(t["map"], t["task"]) for t in tasks}) == 20
    assert len({t["task"] for t in tasks}) == 20
    assert sorted(Counter(t["map"] for t in tasks).values()) == [5, 5, 5, 5]
    assert (manifest["difficulty"], manifest["timing"], manifest["seed"]) == ("hard", "realtime", 0)


def completed_record(cells, task, expected_effort, reference=None):
    paths = [p for root in cells for p in root.glob(f"*/{task['map']}/task_{task['task']:03d}/status.json")]
    assert len(paths) == 1, f"Expected one current cell for {task}: {paths}"
    status = json.loads(paths[0].read_text())
    assert status["state"] == "completed", f"Incomplete task: {paths[0]}"
    path = Path(status["result_path"])
    raw = path.read_bytes()
    record = json.loads(raw)
    assert record["task_id"] == task["task"] and record["seed"] == 0
    assert record["realtime_thinking"] is True and not record.get("rollout_error")
    assert record["reasoning_effort"].lower() in expected_effort
    assert all(record["difficulty_config"][k] == 1 for k in (
        "pedestrian_ratio", "movable_obstacle_ratio", "irregular_ratio", "falling_ratio"))
    points = record["route_reconstruction"]["runtime_route_points"]
    length = sum(math.dist(a, b) for a, b in zip(points, points[1:])) / 100
    assert abs(length - task["length_m"]) < .001
    if reference is not None:
        assert points == reference, f"Route geometry mismatch: {path}"
    subset.episode_metrics(record)
    return record, dict(task=task["task"], map=task["map"], result=str(path),
                       sha256=hashlib.sha256(raw).hexdigest())


def aggregate(records):
    assert len(records) == 20, "All 20 finalized episodes are required"
    assert len({r["task_id"] for r in records}) == 20, "Duplicate task"
    metrics = [subset.episode_metrics(r) for r in records]
    result = {k: round(mean(r[k] for r in metrics), 6) for k in metrics[0]}
    total = lambda key: sum(r[key] for r in records)
    moves = sum(total(f"move_{n}m_count") for n in (1, 2, 4))
    result["behavior"] = dict(
        medianDecisions=median(r["decision_count"] for r in records),
        maxDecisions=max(r["decision_count"] for r in records),
        movesPerEpisode=total("move_count") / 20,
        turnsPerEpisode=total("turn_count") / 20,
        waitsPerEpisode=total("wait_count") / 20,
        commandedMoveMeters=sum(n * total(f"move_{n}m_count") for n in (1, 2, 4)) / moves if moves else None,
        offRouteNoProgressMoves=total("off_route_move_no_progress_count"),
        parseErrors=total("parse_error_count"),
        terminationCounts=dict(Counter(r["termination_reason"] for r in records)),
        inferenceMinutesPerEpisode=mean(r["benchmark_metrics"]["latency"]["cumulative_response_seconds"] / 60 for r in records),
        passiveCollisionPercent=100 * total("passive_collision_count") / total("collision_count") if total("collision_count") else 0,
    )
    return result


def build(args):
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_selection(manifest)
    identities = json.loads((SITE / "app/data.json").read_text())["models"]
    identities.append(dict(name="Gemini 4", fullName="Gemini 4 Argon", provider="Google", color="#4285f4"))
    rows, audit, reference = [], [], {}
    for identity in identities:
        name = identity["name"]
        if name == "Gemini 4":
            cells = [p / "cells" for p in args.gemini_suite]
            effort, allowed = "Medium", {"medium"}
            transport = "Account-pool subscription · explicit medium"
        else:
            root = args.claude_root if name in {"Fable", "Sonnet"} else args.main_root
            cells = [root / "01_hard_realtime_default" / subset.SLUGS[name] / "cells"]
            effort = identity["effort"][0 if name == "Sol" else 1]["label"]
            allowed = {"default", effort.lower()}
            transport = "Claude Code" if name in {"Fable", "Sonnet"} else "Codex CLI" if name in {"Astra", "Sol"} else "OpenRouter"
        records = []
        for task in manifest["tasks"]:
            record, source = completed_record(cells, task, allowed, reference.get(task["task"]))
            reference.setdefault(task["task"], record["route_reconstruction"]["runtime_route_points"])
            records.append(record)
            audit.append(dict(model=name, **source, metrics=subset.episode_metrics(record),
                              behavior={k: record[k] for k in ("decision_count", "termination_reason", "off_route_move_no_progress_count", "collision_count", "passive_collision_count", "success")}))
        rows.append({k: identity[k] for k in ("name", "fullName", "provider", "color")} |
                    dict(effort=effort, transport=transport, **aggregate(records)))
    output = dict(id=manifest["id"], updatedAt=datetime.now(timezone.utc).isoformat(),
                  taskCount=20, mapCount=4, tasksPerMap=5, seed=0,
                  manifestSha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  source="September 2026 historical defaults and October 2026 Gemini 4 explicit-medium runs; original recorded scoring.",
                  selection="Five tasks per map, selected by route geometry and crossing strata before Gemini 4 outcomes. All nine models use the same 20 routes.",
                  aggregation="Equal episode weighting, including navigation failures. Latency is the mean of episode mean response times. Safe success requires reaching the goal with zero recorded safety events.",
                  limitations="One seed; runs collected on different dates and serving transports. Gemini 4 uses explicit medium, the configured default. Original curb scoring retained. Repeated contacts are not distinct accidents; these observations do not isolate latency as a cause.",
                  models=rows)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(dict(manifest=manifest, episodes=audit), indent=2) + "\n")
    for path in [SITE / "app/matched-default-results.json", SITE / "public/data/matched-default-results.json"]:
        path.write_text(json.dumps(output, indent=2) + "\n")
    for row in sorted(rows, key=lambda r: r["collisions"]):
        print(f"{row['fullName']:30} success={row['success']:5.1f}% safe={row['safeSuccess']:4.1f}% collisions={row['collisions']:6.2f} latency={row['latency']:5.2f}s decisions={row['decisions']:6.1f}")
    print(f"Validated {len(audit)} records, with identical route geometry across all models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "main-root", "claude-root", "audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gemini-suite", type=Path, action="append", required=True)
    build(parser.parse_args())
