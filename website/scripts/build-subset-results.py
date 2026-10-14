"""Aggregate a frozen local task selection; publish only configuration summaries.

The manifest and audit stay outside the public website/repository. This reuses
recorded episodes, including failed navigation, and never selects by outcome.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

SITE = Path(__file__).resolve().parents[1]
PHASES = ["05_hard_realtime_efforts", "01_hard_realtime_default", "06_hard_realtime_upper"]
SLUGS = {
    "Astra": "gpt-6-astra", "Sol": "gpt-5_6-sol",
    "Fable": "claude-fable-5-1", "Sonnet": "claude-sonnet-5",
    "Gemini": "google_gemini-3_8-flash",
    "DeepSeek": "deepseek_deepseek-v4-flash-vision-exp",
    "Inkling": "thinkingmachines_inkling", "Grok": "x-ai_grok-4_6",
}


def validate_manifest(manifest):
    tasks = manifest["tasks"]
    assert len(tasks) == 10 and len({(t["map"], t["task"]) for t in tasks}) == 10, "Need 10 distinct tasks"
    counts = Counter(t["map"] for t in tasks)
    assert len(counts) == 2 and set(counts.values()) == {5}, "Need five tasks per map"
    assert manifest["difficulty"] == "hard" and manifest["timing"] == "realtime"
    assert manifest["seed"] == 0, "This recorded campaign uses seed 0"


def episode_metrics(record):
    counts = {key: record[key] for key in (
        "collision_count", "passive_collision_count", "fall_count", "oil_count",
        "water_count", "red_light_violations_count", "illegal_crossing_violations_count",
        "decision_count",
    )}
    assert all(isinstance(v, int) and v >= 0 for v in counts.values()), "Missing/invalid counters"
    collisions, passive = counts["collision_count"], counts["passive_collision_count"]
    assert passive <= collisions
    assert isinstance(record["success"], bool)
    safe = record["success"] and not any(counts[k] for k in (
        "collision_count", "fall_count", "oil_count", "water_count",
        "red_light_violations_count", "illegal_crossing_violations_count",
    ))
    metrics = record["benchmark_metrics"]
    assert metrics["success_indicator"] == int(record["success"])
    assert metrics["collision_count"] == collisions and metrics["passive_collision_count"] == passive
    assert metrics["active_collision_count"] == collisions - passive
    assert metrics["safe_success_indicator"] == int(safe), "Safe-success definition differs"
    latency = record["avg_response_time"]
    spl = metrics["spl"]
    assert math.isfinite(latency) and latency >= 0
    assert math.isfinite(spl) and 0 <= spl <= 1
    return dict(success=100 * int(record["success"]), safeSuccess=100 * int(safe),
                collisions=collisions, passive=passive, active=collisions-passive,
                decisions=counts["decision_count"], latency=latency, spl=spl)


def aggregate(records):
    assert len(records) == 10, "Do not silently rank incomplete configurations"
    episodes = [episode_metrics(r) for r in records]
    return {key: round(mean(e[key] for e in episodes), 6) for key in episodes[0]}


def build(manifest_path, main_root, claude_root, audit_path):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    models = json.loads((SITE / "app/data.json").read_text())["models"]
    result_models, audit = [], []
    for model in models:
        name = model["name"]
        root = claude_root if name in {"Fable", "Sonnet"} else main_root
        phases = [PHASES[1], PHASES[0], PHASES[2]] if name == "Sol" else PHASES
        efforts = []
        for index, phase in enumerate(phases):
            records = []
            for task in manifest["tasks"]:
                paths = list((root / phase / SLUGS[name] / "cells").glob(
                    f"*/{task['map']}/task_{task['task']:03d}/status.json"))
                assert len(paths) == 1, f"Missing/duplicate current cell: {name}, {phase}, {task}"
                status = json.loads(paths[0].read_text())
                assert status["state"] == "completed", f"Incomplete cell: {paths[0]}"
                path = Path(status["result_path"])
                raw = path.read_bytes()
                r = json.loads(raw)
                assert r["task_id"] == task["task"] and r["seed"] == manifest["seed"]
                assert r["realtime_thinking"] is True and not r.get("rollout_error")
                points = r["route_reconstruction"]["runtime_route_points"]
                length = sum(math.dist(a, b) for a, b in zip(points, points[1:])) / 100
                assert abs(length - task["length_m"]) < .001, "Task route differs between configurations"
                config = r["difficulty_config"]
                assert all(config[key] == 1 for key in (
                    "pedestrian_ratio", "movable_obstacle_ratio", "irregular_ratio", "falling_ratio")), "Not hard difficulty"
                actual = r["reasoning_effort"].lower()
                expected = model["effort"][index]["label"].lower()
                assert actual == expected or (phase == PHASES[1] and actual == "default"), (name, actual, expected)
                records.append(r)
                audit.append(dict(model=name, effort=expected, map=task["map"], task=task["task"],
                                  status=str(paths[0]), result=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                  metrics=episode_metrics(r)))
            efforts.append(dict(label=model["effort"][index]["label"], **aggregate(records)))
        result_models.append(dict(name=name, effort=efforts))
    output = dict(
        id=manifest["id"], source="Reaggregated finalized hard real-time campaign episodes (September 2026).",
        taskCount=10, mapCount=2, tasksPerMap=5, seed=manifest["seed"],
        manifestSha256=hashlib.sha256(manifest_bytes).hexdigest(),
        selection="Five tasks per map, spanning recorded route lengths and authored crossing counts. Selection does not use model outcomes.",
        aggregation="Equal episode weighting, including failed navigation. Latency is the mean of episode mean response times. SPL comes from recorded benchmark metrics.",
        models=result_models,
    )
    audit_path.write_text(json.dumps(dict(manifest=manifest, episodes=audit), indent=2) + "\n")
    for destination in [SITE / "app/subset-results.json", SITE / "public/data/subset-results.json"]:
        destination.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Aggregated {len(audit)} episodes into {len(result_models) * 3} configurations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["manifest", "main-root", "claude-root", "audit"]:
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    build(args.manifest, args.main_root, args.claude_root, args.audit)
