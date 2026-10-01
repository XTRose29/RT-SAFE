#!/usr/bin/env python3
"""Write human-readable rollout and unsafe-trigger evidence indexes."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.evaluation_metrics import aggregate_metrics


CATEGORIES = (
    ("human_collision", "Pedestrian / irregular-NPC collision"),
    ("object_collision", "Movable or falling object collision"),
    ("building_collision", "Building collision and rollback"),
    ("touched_road", "Road-surface violation"),
    ("fall", "Fall overlap"),
    ("oil", "Oil overlap / deceleration"),
    ("water", "Water-puddle overlap / slip"),
)


def rel(path: str, root: Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(candidate)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def resolve_result_path(raw_path: str, suite: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    # Result paths are stored relative to the repository root, whereas this
    # summary is intentionally written into a consolidated results directory.
    return suite.parent.parent / candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True, type=Path)
    args = parser.parse_args()
    suite = args.suite_dir.resolve()
    index = json.loads((suite / "video_index.json").read_text(encoding="utf-8"))

    episodes = []
    missing_episode_results = []
    for item in index["rollouts"]:
        result_path = resolve_result_path(item["result"], suite)
        if result_path.is_file():
            episodes.append(json.loads(result_path.read_text(encoding="utf-8")))
        else:
            missing_episode_results.append(str(result_path))

    episode_count = len(index["rollouts"])
    success_count = sum(bool(item["success"]) for item in index["rollouts"])
    total_decisions = sum(int(row.get("decision_count", 0)) for row in episodes)
    total_parse_errors = sum(int(row.get("parse_error_count", 0)) for row in episodes)
    total_invalid = sum(int(row.get("invalid_decision_count", 0)) for row in episodes)
    paper_metrics = aggregate_metrics(episodes)
    failure_reasons = collections.Counter(
        str(row.get("failure_reason") or row.get("stuck_reason") or "step_limit")
        for row in episodes
        if not row.get("success")
    )
    counter_fields = (
        "collision_count",
        "passive_collision_count",
        "human_collision_count",
        "object_collision_count",
        "building_collision_count",
        "off_route_move_no_progress_count",
        "vehicle_collision_count",
        "fall_count",
        "oil_count",
        "water_count",
        "red_light_violations_count",
        "illegal_crossing_violations_count",
    )
    suite_evaluation = {
        "episode_count": episode_count,
        "episode_results_loaded": len(episodes),
        "missing_episode_results": missing_episode_results,
        "success_count": success_count,
        "failure_count": episode_count - success_count,
        "success_rate": success_count / episode_count if episode_count else 0.0,
        "terminal_failure_reasons": dict(sorted(failure_reasons.items())),
        "decision_count": total_decisions,
        "parse_error_count": total_parse_errors,
        "invalid_decision_count": total_invalid,
        "parse_success_rate": (
            (total_decisions - total_parse_errors) / total_decisions
            if total_decisions
            else 0.0
        ),
        "decision_accuracy": (
            (total_decisions - total_invalid) / total_decisions
            if total_decisions
            else 0.0
        ),
        "mean_wall_time_seconds": mean(
            [float(row.get("time_cost", 0.0)) for row in episodes]
        ),
        "mean_sim_time_seconds": mean(
            [float(row.get("sim_time", 0.0)) for row in episodes]
        ),
        "mean_response_time_seconds": mean(
            [float(row.get("avg_response_time", 0.0)) for row in episodes]
        ),
        "benchmark_metrics": paper_metrics,
        "safety_totals": {
            field: sum(int(row.get(field, 0)) for row in episodes)
            for field in counter_fields
        },
        "video_count": len(index["rollouts"]),
        "video_alignment_pass_count": sum(
            bool(item.get("alignment_passed")) for item in index["rollouts"]
        ),
        "all_video_alignment_checks_passed": bool(
            index.get("all_alignment_checks_passed")
        ),
    }
    (suite / "suite_evaluation_summary.json").write_text(
        json.dumps(suite_evaluation, indent=2), encoding="utf-8"
    )

    summary = [
        "# Qwen3-VL-8B RT12 rollout summary",
        "",
        "| Task index | Scenario | Result | Frames | Safety-event steps | Alignment | Mean input render | Video |",
        "|---:|---:|---|---:|---:|---|---:|---|",
    ]
    for item in sorted(index["rollouts"], key=lambda row: row["task_index"]):
        result = "success" if item["success"] else "failed / max steps"
        summary.append(
            f"| {item['task_index']} | {item['task_id']} | {result} | "
            f"{item['frame_count']} | {item['safety_event_count']} | "
            f"{'PASS' if item['alignment_passed'] else 'FAIL'} | "
            f"{item['mean_input_render_wall_seconds']:.3f}s | "
            f"[video]({rel(item['video'], suite)}) |"
        )
    summary.extend(
        [
            "",
            "## Final suite evaluation output",
            "",
            f"- Task success: **{success_count}/{episode_count} "
            f"({suite_evaluation['success_rate']:.1%})**.",
            f"- Parse success: **{suite_evaluation['parse_success_rate']:.1%}**; "
            f"executable-decision accuracy: **{suite_evaluation['decision_accuracy']:.1%}** "
            f"over {total_decisions} decisions.",
            f"- Video alignment: **{suite_evaluation['video_alignment_pass_count']}/"
            f"{suite_evaluation['video_count']} passed**.",
            f"- Mean wall/simulation/model-response time: "
            f"**{suite_evaluation['mean_wall_time_seconds']:.2f}s / "
            f"{suite_evaluation['mean_sim_time_seconds']:.2f}s / "
            f"{suite_evaluation['mean_response_time_seconds']:.2f}s**.",
            f"- SPL/STE/SafeSR: **{paper_metrics['spl']:.3f} / "
            f"{paper_metrics['simulation_time_efficiency']:.3f} / "
            f"{paper_metrics['safe_success_rate']:.3f}**.",
            f"- Collision episode rate and collisions per 100 m: "
            f"**{paper_metrics['collision_episode_rate']:.3f} / "
            f"{paper_metrics['collisions_per_100m']:.3f}**.",
            "- Safety counters remain separate outputs and are not folded into "
            "a hidden composite score; see "
            "[suite_evaluation_summary.json](suite_evaluation_summary.json).",
            "",
            f"All alignment checks passed: `{index['all_alignment_checks_passed']}`.",
            "",
            "Machine-readable details: [video_index.json](video_index.json).",
        ]
    )
    (suite / "ROLLOUT_SUMMARY.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    examples = [
        "# Unsafe-trigger video examples",
        "",
        "Traffic-light and vehicle cases are excluded. Event identity comes from the UE state counters/overlap code recorded in each step manifest.",
        "",
    ]
    covered = index.get("unsafe_trigger_video_examples") or {}
    missing = []
    for category, title in CATEGORIES:
        examples.append(f"## {title}")
        examples.append("")
        rows = covered.get(category) or []
        if not rows:
            missing.append(category)
            examples.append("No naturally occurring example was recorded in the completed task videos.")
            examples.append("")
            continue
        for row in rows[:5]:
            examples.append(
                f"- Scenario {row['task_id']} (index {row['task_index']}), "
                f"step {row['step']}, simulation t={row['sim_time_start_seconds']:.1f}–"
                f"{row['sim_time_end_seconds']:.1f}s: [{row['event']}]"
                f"({rel(row['video'], suite)})"
            )
        examples.append("")
    examples.extend(
        [
            "## Missing natural coverage",
            "",
            (", ".join(missing) if missing else "None; every requested non-traffic trigger class has a recorded example."),
            "",
            "Exact object subtype cannot be inferred from the class-only UE collision counter; subtype claims require visual confirmation or per-actor contact instrumentation.",
        ]
    )
    (suite / "UNSAFE_TRIGGER_VIDEO_EXAMPLES.md").write_text(
        "\n".join(examples) + "\n", encoding="utf-8"
    )
    (suite / "missing_trigger_example_categories.json").write_text(
        json.dumps(missing, indent=2), encoding="utf-8"
    )
    print(json.dumps({"rollouts": len(index["rollouts"]), "missing": missing}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
