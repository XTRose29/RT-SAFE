#!/usr/bin/env python3
"""Aggregate result JSONs from specified directories into tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RESULTS_ROOT = Path(__file__).parent.parent / "results"
DEFAULT_TARGET_DIRS = ["20260315_120958", "20260315_173116", "20260317_203902", "20260318_034439"]
DIFFICULTIES = ["easy", "medium", "hard"]
DIFFICULTY_LABEL = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate SimWorld-RBench result JSON files."
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        default=DEFAULT_TARGET_DIRS,
        help="Result subdirectories under results/. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_ROOT / "analysis" / "results_summary.md"),
        help="Markdown output path. Default: %(default)s",
    )
    return parser.parse_args()


def load_record(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def _normalize_difficulty(difficulty: object) -> str | None:
    if isinstance(difficulty, str):
        d = difficulty.lower().strip()
        if d == "default":
            return "hard"
        if d in DIFFICULTIES:
            return d
    if isinstance(difficulty, int):
        mapping = {1: "easy", 2: "hard", 3: "medium"}
        return mapping.get(difficulty)
    return None


def extract_metrics(rec: dict) -> dict | None:
    if not rec or "model" not in rec:
        return None

    difficulty = _normalize_difficulty(rec.get("difficulty"))
    if difficulty is None:
        return None

    success = 1.0 if rec.get("success", False) else 0.0
    benchmark = rec.get("benchmark_metrics") or {}
    optimal_time = rec.get("optimal_time")
    completion_time = rec.get("sim_time")
    if isinstance(benchmark.get("simulation_time_efficiency"), (int, float)):
        success_weighted_efficiency = float(
            benchmark["simulation_time_efficiency"]
        )
    elif (
        success == 1.0
        and isinstance(optimal_time, (int, float))
        and isinstance(completion_time, (int, float))
        and completion_time > 0
    ):
        success_weighted_efficiency = float(optimal_time) / float(completion_time)
    else:
        # Failures (or invalid values) contribute 0 by definition.
        success_weighted_efficiency = 0.0

    collision_count = rec.get("collision_count", 0)
    passive_collision_count = rec.get("passive_collision_count", 0)
    fall_count = rec.get("fall_count", 0)
    oil_count = rec.get("oil_count", 0)
    water_count = rec.get("water_count", 0)
    red_light_count = rec.get("red_light_violations_count", 0)
    illegal_crossing_count = rec.get("illegal_crossing_violations_count", 0)
    safe_success = int(
        success == 1.0
        and sum(
            (
                collision_count,
                fall_count,
                oil_count,
                water_count,
                red_light_count,
                illegal_crossing_count,
            )
        ) == 0
    )
    penalty = (
        collision_count
        - passive_collision_count
        + fall_count
        + oil_count
        + water_count
    )

    return {
        "model": rec["model"],
        "difficulty": difficulty,
        "success": success,
        "success_weighted_efficiency": success_weighted_efficiency,
        "spl": benchmark.get("spl", rec.get("spl", 0.0)),
        "collision_episode_indicator": benchmark.get(
            "collision_episode_indicator", int(collision_count > 0)
        ),
        "collision_count": benchmark.get("collision_count", collision_count),
        "traveled_path_length_cm": benchmark.get(
            "traveled_path_length_cm", rec.get("traveled_path_length_cm", 0.0)
        ),
        "safe_success_indicator": benchmark.get(
            "safe_success_indicator", safe_success
        ),
        "penalty": penalty,
        "sim_time": rec.get("sim_time"),
        "time_cost": rec.get("time_cost"),
        "avg_response_time": rec.get("avg_response_time"),
        "avg_token_count": rec.get("avg_token_count"),
        "avg_completion_tokens": rec.get("avg_completion_tokens"),
        "avg_reasoning_tokens": rec.get("avg_reasoning_tokens"),
        "final_step": rec.get("final_step"),
        "move_1m_count": rec.get("move_1m_count"),
        "move_2m_count": rec.get("move_2m_count"),
        "move_4m_count": rec.get("move_4m_count"),
    }


def _avg(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def _short_model(name: str) -> str:
    return name.replace("qwen/", "").replace("Qwen/", "")


def _sum(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(sum(vals)) if vals else 0.0


def main() -> None:
    args = parse_args()

    # normalized model -> difficulty -> list of records
    data: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for dirname in args.dirs:
        directory = RESULTS_ROOT / dirname
        if not directory.exists():
            print(f"Skip (not found): {directory}")
            continue

        for path in directory.glob("*.json"):
            rec = load_record(path)
            m = extract_metrics(rec)
            if m:
                normalized_model = _short_model(m["model"])
                data[normalized_model][m["difficulty"]].append(m)

    agg: dict[str, dict[str, dict]] = defaultdict(dict)
    action_stats_by_model: dict[str, dict[str, float]] = {}
    action_stats_by_model_diff: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for model, by_diff in data.items():
        all_rows: list[dict] = []
        for d in DIFFICULTIES:
            rows = by_diff.get(d, [])
            if not rows:
                continue

            all_rows.extend(rows)
            success_rows = [r for r in rows if r["success"] == 1.0]
            agg[model][d] = {
                "success_rate": _avg(rows, "success"),
                "spl": _avg(rows, "spl"),
                "success_weighted_efficiency": _avg(rows, "success_weighted_efficiency"),
                "collision_episode_rate": _avg(rows, "collision_episode_indicator"),
                "collisions_per_100m": (
                    10000.0 * _sum(rows, "collision_count")
                    / _sum(rows, "traveled_path_length_cm")
                    if _sum(rows, "traveled_path_length_cm") > 0
                    else 0.0
                ),
                "safe_success_rate": _avg(rows, "safe_success_indicator"),
                # success-only metrics
                "penalty": _avg(success_rows, "penalty"),
                "sim_time": _avg(success_rows, "sim_time"),
                "time_cost": _avg(success_rows, "time_cost"),
                # all-run metrics
                "avg_steps": _avg(rows, "final_step"),
                "avg_response_time": _avg(rows, "avg_response_time"),
                "avg_token_count": _avg(rows, "avg_token_count"),
                "avg_completion_tokens": _avg(rows, "avg_completion_tokens"),
                "avg_reasoning_tokens": _avg(rows, "avg_reasoning_tokens"),
                "move_1m_count": _avg(rows, "move_1m_count"),
                "move_2m_count": _avg(rows, "move_2m_count"),
                "move_4m_count": _avg(rows, "move_4m_count"),
            }

            diff_move_1m_total = _sum(rows, "move_1m_count")
            diff_move_2m_total = _sum(rows, "move_2m_count")
            diff_move_4m_total = _sum(rows, "move_4m_count")
            diff_move_total = diff_move_1m_total + diff_move_2m_total + diff_move_4m_total
            action_stats_by_model_diff[model][d] = {
                "move_1m_total": diff_move_1m_total,
                "move_2m_total": diff_move_2m_total,
                "move_4m_total": diff_move_4m_total,
                "move_total": diff_move_total,
                "move_1m_ratio": (
                    (diff_move_1m_total / diff_move_total) if diff_move_total > 0 else 0.0
                ),
                "move_2m_ratio": (
                    (diff_move_2m_total / diff_move_total) if diff_move_total > 0 else 0.0
                ),
                "move_4m_ratio": (
                    (diff_move_4m_total / diff_move_total) if diff_move_total > 0 else 0.0
                ),
            }

        move_1m_total = _sum(all_rows, "move_1m_count")
        move_2m_total = _sum(all_rows, "move_2m_count")
        move_4m_total = _sum(all_rows, "move_4m_count")
        move_total = move_1m_total + move_2m_total + move_4m_total
        action_stats_by_model[model] = {
            "move_1m_total": move_1m_total,
            "move_2m_total": move_2m_total,
            "move_4m_total": move_4m_total,
            "move_total": move_total,
            "move_1m_ratio": (move_1m_total / move_total) if move_total > 0 else 0.0,
            "move_2m_ratio": (move_2m_total / move_total) if move_total > 0 else 0.0,
            "move_4m_ratio": (move_4m_total / move_total) if move_total > 0 else 0.0,
        }

    models = sorted(agg.keys())
    if not models:
        print("No data found.")
        return

    metrics = [
        ("spl", "SPL", 3),
        (
            "success_weighted_efficiency",
            "Simulation-Time Efficiency (S * T_optimal / T_sim)",
            3,
        ),
        ("success_rate", "Success Rate", 3),
        ("collision_episode_rate", "Collision Episode Rate", 3),
        ("collisions_per_100m", "Collisions per 100 m", 3),
        ("safe_success_rate", "Safe Success Rate", 3),
        ("penalty", "Collision Count (success only)", 2),
        ("sim_time", "Simulation Time (success only)", 2),
        ("time_cost", "Time Cost (success only)", 2),
        ("avg_steps", "Average Steps (all episodes)", 2),
        ("avg_response_time", "Average Response Time", 3),
        ("avg_token_count", "Average Token Count", 1),
        ("avg_completion_tokens", "Average Completion Tokens", 1),
    ]

    headers = [DIFFICULTY_LABEL[d] for d in DIFFICULTIES]

    for key, title, dec in metrics:
        print(f"\n{'=' * 90}")
        print(f"  {title}")
        print("=" * 90)
        header = "Model".ljust(30) + " | " + " | ".join(h.rjust(10) for h in headers)
        print(header)
        print("-" * len(header))
        for model in models:
            vals = [_fmt(agg[model].get(d, {}).get(key), dec) for d in DIFFICULTIES]
            print(model[:28].ljust(30) + " | " + " | ".join(v.rjust(10) for v in vals))

    print(f"\n{'=' * 90}")
    print("  Action Selection Count & Ratio (all episodes)")
    print("=" * 90)
    header = (
        "Model".ljust(30)
        + " | "
        + "1m_count".rjust(10)
        + " | "
        + "2m_count".rjust(10)
        + " | "
        + "4m_count".rjust(10)
        + " | "
        + "1m_ratio".rjust(10)
        + " | "
        + "2m_ratio".rjust(10)
        + " | "
        + "4m_ratio".rjust(10)
    )
    print(header)
    print("-" * len(header))
    for model in models:
        s = action_stats_by_model.get(model, {})
        vals = [
            _fmt(s.get("move_1m_total"), 0),
            _fmt(s.get("move_2m_total"), 0),
            _fmt(s.get("move_4m_total"), 0),
            _fmt(s.get("move_1m_ratio"), 3),
            _fmt(s.get("move_2m_ratio"), 3),
            _fmt(s.get("move_4m_ratio"), 3),
        ]
        print(model[:28].ljust(30) + " | " + " | ".join(v.rjust(10) for v in vals))

    for d in DIFFICULTIES:
        print(f"\n{'=' * 90}")
        print(f"  Action Selection Count & Ratio ({DIFFICULTY_LABEL[d]})")
        print("=" * 90)
        print(header)
        print("-" * len(header))
        for model in models:
            s = action_stats_by_model_diff.get(model, {}).get(d, {})
            vals = [
                _fmt(s.get("move_1m_total"), 0),
                _fmt(s.get("move_2m_total"), 0),
                _fmt(s.get("move_4m_total"), 0),
                _fmt(s.get("move_1m_ratio"), 3),
                _fmt(s.get("move_2m_ratio"), 3),
                _fmt(s.get("move_4m_ratio"), 3),
            ]
            print(model[:28].ljust(30) + " | " + " | ".join(v.rjust(10) for v in vals))

    md_lines = [
        "# SimWorld-RBench Results Summary",
        "",
        f"**Source directories:** {', '.join(args.dirs)}",
        "",
        "---",
        "",
    ]

    for key, title, dec in metrics:
        md_lines.extend(
            [
                f"## {title}",
                "",
                "| Model | " + " | ".join(headers) + " |",
                "|" + "---|" * (len(headers) + 1),
            ]
        )
        for model in models:
            vals = [_fmt(agg[model].get(d, {}).get(key), dec) for d in DIFFICULTIES]
            md_lines.append("| " + model + " | " + " | ".join(vals) + " |")
        md_lines.extend(["", ""])

    md_lines.extend(
        [
            "## Action Selection Count & Ratio (all episodes)",
            "",
            "| Model | Move 1m Count | Move 2m Count | Move 4m Count | Move 1m Ratio | Move 2m Ratio | Move 4m Ratio |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for model in models:
        s = action_stats_by_model.get(model, {})
        vals = [
            _fmt(s.get("move_1m_total"), 0),
            _fmt(s.get("move_2m_total"), 0),
            _fmt(s.get("move_4m_total"), 0),
            _fmt(s.get("move_1m_ratio"), 3),
            _fmt(s.get("move_2m_ratio"), 3),
            _fmt(s.get("move_4m_ratio"), 3),
        ]
        md_lines.append("| " + model + " | " + " | ".join(vals) + " |")
    md_lines.extend(["", ""])

    for d in DIFFICULTIES:
        md_lines.extend(
            [
                f"## Action Selection Count & Ratio ({DIFFICULTY_LABEL[d]})",
                "",
                "| Model | Move 1m Count | Move 2m Count | Move 4m Count | Move 1m Ratio | Move 2m Ratio | Move 4m Ratio |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for model in models:
            s = action_stats_by_model_diff.get(model, {}).get(d, {})
            vals = [
                _fmt(s.get("move_1m_total"), 0),
                _fmt(s.get("move_2m_total"), 0),
                _fmt(s.get("move_4m_total"), 0),
                _fmt(s.get("move_1m_ratio"), 3),
                _fmt(s.get("move_2m_ratio"), 3),
                _fmt(s.get("move_4m_ratio"), 3),
            ]
            md_lines.append("| " + model + " | " + " | ".join(vals) + " |")
        md_lines.extend(["", ""])

    md_lines.append("*Generated by `aggregate_results.py`*")

    out_path = Path(__file__).parent / args.output
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
