#!/usr/bin/env python3
"""Analyze prompt_style experiment results across multiple result directories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


DEFAULT_DIRS = [
    "20260316_110315",
    "20260316_152854",
    "20260316_165553",
    "20260316_195913",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SimWorld benchmark runs by prompt_style."
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        default=DEFAULT_DIRS,
        help="Result subdirectories under results/. Default: %(default)s",
    )
    parser.add_argument(
        "--results-root",
        default=str(Path(__file__).parent.parent / "results"),
        help="Root path containing result directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent.parent / "results" / "prompt_style_analysis"),
        help="Directory for generated csv/markdown/plots.",
    )
    parser.add_argument(
        "--with-pie",
        action="store_true",
        help="Also generate success/failure pie charts by prompt style.",
    )
    return parser.parse_args()


def safe_avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def safe_std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return pstdev(values)


def fmt(v: float | None, decimals: int = 3) -> str:
    if v is None:
        return "-"
    return f"{v:.{decimals}f}"


def compute_penalty(record: dict) -> float:
    collision_count = float(record.get("collision_count", 0) or 0)
    passive_collision_count = float(record.get("passive_collision_count", 0) or 0)
    fall_count = float(record.get("fall_count", 0) or 0)
    oil_count = float(record.get("oil_count", 0) or 0)
    water_count = float(record.get("water_count", 0) or 0)
    return collision_count - passive_collision_count + fall_count + oil_count + water_count


def load_rows(results_root: Path, dirs: list[str]) -> list[dict]:
    rows: list[dict] = []
    for dirname in dirs:
        folder = results_root / dirname
        if not folder.exists():
            print(f"Skip (not found): {folder}")
            continue

        for path in sorted(folder.glob("*.json")):
            try:
                with path.open(encoding="utf-8") as f:
                    rec = json.load(f)
            except Exception as e:
                print(f"Skip (load failed): {path} ({e})")
                continue

            success = 1.0 if rec.get("success") else 0.0
            sim_time = rec.get("sim_time")
            optimal_time = rec.get("optimal_time")
            if (
                success == 1.0
                and isinstance(sim_time, (int, float))
                and isinstance(optimal_time, (int, float))
                and sim_time > 0
            ):
                efficiency = float(optimal_time) / float(sim_time)
            else:
                efficiency = 0.0

            row = {
                "run_dir": dirname,
                "file_name": path.name,
                "task_id": rec.get("task_id"),
                "difficulty": rec.get("difficulty"),
                "prompt_style": rec.get("prompt_style", "unknown"),
                "model": rec.get("model"),
                "success": success,
                "sim_time": rec.get("sim_time"),
                "time_cost": rec.get("time_cost"),
                "avg_response_time": rec.get("avg_response_time"),
                "avg_token_count": rec.get("avg_token_count"),
                "avg_completion_tokens": rec.get("avg_completion_tokens"),
                "avg_reasoning_tokens": rec.get("avg_reasoning_tokens"),
                "penalty": compute_penalty(rec),
                "efficiency": efficiency,
            }
            rows.append(row)
    return rows


def aggregate(rows: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in key_fields)].append(r)

    out: list[dict] = []
    for keys, grp in grouped.items():
        item = {k: v for k, v in zip(key_fields, keys)}
        item["count"] = len(grp)
        success_vals = [r["success"] for r in grp if isinstance(r.get("success"), (int, float))]
        success_rows = [r for r in grp if r.get("success") == 1.0]

        item["success_rate"] = safe_avg(success_vals)
        item["efficiency_mean"] = safe_avg([r["efficiency"] for r in grp if isinstance(r.get("efficiency"), (int, float))])
        item["efficiency_std"] = safe_std([r["efficiency"] for r in grp if isinstance(r.get("efficiency"), (int, float))])

        item["penalty_mean_success_only"] = safe_avg(
            [r["penalty"] for r in success_rows if isinstance(r.get("penalty"), (int, float))]
        )
        item["sim_time_mean_success_only"] = safe_avg(
            [r["sim_time"] for r in success_rows if isinstance(r.get("sim_time"), (int, float))]
        )
        item["time_cost_mean_success_only"] = safe_avg(
            [r["time_cost"] for r in success_rows if isinstance(r.get("time_cost"), (int, float))]
        )

        item["avg_response_time_mean"] = safe_avg(
            [r["avg_response_time"] for r in grp if isinstance(r.get("avg_response_time"), (int, float))]
        )
        item["avg_token_count_mean"] = safe_avg(
            [r["avg_token_count"] for r in grp if isinstance(r.get("avg_token_count"), (int, float))]
        )
        out.append(item)
    return sorted(out, key=lambda x: tuple(x[k] for k in key_fields))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_bar(
    rows: list[dict],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    styles = sorted({str(r["prompt_style"]) for r in rows})
    task_ids = sorted({r["task_id"] for r in rows if isinstance(r.get("task_id"), int)})
    if not styles or not task_ids:
        return

    matrix: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        task_id = r.get("task_id")
        value = r.get(metric)
        if isinstance(task_id, int) and isinstance(value, (int, float)):
            matrix[str(r["prompt_style"])][task_id].append(float(value))

    x = list(range(len(task_ids)))
    width = 0.35 if len(styles) == 2 else max(0.15, 0.8 / len(styles))

    plt.figure(figsize=(9, 4.8))
    for i, style in enumerate(styles):
        vals = [safe_avg(matrix[style][tid]) for tid in task_ids]
        vals = [0.0 if v is None or math.isnan(v) else v for v in vals]
        offset = (i - (len(styles) - 1) / 2) * width
        plt.bar([xi + offset for xi in x], vals, width=width, label=style)

    plt.xticks(x, [f"task_{tid}" for tid in task_ids])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_penalty_box(rows: list[dict], output_path: Path) -> None:
    styles = sorted({str(r["prompt_style"]) for r in rows})
    data: list[list[float]] = []
    labels: list[str] = []
    for style in styles:
        vals = [
            float(r["penalty"])
            for r in rows
            if str(r.get("prompt_style")) == style
            and r.get("success") == 1.0
            and isinstance(r.get("penalty"), (int, float))
        ]
        if vals:
            data.append(vals)
            labels.append(style)

    if not data:
        return

    plt.figure(figsize=(8, 4.8))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylabel("Penalty (success-only)")
    plt.title("Penalty Distribution by Prompt Style")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_success_pies(rows: list[dict], output_dir: Path) -> None:
    styles = sorted({str(r["prompt_style"]) for r in rows})
    for style in styles:
        style_rows = [r for r in rows if str(r.get("prompt_style")) == style]
        if not style_rows:
            continue
        success_count = sum(1 for r in style_rows if r.get("success") == 1.0)
        fail_count = len(style_rows) - success_count
        plt.figure(figsize=(4.8, 4.8))
        plt.pie(
            [success_count, fail_count],
            labels=["success", "failure"],
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title(f"Outcome Composition - {style}")
        plt.tight_layout()
        plt.savefig(output_dir / f"success_pie_{style}.png", dpi=150)
        plt.close()


def write_report(
    output_path: Path,
    dirs: list[str],
    style_summary: list[dict],
    style_task_summary: list[dict],
) -> None:
    lines: list[str] = [
        "# Prompt Style Analysis",
        "",
        f"**Source directories:** {', '.join(dirs)}",
        "",
        "## Which plots are suitable?",
        "",
        "- Best baseline: **tables + grouped bar charts** (easy to compare prompt_style across tasks).",
        "- Recommended supplement: **boxplot** for penalty variability (shows stability, not just mean).",
        "- Pie chart is optional and only useful for **success/failure composition**; it is weak for multi-metric comparison.",
        "",
        "## Overall by Prompt Style",
        "",
        "| prompt_style | n | success_rate | efficiency_mean | penalty_mean_success_only | sim_time_mean_success_only | time_cost_mean_success_only |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in style_summary:
        lines.append(
            "| "
            + f"{r['prompt_style']} | {r['count']} | {fmt(r['success_rate'])} | {fmt(r['efficiency_mean'])} | "
            + f"{fmt(r['penalty_mean_success_only'])} | {fmt(r['sim_time_mean_success_only'])} | {fmt(r['time_cost_mean_success_only'])} |"
        )

    lines.extend(
        [
            "",
            "## By Prompt Style and Task",
            "",
            "| prompt_style | task_id | n | success_rate | efficiency_mean | penalty_mean_success_only | sim_time_mean_success_only | time_cost_mean_success_only |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in style_task_summary:
        lines.append(
            "| "
            + f"{r['prompt_style']} | {r['task_id']} | {r['count']} | {fmt(r['success_rate'])} | {fmt(r['efficiency_mean'])} | "
            + f"{fmt(r['penalty_mean_success_only'])} | {fmt(r['sim_time_mean_success_only'])} | {fmt(r['time_cost_mean_success_only'])} |"
        )

    lines.extend(
        [
            "",
            "## Generated Figures",
            "",
            "- `success_rate_by_task.png`",
            "- `efficiency_by_task.png`",
            "- `time_cost_by_task_success_only.png`",
            "- `penalty_boxplot_by_style.png`",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(results_root=results_root, dirs=args.dirs)
    if not rows:
        print("No data found.")
        return

    style_summary = aggregate(rows, ("prompt_style",))
    style_task_summary = aggregate(rows, ("prompt_style", "task_id"))

    write_csv(output_dir / "run_level.csv", rows)
    write_csv(output_dir / "style_summary.csv", style_summary)
    write_csv(output_dir / "style_task_summary.csv", style_task_summary)

    plot_grouped_bar(
        rows=rows,
        metric="success",
        title="Success Rate by Task and Prompt Style",
        ylabel="success rate",
        output_path=output_dir / "success_rate_by_task.png",
    )
    plot_grouped_bar(
        rows=rows,
        metric="efficiency",
        title="Success-Weighted Efficiency by Task and Prompt Style",
        ylabel="efficiency",
        output_path=output_dir / "efficiency_by_task.png",
    )

    success_rows = [r for r in rows if r.get("success") == 1.0]
    plot_grouped_bar(
        rows=success_rows,
        metric="time_cost",
        title="Time Cost by Task and Prompt Style (Success-Only)",
        ylabel="time cost",
        output_path=output_dir / "time_cost_by_task_success_only.png",
    )
    plot_penalty_box(rows, output_dir / "penalty_boxplot_by_style.png")

    if args.with_pie:
        plot_success_pies(rows, output_dir)

    write_report(
        output_path=output_dir / "analysis_report.md",
        dirs=args.dirs,
        style_summary=style_summary,
        style_task_summary=style_task_summary,
    )

    print(f"Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
