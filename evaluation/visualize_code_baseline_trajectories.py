#!/usr/bin/env python3
"""Render 2D trajectory overlays for code-baseline result JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = REPO_ROOT / "results" / "code_baselines_final_static_realtime5_easy_20260704"


COLORS = {
    "background": (250, 250, 247),
    "road": (205, 205, 198),
    "sidewalk": (120, 120, 112),
    "crosswalk": (224, 128, 48),
    "route": (32, 32, 32),
    "ped_route": (165, 195, 176),
    "scooter_route": (185, 178, 215),
    "trajectory": (35, 105, 205),
    "trajectory_realtime": (0, 135, 115),
    "step_fill": (255, 255, 255),
    "step_text": (20, 20, 20),
    "start": (34, 145, 70),
    "end": (210, 48, 48),
    "title": (25, 25, 25),
    "muted": (90, 90, 90),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def point_xy(point: Any) -> tuple[float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def collect_bounds(task: dict[str, Any], results: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    points: list[tuple[float, float]] = []
    points.extend(point_xy(p) for p in task["route_info"]["shortest_path"])
    for edge in task.get("edges", []):
        points.append(point_xy(edge["node1"]))
        points.append(point_xy(edge["node2"]))
    for route_key in ("pedestrian_routes", "scooter_routes", "irregular_routes"):
        for route in task.get(route_key, []):
            points.extend(point_xy(p) for p in route.get("route", []))
    for result in results:
        for decision in result.get("decisions", []):
            points.append(point_xy(decision["start"]))
            points.append(point_xy(decision["end"]))
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    pad_x = max(600.0, (max_x - min_x) * 0.08)
    pad_y = max(600.0, (max_y - min_y) * 0.08)
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


class Projector:
    def __init__(self, bounds: tuple[float, float, float, float], size: tuple[int, int], margin: int = 80) -> None:
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.width, self.height = size
        self.margin = margin
        span_x = max(1.0, self.max_x - self.min_x)
        span_y = max(1.0, self.max_y - self.min_y)
        self.scale = min((self.width - 2 * margin) / span_x, (self.height - 2 * margin) / span_y)
        used_w = span_x * self.scale
        used_h = span_y * self.scale
        self.offset_x = (self.width - used_w) / 2
        self.offset_y = (self.height - used_h) / 2

    def __call__(self, point: Any) -> tuple[int, int]:
        x, y = point_xy(point)
        px = self.offset_x + (x - self.min_x) * self.scale
        py = self.height - (self.offset_y + (y - self.min_y) * self.scale)
        return int(round(px)), int(round(py))


def draw_polyline(draw: ImageDraw.ImageDraw, projector: Projector, points: list[Any], fill: tuple[int, int, int], width: int) -> None:
    if len(points) < 2:
        return
    draw.line([projector(p) for p in points], fill=fill, width=width, joint="curve")


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    projector: Projector,
    a: Any,
    b: Any,
    fill: tuple[int, int, int],
    width: int,
    dash_px: int = 18,
) -> None:
    x1, y1 = projector(a)
    x2, y2 = projector(b)
    dx, dy = x2 - x1, y2 - y1
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    steps = max(1, int(length // dash_px))
    for i in range(0, steps, 2):
        t1 = i / steps
        t2 = min(1.0, (i + 1) / steps)
        draw.line(
            [
                (round(x1 + dx * t1), round(y1 + dy * t1)),
                (round(x1 + dx * t2), round(y1 + dy * t2)),
            ],
            fill=fill,
            width=width,
        )


def label_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    text_fill: tuple[int, int, int] = COLORS["step_text"],
    anchor: str = "mm",
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = xy
    if anchor == "mm":
        box = (x - tw // 2 - 5, y - th // 2 - 3, x + tw // 2 + 5, y + th // 2 + 4)
        text_xy = (x - tw // 2, y - th // 2 - 1)
    else:
        box = (x, y, x + tw + 10, y + th + 7)
        text_xy = (x + 5, y + 3)
    draw.rounded_rectangle(box, radius=4, fill=fill, outline=(55, 55, 55), width=1)
    draw.text(text_xy, text, font=font, fill=text_fill)


def render_panel(
    task: dict[str, Any],
    result: dict[str, Any],
    bounds: tuple[float, float, float, float],
    size: tuple[int, int],
) -> Image.Image:
    image = Image.new("RGB", size, COLORS["background"])
    draw = ImageDraw.Draw(image)
    projector = Projector(bounds, size)
    title_font = load_font(24, bold=True)
    small_font = load_font(14)
    step_font = load_font(12, bold=True)
    label_font = load_font(15, bold=True)

    for route in task.get("pedestrian_routes", []):
        draw_polyline(draw, projector, route.get("route", []), COLORS["ped_route"], 2)
    for route in task.get("scooter_routes", []):
        draw_polyline(draw, projector, route.get("route", []), COLORS["scooter_route"], 2)

    for edge in task.get("edges", []):
        color = COLORS["crosswalk"] if edge.get("type") == "crosswalk" else COLORS["sidewalk"]
        if edge.get("type") == "crosswalk":
            draw_dashed_line(draw, projector, edge["node1"], edge["node2"], color, 8)
        else:
            draw.line([projector(edge["node1"]), projector(edge["node2"])], fill=color, width=10)

    route_points = result.get("route", {}).get("shortest_path", [])
    draw_polyline(draw, projector, route_points, COLORS["route"], 4)

    decisions = result.get("decisions", [])
    trajectory = [decisions[0]["start"]] + [d["end"] for d in decisions] if decisions else []
    traj_color = COLORS["trajectory_realtime"] if result.get("env_mode") == "realtime" else COLORS["trajectory"]
    draw_polyline(draw, projector, trajectory, traj_color, 5)

    if trajectory:
        sx, sy = projector(trajectory[0])
        ex, ey = projector(trajectory[-1])
        draw.ellipse((sx - 8, sy - 8, sx + 8, sy + 8), fill=COLORS["start"], outline=(0, 0, 0), width=1)
        draw.ellipse((ex - 8, ey - 8, ex + 8, ey + 8), fill=COLORS["end"], outline=(0, 0, 0), width=1)
        label_box(draw, (sx + 34, sy - 18), "START", label_font, COLORS["start"], (255, 255, 255))
        label_box(draw, (ex + 28, ey + 20), "END", label_font, COLORS["end"], (255, 255, 255))

    for decision in decisions:
        px, py = projector(decision["end"])
        step = str(decision.get("step", "?"))
        label_box(draw, (px, py), step, step_font, COLORS["step_fill"])

    title = f"{result.get('baseline_policy')} baseline - {result.get('env_mode')} env"
    subtitle = (
        f"success={result.get('success')}  steps={result.get('final_step')}  "
        f"collisions={result.get('collision_count')}  red_lights={result.get('red_light_violation_count')}"
    )
    draw.text((24, 20), title, font=title_font, fill=COLORS["title"])
    draw.text((24, 52), subtitle, font=small_font, fill=COLORS["muted"])
    return image


def paste_grid(panels: list[tuple[str, Image.Image]], output: Path) -> None:
    if not panels:
        return
    panel_w, panel_h = panels[0][1].size
    gutter = 24
    width = panel_w * 2 + gutter * 3
    height = panel_h * 2 + gutter * 3
    image = Image.new("RGB", (width, height), (238, 238, 234))
    for idx, (_, panel) in enumerate(panels):
        row, col = divmod(idx, 2)
        x = gutter + col * (panel_w + gutter)
        y = gutter + row * (panel_h + gutter)
        image.paste(panel, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def result_paths(suite_dir: Path) -> list[Path]:
    order = [
        "safety__static__offline",
        "safety__realtime__offline",
        "greedy__static__offline",
        "greedy__realtime__offline",
    ]
    paths: list[Path] = []
    for setting in order:
        path = suite_dir / "runs" / setting / "round_01" / "result.json"
        if path.exists():
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=950)
    args = parser.parse_args()

    suite_dir = args.suite_dir if args.suite_dir.is_absolute() else REPO_ROOT / args.suite_dir
    out_dir = args.out_dir or suite_dir / "trajectory_plots"
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir

    results = [load_json(path) for path in result_paths(suite_dir)]
    if not results:
        raise SystemExit(f"No result JSONs found in {suite_dir}")
    task_path = REPO_ROOT / results[0]["task_file"]
    tasks = load_json(task_path)["tasks"]
    task = tasks[int(results[0]["task_index"])]
    bounds = collect_bounds(task, results)

    panels: list[tuple[str, Image.Image]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        panel = render_panel(task, result, bounds, (args.width, args.height))
        name = f"{result['baseline_policy']}_{result['env_mode']}_trajectory.png"
        panel.save(out_dir / name)
        panels.append((name, panel))

    paste_grid(panels, out_dir / "code_baseline_trajectory_grid.png")
    print(f"Wrote {len(panels) + 1} trajectory plot(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
