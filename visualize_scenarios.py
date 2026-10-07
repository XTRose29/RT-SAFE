"""
Visualize tasks from tasks.json files and save images to the corresponding map folder.
Uses roads.json to draw the full map for context.

Output:
    Images saved to each map folder: task_0.png, task_1.png, ...
"""
import json
import math
import os
import matplotlib.pyplot as plt

# List of tasks.json paths to visualize
TASKS_FILES = [
    "data/map1_10roads/tasks.json",
    "data/map2_12roads/tasks.json",
    "data/map3_15roads/tasks.json",
    "data/map4_18roads/tasks.json",
    "data/map5_20roads/tasks.json",
]
SIDEWALK_OFFSET = 700
ROAD_SCALE = 100  # roads.json units to map units (segment_length)


def _build_map_edges_from_roads(roads_path: str):
    """Build map edges (sidewalks + crosswalks) from roads.json, same coord system as tasks."""
    with open(roads_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    roads = data["roads"]
    edges = []

    def vec(x, y):
        return [float(x), float(y)]

    def sub(a, b):
        return [a[0] - b[0], a[1] - b[1]]

    def add(a, b):
        return [a[0] + b[0], a[1] + b[1]]

    def scale(v, s):
        return [v[0] * s, v[1] * s]

    def norm(v):
        L = math.sqrt(v[0] ** 2 + v[1] ** 2)
        return [v[0] / L, v[1] / L] if L > 1e-9 else [1, 0]

    for road in roads:
        start = vec(road["start"]["x"], road["start"]["y"])
        end = vec(road["end"]["x"], road["end"]["y"])
        start = scale(start, ROAD_SCALE)
        end = scale(end, ROAD_SCALE)
        d = sub(end, start)
        direction = norm(d)
        normal = [direction[1], -direction[0]]
        off = SIDEWALK_OFFSET
        p1 = add(sub(start, scale(normal, off)), scale(direction, off))
        p2 = add(sub(end, scale(normal, off)), scale(direction, -off))
        p3 = add(add(end, scale(normal, off)), scale(direction, -off))
        p4 = add(add(start, scale(normal, off)), scale(direction, off))
        edges.append((p1, p2, "sidewalk"))
        edges.append((p3, p4, "sidewalk"))
        edges.append((p1, p4, "crosswalk"))
        edges.append((p2, p3, "crosswalk"))

    return edges


def visualize_and_save_task(task: dict, map_edges: list, output_path: str):
    """
    Visualize a single task on the full map and save to file.

    Args:
        task: Task dict with route_info, pedestrian_routes, etc.
        map_edges: List of (node1_pos, node2_pos, edge_type) from Map
        output_path: Path to save the image (e.g., data/map1_10roads/task_0.png)
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    # Draw full map (all edges from roads.json)
    for p1, p2, etype in map_edges:
        color = "#66d9ef" if etype == "crosswalk" else "#555577"
        lw = 2 if etype == "crosswalk" else 3
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=lw,
                solid_capstyle="round", zorder=1, alpha=0.8)

    # Draw pedestrian routes (lighter, thinner)
    if "pedestrian_routes" in task and task["pedestrian_routes"]:
        for ped_route in task["pedestrian_routes"]:
            route = ped_route.get("route", [])
            if len(route) >= 2:
                xs = [p[0] for p in route]
                ys = [p[1] for p in route]
                ax.plot(xs, ys, color="#66d9ef", linewidth=1, alpha=0.5, zorder=2)

    # Draw agent shortest path (main route, prominent)
    if "route_info" in task and "shortest_path" in task["route_info"]:
        path = task["route_info"]["shortest_path"]
        if len(path) >= 2:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color="#a6e22e", linewidth=4, alpha=0.9, zorder=4, label="Agent route")

    # Draw start and end points
    start = task.get("start_point", [0, 0])
    end = task.get("end_point", [0, 0])
    ax.scatter([start[0]], [start[1]], c="#a6e22e", s=120, marker="o", edgecolors="white",
               linewidths=2, zorder=5, label="Start")
    ax.scatter([end[0]], [end[1]], c="#f92672", s=120, marker="s", edgecolors="white",
               linewidths=2, zorder=5, label="End")

    # Styling
    ax.set_aspect("equal")
    ax.set_xlabel("X", color="white", fontsize=11)
    ax.set_ylabel("Y", color="white", fontsize=11)
    ax.set_title(
        f"Task {task.get('task_id', '?')} | Hops: {task.get('total_hops', '?')} | "
        f"Crosswalks: {task.get('crosswalk_hops', '?')}",
        color="white", fontsize=12, fontweight="bold",
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444466")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7,
              facecolor="#2e2e3e", edgecolor="#555577", labelcolor="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_tasks_file(tasks_file_path: str):
    """
    Read a tasks.json file, visualize all tasks on the full map, and save to the map folder.

    Args:
        tasks_file_path: Path to tasks.json (e.g., data/map1_10roads/tasks.json)
    """
    tasks_file_path = os.path.normpath(tasks_file_path)
    if not os.path.exists(tasks_file_path):
        raise FileNotFoundError(f"Tasks file not found: {tasks_file_path}")

    with open(tasks_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    if not tasks:
        print(f"No tasks found in {tasks_file_path}")
        return

    # Output folder = directory containing tasks.json (e.g., data/map1_10roads)
    output_dir = os.path.dirname(os.path.abspath(tasks_file_path))
    roads_path = os.path.join(output_dir, "roads.json")
    if not os.path.exists(roads_path):
        raise FileNotFoundError(f"Roads file not found: {roads_path}")

    # Build full map edges from roads.json (same coord system as tasks)
    map_edges = _build_map_edges_from_roads(roads_path)

    print(f"Loaded {len(tasks)} tasks from {tasks_file_path}")
    print(f"Map: {len(map_edges)} edges from {roads_path}")
    print(f"Output dir: {output_dir}")
    print()

    for task in tasks:
        task_id = task.get("task_id", len(tasks))
        output_path = os.path.join(output_dir, f"task_{task_id}.png")
        visualize_and_save_task(task, map_edges, output_path)


def main():
    try:
        for tasks_file in TASKS_FILES:
            if os.path.exists(tasks_file):
                visualize_tasks_file(tasks_file)
            else:
                print(f"Skipped (not found): {tasks_file}")
        print("\nDone.")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
