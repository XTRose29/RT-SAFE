"""
可视化脚本：同时展示 roads、walker_waypoints 和 obstacles。
支持：滚轮缩放、鼠标中键/右键拖拽平移、双击还原视图。
用法: python visualize_waypoints_obstacles.py
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse


# ───────────────────────── 数据加载 ──────────────────────────

def load_roads(roads_path: str):
    """加载 roads.json。"""
    with open(roads_path, "r") as f:
        data = json.load(f)
    return data["roads"]


def load_walker_waypoints(wp_path: str):
    """加载 walker_waypoints.json（按 sidewalk 组织的新格式）。"""
    with open(wp_path, "r") as f:
        data = json.load(f)
    return data["sidewalks"]


def load_obstacles(obs_path: str):
    """加载 obstacles.json (progen_world 格式)。"""
    with open(obs_path, "r") as f:
        data = json.load(f)
    return data["nodes"]


def classify_obstacle(instance_name: str) -> str:
    """根据 instance_name 判断障碍物类型。"""
    if instance_name.startswith("BP_Tree"):
        return "tree"
    elif instance_name.startswith("RT_"):
        return "rt"
    else:
        return "other"


# ───────────────── 鼠标交互：缩放 & 平移 ─────────────────────

class MouseInteraction:
    """给 matplotlib axes 绑定滚轮缩放 + 鼠标拖拽平移 + 双击还原。"""

    ZOOM_FACTOR = 1.3  # 每次滚轮缩放倍率

    def __init__(self, ax):
        self.ax = ax
        self.fig = ax.figure
        self._press_event = None          # 按下时的事件
        self._xlim_on_press = None
        self._ylim_on_press = None
        self._home_xlim = None            # 初始视图范围（用于双击还原）
        self._home_ylim = None

        # 连接事件
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def save_home(self):
        """保存当前视图范围作为"主页"。"""
        self._home_xlim = self.ax.get_xlim()
        self._home_ylim = self.ax.get_ylim()

    # ── 滚轮缩放（以光标为中心）──
    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        xdata, ydata = event.xdata, event.ydata
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()

        if event.button == "up":
            scale = 1 / self.ZOOM_FACTOR
        elif event.button == "down":
            scale = self.ZOOM_FACTOR
        else:
            return

        new_xlim = [xdata + (x - xdata) * scale for x in xlim]
        new_ylim = [ydata + (y - ydata) * scale for y in ylim]
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.fig.canvas.draw_idle()

    # ── 按下（右键 or 中键开始拖拽；双击左键还原）──
    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        # 双击左键 → 还原
        if event.dblclick and event.button == 1:
            if self._home_xlim and self._home_ylim:
                self.ax.set_xlim(self._home_xlim)
                self.ax.set_ylim(self._home_ylim)
                self.fig.canvas.draw_idle()
            return
        # 右键(3) 或 中键(2) → 开始拖拽
        if event.button in (2, 3):
            self._press_event = event
            self._xlim_on_press = self.ax.get_xlim()
            self._ylim_on_press = self.ax.get_ylim()

    def _on_release(self, event):
        self._press_event = None

    # ── 拖拽平移 ──
    def _on_motion(self, event):
        if self._press_event is None or event.inaxes != self.ax:
            return
        dx = event.xdata - self._press_event.xdata
        dy = event.ydata - self._press_event.ydata
        self.ax.set_xlim(self._xlim_on_press[0] - dx, self._xlim_on_press[1] - dx)
        self.ax.set_ylim(self._ylim_on_press[0] - dy, self._ylim_on_press[1] - dy)
        self.fig.canvas.draw_idle()


# ──────────────────────── 绘图主函数 ──────────────────────────

def visualize(roads_path: str, wp_path: str, obs_path: str):
    roads = load_roads(roads_path)
    sidewalks = load_walker_waypoints(wp_path)
    obstacles = load_obstacles(obs_path)

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    # ── 1. 道路 ──────────────────────────────────────────────
    for road in roads:
        sx, sy = road["start"]["x"], road["start"]["y"]
        ex, ey = road["end"]["x"], road["end"]["y"]
        ax.plot(
            [sx, ex], [sy, ey],
            color="#555577", linewidth=4, solid_capstyle="round", zorder=1,
        )

    # ── 2. Walker waypoints（按 sidewalk 组织，三列分色）──────
    lane_style = {
        "far_road":  {"color": "#66d9ef", "marker": ".", "s": 12, "label": "WP: far_road"},
        "middle":    {"color": "#a6e22e", "marker": ".", "s": 12, "label": "WP: middle"},
        "near_road": {"color": "#e6db74", "marker": ".", "s": 12, "label": "WP: near_road"},
    }
    # 汇总所有 sidewalk 的三列点
    total_wp = 0
    lane_points: dict[str, list] = {"far_road": [], "middle": [], "near_road": []}
    for sw in sidewalks:
        for lane in ("far_road", "middle", "near_road"):
            lane_points[lane].extend(sw[lane])
            total_wp += len(sw[lane])

    for lane, pts in lane_points.items():
        style = lane_style[lane]
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        ax.scatter(xs, ys, c=style["color"], marker=style["marker"],
                   s=style["s"], alpha=0.6, linewidths=0, zorder=2, label=style["label"])

    # ── 3. Obstacles（按类型分色，用大标记）─────────────────
    obs_style = {
        "tree":  {"color": "#f92672", "marker": "^", "s": 60, "label": "Obstacle: tree"},
        "rt":    {"color": "#fd971f", "marker": "s", "s": 50, "label": "Obstacle: RT"},
        "other": {"color": "#f8f8f2", "marker": "D", "s": 40, "label": "Obstacle: other"},
    }
    obs_groups: dict[str, list] = {}
    for node in obstacles:
        obs_type = classify_obstacle(node["instance_name"])
        obs_groups.setdefault(obs_type, []).append(node)

    for obs_type, nodes in obs_groups.items():
        style = obs_style.get(obs_type, {"color": "red", "marker": "x", "s": 40, "label": obs_type})
        xs = [n["properties"]["location"]["x"] for n in nodes]
        ys = [n["properties"]["location"]["y"] for n in nodes]
        ax.scatter(xs, ys, c=style["color"], marker=style["marker"],
                   s=style["s"], edgecolors="white", linewidths=0.3,
                   alpha=0.85, zorder=3, label=style["label"])

    # ── 4. 美化 ──────────────────────────────────────────────
    ax.set_aspect("equal")
    ax.set_xlabel("X", color="white", fontsize=11)
    ax.set_ylabel("Y", color="white", fontsize=11)
    ax.set_title(
        "Walker Waypoints & Obstacles  (scroll=zoom, right-drag=pan, dblclick=reset)",
        color="white", fontsize=13, fontweight="bold",
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444466")

    # legend（去重）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="upper left", fontsize=9, framealpha=0.7,
        facecolor="#2e2e3e", edgecolor="#555577", labelcolor="white",
    )

    # 统计信息
    stats_text = (
        f"Roads: {len(roads)}   "
        f"Sidewalks: {len(sidewalks)}   "
        f"Waypoints: {total_wp}   "
        f"Obstacles: {len(obstacles)}"
    )
    ax.text(
        0.5, -0.04, stats_text,
        transform=ax.transAxes, ha="center", fontsize=10, color="#888899",
    )

    plt.tight_layout()

    # ── 5. 绑定鼠标交互 ─────────────────────────────────────
    interaction = MouseInteraction(ax)
    interaction.save_home()

    plt.show()


def main():
    roads = r"data\map1_10roads\roads.json"
    waypoints = r"data\map1_10roads\walker_waypoints.json"
    obstacles = r"data\map1_10roads\obstacles.json"
    visualize(roads, waypoints, obstacles)


if __name__ == "__main__":
    main()
