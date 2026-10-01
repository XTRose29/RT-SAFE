"""
Walker waypoint sampler: 为不规则行走的行人采样密集 waypoint 点.
仿照 obstacle_sampler，使用独立初始化的 map：更密的纵向距离、更宽的横向 offset.
同时在 sidewalk 的 middle 点上采样 falling objects 并写入 obstacles.json.
"""
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from simworld.map.map import Node
from rt_map import RTMap
from simworld.config import Config

# 行人 waypoint 用图：纵向更密、横向更宽（相对 obstacle 的 800/120）
DEFAULT_WAYPOINTS_DISTANCE = 400       # 纵向：沿路每 400 一格（obstacle 为 800）
DEFAULT_WAYPOINTS_NORMAL_DISTANCE = 200  # 横向：offset 基准 200（obstacle 为 120）

# Falling object blueprint 候选列表
FALLING_OBJECT_BLUEPRINTS = [
    {"id_prefix": "RT_Falling_Basketball", "instance_name": "RT_Falling_Basketball_C"},
    {"id_prefix": "RT_Falling_Box",        "instance_name": "RT_Falling_Box_C"},
    {"id_prefix": "RT_Falling_Box2",       "instance_name": "RT_Falling_Box2_C"},
    {"id_prefix": "RT_Falling_Soccer",     "instance_name": "RT_Falling_Soccer_C"},
]
DEFAULT_FALLING_OBJECTS_PER_SIDEWALK = 10
FALLING_OBJECT_Z = 500  # 高度，物体从此处下落


class WalkerWaypointSampler:
    """在 sidewalk 上采样密集 waypoint，供不规则行走的行人使用。"""

    def __init__(
        self,
        config: Config,
        waypoints_distance: float = DEFAULT_WAYPOINTS_DISTANCE,
        waypoints_normal_distance: float = DEFAULT_WAYPOINTS_NORMAL_DISTANCE,
    ):
        """初始化 sampler。

        Args:
            config: 配置（含 traffic.sidewalk_offset 等）
            waypoints_distance: 沿路插值间距（越小越密）
            waypoints_normal_distance: 横向 offset 基准（越大横向越宽）
        """
        self.config = config
        self.waypoints_distance = waypoints_distance
        self.waypoints_normal_distance = waypoints_normal_distance
        self.waypoints: List[Dict[str, Any]] = []

    def build_map(self, roads_path: str) -> RTMap:
        """用更密纵向、更宽横向参数重新初始化 map，用于采样行人 waypoint。

        Args:
            roads_path: roads.json 路径

        Returns:
            初始化好的 RTMap（已插值）
        """
        map = RTMap(self.config)
        map.initialize_map_from_file(
            roads_path,
            self.config["traffic.sidewalk_offset"],
            True,
            waypoints_distance=self.waypoints_distance,
            waypoints_normal_distance=self.waypoints_normal_distance,
        )
        return map

    @staticmethod
    def _nodes_to_points(nodes: list) -> List[Dict[str, float]]:
        """将 Node 列表转为 [{"x": ..., "y": ...}, ...]"""
        return [{"x": n.position.x, "y": n.position.y} for n in nodes]

    def sample_waypoints(self, map: RTMap) -> List[Dict[str, Any]]:
        """按 sidewalk 组织采样结果，每条 sidewalk 维护三列有序点。

        Args:
            map: 由 build_map() 得到的 RTMap

        Returns:
            列表，每项为一条 sidewalk::

                {
                    "sidewalk_id": int,
                    "start": {"x": float, "y": float},
                    "end":   {"x": float, "y": float},
                    "far_road":  [{"x": ..., "y": ...}, ...],
                    "middle":    [{"x": ..., "y": ...}, ...],
                    "near_road": [{"x": ..., "y": ...}, ...],
                }
        """
        sidewalk_groups = map.get_sidewalk_groups()
        result = []
        for idx, group in enumerate(sidewalk_groups):
            result.append({
                "sidewalk_id": idx,
                "start": {"x": group["start"].x, "y": group["start"].y},
                "end":   {"x": group["end"].x,   "y": group["end"].y},
                "far_road":  self._nodes_to_points(group["far_road"]),
                "middle":    self._nodes_to_points(group["middle"]),
                "near_road": self._nodes_to_points(group["near_road"]),
            })
        self.waypoints = result
        return result

    def run(
        self,
        roads_path: str,
        output_json_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """一站式：构建 map -> 采样 waypoint -> 可选写 JSON。

        Args:
            roads_path: roads.json 路径
            output_json_path: 若提供则把 waypoints 写入该 JSON

        Returns:
            按 sidewalk 组织的 waypoints 列表
        """
        map = self.build_map(roads_path)
        waypoints = self.sample_waypoints(map)
        if output_json_path:
            self.export_waypoints_json(waypoints, output_json_path)
        return waypoints

    def sample_falling_objects(
        self,
        waypoints: List[Dict[str, Any]],
        num_per_sidewalk: int = DEFAULT_FALLING_OBJECTS_PER_SIDEWALK,
    ) -> List[Dict[str, Any]]:
        """从每条 sidewalk 的 middle 点中采样 falling objects.

        为每条 sidewalk 随机选 num_per_sidewalk 个 middle 点，
        生成 obstacles.json 格式的节点条目，并在每条 sidewalk 上
        记录其对应的 obstacle ID 列表（falling_objects 字段）。

        Args:
            waypoints: sample_waypoints() 返回的列表（会被原地修改，添加 falling_objects 字段）
            num_per_sidewalk: 每条 sidewalk 采样的 falling object 数量

        Returns:
            obstacles.json 格式的 node 列表
        """
        obstacle_entries: List[Dict[str, Any]] = []
        global_idx = 0

        for sidewalk in waypoints:
            middle = sidewalk.get("middle", [])
            if not middle:
                sidewalk["falling_objects"] = []
                continue

            k = min(num_per_sidewalk, len(middle))
            sampled = random.sample(middle, k)

            fo_ids: List[str] = []
            for point in sampled:
                bp = random.choice(FALLING_OBJECT_BLUEPRINTS)
                obj_id = f"GEN_RT_FO_{bp['id_prefix']}_{global_idx}"
                obstacle_entries.append({
                    "id": obj_id,
                    "instance_name": bp["instance_name"],
                    "properties": {
                        "location": {"x": point["x"], "y": point["y"], "z": FALLING_OBJECT_Z},
                        "orientation": {"pitch": 0, "yaw": 0, "roll": 0},
                        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                    },
                })
                fo_ids.append(obj_id)
                global_idx += 1

            sidewalk["falling_objects"] = fo_ids

        return obstacle_entries

    @staticmethod
    def append_to_obstacles_json(
        obstacles_path: str,
        new_entries: List[Dict[str, Any]],
        id_prefix: str = "GEN_RT_FO_",
    ) -> None:
        """将 falling object 条目追加到已有的 obstacles.json 中。

        先移除旧的 falling object 条目（按 id_prefix 匹配），再追加新的。

        Args:
            obstacles_path: obstacles.json 文件路径
            new_entries: sample_falling_objects() 返回的条目列表
            id_prefix: 旧条目的 ID 前缀，用于清理
        """
        with open(obstacles_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 移除旧的 falling object 条目
        data["nodes"] = [
            n for n in data["nodes"]
            if not n.get("id", "").startswith(id_prefix)
        ]

        # 追加新条目
        data["nodes"].extend(new_entries)

        with open(obstacles_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_waypoints_json(self, waypoints: List[Dict[str, Any]], output_path: str) -> None:
        """将 waypoints 导出为 JSON 文件。

        Args:
            waypoints: sample_waypoints() 返回的列表
            output_path: 输出文件路径
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"sidewalks": waypoints}, f, indent=2, ensure_ascii=False)


# Example usage
if __name__ == "__main__":
    random.seed(42)
    config = Config(str(Path(__file__).resolve().parents[1] / "config.yaml"))

    # 行人 waypoint 用：更密纵向、更宽横向
    sampler = WalkerWaypointSampler(
        config,
        waypoints_distance=500,
        waypoints_normal_distance=200,
    )

    maps = [
        "map1_10roads",
        "map2_12roads",
        "map3_15roads",
        "map4_18roads",
        "map5_20roads",
    ]

    for map_name in maps:
        print(f"Processing {map_name}...")
        roads_path = rf"E:\Projects\SimWorld-RBench\data\{map_name}\roads.json"
        output_path = rf"data\{map_name}\walker_waypoints.json"
        obstacles_path = rf"data\{map_name}\obstacles.json"

        # 1. 采样 waypoints
        map_obj = sampler.build_map(roads_path)
        sidewalks = sampler.sample_waypoints(map_obj)

        # 2. 采样 falling objects（会在每条 sidewalk 上写入 falling_objects 字段）
        fo_entries = sampler.sample_falling_objects(sidewalks)

        # 3. 写入 walker_waypoints.json（含 falling_objects ID 列表）
        sampler.export_waypoints_json(sidewalks, output_path)

        # 4. 追加 falling objects 到 obstacles.json
        if os.path.exists(obstacles_path):
            sampler.append_to_obstacles_json(obstacles_path, fo_entries)
            print(f"  - Appended {len(fo_entries)} falling objects to {obstacles_path}")
        else:
            print(f"  - Warning: {obstacles_path} not found, skipping falling objects")

        total_pts = sum(
            len(s["far_road"]) + len(s["middle"]) + len(s["near_road"])
            for s in sidewalks
        )
        print(f"  - {len(sidewalks)} sidewalks, {total_pts} total waypoints")
        print(f"  - Saved to {output_path}")

    print("\nAll maps processed successfully!")
