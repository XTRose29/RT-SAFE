"""Deterministic validation of benchmark configs, maps, and UE asset metadata."""

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SimWorld"))

from simworld.config import Config
from utils.task_routes import point_on_segment, reconstruct_route_points


def segment_intersects_expanded_bounds(
    start,
    end,
    bounds,
    clearance_cm,
):
    """Return whether a route segment enters an AABB plus its safety margin."""
    x_min = bounds["x"] * 100.0 - clearance_cm
    y_min = bounds["y"] * 100.0 - clearance_cm
    x_max = (bounds["x"] + bounds["width"]) * 100.0 + clearance_cm
    y_max = (bounds["y"] + bounds["height"]) * 100.0 + clearance_cm
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    entry = 0.0
    exit_ = 1.0

    for direction, distance in (
        (-dx, x0 - x_min),
        (dx, x_max - x0),
        (-dy, y0 - y_min),
        (dy, y_max - y0),
    ):
        if direction == 0:
            if distance < 0:
                return False
            continue
        fraction = distance / direction
        if direction < 0:
            entry = max(entry, fraction)
        else:
            exit_ = min(exit_, fraction)
        if entry > exit_:
            return False
    return True


class BenchmarkAssetTests(unittest.TestCase):
    MAPS = (
        "map1_10roads",
        "map2_12roads",
        "map3_15roads",
        "map4_18roads",
        "map5_20roads",
    )

    def test_runtime_config_has_every_directly_indexed_signal_key(self):
        config = Config(str(REPO_ROOT / "config.yaml"))
        for key in (
            "green_light_duration",
            "yellow_light_duration",
            "pedestrian_phase_duration",
            "pedestrian_green_light_duration",
        ):
            self.assertIsInstance(
                config[f"traffic.traffic_signal.{key}"],
                (int, float),
            )
        self.assertNotIn("citygen", config.config)
        self.assertNotIn("assets_rp", config.config)
        self.assertEqual(
            config["traffic.traffic_signal.intersection_controller_model_path"],
            "/Game/RealTimeBench/Traffic/RT_BP_Intersection.RT_BP_Intersection_C",
        )
        self.assertEqual(
            1.35,
            config["traffic.traffic_signal.traffic_light_scale"],
        )
        self.assertEqual(
            1.90,
            config["traffic.traffic_signal.pedestrian_light_scale"],
        )
        self.assertEqual(
            [3.0, 9.0],
            [
                config[
                    "traffic.traffic_signal.conflict_vehicle_min_launch_distance_m"
                ],
                config[
                    "traffic.traffic_signal.conflict_vehicle_max_launch_distance_m"
                ],
            ],
        )

    def test_all_map_task_dependencies_are_present(self):
        required_sidecars = (
            "buildings.json",
            "elements.json",
            "obstacles.json",
            "progen_world.json",
            "roads.json",
            "routes.json",
            "tasks.json",
            "walker_waypoints.json",
        )
        for map_name in self.MAPS:
            map_dir = REPO_ROOT / "data" / map_name
            with self.subTest(map=map_name):
                for name in required_sidecars:
                    self.assertTrue((map_dir / name).is_file(), name)
                tasks = json.loads(
                    (map_dir / "tasks.json").read_text(encoding="utf-8")
                )["tasks"]
                self.assertGreater(len(tasks), 0)
                for task in tasks:
                    map_path = REPO_ROOT / task["map_path"]
                    self.assertTrue(map_path.is_file(), task["map_path"])
                    self.assertGreater(task["total_hops"], 0)
                    self.assertGreater(
                        len(task["route_info"]["shortest_path"]),
                        1,
                    )

    def test_every_task_route_follows_its_ordered_edges(self):
        checked_tasks = 0
        stale_routes = set()
        for map_name in self.MAPS:
            tasks = json.loads(
                (REPO_ROOT / "data" / map_name / "tasks.json").read_text(
                    encoding="utf-8"
                )
            )["tasks"]
            for task in tasks:
                with self.subTest(map=map_name, task=task["task_id"]):
                    route = reconstruct_route_points(task)
                    edges = task["edges"]
                    self.assertEqual(len(edges) + 1, len(route))
                    self.assertEqual(task["start_point"], route[0])
                    self.assertEqual(task["end_point"], route[-1])
                    for index, edge in enumerate(edges):
                        self.assertTrue(
                            point_on_segment(
                                route[index], edge["node1"], edge["node2"]
                            )
                        )
                        self.assertTrue(
                            point_on_segment(
                                route[index + 1], edge["node1"], edge["node2"]
                            )
                        )
                    if route != task["route_info"]["shortest_path"]:
                        stale_routes.add((map_name, task["task_id"]))
                    checked_tasks += 1

        self.assertEqual(36, checked_tasks)
        self.assertEqual(
            {
                ("map1_10roads", 3),
                ("map2_12roads", 12),
                ("map3_15roads", 22),
                ("map4_18roads", 29),
                ("map4_18roads", 30),
                ("map5_20roads", 36),
                ("map5_20roads", 38),
                ("map5_20roads", 39),
            },
            stale_routes,
        )

    def test_route_reconstruction_fails_closed_on_disconnected_edges(self):
        task = {
            "start_point": [0, 0],
            "end_point": [30, 0],
            "edges": [
                {"node1": [0, 0], "node2": [10, 0]},
                {"node1": [20, 0], "node2": [30, 0]},
            ],
        }
        with self.assertRaisesRegex(ValueError, "exactly one endpoint"):
            reconstruct_route_points(task)

    def test_buildings_leave_one_metre_clearance_from_runtime_routes(self):
        violations = []
        for map_name in self.MAPS:
            map_dir = REPO_ROOT / "data" / map_name
            buildings = json.loads(
                (map_dir / "buildings.json").read_text(encoding="utf-8")
            )["buildings"]
            tasks = json.loads(
                (map_dir / "tasks.json").read_text(encoding="utf-8")
            )["tasks"]
            for task in tasks:
                route = reconstruct_route_points(task)
                for building_index, building in enumerate(buildings):
                    if any(
                        segment_intersects_expanded_bounds(
                            start,
                            end,
                            building["bounds"],
                            clearance_cm=100.0,
                        )
                        for start, end in zip(route, route[1:])
                    ):
                        violations.append(
                            (map_name, task["task_id"], building_index)
                        )

        self.assertEqual([], violations)

    def test_building_bounds_centres_match_spawned_unreal_actors(self):
        for map_name in self.MAPS:
            map_dir = REPO_ROOT / "data" / map_name
            road_count = int(map_name.split("_")[1].removesuffix("roads"))
            buildings = json.loads(
                (map_dir / "buildings.json").read_text(encoding="utf-8")
            )["buildings"]
            nodes = {
                node["id"]: node
                for node in json.loads(
                    (map_dir / "progen_world.json").read_text(encoding="utf-8")
                )["nodes"]
            }
            for building_index, building in enumerate(buildings):
                actor_id = (
                    f"GEN_{building['type']}_{road_count + building_index}"
                )
                with self.subTest(map=map_name, actor=actor_id):
                    self.assertIn(actor_id, nodes)
                    location = nodes[actor_id]["properties"]["location"]
                    self.assertAlmostEqual(
                        building["center"]["x"],
                        location["x"] / 100.0,
                        places=4,
                    )
                    self.assertAlmostEqual(
                        building["center"]["y"],
                        location["y"] / 100.0,
                        places=4,
                    )

    def test_ue_asset_mapping_is_nonempty(self):
        assets = json.loads(
            (REPO_ROOT / "data" / "ue_assets.json").read_text(encoding="utf-8")
        )
        self.assertGreater(len(assets), 100)
