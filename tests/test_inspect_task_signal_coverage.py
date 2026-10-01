import unittest

from tools.inspect_task_signal_coverage import (
    reconstruct_route_points_from_task_edges,
)


class OrderedCoverageRouteTests(unittest.TestCase):
    def test_coverage_preflight_uses_ordered_edges_not_stale_generated_path(self):
        task = {
            "start_point": [-7797.6773, -700.0],
            "end_point": [-8193.4948, 700.0],
            "edges": [
                {
                    "node1": [-700.0, -700.0],
                    "node2": [-9300.0, -700.0],
                    "type": "sidewalk",
                },
                {
                    "node1": [-700.0, -700.0],
                    "node2": [-700.0, 700.0],
                    "type": "crosswalk",
                },
                {
                    "node1": [-9300.0, 700.0],
                    "node2": [-700.0, 700.0],
                    "type": "sidewalk",
                },
            ],
            "route_info": {
                "shortest_path": [
                    [-7797.6773, -700.0],
                    [-9300.0, -700.0],
                    [-9300.0, 700.0],
                    [-8193.4948, 700.0],
                ]
            },
        }

        self.assertEqual(
            [
                [-7797.6773, -700.0],
                [-700.0, -700.0],
                [-700.0, 700.0],
                [-8193.4948, 700.0],
            ],
            reconstruct_route_points_from_task_edges(task),
        )

    def test_coverage_preflight_fails_closed_on_disconnected_ordered_edges(self):
        task = {
            "start_point": [0.0, 0.0],
            "end_point": [20.0, 10.0],
            "edges": [
                {"node1": [0.0, 0.0], "node2": [10.0, 0.0]},
                {"node1": [20.0, 0.0], "node2": [20.0, 10.0]},
            ],
        }

        with self.assertRaisesRegex(ValueError, "do not share exactly one"):
            reconstruct_route_points_from_task_edges(task)


if __name__ == "__main__":
    unittest.main()
