import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SimWorld"))

from base.rt_agent import RTAgent


def snapshot(state, *, consistent=True, on_crosswalk=False):
    return {
        "relevant": True,
        "crosswalk_id": 2,
        "distance_cm": 250.0,
        "pedestrian_state": state,
        "phase_remaining_time_s": 4.2,
        "consistent": consistent,
        "on_crosswalk": on_crosswalk,
        "signal_states": [],
    }


class RoutePermissionIntegrationTests(unittest.TestCase):
    def test_walk_allows_entry(self):
        result = RTAgent._annotate_route_crossing_permission(snapshot("WALK"))
        self.assertEqual("CROSS", result["route_crossing_permission"])

    def test_dont_walk_stops_new_entry(self):
        result = RTAgent._annotate_route_crossing_permission(snapshot("DONT_WALK"))
        self.assertEqual("STOP", result["route_crossing_permission"])

    def test_dont_walk_still_clears_an_agent_already_in_crosswalk(self):
        result = RTAgent._annotate_route_crossing_permission(
            snapshot("DONT_WALK", on_crosswalk=True)
        )
        self.assertEqual("CLEAR_ONLY", result["route_crossing_permission"])

    def test_flashing_only_clears_existing_crossing(self):
        outside = RTAgent._annotate_route_crossing_permission(
            snapshot("FLASHING_DONT_WALK", on_crosswalk=False)
        )
        inside = RTAgent._annotate_route_crossing_permission(
            snapshot("FLASHING_DONT_WALK", on_crosswalk=True)
        )
        self.assertEqual("STOP", outside["route_crossing_permission"])
        self.assertEqual("CLEAR_ONLY", inside["route_crossing_permission"])

    def test_conflict_fails_safe(self):
        result = RTAgent._annotate_route_crossing_permission(
            snapshot("CONFLICT", consistent=False)
        )
        self.assertEqual("STOP", result["route_crossing_permission"])

    def test_prompt_prioritizes_pedestrian_permission(self):
        text = RTAgent._format_traffic_light_context(snapshot("DONT_WALK"))
        self.assertTrue(
            text.startswith(
                "AUTHORITATIVE ROUTE CROSSING PERMISSION FOR THIS PEDESTRIAN"
            )
        )
        self.assertIn("Permission: STOP", text)
        self.assertIn("vehicle GREEN never authorizes", text)
        self.assertIn("may approach the near curb", text)


if __name__ == "__main__":
    unittest.main()
