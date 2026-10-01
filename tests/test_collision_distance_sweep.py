"""Tests for the live collision-distance sweep's deterministic bookkeeping."""

import argparse
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from run_collision_distance_sweep import (
    _cached_trial_matches,
    _range,
    _seeds,
    _summary,
)


class CollisionDistanceSweepTests(unittest.TestCase):
    def test_range_parser_accepts_ordered_nonnegative_centimetres(self):
        self.assertEqual((400.0, 800.0), _range("400:800"))
        with self.assertRaises(argparse.ArgumentTypeError):
            _range("800:400")

    def test_seed_parser_requires_at_least_one_integer(self):
        self.assertEqual([1, 3, 8], _seeds("1,3,8"))
        with self.assertRaises(argparse.ArgumentTypeError):
            _seeds("")

    def test_summary_reports_probability_conditioned_on_valid_launches(self):
        trials = [
            {
                "range_label": "range_400_800",
                "launch_distance_range_cm": [400.0, 800.0],
                "passed": True,
                "outcome": "collision",
            },
            {
                "range_label": "range_400_800",
                "launch_distance_range_cm": [400.0, 800.0],
                "passed": True,
                "outcome": "miss",
            },
            {
                "range_label": "range_400_800",
                "launch_distance_range_cm": [400.0, 800.0],
                "passed": False,
                "outcome": None,
            },
        ]

        result = _summary(Path("results"), trials)["ranges"][0]

        self.assertEqual(3, result["attempted"])
        self.assertEqual(2, result["valid_launched_trials"])
        self.assertEqual(1, result["collisions"])
        self.assertEqual(1, result["misses"])
        self.assertEqual(0.5, result["collision_probability_given_launch"])
        self.assertIsNotNone(result["collision_probability_wilson_95"])

    def test_resume_cache_requires_matching_experiment_contract(self):
        payload = {
            "passed": True,
            "task_file": "data/map1_10roads/tasks.json",
            "task_number": 2,
            "mode": "red_light",
            "seed": 7,
            "experiment": {
                "pressure_fast_forward": True,
                "post_launch_agent_motion": "continue",
                "launch_distance_range_cm": [400.0, 800.0],
                "walking_speed_cm_s": 200.0,
            },
            "full_task_recording": {"simulation_step_seconds": 1.0},
            "camera": {"resolution": [320, 240], "fov_deg": 70.0},
        }
        arguments = {
            "task_file": "data/map1_10roads/tasks.json",
            "task_number": 2,
            "mode": "red_light",
            "seed": 7,
            "distance_range": (400.0, 800.0),
            "walking_speed_cm_s": 200.0,
            "simulation_step_seconds": 1.0,
            "camera_resolution": (320, 240),
        }

        self.assertTrue(_cached_trial_matches(payload, **arguments))
        self.assertFalse(
            _cached_trial_matches(payload, **{**arguments, "seed": 8})
        )


if __name__ == "__main__":
    unittest.main()
