import unittest

from evaluation.aggregate_results import extract_metrics


class AggregateResultsTests(unittest.TestCase):
    def test_extracts_paper_metrics_from_episode_result(self):
        row = extract_metrics({
            "model": "example",
            "difficulty": "easy",
            "success": True,
            "benchmark_metrics": {
                "spl": 0.75,
                "simulation_time_efficiency": 0.6,
                "collision_episode_indicator": 1,
                "collision_count": 2,
                "traveled_path_length_cm": 1200,
                "safe_success_indicator": 0,
            },
        })

        self.assertAlmostEqual(0.75, row["spl"])
        self.assertAlmostEqual(0.6, row["success_weighted_efficiency"])
        self.assertEqual(1, row["collision_episode_indicator"])
        self.assertEqual(0, row["safe_success_indicator"])

    def test_safe_success_fallback_checks_all_safety_events(self):
        row = extract_metrics({
            "model": "example",
            "difficulty": "default",
            "success": True,
            "collision_count": 0,
            "fall_count": 0,
            "oil_count": 0,
            "water_count": 0,
            "red_light_violations_count": 1,
            "illegal_crossing_violations_count": 0,
        })

        self.assertEqual("hard", row["difficulty"])
        self.assertEqual(0, row["safe_success_indicator"])


if __name__ == "__main__":
    unittest.main()
