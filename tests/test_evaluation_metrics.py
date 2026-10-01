import unittest

from utils.evaluation_metrics import aggregate_metrics, episode_metrics, percentile


class EvaluationMetricsTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertAlmostEqual(3.7, percentile([1, 2, 3, 4], 90))

    def test_episode_metrics_match_paper_definitions(self):
        metrics = episode_metrics(
            success=True,
            shortest_path_length_cm=1000,
            traveled_path_length_cm=1250,
            sim_time_seconds=10,
            collision_count=2,
            passive_collision_count=1,
            fall_count=0,
            oil_count=0,
            water_count=0,
            red_light_violation_count=0,
            illegal_crossing_violation_count=0,
            decision_trace=[{
                "raw_model_response": "move",
                "response_time_seconds": 2.0,
                "modeled_thinking_latency_seconds": 1.5,
            }],
            conflict_event_groups=[[{
                "disposition": "launched",
                "collision_triggered": True,
            }]],
            total_tokens=120,
            completion_tokens=20,
            reasoning_tokens=5,
            parse_error_count=1,
            invalid_decision_count=2,
            collision_type_counts={"human": 1, "object": 1, "building": 0},
            passive_collision_type_counts={
                "human": 1, "object": 0, "building": 0,
            },
            vehicle_collision_count=0,
        )

        self.assertAlmostEqual(0.8, metrics["spl"])
        self.assertAlmostEqual(0.5, metrics["simulation_time_efficiency"])
        self.assertAlmostEqual(16.0, metrics["collisions_per_100m"])
        self.assertEqual(1, metrics["active_collision_count"])
        self.assertEqual(
            {"human": 1, "object": 1, "building": 0, "vehicle": 0},
            metrics["collision_type_counts"],
        )
        self.assertEqual(1, metrics["passive_collision_type_counts"]["human"])
        self.assertEqual(1, metrics["active_collision_type_counts"]["object"])
        self.assertEqual(0, metrics["safe_success_indicator"])
        self.assertEqual(1.0, metrics["conflict_vehicle"]["impact_given_valid_launch"])
        self.assertEqual(100, metrics["tokens"]["prompt"])

    def test_aggregate_metrics_is_episode_weighted_and_distance_normalized(self):
        first = {
            "benchmark_metrics": {
                "success_indicator": 1,
                "spl": 1.0,
                "simulation_time_efficiency": 0.8,
                "collision_episode_indicator": 0,
                "collision_count": 0,
                "collision_type_counts": {
                    "human": 0, "object": 0, "building": 0, "vehicle": 0,
                },
                "traveled_path_length_cm": 1000,
                "safe_success_indicator": 1,
                "conflict_vehicle": {"valid_launch_count": 1, "impact_count": 0},
                "latency": {"valid_response_count": 1, "cumulative_response_seconds": 1, "response_samples_seconds": [1]},
                "tokens": {},
                "errors": {},
            }
        }
        second = {
            "benchmark_metrics": {
                "success_indicator": 0,
                "spl": 0.0,
                "simulation_time_efficiency": 0.0,
                "collision_episode_indicator": 1,
                "collision_count": 2,
                "collision_type_counts": {
                    "human": 1, "object": 0, "building": 0, "vehicle": 1,
                },
                "active_collision_count": 1,
                "passive_collision_count": 1,
                "active_collision_type_counts": {
                    "human": 1, "object": 0, "building": 0, "vehicle": 0,
                },
                "passive_collision_type_counts": {
                    "human": 0, "object": 0, "building": 0, "vehicle": 1,
                },
                "traveled_path_length_cm": 1000,
                "safe_success_indicator": 0,
                "conflict_vehicle": {"valid_launch_count": 1, "impact_count": 1},
                "latency": {"valid_response_count": 1, "cumulative_response_seconds": 3, "response_samples_seconds": [3]},
                "tokens": {},
                "errors": {},
            }
        }

        summary = aggregate_metrics([first, second])

        self.assertAlmostEqual(0.5, summary["success_rate"])
        self.assertAlmostEqual(0.5, summary["spl"])
        self.assertAlmostEqual(10.0, summary["collisions_per_100m"])
        self.assertAlmostEqual(1.0, summary["mean_collision_count"])
        self.assertEqual(1, summary["collision_type_counts"]["human"])
        self.assertEqual(1, summary["collision_type_counts"]["vehicle"])
        self.assertAlmostEqual(
            0.5, summary["mean_collision_type_counts"]["human"]
        )
        self.assertEqual(1, summary["active_collision_count"])
        self.assertEqual(1, summary["passive_collision_count"])
        self.assertEqual(1, summary["active_collision_type_counts"]["human"])
        self.assertEqual(1, summary["passive_collision_type_counts"]["vehicle"])
        self.assertAlmostEqual(2.0, summary["latency"]["mean_response_seconds"])
        self.assertAlmostEqual(2.0, summary["latency"]["median_response_seconds"])


if __name__ == "__main__":
    unittest.main()
