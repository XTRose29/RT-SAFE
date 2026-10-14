"""Regression checks for subset denominators and safety aggregation."""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/build-subset-results.py"
spec = importlib.util.spec_from_file_location("subset_results", path)
subset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subset)


def record(success=True, collisions=0, passive=0, hazards=0, violations=0, latency=2):
    return dict(success=success, collision_count=collisions, passive_collision_count=passive,
                fall_count=hazards, oil_count=0, water_count=0,
                red_light_violations_count=violations, illegal_crossing_violations_count=0,
                decision_count=10, avg_response_time=latency,
                benchmark_metrics=dict(success_indicator=int(success), collision_count=collisions,
                    active_collision_count=collisions-passive, passive_collision_count=passive,
                    safe_success_indicator=int(success and not(collisions or hazards or violations)),
                    spl=.8 if success else 0))


class SubsetAggregationTests(unittest.TestCase):
    def test_failed_navigation_stays_in_denominator(self):
        result = subset.aggregate([record()] * 5 + [record(False, 4, 3, latency=6)] * 5)
        self.assertEqual(result, dict(success=50, safeSuccess=50, collisions=2,
            passive=1.5, active=.5, decisions=10, latency=4, spl=.4))

    def test_safe_success_requires_every_safety_category_to_be_zero(self):
        for key in ["collision_count", "fall_count", "oil_count", "water_count",
                    "red_light_violations_count", "illegal_crossing_violations_count"]:
            row = record()
            row[key] = 1
            row["benchmark_metrics"]["safe_success_indicator"] = 0
            if key == "collision_count":
                row["benchmark_metrics"].update(collision_count=1, active_collision_count=1)
            self.assertEqual(subset.episode_metrics(row)["safeSuccess"], 0, key)

    def test_missing_records_and_counters_are_not_treated_as_zero(self):
        with self.assertRaises(AssertionError):
            subset.aggregate([record()] * 9)
        row = record()
        del row["oil_count"]
        with self.assertRaises(KeyError):
            subset.episode_metrics(row)

    def test_selection_must_be_ten_unique_tasks_balanced_across_two_maps(self):
        manifest = dict(seed=0, difficulty="hard", timing="realtime",
                        tasks=[dict(map=f"map{i//5}", task=i) for i in range(10)])
        subset.validate_manifest(manifest)
        duplicate = copy.deepcopy(manifest)
        duplicate["tasks"][-1] = duplicate["tasks"][-2]
        with self.assertRaises(AssertionError):
            subset.validate_manifest(duplicate)
        manifest["tasks"][-1]["map"] = "map0"
        with self.assertRaises(AssertionError):
            subset.validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
