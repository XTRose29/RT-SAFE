"""Protect the new comparison from incomplete or duplicated task sets."""
import copy
import importlib.util
from pathlib import Path
import unittest
from test_subset_results import record

path = Path(__file__).resolve().parents[1] / "scripts/build-matched-default-results.py"
spec = importlib.util.spec_from_file_location("matched", path)
matched = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matched)


def episode(task, success=True):
    row = record(success=success, collisions=0 if success else 4, passive=0 if success else 3)
    row.update(task_id=task, move_count=8, move_1m_count=0, move_2m_count=8,
               move_4m_count=0, turn_count=1, wait_count=1,
               off_route_move_no_progress_count=0, parse_error_count=0,
               termination_reason="success" if success else "max_steps")
    row["benchmark_metrics"]["latency"] = dict(cumulative_response_seconds=20)
    return row


class MatchedDefaultTests(unittest.TestCase):
    def test_incomplete_and_duplicate_tasks_cannot_be_ranked(self):
        rows = [episode(i) for i in range(20)]
        with self.assertRaises(AssertionError):
            matched.aggregate(rows[:-1])
        with self.assertRaises(AssertionError):
            matched.aggregate(rows[:-1] + [rows[0]])

    def test_navigation_failures_count_in_all_twenty_episode_means(self):
        result = matched.aggregate([episode(i, i < 15) for i in range(20)])
        self.assertEqual(result["success"], 75)
        self.assertEqual(result["collisions"], 1)
        self.assertEqual(result["passive"], .75)
        self.assertEqual(result["behavior"]["terminationCounts"]["max_steps"], 5)

    def test_selection_has_five_distinct_tasks_on_each_of_four_maps(self):
        manifest = dict(seed=0, difficulty="hard", timing="realtime",
                        tasks=[dict(map=f"map{i//5}", task=i) for i in range(20)])
        matched.validate_selection(manifest)
        changed = copy.deepcopy(manifest)
        changed["tasks"][-1] = changed["tasks"][0]
        with self.assertRaises(AssertionError):
            matched.validate_selection(changed)
        changed = copy.deepcopy(manifest)
        changed["tasks"][-1]["map"] = "map0"
        with self.assertRaises(AssertionError):
            matched.validate_selection(changed)


if __name__ == "__main__":
    unittest.main()
