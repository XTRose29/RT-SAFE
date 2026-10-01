import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SimWorld"))


# SimWorld uses lazy optional imports in this release; no global module stubs needed.
from base.rt_evaluator import RTEvaluator
from simworld.utils.vector import Vector


def dynamic_metadata(actor_id, kind, location):
    return {
        actor_id: {
            "id": actor_id,
            "kind": kind,
            "type": "test_actor",
            "initial_location_cm": list(location),
        }
    }


class FakeUnrealCV:
    def __init__(self, locations):
        self.locations = dict(locations)

    def get_objects(self):
        return list(self.locations)

    def get_location_batch(self, names):
        return [self.locations[name] for name in names]


class DynamicObstacleEvaluatorTests(unittest.TestCase):
    def make_evaluator(self, locations, metadata, static_obstacles=None):
        unrealcv = FakeUnrealCV(locations)
        communicator = SimpleNamespace(unrealcv=unrealcv)
        agent = SimpleNamespace(
            static_obstacles=list(static_obstacles or []),
            dynamic_obstacles=metadata,
            position=Vector(0.0, 0.0),
            shortest_path=[Vector(100.0, 0.0)],
            speed=200.0,
            sim_time_elapsed=0.0,
        )
        return RTEvaluator(communicator, agent), unrealcv

    def test_registered_movable_uses_live_positions_and_motion_telemetry(self):
        actor_id = "GEN_RT_RT_Box_1"
        evaluator, unrealcv = self.make_evaluator(
            {actor_id: (100.0, 0.0, 2.0)},
            dynamic_metadata(actor_id, "movable_obstacle", (100.0, 0.0, 2.0)),
        )

        first = evaluator._get_npc_positions()
        unrealcv.locations[actor_id] = (336.0, 0.0, 2.0)
        second = evaluator._get_npc_positions()
        telemetry = evaluator.dynamic_actor_telemetry()

        self.assertEqual(first[actor_id].x, 100.0)
        self.assertEqual(second[actor_id].x, 336.0)
        self.assertEqual(telemetry["registered_count"], 1)
        self.assertEqual(telemetry["sampled_live_count"], 1)
        self.assertEqual(telemetry["observed_moving_count"], 1)
        self.assertAlmostEqual(telemetry["max_displacement_cm"], 236.0)

    def test_stationary_actions_are_unsafe_when_actor_crosses_agent(self):
        actor_id = "GEN_RT_RT_Ball_1"
        evaluator, _ = self.make_evaluator(
            {actor_id: (200.0, 0.0, 2.0)},
            dynamic_metadata(actor_id, "movable_obstacle", (200.0, 0.0, 2.0)),
        )
        start = {actor_id: Vector(200.0, 0.0)}
        end = {actor_id: Vector(0.0, 0.0)}
        candidates = evaluator._evaluate_all_candidates(
            Vector(0.0, 0.0),
            start,
            end,
            100.0,
            [Vector(100.0, 0.0)],
        )

        self.assertEqual(candidates["L30"]["safety_score"], 0.0)
        self.assertEqual(candidates["WAIT_1"]["safety_score"], 0.0)

    def test_score_gap_separates_decision_regret_from_execution_delta(self):
        evaluator, _ = self.make_evaluator({}, {})
        evaluator.agent.step_num = 1

        initial_state = {
            "agent_position": Vector(0.0, 0.0),
            "npc_positions": {},
            "dynamic_actor_states": {},
            "path_distance": 1000.0,
            "step_num": 1,
            "candidate_waypoints": [Vector(100.0, 0.0), Vector(400.0, 0.0)],
        }
        candidate_scores = {
            "1": {"total_score": 1.0},
            "2": {"total_score": 1.5},
        }

        with patch.object(
            evaluator, "_get_npc_positions", return_value={}
        ), patch.object(
            evaluator, "_calculate_path_distance", return_value=800.0
        ), patch.object(
            evaluator, "_calculate_safety_score", return_value=1.0
        ), patch.object(
            evaluator, "_calculate_progress_score", return_value=1.0
        ), patch.object(
            evaluator, "_calculate_cost_score", return_value=0.0
        ), patch.object(
            evaluator, "_evaluate_all_candidates", return_value=candidate_scores
        ):
            scores = evaluator.end_step_evaluation(
                initial_state,
                action_type="MOVE",
                action_choice="1",
            )

        self.assertEqual(scores["total_score"], 2.0)
        self.assertEqual(scores["best_score"], 1.5)
        self.assertEqual(scores["chosen_candidate_score"], 1.0)
        self.assertEqual(scores["score_gap"], 0.5)
        self.assertEqual(scores["realization_score_delta"], 1.0)
        self.assertFalse(scores["is_optimal"])
        self.assertGreaterEqual(scores["score_gap"], 0.0)

    def test_falling_object_retains_vertical_clearance(self):
        actor_id = "GEN_RT_FO_Box_1"
        evaluator, _ = self.make_evaluator(
            {actor_id: (0.0, 0.0, 500.0)},
            dynamic_metadata(actor_id, "falling_object", (0.0, 0.0, 500.0)),
        )
        overhead = evaluator._dynamic_relative_distance(
            actor_id,
            Vector(0.0, 0.0),
            Vector(0.0, 0.0),
            Vector(0.0, 0.0),
            Vector(0.0, 0.0),
            actor_start_state=(0.0, 0.0, 500.0),
            actor_end_state=(0.0, 0.0, 500.0),
        )
        falling_through_agent = evaluator._dynamic_relative_distance(
            actor_id,
            Vector(0.0, 0.0),
            Vector(0.0, 0.0),
            Vector(0.0, 0.0),
            Vector(0.0, 0.0),
            actor_start_state=(0.0, 0.0, 500.0),
            actor_end_state=(0.0, 0.0, 110.0),
        )

        self.assertAlmostEqual(overhead, 390.0)
        self.assertAlmostEqual(falling_through_agent, 0.0)


if __name__ == "__main__":
    unittest.main()
