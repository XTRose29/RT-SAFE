import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
unrealcv_dependencies = REPO_ROOT / ".py312deps"
if unrealcv_dependencies.is_dir():
    sys.path.append(str(unrealcv_dependencies))
simworld_package = types.ModuleType("simworld")
simworld_package.__path__ = [str(REPO_ROOT / "SimWorld" / "simworld")]
sys.modules.setdefault("simworld", simworld_package)

from evaluation.replay_recorded_rollout_video import (
    NoModelReplayLLM,
    action_text,
    candidate_waypoints,
    make_action,
    parsed_reasoning,
    phase_durations,
)


class RecordedRolloutReplayTests(unittest.TestCase):
    def manifest(self):
        return {
            "step": 3,
            "model_output": {
                "raw_response": (
                    "Action: move_to\nParam: 5\n"
                    "Reasoning: The path is clear."
                ),
                "parsed_action": {
                    "type": "move_to",
                    "param": "5",
                    "reasoning": "The path is clear.",
                },
            },
            "metrics": {"response_time": 4.5},
            "timing": {
                "model_inference_wall_seconds": 4.5,
                "simulation_thinking_latency_seconds": 7.25,
            },
            "waypoint_execution": {"move_command_duration_seconds": 2.0},
            "observation_geometry": {
                "annotated_candidates": [
                    {
                        "index": index,
                        "world_position_cm": {"x": index * 10, "y": index * 20},
                    }
                    for index in range(1, 8)
                ]
            },
        }

    def test_realtime_uses_simulated_thinking_duration(self):
        self.assertEqual((7.25, 2.0), phase_durations(self.manifest(), realtime=True))

    def test_static_shows_wall_inference_without_advancing_ue(self):
        self.assertEqual((4.5, 2.0), phase_durations(self.manifest(), realtime=False))

    def test_action_and_reasoning_are_loaded_from_manifest(self):
        manifest = self.manifest()
        action = make_action(manifest)
        self.assertEqual("move_to", action.action_type)
        self.assertEqual("5", action.action_param)
        self.assertEqual("MOVE_TO 5", action_text(manifest))
        self.assertEqual("The path is clear.", parsed_reasoning(manifest))

    def test_recorded_candidate_order_is_preserved(self):
        waypoints = candidate_waypoints(self.manifest())
        self.assertEqual(7, len(waypoints))
        self.assertEqual((10.0, 20.0), (waypoints[0].x, waypoints[0].y))
        self.assertEqual((70.0, 140.0), (waypoints[-1].x, waypoints[-1].y))

    def test_no_model_sentinel_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "attempted to access LLM"):
            NoModelReplayLLM().generate_response_openai


if __name__ == "__main__":
    unittest.main()
