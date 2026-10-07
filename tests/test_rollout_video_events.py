"""Tests for self-contained safety labels in annotated rollout videos."""

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.render_rollout_input_video import (  # noqa: E402
    classify_events,
    current_policy_image_name,
    is_turn_action,
    policy_input_grid_shape,
    prepare_frame_directory,
    simulation_frame_durations,
    structured_event_times,
    wrap_text,
)
from evaluation.finalize_qwen_rollout_videos import (  # noqa: E402
    safety_event_category,
)


class RolloutVideoEventTests(unittest.TestCase):
    def test_terminal_inference_vehicle_impact_is_not_double_labeled(self):
        manifest = {
            "feedback": "Vehicle collision occurred during model inference.",
            "collision_details": {"vehicle": 1},
            "passive_collision_details": {"vehicle": 1},
        }

        self.assertEqual(
            ["passive vehicle collision x1"],
            classify_events(manifest),
        )

    def test_suite_index_uses_specific_vehicle_and_traffic_categories(self):
        self.assertEqual(
            "vehicle_collision",
            safety_event_category("passive vehicle collision x1"),
        )
        self.assertEqual(
            "illegal_crossing",
            safety_event_category("illegal crossing + vehicle launch 3.0m"),
        )
        self.assertEqual(
            "red_light_violation",
            safety_event_category("red light + vehicle launch 4.0m"),
        )

    def test_current_policy_image_is_last_after_action_frame_history(self):
        names = [f"history_{index}.png" for index in range(3)] + ["current.png"]

        self.assertEqual("current.png", current_policy_image_name(names, 4))

    def test_policy_image_sequence_has_no_benchmark_count_cap(self):
        names = [f"frame_{index}.png" for index in range(12)]
        self.assertEqual("frame_11.png", current_policy_image_name(names, 4))

    def test_policy_image_sequence_rejects_an_empty_bundle(self):
        with self.assertRaisesRegex(RuntimeError, "at least one policy image"):
            current_policy_image_name([], 4)

    def test_policy_input_grid_preserves_every_model_image(self):
        self.assertEqual((1, 1), policy_input_grid_shape(1))
        self.assertEqual((2, 1), policy_input_grid_shape(2))
        self.assertEqual((2, 2), policy_input_grid_shape(4))
        self.assertEqual((4, 2), policy_input_grid_shape(8))
        self.assertEqual((4, 3), policy_input_grid_shape(12))

    def test_simulation_frame_durations_cover_exact_timeline(self):
        manifests = [
            {
                "timing": {
                    "sim_time_start_seconds": 0.0,
                    "sim_time_end_seconds": 3.5,
                }
            },
            {
                "timing": {
                    "sim_time_start_seconds": 3.5,
                    "sim_time_end_seconds": 8.25,
                }
            },
            {
                "timing": {
                    "sim_time_start_seconds": 8.25,
                    "sim_time_end_seconds": 10.0,
                }
            },
        ]

        durations = simulation_frame_durations(manifests)

        self.assertEqual([3.5, 4.75, 1.75], durations)
        self.assertEqual(10.0, sum(durations))

    def test_simulation_frame_durations_reject_non_monotonic_time(self):
        manifests = [
            {
                "timing": {
                    "sim_time_start_seconds": 2.0,
                    "sim_time_end_seconds": 3.0,
                }
            },
            {
                "timing": {
                    "sim_time_start_seconds": 1.0,
                    "sim_time_end_seconds": 4.0,
                }
            },
        ]

        with self.assertRaisesRegex(RuntimeError, "Non-positive"):
            simulation_frame_durations(manifests)

    def test_structured_red_light_launch_is_labeled_with_event_time(self):
        manifest = {
            "feedback": "Action completed successfully.",
            "safety_events": {
                "red_light_violations": [
                    {
                        "sim_time_s": 42.75,
                        "pedestrian_state": "FLASHING_DONT_WALK",
                        "conflict_vehicle": {
                            "status": "launched",
                            "launch_distance_cm": 375.0,
                        },
                    }
                ]
            },
        }

        self.assertEqual(
            [
                "red light + vehicle launch 3.8m "
                "(FLASHING_DONT_WALK)"
            ],
            classify_events(manifest),
        )
        self.assertEqual([42.75], structured_event_times(manifest))

    def test_structured_illegal_crossing_is_visible_without_feedback_text(self):
        manifest = {
            "safety_events": {
                "illegal_crossing_violations": [{"sim_time_s": 7.0}]
            }
        }

        self.assertEqual(["illegal crossing"], classify_events(manifest))
        self.assertEqual([7.0], structured_event_times(manifest))

    def test_turn_classification_ignores_stale_execution_metadata(self):
        manifest = {
            "model_output": {"parsed_action": {"type": "wait"}},
            "turn_execution": {"requested_param": "L90"},
        }

        self.assertFalse(is_turn_action(manifest))

    def test_turn_classification_accepts_actual_parsed_turn(self):
        manifest = {
            "model_output": {
                "parsed_action": {"type": "turn_around"}
            },
            "turn_execution": {"requested_param": "L90"},
        }

        self.assertTrue(is_turn_action(manifest))

    def test_prepare_frame_directory_removes_only_stale_render_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_dir = Path(temp_dir) / "annotated_input_frames"
            frame_dir.mkdir()
            stale = frame_dir / "frame_0030.png"
            unrelated = frame_dir / "notes.txt"
            stale.write_bytes(b"old frame")
            unrelated.write_text("keep", encoding="utf-8")

            prepare_frame_directory(frame_dir)

            self.assertFalse(stale.exists())
            self.assertEqual("keep", unrelated.read_text(encoding="utf-8"))

    def test_combined_safety_labels_wrap_without_ellipsis(self):
        image = Image.new("RGB", (720, 640))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        events = ["PASSIVE HUMAN COLLISION X1", "WATER/SLIP"]

        lines = []
        for event in events:
            lines.extend(wrap_text(draw, event, font, 692))

        self.assertEqual(events, lines)
        self.assertNotIn("...", " ".join(lines))


if __name__ == "__main__":
    unittest.main()
