import json
import tempfile
import time
import unittest
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
unrealcv_dependencies = REPO_ROOT / ".py312deps"
if unrealcv_dependencies.is_dir():
    # UnrealCV is pure Python; append this directory so the active interpreter's
    # native packages (notably NumPy) retain precedence.
    sys.path.append(str(unrealcv_dependencies))
simworld_package = types.ModuleType("simworld")
simworld_package.__path__ = [str(REPO_ROOT / "SimWorld" / "simworld")]
sys.modules.setdefault("simworld", simworld_package)

from base.rt_agent import RTAgent, validate_vlm_camera_frame
from llm.rt_llm import RTLLM
from simworld.utils.vector import Vector


class ConcurrentRealtimeInferenceTests(unittest.TestCase):
    def test_model_call_runs_between_resume_and_pause(self):
        events = []

        class FakeUnrealCV:
            def set_camera_resolution(self, *args):
                events.append("resolution")

            def set_camera_fov(self, *args):
                events.append("fov")

            def resume_simulation(self):
                events.append("resume")

            def pause_simulation(self):
                events.append("pause")

        agent = object.__new__(RTAgent)
        agent.step_num = 0
        agent.sim_time_elapsed = 0.0
        agent.concurrent_realtime_inference = True
        agent.realtime_thinking = True
        agent.token_based = False
        agent.fixed_action_seconds = None
        agent.camera_id = 1
        agent.fov = 100
        agent.communicator = SimpleNamespace(unrealcv=FakeUnrealCV())
        agent.disable_step_evaluation = True
        agent.record_per_step = False
        agent.record_dir = None
        agent._legacy_occupancy_crosswalks = set()
        agent.decision_trace = []
        agent.decision_count = 1
        agent.position = SimpleNamespace(x=1.0, y=2.0)
        agent.logger = SimpleNamespace(info=lambda *args: None)
        agent.sync_ue = lambda: events.append("sync")
        agent._remember_collision_free_state = lambda *args: None
        agent.get_observation = lambda: {
            "ego_view": object(),
            "waypoints": [],
        }
        agent._observation_geometry = lambda waypoints, **kwargs: {}
        callback_deltas = []
        agent.simulation_step_callback = callback_deltas.append

        def plan(*args):
            events.append("plan_start")
            time.sleep(0.02)
            events.append("plan_end")
            return None, 0.02, 0, 0, {"response_time": 0.02}

        agent.plan = plan

        agent.step("llm")

        self.assertLess(events.index("resume"), events.index("plan_start"))
        self.assertLess(events.index("plan_end"), events.index("pause"))
        self.assertEqual(1, agent.step_num)
        self.assertGreaterEqual(agent.sim_time_elapsed, 0.01)
        self.assertEqual(1, len(callback_deltas))
        self.assertAlmostEqual(
            agent.sim_time_elapsed,
            callback_deltas[0],
            places=6,
        )

        self.assertTrue(agent.decision_trace[0]["concurrent_realtime_inference"])
        self.assertGreaterEqual(agent.decision_trace[0]["modeled_thinking_latency_seconds"], 0.01)

    def test_concurrent_inference_restores_accepted_actor_pose_after_pause(self):
        events = []

        class FakeUnrealCV:
            def __init__(self):
                self.location = (100.0, 200.0, 92.0)
                self.orientation = (0.0, 90.0, 0.0)

            def set_camera_resolution(self, *args):
                events.append("resolution")

            def set_camera_fov(self, *args):
                events.append("fov")

            def get_location(self, *args):
                return self.location

            def get_orientation(self, *args):
                return self.orientation

            def resume_simulation(self):
                events.append("resume")
                # Reproduce a stale Blueprint timeline displacing the actor
                # and reversing yaw while Python is blocked in model inference.
                self.location = (160.0, 260.0, 92.0)
                self.orientation = (0.0, -89.0, 0.0)

            def pause_simulation(self):
                events.append("pause")

            def set_location(self, location, *args):
                events.append("restore_location")
                self.location = tuple(location)

            def set_orientation(self, orientation, *args):
                events.append("restore_orientation")
                self.orientation = tuple(orientation)

        unrealcv = FakeUnrealCV()
        agent = object.__new__(RTAgent)
        agent.step_num = 0
        agent.sim_time_elapsed = 0.0
        agent.concurrent_realtime_inference = True
        agent.realtime_thinking = True
        agent.token_based = False
        agent.fixed_action_seconds = None
        agent.camera_id = 1
        agent.fov = 100
        agent.name = "RT_AGENT"
        agent.task_edges = [object()]
        agent.first_person_actor_base_height_cm = 92.0
        agent.position = Vector(100.0, 200.0)
        agent.direction = 90.0
        agent.communicator = SimpleNamespace(unrealcv=unrealcv)
        agent.disable_step_evaluation = True
        agent.record_per_step = False
        agent.record_dir = None
        agent._legacy_occupancy_crosswalks = set()
        agent.decision_trace = []
        agent.decision_count = 1
        agent.logger = SimpleNamespace(info=lambda *args: None)
        agent.sync_ue = lambda: events.append("sync")
        agent._remember_collision_free_state = lambda *args: None
        agent.get_observation = lambda: {
            "ego_view": object(),
            "waypoints": [],
        }
        agent._observation_geometry = lambda waypoints, **kwargs: {}
        agent.simulation_step_callback = None
        agent.plan = lambda *args: (
            None,
            0.02,
            0,
            0,
            {"response_time": 0.02},
        )

        agent.step("llm")

        self.assertLess(events.index("pause"), events.index("restore_location"))
        self.assertLess(
            events.index("restore_location"),
            events.index("restore_orientation"),
        )
        self.assertEqual((100.0, 200.0, 92.0), unrealcv.location)
        self.assertEqual((0.0, 90.0, 0.0), unrealcv.orientation)
        self.assertAlmostEqual(100.0, agent.position.x)
        self.assertAlmostEqual(200.0, agent.position.y)
        self.assertAlmostEqual(90.0, agent.yaw)

    def test_token_based_mode_keeps_replay_path(self):
        agent = object.__new__(RTAgent)
        agent.concurrent_realtime_inference = True
        agent.realtime_thinking = True
        agent.token_based = True

        should_run_concurrently = (
            agent.concurrent_realtime_inference
            and agent.realtime_thinking
            and not agent.token_based
        )

        self.assertFalse(should_run_concurrently)


    def test_static_mode_does_not_model_inference_as_simulation_time(self):
        events = []

        class FakeUnrealCV:
            def set_camera_resolution(self, *args):
                events.append("resolution")

            def set_camera_fov(self, *args):
                events.append("fov")

        agent = object.__new__(RTAgent)
        agent.step_num = 0
        agent.sim_time_elapsed = 0.0
        agent.concurrent_realtime_inference = True
        agent.realtime_thinking = False
        agent.token_based = False
        agent.fixed_action_seconds = None
        agent.camera_id = 1
        agent.fov = 100
        agent.communicator = SimpleNamespace(unrealcv=FakeUnrealCV())
        agent.disable_step_evaluation = True
        agent.record_per_step = False
        agent.record_dir = None
        agent._legacy_occupancy_crosswalks = set()
        agent.logger = SimpleNamespace(info=lambda *args: None)
        agent.sync_ue = lambda: events.append("sync")
        agent._remember_collision_free_state = lambda *args: None
        agent.get_observation = lambda: {"ego_view": object(), "waypoints": []}
        agent._observation_geometry = lambda waypoints, **kwargs: {}
        agent.simulation_step_callback = None
        agent.decision_trace = []
        agent.decision_count = 1
        agent.position = SimpleNamespace(x=1.0, y=2.0)
        agent.plan = lambda *args: (
            None,
            12.5,
            0,
            0,
            {"response_time": 12.5},
        )

        agent.step("llm")

        self.assertEqual(0.0, agent.sim_time_elapsed)
        self.assertEqual(
            0.0,
            agent.decision_trace[0]["modeled_thinking_latency_seconds"],
        )


class LLMInputIntegrityTests(unittest.TestCase):
    def test_unknown_explicit_action_is_not_rewritten_as_move(self):
        llm = object.__new__(RTLLM)
        action = llm._parse_structured_response("Action: fly\nParam: 1")
        self.assertFalse(action.is_valid())
        self.assertIsNone(action.action_type)

    def test_reasoning_waypoint_mention_is_not_executed_without_action_fields(self):
        llm = object.__new__(RTLLM)
        action = llm._parse_structured_response(
            "Reasoning: waypoint 6 is blocked by pedestrians; I must wait."
        )
        self.assertFalse(action.is_valid())
        self.assertIsNone(action.action_type)

    def test_truncated_move_reasoning_fails_closed(self):
        llm = object.__new__(RTLLM)
        action = llm._parse_structured_response(
            "Reasoning: Moving to waypoint 5 could minimize exposure, but"
        )
        self.assertFalse(action.is_valid())
        self.assertIsNone(action.action_type)

    def test_pil_observation_declares_png_mime(self):
        llm = object.__new__(RTLLM)
        data_url = llm._image_data_url(Image.new("RGB", (2, 2), "red"))
        self.assertTrue(data_url.startswith("data:image/png;base64,iVBOR"))

    def test_expected_camera_frame_is_accepted(self):
        frame = np.zeros((640, 720, 3), dtype=np.uint8)
        frame[0, 0] = (10, 20, 30)
        self.assertIs(validate_vlm_camera_frame(frame), frame)

    def test_configured_larger_camera_frame_is_accepted(self):
        frame = np.full((960, 1080, 3), 127, dtype=np.uint8)
        self.assertIs(
            validate_vlm_camera_frame(
                frame,
                expected_width=1080,
                expected_height=960,
            ),
            frame,
        )

    def test_black_camera_frame_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "blank/black"):
            validate_vlm_camera_frame(
                np.zeros((640, 720, 3), dtype=np.uint8)
            )

    def test_wrong_camera_shape_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "shape"):
            validate_vlm_camera_frame(
                np.zeros((480, 640, 3), dtype=np.uint8)
            )

    def test_chat_api_timeout_is_applied_and_failure_can_propagate(self):
        captured = {}

        def fail_request(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("model endpoint unavailable")

        llm = object.__new__(RTLLM)
        llm.api_mode = "chat_completions"
        llm.max_tokens = 8
        llm.model_name = "qwen-test"
        llm.provider = "self-hosted"
        llm.reasoning = False
        llm.reasoning_effort = None
        llm.extra_body = {}
        llm.request_timeout = 7.5
        llm.raise_on_api_error = True
        llm.logger = SimpleNamespace(error=lambda *args: None)
        llm.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=fail_request)
            )
        )

        with self.assertRaisesRegex(RuntimeError, "endpoint unavailable"):
            llm.generate_response_openai("system", "user", images=[])

        self.assertEqual(7.5, captured["timeout"])

class SparseImageRetentionTests(unittest.TestCase):
    def test_first_and_rolling_last_images_are_kept_with_all_manifests(self):
        agent = object.__new__(RTAgent)
        agent.record_images_first_steps = 2
        agent.record_images_last_steps = 2
        with tempfile.TemporaryDirectory() as temporary:
            agent.record_dir = temporary
            root = Path(temporary)
            for step in range(7):
                stem = f"step_{step:04d}"
                directory = root / stem
                directory.mkdir()
                image_name = f"{stem}_input_frame_00.png"
                Image.new("RGB", (8, 6), "white").save(
                    directory / image_name
                )
                Image.new("RGB", (8, 6), "black").save(
                    directory / f"{stem}_traffic_light_debug.png"
                )
                (directory / f"{stem}_manifest.json").write_text(
                    json.dumps({
                        "input_images": [image_name],
                        "model_input_image_count": 1,
                        "classifier_input_images": [],
                        "recording_action_frames": [],
                        "demo_action_images": [],
                        "output_image": None,
                        "demo_input_image": None,
                        "demo_output_image": None,
                        "feedback": f"feedback-{step}",
                        "image_retention": {
                            "policy": "first_last_ring_v1",
                            "retained": True,
                            "first_steps": 2,
                            "last_steps": 2,
                        },
                    }),
                    encoding="utf-8",
                )
                agent._prune_middle_step_images(step)

            for step in range(7):
                stem = f"step_{step:04d}"
                directory = root / stem
                manifest = json.loads(
                    (directory / f"{stem}_manifest.json").read_text()
                )
                should_retain = step < 2 or step >= 5
                self.assertEqual(
                    should_retain,
                    manifest["image_retention"]["retained"],
                )
                self.assertEqual(
                    should_retain,
                    bool(list(directory.glob("*.png"))),
                )
                self.assertEqual(f"feedback-{step}", manifest["feedback"])


if __name__ == "__main__":
    unittest.main()
