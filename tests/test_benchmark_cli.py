import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_cli", ROOT / "benchmark" / "run.py"
)
benchmark_cli = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark_cli)


class BenchmarkConfigTests(unittest.TestCase):
    def test_doctor_finds_module_in_supported_local_dependency_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "unrealcv"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            with patch.object(
                benchmark_cli.importlib.util,
                "find_spec",
                return_value=None,
            ):
                self.assertEqual(
                    str(root),
                    benchmark_cli.python_module_location(
                        "unrealcv", [root]
                    ),
                )

    def test_doctor_reports_missing_required_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(
                benchmark_cli.importlib.util,
                "find_spec",
                return_value=None,
            ):
                self.assertIsNone(
                    benchmark_cli.python_module_location(
                        "unrealcv", [Path(temporary)]
                    )
                )

    def test_qwen_doctor_auto_prefers_a_live_external_endpoint(self):
        with patch.object(benchmark_cli, "tcp_open", return_value=True):
            self.assertEqual(
                ("external Qwen endpoint", True, "127.0.0.1:30001"),
                benchmark_cli.qwen_doctor_check({}),
            )

    def test_qwen_doctor_external_fails_closed_when_endpoint_is_down(self):
        with patch.object(benchmark_cli, "tcp_open", return_value=False):
            self.assertEqual(
                ("external Qwen endpoint", False, "qwen.example:30123"),
                benchmark_cli.qwen_doctor_check(
                    {
                        "qwen_mode": "external",
                        "qwen_host": "qwen.example",
                        "qwen_port": 30123,
                    }
                ),
            )

    def test_all_versioned_configs_are_valid(self):
        paths = sorted((ROOT / "benchmark" / "configs").glob("*.json"))
        self.assertGreaterEqual(len(paths), 4)
        for path in paths:
            with self.subTest(path=path):
                resolved, config = benchmark_cli.load_config(path)
                self.assertEqual(resolved, path.resolve())
                self.assertTrue(config["rollout"]["record_per_step"])
                self.assertTrue(config["rollout"]["uniform_waypoint_movement"])
                self.assertTrue(config["rollout"]["use_action_frames"])

    def test_run_and_doctor_default_to_finalized_collision_preset(self):
        self.assertEqual(
            benchmark_cli.DEFAULT_CONFIG,
            benchmark_cli.parse_args(["run"]).config,
        )
        self.assertEqual(
            benchmark_cli.DEFAULT_CONFIG,
            benchmark_cli.parse_args(["doctor"]).config,
        )

    def test_default_preset_explicitly_records_reference_condition(self):
        _, config = benchmark_cli.load_config(benchmark_cli.DEFAULT_CONFIG)
        rollout = config["rollout"]

        expected = {
            "maps": [
                "map1_10roads",
                "map2_12roads",
                "map3_15roads",
                "map4_18roads",
                "map5_20roads",
            ],
            "difficulties": ["easy"],
            "env_modes": ["realtime"],
            "rounds": 1,
            "seed": 0,
            "model": "qwen3-vl-8b",
            "enable_thinking": False,
            "qwen_max_tokens": 128,
            "token_based": False,
            "observation_width": 720,
            "observation_height": 640,
            "observation_fov_deg": 100.0,
            "observation_camera_pitch_deg": -25.0,
            "concurrent_realtime_inference": True,
            "traffic_policy": "visual_only",
            "pedestrians": True,
            "movable_obstacles": True,
            "irregular_npcs": True,
            "falling_objects": True,
            "load_all_unsafe_triggers": False,
            "red_light_conflict_vehicle": True,
            "conflict_vehicle_launch_probability": 1.0,
            "static_signal_vehicles": True,
            "conflict_vehicle_min_launch_distance_m": 3.0,
            "conflict_vehicle_max_launch_distance_m": 9.0,
            "conflict_vehicle_impact_radius_m": 1.0,
            "conflict_vehicle_release_wait_timeout_s": 12.0,
        }
        self.assertEqual(
            expected,
            {key: rollout[key] for key in expected},
        )

        arguments = benchmark_cli.rollout_arguments(config)
        for flag in (
            "--observation-width",
            "--observation-height",
            "--observation-fov-deg",
            "--observation-camera-pitch-deg",
            "--concurrent-realtime-inference",
        ):
            self.assertIn(flag, arguments)

    def test_easy_realtime_config_targets_all_maps_and_one_mode(self):
        _, config = benchmark_cli.load_config("all_tasks_easy_realtime")
        arguments = benchmark_cli.rollout_arguments(config)
        self.assertIn("--maps", arguments)
        self.assertEqual(config["rollout"]["difficulties"], ["easy"])
        self.assertEqual(config["rollout"]["env_modes"], ["realtime"])
        self.assertNotIn("--task-indices", arguments)

    def test_smoke_preserves_default_artifact_and_vehicle_staging(self):
        _, smoke = benchmark_cli.load_config("smoke")
        _, default = benchmark_cli.load_config(benchmark_cli.DEFAULT_CONFIG)
        for key in ("record_png_compress_level", "static_signal_vehicles"):
            with self.subTest(key=key):
                self.assertEqual(
                    default["rollout"][key],
                    smoke["rollout"][key],
                )

    def test_collision_preset_preserves_easy_realtime_semantics(self):
        _, baseline = benchmark_cli.load_config("all_tasks_easy_realtime")
        _, collision = benchmark_cli.load_config(
            "all_tasks_easy_realtime_collision"
        )
        reliability_and_collision_keys = {
            "cell_max_attempts",
            "conflict_vehicle_launch_probability",
            "conflict_vehicle_impact_radius_m",
            "conflict_vehicle_max_launch_distance_m",
            "conflict_vehicle_min_launch_distance_m",
            "conflict_vehicle_release_wait_timeout_s",
            "continue_on_error",
            "red_light_conflict_vehicle",
            "retry_errors",
            "static_signal_vehicles",
            "ue_max_attempts",
            "ue_no_rhi_thread",
        }
        baseline_rollout = baseline["rollout"]
        collision_rollout = collision["rollout"]
        semantic_keys = set(baseline_rollout) - reliability_and_collision_keys
        self.assertEqual(
            {key: baseline_rollout[key] for key in semantic_keys},
            {key: collision_rollout[key] for key in semantic_keys},
        )
        self.assertEqual(
            3.0, collision_rollout["conflict_vehicle_min_launch_distance_m"]
        )
        self.assertEqual(
            9.0, collision_rollout["conflict_vehicle_max_launch_distance_m"]
        )
        self.assertEqual(
            1.0, collision_rollout["conflict_vehicle_impact_radius_m"]
        )
        self.assertEqual(
            12.0, collision_rollout["conflict_vehicle_release_wait_timeout_s"]
        )
        self.assertTrue(collision_rollout["static_signal_vehicles"])

    def test_video_rejects_missing_step_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "rollout": {
                            "rollout_profile": "optimized",
                            "record_per_step": False,
                            "uniform_waypoint_movement": True,
                        },
                        "video": {"enabled": True},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "record_per_step"):
                benchmark_cli.load_config(path)

    def test_video_rejects_nonuniform_waypoint_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "rollout": {
                            "rollout_profile": "full",
                            "uniform_waypoint_movement": False,
                        },
                        "video": {"enabled": True},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "uniform_waypoint_movement"):
                benchmark_cli.load_config(path)

    def test_boolean_false_uses_no_prefix(self):
        config = {
            "rollout": {
                "red_light_conflict_vehicle": False,
                "pedestrians": True,
                "use_action_frames": False,
            }
        }
        self.assertEqual(
            benchmark_cli.rollout_arguments(config),
            [
                "--no-red-light-conflict-vehicle",
                "--pedestrians",
                "--no-use-action-frames",
            ],
        )

    def test_status_excludes_archived_failed_attempt_videos(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary) / "suite"
            active = suite / "runs" / "task_001" / "round_001"
            archived = active / "failed_attempts" / "old"
            archived.mkdir(parents=True)
            (suite / "experiment_manifest.json").write_text(
                json.dumps({"rollout_count": 1}),
                encoding="utf-8",
            )
            result = active / "task_1.json"
            result.write_text("{}", encoding="utf-8")
            (active / "run_status.json").write_text(
                json.dumps(
                    {"status": "completed", "result_path": str(result)}
                ),
                encoding="utf-8",
            )
            (active / "rollout_agent_input_annotated.mp4").write_bytes(
                b"current"
            )
            (active / "rollout_alignment_check.json").write_text(
                json.dumps({"passed": True}),
                encoding="utf-8",
            )
            (archived / "rollout_agent_input_annotated.mp4").write_bytes(
                b"old"
            )
            (archived / "rollout_alignment_check.json").write_text(
                json.dumps({"passed": False}),
                encoding="utf-8",
            )

            payload = benchmark_cli.status_payload(suite)
            self.assertEqual(1, payload["videos"])
            self.assertEqual({"completed": 1}, payload["status_counts"])
            self.assertEqual(1, payload["alignment_passed"])
            self.assertEqual(0, payload["alignment_failed"])

            result.unlink()
            invalidated = benchmark_cli.status_payload(suite)
            self.assertEqual(
                {"invalidated": 1},
                invalidated["status_counts"],
            )


if __name__ == "__main__":
    unittest.main()
