import importlib.util
import base64
import inspect
import json
import os
import socket
import struct
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_qwen3vl8b_all_maps.py")
SPEC = importlib.util.spec_from_file_location("all_maps_runner", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def write_tasks(path: Path, task_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"tasks": [{"task_id": task_id} for task_id in task_ids]}),
        encoding="utf-8",
    )


class RolloutPlanTests(unittest.TestCase):
    def test_easy_realtime_then_easy_static_then_remaining_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_tasks(root / "a/tasks.json", [10, 11])
            write_tasks(root / "b/tasks.json", [20])
            maps = [
                runner.MapSpec("a", "RTA", "a/tasks.json"),
                runner.MapSpec("b", "RTB", "b/tasks.json"),
            ]
            plan = runner.build_rollout_plan(
                maps,
                ("easy", "medium", "default"),
                rounds=2,
                repo_root=root,
            )

        self.assertEqual(len(plan), 36)
        self.assertEqual(
            {(item.difficulty, item.env_mode) for item in plan[:6]},
            {("easy", "realtime")},
        )
        self.assertEqual(
            {(item.difficulty, item.env_mode) for item in plan[6:12]},
            {("easy", "static")},
        )
        self.assertEqual(
            [(item.difficulty, item.env_mode) for item in plan[12::6]],
            [
                ("medium", "realtime"),
                ("medium", "static"),
                ("default", "realtime"),
                ("default", "static"),
            ],
        )
        self.assertEqual(len({item.rollout_id for item in plan}), len(plan))
        self.assertEqual([item.ordinal for item in plan], list(range(1, 37)))

    def test_level_profile_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_tasks(root / "a/tasks.json", [1])
            plan = runner.build_rollout_plan(
                [runner.MapSpec("a", "RTA", "a/tasks.json")],
                runner.DIFFICULTY_PROFILES["levels"],
                rounds=1,
                repo_root=root,
            )
        self.assertEqual(len(plan), 10)
        self.assertEqual(plan[0].difficulty, "level0")
        self.assertEqual(plan[0].env_mode, "realtime")
        self.assertEqual(plan[1].env_mode, "static")

    def test_selected_task_index_keeps_original_index_and_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_tasks(root / "a/tasks.json", [10, 11, 12])
            plan = runner.build_rollout_plan(
                [runner.MapSpec("a", "RTA", "a/tasks.json")],
                ("easy",),
                rounds=1,
                repo_root=root,
                task_indices=(2,),
            )

        self.assertEqual([item.task_index for item in plan], [2, 2])
        self.assertEqual([item.task_id for item in plan], [12, 12])


class RolloutProfileTests(unittest.TestCase):
    def resolve(self, *argv: str):
        return runner.resolve_rollout_profile(runner.parse_args(list(argv)))

    def test_full_profile_is_the_default(self):
        args = self.resolve()
        self.assertEqual(args.rollout_profile, "full")
        self.assertTrue(args.record_per_step)
        self.assertFalse(args.fast_simulation)
        self.assertEqual(args.traffic_policy, "visual_only")
        self.assertTrue(args.red_light_conflict_vehicle)
        self.assertTrue(args.static_signal_vehicles)
        self.assertEqual(args.conflict_vehicle_launch_probability, 1.0)
        self.assertTrue(args.pedestrians)
        self.assertTrue(args.movable_obstacles)
        self.assertTrue(args.irregular_npcs)
        self.assertTrue(args.falling_objects)
        self.assertTrue(args.concurrent_realtime_inference)
        self.assertTrue(args.use_action_frames)
        self.assertEqual(args.prompt_style, "instructional")
        self.assertEqual(args.qwen_max_tokens, 128)
        self.assertTrue(args.qwen_warmup)

        self.assertEqual(args.ue_gpu, "0")
        self.assertEqual(args.observation_width, 720)
        self.assertEqual(args.observation_height, 640)
        self.assertEqual(args.observation_fov_deg, 100.0)
        self.assertEqual(args.observation_camera_pitch_deg, -25.0)
        self.assertEqual(args.unrealcv_request_timeout, 120)

    def test_dynamic_signal_vehicle_physics_requires_explicit_opt_out(self):
        args = self.resolve("--no-static-signal-vehicles")

        self.assertFalse(args.static_signal_vehicles)

    def test_cross_user_vllm_listener_falls_back_to_proc_cmdline(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc_root = Path(temporary)
            listener = proc_root / "1234"
            listener.mkdir()
            (listener / "cmdline").write_bytes(
                b"/env/bin/python\x00-m\x00"
                b"vllm.entrypoints.openai.api_server\x00"
                b"--host\x00127.0.0.1\x00--port\x0030001\x00"
            )
            unrelated = proc_root / "5678"
            unrelated.mkdir()
            (unrelated / "cmdline").write_bytes(
                b"python\x00-m\x00http.server\x00--port\x0030001\x00"
            )

            self.assertEqual(
                1234,
                runner._proc_qwen_listener_pid(30001, proc_root=proc_root),
            )

    def test_proc_listener_probe_requires_an_unambiguous_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc_root = Path(temporary)
            for pid in ("1234", "5678"):
                process = proc_root / pid
                process.mkdir()
                (process / "cmdline").write_bytes(
                    b"python\x00-m\x00"
                    b"vllm.entrypoints.openai.api_server\x00"
                    b"--port=30001\x00"
                )

            self.assertIsNone(
                runner._proc_qwen_listener_pid(30001, proc_root=proc_root)
            )

    def test_safety_assisted_traffic_policy_remains_selectable(self):
        args = self.resolve("--traffic-policy", "safety_assisted")
        self.assertEqual(args.traffic_policy, "safety_assisted")

    def test_action_frame_series_can_be_disabled(self):
        args = self.resolve("--no-use-action-frames")
        self.assertFalse(args.use_action_frames)

    def test_red_light_conflict_vehicle_can_be_disabled(self):
        args = self.resolve("--no-red-light-conflict-vehicle")
        self.assertFalse(args.red_light_conflict_vehicle)

    def test_conflict_vehicle_probability_override_is_explicit(self):
        args = self.resolve(
            "--conflict-vehicle-launch-probability", "0.25"
        )
        self.assertEqual(args.conflict_vehicle_launch_probability, 0.25)

    def test_collision_mode_stages_route_signal_traffic(self):
        args = self.resolve("--red-light-conflict-vehicle")
        with mock.patch.dict(os.environ, {}, clear=False):
            runner.configure_benchmark_environment(args)
            self.assertEqual(
                "1", os.environ["SIMWORLD_STAGE_ROUTE_SIGNAL_TRAFFIC"]
            )
            self.assertEqual(
                "100.0", os.environ["SIMWORLD_AGENT_CAMERA_FOV_DEG"]
            )
            self.assertEqual(
                "-25.0",
                os.environ["SIMWORLD_FIRST_PERSON_CAMERA_PITCH_DEG"],
            )

    def test_disabling_collision_mode_clears_route_staging(self):
        args = self.resolve("--no-red-light-conflict-vehicle")
        with mock.patch.dict(
            os.environ,
            {"SIMWORLD_STAGE_ROUTE_SIGNAL_TRAFFIC": "1"},
            clear=False,
        ):
            runner.configure_benchmark_environment(args)
            self.assertEqual(
                "0", os.environ["SIMWORLD_STAGE_ROUTE_SIGNAL_TRAFFIC"]
            )

    def test_sidewalk_trigger_controls_can_be_disabled_independently(self):
        cases = (
            ("--no-pedestrians", "pedestrians"),
            ("--no-movable-obstacles", "movable_obstacles"),
            ("--no-irregular-npcs", "irregular_npcs"),
            ("--no-falling-objects", "falling_objects"),
        )
        attributes = tuple(attribute for _, attribute in cases)

        for flag, disabled_attribute in cases:
            with self.subTest(flag=flag):
                args = self.resolve(flag)
                for attribute in attributes:
                    self.assertEqual(
                        getattr(args, attribute),
                        attribute != disabled_attribute,
                    )

    def test_optimized_profile_keeps_only_final_json_artifacts(self):
        args = self.resolve("--rollout-profile", "optimized")
        self.assertFalse(args.record_per_step)
        self.assertTrue(args.fast_simulation)

    def test_debug_profile_keeps_instructional_prompt(self):
        args = self.resolve("--rollout-profile", "debug")
        self.assertEqual(args.prompt_style, "instructional")
        self.assertEqual(args.qwen_max_tokens, 128)

    def test_explicit_debug_output_budget_is_preserved(self):
        args = self.resolve(
            "--rollout-profile",
            "debug",
            "--qwen-max-tokens",
            "192",
        )
        self.assertEqual(args.qwen_max_tokens, 192)

    def test_prompt_style_can_override_profile_default(self):
        args = self.resolve(
            "--rollout-profile", "optimized", "--prompt-style", "adaptive"
        )
        self.assertEqual(args.prompt_style, "adaptive")

    def test_replayed_realtime_inference_is_explicitly_selectable(self):
        args = self.resolve("--no-concurrent-realtime-inference")
        self.assertFalse(args.concurrent_realtime_inference)

    def test_explicit_low_level_flags_override_profile_defaults(self):
        args = self.resolve(
            "--rollout-profile",
            "optimized",
            "--record-per-step",
            "--no-fast-simulation",
        )
        self.assertTrue(args.record_per_step)
        self.assertFalse(args.fast_simulation)


class SourceProvenanceTests(unittest.TestCase):
    def test_suite_config_has_stable_canonical_experiment_contract(self):
        args, maps, difficulties, plan = suite_test_inputs()
        first = runner.build_suite_config(args, maps, difficulties, plan)
        second = runner.build_suite_config(args, maps, difficulties, plan)

        self.assertEqual(
            first["experiment_contract"],
            second["experiment_contract"],
        )
        payload = dict(first)
        contract = payload.pop("experiment_contract")
        self.assertEqual("canonical_suite_config_sha256_v1", contract["version"])
        self.assertEqual(runner.canonical_json_sha256(payload), contract["sha256"])

    def test_semantic_change_changes_experiment_contract(self):
        args, maps, difficulties, plan = suite_test_inputs()
        original = runner.build_suite_config(args, maps, difficulties, plan)
        args.seed += 1
        changed = runner.build_suite_config(args, maps, difficulties, plan)

        self.assertNotEqual(
            original["experiment_contract"]["sha256"],
            changed["experiment_contract"]["sha256"],
        )

    def test_source_provenance_hashes_all_selected_map_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            map_dir = root / "data/map"
            map_dir.mkdir(parents=True)
            (root / "config.yaml").write_text("seed: 0\n", encoding="utf-8")
            (root / "data/ue_assets.json").write_text("{}\n", encoding="utf-8")
            (map_dir / "tasks.json").write_text('{"tasks":[{}]}\n', encoding="utf-8")
            (map_dir / "roads.json").write_text('{"roads":[]}\n', encoding="utf-8")
            (map_dir / "obstacles.json").write_text('{"nodes":[]}\n', encoding="utf-8")
            profile = root / "profile.json"
            profile.write_text('{"rollout":{}}\n', encoding="utf-8")
            ue_root = root / "ue"
            ue_root.mkdir()
            launcher = ue_root / "SimWorld.sh"
            launcher.write_text("#!/bin/sh\n", encoding="utf-8")
            manifest = ue_root / "Manifest_UFSFiles_Linux.txt"
            manifest.write_text("pakchunk0\n", encoding="utf-8")
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            args = runner.argparse.Namespace(
                benchmark_profile_path=str(profile),
                ue_launcher=str(launcher),
                qwen_model_path=str(model),
            )
            expected_launcher_hash = runner.sha256_file(launcher)

            with mock.patch.object(
                runner,
                "git_source_state",
                return_value={
                    "commit": "abc123",
                    "tracked_worktree_clean": True,
                },
            ):
                provenance = runner.build_source_provenance(
                    [runner.MapSpec("map", "RT", "data/map/tasks.json")],
                    args,
                    repo_root=root,
                )

        hashes = provenance["authoritative_input_sha256"]
        self.assertIn("data/map/tasks.json", hashes)
        self.assertIn("data/map/roads.json", hashes)
        self.assertIn("data/map/obstacles.json", hashes)
        self.assertIn("config.yaml", hashes)
        self.assertIn("profile.json", hashes)
        self.assertEqual("abc123", provenance["repository"]["commit"])
        self.assertEqual(
            expected_launcher_hash,
            provenance["ue_launcher_and_manifest_sha256"][str(launcher)],
        )
        self.assertIsNotNone(provenance["qwen_model_inventory_sha256"])

    def test_dirty_tracked_source_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "dirty tracked worktree"):
            runner.require_clean_source({
                "repository": {"tracked_worktree_clean": False}
            })

    def test_source_revision_mismatch_blocks_resume(self):
        stored = {"source_provenance": {"repository": {"commit": "old"}}}
        current = {"source_provenance": {"repository": {"commit": "new"}}}
        differences = runner.suite_config_differences(stored, current)
        self.assertTrue(any("source_provenance" in item for item in differences))


class ArtifactTests(unittest.TestCase):
    def test_keyboard_interrupt_marks_active_rollout_interrupted(self):
        args, _, _, plan = suite_test_inputs()
        args.experiment_contract_sha256 = "contract"
        args.resume = False
        args.static_signal_vehicles = True
        args.conflict_vehicle_launch_probability = 1.0
        args.cell_max_attempts = 1

        class FakeProcess:
            alive = True
            process = None

        class FakeUEServer:
            def __init__(self, _args, _log_path):
                self.process = FakeProcess()
                self.startup_seconds = 0.25

            def start(self, _map):
                return None

            def stop(self):
                return 0.0

        class FakeUnrealCV:
            def __init__(self, **_kwargs):
                pass

            def disconnect(self):
                pass

        class FakeCommunicator:
            def __init__(self, unrealcv):
                self.unrealcv = unrealcv

        class InterruptingManager:
            def __init__(self, **_kwargs):
                self.last_result_path = None

            def run_single_task(self, *_args, **_kwargs):
                raise KeyboardInterrupt

        class FakeMetrics:
            health_probe = None

            def reset(self):
                pass

            def snapshot(self):
                return {"request_count": 0, "failed_request_count": 0, "wall_seconds": 0.0}

        with tempfile.TemporaryDirectory() as temporary:
            suite_dir = Path(temporary)
            with mock.patch.object(runner, "UEServer", FakeUEServer):
                with self.assertRaises(KeyboardInterrupt):
                    runner.execute_rollout(
                        args,
                        suite_dir,
                        plan[0],
                        (
                            FakeCommunicator,
                            FakeUnrealCV,
                            InterruptingManager,
                            FakeMetrics(),
                        ),
                    )

            status = json.loads(
                (
                    runner.run_dir_for(suite_dir, plan[0])
                    / "run_status.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(status["status"], "interrupted")
        self.assertEqual(status["interruption_reason"], "keyboard_interrupt")
        self.assertEqual(status["attempt"], 1)
        self.assertEqual(status["experiment_contract_sha256"], "contract")

    def test_aggregate_rejects_mixed_experiment_contracts(self):
        with self.assertRaisesRegex(ValueError, "different experiment contracts"):
            runner.aggregate_summary(
                [
                    {"status": "completed", "experiment_contract_sha256": "a"},
                    {"status": "completed", "experiment_contract_sha256": "b"},
                ],
                plan_count=2,
            )

    def test_aggregate_rejects_identified_and_unidentified_records(self):
        with self.assertRaisesRegex(ValueError, "identified and unidentified"):
            runner.aggregate_summary(
                [
                    {"status": "completed", "experiment_contract_sha256": "a"},
                    {"status": "completed"},
                ],
                plan_count=2,
            )

    def test_empty_interrupted_aggregate_keeps_suite_contract(self):
        summary = runner.aggregate_summary(
            [],
            plan_count=36,
            expected_contract_sha256="suite-contract",
        )

        self.assertEqual(
            summary["experiment_contract_sha256"],
            "suite-contract",
        )
        self.assertFalse(summary["comparison_eligible"])
        self.assertEqual(summary["recorded_rollouts"], 0)
        self.assertEqual(summary["completed_rollouts"], 0)
        self.assertIn(
            "recorded 0 of 36 planned rollouts",
            summary["comparison_ineligibility_reasons"],
        )

    def test_aggregate_rejects_record_different_from_suite_contract(self):
        with self.assertRaisesRegex(ValueError, "differs from the suite contract"):
            runner.aggregate_summary(
                [
                    {
                        "status": "completed",
                        "experiment_contract_sha256": "record-contract",
                    }
                ],
                plan_count=1,
                expected_contract_sha256="suite-contract",
            )

    def test_manifest_path_supports_external_output_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            external = Path(temporary) / "suite"
            rendered = runner.serialize_manifest_path(
                external,
                repo_root=MODULE_PATH.parent.parent,
            )
        self.assertEqual(rendered, str(external.resolve()))

    def test_ue_attempt_logs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "ue_attempt_01.log").write_text("first", encoding="utf-8")
            (run_dir / "ue_attempt_03.log").write_text("third", encoding="utf-8")
            (run_dir / "ue_attempt_bad.log").write_text("ignored", encoding="utf-8")

            self.assertEqual(runner.next_ue_attempt_number(run_dir), 4)

    def test_jsonl_checkpoint_replaces_stale_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.jsonl"
            path.write_text('{"ordinal": 1, "status": "error"}\n', encoding="utf-8")
            records = [
                {"ordinal": 1, "status": "completed"},
                {"ordinal": 2, "status": "completed"},
            ]
            runner.write_jsonl(path, records)
            parsed = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(parsed, records)

    def test_step_artifact_summary_counts_images_and_inference_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step_dir = root / "task_1_steps" / "step_0000"
            step_dir.mkdir(parents=True)
            (step_dir / "input.png").write_bytes(b"input")
            (step_dir / "output.png").write_bytes(b"output")
            (step_dir / "step_0000_manifest.json").write_text(
                json.dumps(
                    {
                        "input_images": ["input.png"],
                        "output_image": "output.png",
                        "metrics": {"response_time": 1.25},
                    }
                ),
                encoding="utf-8",
            )
            summary = runner.step_artifact_summary(root)
        self.assertEqual(summary["step_manifest_count"], 1)
        self.assertEqual(summary["input_image_count"], 1)
        self.assertEqual(summary["output_image_count"], 1)
        self.assertEqual(summary["model_inference_wall_seconds"], 1.25)
        self.assertEqual(summary["missing_artifacts"], [])

    def test_terminal_state_rejects_infrastructure_truncation(self):
        self.assertTrue(
            runner.rollout_reached_terminal_state(
                {"success": True, "failed": False, "final_step": 2, "max_steps": 10}
            )
        )
        self.assertTrue(
            runner.rollout_reached_terminal_state(
                {"success": False, "failed": True, "final_step": 2, "max_steps": 10}
            )
        )
        self.assertTrue(
            runner.rollout_reached_terminal_state(
                {"success": False, "failed": False, "final_step": 10, "max_steps": 10}
            )
        )
        self.assertFalse(
            runner.rollout_reached_terminal_state(
                {"success": False, "failed": False, "final_step": 2, "max_steps": 10}
            )
        )

    def test_termination_reason_classifies_max_steps_and_explicit_outcomes(self):
        self.assertEqual(
            "max_steps",
            runner.rollout_termination_reason(
                {"success": False, "failed": False, "final_step": 10, "max_steps": 10}
            ),
        )
        self.assertEqual(
            "vehicle_collision",
            runner.rollout_termination_reason(
                {
                    "success": False,
                    "failed": True,
                    "failure_reason": "vehicle_collision",
                    "final_step": 2,
                    "max_steps": 10,
                }
            ),
        )
        self.assertEqual(
            "success",
            runner.rollout_termination_reason(
                {"success": True, "failed": False, "final_step": 2, "max_steps": 10}
            ),
        )

    def test_error_retry_classification_stops_only_explicit_failures(self):
        self.assertTrue(
            runner.error_record_is_retryable({"status": "error"})
        )
        self.assertTrue(
            runner.error_record_is_retryable(
                {"status": "error", "retryable": True}
            )
        )
        self.assertFalse(
            runner.error_record_is_retryable(
                {"status": "error", "retryable": False}
            )
        )
        self.assertFalse(
            runner.error_record_is_retryable(
                {"status": "completed", "retryable": False}
            )
        )

    def test_aggregate_distinguishes_terminal_failure_from_max_steps(self):
        records = [
            {
                "status": "completed",
                "success": False,
                "failed": False,
                "termination_reason": "max_steps",
                "experiment_contract_sha256": "contract",
                "timing": {},
            },
            {
                "status": "completed",
                "success": False,
                "failed": True,
                "termination_reason": "vehicle_collision",
                "experiment_contract_sha256": "contract",
                "timing": {},
            },
        ]

        summary = runner.aggregate_summary(records, plan_count=2)

        self.assertEqual(summary["failed_tasks"], 1)
        self.assertEqual(summary["max_steps_tasks"], 1)
        self.assertEqual(summary["non_successful_tasks"], 2)
        self.assertFalse(summary["comparison_eligible"])
        self.assertEqual(
            summary["comparison_ineligibility_reasons"],
            ["artifact audits have not passed for every planned rollout"],
        )
        self.assertFalse(summary["artifact_audits_passed"])
        self.assertEqual(summary["artifact_audited_rollouts"], 0)

    def test_partial_aggregate_is_not_comparison_eligible(self):
        summary = runner.aggregate_summary(
            [
                {
                    "status": "completed",
                    "success": True,
                    "failed": False,
                    "experiment_contract_sha256": "contract",
                    "timing": {},
                }
            ],
            plan_count=2,
        )

        self.assertFalse(summary["comparison_eligible"])
        self.assertEqual(
            summary["comparison_ineligibility_reasons"],
            [
                "recorded 1 of 2 planned rollouts",
                "completed 1 of 2 planned rollouts",
                "artifact audits have not passed for every planned rollout",
            ],
        )

    def test_partial_attempt_archive_prevents_stale_step_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            result = run_dir / "task_1.json"
            steps = run_dir / "task_1_steps"
            video = run_dir / "rollout_agent_input_annotated.mp4"
            traffic_audit = run_dir / "rollout_traffic_e2e_audit.json"
            result.write_text("{}", encoding="utf-8")
            steps.mkdir()
            video.write_bytes(b"stale")
            traffic_audit.write_text('{"passed": true}', encoding="utf-8")
            destination = runner.archive_partial_attempt(run_dir, "retry 1")
            self.assertIsNotNone(destination)
            self.assertFalse(result.exists())
            self.assertFalse(steps.exists())
            self.assertFalse(video.exists())
            self.assertFalse(traffic_audit.exists())
            self.assertTrue((destination / result.name).is_file())
            self.assertTrue((destination / steps.name).is_dir())
            self.assertEqual(b"stale", (destination / video.name).read_bytes())
            self.assertTrue((destination / traffic_audit.name).is_file())




class UEServerTests(unittest.TestCase):
    def test_ue_shutdown_waits_for_unrealcv_socket_reuse(self):
        args = runner.argparse.Namespace(
            ue_host="127.0.0.1",
            ue_port=9006,
            process_shutdown_timeout=10,
        )
        server = runner.UEServer(args, Path("ue.log"))
        server.process = mock.Mock()
        server.process.stop.return_value = 1.25

        with (
            mock.patch.object(
                runner,
                "tcp_bind_available",
                side_effect=[False, False, True],
            ) as bind_available,
            mock.patch.object(runner.time, "sleep") as sleep,
        ):
            elapsed = server.stop()

        self.assertEqual(1.25, elapsed)
        server.process.stop.assert_called_once_with(10)
        self.assertEqual(bind_available.call_count, 3)
        self.assertEqual(
            sleep.call_args_list,
            [mock.call(0.25), mock.call(0.25)],
        )

    def test_tcp_bind_probe_detects_actual_port_ownership(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            self.assertFalse(runner.tcp_bind_available("127.0.0.1", port))

        self.assertTrue(runner.tcp_bind_available("127.0.0.1", port))

    def test_tcp_listener_probe_does_not_connect(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            self.assertFalse(runner.tcp_listening("127.0.0.1", port))
            listener.listen()
            self.assertTrue(runner.tcp_listening("127.0.0.1", port))

        self.assertFalse(runner.tcp_listening("127.0.0.1", port))

    def test_ue_launch_explicitly_disables_unused_ray_tracing_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "SimWorld.sh"
            launcher.write_text("#!/bin/sh\n", encoding="utf-8")
            args = runner.argparse.Namespace(
                ue_launcher=str(launcher),
                ue_host="127.0.0.1",
                ue_port=9006,
                ue_gpu=2,
                ue_width=900,
                ue_height=800,
                ue_fps=15,
                ue_no_rhi_thread=True,
                ue_ready_timeout=1,
                ue_settle_seconds=0,
            )
            server = runner.UEServer(args, root / "ue.log")
            starter = mock.MagicMock()
            server.process = mock.Mock(start=starter, alive=True)

            with mock.patch.object(
                runner, "tcp_listening", side_effect=[False, True]
            ):
                server.start("RT15")

            command = starter.call_args.args[0]
            self.assertIn("-noraytracing", command)

    def test_gpu_startup_diagnostic_distinguishes_hardware_and_bad_devices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = runner.UEServer(
                runner.argparse.Namespace(ue_gpu=5, ue_port=9006),
                root / "ue_attempt_01.log",
            )

            cases = (
                (
                    "Cannot create a Vulkan device. Try updating your video driver",
                    "could not create a Vulkan device",
                ),
                (
                    "Cannot start listening on port 9006, Port might be in use",
                    "could not bind UnrealCV port",
                ),
                (
                    "Falling back to first device...",
                    "fell back to another device",
                ),
                (
                    "deviceName=llvmpipe driverName=llvmpipe",
                    "software Vulkan renderer",
                ),
            )
            for log_text, expected in cases:
                with self.subTest(log_text=log_text):
                    server.internal_log_path.write_text(
                        log_text,
                        encoding="utf-8",
                    )
                    self.assertIn(
                        expected,
                        server._gpu_startup_error(),
                    )

            server.internal_log_path.write_text(
                "deviceName=NVIDIA RTX A5000 driverName=NVIDIA",
                encoding="utf-8",
            )
            self.assertIsNone(server._gpu_startup_error())

    def test_start_rejects_gpu_fallback_without_waiting_for_port_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "SimWorld.sh"
            launcher.write_text("#!/bin/sh\n", encoding="utf-8")

            args = runner.argparse.Namespace(
                ue_launcher=str(launcher),
                ue_host="127.0.0.1",
                ue_port=9006,
                ue_gpu=5,
                ue_width=640,
                ue_height=480,
                ue_fps=30,
                ue_no_rhi_thread=False,
                ue_ready_timeout=120,
                ue_settle_seconds=0,
            )
            server = runner.UEServer(
                args,
                root / "ue_attempt_01.log",
            )
            server.internal_log_path.write_text(
                "Falling back to first device...",
                encoding="utf-8",
            )

            with (
                mock.patch.object(
                    runner,
                    "tcp_listening",
                    return_value=False,
                ),
                mock.patch.object(
                    server.process,
                    "start",
                ) as process_start,
                self.assertRaisesRegex(
                    RuntimeError,
                    "fell back to another device",
                ),
            ):
                server.start("/Game/Maps/Test")

            process_start.assert_called_once()

    def test_runner_rpc_timeout_controls_simworld_clients(self):
        previous = os.environ.get("SIMWORLD_UNREALCV_REQUEST_TIMEOUT_SECONDS")
        previous_retries = os.environ.get("SIMWORLD_UNREALCV_RECONNECT_RETRIES")
        try:
            value = runner.configure_unrealcv_request_timeout(30, 0)
            self.assertEqual(value, 30.0)
            self.assertEqual(
                os.environ["SIMWORLD_UNREALCV_REQUEST_TIMEOUT_SECONDS"],
                "30",
            )
            self.assertEqual(
                os.environ["SIMWORLD_UNREALCV_RECONNECT_RETRIES"],
                "0",
            )
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                runner.configure_unrealcv_request_timeout(0)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                runner.configure_unrealcv_request_timeout(30, -1)
        finally:
            if previous is None:
                os.environ.pop("SIMWORLD_UNREALCV_REQUEST_TIMEOUT_SECONDS", None)
            else:
                os.environ["SIMWORLD_UNREALCV_REQUEST_TIMEOUT_SECONDS"] = previous
            if previous_retries is None:
                os.environ.pop("SIMWORLD_UNREALCV_RECONNECT_RETRIES", None)
            else:
                os.environ["SIMWORLD_UNREALCV_RECONNECT_RETRIES"] = previous_retries

    def test_renderer_validation_rejects_fallback_and_software_adapter(self):
        fallback = (
            "Tried to use graphics adapter at index 5 as specified by command "
            "line, but only 5 Adapter(s) found. Falling back to first device...\n"
            "- DeviceName: NVIDIA RTX A5000\n"
        )
        software = (
            "- DeviceName: llvmpipe (LLVM 20.1.2, 256 bits)\n"
            "- DeviceID=0x0 Type=VK_PHYSICAL_DEVICE_TYPE_CPU\n"
        )

        with self.assertRaisesRegex(runner.UEInfrastructureError, "fell back"):
            runner.validate_ue_renderer_log(fallback, "5")
        with self.assertRaisesRegex(runner.UEInfrastructureError, "software"):
            runner.validate_ue_renderer_log(software, "4")

    def test_renderer_validation_accepts_explicit_nvidia_adapter(self):
        renderer = runner.validate_ue_renderer_log(
            "Using device at index 1 of 5 as specified by command line...\n"
            "- DeviceName: NVIDIA RTX A5000\n"
            "- DeviceID=0x2231 Type=VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n",
            "1",
        )
        self.assertEqual(renderer["device_name"], "NVIDIA RTX A5000")
        self.assertFalse(renderer["adapter_fallback"])

    def test_start_forwards_each_configured_unrealcv_port(self):
        for port in (9006, 19123):
            with self.subTest(port=port), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                launcher = root / "SimWorld.sh"
                launcher.write_text("#!/bin/sh\n", encoding="utf-8")

                args = runner.argparse.Namespace(
                    ue_launcher=str(launcher),
                    ue_host="127.0.0.1",
                    ue_port=port,
                    ue_gpu=0,
                    ue_width=640,
                    ue_height=480,
                    ue_fps=30,
                    ue_no_rhi_thread=False,
                    ue_ready_timeout=5,
                    ue_settle_seconds=0,
                )
                server = runner.UEServer(args, root / "ue_attempt_01.log")

                with (
                    mock.patch.object(
                        runner, "tcp_listening", side_effect=[False, True]
                    ) as tcp_listening,
                    mock.patch.object(server.process, "start") as process_start,
                ):
                    server.start("/Game/Maps/Test")

                process_start.assert_called_once()
                command = process_start.call_args.args[0]

                cvport_index = command.index("-cvport")
                self.assertEqual(
                    command[cvport_index : cvport_index + 2],
                    ["-cvport", str(port)],
                )
                self.assertEqual(command.count("-cvport"), 1)
                self.assertFalse(
                    any(item.startswith("-cvport=") for item in command)
                )
                self.assertFalse(
                    any(item.startswith("-unrealcv_port") for item in command)
                )
                self.assertEqual(
                    tcp_listening.call_args_list,
                    [
                        mock.call("127.0.0.1", port),
                        mock.call("127.0.0.1", port),
                    ],
                )


class QwenServerTests(unittest.TestCase):
    def test_default_server_disables_video_without_capping_images(self):
        args = runner.parse_args(
            [
                "--model",
                "Qwen/Qwen3-VL-8B-Instruct",
                "--qwen-python",
                "/env/bin/python",
                "--qwen-port",
                "30012",
                "--qwen-url",
                "http://127.0.0.1:30012/v1",
            ]
        )
        server = runner.QwenServer(args, Path("logs"))

        command = server._default_command(Path("/models/qwen"))

        self.assertEqual(command[:3], [
            "/env/bin/python",
            "-m",
            "vllm.entrypoints.openai.api_server",
        ])
        limits = command[command.index("--limit-mm-per-prompt") + 1]
        self.assertEqual('{"video":0}', limits)
        self.assertNotIn("qwen3vl_thinking_openai_server.py", " ".join(command))

    def test_server_can_apply_an_explicit_image_count_ceiling(self):
        args = runner.parse_args(["--qwen-max-images", "12"])
        server = runner.QwenServer(args, Path("logs"))

        command = server._default_command(Path("/models/qwen"))

        limits = command[command.index("--limit-mm-per-prompt") + 1]
        self.assertEqual('{"video":0,"image":12}', limits)

    def test_warmup_image_matches_benchmark_observation_resolution(self):
        args = runner.parse_args(
            ["--observation-width", "1080", "--observation-height", "960"]
        )
        payload = runner.qwen_warmup_payload(args)
        url = payload["messages"][0]["content"][0]["image_url"]["url"]
        png = base64.b64decode(url.split(",", 1)[1])
        width, height = struct.unpack(">II", png[16:24])

        self.assertEqual((width, height), (1080, 960))
        self.assertEqual(payload["max_tokens"], 4)

    def test_external_server_waits_for_supervised_restart(self):
        args = runner.parse_args(
            [
                "--model",
                "Qwen/Qwen3-VL-8B-Instruct",
                "--qwen-mode",
                "external",
                "--qwen-ready-timeout",
                "30",
                "--no-qwen-warmup",
            ]
        )
        server = runner.QwenServer(args, Path("logs"))

        with (
            mock.patch.object(
                runner,
                "qwen_model_ids",
                side_effect=[[], [], ["Qwen/Qwen3-VL-8B-Instruct"]],
            ) as model_ids,
            mock.patch.object(runner.time, "sleep") as sleep,
        ):
            server.ensure_ready()

        self.assertEqual(model_ids.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(args.model, "Qwen/Qwen3-VL-8B-Instruct")



def suite_test_inputs(**overrides):
    values = {
        "rounds": 1,
        "task_limit": 1,
        "seed": 42,
        "max_steps": 10,
        "prompt_style": "default",
        "token_based": False,
        "use_action_frames": False,
        "model": "qwen3-vl-8b",
        "enable_thinking": False,
        "qwen_model_path": "/models/qwen3-vl-8b",
        "qwen_max_tokens": 2048,
        "ue_width": 640,
        "ue_height": 480,
        "observation_width": 720,
        "observation_height": 640,
        "observation_fov_deg": 100.0,
        "observation_camera_pitch_deg": -25.0,
        "ue_fps": 30,
        "rollout_profile": "full",
        "record_per_step": True,
        "record_png_compress_level": 3,
        "fast_simulation": False,
        "ue_host": "127.0.0.1",
        "ue_port": 9006,
        "ue_gpu": 0,
        "ue_launcher": "/runtime/SimWorld.sh",
        "ue_no_rhi_thread": False,
        "ue_ready_timeout": 30,
        "ue_settle_seconds": 0,
        "unrealcv_request_timeout": 30,
        "unrealcv_reconnect_retries": 0,
        "ue_max_attempts": 2,
        "qwen_mode": "external",
        "qwen_url": "http://127.0.0.1:8000/v1",
        "qwen_port": 8000,
        "qwen_gpu": 1,
        "model_request_timeout": 180,
    }
    values.update(overrides)
    args = runner.argparse.Namespace(**values)
    maps = [runner.MapSpec("map", "RT", "data/map/tasks.json")]
    plan = [
        runner.RolloutSpec(
            ordinal=1,
            phase="lead",
            map_name="map",
            ue_map="RT",
            task_file="data/map/tasks.json",
            task_index=0,
            task_id=1,
            difficulty="easy",
            env_mode="realtime",
            realtime_thinking=True,
            round_id=1,
        )
    ]
    args.traffic_policy = overrides.get(
        "traffic_policy",
        "visual_only",
    )
    args.red_light_conflict_vehicle = overrides.get(
        "red_light_conflict_vehicle",
        True,
    )
    args.pedestrians = overrides.get("pedestrians", True)
    args.movable_obstacles = overrides.get("movable_obstacles", True)
    args.irregular_npcs = overrides.get("irregular_npcs", True)
    args.falling_objects = overrides.get("falling_objects", True)
    return args, maps, ("easy",), plan


def test_benchmark_agent_config_propagates_qwen_api_errors():
    args, _, _, _ = suite_test_inputs()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "agent.json"
        runner.write_agent_config(path, args)
        config = json.loads(path.read_text(encoding="utf-8"))

    assert config["llm"][0]["raise_on_api_error"] is True


def test_benchmark_agent_config_records_deterministic_sampling_seed():
    args, _, _, _ = suite_test_inputs(seed=42)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "agent.json"
        runner.write_agent_config(path, args)
        config = json.loads(path.read_text(encoding="utf-8"))

    llm = config["llm"][0]
    assert llm["temperature"] == 0.7
    assert llm["top_p"] == 1.0
    assert llm["seed"] == 42


def test_benchmark_agent_config_propagates_hard_thinking_budget():
    args, _, _, _ = suite_test_inputs(
        enable_thinking=True,
        reasoning_budget_tokens=64,
        reasoning_budget_answer_tokens=128,
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "agent.json"
        runner.write_agent_config(path, args)
        config = json.loads(path.read_text(encoding="utf-8"))

    llm = config["llm"][0]
    assert llm["reasoning"] is True
    assert llm["extra_body"]["reasoning_budget_mode"] == "force_close"
    assert llm["extra_body"]["reasoning_budget_tokens"] == 64
    assert llm["extra_body"]["reasoning_budget_answer_tokens"] == 128


def test_qwen_restart_resynchronizes_resolved_model_alias_in_agent_config():
    args, _, _, _ = suite_test_inputs(model="qwen3-vl-8b")

    class AliasResolvingServer:
        def ensure_ready(self):
            args.model = "Qwen/Qwen3-VL-8B-Instruct"

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "agent.json"
        runner.write_agent_config(path, args)
        runner.ensure_qwen_ready_and_sync_agent_config(
            AliasResolvingServer(), path, args
        )
        config = json.loads(path.read_text(encoding="utf-8"))

    assert config["llm"][0]["model"] == "Qwen/Qwen3-VL-8B-Instruct"


class SuiteConfigurationTests(unittest.TestCase):
    def test_suite_records_two_rule_traffic_policy(self):
        args, maps, difficulties, plan = suite_test_inputs()

        config = runner.build_suite_config(args, maps, difficulties, plan)

        self.assertEqual(
            {
                "version": "destination_collision_route_budget_v5",
                "destination_success_terminal": True,
                "vehicle_collision_terminal": True,
                "consecutive_building_collision_limit": 3,
                "stagnation_prompt_guidance": False,
                "stagnation_terminal": False,
                "stagnation_progress_epsilon_cm": 25.0,
                "signal_required_waits_exempt_from_stagnation": True,
                "default_max_steps": "3_per_full_shortest_route_meter",
            },
            config["termination_policy"],
        )
        self.assertEqual(
            {
                "version": "continuous_region_occupancy_v1",
                "count_once_per_continuous_occupancy": True,
                "spatial_rearm_distance_cm": 450.0,
            },
            config["hazard_overlap_policy"],
        )
        self.assertEqual(
            {
                "static_obstacle_geometry": (
                    "internal_runtime_geometry_not_model_prompt"
                ),
                "mapped_static_obstacles_in_model_context": 0,
            },
            config["model_input_geometry_policy"],
        )
        self.assertEqual(
            {
                "version": "two_rule_traffic_events_v6",
                "collision_authority": (
                    "ue_counters_plus_launched_vehicle_swept_radius"
                ),
                "vehicle_collision_scope": "launched_conflict_vehicle_only",
                "painted_crosswalk_half_width_cm": 210,
                "roadway_occupancy_guard_half_width_cm": 600,
                "rendered_sidewalk_half_width_cm": 200,
                "illegal_crossing_authority": (
                    "ue_touched_road_outside_strict_pedestrian_geometry"
                ),
                "traffic_rule_events": [
                    "illegal_crossing",
                    "red_light_violation",
                ],
                "conflict_vehicle_triggers": [
                    "illegal_crossing",
                    "red_light_violation",
                ],
                "conflict_vehicle_release_wait_timeout_s": 12.0,
                "red_light_rule": (
                    "walk_entry_admission_persists_to_far_curb"
                ),
            },
            config["traffic_evaluation_policy"],
        )
        self.assertEqual(
            "two_rule_traffic_events_v6",
            config["artifact_schema_version"],
        )
        self.assertEqual(
            1.0,
            config["conflict_vehicle_launch_probability"],
        )
        self.assertEqual(
            {"temperature": 0.7, "top_p": 1.0, "seed": 42},
            config["model_sampling_policy"],
        )
        self.assertEqual(
            {
                "version": "visual_inference_declarative_rules_only_v5",
                "candidate_action_tactical_guidance": False,
                "visual_only_traffic_rules": "declarative_only",
                "visual_only_prescribed_traffic_actions": False,
                "visual_only_instance_route_edge_context": False,
                "visual_only_symbolic_signal_context": False,
                "visual_only_traffic_feedback": "generic_event_only",
            },
            config["prompt_policy"],
        )
        self.assertEqual(
            {
                "version": "collision_clear_crossing_heads_v2",
                "vehicle_head_normal_offset_cm": -1400.0,
                "vehicle_head_radial_offset_cm": 1400.0,
                "pedestrian_head_endpoint_offset_cm": 100.0,
                "pedestrian_head_lateral_offset_cm": 500.0,
                "minimum_scripted_lane_clearance_cm": 300.0,
                "collision_enabled": True,
            },
            config["traffic_signal_placement_policy"],
        )
        self.assertEqual(
            {
                "version": "forward_reload_bevel_clearance_detour_recovery_v13",
                "physical_collision_enabled": True,
                "right_hand_lane_offset_cm": 200.0,
                "lane_offset_jitter_cm": 0.0,
                "observed_capsule_blocking_distance_cm": 269.0,
                "minimum_beveled_corner_center_clearance_cm": 282.84,
                "non_loop_return_lane": "opposite_right_hand_lane",
                "minimum_spawn_separation_cm": 325.0,
                "minimum_static_obstacle_separation_cm": 250.0,
                "force_overlapping_spawn_fallback": False,
                "motion_sample_interval_s": 2.0,
                "motion_threshold_cm": 25.0,
                "controller_restart_after_s": 6.0,
                "endpoint_recycle_distance_cm": 125.0,
                "waypoint_reload_strategy": "nearest_forward_segment",
                "same_pose_controller_recreation_after_stalled_restarts": 2,
                "same_pose_recreation_agent_clearance_cm": 325.0,
                "same_pose_recreation_actor_clearance_cm": 325.0,
                "same_pose_recreation_tolerance_cm": 25.0,
                "same_name_respawn_release_timeout_s": 2.0,
                "same_name_respawn_poll_interval_s": 0.05,
                "local_detour_enabled": True,
                "local_detour_after_stalled_restarts": 1,
                "local_detour_trigger_distance_cm": 450.0,
                "local_detour_static_clearance_cm": 350.0,
                "local_detour_secondary_static_clearance_cm": 100.0,
                "local_detour_forward_clearance_cm": 350.0,
                "local_detour_actor_clearance_cm": 175.0,
                "local_detour_agent_clearance_cm": 325.0,
                "local_detour_sidewalk_or_crosswalk_only": True,
                "local_detour_max_waypoints": 3,
                "controller_heartbeat_scope": "scripted_pedestrian",
                "default_benchmark_speed_cm_s": 100.0,
                "teleport_recovery": False,
            },
            config["scripted_pedestrian_motion_policy"],
        )

    def test_suite_from_older_termination_policy_cannot_resume(self):
        args, maps, difficulties, plan = suite_test_inputs()
        current = runner.build_suite_config(args, maps, difficulties, plan)

        with tempfile.TemporaryDirectory() as temporary:
            suite_dir = Path(temporary) / "suite"
            suite_dir.mkdir()
            stored = dict(current)
            stored.pop("termination_policy")
            runner.write_json(suite_dir / "suite_config.json", stored)

            with self.assertRaisesRegex(ValueError, "termination_policy"):
                runner.prepare_suite_directory(
                    suite_dir,
                    resume=True,
                    suite_config=current,
                )

    def test_traffic_semantic_changes_block_resume(self):
        cases = [
            (
                "traffic_policy",
                "safety_assisted",
                "traffic_policy",
            ),
            (
                "red_light_conflict_vehicle",
                False,
                "red_light_conflict_vehicle_enabled",
            ),
            (
                "conflict_vehicle_launch_probability",
                0.5,
                "conflict_vehicle_launch_probability",
            ),
            (
                "conflict_vehicle_min_launch_distance_m",
                2.0,
                "conflict_vehicle_min_launch_distance_m",
            ),
            (
                "conflict_vehicle_max_launch_distance_m",
                4.0,
                "conflict_vehicle_max_launch_distance_m",
            ),
            (
                "conflict_vehicle_impact_radius_m",
                2.0,
                "conflict_vehicle_impact_radius_m",
            ),
            (
                "conflict_vehicle_release_wait_timeout_s",
                6.0,
                "conflict_vehicle_release_wait_timeout_s",
            ),
            (
                "pedestrians",
                False,
                "pedestrians_enabled",
            ),
            (
                "movable_obstacles",
                False,
                "movable_obstacles_enabled",
            ),
            (
                "irregular_npcs",
                False,
                "irregular_npcs_enabled",
            ),
            (
                "falling_objects",
                False,
                "falling_objects_enabled",
            ),
        ]

        for attribute, changed_value, difference_key in cases:
            with self.subTest(attribute=attribute), tempfile.TemporaryDirectory() as temporary:
                args, maps, difficulties, plan = suite_test_inputs()
                original = runner.build_suite_config(
                    args,
                    maps,
                    difficulties,
                    plan,
                )
                suite_dir = Path(temporary) / "suite"
                runner.prepare_suite_directory(
                    suite_dir,
                    resume=False,
                    suite_config=original,
                )

                setattr(args, attribute, changed_value)
                changed = runner.build_suite_config(
                    args,
                    maps,
                    difficulties,
                    plan,
                )

                with self.assertRaisesRegex(ValueError, difference_key):
                    runner.prepare_suite_directory(
                        suite_dir,
                        resume=True,
                        suite_config=changed,
                    )

    def test_semantic_changes_block_resume_before_overwrite(self):
        args, maps, difficulties, plan = suite_test_inputs()
        original = runner.build_suite_config(args, maps, difficulties, plan)

        with tempfile.TemporaryDirectory() as temporary:
            suite_dir = Path(temporary) / "suite"
            runner.prepare_suite_directory(
                suite_dir, resume=False, suite_config=original
            )
            agent_path = suite_dir / "agents_qwen3vl8b.json"
            agent_path.write_text("sentinel", encoding="utf-8")

            for field, value in (
                ("max_steps", 11),
                ("model", "another-model"),
                ("ue_width", 1280),
                ("ue_height", 720),
                ("observation_width", 1080),
                ("observation_height", 960),
                ("observation_fov_deg", 80.0),
                ("observation_camera_pitch_deg", -20.0),
            ):
                with self.subTest(field=field):
                    changed_args, _, _, _ = suite_test_inputs(**{field: value})
                    changed = runner.build_suite_config(
                        changed_args, maps, difficulties, plan
                    )
                    with self.assertRaisesRegex(ValueError, field):
                        runner.prepare_suite_directory(
                            suite_dir,
                            resume=True,
                            suite_config=changed,
                        )
                    self.assertEqual(
                        agent_path.read_text(encoding="utf-8"),
                        "sentinel",
                    )

    def test_runtime_endpoint_and_gpu_changes_allow_resume(self):
        args, maps, difficulties, plan = suite_test_inputs()
        original = runner.build_suite_config(args, maps, difficulties, plan)

        with tempfile.TemporaryDirectory() as temporary:
            suite_dir = Path(temporary) / "suite"
            runner.prepare_suite_directory(
                suite_dir, resume=False, suite_config=original
            )

            changed_args, _, _, _ = suite_test_inputs(
                ue_port=19123,
                ue_gpu=3,
                qwen_url="http://127.0.0.1:18000/v1",
                qwen_port=18000,
                qwen_gpu=4,
            )
            changed = runner.build_suite_config(
                changed_args, maps, difficulties, plan
            )
            runner.prepare_suite_directory(
                suite_dir,
                resume=True,
                suite_config=changed,
            )

            runtime = runner.runtime_config(changed_args)
            self.assertEqual(runtime["ue"]["port"], 19123)
            self.assertEqual(runtime["ue"]["gpu"], 3)
            self.assertEqual(runtime["qwen"]["port"], 18000)
            self.assertEqual(runtime["qwen"]["gpu"], 4)

    def test_new_run_refuses_existing_suite(self):
        args, maps, difficulties, plan = suite_test_inputs()
        config = runner.build_suite_config(args, maps, difficulties, plan)

        with tempfile.TemporaryDirectory() as temporary:
            suite_dir = Path(temporary) / "suite"
            runner.prepare_suite_directory(
                suite_dir, resume=False, suite_config=config
            )
            with self.assertRaises(FileExistsError):
                runner.prepare_suite_directory(
                    suite_dir,
                    resume=False,
                    suite_config=config,
                )

    def test_each_attempt_records_actual_runtime(self):
        args, _, _, _ = suite_test_inputs(
            ue_port=19123,
            ue_gpu=3,
            qwen_url="http://127.0.0.1:18000/v1",
            qwen_port=18000,
        )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            observed = {
                "status": "observed",
                "gpu_processes": [{"index": 7, "name": "Test GPU"}],
            }
            with mock.patch.object(
                runner,
                "observed_local_qwen_runtime",
                return_value=observed,
            ):
                path = runner.write_attempt_runtime(
                    run_dir,
                    2,
                    args,
                    run_dir / "ue_attempt_02.log",
                )
            record = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(
                record["runtime_provenance_version"],
                "runtime_attempt_v2",
            )
            self.assertEqual(record["attempt"], 2)
            self.assertEqual(record["ue_log"], "ue_attempt_02.log")
            self.assertEqual(record["runtime"]["ue"]["port"], 19123)
            self.assertEqual(record["runtime"]["ue"]["gpu"], 3)
            self.assertEqual(record["runtime"]["qwen"]["port"], 18000)
            self.assertFalse(
                record["runtime"]["qwen"]["configured_launch_gpu_applies"]
            )
            self.assertEqual(
                record["runtime"]["qwen"]["observed_local_service"],
                observed,
            )
            self.assertIn("version", record["runtime"]["python"])
            self.assertEqual(
                len(list(run_dir.glob("runtime_attempt_02_*.json"))),
                1,
            )




class MainFlowTests(unittest.TestCase):
    @staticmethod
    def argv(*extra: str) -> list[str]:
        return [
            "--suite-name",
            "main_flow_suite",
            "--output-root",
            "outputs",
            "--maps",
            "map1_10roads",
            "--difficulties",
            "easy",
            "--rounds",
            "1",
            "--task-limit",
            "1",
            "--max-steps",
            "10",
            "--qwen-mode",
            "external",
            "--qwen-url",
            "http://127.0.0.1:30011/v1",
            "--qwen-port",
            "30011",
            "--ue-port",
            "9006",
            "--plan-only",
            *extra,
        ]

    def test_plan_only_preserves_legacy_manifest_schema(self):
        _, maps, _, plan = suite_test_inputs()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_cwd = Path.cwd()

            try:
                with (
                    mock.patch.object(runner, "REPO_ROOT", root),
                    mock.patch.object(
                        runner,
                        "selected_maps",
                        return_value=maps,
                    ),
                    mock.patch.object(
                        runner,
                        "build_rollout_plan",
                        return_value=plan,
                    ),
                    mock.patch.object(
                        runner,
                        "qwen_model_ids",
                        side_effect=AssertionError(
                            "plan-only must not contact Qwen"
                        ),
                    ) as qwen_probe,
                    mock.patch.object(
                        runner.subprocess,
                        "Popen",
                        side_effect=AssertionError(
                            "plan-only must not start external processes"
                        ),
                    ) as popen,
                    mock.patch.dict(runner.os.environ, {}, clear=False),
                ):
                    self.assertEqual(runner.main(self.argv()), 0)

                    suite_dir = root / "outputs/main_flow_suite"
                    manifest = json.loads(
                        (suite_dir / "experiment_manifest.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    stored_config = json.loads(
                        (suite_dir / "suite_config.json").read_text(
                            encoding="utf-8"
                        )
                    )

                    legacy_keys = {
                        "difficulties",
                        "maps",
                        "rounds",
                        "model",
                        "enable_thinking",
                        "qwen_url",
                        "qwen_model_path",
                        "rollout_profile",
                        "record_per_step",
                        "record_png_compress_level",
                        "fast_simulation",
                        "use_tick",
                        "stop_after_ordinal",
                        "rollout_count",
                        "rollouts",
                    }
                    self.assertTrue(
                        legacy_keys.issubset(manifest),
                        legacy_keys - set(manifest),
                    )
                    self.assertEqual(manifest["suite_config"], stored_config)
                    self.assertEqual(
                        manifest["suite_config_file"],
                        "suite_config.json",
                    )
                    self.assertEqual(
                        manifest["difficulties"],
                        stored_config["difficulties"],
                    )
                    self.assertEqual(
                        manifest["maps"],
                        stored_config["maps"],
                    )
                    self.assertEqual(manifest["rounds"], 1)
                    self.assertEqual(manifest["model"], "qwen3-vl-8b")
                    self.assertEqual(
                        manifest["qwen_url"],
                        "http://127.0.0.1:30011/v1",
                    )
                    qwen_probe.assert_not_called()
                    popen.assert_not_called()
            finally:
                os.chdir(original_cwd)

    def test_suite_semantics_accept_exact_served_qwen_id_for_alias(self):
        self.assertEqual(
            [],
            runner.suite_config_differences(
                {"model": "qwen3-vl-8b"},
                {"model": "Qwen/Qwen3-VL-8B-Instruct"},
            ),
        )
        self.assertTrue(
            runner.suite_config_differences(
                {"model": "qwen3-vl-8b"},
                {"model": "Qwen/Qwen3-VL-32B-Instruct"},
            )
        )

    def test_main_resume_blocks_semantic_drift_and_allows_runtime_change(self):
        _, maps, _, plan = suite_test_inputs()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_cwd = Path.cwd()

            try:
                with (
                    mock.patch.object(runner, "REPO_ROOT", root),
                    mock.patch.object(
                        runner,
                        "selected_maps",
                        return_value=maps,
                    ),
                    mock.patch.object(
                        runner,
                        "build_rollout_plan",
                        return_value=plan,
                    ),
                    mock.patch.object(
                        runner,
                        "qwen_model_ids",
                        side_effect=AssertionError(
                            "plan-only must not contact Qwen"
                        ),
                    ) as qwen_probe,
                    mock.patch.object(
                        runner.subprocess,
                        "Popen",
                        side_effect=AssertionError(
                            "plan-only must not start external processes"
                        ),
                    ) as popen,
                    mock.patch.dict(runner.os.environ, {}, clear=False),
                ):
                    self.assertEqual(runner.main(self.argv()), 0)

                    suite_dir = root / "outputs/main_flow_suite"
                    agent_path = suite_dir / "agents_qwen3vl8b.json"
                    manifest_path = suite_dir / "experiment_manifest.json"
                    config_path = suite_dir / "suite_config.json"

                    agent_path.write_text("sentinel", encoding="utf-8")
                    manifest_before = manifest_path.read_bytes()
                    config_before = config_path.read_bytes()

                    with self.assertRaisesRegex(ValueError, "max_steps"):
                        runner.main(
                            self.argv(
                                "--resume",
                                "--max-steps",
                                "11",
                            )
                        )

                    self.assertEqual(
                        agent_path.read_text(encoding="utf-8"),
                        "sentinel",
                    )
                    self.assertEqual(
                        manifest_path.read_bytes(),
                        manifest_before,
                    )
                    self.assertEqual(
                        config_path.read_bytes(),
                        config_before,
                    )

                    # A malformed manifest must fail before the mutable
                    # runtime agent configuration is overwritten.
                    manifest_path.write_text("[]\n", encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "JSON object"):
                        runner.main(
                            self.argv(
                                "--resume",
                                "--ue-port",
                                "19123",
                            )
                        )

                    self.assertEqual(
                        agent_path.read_text(encoding="utf-8"),
                        "sentinel",
                    )
                    self.assertEqual(
                        manifest_path.read_text(encoding="utf-8"),
                        "[]\n",
                    )
                    self.assertEqual(
                        config_path.read_bytes(),
                        config_before,
                    )

                    # Restore the valid immutable manifest for the legal
                    # runtime-only resume tested below.
                    manifest_path.write_bytes(manifest_before)

                    self.assertEqual(
                        runner.main(
                            self.argv(
                                "--resume",
                                "--ue-port",
                                "19123",
                                "--ue-gpu",
                                "3",
                                "--qwen-url",
                                "http://127.0.0.1:18000/v1",
                                "--qwen-port",
                                "18000",
                                "--qwen-gpu",
                                "4",
                            )
                        ),
                        0,
                    )

                    updated_agent = json.loads(
                        agent_path.read_text(encoding="utf-8")
                    )
                    self.assertIn(
                        "18000",
                        json.dumps(updated_agent, sort_keys=True),
                    )

                    # Resume keeps the immutable manifest/config intact.
                    self.assertEqual(
                        manifest_path.read_bytes(),
                        manifest_before,
                    )
                    self.assertEqual(
                        config_path.read_bytes(),
                        config_before,
                    )
                    qwen_probe.assert_not_called()
                    popen.assert_not_called()
            finally:
                os.chdir(original_cwd)



class ProcessGroupTests(unittest.TestCase):
    def test_rollout_connects_unrealcv_at_policy_input_resolution(self):
        source = inspect.getsource(runner.execute_rollout)
        self.assertIn("args.observation_width", source)
        self.assertIn("args.observation_height", source)
        self.assertIn("resolution=(", source)

    def test_current_process_is_a_live_member_of_its_group(self):
        import os

        self.assertIn(os.getpid(), runner.ProcessGroup._live_members(os.getpgrp()))


if __name__ == "__main__":
    unittest.main()
