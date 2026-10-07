from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from evaluation.run_openai_luna_all36 import (
    all_tasks,
    build_summary,
    claim_cell,
    expected_rollouts_for_suite,
    latest_ue_generation,
    parse_args,
)
from evaluation.run_api_easy_instructional_matrix import (
    parse_args as parse_api_args,
    result_files,
    retry_delay_seconds,
    wait_for_benchmark_launch_slot,
)


class LunaCampaignRunnerTests(unittest.TestCase):
    def test_expected_rollouts_uses_persisted_subset_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            (suite / "execution_profile.json").write_text(
                json.dumps({"selected_rollouts": 75}),
                encoding="utf-8",
            )

            self.assertEqual(75, expected_rollouts_for_suite(suite))

    def test_latest_ue_generation_preserves_logs_across_worker_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = Path(temporary)
            ue_logs = worker / "ue"
            ue_logs.mkdir()
            (ue_logs / "map1_10roads_001.log").touch()
            (ue_logs / "map2_10roads_004.json").touch()
            (ue_logs / "unrelated.log").touch()

            self.assertEqual(4, latest_ue_generation(worker))

    def test_latest_ue_generation_defaults_to_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(0, latest_ue_generation(Path(temporary)))

    def test_dynamic_claim_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary) / "cell"
            with claim_cell(cell, True) as first:
                self.assertTrue(first)
                with claim_cell(cell, True) as second:
                    self.assertFalse(second)
            with claim_cell(cell, True) as third:
                self.assertTrue(third)

    def test_parallel_profile_defaults_keep_sparse_audit_images(self):
        args = parse_args([])
        self.assertTrue(args.fast_simulation)
        self.assertEqual(3, args.record_images_first_steps)
        self.assertEqual(3, args.record_images_last_steps)
        self.assertEqual(0, args.record_png_compress_level)

    def test_api_matrix_accepts_full_step_image_recording(self):
        args = parse_api_args([
            "--record-images-first-steps", "0",
            "--record-images-last-steps", "0",
            "--record-output-images",
        ])

        self.assertEqual(0, args.record_images_first_steps)
        self.assertEqual(0, args.record_images_last_steps)
        self.assertTrue(args.record_output_images)

    def test_task_plan_uses_authoritative_edges_for_route_budget(self):
        task = next(item for item in all_tasks() if item.task_id == 12)

        self.assertAlmostEqual(task.route_length_m, 159.911721, places=6)
        self.assertEqual(task.max_steps, 477)

    def test_final_summary_excludes_fresh_validation_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            cell = (
                suite
                / "cells"
                / "easy_realtime_instructional_none"
                / "map1_10roads"
                / "task_000"
            )
            attempt = cell / "attempt_001"
            attempt.mkdir(parents=True)
            result_path = attempt / "task_0.json"
            result_path.write_text("{}\n", encoding="utf-8")
            (cell / "status.json").write_text(
                json.dumps({
                    "state": "completed",
                    "result_path": str(result_path),
                }),
                encoding="utf-8",
            )

            with patch(
                "evaluation.run_openai_luna_all36.validate_attempt",
                return_value=(result_path, ["contract mismatch"]),
            ):
                summary = build_summary(
                    suite, validate_completed=True
                )

        self.assertEqual(summary["status_counts"], {"completed": 1})
        self.assertEqual(summary["completed"], 0)
        self.assertEqual(len(summary["fresh_validation_failures"]), 1)
        self.assertEqual(
            summary["fresh_validation_failures"][0]["issues"],
            ["contract mismatch"],
        )

    def test_result_files_ignore_auxiliary_alignment_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            result = attempt / "task_0.json"
            result.touch()
            (attempt / "task_0_alignment_validation.json").touch()
            self.assertEqual([result], result_files(attempt))
    def test_benchmark_launch_slot_enforces_minimum_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            (suite / ".benchmark_launch.lock").write_text("95\n", encoding="utf-8")
            with patch(
                "evaluation.run_api_easy_instructional_matrix.time.time",
                side_effect=[100.0, 100.0],
            ), patch(
                "evaluation.run_api_easy_instructional_matrix.time.sleep"
            ) as sleeper:
                delay = wait_for_benchmark_launch_slot(suite, 10.0)
            self.assertEqual(5.0, delay)
            sleeper.assert_called_once_with(5.0)
            self.assertEqual(
                "105.000000",
                (suite / ".benchmark_launch.lock").read_text(encoding="utf-8").strip(),
            )

    def test_provider_throttle_retry_honors_retry_after(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "rollout.log"
            log.write_text(
                "Error code: 402 in_flight_budget_exhausted "
                "headers: {'Retry-After': '120'}",
                encoding="utf-8",
            )
            self.assertEqual(120.0, retry_delay_seconds(log, 1))
            log.write_text("ordinary rollout failure", encoding="utf-8")
            self.assertEqual(5.0, retry_delay_seconds(log, 1))


    def test_condition_filter_accepts_known_ids(self):
        args = parse_args([
            "--condition-ids",
            "easy_realtime_instructional_none",
            "easy_static_instructional_low",
        ])
        self.assertEqual(
            [
                "easy_realtime_instructional_none",
                "easy_static_instructional_low",
            ],
            args.condition_ids,
        )

    def test_condition_filter_rejects_unknown_ids(self):
        with self.assertRaises(SystemExit):
            parse_args(["--condition-ids", "not_a_condition"])


if __name__ == "__main__":
    unittest.main()
