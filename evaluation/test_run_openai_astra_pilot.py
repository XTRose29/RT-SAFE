from __future__ import annotations

import unittest
from pathlib import Path

from evaluation import run_openai_astra_pilot as pilot
from evaluation.run_openai_benchmark import (
    _catalog_pricing,
    parse_args as parse_rollout_args,
)


class AstraPilotTests(unittest.TestCase):
    def setUp(self) -> None:
        names = (
            "MODEL",
            "INPUT_PRICE",
            "OUTPUT_PRICE",
            "OBSERVATION_WIDTH",
            "OBSERVATION_HEIGHT",
            "UE_WIDTH",
            "UE_HEIGHT",
            "all_conditions",
            "all_tasks",
            "planned_manifest",
            "runner_command",
            "__file__",
        )
        original = {name: getattr(pilot.campaign, name) for name in names}

        def restore_campaign() -> None:
            for name, value in original.items():
                setattr(pilot.campaign, name, value)

        self.addCleanup(restore_campaign)
        pilot.configure_campaign()

    def test_matrix_is_five_maps_by_four_settings_by_two_efforts(self):
        tasks = pilot.all_tasks()
        conditions = pilot.all_conditions()

        self.assertEqual([0, 8, 16, 24, 32], [task.task_id for task in tasks])
        self.assertEqual(5, len({task.map_name for task in tasks}))
        self.assertEqual(8, len(conditions))
        self.assertEqual({"low", "high"}, {item.effort for item in conditions})
        self.assertEqual(40, len(tasks) * len(conditions))

    def test_manifest_records_full_step_images(self):
        manifest = pilot.planned_manifest()

        self.assertEqual("gpt-6-astra", manifest["model"])
        self.assertEqual(40, manifest["rollout_count"])
        self.assertTrue(manifest["artifact_policy"]["retain_all_input_images"])
        self.assertTrue(manifest["artifact_policy"]["retain_all_post_action_images"])
        self.assertEqual([1080, 960], manifest["observation_resolution"])
        self.assertEqual([1920, 1080], manifest["ue_resolution"])

    def test_only_resolution_changes_in_inherited_ue_configuration(self):
        args = pilot.campaign.parse_args(["--suite-dir", "/tmp/astra-test"])
        ue_args = pilot.campaign.make_ue_args(args, gpu=7, port=9123)

        self.assertEqual((1920, 1080), (ue_args.ue_width, ue_args.ue_height))
        self.assertEqual(30, ue_args.ue_fps)

    def test_rollout_command_forwards_full_image_retention(self):
        args = pilot.campaign.parse_args(
            [
                "--suite-dir",
                "/tmp/astra-test",
                "--record-images-first-steps",
                "0",
                "--record-images-last-steps",
                "0",
            ]
        )
        command = pilot.runner_command(
            args,
            pilot.all_tasks()[0],
            pilot.all_conditions()[0],
            Path("/tmp/astra-test/attempt"),
            9123,
        )

        self.assertIn("gpt-6-astra", command)
        self.assertIn("--record-output-images", command)
        self.assertEqual(
            "0", command[command.index("--record-images-first-steps") + 1]
        )
        self.assertEqual(
            "0", command[command.index("--record-images-last-steps") + 1]
        )

    def test_pricing_catalog_includes_astra_and_snapshots(self):
        self.assertEqual(10.0, _catalog_pricing("gpt-6-astra")["input"])
        self.assertEqual(50.0, _catalog_pricing("gpt-6-astra-2026-09-07")["output"])

    def test_astra_rejects_unsupported_none_effort(self):
        with self.assertRaises(SystemExit):
            parse_rollout_args(
                ["--model", "gpt-6-astra", "--reasoning-effort", "none"]
            )


if __name__ == "__main__":
    unittest.main()
