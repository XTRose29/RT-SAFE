from __future__ import annotations

import unittest
from pathlib import Path

from evaluation import run_openai_sol_task0 as pilot


class SolTask0PilotTests(unittest.TestCase):
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

    def test_matrix_is_task0_easy_realtime_low_high(self):
        tasks = pilot.all_tasks()
        conditions = pilot.all_conditions()

        self.assertEqual([0], [task.task_id for task in tasks])
        self.assertEqual({"easy"}, {item.difficulty for item in conditions})
        self.assertEqual({"realtime"}, {item.env_mode for item in conditions})
        self.assertEqual(["low", "high"], [item.effort for item in conditions])

    def test_manifest_and_command_retain_every_step_image(self):
        manifest = pilot.planned_manifest()
        self.assertEqual("gpt-5.6-sol", manifest["model"])
        self.assertEqual(2, manifest["rollout_count"])
        self.assertTrue(manifest["artifact_policy"]["retain_all_input_images"])
        self.assertTrue(
            manifest["artifact_policy"]["retain_all_post_action_images"]
        )

        args = pilot.campaign.parse_args(["--suite-dir", "/tmp/sol-task0-test"])
        command = pilot.runner_command(
            args,
            pilot.all_tasks()[0],
            pilot.all_conditions()[0],
            Path("/tmp/sol-task0-test/attempt"),
            9123,
        )
        self.assertIn("gpt-5.6-sol", command)
        self.assertIn("--record-output-images", command)
        self.assertEqual(
            "0", command[command.index("--record-images-first-steps") + 1]
        )
        self.assertEqual(
            "0", command[command.index("--record-images-last-steps") + 1]
        )


if __name__ == "__main__":
    unittest.main()
