import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "open_source_experiments", ROOT / "benchmark" / "experiments.py"
)
experiments = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiments)


class OpenSourceExperimentTests(unittest.TestCase):
    def args(self):
        return experiments.parse_args(
            [
                "plan",
                "--program-name",
                "unit",
                "--output-root",
                "/tmp/unit-results",
            ]
        )

    def test_default_program_has_three_controlled_studies(self):
        conditions = experiments.build_conditions()
        by_study = {
            study: [item for item in conditions if item["study"] == study]
            for study in experiments.STUDIES
        }

        self.assertEqual(len(by_study["size"]), 6)
        self.assertEqual(len(by_study["reasoning"]), 4)
        self.assertEqual(len(by_study["timing"]), 1)
        self.assertEqual(
            sum(item["rollout_count"] for item in conditions),
            432,
        )

    def test_size_study_changes_only_instruct_checkpoint(self):
        conditions = experiments.build_conditions(["size"])

        self.assertEqual(
            {tuple(item["env_modes"]) for item in conditions},
            {("realtime",)},
        )
        self.assertEqual(
            {item["enable_thinking"] for item in conditions},
            {False},
        )
        self.assertEqual(
            {item["thinking_budget_tokens"] for item in conditions},
            {None},
        )

    def test_reasoning_study_uses_true_hard_budgets(self):
        conditions = experiments.build_conditions(["reasoning"])
        baseline = conditions[0]
        thinking = conditions[1:]

        self.assertEqual(baseline["model"], experiments.INSTRUCT_8B)
        self.assertFalse(baseline["enable_thinking"])
        self.assertEqual(
            [item["thinking_budget_tokens"] for item in thinking],
            [64, 128, 256],
        )
        self.assertEqual(
            {item["model"] for item in thinking},
            {experiments.THINKING_8B},
        )

        values = experiments.runner_args_for(
            thinking[0],
            self.args(),
            "/models/thinking",
        )
        self.assertIn("--enable-thinking", values)
        self.assertEqual(
            values[values.index("--reasoning-budget-tokens") + 1],
            "64",
        )
        launch = values[values.index("--qwen-launch-command") + 1]
        self.assertIn("qwen3vl_thinking_openai_server.py", launch)
        self.assertIn("--require-cuda", launch)

    def test_timing_study_keeps_one_model_and_both_modes(self):
        condition = experiments.build_conditions(["timing"])[0]

        self.assertEqual(condition["model"], experiments.INSTRUCT_8B)
        self.assertFalse(condition["enable_thinking"])
        self.assertEqual(condition["env_modes"], ["static", "realtime"])
        self.assertEqual(condition["rollout_count"], 72)

    def test_run_requires_real_checkpoint_directories(self):
        condition = experiments.build_conditions(["reasoning"])[0]
        with self.assertRaisesRegex(ValueError, "Missing --model-path"):
            experiments.model_path_for(
                condition["model"],
                {},
                require_existing=True,
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            self.assertEqual(
                experiments.model_path_for(
                    condition["model"],
                    {condition["model"]: path},
                    require_existing=True,
                ),
                str(path.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
