import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "evaluation" / "run_codex_cli_sol_matrix.py"


class CodexCLIMatrixTests(unittest.TestCase):
    def _run(self, mode: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--mode",
                mode,
                "--task-ids",
                "0",
                "--workers",
                "1",
                "--gpus",
                "7",
                "--ports",
                "9048",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_default_plan_is_easy_static_and_realtime_low_high(self):
        completed = self._run("plan")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        plan = json.loads(completed.stdout)
        self.assertFalse(plan["starts_rollouts"])
        self.assertEqual(plan["difficulties"], ["easy"])
        self.assertEqual(plan["environment_modes"], ["realtime", "static"])
        self.assertEqual(plan["rollout_count"], 4)
        self.assertEqual(plan["models"], [{
            "model": "gpt-5.6-sol",
            "reasoning_efforts": ["low", "high"],
            "billing_mode": "chatgpt_codex_credits",
            "standard_short_context_price_usd_per_million": None,
        }])
        self.assertEqual(
            plan["shared_protocol"]["inference_surface"],
            "chatgpt_codex_cli",
        )
        self.assertNotIn(
            "--key-file", plan["per_model_commands"][0]["command"]
        )

    def test_run_requires_explicit_codex_credit_confirmation(self):
        completed = self._run("run")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--confirm-codex-cli-run", completed.stderr)


if __name__ == "__main__":
    unittest.main()
