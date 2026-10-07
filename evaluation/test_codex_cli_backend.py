from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation import run_openai_benchmark
from PIL import Image

from llm.codex_cli_backend import CodexCLIBackend, _serializable_usage


class CodexCLIBackendTests(unittest.TestCase):
    def test_serializable_usage_uses_final_completed_turn(self):
        usage = _serializable_usage(
            [
                {"type": "turn.completed", "usage": {"input_tokens": 2}},
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 40,
                        "cache_write_input_tokens": 3,
                        "output_tokens": 12,
                        "reasoning_output_tokens": 7,
                    },
                },
            ]
        )
        self.assertEqual(
            usage,
            {
                "prompt_tokens": 100,
                "completion_tokens": 12,
                "total_tokens": 112,
                "reasoning_tokens": 7,
                "cached_tokens": 40,
                "cache_write_tokens": 3,
                "cost_usd": None,
            },
        )

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "must-not-leak", "OPENROUTER_API_KEY": "must-not-leak"},
    )
    def test_generate_uses_ephemeral_chatgpt_auth_without_api_keys(self):
        backend = CodexCLIBackend(
            model="gpt-5.6-sol",
            reasoning_effort="low",
            timeout=30,
            executable="/usr/bin/true",
        )

        def fake_run(command, **kwargs):
            final_path = command[command.index("-o") + 1]
            with open(final_path, "w", encoding="utf-8") as stream:
                stream.write("Action: move_to\nParam: 1\n")
            self.assertIn("--ephemeral", command)
            self.assertIn("--ignore-user-config", command)
            self.assertIn("--sandbox", command)
            self.assertIn("gpt-5.6-sol", command)
            self.assertTrue(any("model_reasoning_effort" in value for value in command))
            self.assertIn("-i", command)
            self.assertIsNone(kwargs["env"].get("OPENAI_API_KEY"))
            self.assertIsNone(kwargs["env"].get("OPENROUTER_API_KEY"))
            self.assertIn(
                "<benchmark_system>\nsystem\n</benchmark_system>", kwargs["input"]
            )
            return SimpleNamespace(
                returncode=0,
                stdout="\n".join(
                    [
                        json.dumps(
                            {
                                "type": "item.completed",
                                "item": {
                                    "type": "agent_message",
                                    "text": "Action: move_to\nParam: 1",
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "turn.completed",
                                "usage": {
                                    "input_tokens": 200,
                                    "cached_input_tokens": 50,
                                    "output_tokens": 10,
                                    "reasoning_output_tokens": 4,
                                },
                            }
                        ),
                    ]
                ),
                stderr="",
            )

        with patch("llm.codex_cli_backend.subprocess.run", side_effect=fake_run):
            result = backend.generate(
                system_prompt="system",
                user_prompt="user",
                images=[Image.new("RGB", (8, 8), "red")],
            )
        self.assertEqual(result.text, "Action: move_to\nParam: 1")
        self.assertEqual(result.usage["total_tokens"], 210)

    def test_single_task_runner_accepts_explicit_codex_cli_mode(self):
        args = run_openai_benchmark.parse_args(
            [
                "--provider",
                "codex-cli",
                "--api-mode",
                "codex_cli",
                "--model",
                "gpt-5.6-sol",
                "--reasoning-effort",
                "low",
                "--max-output-tokens",
                "none",
                "--dry-run",
            ]
        )
        self.assertEqual(args.provider, "codex-cli")
        self.assertEqual(args.api_mode, "codex_cli")
        self.assertIsNone(args.key_file)
        self.assertIsNone(args.max_output_tokens)


if __name__ == "__main__":
    unittest.main()
