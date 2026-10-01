from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from evaluation.run_openai_benchmark import (
    LOCAL_DEPS,
    build_cost_estimate,
    build_llm_config,
    cleanup_runtime,
    load_api_key,
    main,
    parse_args,
    resolve_api_key,
    resolve_max_steps,
)
from llm.prompt import get_system_prompt_openai_action_only


class OpenAIBenchmarkRunnerTest(unittest.TestCase):
    def test_optional_bundled_unrealcv_dependencies_are_bootstrapped(self):
        import sys

        if LOCAL_DEPS.is_dir():
            self.assertIn(str(LOCAL_DEPS), sys.path)
        else:
            self.assertNotIn(str(LOCAL_DEPS), sys.path)

    def test_load_api_key_accepts_raw_and_env_file_formats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "key.txt"
            path.write_text("sk-test-raw\n", encoding="utf-8")
            self.assertEqual(load_api_key(path), "sk-test-raw")
            path.write_text('OPENAI_API_KEY="sk-test-env"\n', encoding="utf-8")
            self.assertEqual(load_api_key(path), "sk-test-env")

    def test_dry_run_config_contains_no_credential(self):
        args = parse_args(["--model", "model-under-test", "--dry-run"])
        config = build_llm_config(args)
        self.assertEqual(config["llm"][0]["api_mode"], "responses")
        self.assertEqual(config["llm"][0]["reasoning_effort"], "none")
        self.assertIn("Return exactly these two lines", config["llm"][0]["prompt_suffix"])
        self.assertIn("Do not output reasoning", config["llm"][0]["prompt_suffix"])
        self.assertNotIn("api_key", config["llm"][0])
        self.assertTrue(config["llm"][0]["raise_on_api_error"])

    def test_instructional_prompt_does_not_receive_action_only_suffix(self):
        args = parse_args(["--prompt-style", "instructional", "--dry-run"])
        config = build_llm_config(args)

        self.assertIsNone(config["llm"][0]["prompt_suffix"])

    def test_default_reasoning_effort_is_preserved_as_omission_sentinel(self):
        args = parse_args(["--reasoning-effort", "default", "--dry-run"])
        config = build_llm_config(args)["llm"][0]

        self.assertTrue(config["reasoning"])
        self.assertEqual(config["reasoning_effort"], "default")

    def test_cost_safe_defaults_select_luna_standard_and_small_output_cap(self):
        args = parse_args(["--dry-run"])
        self.assertEqual(args.model, "gpt-5.6-luna")
        self.assertEqual(args.reasoning_effort, "none")
        self.assertEqual(args.prompt_style, "openai_action_only")
        self.assertEqual(args.max_output_tokens, 128)
        self.assertEqual(args.estimated_output_tokens_per_step, 32)
        self.assertEqual(args.service_tier, "default")

    def test_openai_static_prompt_is_action_only_and_describes_paused_inference(self):
        prompt = get_system_prompt_openai_action_only(realtime_thinking=False)

        self.assertIn("The simulator is paused while you decide", prompt)
        self.assertIn("reasoning duration does not", prompt)
        self.assertIn("Return exactly these two lines", prompt)
        self.assertNotIn("Reasoning: [ONE sentence", prompt)
        self.assertNotIn("environment continues to evolve during both your reasoning", prompt)

    def test_environment_api_key_takes_precedence(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-env"}, clear=False):
            self.assertEqual(
                resolve_api_key(Path("/definitely/not/present.txt")),
                "sk-test-env",
            )

    def test_known_model_cost_estimate(self):
        args = parse_args(["--model", "gpt-5.6", "--estimate-only", "--max-steps", "60"])
        estimate = build_cost_estimate(args)
        self.assertEqual(estimate["estimated_input_tokens"], 120_000)
        self.assertEqual(estimate["estimated_output_tokens"], 1_920)
        self.assertAlmostEqual(estimate["estimated_cost_usd"], 0.5184)
        self.assertAlmostEqual(estimate["output_cap_scenario_cost_usd"], 0.6336)

    def test_luna_cost_uses_current_model_page_price(self):
        args = parse_args(
            ["--model", "gpt-5.6-luna", "--estimate-only", "--max-steps", "60"]
        )
        estimate = build_cost_estimate(args)

        self.assertEqual(estimate["pricing_as_of"], "2026-09-01")
        self.assertEqual(estimate["input_price_usd_per_million_tokens"], 0.2)
        self.assertEqual(estimate["output_price_usd_per_million_tokens"], 1.2)
        self.assertAlmostEqual(estimate["estimated_cost_usd"], 0.026304)

    def test_omitted_output_cap_is_not_sent_or_estimated(self):
        args = parse_args(
            ["--model", "gpt-5.6-luna", "--max-output-tokens", "none", "--max-steps", "60"]
        )
        config = build_llm_config(args)
        estimate = build_cost_estimate(args)
        self.assertIsNone(config["llm"][0]["max_tokens"])
        self.assertIsNone(estimate["configured_output_cap_tokens"])
        self.assertIsNone(estimate["output_cap_scenario_cost_usd"])

    def test_sparse_input_only_artifacts_are_the_openai_defaults(self):
        args = parse_args(["--dry-run"])

        self.assertEqual(0, args.record_png_compress_level)
        self.assertEqual(3, args.record_images_first_steps)
        self.assertEqual(3, args.record_images_last_steps)
        self.assertFalse(args.record_output_images)
        self.assertFalse(args.record_demo_images)

    def test_token_based_mode_is_selectable(self):
        args = parse_args(["--model", "gpt-5.6-luna", "--token-based"])
        self.assertTrue(args.token_based)

    def test_traffic_controls_default_on_and_remain_independently_selectable(self):
        defaults = parse_args(["--model", "gpt-5.6-luna"])
        self.assertEqual("visual_only", defaults.traffic_policy)
        self.assertTrue(defaults.red_light_conflict_vehicle)
        self.assertTrue(defaults.static_signal_vehicles)
        self.assertIsNone(defaults.red_light_conflict_vehicle_probability)

        disabled = parse_args(
            [
                "--model",
                "gpt-5.6-luna",
                "--traffic-policy",
                "safety_assisted",
                "--no-red-light-conflict-vehicle",
                "--no-static-signal-vehicles",
            ]
        )
        self.assertEqual("safety_assisted", disabled.traffic_policy)
        self.assertFalse(disabled.red_light_conflict_vehicle)
        self.assertFalse(disabled.static_signal_vehicles)

    def test_cleanup_does_not_mask_a_dead_ue_connection(self):
        class DeadManager:
            def cleanup(self):
                raise ConnectionError("socket is closed")

        class UnrealCV:
            disconnected = False

            def disconnect(self):
                self.disconnected = True

        communicator = type("Communicator", (), {"unrealcv": UnrealCV()})()
        stderr = StringIO()

        with redirect_stderr(stderr):
            cleanup_runtime(communicator, DeadManager(), skip_cleanup=False)

        self.assertTrue(communicator.unrealcv.disconnected)
        self.assertIn("cleanup was incomplete", stderr.getvalue())

    def test_unknown_model_accepts_explicit_prices(self):
        args = parse_args(
            [
                "--model",
                "future-model",
                "--input-price-per-million",
                "2",
                "--output-price-per-million",
                "8",
                "--max-steps",
                "60",
            ]
        )
        estimate = build_cost_estimate(args)
        self.assertAlmostEqual(estimate["estimated_cost_usd"], 0.25536)

    def test_automatic_max_steps_is_three_times_route_length_in_meters(self):
        args = parse_args(["--model", "gpt-5.6-luna", "--task-index", "1"])
        self.assertEqual(resolve_max_steps(args), 180)

    def test_automatic_max_steps_uses_authoritative_task_edges(self):
        args = parse_args(
            [
                "--model",
                "gpt-5.6-luna",
                "--task-file",
                "data/map2_12roads/tasks.json",
                "--task-index",
                "4",
            ]
        )

        # The generated route_info path for task 12 is stale and only yields
        # 120 steps. Its authoritative ordered-edge route is 159.91 m.
        self.assertEqual(resolve_max_steps(args), 477)

    def test_explicit_max_steps_overrides_route_calculation(self):
        args = parse_args(
            ["--model", "gpt-5.6-luna", "--task-index", "1", "--max-steps", "37"]
        )
        self.assertEqual(resolve_max_steps(args), 37)

    def test_estimate_only_does_not_require_key_file(self):
        output = StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "--model",
                    "gpt-5.6-luna",
                    "--estimate-only",
                    "--key-file",
                    "/definitely/not/present.txt",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("no API call made", output.getvalue())
        self.assertIn("Estimated API cost:", output.getvalue())


    def test_openrouter_claude_dry_run_uses_secret_free_chat_config(self):
        args = parse_args(
            [
                "--provider", "openrouter",
                "--model", "anthropic/claude-haiku-4.5",
                "--api-mode", "chat_completions",
                "--reasoning-effort", "none",
                "--dry-run",
            ]
        )
        config = build_llm_config(args)["llm"][0]

        self.assertEqual(config["provider"], "openrouter")
        self.assertEqual(config["url"], "https://openrouter.ai/api/v1")
        self.assertEqual(config["api_mode"], "chat_completions")
        self.assertFalse(config["reasoning"])
        self.assertNotIn("api_key", config)
        self.assertEqual(
            args.key_file,
            Path(__file__).resolve().parents[1].parent / "openrouter_api.txt",
        )

    def test_openrouter_key_file_format_is_supported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "key.txt"
            path.write_text("OPENROUTER_API_KEY=sk-or-test\n", encoding="utf-8")
            self.assertEqual(
                load_api_key(path, "OPENROUTER_API_KEY"), "sk-or-test"
            )

    def test_claude_low_requires_room_for_minimum_thinking_budget(self):
        base = [
            "--provider", "openrouter",
            "--model", "anthropic/claude-haiku-4.5",
            "--api-mode", "chat_completions",
            "--reasoning-effort", "low",
        ]
        with self.assertRaises(SystemExit):
            parse_args(base + ["--max-output-tokens", "128"])

        args = parse_args(base + ["--max-output-tokens", "1280"])
        self.assertEqual(args.max_output_tokens, 1280)

    def test_openrouter_models_reject_silently_mapped_reasoning_efforts(self):
        base = [
            "--provider", "openrouter",
            "--model", "google/gemini-3.8-flash",
            "--api-mode", "chat_completions",
        ]
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(base + ["--reasoning-effort", "none"])
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(base + ["--reasoning-effort", "xhigh"])

        args = parse_args(base + ["--reasoning-effort", "medium"])
        self.assertEqual(args.reasoning_effort, "medium")

    def test_optional_reasoning_model_accepts_none(self):
        args = parse_args(
            [
                "--provider", "openrouter",
                "--model", "thinkingmachines/inkling",
                "--api-mode", "chat_completions",
                "--reasoning-effort", "none",
            ]
        )
        self.assertEqual(args.reasoning_effort, "none")

    def test_new_claude_adaptive_effort_does_not_require_legacy_budget_floor(self):
        args = parse_args(
            [
                "--provider", "openrouter",
                "--model", "anthropic/claude-sonnet-5",
                "--api-mode", "chat_completions",
                "--reasoning-effort", "low",
                "--max-output-tokens", "128",
            ]
        )
        self.assertEqual(args.max_output_tokens, 128)

    def test_claude_catalog_cost_is_available_offline(self):
        args = parse_args(
            [
                "--provider", "openrouter",
                "--model", "anthropic/claude-haiku-4.5",
                "--api-mode", "chat_completions",
                "--max-steps", "60",
            ]
        )
        estimate = build_cost_estimate(args)

        self.assertEqual(estimate["input_price_usd_per_million_tokens"], 1.0)
        self.assertEqual(estimate["output_price_usd_per_million_tokens"], 5.0)
        self.assertAlmostEqual(estimate["estimated_cost_usd"], 0.1296)

if __name__ == "__main__":
    unittest.main()
