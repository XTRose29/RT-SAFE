"""Tests for the Claude Code subscription backend using a fake ``claude`` executable."""

from __future__ import annotations

import base64
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import run_openai_benchmark  # noqa: F401  (installs the simworld stub)
import numpy as np
from PIL import Image

from evaluation import run_claude_code_campaign as campaign
from llm import claude_code_backend as backend_module
from llm.claude_code_backend import (
    CLI_IMAGE_BYTE_LIMIT,
    ClaudeCodeAuthError,
    ClaudeCodeBackend,
    ClaudeCodeCampaignStopped,
    ClaudeCodeEffortMismatch,
    ClaudeCodeInputError,
    ClaudeCodeModelMismatch,
    ClaudeCodeProtocolError,
    ClaudeCodeTimeout,
    ClaudeCodeUsageLimit,
    pid_alive,
    validate_managed_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTH = {
    "loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
    "email": "user@example.com", "orgId": "org-1", "subscriptionType": "max",
}
FAKE_CLAUDE = r'''#!{python}
import json, os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
config = json.loads((HERE / "config.json").read_text())
args = sys.argv[1:]
def record(entry):
    with open(HERE / "calls.jsonl", "a") as stream:
        stream.write(json.dumps(entry) + "\n")
if args == ["--version"]:
    record({"argv": args, "env": dict(os.environ)})
    print(config.get("version", "2.1.270 (Claude Code)"))
    sys.exit(0)
if args[:2] == ["auth", "status"]:
    record({"argv": args, "env": dict(os.environ)})
    print(json.dumps(config["auth"]))
    sys.exit(0)
def flag(name):
    return args[args.index(name) + 1] if name in args else None
model, effort = flag("--model"), flag("--effort")
messages = [json.loads(line) for line in sys.stdin if line.strip()]
prompt_path = flag("--system-prompt-file")
record({"argv": args, "env": dict(os.environ), "cwd": os.getcwd(), "stdin": messages,
        "system_prompt": Path(prompt_path).read_text() if prompt_path else None})
if config.get("mode") == "hang":
    child = subprocess.Popen(["sleep", "300"])
    (HERE / "pids.json").write_text(json.dumps({"cli": os.getpid(), "grandchild": child.pid}))
    time.sleep(300)
def emit(event):
    print(json.dumps(event), flush=True)
emit({"type": "system", "subtype": "init", "model": config.get("init_model", model),
      "apiKeySource": config.get("api_key_source", "none"),
      "claude_code_version": config.get("init_version", "2.1.270"),
      "tools": config.get("tools", []), "mcp_servers": [], "permissionMode": "dontAsk"})
for message in messages:
    if message.get("type") == "control_request":
        emit({"type": "control_response", "response": {
            "subtype": "success", "request_id": message["request_id"],
            "response": {"effective": {}, "sources": [], "applied": {
                "model": config.get("applied_model", model),
                "effort": config.get("applied_effort", effort)}}}})
for event in config.get("extra_events", []):
    emit(event)
emit({"type": "rate_limit_event", "rate_limit_info": config.get(
    "rate_limit", {"status": "allowed", "resetsAt": 1789348622, "isUsingOverage": False})})
text = config.get("text", "Reasoning: The crossing is clear.\nAction: move_to\nParam: 4")
emit({"type": "assistant", "message": {"model": config.get("served_model", model),
      "content": [{"type": "text", "text": text}], "stop_reason": None}})
emit({"type": "result", "subtype": config.get("result_subtype", "success"),
      "is_error": config.get("is_error", False), "result": text, "num_turns": 1,
      "permission_denials": [], "session_id": "fake-session", "duration_ms": 1300,
      "duration_api_ms": 1000, "ttft_ms": 400, "stop_reason": "end_turn",
      "total_cost_usd": 0.0123, "api_error_status": config.get("api_error_status"),
      "usage": {"input_tokens": 1000, "cache_read_input_tokens": 3000,
                "cache_creation_input_tokens": 200, "output_tokens": 150,
                "output_tokens_details": {"thinking_tokens": 120}},
      "modelUsage": config.get("model_usage", {model: {
          "inputTokens": 1000, "outputTokens": 150, "thinkingTokens": 120,
          "canonicalModel": config.get("canonical", model)}})})
sys.exit(config.get("exit_code", 0))
'''


class FakeClaudeMixin:
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="fake_claude_")
        self.root = Path(self._temporary.name)
        self.executable = self.root / "claude"
        self.executable.write_text(FAKE_CLAUDE.replace("{python}", sys.executable))
        self.executable.chmod(self.executable.stat().st_mode | stat.S_IXUSR)
        self.stop_file = self.root / "stop.json"
        self.configure()

    def tearDown(self):
        self._temporary.cleanup()

    def configure(self, **overrides):
        config = {"auth": DEFAULT_AUTH} | overrides
        (self.root / "config.json").write_text(json.dumps(config))

    def calls(self):
        path = self.root / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def policy_calls(self):
        return [call for call in self.calls() if "-p" in call["argv"]]

    def make(self, model="claude-sonnet-5", effort="low", timeout=30.0):
        return ClaudeCodeBackend(
            model=model, reasoning_effort=effort, timeout=timeout,
            executable=str(self.executable), stop_file=str(self.stop_file),
            managed_settings_paths=(),
        )

    @staticmethod
    def frame(color, size=(720, 640)):
        return Image.new("RGB", size, color)


class ClaudeCodeBackendTests(FakeClaudeMixin, unittest.TestCase):
    def test_command_construction(self):
        backend = self.make(model="claude-fable-5-1", effort="medium")
        command = backend.build_command(Path("/tmp/system_prompt.txt"))
        self.assertEqual(command[0], str(self.executable))
        self.assertEqual(command[command.index("--model") + 1], "claude-fable-5-1")
        self.assertEqual(command[command.index("--effort") + 1], "medium")
        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertEqual(command[command.index("--setting-sources") + 1], "")
        self.assertEqual(command[command.index("--input-format") + 1], "stream-json")
        self.assertEqual(command[command.index("--output-format") + 1], "stream-json")
        self.assertEqual(command[command.index("--permission-mode") + 1], "dontAsk")
        self.assertEqual(command[command.index("--system-prompt-file") + 1], "/tmp/system_prompt.txt")
        # Without a session name Claude Code sends a hidden title request to the same model.
        self.assertEqual(command[command.index("--name") + 1], "simworld-policy")
        for required in ("-p", "--safe-mode", "--no-session-persistence", "--strict-mcp-config",
                         "--disable-slash-commands", "--verbose"):
            self.assertIn(required, command)
        for forbidden in ("--bare", "--fallback-model", "--max-budget-usd", "--dangerously-skip-permissions"):
            self.assertNotIn(forbidden, command)

    def test_effort_forwarding_low_medium_high(self):
        for effort in ("low", "medium", "high"):
            with self.subTest(effort=effort):
                result = self.make(effort=effort).generate(
                    system_prompt="SYSTEM", user_prompt="USER", images=[self.frame("red")])
                call = self.policy_calls()[-1]
                self.assertEqual(call["argv"][call["argv"].index("--effort") + 1], effort)
                self.assertEqual(call["stdin"][0]["request"]["subtype"], "get_settings")
                self.assertEqual(call["system_prompt"], "SYSTEM")
                metadata = result.usage["claude_code"]
                self.assertEqual(metadata["requested_effort"], effort)
                self.assertEqual(metadata["effective_effort"], effort)

    def test_effort_not_applied_fails_closed_and_stops_campaign(self):
        self.configure(applied_effort="max")
        backend = self.make(effort="low")
        with self.assertRaises(ClaudeCodeEffortMismatch):
            backend.generate(system_prompt="s", user_prompt="u", images=[self.frame("red")])
        self.assertEqual(json.loads(self.stop_file.read_text())["reason"], "ClaudeCodeEffortMismatch")
        before = len(self.policy_calls())
        with self.assertRaises(ClaudeCodeCampaignStopped):
            backend.generate(system_prompt="s", user_prompt="u", images=[self.frame("red")])
        self.assertEqual(len(self.policy_calls()), before, "no process may start after a stop")

    def test_model_identity_and_fallback_rejection(self):
        cases = {
            "served_older_sonnet": {"served_model": "claude-sonnet-4-6"},
            "cli_resolved_other_model": {"init_model": "claude-opus-5"},
            "applied_other_model": {"applied_model": "claude-haiku-4-5"},
            "canonicalized_as_other": {"canonical": "claude-sonnet-4-6"},
            "auxiliary_model_billed": {"model_usage": {
                "claude-sonnet-5": {"canonicalModel": "claude-sonnet-5"},
                "claude-haiku-4-5": {"canonicalModel": "claude-haiku-4-5"}}},
            "substitution_event": {"extra_events": [{
                "type": "system", "subtype": "model_consent_fallback",
                "content": "Switched to Opus 5"}]},
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                self.stop_file.unlink(missing_ok=True)
                self.configure(**overrides)
                with self.assertRaises(ClaudeCodeModelMismatch):
                    self.make().generate(system_prompt="s", user_prompt="u", images=[self.frame("red")])
                self.assertTrue(self.stop_file.exists())

    def test_dated_snapshot_of_same_model_is_accepted(self):
        self.configure(served_model="claude-sonnet-5-20260801")
        result = self.make().generate(system_prompt="s", user_prompt="u", images=[self.frame("red")])
        self.assertEqual(result.usage["claude_code"]["served_models"], ["claude-sonnet-5-20260801"])

    def test_aliases_and_unsupported_models_are_rejected(self):
        for model in ("sonnet", "fable", "claude-sonnet-5[1m]", "claude-opus-5", "claude-fable-5"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                self.make(model=model)
        for effort in (None, "none", "default"):
            with self.subTest(effort=effort), self.assertRaises(ValueError):
                self.make(effort=effort)

    def test_cli_too_old_for_fable_5_1(self):
        self.configure(version="2.1.237 (Claude Code)")
        with self.assertRaises(ClaudeCodeModelMismatch):
            self.make(model="claude-fable-5-1")
        self.make(model="claude-sonnet-5")  # Sonnet 5 needs only 2.1.197

    def test_ordered_multi_image_input_is_lossless_and_under_cli_limit(self):
        frames = [self.frame(color) for color in ("red", "green", "blue")]
        rng = np.random.default_rng(0)
        # Smooth gradient: realistic compressibility, still exercises lossless WebP.
        gradient = np.tile(np.linspace(0, 255, 720, dtype=np.uint8), (640, 1))
        frames.append(Image.fromarray(np.stack([gradient, gradient[::-1], gradient.T[:640, :720] if False else gradient], -1)))
        noise = Image.fromarray(rng.integers(0, 256, (640, 720, 3), dtype=np.uint8))
        result = self.make().generate(system_prompt="s", user_prompt="USER TEXT", images=[*frames, noise])
        content = self.policy_calls()[-1]["stdin"][1]["message"]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "USER TEXT"})
        images = content[1:]
        self.assertEqual(len(images), 5)
        manifest = result.usage["claude_code"]["images"]
        self.assertEqual([item["position"] for item in manifest], [0, 1, 2, 3, 4])
        for index, (block, original) in enumerate(zip(images[:4], frames)):
            raw = base64.b64decode(block["source"]["data"])
            self.assertLessEqual(len(raw), CLI_IMAGE_BYTE_LIMIT)
            self.assertEqual(block["source"]["media_type"], "image/webp")
            decoded = Image.open(io.BytesIO(raw)).convert("RGB")
            self.assertEqual(decoded.tobytes(), original.tobytes(), f"frame {index} not lossless")
            self.assertTrue(manifest[index]["pixel_identical_to_api_arms"])
        # Incompressible noise exceeds the limit losslessly: bounded lossy fallback, flagged.
        raw = base64.b64decode(images[4]["source"]["data"])
        self.assertLessEqual(len(raw), CLI_IMAGE_BYTE_LIMIT)
        self.assertFalse(manifest[4]["pixel_identical_to_api_arms"])
        self.assertTrue(manifest[4]["encoding"].startswith("webp_lossy_q"))

    def test_numpy_frames_keep_api_arm_jpeg_bytes(self):
        from utils.img2base64 import np_to_base64

        array = np.full((64, 48, 3), 200, dtype=np.uint8)
        self.make().generate(system_prompt="s", user_prompt="u", images=[array])
        block = self.policy_calls()[-1]["stdin"][1]["message"]["content"][1]
        self.assertEqual(block["source"], {"type": "base64", "media_type": "image/jpeg", "data": np_to_base64(array)})

    def test_missing_images_fail_closed_before_any_process(self):
        backend = self.make()
        for images in ([], [self.frame("red"), None]):
            with self.subTest(count=len(images)), self.assertRaises(ClaudeCodeInputError):
                backend.generate(system_prompt="s", user_prompt="u", images=images)
        self.assertEqual(self.policy_calls(), [])

    def test_json_and_token_usage_parsing(self):
        self.configure(extra_events=[{"type": "system", "subtype": "api_retry", "attempt": 1,
                                      "max_retries": 10, "retry_delay_ms": 500, "error_status": 529}])
        result = self.make().generate(system_prompt="s", user_prompt="u", images=[self.frame("red")])
        usage = result.usage
        self.assertEqual(result.text, "Reasoning: The crossing is clear.\nAction: move_to\nParam: 4")
        self.assertEqual(usage["prompt_tokens"], 4200)
        self.assertEqual(usage["completion_tokens"], 150)
        self.assertEqual(usage["total_tokens"], 4350)
        self.assertEqual(usage["reasoning_tokens"], 120)
        self.assertEqual(usage["cached_tokens"], 3000)
        self.assertEqual(usage["cache_write_tokens"], 200)
        self.assertIsNone(usage["cost_usd"])
        metadata = usage["claude_code"]
        self.assertEqual(metadata["api_list_price_equivalent_usd"], 0.0123)
        self.assertEqual(metadata["api_retries"][0]["error_status"], 529)
        self.assertEqual(metadata["rate_limit"]["status"], "allowed")
        self.assertEqual(metadata["latency"]["duration_api_ms"], 1000)
        self.assertEqual(metadata["api_key_source"], "none")
        self.assertEqual(metadata["auth"]["subscription_type"], "max")

    def test_timeout_terminates_whole_process_tree(self):
        self.configure(mode="hang")
        backend = self.make(timeout=2.0)
        with self.assertRaises(ClaudeCodeTimeout):
            backend.generate(system_prompt="s", user_prompt="u", images=[self.frame("red")])
        pids = json.loads((self.root / "pids.json").read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(pid_alive(pid) for pid in pids.values()):
            time.sleep(0.1)
        self.assertFalse(pid_alive(pids["cli"]))
        self.assertFalse(pid_alive(pids["grandchild"]))

    def test_usage_exhaustion_and_credit_use_fail_closed(self):
        cases = {
            "plan_limit_rejected": {"rate_limit": {"status": "rejected", "rateLimitType": "seven_day"}},
            "overage_credit_use": {"rate_limit": {"status": "allowed", "isUsingOverage": True}},
            "http_429_result": {"is_error": True, "result_subtype": "error_during_execution",
                                "api_error_status": 429, "text": "usage limit"},
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                self.stop_file.unlink(missing_ok=True)
                self.configure(**overrides)
                with self.assertRaises(ClaudeCodeUsageLimit):
                    self.make().generate(system_prompt="s", user_prompt="u", images=[self.frame("red")])
                self.assertTrue(self.stop_file.exists())

    def test_protocol_violations(self):
        for name, overrides in {"tools_enabled": {"tools": ["Read"]}, "empty_response": {"text": ""}}.items():
            with self.subTest(case=name):
                self.configure(**overrides)
                with self.assertRaises(ClaudeCodeProtocolError):
                    self.make().generate(system_prompt="s", user_prompt="u", images=[self.frame("red")])


class SubscriptionRouteTests(FakeClaudeMixin, unittest.TestCase):
    def test_non_subscription_auth_is_rejected(self):
        for name, auth in {
            "console_api_billing": DEFAULT_AUTH | {"authMethod": "console"},
            "bedrock": DEFAULT_AUTH | {"apiProvider": "bedrock"},
            "pro_plan_not_allowed": DEFAULT_AUTH | {"subscriptionType": "pro"},
            "logged_out": DEFAULT_AUTH | {"loggedIn": False},
        }.items():
            with self.subTest(case=name):
                self.configure(auth=auth)
                with self.assertRaises(ClaudeCodeAuthError):
                    self.make()

    def test_api_key_source_in_request_is_rejected(self):
        self.configure(api_key_source="ANTHROPIC_API_KEY")
        with self.assertRaises(ClaudeCodeAuthError):
            self.make().generate(system_prompt="s", user_prompt="u", images=[self.frame("red")])

    def test_cli_version_change_mid_run_is_rejected(self):
        self.configure(init_version="2.1.271")
        with self.assertRaises(ClaudeCodeAuthError):
            self.make().generate(system_prompt="s", user_prompt="u", images=[self.frame("red")])

    def test_route_overrides_never_reach_the_cli(self):
        overrides = {
            "ANTHROPIC_API_KEY": "must-not-leak", "ANTHROPIC_AUTH_TOKEN": "x",
            "ANTHROPIC_BASE_URL": "http://evil", "CLAUDE_CODE_USE_BEDROCK": "1",
            "CLAUDE_CODE_USE_VERTEX": "1", "CLAUDE_CODE_USE_FOUNDRY": "1",
            "CLAUDE_CODE_OAUTH_TOKEN": "x", "ANTHROPIC_MODEL": "claude-haiku-4-5",
            "CLAUDE_CODE_EFFORT_LEVEL": "max", "CLAUDE_EFFORT": "max",
            "MAX_THINKING_TOKENS": "1024", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "256",
            "HTTPS_PROXY": "http://proxy", "CLAUDE_CONFIG_DIR": "/elsewhere",
        }
        with patch.dict(os.environ, overrides):
            backend = self.make()
            backend.generate(system_prompt="s", user_prompt="u", images=[self.frame("red")])
        self.assertEqual(sorted(overrides), sorted(set(backend.scrubbed_environment) & set(overrides)))
        for call in self.calls():
            for name in overrides:
                self.assertNotIn(name, call["env"], f"{name} leaked into {call['argv'][:2]}")
            self.assertEqual(call["env"]["DISABLE_AUTOUPDATER"], "1")
            self.assertEqual(call["env"]["CLAUDE_CODE_NO_MODEL_FALLBACK"], "1")
            self.assertEqual(call["env"]["DISABLE_PROMPT_CACHING"], "1")

    def test_managed_settings_overrides_are_rejected(self):
        for name, settings in {
            "env_base_url": {"env": {"ANTHROPIC_BASE_URL": "http://gateway"}},
            "api_key_helper": {"apiKeyHelper": "/bin/echo key"},
            "console_login": {"forceLoginMethod": "console"},
            "fallback_model": {"fallbackModel": ["claude-haiku-4-5"]},
        }.items():
            with self.subTest(case=name):
                path = self.root / f"{name}.json"
                path.write_text(json.dumps(settings))
                with self.assertRaises(ClaudeCodeAuthError):
                    validate_managed_settings([path])
        benign = self.root / "benign.json"
        benign.write_text(json.dumps({"env": {"DISABLE_TELEMETRY": "1"}, "forceLoginMethod": "claudeai"}))
        validate_managed_settings([benign])


class RTLLMIntegrationTests(FakeClaudeMixin, unittest.TestCase):
    def llm(self):
        from llm.rt_llm import RTLLM

        with patch.dict(os.environ, {
            "SIMWORLD_CLAUDE_CODE_EXECUTABLE": str(self.executable),
            "SIMWORLD_CLAUDE_CODE_STOP_FILE": str(self.stop_file),
        }):
            return RTLLM("claude-sonnet-5", provider="claude-code", api_mode="claude_code",
                         reasoning=True, reasoning_effort="low", request_timeout=30,
                         raise_on_api_error=True)

    def test_valid_action_uses_benchmark_schema(self):
        action, _, _, text, total, completion, reasoning = self.llm().generate_response_openai(
            system_prompt="s", user_prompt="u", images=[self.frame("red")])
        self.assertTrue(action.is_valid())
        self.assertEqual((action.action_type, action.action_param), ("move_to", "4"))
        self.assertEqual((total, completion, reasoning), (4350, 150, 120))

    def test_malformed_action_is_parse_error_never_executed(self):
        self.configure(text="I think going left might be fine, maybe waypoint 3.")
        llm = self.llm()
        action, *_ = llm.generate_response_openai(system_prompt="s", user_prompt="u", images=[self.frame("red")])
        self.assertFalse(action.is_valid())
        self.assertIsNone(action.action_type)
        self.assertEqual(llm.last_usage_metadata["claude_code"]["requested_model"], "claude-sonnet-5")

    def test_provider_and_api_mode_must_match(self):
        from llm.rt_llm import RTLLM

        with self.assertRaises(ValueError):
            RTLLM("claude-sonnet-5", provider="claude-code", api_mode="responses")


class RunnerAndCampaignTests(FakeClaudeMixin, unittest.TestCase):
    def test_single_task_runner_requires_explicit_effort_and_no_cap(self):
        base = ["--provider", "claude-code", "--api-mode", "claude_code", "--model", "claude-sonnet-5"]
        args = run_openai_benchmark.parse_args([*base, "--reasoning-effort", "low", "--max-output-tokens", "none"])
        self.assertIsNone(args.max_output_tokens)
        self.assertIsNone(args.key_file)
        for bad in (["--reasoning-effort", "none", "--max-output-tokens", "none"],
                    ["--reasoning-effort", "low", "--max-output-tokens", "128"],
                    ["--reasoning-effort", "low", "--max-output-tokens", "none", "--api-mode", "responses"]):
            with self.subTest(bad=bad), self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
                run_openai_benchmark.parse_args([*base, *bad])

    def _matrix(self, mode, env_modes, extra=()):
        environment = os.environ | {
            "SIMWORLD_CAMPAIGN_PROVIDER": "claude-code", "SIMWORLD_CAMPAIGN_API_MODE": "claude_code",
            "SIMWORLD_CAMPAIGN_MODEL": "claude-fable-5-1", "SIMWORLD_CAMPAIGN_EFFORTS": "low,medium,high",
            "SIMWORLD_CAMPAIGN_ENV_MODES": env_modes, "SIMWORLD_CAMPAIGN_DIFFICULTIES": "easy",
        }
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "evaluation/run_openai_frontier_matrix.py"), "--mode", mode,
             "--suite-dir", str(self.root / "suite"), "--workers", "1", "--gpus", "7", "--ports", "20000", *extra],
            cwd=REPO_ROOT, env=environment, text=True, capture_output=True, check=False,
        )

    def test_matrix_refuses_run_without_confirmation(self):
        completed = self._matrix("run", "realtime")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--confirm-claude-code-run", completed.stderr)

    def test_matrix_cells_use_subscription_transport(self):
        completed = self._matrix("cells", "realtime")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        cells = json.loads(completed.stdout)
        self.assertEqual(cells["cell_count"], 108)
        self.assertEqual(cells["suite_manifest"]["inference_surface"], "claude_code_subscription_cli")
        self.assertFalse(cells["suite_manifest"]["strict_model_transport_comparable_across_surfaces"])
        for cell in cells["cells"]:
            self.assertNotIn("--key-file", cell["runner_command"])
            self.assertIn("claude-code", cell["runner_command"])

    def test_campaign_dry_run_manifest_counts(self):
        output = self.root / "manifest.json"
        with patch("sys.stdout", io.StringIO()):
            code = campaign.main(["--mode", "plan", "--results-root", str(self.root / "full"),
                                  "--output", str(output), "--claude-executable", str(self.executable)])
        manifest = json.loads(output.read_text())
        self.assertEqual(code, 0, [k for k, v in manifest["protocol_checks"].items() if not v["pass"]])
        self.assertEqual(manifest["counts"], {"by_arm": {"fable_5_1": 216, "sonnet_5": 216}, "total": 432})
        self.assertTrue(manifest["all_checks_pass"])
        self.assertEqual(manifest["cli_preflight"]["auth"]["subscription_type"], "max")
        self.assertFalse(manifest["starts_rollouts"])

    def test_campaign_subset_selection(self):
        output = self.root / "subset.json"
        with patch("sys.stdout", io.StringIO()):
            code = campaign.main(["--mode", "plan", "--results-root", str(self.root / "subset"),
                                  "--output", str(output), "--claude-executable", str(self.executable),
                                  "--efforts", "low", "--env-modes", "realtime", "--task-ids", "35",
                                  "--parallel-arms", "--workers", "1", "--gpus", "6", "7",
                                  "--ports", "20000", "21000"])
        manifest = json.loads(output.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(manifest["counts"], {"by_arm": {"fable_5_1": 1, "sonnet_5": 1}, "total": 2})
        self.assertEqual([arm["gpus"] for arm in manifest["arms"]], [[6], [7]])
        self.assertEqual([arm["base_ports"] for arm in manifest["arms"]], [[20000], [21000]])

    def test_campaign_run_requires_confirmation(self):
        with patch("sys.stderr", io.StringIO()) as stderr:
            code = campaign.main(["--mode", "run", "--results-root", str(self.root / "run"),
                                  "--claude-executable", str(self.executable)])
        self.assertEqual(code, 2)
        self.assertIn("--confirm-claude-code-run", stderr.getvalue())


class UsageGuardTests(unittest.TestCase):
    RAW = {
        "subscription_type": "max", "rate_limits_available": True,
        "rate_limits": {
            "five_hour": {"utilization": 2, "resets_at": "2026-09-14T05:10:00+00:00"},
            "seven_day": {"utilization": 69, "resets_at": "2026-09-15T10:00:00+00:00"},
            "model_scoped": [{"display_name": "Fable", "utilization": 78, "resets_at": "2026-09-15T10:00:00+00:00"}],
            "extra_usage": {"is_enabled": False, "user_disabled": True, "spend_limit_reached": False},
            "spend": {"used": {"amount_minor": 0, "currency": "USD", "exponent": 2}, "enabled": False},
        },
    }

    def args(self, *extra):
        return campaign.parse_args(["--gpus", "1", "2", "3", "4", "--ports", "1", "2", "3", "4", *extra])

    def test_current_snapshot_passes_default_thresholds(self):
        usage = campaign.summarize_usage(self.RAW)
        self.assertEqual(campaign.guard_violations(usage, self.args(), 0, "Fable"), [])

    def test_thresholds_stop_before_exhaustion(self):
        usage = campaign.summarize_usage(self.RAW)
        self.assertTrue(campaign.scope_violations(usage, self.args("--min-model-scoped-remaining", "25"), "Fable"))
        self.assertEqual(campaign.scope_violations(usage, self.args("--min-model-scoped-remaining", "25"), None), [])
        self.assertTrue(campaign.account_violations(usage, self.args("--min-weekly-remaining", "31"), 0))
        self.assertTrue(campaign.account_violations(usage, self.args("--min-five-hour-remaining", "98"), 0))

    def test_usage_credits_are_never_spent_by_default(self):
        raw = json.loads(json.dumps(self.RAW))
        raw["rate_limits"]["extra_usage"]["is_enabled"] = True
        self.assertTrue(campaign.account_violations(campaign.summarize_usage(raw), self.args(), 0))
        raw["rate_limits"]["extra_usage"]["is_enabled"] = False
        raw["rate_limits"]["spend"]["used"]["amount_minor"] = 5
        self.assertTrue(campaign.account_violations(campaign.summarize_usage(raw), self.args(), 0))

    def test_malformed_usage_payload_fails_closed(self):
        with self.assertRaises(RuntimeError):
            campaign.summarize_usage({"rate_limits_available": False})


if __name__ == "__main__":
    unittest.main()
