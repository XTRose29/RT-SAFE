"""Bounded Codex CLI visual policy, using the user's existing CLI sign-in.

This is a policy-only invocation: shell, agent delegation, apps, web search,
image generation, and file-viewing tools are disabled. The controller alone
executes the validated navigation action in Unreal.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from .policy import SYSTEM_PROMPT, parse_action

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"type": "string", "enum": ["move_to", "turn_around", "wait"]},
        "action_param": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["action_type", "action_param", "reasoning"],
    "additionalProperties": False,
}


def inspect_cli_events(text: str) -> tuple[list[dict], dict]:
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    usage = None
    for event in events:
        if event.get("type") in ("error", "turn.failed"):
            raise RuntimeError("Codex policy failed: " + json.dumps(event))
        item = event.get("item", {})
        kind = item.get("type")
        if kind and kind not in ("agent_message", "reasoning"):
            raise RuntimeError(
                f"Unexpected Codex policy item {kind}; no tools are allowed"
            )
        if event.get("type") == "turn.completed":
            if usage is not None:
                raise RuntimeError("Policy must complete exactly one turn")
            usage = event["usage"]
    if usage is None:
        raise RuntimeError("Codex policy did not complete")
    return events, usage


class CodexVisualPolicy:
    MODEL = "gpt-5.6-luna"

    def __init__(self, *, max_calls=8, max_tokens=100_000):
        if not 1 <= max_calls <= 16 or not 1 <= max_tokens <= 200_000:
            raise ValueError(
                "CLI pilot supports at most 16 decisions and 200,000 tokens"
            )
        self.executable = shutil.which("codex")
        if not self.executable:
            raise RuntimeError("Codex CLI is not installed")
        self.version = subprocess.check_output(
            [self.executable, "--version"], text=True, timeout=10
        ).strip()
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.calls, self.tokens = 0, 0
        self.spent = None  # ChatGPT-plan CLI output does not report a dollar charge.

    def decide(self, image_png: bytes, user_prompt: str):
        if self.calls >= self.max_calls or self.tokens + 10_000 > self.max_tokens:
            raise RuntimeError("CLI pilot decision/token limit reached before request")
        self.calls += 1
        with tempfile.TemporaryDirectory(prefix="rtsafe-policy-") as directory:
            root = Path(directory)
            (root / "instructions.txt").write_text(SYSTEM_PROMPT)
            (root / "action.schema.json").write_text(json.dumps(ACTION_SCHEMA))
            (root / "observation.png").write_bytes(image_png)
            output = root / "action.json"
            command = [
                self.executable,
                "exec",
                "--strict-config",
                "--ignore-user-config",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--model",
                self.MODEL,
                "--json",
                "--cd",
                directory,
                "--image",
                str(root / "observation.png"),
                "--output-schema",
                str(root / "action.schema.json"),
                "--output-last-message",
                str(output),
            ]
            settings = {
                "model_reasoning_effort": "low",
                "approval_policy": "never",
                "web_search": "disabled",
                "model_instructions_file": str(root / "instructions.txt"),
                "features.shell_tool": False,
                "features.unified_exec": False,
                "features.multi_agent": False,
                "features.apps": False,
                "features.image_generation": False,
                "features.view_image": False,
                "features.skill_search": False,
                "features.skip_host_skill_discovery": True,
                "suppress_unstable_features_warning": True,
            }
            for key, value in settings.items():
                command += ["-c", key + "=" + json.dumps(value)]
            command.append("-")
            environment = {
                key: value
                for key, value in os.environ.items()
                if key
                in (
                    "PATH",
                    "HOME",
                    "LANG",
                    "LC_ALL",
                    "HTTPS_PROXY",
                    "HTTP_PROXY",
                    "NO_PROXY",
                    "SSL_CERT_FILE",
                    "SSL_CERT_DIR",
                )
            }
            start = time.monotonic()
            result = subprocess.run(
                command,
                input=user_prompt,
                text=True,
                capture_output=True,
                timeout=60,
                env=environment,
            )
            elapsed = time.monotonic() - start
            if result.returncode:
                raise RuntimeError(
                    "Codex policy invocation failed: "
                    + result.stdout[-3000:]
                    + result.stderr[-1000:]
                )
            events, usage = inspect_cli_events(result.stdout)
            self.tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            if self.tokens > self.max_tokens:
                raise RuntimeError(
                    "CLI token limit exceeded; no further requests will be sent"
                )
            action = parse_action(output.read_text())
            return action, {
                "kind": "codex-cli-visual-pilot",
                "requested_model": self.MODEL,
                "reasoning_effort": "low",
                "cli_version": self.version,
                "latency_s": elapsed,
                "usage": usage,
                "cumulative_tokens": self.tokens,
                "cost_usd": None,
                "billing_note": "CLI does not report dollar cost",
                "tool_calls": 0,
                "raw_response": action.as_dict(),
                "events": events,
                "user_prompt": user_prompt,
            }
