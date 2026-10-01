"""Claude Code subscription transport for separately labeled SimWorld arms.

This backend is neither the Anthropic Messages API nor OpenRouter. Every
decision is one fresh, non-interactive ``claude -p`` process authenticated by
the local claude.ai subscription (OAuth). Built-in tools, MCP servers,
settings files, skills, slash commands and session persistence are disabled.
The benchmark system prompt replaces Claude Code's default coding prompt via
``--system-prompt-file``, and the benchmark user prompt plus the ordered
observation images travel as one ``stream-json`` user message, so no file or
tool round trip is involved.

Claude Code still adds a small product layer (a billing header, an Agent SDK
identity line, an environment reminder and, on Fable 5.1, a short reporting
guideline), so results form their own access-surface arm: same environment,
observations, prompts and scoring as the API arms, not identical transport.

Every guard fails closed: a wrong served model, any model-substitution event,
an effort that was not applied, an authentication-route change, usage
exhaustion, or any usage-credit (overage) consumption raises instead of
returning an action.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# Minimum CLI versions from https://code.claude.com/docs/en/model-config.
# Older CLIs pass unknown IDs through but label them as another model
# (2.1.237 reports canonicalModel "claude-fable-5" for "claude-fable-5-1").
MINIMUM_CLI_VERSION = {
    "claude-fable-5-1": (2, 1, 257),
    "claude-sonnet-5": (2, 1, 197),
}
SUPPORTED_EFFORTS = ("low", "medium", "high", "xhigh", "max")
MODEL_ALIASES = {"fable", "sonnet", "opus", "haiku", "best", "default", "opusplan"}
DEFAULT_ALLOWED_SUBSCRIPTIONS = ("max",)
# Claude Code announces model substitution with these system events.
MODEL_SUBSTITUTION_EVENTS = frozenset({
    "model_fallback",
    "model_consent_fallback",
    "model_refusal_fallback",
    "model_refusal_no_fallback",
})
# Only these variables reach the child; everything else (API keys, base URLs,
# Bedrock/Vertex/Foundry switches, model/effort overrides, output/thinking
# caps, proxies) is dropped so it cannot move billing or alter the arm.
INHERITED_ENVIRONMENT = (
    "HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR",
)
FORCED_ENVIRONMENT = {
    # Keep the CLI version fixed for the whole campaign.
    "DISABLE_AUTOUPDATER": "1",
    # Error out instead of silently swapping models (e.g. when Fable would
    # need usage credits or its scoped limit is exhausted).
    "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
    # Claude Code marks the trailing environment reminder, i.e. the whole
    # prompt including the per-step frames, for a 1-hour cache write (2x
    # input price). Frames change every decision, so that entry is never
    # read back; disabling caching bills input at 1x and changes nothing the
    # model sees.
    "DISABLE_PROMPT_CACHING": "1",
}
ROUTE_OVERRIDE_PATTERN = re.compile(
    r"^(ANTHROPIC_|CLAUDE_CODE_USE_|CLAUDE_CODE_OAUTH_TOKEN$|CLAUDE_CODE_EFFORT_LEVEL$|"
    r"CLAUDE_EFFORT$|MAX_THINKING_TOKENS$|CLAUDE_CODE_MAX_OUTPUT_TOKENS$|"
    r"CLAUDE_CODE_SUBAGENT_MODEL$|CLAUDE_CONFIG_DIR$|AWS_BEARER_TOKEN_BEDROCK$|"
    r"HTTPS?_PROXY$|ALL_PROXY$|https?_proxy$|all_proxy$)"
)
MANAGED_SETTINGS_PATHS = (
    Path("/etc/claude-code/managed-settings.json"),
    Path("/etc/claude-code/managed-settings.d"),
)
# Claude Code 2.1.270 silently re-encodes any image larger than this many
# bytes to JPEG before sending it (verified offline for PNG and WebP).
CLI_IMAGE_BYTE_LIMIT = 512_000
# Lossless WebP attempts, fastest first: (method, quality/effort). On real
# 720x640 frames m0 fits ~76% under the limit in ~25 ms, m2 all but ~4% in
# ~180 ms; m6 gains almost nothing and takes seconds on a loaded host.
WEBP_LOSSLESS_ATTEMPTS = ((0, 0), (2, 50))
WEBP_LOSSY_FALLBACK_QUALITIES = (95, 90, 85, 80)
SESSION_NAME = "simworld-policy"
STOP_FILE_ENVIRONMENT = "SIMWORLD_CLAUDE_CODE_STOP_FILE"
EXECUTABLE_ENVIRONMENT = "SIMWORLD_CLAUDE_CODE_EXECUTABLE"
_SETTINGS_REQUEST_ID = "simworld_settings"
_PR_SET_PDEATHSIG = 1


class ClaudeCodeError(RuntimeError):
    """Base class; every subclass aborts the decision (fail closed)."""

    # Campaign-fatal errors also write the arm's stop file so sibling workers
    # stop before spending more allowance.
    campaign_fatal = False


class ClaudeCodeInputError(ClaudeCodeError):
    pass


class ClaudeCodeProtocolError(ClaudeCodeError):
    pass


class ClaudeCodeTimeout(ClaudeCodeError):
    pass


class ClaudeCodeAuthError(ClaudeCodeError):
    campaign_fatal = True


class ClaudeCodeModelMismatch(ClaudeCodeError):
    campaign_fatal = True


class ClaudeCodeEffortMismatch(ClaudeCodeError):
    campaign_fatal = True


class ClaudeCodeUsageLimit(ClaudeCodeError):
    campaign_fatal = True


class ClaudeCodeCampaignStopped(ClaudeCodeError):
    campaign_fatal = True


@dataclass(frozen=True)
class ClaudeCodeResult:
    text: str
    usage: dict[str, Any]


def parse_version(text: str) -> tuple[int, int, int]:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    if not match:
        raise ClaudeCodeProtocolError(f"Unrecognized Claude Code version output: {text!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def child_environment(parent: Mapping[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    """Return the allowlisted child environment and the names (never values) dropped."""
    parent = dict(os.environ if parent is None else parent)
    environment = {name: parent[name] for name in INHERITED_ENVIRONMENT if parent.get(name)}
    environment.update(FORCED_ENVIRONMENT)
    scrubbed = sorted(name for name in parent if ROUTE_OVERRIDE_PATTERN.match(name))
    return environment, scrubbed


def validate_auth_status(payload: Mapping[str, Any], allowed_subscriptions: Iterable[str]) -> dict[str, Any]:
    """Require claude.ai OAuth on the first-party route with an allowed plan."""
    allowed = set(allowed_subscriptions)
    problems = []
    if payload.get("loggedIn") is not True:
        problems.append("not logged in")
    if payload.get("authMethod") != "claude.ai":
        problems.append(f"authMethod={payload.get('authMethod')!r} (need 'claude.ai')")
    if payload.get("apiProvider") != "firstParty":
        problems.append(f"apiProvider={payload.get('apiProvider')!r} (need 'firstParty')")
    if payload.get("subscriptionType") not in allowed:
        problems.append(
            f"subscriptionType={payload.get('subscriptionType')!r} (allowed {sorted(allowed)})"
        )
    if problems:
        raise ClaudeCodeAuthError("Claude Code is not on the subscription route: " + "; ".join(problems))
    account = f"{payload.get('orgId')}|{payload.get('email')}"
    return {
        "auth_method": payload["authMethod"],
        "api_provider": payload["apiProvider"],
        "subscription_type": payload["subscriptionType"],
        # Detects an account switch without storing identifiers in results.
        "account_fingerprint": hashlib.sha256(account.encode()).hexdigest()[:16],
    }


def validate_managed_settings(paths: Iterable[Path] = MANAGED_SETTINGS_PATHS) -> None:
    """Admin policy still applies under --safe-mode; reject route overrides there."""
    for root in paths:
        files = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
        for path in files:
            try:
                settings = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ClaudeCodeAuthError(f"Unreadable managed settings {path}: {exc}") from exc
            problems = [
                f"env sets {name}" for name in sorted(settings.get("env") or {})
                if ROUTE_OVERRIDE_PATTERN.match(name)
            ]
            problems += [
                f"sets {key}" for key in ("apiKeyHelper", "awsAuthRefresh", "awsCredentialExport", "fallbackModel")
                if key in settings
            ]
            if str(settings.get("forceLoginMethod", "claudeai")).lower() not in {"claudeai", "claude.ai"}:
                problems.append(f"forceLoginMethod={settings['forceLoginMethod']!r}")
            if problems:
                raise ClaudeCodeAuthError(
                    f"Managed settings {path} can override the subscription route: {'; '.join(problems)}"
                )


def served_model_matches(requested: str, served: str | None) -> bool:
    """Exact ID, or a dated snapshot of the same ID; never another family/version."""
    if not served:
        return False
    return served == requested or re.fullmatch(re.escape(requested) + r"-\d{8}", served) is not None


def _webp(image: Any, *, lossless: bool, method: int, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", lossless=lossless, method=method, quality=quality, exact=True)
    return buffer.getvalue()


def encode_image(image: Any) -> tuple[str, str, dict[str, Any]]:
    """Encode one observation so Claude Code forwards it unchanged.

    The API arms send PIL observations as PNG, but real 720x640 PNG frames
    exceed Claude Code's 512,000-byte limit and would be silently re-encoded
    to JPEG. They are therefore sent as lossless WebP (identical pixels);
    only if even that exceeds the limit is a high-quality lossy WebP used,
    and the image record says so. Arrays keep the API arms' exact JPEG bytes.
    """
    from PIL import Image

    started = time.perf_counter()
    if not isinstance(image, Image.Image):
        from utils.img2base64 import np_to_base64

        data = np_to_base64(image)
        size = len(base64.b64decode(data))
        return "image/jpeg", data, {
            "encoding": "jpeg_same_bytes_as_api_arms",
            "pixel_identical_to_api_arms": size <= CLI_IMAGE_BYTE_LIMIT,
            "bytes": size,
            "encode_seconds": round(time.perf_counter() - started, 4),
        }
    source = image if image.mode in ("RGB", "RGBA") else image.convert("RGB")
    for method, quality in WEBP_LOSSLESS_ATTEMPTS:
        raw = _webp(source, lossless=True, method=method, quality=quality)
        if len(raw) <= CLI_IMAGE_BYTE_LIMIT:
            return "image/webp", base64.b64encode(raw).decode("ascii"), {
                "encoding": f"webp_lossless_m{method}",
                "pixel_identical_to_api_arms": True,
                "bytes": len(raw),
                "encode_seconds": round(time.perf_counter() - started, 4),
            }
    for quality in WEBP_LOSSY_FALLBACK_QUALITIES:
        raw = _webp(source, lossless=False, method=4, quality=quality)
        if len(raw) <= CLI_IMAGE_BYTE_LIMIT:
            return "image/webp", base64.b64encode(raw).decode("ascii"), {
                "encoding": f"webp_lossy_q{quality}",
                "pixel_identical_to_api_arms": False,
                "bytes": len(raw),
                "encode_seconds": round(time.perf_counter() - started, 4),
            }
    raise ClaudeCodeInputError("Observation could not be encoded under Claude Code's image byte limit")


def build_user_message(user_prompt: str, images: Sequence[Any]) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    """Text first, then images oldest to newest: the API arms' content order."""
    if not images:
        raise ClaudeCodeInputError("No observation images were supplied; refusing a blind decision")
    missing = [index for index, image in enumerate(images) if image is None]
    if missing:
        raise ClaudeCodeInputError(f"Observation image(s) missing at positions {missing}")
    started = time.perf_counter()
    # Encoders release the GIL, so frames encode concurrently; map keeps order.
    with ThreadPoolExecutor(max_workers=min(4, len(images))) as pool:
        encoded = list(pool.map(encode_image, images))
    encode_wall = time.perf_counter() - started
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    manifest = []
    for position, (media_type, data, record) in enumerate(encoded):
        content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        manifest.append({
            "position": position,
            "media_type": media_type,
            "base64_sha256": hashlib.sha256(data.encode("ascii")).hexdigest()[:16],
        } | record)
    message = {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
        "session_id": "",
    }
    return message, manifest, encode_wall


def _descendants(pid: int) -> list[int]:
    """All descendants of ``pid`` via the Linux /proc children lists."""
    found: list[int] = []
    pending = [pid]
    while pending:
        parent = pending.pop()
        for children_file in Path(f"/proc/{parent}/task").glob("*/children"):
            try:
                children = children_file.read_text().split()
            except OSError:
                continue
            for child in map(int, children):
                if child not in found:
                    found.append(child)
                    pending.append(child)
    return found


def pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    # A zombie has already exited; it only awaits reaping by its parent.
    return stat.rsplit(")", 1)[-1].split()[0] != "Z"


def terminate_process_tree(process: subprocess.Popen, grace_seconds: float = 5.0) -> list[int]:
    """SIGTERM the CLI and every descendant, then SIGKILL survivors."""
    targets = [process.pid, *_descendants(process.pid)]
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None and not any(pid_alive(pid) for pid in targets[1:]):
            break
        time.sleep(0.05)
    for pid in [*targets, *_descendants(process.pid)]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return targets


def _die_with_parent() -> None:
    # The CLI stays in the rollout's process group (so the matrix's killpg
    # reaches it) and additionally dies if the rollout process is killed.
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    except OSError:
        pass


def parse_stream(stdout: str) -> dict[str, Any]:
    """Group the CLI's stream-json events by the parts the guards need."""
    parsed: dict[str, Any] = {
        "init": None, "settings": None, "assistant": [], "result": None,
        "rate_limits": [], "api_retries": [], "substitutions": [], "unparsed_lines": 0,
    }
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            parsed["unparsed_lines"] += 1
            continue
        if not isinstance(event, dict):
            continue
        kind, subtype = event.get("type"), event.get("subtype")
        if kind == "system" and subtype == "init":
            parsed["init"] = event
        elif kind == "system" and subtype in MODEL_SUBSTITUTION_EVENTS:
            parsed["substitutions"].append(event)
        elif kind == "system" and subtype == "api_retry":
            parsed["api_retries"].append(event)
        elif kind == "control_response":
            response = event.get("response") or {}
            if response.get("request_id") == _SETTINGS_REQUEST_ID:
                parsed["settings"] = response
        elif kind == "assistant":
            parsed["assistant"].append(event)
        elif kind == "rate_limit_event":
            parsed["rate_limits"].append(event.get("rate_limit_info") or {})
        elif kind == "result":
            parsed["result"] = event
    return parsed


class ClaudeCodeBackend:
    """Invoke one fresh, tool-less, subscription-authenticated ``claude -p`` per decision."""

    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        timeout: float | None,
        executable: str | None = None,
        allowed_subscriptions: Iterable[str] = DEFAULT_ALLOWED_SUBSCRIPTIONS,
        stop_file: str | None = None,
        managed_settings_paths: Iterable[Path] = MANAGED_SETTINGS_PATHS,
    ) -> None:
        if model.lower() in MODEL_ALIASES or "[" in model or model not in MINIMUM_CLI_VERSION:
            raise ValueError(
                f"Claude Code arm accepts only the full model IDs {sorted(MINIMUM_CLI_VERSION)}; "
                f"got {model!r} (aliases can re-point to another model)"
            )
        if reasoning_effort not in SUPPORTED_EFFORTS:
            raise ValueError(
                f"Claude Code effort must be explicit and one of {list(SUPPORTED_EFFORTS)}; "
                f"got {reasoning_effort!r} (omitting --effort silently applies the CLI default)"
            )
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        requested = executable or os.environ.get(EXECUTABLE_ENVIRONMENT) or "claude"
        resolved = shutil.which(requested)
        if resolved is None:
            raise ValueError(f"Claude Code executable was not found: {requested}")
        self.executable = resolved
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.allowed_subscriptions = tuple(allowed_subscriptions)
        self.stop_file = stop_file if stop_file is not None else os.environ.get(STOP_FILE_ENVIRONMENT)
        self.environment, self.scrubbed_environment = child_environment()
        # One stable, empty, per-user cwd keeps Claude Code's injected
        # environment reminder identical across decisions and workers.
        self.workdir = Path(tempfile.gettempdir()) / f"simworld_claude_code_cwd_{os.getuid()}"
        self.workdir.mkdir(mode=0o700, exist_ok=True)
        if self.workdir.stat().st_uid != os.getuid() or any(self.workdir.iterdir()):
            raise ClaudeCodeInputError(f"Policy cwd must be an empty directory we own: {self.workdir}")
        self._check_stop_file()
        self.cli_version = parse_version(self._run_cli("--version").stdout)
        minimum = MINIMUM_CLI_VERSION[model]
        if self.cli_version < minimum:
            raise ClaudeCodeModelMismatch(
                f"{model} requires Claude Code >= {'.'.join(map(str, minimum))}; "
                f"{self.executable} is {'.'.join(map(str, self.cli_version))}"
            )
        validate_managed_settings(managed_settings_paths)
        self.auth = self._auth_status()

    # ------------------------------------------------------------------ setup
    def _run_cli(self, *arguments: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.executable, *arguments], text=True, capture_output=True,
            timeout=timeout, env=self.environment, cwd=self.workdir, check=False,
        )

    def _auth_status(self) -> dict[str, Any]:
        completed = self._run_cli("auth", "status")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ClaudeCodeAuthError(
                f"`claude auth status` returned no JSON (exit {completed.returncode})"
            ) from exc
        return validate_auth_status(payload, self.allowed_subscriptions)

    def build_command(self, system_prompt_path: Path) -> list[str]:
        return [
            self.executable,
            "-p",
            "--model", self.model,
            "--effort", self.reasoning_effort,
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--safe-mode",
            "--tools", "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--setting-sources", "",
            "--settings", json.dumps({"switchModelsOnFlag": False}, separators=(",", ":")),
            "--permission-mode", "dontAsk",
            # A named session skips Claude Code's automatic title generation,
            # which otherwise sends an extra request to the same model.
            "--name", SESSION_NAME,
            "--system-prompt-file", str(system_prompt_path),
        ]

    # ----------------------------------------------------------------- guards
    def _check_stop_file(self) -> None:
        if self.stop_file and Path(self.stop_file).exists():
            raise ClaudeCodeCampaignStopped(f"Campaign stop file is present: {self.stop_file}")

    def _write_stop_file(self, error: ClaudeCodeError) -> None:
        if not self.stop_file:
            return
        path = Path(self.stop_file)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "time": datetime.now(timezone.utc).isoformat(),
            "reason": type(error).__name__,
            "detail": str(error),
            "pid": os.getpid(),
            "model": self.model,
            "effort": self.reasoning_effort,
        }, indent=2) + "\n", encoding="utf-8")

    def _validate(self, parsed: dict[str, Any], returncode: int, stderr_tail: str) -> dict[str, Any]:
        # Usage and substitution signals first: they explain most other failures.
        for info in parsed["rate_limits"]:
            if info.get("isUsingOverage"):
                raise ClaudeCodeUsageLimit(f"Request consumed usage credits (overage): {info}")
            if info.get("status") == "rejected":
                raise ClaudeCodeUsageLimit(
                    f"Plan limit reached ({info.get('rateLimitType') or 'unknown window'}); "
                    f"resets at {info.get('resetsAt')}"
                )
        if parsed["substitutions"]:
            event = parsed["substitutions"][0]
            raise ClaudeCodeModelMismatch(
                f"Claude Code substituted the model ({event.get('subtype')}): "
                f"{event.get('content') or event.get('trigger')}"
            )
        init = parsed["init"]
        if init is None:
            raise ClaudeCodeProtocolError(f"Claude Code exited {returncode} without an init event: {stderr_tail}")
        if init.get("apiKeySource") != "none":
            raise ClaudeCodeAuthError(
                f"Request did not use subscription OAuth: apiKeySource={init.get('apiKeySource')!r}"
            )
        if parse_version(str(init.get("claude_code_version"))) != self.cli_version:
            raise ClaudeCodeAuthError(f"Claude Code version changed mid-run: {init.get('claude_code_version')}")
        if init.get("model") != self.model:
            raise ClaudeCodeModelMismatch(f"CLI resolved model {init.get('model')!r}, requested {self.model!r}")
        if init.get("tools") or init.get("mcp_servers"):
            raise ClaudeCodeProtocolError(
                f"Tools must be disabled: tools={init.get('tools')} mcp={init.get('mcp_servers')}"
            )
        settings = parsed["settings"] or {}
        if settings.get("subtype") != "success":
            raise ClaudeCodeProtocolError(f"get_settings control request failed: {settings}")
        applied = (settings.get("response") or {}).get("applied") or {}
        if applied.get("model") != self.model:
            raise ClaudeCodeModelMismatch(f"Applied model {applied.get('model')!r}, requested {self.model!r}")
        if applied.get("effort") != self.reasoning_effort:
            raise ClaudeCodeEffortMismatch(
                f"Applied effort {applied.get('effort')!r}, requested {self.reasoning_effort!r}"
            )
        result = parsed["result"]
        if result is None:
            raise ClaudeCodeProtocolError(f"Claude Code exited {returncode} without a result event: {stderr_tail}")
        if result.get("is_error") or result.get("subtype") != "success":
            detail = str(result.get("result") or result.get("errors") or stderr_tail)
            status = result.get("api_error_status")
            lowered = detail.lower()
            if status == 429 or "usage limit" in lowered or "limit reached" in lowered:
                raise ClaudeCodeUsageLimit(f"Claude Code usage limit: {detail[:300]}")
            if status in {401, 403} or "authentication" in lowered or "log in" in lowered:
                raise ClaudeCodeAuthError(f"Claude Code authentication failure: {detail[:300]}")
            raise ClaudeCodeProtocolError(
                f"Claude Code result {result.get('subtype')!r} (api_status={status}): {detail[:300]}"
            )
        served = [(event.get("message") or {}).get("model") for event in parsed["assistant"]]
        if not served:
            raise ClaudeCodeProtocolError("Claude Code returned no assistant message")
        # modelUsage is keyed by the *requested* model even when another model
        # served the turn, so each assistant message's model is the authority.
        wrong = [model for model in served if not served_model_matches(self.model, model)]
        if wrong:
            raise ClaudeCodeModelMismatch(f"Served model(s) {wrong} differ from requested {self.model!r}")
        for event in parsed["assistant"]:
            message = event.get("message") or {}
            if event.get("isApiErrorMessage") or message.get("stop_reason") == "refusal":
                raise ClaudeCodeProtocolError(f"Assistant message is an API error/refusal: {message.get('stop_reason')}")
        if result.get("num_turns") != 1 or result.get("permission_denials"):
            raise ClaudeCodeProtocolError(
                f"Expected one tool-free turn; got num_turns={result.get('num_turns')} "
                f"denials={result.get('permission_denials')}"
            )
        model_usage = result.get("modelUsage") or {}
        auxiliary = sorted(set(model_usage) - {self.model})
        if auxiliary:
            raise ClaudeCodeModelMismatch(f"Unexpected models in modelUsage: {auxiliary}")
        canonical = (model_usage.get(self.model) or {}).get("canonicalModel")
        if canonical is not None and canonical != self.model:
            raise ClaudeCodeModelMismatch(f"CLI canonicalized {self.model!r} as {canonical!r}")
        if returncode != 0:
            raise ClaudeCodeProtocolError(f"Claude Code exited {returncode} after a result: {stderr_tail}")
        return {"applied": applied, "served": served, "canonical": canonical}

    # --------------------------------------------------------------- generate
    def generate(self, *, system_prompt: str, user_prompt: str, images: Iterable[Any]) -> ClaudeCodeResult:
        try:
            return self._generate(system_prompt=system_prompt, user_prompt=user_prompt, images=list(images))
        except ClaudeCodeError as error:
            if error.campaign_fatal and not isinstance(error, ClaudeCodeCampaignStopped):
                self._write_stop_file(error)
            raise

    def _generate(self, *, system_prompt: str, user_prompt: str, images: list[Any]) -> ClaudeCodeResult:
        self._check_stop_file()
        message, image_manifest, encode_wall = build_user_message(user_prompt, images)
        settings_request = {
            "type": "control_request",
            "request_id": _SETTINGS_REQUEST_ID,
            "request": {"subtype": "get_settings"},
        }
        stdin = json.dumps(settings_request) + "\n" + json.dumps(message, separators=(",", ":")) + "\n"
        with tempfile.TemporaryDirectory(prefix="simworld_claude_code_") as temporary:
            system_prompt_path = Path(temporary) / "system_prompt.txt"
            system_prompt_path.write_text(system_prompt, encoding="utf-8")
            started = time.monotonic()
            process = subprocess.Popen(
                self.build_command(system_prompt_path),
                cwd=self.workdir, env=self.environment, text=True,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=_die_with_parent,
            )
            try:
                stdout, stderr = process.communicate(stdin, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                killed = terminate_process_tree(process)
                process.communicate()
                raise ClaudeCodeTimeout(f"Claude Code exceeded {self.timeout:g}s; terminated pids {killed}") from None
            except BaseException:
                terminate_process_tree(process)
                process.communicate()
                raise
            wall_seconds = time.monotonic() - started

        parsed = parse_stream(stdout)
        stderr_tail = (stderr or "").strip()[-500:]
        checks = self._validate(parsed, process.returncode, stderr_tail)
        result = parsed["result"]
        text = str(result.get("result") or "").strip()
        if not text:
            raise ClaudeCodeProtocolError("Claude Code returned an empty final response")
        usage = result.get("usage") or {}
        uncached = int(usage.get("input_tokens") or 0)
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_write = int(usage.get("cache_creation_input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        if uncached + cache_read + cache_write == 0 or output_tokens == 0:
            raise ClaudeCodeProtocolError("Claude Code completed without machine-readable usage")
        thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
        if thinking is None:
            thinking = ((result.get("modelUsage") or {}).get(self.model) or {}).get("thinkingTokens")
        prompt_tokens = uncached + cache_read + cache_write
        metadata = {
            "access_surface": "claude_code_subscription_cli",
            "requested_model": self.model,
            "served_models": checks["served"],
            "canonical_model": checks["canonical"],
            "requested_effort": self.reasoning_effort,
            "effective_effort": checks["applied"].get("effort"),
            "effective_effort_source": "get_settings.applied",
            "cli_version": ".".join(map(str, self.cli_version)),
            "cli_executable": self.executable,
            "auth": self.auth,
            "api_key_source": parsed["init"].get("apiKeySource"),
            "scrubbed_environment": self.scrubbed_environment,
            "policy_cwd": str(self.workdir),
            "session_id": result.get("session_id"),
            "exit_status": process.returncode,
            "stop_reason": result.get("stop_reason"),
            "num_turns": result.get("num_turns"),
            "api_retries": [
                {key: event.get(key) for key in ("attempt", "max_retries", "retry_delay_ms", "error_status", "error")}
                for event in parsed["api_retries"]
            ],
            "rate_limit": parsed["rate_limits"][-1] if parsed["rate_limits"] else None,
            "latency": {
                "image_encode_seconds": round(encode_wall, 4),
                "process_wall_seconds": round(wall_seconds, 4),
                "duration_ms": result.get("duration_ms"),
                "duration_api_ms": result.get("duration_api_ms"),
                "ttft_ms": result.get("ttft_ms"),
                "cli_overhead_seconds": (
                    round(wall_seconds - result["duration_api_ms"] / 1000.0, 4)
                    if isinstance(result.get("duration_api_ms"), (int, float)) else None
                ),
            },
            "images": image_manifest,
            "uncached_input_tokens": uncached,
            "thinking_tokens_reported": thinking is not None,
            # Claude Code's API-list-price estimate; NOT the subscription bill.
            "api_list_price_equivalent_usd": result.get("total_cost_usd"),
            "unparsed_stdout_lines": parsed["unparsed_lines"],
            "stderr_tail": stderr_tail,
        }
        return ClaudeCodeResult(text=text, usage={
            "prompt_tokens": prompt_tokens,
            # Output tokens include thinking, as in the Responses API arm.
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "reasoning_tokens": None if thinking is None else int(thinking),
            "cached_tokens": cache_read,
            "cache_write_tokens": cache_write,
            # Subscription usage has no per-request dollar price.
            "cost_usd": None,
            "claude_code": metadata,
        })
