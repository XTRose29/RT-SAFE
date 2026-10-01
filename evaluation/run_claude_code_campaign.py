#!/usr/bin/env python3
"""Plan or run the Claude Code subscription arms behind an account-usage guard.

Full design (seed 0, all 36 tasks; easy by default):

* ``claude-fable-5-1``: efforts low/medium/high, realtime and static  -> 216 cells
* ``claude-sonnet-5``:  efforts low/medium/high, realtime and static  -> 216 cells

``xhigh`` and ``max`` are also accepted; Claude Code's default effort is ``high``.

``--difficulties``, ``--efforts``, ``--env-modes`` and ``--task-ids`` select a subset (for
example a one-task smoke, or realtime/low only). The default ``plan`` mode
writes a dry-run manifest and never starts Unreal Engine or sends a model
turn. ``run`` additionally requires ``--confirm-claude-code-run`` and a
passing auth/version/usage preflight. While arms run, the guard polls plan
usage through Claude Code's own read-only ``get_usage`` control request (no
model turn is submitted) and interrupts coordinators before the configured
remaining-allowance thresholds. It never enables usage credits or changes a
spend limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm.claude_code_backend import (  # noqa: E402
    CLI_IMAGE_BYTE_LIMIT,
    DEFAULT_ALLOWED_SUBSCRIPTIONS,
    EXECUTABLE_ENVIRONMENT,
    FORCED_ENVIRONMENT,
    MINIMUM_CLI_VERSION,
    ROUTE_OVERRIDE_PATTERN,
    STOP_FILE_ENVIRONMENT,
    ClaudeCodeError,
    child_environment,
    parse_version,
    validate_auth_status,
)

MATRIX_RUNNER = REPO_ROOT / "evaluation" / "run_openai_frontier_matrix.py"
DEFAULT_RESULTS_ROOT = Path("results/claude-code/easy_seed0")
DEFAULT_EXECUTABLE = "claude"
ALL_EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORTS = ("low", "medium", "high")
ALL_DIFFICULTIES = ("easy", "medium", "hard")
ARMS = (
    {
        "key": "fable_5_1",
        "model": "claude-fable-5-1",
        "env_modes": ("realtime", "static"),
        # Display name of the model-scoped weekly window in get_usage.
        "usage_scope": "Fable",
    },
    {
        "key": "sonnet_5",
        "model": "claude-sonnet-5",
        "env_modes": ("realtime", "static"),
        "usage_scope": None,
    },
)
FULL_DESIGN_CELLS = {"fable_5_1": 216, "sonnet_5": 216}
TASK_COUNT = 36
TOUCHED_ROAD_COMMIT = "b5fde3343d0e4f3e81f60b0f4cbe54ec9221922d"
SOURCE_FILES = (
    "base/rt_agent.py",
    "manager/world_manager.py",
    "llm/prompt.py",
    "llm/rt_llm.py",
    "llm/claude_code_backend.py",
    "evaluation/run_openai_benchmark.py",
    "evaluation/run_openai_frontier_matrix.py",
    "evaluation/run_claude_code_campaign.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A per-process temporary name: two campaigns (one per lane) can share a
    # results root and rewrite the same file (for example latest_usage.json).
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "_")


def git_output(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        # ZIP source releases have no git metadata; file hashes remain recorded.
        return ""


def source_snapshot() -> dict[str, Any]:
    return {
        "git_commit": git_output("rev-parse", "HEAD") or None,
        "git_branch": git_output("branch", "--show-current"),
        "dirty_worktree_entries": git_output("status", "--short", "--untracked-files=all").splitlines(),
        "source_sha256": {
            relative: (
                hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
                if (REPO_ROOT / relative).is_file() else None
            )
            for relative in SOURCE_FILES
        },
    }


# ------------------------------------------------------------------ arms
def selected_arms(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Arms in order, restricted to the requested efforts and environment modes."""
    by_key = {arm["key"]: arm for arm in ARMS}
    arms = []
    for key in args.arms:
        arm = dict(by_key[key])
        modes = tuple(mode for mode in arm["env_modes"] if mode in args.env_modes)
        if not modes:
            raise SystemExit(f"arm {key} has no environment mode in {args.env_modes}")
        arm["env_modes"] = modes
        arm["efforts"] = tuple(args.efforts)
        tasks = len(args.task_ids) if args.task_ids else TASK_COUNT
        arm["expected_cells"] = (
            tasks * len(arm["efforts"]) * len(modes) * len(args.difficulties)
        )
        arms.append(arm)
    return arms


def arm_resources(args: argparse.Namespace, index: int) -> tuple[list[int], list[int]]:
    """Parallel arms take consecutive, disjoint GPU/port slices."""
    start = index * args.workers if args.parallel_arms else 0
    return args.gpus[start:start + args.workers], args.ports[start:start + args.workers]


def campaign_tag(
    arms: list[dict[str, Any]], efforts: list[str],
    difficulties: list[str] | tuple[str, ...] = ("easy",),
) -> str:
    return (
        "+".join(arm["key"] for arm in arms)
        + "-" + "+".join(difficulties)
        + "-" + "+".join(efforts)
    )


def arm_stop_file(args: argparse.Namespace, arm: dict[str, Any]) -> Path:
    return args.results_root / model_slug(arm["model"]) / "CLAUDE_CODE_STOP.json"


def global_stop_file(args: argparse.Namespace) -> Path:
    return args.results_root / "CLAUDE_CODE_STOP.json"


def arm_environment(arm: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "SIMWORLD_CAMPAIGN_MODEL": arm["model"],
        "SIMWORLD_CAMPAIGN_PROVIDER": "claude-code",
        "SIMWORLD_CAMPAIGN_API_MODE": "claude_code",
        "SIMWORLD_CAMPAIGN_DIFFICULTIES": ",".join(args.difficulties),
        "SIMWORLD_CAMPAIGN_ENV_MODES": ",".join(arm["env_modes"]),
        "SIMWORLD_CAMPAIGN_EFFORTS": ",".join(arm["efforts"]),
        "SIMWORLD_CAMPAIGN_INPUT_PRICE": "0",
        "SIMWORLD_CAMPAIGN_OUTPUT_PRICE": "0",
        EXECUTABLE_ENVIRONMENT: args.claude_executable,
        STOP_FILE_ENVIRONMENT: str(arm_stop_file(args, arm)),
    })
    return environment


def matrix_command(arm: dict[str, Any], args: argparse.Namespace, mode: str, index: int) -> list[str]:
    gpus, ports = arm_resources(args, index)
    command = [
        sys.executable, str(MATRIX_RUNNER),
        "--mode", mode,
        "--suite-dir", str(args.results_root / model_slug(arm["model"])),
        "--workers", str(args.workers),
        "--worker-count", str(args.workers),
        "--gpus", *map(str, gpus),
        "--ports", *map(str, ports),
        "--max-attempts", str(args.max_attempts),
        "--request-timeout", str(args.request_timeout),
        "--unrealcv-request-timeout", str(args.unrealcv_request_timeout),
        "--cell-timeout", str(args.cell_timeout),
        "--ue-ready-timeout", str(args.ue_ready_timeout),
        "--ue-settle-seconds", str(args.ue_settle_seconds),
        "--ue-start-stagger-seconds", str(args.ue_start_stagger_seconds),
        "--process-shutdown-timeout", str(args.process_shutdown_timeout),
        "--status-interval", str(args.status_interval),
        "--ue-port-stride", str(args.ue_port_stride),
        "--record-png-compress-level", str(args.record_png_compress_level),
        "--record-images-first-steps", str(args.record_images_first_steps),
        "--record-images-last-steps", str(args.record_images_last_steps),
        "--dynamic-queue",
        "--fast-simulation",
        "--ue-no-rhi-thread",
        "--no-record-output-images",
        "--resume",
    ]
    if args.task_ids:
        command += ["--task-ids", *map(str, args.task_ids)]
    return command


def matrix_json(arm: dict[str, Any], args: argparse.Namespace, mode: str, index: int) -> dict[str, Any]:
    completed = subprocess.run(
        matrix_command(arm, args, mode, index), cwd=REPO_ROOT, env=arm_environment(arm, args),
        text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{mode} for {arm['model']} failed: {completed.stderr.strip()[-800:]}")
    return json.loads(completed.stdout)


# ----------------------------------------------------------------- preflight
def cli_preflight(executable: str, allowed_subscriptions: tuple[str, ...]) -> dict[str, Any]:
    """Version and auth route of the CLI; no model turn is sent."""
    environment, scrubbed = child_environment()
    report: dict[str, Any] = {"executable": executable, "scrubbed_environment": scrubbed}
    try:
        version_text = subprocess.run(
            [executable, "--version"], env=environment, text=True,
            capture_output=True, timeout=60, check=False,
        ).stdout.strip()
        version = parse_version(version_text)
        report["version"] = ".".join(map(str, version))
        report["meets_minimum"] = {model: version >= minimum for model, minimum in MINIMUM_CLI_VERSION.items()}
    except (OSError, subprocess.SubprocessError, ClaudeCodeError) as exc:
        report["version_error"] = str(exc)
        report["meets_minimum"] = {model: False for model in MINIMUM_CLI_VERSION}
    try:
        status = subprocess.run(
            [executable, "auth", "status"], env=environment, text=True,
            capture_output=True, timeout=60, check=False,
        )
        report["auth"] = validate_auth_status(json.loads(status.stdout), allowed_subscriptions)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ClaudeCodeError) as exc:
        report["auth_error"] = str(exc)
    return report


def _kill_group(process: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue


def read_plan_usage(executable: str, timeout: float = 30.0, attempts: int = 3,
                    backoff: float = 5.0) -> dict[str, Any]:
    """Plan usage via ``get_usage``, retried before giving up.

    The control request intermittently answers without a rate-limit block
    (observed 2026-09-22, alongside transient API DNS failures). A single such
    answer used to stop the whole campaign, so the probe is retried; if every
    attempt fails the error still propagates and the guard still fails closed.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _read_plan_usage_once(executable, timeout)
        except RuntimeError as exc:
            last = exc
            if attempt < attempts:
                time.sleep(backoff)
    assert last is not None
    raise last


def _read_plan_usage_once(executable: str, timeout: float = 30.0) -> dict[str, Any]:
    """One ``get_usage`` control request.

    Only the control request is written to stdin; no user message is ever
    sent, so no model turn can start. It reads the same data as /usage.
    """
    environment, _ = child_environment()
    with tempfile.TemporaryDirectory(prefix="simworld_claude_usage_") as workdir:
        process = subprocess.Popen(
            [executable, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
             "--verbose", "--no-session-persistence", "--safe-mode", "--tools", "",
             "--strict-mcp-config", "--disable-slash-commands", "--setting-sources", "",
             "--model", "claude-sonnet-5"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, env=environment, cwd=workdir, start_new_session=True,
        )
        lines: "queue.Queue[str | None]" = queue.Queue()

        def pump() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps({
                "type": "control_request", "request_id": "simworld_usage",
                "request": {"subtype": "get_usage", "skip_behaviors": True},
            }) + "\n")
            process.stdin.flush()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    line = lines.get(timeout=max(0.05, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if line is None:
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") in {"assistant", "result"}:
                    raise RuntimeError("usage probe unexpectedly produced a model turn")
                response = event.get("response") or {}
                if event.get("type") == "control_response" and response.get("request_id") == "simworld_usage":
                    if response.get("subtype") != "success":
                        raise RuntimeError(f"get_usage failed: {response.get('error')}")
                    return summarize_usage(response.get("response") or {})
            raise RuntimeError("Claude Code did not answer get_usage in time")
        finally:
            _kill_group(process)


def summarize_usage(raw: dict[str, Any]) -> dict[str, Any]:
    """Reduce a get_usage payload to the fields the guard enforces."""
    limits = raw.get("rate_limits")
    if not raw.get("rate_limits_available") or not isinstance(limits, dict):
        raise RuntimeError("get_usage returned no plan rate limits")

    def window(name: str) -> dict[str, Any]:
        entry = limits.get(name) or {}
        if entry.get("utilization") is None:
            raise RuntimeError(f"get_usage has no {name} utilization")
        return {"utilization": float(entry["utilization"]), "resets_at": entry.get("resets_at")}

    extra = limits.get("extra_usage") or {}
    spend = limits.get("spend") or {}
    used = spend.get("used") or {}
    return {
        "checked_at": utc_now(),
        "subscription_type": raw.get("subscription_type"),
        "five_hour": window("five_hour"),
        "seven_day": window("seven_day"),
        "model_scoped": [
            {"display_name": entry.get("display_name"), "utilization": float(entry["utilization"]),
             "resets_at": entry.get("resets_at")}
            for entry in (limits.get("model_scoped") or []) if entry.get("utilization") is not None
        ],
        "usage_credits": {
            "enabled": bool(extra.get("is_enabled")) or bool(spend.get("enabled")),
            "user_disabled": extra.get("user_disabled"),
            "monthly_limit": extra.get("monthly_limit"),
            "spend_limit_reached": bool(extra.get("spend_limit_reached")),
            "used_minor": int(used.get("amount_minor") or 0),
            "currency": used.get("currency"),
            "exponent": int(used.get("exponent") or 2),
            "spend_limit": spend.get("limit"),
        },
    }


def account_violations(usage: dict[str, Any], args: argparse.Namespace, baseline_used_minor: int) -> list[str]:
    """Account-wide reasons to stop every arm; empty means proceed."""
    problems = []
    if usage.get("subscription_type") not in args.allowed_subscriptions:
        problems.append(f"subscription_type={usage.get('subscription_type')!r}")
    five_hour_remaining = 100.0 - usage["five_hour"]["utilization"]
    if five_hour_remaining <= args.min_five_hour_remaining:
        problems.append(f"five-hour remaining {five_hour_remaining:g}% <= {args.min_five_hour_remaining:g}%")
    weekly_remaining = 100.0 - usage["seven_day"]["utilization"]
    if weekly_remaining <= args.min_weekly_remaining:
        problems.append(f"weekly remaining {weekly_remaining:g}% <= {args.min_weekly_remaining:g}%")
    credits = usage["usage_credits"]
    if credits["enabled"] and not args.acknowledge_enabled_usage_credits:
        problems.append("usage credits are enabled on the account; refusing to risk credit spend")
    if credits["spend_limit_reached"]:
        problems.append("usage-credit spend limit reached")
    allowed_minor = round(args.max_usage_credit_spend_usd * 10 ** credits["exponent"])
    spent_minor = credits["used_minor"] - baseline_used_minor
    if spent_minor > allowed_minor:
        problems.append(f"usage credits spent during campaign {spent_minor} minor units > allowed {allowed_minor}")
    return problems


def scope_violations(usage: dict[str, Any], args: argparse.Namespace, scope: str | None) -> list[str]:
    """Model-scoped window (e.g. Fable weekly) problems that stop only that arm."""
    if scope is None:
        return []
    for entry in usage["model_scoped"]:
        if entry["display_name"] == scope:
            remaining = 100.0 - entry["utilization"]
            if remaining <= args.min_model_scoped_remaining:
                return [f"{scope} weekly remaining {remaining:g}% <= {args.min_model_scoped_remaining:g}%"]
    return []


def guard_violations(usage: dict[str, Any], args: argparse.Namespace, baseline_used_minor: int,
                     scope: str | None) -> list[str]:
    return account_violations(usage, args, baseline_used_minor) + scope_violations(usage, args, scope)


# ---------------------------------------------------------------- dry run
def protocol_checks(
    arm_results: list[dict[str, Any]], arms: list[dict[str, Any]],
    difficulties: list[str],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    def check(name: str, ok: bool, evidence: Any) -> None:
        checks[name] = {"pass": bool(ok), "evidence": evidence}

    def flag(command: list[str], name: str) -> str | None:
        return command[command.index(name) + 1] if name in command else None

    cells = [cell for result in arm_results for cell in result["cells"]]
    counts = {result["arm"]: result["cell_count"] for result in arm_results}
    expected = {arm["key"]: arm["expected_cells"] for arm in arms}
    check("cell_counts", counts == expected and len(cells) == sum(expected.values()),
          {"by_arm": counts, "total": len(cells), "expected": expected})
    check("unique_cell_dirs", len({cell["cell_dir"] for cell in cells}) == len(cells), len(cells))
    check("seed_0", all(cell["seed"] == 0 and flag(cell["runner_command"], "--seed") == "0" for cell in cells), "--seed 0")
    check("selected_difficulties", all(
        cell["difficulty"] in difficulties
        and flag(cell["runner_command"], "--difficulty") == cell["difficulty"]
        for cell in cells
    ) and {cell["difficulty"] for cell in cells} == set(difficulties),
          list(difficulties))
    effort_ok = mode_ok = True
    for result, arm in zip(arm_results, arms):
        arm_cells = result["cells"]
        effort_ok &= {cell["effort"] for cell in arm_cells} == set(arm["efforts"])
        effort_ok &= all(flag(cell["runner_command"], "--reasoning-effort") == cell["effort"] for cell in arm_cells)
        mode_ok &= {cell["env_mode"] for cell in arm_cells} == set(arm["env_modes"])
        mode_ok &= all(
            ("--static-thinking" in cell["runner_command"]) == (cell["env_mode"] == "static") for cell in arm_cells
        )
        mode_ok &= all(cell["model"] == arm["model"] for cell in arm_cells)
    check("explicit_efforts", effort_ok, {arm["key"]: list(arm["efforts"]) for arm in arms})
    check("static_realtime_semantics", mode_ok, {arm["key"]: list(arm["env_modes"]) for arm in arms} | {
        "static": "--static-thinking pauses UE during inference",
        "realtime": "UE advances concurrently during inference (SIMWORLD_CONCURRENT_REALTIME_INFERENCE=1)",
    })
    suites = [result["suite_manifest"] for result in arm_results]
    matrix_source = MATRIX_RUNNER.read_text(encoding="utf-8")
    agent_source = (REPO_ROOT / "base/rt_agent.py").read_text(encoding="utf-8")
    prompt_source = (REPO_ROOT / "llm/prompt.py").read_text(encoding="utf-8")
    check("realtime_concurrent_static_paused_sources", (
        'environment["SIMWORLD_CONCURRENT_REALTIME_INFERENCE"] = "1"' in matrix_source
        and "If realtime_thinking is False, environment is paused during thinking" in agent_source
    ), "run_openai_frontier_matrix.run_monitored + base/rt_agent.py step()")
    check("temporal_image_history_default", (
        all(suite["action_frame_mode"] == "video_history" for suite in suites)
        and all(flag(cell["runner_command"], "--action-frame-mode") == "video_history" for cell in cells)
        and "(approximately every 0.5s)" in prompt_source
    ), "ordered stills from the last action (~0.5 s cadence) + current annotated view, no image cap")
    check("no_thinking_or_output_cap", (
        all(set(suite["max_output_tokens_by_effort"].values()) == {None} for suite in suites)
        and all(flag(cell["runner_command"], "--max-output-tokens") == "none" for cell in cells)
        and not any("TOKENS" in name for name in FORCED_ENVIRONMENT)
        and ROUTE_OVERRIDE_PATTERN.match("MAX_THINKING_TOKENS") is not None
        and ROUTE_OVERRIDE_PATTERN.match("CLAUDE_CODE_MAX_OUTPUT_TOKENS") is not None
    ), "no benchmark cap; MAX_THINKING_TOKENS / CLAUDE_CODE_MAX_OUTPUT_TOKENS scrubbed; adaptive thinking")
    check("subscription_transport_only", all(
        flag(cell["runner_command"], "--provider") == "claude-code"
        and flag(cell["runner_command"], "--api-mode") == "claude_code"
        and "--key-file" not in cell["runner_command"]
        for cell in cells
    ), "--provider claude-code --api-mode claude_code, no key file")
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", TOUCHED_ROAD_COMMIT, "HEAD"],
                       cwd=REPO_ROOT, check=True, capture_output=True)
        has_commit = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        has_commit = False
    # A clean public repository intentionally does not retain private history.
    # Accept the pinned release source as an alternative to historical ancestry.
    release_manifest = REPO_ROOT / "benchmark" / "release_sources.json"
    release_match = False
    if release_manifest.is_file():
        expected = json.loads(release_manifest.read_text()).get("sha256", {})
        release_match = bool(expected) and all(
            (REPO_ROOT / name).is_file()
            and hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() == digest
            for name, digest in expected.items()
        )
    touched_road = re.search(
        r"illegal_crossing_detection_source = \(\s*'ue_touched_road'\s*if ue_unsafe_touched_road\s*else None",
        agent_source,
    )
    check("ue_only_touched_road_illegal_crossing", (
        (has_commit or release_match) and touched_road is not None
        and "Unreal's TouchedRoad state is the sole illegal-crossing authority" in agent_source
    ), {"historical_commit": TOUCHED_ROAD_COMMIT, "commit_in_history": has_commit, "pinned_release_sources_match": release_match})
    check("walk_entry_persists_through_flashing_clearance", (
        "'flashing_clearance_continuation_legal': True" in agent_source
        and "If WALK changes to FLASHING DON'T WALK or steady DON'T WALK after entry, continuing to the far" in prompt_source
        and "corrected crossing-continuation rule missing" in matrix_source
    ), "admitted-on-WALK crossings stay legal (CLEAR_ONLY); prompt rule; per-step validator")
    return checks


def build_dry_run_manifest(args: argparse.Namespace, *, with_preflight: bool = True) -> dict[str, Any]:
    arms = selected_arms(args)
    arm_results = []
    for index, arm in enumerate(arms):
        cells = matrix_json(arm, args, "cells", index)
        gpus, ports = arm_resources(args, index)
        arm_results.append({
            "arm": arm["key"],
            "model": arm["model"],
            "efforts": list(arm["efforts"]),
            "env_modes": list(arm["env_modes"]),
            "suite_dir": str(args.results_root / model_slug(arm["model"])),
            "gpus": gpus,
            "base_ports": ports,
            "stop_file": str(arm_stop_file(args, arm)),
            "cell_count": cells["cell_count"],
            "full_design_cells": FULL_DESIGN_CELLS[arm["key"]],
            "suite_manifest": cells["suite_manifest"],
            "cells": cells["cells"],
        })
    checks = protocol_checks(arm_results, arms, list(args.difficulties))
    manifest = {
        "schema_version": 2,
        "mode": "dry_run",
        "starts_rollouts": False,
        "created_at": utc_now(),
        "results_root": str(args.results_root),
        "selection": {"difficulties": list(args.difficulties),
                      "efforts": list(args.efforts), "env_modes": list(args.env_modes),
                      "task_ids": args.task_ids or "all 36"},
        "counts": {
            "by_arm": {result["arm"]: result["cell_count"] for result in arm_results},
            "total": sum(result["cell_count"] for result in arm_results),
        },
        "protocol_checks": checks,
        "all_checks_pass": all(item["pass"] for item in checks.values()),
        "topology": {
            "parallel_arms": args.parallel_arms,
            "workers_per_arm": args.workers,
            "ue_port_stride": args.ue_port_stride,
            "dynamic_queue": True,
        },
        "image_policy": (
            f"PIL frames sent as lossless WebP (pixel-identical to the API arms' PNG) so every image is "
            f"<= {CLI_IMAGE_BYTE_LIMIT} bytes, which Claude Code forwards unchanged; frames still over the "
            "limit fall back to high-quality lossy WebP and are flagged per image"
        ),
        "guard": guard_settings(args),
        "source": source_snapshot(),
        "arms": arm_results,
    }
    if with_preflight:
        manifest["cli_preflight"] = cli_preflight(args.claude_executable, tuple(args.allowed_subscriptions))
    if args.include_usage_snapshot:
        manifest["usage_snapshot"] = read_plan_usage(args.claude_executable, args.usage_query_timeout)
    return manifest


def guard_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "usage_source": "claude -p get_usage control request (read-only; no model turn)",
        "poll_seconds": args.usage_poll_seconds,
        "min_five_hour_remaining_percent": args.min_five_hour_remaining,
        "min_weekly_remaining_percent": args.min_weekly_remaining,
        "min_model_scoped_remaining_percent": args.min_model_scoped_remaining,
        "max_usage_credit_spend_usd": args.max_usage_credit_spend_usd,
        "acknowledge_enabled_usage_credits": args.acknowledge_enabled_usage_credits,
        "never_enables_usage_credits_or_raises_spend_limits": True,
        "per_decision_fail_closed": "backend raises and writes the arm stop file",
        "global_stop_file": str(global_stop_file(args)),
    }


# --------------------------------------------------------------------- run
def stop_process_group(process: subprocess.Popen, *, grace: float = 180.0) -> int:
    if process.poll() is not None:
        return int(process.returncode or 0)
    os.killpg(process.pid, signal.SIGINT)
    try:
        return process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return process.wait(timeout=30)


def write_stop(path: Path, reason: str, detail: Any) -> None:
    if not path.exists():
        write_json(path, {"time": utc_now(), "reason": reason, "detail": detail, "pid": os.getpid()})


def run_arms(args: argparse.Namespace, arms: list[dict[str, Any]], indices: list[int],
             baseline: int, events: Path) -> dict[str, int]:
    """Run the given arms' coordinators concurrently under one usage guard."""
    running: dict[str, tuple[dict[str, Any], subprocess.Popen, Any]] = {}
    for arm, index in zip(arms, indices):
        suite = args.results_root / model_slug(arm["model"])
        suite.mkdir(parents=True, exist_ok=True)
        command = matrix_command(arm, args, "coordinator", index)
        write_json(suite / "guarded_command.json", command)
        log = (suite / "guarded_coordinator.log").open("ab", buffering=0)
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=arm_environment(arm, args),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        running[arm["key"]] = (arm, process, log)
        append_jsonl(events, {"time": utc_now(), "event": "arm_start", "arm": arm["key"],
                              "pid": process.pid, "gpus": arm_resources(args, index)[0],
                              "ports": arm_resources(args, index)[1]})
    codes: dict[str, int] = {}
    guard_errors = 0

    def stop_arm(key: str, code: int, reason: str, detail: Any) -> None:
        arm, process, log = running.pop(key)
        write_stop(arm_stop_file(args, arm), reason, detail)
        append_jsonl(events, {"time": utc_now(), "event": "arm_stop", "arm": key, "reason": reason, "detail": detail})
        stop_process_group(process)
        log.close()
        codes[key] = code

    try:
        while running:
            deadline = time.monotonic() + args.usage_poll_seconds
            while running and time.monotonic() < deadline:
                for key in list(running):
                    arm, process, log = running[key]
                    if process.poll() is not None:
                        running.pop(key)
                        log.close()
                        codes[key] = int(process.returncode or 0)
                        append_jsonl(events, {"time": utc_now(), "event": "arm_exit", "arm": key,
                                              "return_code": codes[key]})
                    elif arm_stop_file(args, arm).exists():
                        stop_arm(key, 75, "arm_stop_file", json.loads(arm_stop_file(args, arm).read_text()))
                if global_stop_file(args).exists():
                    for key in list(running):
                        stop_arm(key, 75, "global_stop_file", str(global_stop_file(args)))
                time.sleep(2)
            if not running:
                break
            try:
                usage = read_plan_usage(args.claude_executable, args.usage_query_timeout)
                preflight = cli_preflight(args.claude_executable, tuple(args.allowed_subscriptions))
                guard_errors = 0
            except Exception as exc:  # the guard itself must fail closed
                guard_errors += 1
                append_jsonl(events, {"time": utc_now(), "event": "usage_guard_error",
                                      "consecutive_errors": guard_errors, "error": str(exc)})
                if guard_errors >= args.maximum_guard_errors:
                    write_stop(global_stop_file(args), "usage_guard_unavailable", str(exc))
                    for key in list(running):
                        stop_arm(key, 76, "usage_guard_unavailable", str(exc))
                continue
            write_json(args.results_root / "latest_usage.json", usage)
            append_jsonl(events, {"time": utc_now(), "event": "usage_check"} | usage)
            problems = account_violations(usage, args, baseline)
            if "auth" not in preflight:
                problems.append(f"auth route check failed: {preflight.get('auth_error')}")
            if problems:
                write_stop(global_stop_file(args), "account_guard", problems)
                for key in list(running):
                    stop_arm(key, 75, "account_guard", problems)
                break
            for key in list(running):
                scoped = scope_violations(usage, args, running[key][0]["usage_scope"])
                if scoped:
                    stop_arm(key, 75, "model_scope_guard", scoped)
    except KeyboardInterrupt:
        for key in list(running):
            stop_arm(key, 130, "keyboard_interrupt", None)
    return codes


def run(args: argparse.Namespace) -> int:
    if not args.confirm_claude_code_run:
        print("Refusing to start: pass --confirm-claude-code-run after reviewing the dry-run "
              "manifest and the current usage snapshot.", file=sys.stderr)
        return 2
    arms = selected_arms(args)
    stale = [str(path) for path in [global_stop_file(args), *(arm_stop_file(args, arm) for arm in arms)] if path.exists()]
    if stale:
        print(f"Refusing to start: stop file(s) present: {stale}", file=sys.stderr)
        return 75
    manifest = build_dry_run_manifest(args)
    if not manifest["all_checks_pass"]:
        failed = [name for name, item in manifest["protocol_checks"].items() if not item["pass"]]
        print(f"Refusing to start: protocol checks failed: {failed}", file=sys.stderr)
        return 2
    preflight = manifest["cli_preflight"]
    if "auth" not in preflight:
        print(f"Refusing to start: {preflight.get('auth_error')}", file=sys.stderr)
        return 2
    too_old = [arm["model"] for arm in arms if not preflight["meets_minimum"].get(arm["model"])]
    if too_old:
        print(f"Refusing to start: {args.claude_executable} is too old for {too_old}", file=sys.stderr)
        return 2
    args.results_root.mkdir(parents=True, exist_ok=True)
    events = args.results_root / "campaign_events.jsonl"
    # Campaigns for different arms may share a results root (Fable and Sonnet
    # low run concurrently), so per-campaign files carry the arm/effort tag.
    tag = campaign_tag(arms, args.efforts, args.difficulties)
    initial = read_plan_usage(args.claude_executable, args.usage_query_timeout)
    baseline = initial["usage_credits"]["used_minor"]
    write_json(args.results_root / f"usage_before_{tag}.json", initial)
    write_json(args.results_root / f"dry_run_manifest_{tag}.json", manifest)
    write_json(args.results_root / f"campaign_manifest_{tag}.json", {
        "created_at": utc_now(),
        "arms": [{key: arm[key] for key in ("key", "model", "efforts", "env_modes", "expected_cells")} for arm in arms],
        "parallel_arms": args.parallel_arms,
        "counts": manifest["counts"],
        "guard": guard_settings(args),
        "cli_preflight": preflight,
        "initial_usage": initial,
        "source": manifest["source"],
        "command_line": sys.argv,
    })
    append_jsonl(events, {"time": utc_now(), "event": "campaign_start", "tag": tag, "pid": os.getpid()} | initial)
    problems = account_violations(initial, args, baseline)
    if problems:
        write_stop(global_stop_file(args), "account_guard_before_start", problems)
        print(f"Refusing to start: {problems}", file=sys.stderr)
        return 75
    codes: dict[str, int] = {}
    if args.parallel_arms:
        startable = [(arm, index) for index, arm in enumerate(arms)
                     if not scope_violations(initial, args, arm["usage_scope"])]
        for arm in arms:
            if arm not in [item[0] for item in startable]:
                codes[arm["key"]] = 75
                append_jsonl(events, {"time": utc_now(), "event": "arm_skipped_scope_guard", "arm": arm["key"]})
        codes |= run_arms(args, [arm for arm, _ in startable], [index for _, index in startable], baseline, events)
    else:
        for index, arm in enumerate(arms):
            usage = read_plan_usage(args.claude_executable, args.usage_query_timeout)
            problems = guard_violations(usage, args, baseline, arm["usage_scope"])
            if problems:
                codes[arm["key"]] = 75
                append_jsonl(events, {"time": utc_now(), "event": "before_arm_stop", "arm": arm["key"],
                                      "problems": problems})
                continue
            codes |= run_arms(args, [arm], [index], baseline, events)
            if global_stop_file(args).exists():
                break
    final = read_plan_usage(args.claude_executable, args.usage_query_timeout)
    write_json(args.results_root / f"usage_after_{tag}.json", final)
    append_jsonl(events, {"time": utc_now(), "event": "campaign_end", "tag": tag, "arm_codes": codes} | final)
    print(json.dumps({"arm_return_codes": codes, "usage_before": initial, "usage_after": final}, indent=2))
    return 0 if codes and all(code == 0 for code in codes.values()) else max(codes.values(), default=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "run"), default="plan")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=None,
                        help="Dry-run manifest path (default: <results-root>/dry_run_manifest.json).")
    parser.add_argument("--arms", nargs="+", choices=[arm["key"] for arm in ARMS],
                        default=[arm["key"] for arm in ARMS])
    parser.add_argument("--difficulties", nargs="+", choices=ALL_DIFFICULTIES,
                        default=["easy"])
    parser.add_argument("--efforts", nargs="+", choices=ALL_EFFORTS, default=list(DEFAULT_EFFORTS))
    parser.add_argument("--env-modes", nargs="+", choices=("realtime", "static"),
                        default=["realtime", "static"],
                        help="Restrict arms to these modes (Fable is realtime-only by design).")
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--parallel-arms", action="store_true",
                        help="Run arms concurrently; arm i uses GPUs/ports [i*workers:(i+1)*workers].")
    parser.add_argument("--claude-executable", default=os.environ.get(EXECUTABLE_ENVIRONMENT, DEFAULT_EXECUTABLE))
    parser.add_argument("--allowed-subscriptions", nargs="+", default=list(DEFAULT_ALLOWED_SUBSCRIPTIONS))
    parser.add_argument("--confirm-claude-code-run", action="store_true")
    parser.add_argument("--include-usage-snapshot", action="store_true",
                        help="Add a read-only get_usage snapshot to the dry-run manifest.")
    parser.add_argument("--workers", type=int, default=2, help="UE workers per arm.")
    parser.add_argument("--gpus", nargs="+", type=int, default=[6, 7, 5, 0])
    parser.add_argument("--ports", nargs="+", type=int, default=[20000, 20500, 21000, 21500])
    parser.add_argument("--ue-port-stride", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    # 2026-09-19 17:45 UTC: 300 s, as used on serv4. On the loaded shared host a fresh UE
    # can need >120 s for "vrun Editor.AsyncSkinnedAssetCompilation", which killed 4 Fable
    # attempts at startup.
    parser.add_argument("--unrealcv-request-timeout", type=float, default=300.0)
    # Wall-clock watchdog per rollout only; the benchmark's limit is max_steps.
    # On the shared host a full-length hard rollout (up to ~560 steps at
    # 60-150 s/step) can exceed 4 h, and killing it discards a valid failure
    # episode and retries it, so allow 48 h. Hangs are caught separately by the
    # UnrealCV and API request timeouts.
    parser.add_argument("--cell-timeout", type=float, default=172800.0)
    parser.add_argument("--ue-ready-timeout", type=float, default=300.0)
    parser.add_argument("--ue-settle-seconds", type=float, default=30.0)
    parser.add_argument("--ue-start-stagger-seconds", type=float, default=30.0)
    parser.add_argument("--process-shutdown-timeout", type=float, default=20.0)
    parser.add_argument("--status-interval", type=float, default=30.0)
    parser.add_argument("--record-png-compress-level", type=int, default=3)
    parser.add_argument("--record-images-first-steps", type=int, default=3)
    parser.add_argument("--record-images-last-steps", type=int, default=3)
    # Guard thresholds are remaining allowance (100 - utilization).
    parser.add_argument("--min-five-hour-remaining", type=float, default=20.0)
    parser.add_argument("--min-weekly-remaining", type=float, default=10.0)
    parser.add_argument("--min-model-scoped-remaining", type=float, default=5.0)
    parser.add_argument("--max-usage-credit-spend-usd", type=float, default=0.0)
    parser.add_argument("--acknowledge-enabled-usage-credits", action="store_true",
                        help="Proceed even if usage credits are enabled (spend stays capped by --max-usage-credit-spend-usd).")
    parser.add_argument("--usage-poll-seconds", type=float, default=60.0)
    parser.add_argument("--usage-query-timeout", type=float, default=30.0)
    parser.add_argument("--maximum-guard-errors", type=int, default=3)
    args = parser.parse_args(argv)
    args.results_root = args.results_root.resolve()
    needed = args.workers * (len(args.arms) if args.parallel_arms else 1)
    if args.workers <= 0 or len(args.gpus) < needed or len(args.ports) < needed:
        parser.error(f"{needed} GPUs and base ports are required ({args.workers} per arm)")
    if len(set(args.gpus[:needed])) != needed or len(set(args.ports[:needed])) != needed:
        parser.error("GPUs and base ports must be unique per worker")
    for name in ("min_five_hour_remaining", "min_weekly_remaining", "min_model_scoped_remaining"):
        if not 0 <= getattr(args, name) < 100:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 100)")
    if args.max_usage_credit_spend_usd < 0:
        parser.error("--max-usage-credit-spend-usd cannot be negative")
    if args.usage_poll_seconds <= 0:
        parser.error("--usage-poll-seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "run":
        return run(args)
    manifest = build_dry_run_manifest(args)
    output = args.output or args.results_root / "dry_run_manifest.json"
    write_json(output, manifest)
    print(json.dumps({
        "dry_run_manifest": str(output),
        "counts": manifest["counts"],
        "all_checks_pass": manifest["all_checks_pass"],
        "failed_checks": [name for name, item in manifest["protocol_checks"].items() if not item["pass"]],
        "cli_preflight": manifest.get("cli_preflight"),
    }, indent=2))
    return 0 if manifest["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
