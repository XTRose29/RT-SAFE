#!/usr/bin/env python3
"""Run one SimWorld realtime task with an OpenAI vision-capable model."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import types
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = REPO_ROOT / ".py312deps"
if LOCAL_DEPS.is_dir() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid importing optional city-generation dependencies that the realtime
# benchmark does not use.
_simworld = types.ModuleType("simworld")
_simworld.__path__ = [str(REPO_ROOT / "SimWorld" / "simworld")]
sys.modules.setdefault("simworld", _simworld)

from utils.route_steps import route_length_to_max_steps
from utils.task_routes import reconstruct_route_points, route_length_cm
from llm.reasoning_capabilities import openrouter_reasoning_profile


DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
PRICING_USD_PER_MILLION_TOKENS = {
    # Standard short-context rates. Reasoning tokens are billed as output tokens.
    "gpt-6-astra": {
        "input": 10.00,
        "output": 50.00,
        "label": "GPT-6 Astra",
        "pricing_as_of": "2026-09-07",
        "source": "https://developers.openai.com/api/docs/models/gpt-6-astra",
    },
    "gpt-5.6": {
        "input": 4.00,
        "output": 20.00,
        "label": "GPT-5.6 Sol",
        "pricing_as_of": "2026-09-01",
        "source": PRICING_SOURCE,
    },
    "gpt-5.6-sol": {
        "input": 4.00,
        "output": 20.00,
        "label": "GPT-5.6 Sol",
        "pricing_as_of": "2026-09-01",
        "source": PRICING_SOURCE,
    },
    "gpt-5.6-terra": {
        "input": 2.00,
        "output": 12.00,
        "label": "GPT-5.6 Terra",
        "pricing_as_of": "2026-09-01",
        "source": PRICING_SOURCE,
    },
    "gpt-5.6-luna": {
        "input": 0.20,
        "output": 1.20,
        "label": "GPT-5.6 Luna",
        "pricing_as_of": "2026-09-01",
        "source": PRICING_SOURCE,
    },
    "anthropic/claude-haiku-4.5": {
        "input": 1.00,
        "output": 5.00,
        "label": "Claude Haiku 4.5 via OpenRouter",
        "pricing_as_of": "2026-09-05",
        "source": "https://openrouter.ai/anthropic/claude-haiku-4.5",
    },
    "gpt-realtime-2.1-mini": {
        "input": 0.80,
        "output": 2.40,
        "label": "GPT-Realtime-2.1 Mini (conservative image-input rate)",
        "pricing_as_of": "2026-09-05",
        "source": "https://developers.openai.com/api/docs/models/gpt-realtime-2.1-mini",
    },
}

# Signed-in CLI surfaces consume plan allowance, not per-token API billing.
SUBSCRIPTION_BILLING_MODES = {
    "codex-cli": "chatgpt_codex_credits",
    "claude-code": "claude_subscription_plan_limits",
}

OPENAI_ACTION_ONLY_PROMPT_SUFFIX = """\
OUTPUT CONTRACT (this overrides any earlier response-format instruction):
Return exactly these two lines and no other text:
Action: move_to|turn_around|wait
Param: 1-7 for move_to, L30/L60/L90/R30/R60/R90 for turn_around, or 1-3 for wait
Do not output reasoning, explanations, analysis, JSON, Markdown, or any additional fields."""


def load_api_key(
    key_file: Path, environment_variable: str = "OPENAI_API_KEY"
) -> str:
    """Read a raw key or NAME=... file without exposing its value."""
    try:
        raw = key_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(
            f"API key file not found: {key_file}. "
            f"Create it with one raw key line or {environment_variable}=<key>."
        ) from exc
    except OSError as exc:
        raise ValueError(f"Could not read API key file {key_file}: {exc}") from exc

    accepted_prefixes = (
        f"{environment_variable}=",
        "OPENAI_API_KEY=",
        "OPENROUTER_API_KEY=",
    )
    matching_prefix = next(
        (prefix for prefix in accepted_prefixes if raw.startswith(prefix)), None
    )
    if matching_prefix:
        raw = raw.split("=", 1)[1].strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
            raw = raw[1:-1]
    if not raw:
        raise ValueError(f"API key file is empty: {key_file}")
    if any(char.isspace() for char in raw):
        raise ValueError("API key must be a single value without whitespace")
    return raw


def resolve_api_key(
    key_file: Path, environment_variable: str = "OPENAI_API_KEY"
) -> str:
    """Prefer the process environment, then fall back to the ignored key file."""
    environment_key = os.environ.get(environment_variable, "").strip()
    if environment_key:
        if any(char.isspace() for char in environment_key):
            raise ValueError(
                f"{environment_variable} must be a single value without whitespace"
            )
        return environment_key
    return load_api_key(key_file, environment_variable)


def build_llm_config(args: argparse.Namespace) -> dict:
    reasoning_effort = args.reasoning_effort
    prompt_suffix = (
        OPENAI_ACTION_ONLY_PROMPT_SUFFIX
        if args.prompt_style == "openai_action_only"
        else None
    )
    return {
        "llm": [
            {
                "model": args.model,
                "provider": args.provider,
                "url": args.base_url,
                "api_mode": args.api_mode,
                "reasoning": reasoning_effort != "none",
                "reasoning_effort": reasoning_effort,
                "max_tokens": args.max_output_tokens,
                "image_detail": args.image_detail,
                "text_verbosity": args.text_verbosity,
                "service_tier": args.service_tier,
                "store": False,
                "request_timeout": args.request_timeout,
                # Provider/API failures are infrastructure events, not model
                # actions. Let the matrix retry the entire isolated rollout.
                "raise_on_api_error": True,
                # The instructional prompt intentionally exposes its concise
                # Reasoning line. Only the action-only condition may override
                # that response contract.
                "prompt_suffix": prompt_suffix,
            }
        ]
    }


def resolve_max_steps(args: argparse.Namespace) -> int:
    """Resolve the explicit cap or compute three steps per full route meter."""
    if args.max_steps > 0:
        return args.max_steps

    task_file = Path(args.task_file)
    if not task_file.is_absolute():
        task_file = REPO_ROOT / task_file
    try:
        task_data = json.loads(task_file.read_text(encoding="utf-8"))
        task = task_data["tasks"][args.task_index]
        # Keep the preflight cap aligned with WorldManager execution and SPL.
        # Generated route_info paths can be stale after task edges are edited.
        points = reconstruct_route_points(task)
        path_length_cm = route_length_cm(points)
    except FileNotFoundError as exc:
        raise ValueError(f"Task file not found: {task_file}") from exc
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not compute max steps for task index {args.task_index} from {task_file}: {exc}"
        ) from exc
    if len(points) < 2:
        raise ValueError(f"Task index {args.task_index} has fewer than two route points")
    return route_length_to_max_steps(path_length_cm)


def _catalog_pricing(model: str) -> dict[str, Any] | None:
    """Resolve aliases and dated snapshots without guessing across model families."""
    normalized = model.lower()
    if normalized in PRICING_USD_PER_MILLION_TOKENS:
        return dict(PRICING_USD_PER_MILLION_TOKENS[normalized])
    for family in (
        "gpt-6-astra",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    ):
        if normalized.startswith(f"{family}-"):
            return dict(PRICING_USD_PER_MILLION_TOKENS[family])
    return None


def build_cost_estimate(args: argparse.Namespace) -> dict[str, Any]:
    """Build a local planning estimate; this function never calls an API."""
    steps = args.expected_steps if args.expected_steps is not None else args.max_steps
    input_tokens = steps * args.estimated_input_tokens_per_step
    output_tokens = steps * args.estimated_output_tokens_per_step
    output_cap_tokens = (
        steps * args.max_output_tokens if args.max_output_tokens is not None else None
    )

    billing_mode = SUBSCRIPTION_BILLING_MODES.get(args.provider, "api_usage_based")
    pricing = None
    pricing_source = None
    if args.provider in SUBSCRIPTION_BILLING_MODES:
        # Codex exec / Claude Code add their own model-visible instructions
        # and consume the signed-in account's plan allowance. API-dollar estimates
        # are therefore inapplicable and would understate the input materially.
        pricing = None
    elif args.input_price_per_million is not None:
        pricing = {
            "input": args.input_price_per_million,
            "output": args.output_price_per_million,
            "label": "CLI override",
        }
        pricing_source = "CLI override"
    else:
        pricing = _catalog_pricing(args.model)
        if pricing is not None:
            pricing_source = pricing.get("source", PRICING_SOURCE)

    estimate: dict[str, Any] = {
        "model": args.model,
        "provider": args.provider,
        "billing_mode": billing_mode,
        "expected_steps": steps,
        "estimated_input_tokens_per_step": args.estimated_input_tokens_per_step,
        "estimated_output_tokens_per_step": args.estimated_output_tokens_per_step,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens": input_tokens + output_tokens,
        "configured_output_cap_tokens": output_cap_tokens,
        "pricing_as_of": pricing.get("pricing_as_of") if pricing else None,
        "pricing_source": pricing_source,
        "input_price_usd_per_million_tokens": pricing["input"] if pricing else None,
        "output_price_usd_per_million_tokens": pricing["output"] if pricing else None,
        "estimated_cost_usd": None,
        "output_cap_scenario_cost_usd": None,
    }
    if pricing is not None:
        input_cost = input_tokens * pricing["input"] / 1_000_000
        estimate["estimated_cost_usd"] = (
            input_cost + output_tokens * pricing["output"] / 1_000_000
        )
        if output_cap_tokens is not None:
            estimate["output_cap_scenario_cost_usd"] = (
                input_cost + output_cap_tokens * pricing["output"] / 1_000_000
            )
    return estimate


def print_cost_estimate(estimate: dict[str, Any]) -> None:
    """Print a human-readable estimate before any paid or simulator work begins."""
    cost = estimate["estimated_cost_usd"]
    cap_cost = estimate["output_cap_scenario_cost_usd"]
    label = {
        "chatgpt_codex_credits": "Codex CLI pilot",
        "claude_subscription_plan_limits": "Claude Code subscription arm",
    }.get(estimate["billing_mode"], "API benchmark")
    no_call_label = (
        "no model call made"
        if estimate["billing_mode"] in SUBSCRIPTION_BILLING_MODES.values()
        else "no API call made"
    )
    print(f"=== SimWorld {label} preflight estimate ({no_call_label}) ===")
    print(f"Model: {estimate['model']}")
    print(f"Expected steps: {estimate['expected_steps']}")
    print(
        "Planned tokens: "
        f"{estimate['estimated_input_tokens']:,} input + "
        f"{estimate['estimated_output_tokens']:,} output = "
        f"{estimate['estimated_total_tokens']:,} total"
    )
    print(
        "Per-step assumptions: "
        f"{estimate['estimated_input_tokens_per_step']:,} input, "
        f"{estimate['estimated_output_tokens_per_step']:,} output"
    )
    if estimate["configured_output_cap_tokens"] is None:
        print("Configured output-cap scenario: omitted (provider/model limits still apply)")
    else:
        print(
            f"Configured output-cap scenario: "
            f"{estimate['configured_output_cap_tokens']:,} output tokens"
        )
    if cost is None:
        if estimate["billing_mode"] == "chatgpt_codex_credits":
            print("Estimated API cost: not applicable (uses signed-in Codex credits/allowance)")
            print("Runtime Codex token usage is recorded for every decision.")
        elif estimate["billing_mode"] == "claude_subscription_plan_limits":
            print("Estimated API cost: not applicable (uses Claude subscription plan limits)")
            print("Runtime Claude Code token usage is recorded for every decision.")
        else:
            print("Estimated API cost: unavailable (model price is not in the local catalog)")
            print(
                "Set both --input-price-per-million and --output-price-per-million "
                "to calculate dollars."
            )
    else:
        print(
            "Pricing: "
            f"${estimate['input_price_usd_per_million_tokens']:g}/M input, "
            f"${estimate['output_price_usd_per_million_tokens']:g}/M output"
        )
        print(f"Estimated API cost: ${cost:.4f}")
        if cap_cost is not None:
            print(f"Output-cap scenario cost: ${cap_cost:.4f}")
    print(
        "Note: image/text input and reasoning usage vary at runtime; reasoning tokens "
        "are included in output-token billing. The output-cap scenario is not a total-cost guarantee."
    )
    print("================================================================")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI API model ID to evaluate (default: {DEFAULT_OPENAI_MODEL}).",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "openrouter", "codex-cli", "claude-code"),
        default="openai",
        help="API provider (default: openai).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL (OpenRouter defaults to https://openrouter.ai/api/v1).",
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        default=None,
        help="File containing a raw API key or matching environment assignment.",
    )
    parser.add_argument(
        "--api-mode",
        choices=("responses", "chat_completions", "realtime", "codex_cli", "claude_code"),
        default="responses",
        help="OpenAI endpoint adapter. Responses is the default for current reasoning models.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max", "default"),
        default="none",
        help=(
            "Reasoning effort. 'default' deliberately omits reasoning.effort; "
            "GPT-5.6 Luna currently resolves that omission to medium."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        default="128",
        help=(
            "Per-call output-token cap, or 'none' to omit max_output_tokens from the "
            "API request (provider/model limits still apply)."
        ),
    )
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    parser.add_argument("--text-verbosity", choices=("low", "medium", "high"), default="low")
    parser.add_argument(
        "--service-tier",
        choices=("auto", "default", "flex", "priority"),
        default="default",
        help="Processing tier (default: standard/default, matching the cost estimate).",
    )
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument(
        "--claude-code-executable",
        default=None,
        help=(
            "Claude Code CLI for --provider claude-code "
            "(default: $SIMWORLD_CLAUDE_CODE_EXECUTABLE, else claude on PATH)."
        ),
    )

    cost_group = parser.add_argument_group("preflight token and cost estimate")
    cost_group.add_argument(
        "--expected-steps",
        type=int,
        default=None,
        help="Steps used in the estimate (default: --max-steps).",
    )
    cost_group.add_argument(
        "--estimated-input-tokens-per-step",
        type=int,
        default=2000,
        help="Planning assumption including prompt text and rendered image input (default: 2000).",
    )
    cost_group.add_argument(
        "--estimated-output-tokens-per-step",
        type=int,
        default=32,
        help=(
            "Planning assumption including visible and reasoning output (default: 32; "
            "prior Luna action-only rollouts averaged about 14)."
        ),
    )
    cost_group.add_argument(
        "--input-price-per-million",
        type=float,
        default=None,
        help="Override input price in USD per million tokens; must be paired with output price.",
    )
    cost_group.add_argument(
        "--output-price-per-million",
        type=float,
        default=None,
        help="Override output price in USD per million tokens; must be paired with input price.",
    )
    cost_group.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print the preflight estimate and exit without reading the key or connecting to UE.",
    )

    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help=(
            "Decision-step cap. Default -1 allows three decisions per full meter of shortest-route length; "
            "a positive value overrides it."
        ),
    )
    parser.add_argument("--ue-ip", default="127.0.0.1")
    parser.add_argument("--ue-port", type=int, default=9000)
    parser.add_argument(
        "--prompt-style",
        default="openai_action_only",
        help=(
            "System-prompt variant (default: openai_action_only, which permits hidden "
            "reasoning but emits only Action and Param)."
        ),
    )
    parser.add_argument(
        "--traffic-policy",
        choices=("visual_only", "safety_assisted"),
        default="visual_only",
        help=(
            "visual_only keeps traffic visible and evaluated without symbolic "
            "traffic prompt fields or crossing action overrides; "
            "safety_assisted retains the earlier assisted behavior."
        ),
    )
    parser.add_argument(
        "--red-light-conflict-vehicle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Launch a collision-enabled conflict vehicle after an observed "
            "red-light crossing violation (default: enabled)."
        ),
    )
    parser.add_argument(
        "--static-signal-vehicles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep ordinary signal-controlled background cars upright and "
            "non-colliding until one is selected as a benchmark conflict "
            "vehicle (maintained easy/realtime default: enabled)."
        ),
    )
    parser.add_argument(
        "--red-light-conflict-launch-distance-min-cm",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--red-light-conflict-launch-distance-max-cm",
        type=float,
        default=900.0,
    )
    parser.add_argument(
        "--red-light-conflict-collision-radius-cm",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--red-light-conflict-vehicle-probability",
        type=float,
        default=None,
        help=(
            "Override the maintained 1.0 launch probability. Values below "
            "1.0 are explicit ablations."
        ),
    )
    parser.add_argument("--red-light-conflict-nominal-speed-cm-s", type=float, default=450.0)
    parser.add_argument(
        "--pedestrian-signal-compliance-probability",
        type=float,
        default=1.0,
    )
    parser.add_argument("--static-thinking", action="store_true")
    parser.add_argument(
        "--token-based",
        action="store_true",
        help="Advance real-time thinking by 0.02 * output_tokens + 1.0 seconds instead of API latency.",
    )
    parser.add_argument("--async-time", action="store_true")
    parser.add_argument(
        "--action-frame-mode",
        choices=("image_history", "video_history"),
        default="image_history",
    )
    parser.add_argument("--no-record-per-step", action="store_true")
    parser.add_argument("--record-png-compress-level", type=int, default=0)
    parser.add_argument("--record-images-first-steps", type=int, default=3)
    parser.add_argument("--record-images-last-steps", type=int, default=3)
    parser.add_argument(
        "--record-output-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retain annotated post-action images in audited steps.",
    )
    parser.add_argument(
        "--record-demo-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retain unannotated demo images in audited steps.",
    )
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the secret-free resolved configuration without reading the key or connecting to UE.",
    )
    args = parser.parse_args(argv)
    if args.provider == "openrouter":
        if args.api_mode != "chat_completions":
            parser.error("OpenRouter rollouts currently require --api-mode chat_completions")
        if args.base_url is None:
            args.base_url = DEFAULT_OPENROUTER_URL
        if args.key_file is None:
            args.key_file = REPO_ROOT.parent / "openrouter_api.txt"
    elif args.provider == "codex-cli":
        if args.api_mode != "codex_cli":
            parser.error("Codex CLI rollouts require --api-mode codex_cli")
        if args.reasoning_effort == "none":
            parser.error("GPT-5.6 Sol in Codex CLI does not support effort 'none'")
    elif args.provider == "claude-code":
        if args.api_mode != "claude_code":
            parser.error("Claude Code rollouts require --api-mode claude_code")
        if args.reasoning_effort in {"none", "default"}:
            parser.error(
                "Claude Code rollouts require an explicit --reasoning-effort; "
                "omitting --effort silently applies the CLI default"
            )
    elif args.key_file is None:
        args.key_file = REPO_ROOT / "openai_api.txt"
    if str(args.max_output_tokens).lower() in {"none", "null", "omit"}:
        args.max_output_tokens = None
    else:
        try:
            args.max_output_tokens = int(args.max_output_tokens)
        except (TypeError, ValueError):
            parser.error("--max-output-tokens must be a positive integer or 'none'")
        if args.max_output_tokens <= 0:
            parser.error("--max-output-tokens must be positive or 'none'")
    if args.provider == "claude-code" and args.max_output_tokens is not None:
        parser.error(
            "Claude Code rollouts impose no output/thinking cap; "
            "use --max-output-tokens none"
        )
    if (
        args.provider == "openrouter"
        and "claude-haiku-4.5" in args.model.lower()
        and args.reasoning_effort != "none"
        and args.max_output_tokens is not None
        and args.max_output_tokens <= 1024
    ):
        parser.error(
            "Claude reasoning through OpenRouter requires --max-output-tokens greater "
            "than its 1024-token minimum thinking budget; use 1280 or higher"
        )
    if args.provider == "openrouter":
        reasoning_profile = openrouter_reasoning_profile(args.model)
        if reasoning_profile is not None and args.reasoning_effort != "default":
            supported_efforts = set(reasoning_profile["efforts"])
            if args.reasoning_effort not in supported_efforts:
                supported = ", ".join(reasoning_profile["efforts"])
                parser.error(
                    f"{args.model} does not support reasoning effort "
                    f"{args.reasoning_effort!r}; choose one of: {supported}, default"
                )
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 (automatic) or positive")
    if args.model.lower().startswith("gpt-6-astra") and args.reasoning_effort == "none":
        parser.error(
            "GPT-6 Astra does not support reasoning effort 'none'; use low, medium, "
            "high, xhigh, max, or default"
        )
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if not 0 <= args.record_png_compress_level <= 9:
        parser.error("--record-png-compress-level must be in [0, 9]")
    if args.record_images_first_steps < 0 or args.record_images_last_steps < 0:
        parser.error("image retention counts must be nonnegative")
    if args.expected_steps is not None and args.expected_steps <= 0:
        parser.error("--expected-steps must be positive")
    if args.estimated_input_tokens_per_step <= 0:
        parser.error("--estimated-input-tokens-per-step must be positive")
    if args.estimated_output_tokens_per_step <= 0:
        parser.error("--estimated-output-tokens-per-step must be positive")
    price_overrides = (args.input_price_per_million, args.output_price_per_million)
    if (price_overrides[0] is None) != (price_overrides[1] is None):
        parser.error("input and output price overrides must be provided together")
    if any(price is not None and price < 0 for price in price_overrides):
        parser.error("price overrides cannot be negative")
    return args


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "openai_model"


def cleanup_runtime(communicator, manager, *, skip_cleanup: bool) -> None:
    """Best-effort teardown that preserves the original rollout failure.

    A UE crash closes UnrealCV before ``WorldManager.cleanup`` can remove its
    actors. Teardown must not replace the useful rollout exception with a
    second socket error; a fresh UE process owns a fresh world on the retry.
    """
    if communicator is None:
        return
    try:
        if skip_cleanup or manager is None:
            communicator.unrealcv.disconnect()
        else:
            manager.cleanup()
    except Exception as exc:
        print(
            f"Warning: UE cleanup was incomplete after rollout termination: {exc}",
            file=sys.stderr,
        )
        try:
            communicator.unrealcv.disconnect()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        args.max_steps = resolve_max_steps(args)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    cost_estimate = build_cost_estimate(args)
    print_cost_estimate(cost_estimate)
    if args.estimate_only:
        return 0

    llm_config = build_llm_config(args)
    results_dir = args.results_dir or (
        f"results/{args.provider}_{_safe_model_name(args.model)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    resolved = {
        "llm": llm_config["llm"][0],
        "task_file": args.task_file,
        "task_index": args.task_index,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "realtime_thinking": not args.static_thinking,
        "token_based": args.token_based,
        "use_tick": not args.async_time,
        "action_frame_mode": args.action_frame_mode,
        "prompt_style": args.prompt_style,
        "traffic_policy": args.traffic_policy,
        "red_light_conflict_vehicle_enabled": (
            args.red_light_conflict_vehicle
        ),
        "static_signal_vehicles": args.static_signal_vehicles,
        "results_dir": results_dir,
        "artifact_policy": {
            "record_per_step": not args.no_record_per_step,
            "record_png_compress_level": args.record_png_compress_level,
            "record_images_first_steps": args.record_images_first_steps,
            "record_images_last_steps": args.record_images_last_steps,
            "record_output_images": args.record_output_images,
            "record_demo_images": args.record_demo_images,
        },
        "preflight_cost_estimate": cost_estimate,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return 0

    if args.claude_code_executable:
        os.environ["SIMWORLD_CLAUDE_CODE_EXECUTABLE"] = args.claude_code_executable
    credential_environment = None
    api_key = None
    if args.provider not in SUBSCRIPTION_BILLING_MODES:
        credential_environment = (
            "OPENROUTER_API_KEY"
            if args.provider == "openrouter"
            else "OPENAI_API_KEY"
        )
        try:
            api_key = resolve_api_key(
                args.key_file.resolve(), credential_environment
            )
        except ValueError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "preflight_cost_estimate.json").write_text(
        json.dumps(cost_estimate, indent=2) + "\n", encoding="utf-8"
    )

    artifact_environment = {
        "SIMWORLD_RECORD_PNG_COMPRESS_LEVEL": str(args.record_png_compress_level),
        "SIMWORLD_RECORD_IMAGES_FIRST_STEPS": str(
            args.record_images_first_steps
        ),
        "SIMWORLD_RECORD_IMAGES_LAST_STEPS": str(
            args.record_images_last_steps
        ),
        "SIMWORLD_RECORD_OUTPUT_IMAGES": (
            "1" if args.record_output_images else "0"
        ),
        "SIMWORLD_RECORD_DEMO_IMAGES": (
            "1" if args.record_demo_images else "0"
        ),
    }
    previous_artifact_environment = {
        name: os.environ.get(name) for name in artifact_environment
    }
    os.environ.update(artifact_environment)
    previous_api_key = (
        os.environ.get(credential_environment) if credential_environment else None
    )
    if credential_environment:
        os.environ[credential_environment] = api_key
    communicator = None
    manager = None
    try:
        from base.rt_communicator import RTCommunicator
        from base.rt_unrealcv import RTUnrealCV
        from manager.world_manager import WorldManager

        with tempfile.TemporaryDirectory(prefix=f"simworld_{args.provider}_") as temp_dir:
            agent_config_path = Path(temp_dir) / "agent.json"
            agent_config_path.write_text(json.dumps(llm_config), encoding="utf-8")
            communicator = RTCommunicator(RTUnrealCV(port=args.ue_port, ip=args.ue_ip))
            manager = WorldManager(
                communicator=communicator,
                agent_path=str(agent_config_path),
                task_file_path=args.task_file,
                seed=args.seed,
                control_mode="llm",
                token_based=args.token_based,
                use_tick=not args.async_time,
                realtime_thinking=not args.static_thinking,
                record_per_step=not args.no_record_per_step,
                use_action_frames=args.action_frame_mode == "video_history",
                prompt_style=args.prompt_style,
                traffic_policy=args.traffic_policy,
                red_light_conflict_vehicle_enabled=(
                    args.red_light_conflict_vehicle
                ),
                static_signal_vehicles=args.static_signal_vehicles,
                red_light_conflict_launch_distance_min_cm=(
                    args.red_light_conflict_launch_distance_min_cm
                ),
                red_light_conflict_launch_distance_max_cm=(
                    args.red_light_conflict_launch_distance_max_cm
                ),
                red_light_conflict_collision_radius_cm=(
                    args.red_light_conflict_collision_radius_cm
                ),
                red_light_conflict_vehicle_probability=(
                    args.red_light_conflict_vehicle_probability
                ),
                red_light_conflict_nominal_speed_cm_s=(
                    args.red_light_conflict_nominal_speed_cm_s
                ),
                pedestrian_signal_compliance_probability=(
                    args.pedestrian_signal_compliance_probability
                ),
                max_steps=args.max_steps,
                results_dir=results_dir,
            )
            manager._batch_suffix = (
                f"d{args.difficulty}_{args.provider}_{_safe_model_name(args.model)}_"
                f"{args.api_mode}_{args.reasoning_effort}"
            )
            manager.run_single_task(args.task_index, difficulty=args.difficulty)
            return 0 if manager.agent.success else 2
    finally:
        if credential_environment:
            if previous_api_key is None:
                os.environ.pop(credential_environment, None)
            else:
                os.environ[credential_environment] = previous_api_key
        for name, previous in previous_artifact_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        cleanup_runtime(
            communicator,
            manager,
            skip_cleanup=args.skip_cleanup,
        )


if __name__ == "__main__":
    raise SystemExit(main())
