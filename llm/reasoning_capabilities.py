"""Pinned reasoning controls for model IDs used by the benchmark.

OpenRouter exposes a normalized ``reasoning.effort`` field, but each model has
its own allowed effort set and some models do not permit reasoning to be
disabled.  Keeping the small benchmark matrix here prevents OpenRouter from
silently mapping an unsupported requested effort to a different level.
"""

from __future__ import annotations

from typing import Any


# Verified against OpenRouter's ``GET /api/v1/models`` catalog on 2026-09-13.
OPENROUTER_REASONING_PROFILES: dict[str, dict[str, Any]] = {
    "anthropic/claude-fable-5.1": {
        "efforts": ("low", "medium", "high", "xhigh", "max"),
        "default": "high",
        "mandatory": True,
    },
    "anthropic/claude-sonnet-5": {
        "efforts": ("none", "low", "medium", "high", "xhigh", "max"),
        "default": "high",
        "mandatory": False,
    },
    "google/gemini-3.8-flash": {
        "efforts": ("low", "medium", "high"),
        "default": "medium",
        "mandatory": True,
    },
    "moonshotai/kimi-k3": {
        "efforts": ("none", "low", "high", "max"),
        "default": "max",
        "mandatory": False,
    },
    "deepseek/deepseek-v4-flash-vision-exp": {
        "efforts": ("none", "low", "high", "max"),
        "default": "high",
        "mandatory": False,
    },
    "thinkingmachines/inkling": {
        "efforts": ("none", "minimal", "low", "medium", "high", "max"),
        "default": "high",
        "mandatory": False,
    },
    "z-ai/glm-5.3-flash": {
        "efforts": ("low", "high", "max"),
        "default": "max",
        "mandatory": True,
    },
    "x-ai/grok-4.6": {
        "efforts": ("low", "medium", "high", "xhigh"),
        "default": "high",
        "mandatory": True,
    },
}


def openrouter_reasoning_profile(model_name: str) -> dict[str, Any] | None:
    """Return a copied profile for an exact ID or its pinned dated snapshot."""
    normalized = model_name.lower()
    for model_id, profile in OPENROUTER_REASONING_PROFILES.items():
        if normalized == model_id or normalized.startswith(f"{model_id}-"):
            return {
                "efforts": tuple(profile["efforts"]),
                "default": profile["default"],
                "mandatory": bool(profile["mandatory"]),
            }
    return None
