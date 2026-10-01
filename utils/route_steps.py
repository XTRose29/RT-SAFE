"""Shared route-length-to-step-budget policy."""

from __future__ import annotations


ROUTE_CM_PER_BASE_STEP = 100.0
ROUTE_STEP_MULTIPLIER = 3


def route_length_to_max_steps(path_length_cm: float) -> int:
    """Allow three decision steps for every full meter of shortest-route length."""
    return max(1, ROUTE_STEP_MULTIPLIER * int(path_length_cm / ROUTE_CM_PER_BASE_STEP))
