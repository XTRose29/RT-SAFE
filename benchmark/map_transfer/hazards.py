"""Explicit hazard dynamics for the native transfer profile.

Parameters match the current Python source (trip recovery 6 s; next movement
on oil at half speed; water recorded without a destination perturbation).
Detection remains native continuous overlap, unlike paper endpoint polling.
"""

from __future__ import annotations


class HazardEffects:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.oil_next_move = False
        self.recovery_until = 0.0
        self.history = []

    def enter(self, kind: str, now: float) -> dict:
        event = {"kind": kind, "at_simulation_time": now}
        if kind == "oil":
            self.oil_next_move = True
            event["effect"] = "next_move_slowdown"
            event["speed_factor"] = self.config.get("oil_speed_factor", 0.5)
        elif kind == "trip":
            seconds = self.config.get("trip_recovery_seconds", 6.0)
            self.recovery_until = max(now, self.recovery_until) + seconds
            event.update(effect="recovery_delay", seconds=seconds)
        elif kind == "water":
            event["effect"] = "recorded_without_motion_perturbation"
        else:
            raise ValueError("Unknown environmental hazard")
        self.history.append(event)
        return event

    def start_move(self) -> float:
        factor = self.config.get("oil_speed_factor", 0.5) if self.oil_next_move else 1.0
        self.oil_next_move = False
        return factor

    def remaining(self, now: float) -> float:
        return max(0.0, self.recovery_until - now)
