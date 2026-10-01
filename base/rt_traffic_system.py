"""Intersection-level traffic-system state shared by UE, rendering, and prompts.

The packaged RT Blueprint currently exposes only two booleans per signal
(``vehicle_green`` and ``pedestrian_walk``).  This module deliberately models
the richer real-world phase plan while also recording which phases cannot yet
be distinguished through the Blueprint API (yellow and pedestrian clearance).
"""

from dataclasses import asdict, dataclass, replace
from enum import Enum
import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional


class IntersectionPhase(str, Enum):
    VEHICLE_GREEN = "VEHICLE_GREEN"
    VEHICLE_YELLOW = "VEHICLE_YELLOW"
    ALL_RED = "ALL_RED"
    PEDESTRIAN_WALK = "PEDESTRIAN_WALK"
    PEDESTRIAN_CLEARANCE = "PEDESTRIAN_CLEARANCE"
    CONFLICT = "CONFLICT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class TrafficPhaseTiming:
    """Configurable timings for one real-world intersection phase plan."""

    vehicle_green_s: float = 10.0
    vehicle_yellow_s: float = 2.0
    all_red_s: float = 1.0
    pedestrian_walk_s: float = 20.0
    pedestrian_clearance_s: float = 5.0
    pedestrian_walking_speed_mps: float = 1.067
    minimum_pedestrian_walk_s: float = 7.0
    agent_crossing_speed_mps: float = 2.0
    # Aggregate budget for all VLM decisions made during one crossing, rather
    # than the latency of a single request.  A 14 m crossing typically needs
    # several waypoint decisions in the realtime benchmark.
    vlm_latency_budget_s: float = 40.0
    action_execution_budget_s: float = 10.0
    crossing_safety_buffer_s: float = 5.0
    # A newly released WALK phase can begin while the agent is finishing its
    # previous observation/action cycle.  Reserve one bounded control-loop
    # interval so the first synchronized prompt can still admit a crossing;
    # otherwise a WALK calibrated to exactly the crossing budget is usable
    # only at the mathematically exact phase boundary.
    phase_detection_budget_s: float = 10.0

    @classmethod
    def from_config(cls, config) -> "TrafficPhaseTiming":
        def get(key: str, default: float) -> float:
            try:
                return float(config.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        legacy_vehicle_green = get(
            "traffic.traffic_signal.green_light_duration",
            cls.vehicle_green_s,
        )
        legacy_vehicle_yellow = get(
            "traffic.traffic_signal.yellow_light_duration",
            cls.vehicle_yellow_s,
        )
        legacy_pedestrian_green = get(
            "traffic.traffic_signal.pedestrian_green_light_duration",
            cls.pedestrian_walk_s + cls.pedestrian_clearance_s,
        )
        clearance = get(
            "traffic.traffic_signal.pedestrian_clearance_duration",
            cls.pedestrian_clearance_s,
        )
        walk_default = max(0.0, legacy_pedestrian_green - clearance)

        timing = cls(
            vehicle_green_s=get(
                "traffic.traffic_signal.vehicle_green_duration",
                legacy_vehicle_green,
            ),
            vehicle_yellow_s=get(
                "traffic.traffic_signal.vehicle_yellow_duration",
                legacy_vehicle_yellow,
            ),
            all_red_s=get(
                "traffic.traffic_signal.all_red_duration",
                cls.all_red_s,
            ),
            pedestrian_walk_s=get(
                "traffic.traffic_signal.pedestrian_walk_duration",
                walk_default,
            ),
            pedestrian_clearance_s=clearance,
            pedestrian_walking_speed_mps=get(
                "traffic.traffic_signal.pedestrian_walking_speed_mps",
                cls.pedestrian_walking_speed_mps,
            ),
            minimum_pedestrian_walk_s=get(
                "traffic.traffic_signal.minimum_pedestrian_walk_duration",
                cls.minimum_pedestrian_walk_s,
            ),
            agent_crossing_speed_mps=get(
                "traffic.traffic_signal.agent_crossing_speed_mps",
                cls.agent_crossing_speed_mps,
            ),
            vlm_latency_budget_s=get(
                "traffic.traffic_signal.vlm_latency_budget_s",
                cls.vlm_latency_budget_s,
            ),
            action_execution_budget_s=get(
                "traffic.traffic_signal.action_execution_budget_s",
                cls.action_execution_budget_s,
            ),
            crossing_safety_buffer_s=get(
                "traffic.traffic_signal.crossing_safety_buffer_s",
                cls.crossing_safety_buffer_s,
            ),
            phase_detection_budget_s=get(
                "traffic.traffic_signal.phase_detection_budget_s",
                cls.phase_detection_budget_s,
            ),
        )
        timing.validate()
        return timing

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.vehicle_green_s == 0:
            raise ValueError("vehicle_green_s must be positive")
        if self.pedestrian_walk_s == 0:
            raise ValueError("pedestrian_walk_s must be positive")
        if self.pedestrian_walking_speed_mps <= 0:
            raise ValueError("pedestrian_walking_speed_mps must be positive")
        if self.agent_crossing_speed_mps <= 0:
            raise ValueError("agent_crossing_speed_mps must be positive")

    def agent_crossing_budget_s(self, distance_m: float) -> float:
        """Return the fixed admission budget exposed to the VLM.

        The signal controller never extends a live WALK because the agent has
        entered.  Instead, the configured WALK window is sized in advance for
        physical travel, VLM inference, action execution, and a safety margin.
        """
        distance_m = max(0.0, float(distance_m))
        return (
            distance_m / self.agent_crossing_speed_mps
            + self.vlm_latency_budget_s
            + self.action_execution_budget_s
            + self.crossing_safety_buffer_s
        )

    def calibrated_for_crosswalks(
        self,
        crosswalks: Iterable[Any],
    ) -> "TrafficPhaseTiming":
        """Return a geometry-calibrated, MUTCD-aligned timing plan.

        Crosswalk geometry is stored in centimetres.  The pedestrian change
        interval is long enough to traverse the longest supplied crosswalk at
        the configured design walking speed.  Explicitly configured durations
        remain lower bounds so map owners can accommodate slower users.
        """
        lengths_m = []
        for crosswalk in crosswalks or []:
            start = getattr(crosswalk, "start", None)
            end = getattr(crosswalk, "end", None)
            if start is None or end is None:
                continue
            try:
                length_cm = start.distance(end)
            except AttributeError:
                dx = float(end.x) - float(start.x)
                dy = float(end.y) - float(start.y)
                length_cm = math.hypot(dx, dy)
            if length_cm > 0:
                lengths_m.append(float(length_cm) / 100.0)

        longest_crosswalk_m = max(lengths_m) if lengths_m else 0.0
        required_clearance_s = (
            math.ceil(longest_crosswalk_m / self.pedestrian_walking_speed_mps)
            if lengths_m
            else self.pedestrian_clearance_s
        )
        required_agent_walk_s = (
            math.ceil(
                self.agent_crossing_budget_s(longest_crosswalk_m)
                + self.phase_detection_budget_s
            )
            if lengths_m
            else self.pedestrian_walk_s
        )
        calibrated = replace(
            self,
            pedestrian_walk_s=max(
                self.pedestrian_walk_s,
                self.minimum_pedestrian_walk_s,
                float(required_agent_walk_s),
            ),
            pedestrian_clearance_s=max(
                self.pedestrian_clearance_s,
                float(required_clearance_s),
            ),
        )
        calibrated.validate()
        return calibrated

    @property
    def blueprint_pedestrian_green_s(self) -> float:
        """Duration passed to the legacy BP until it exposes clearance."""
        return self.pedestrian_walk_s + self.pedestrian_clearance_s

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


TIMING_ENVIRONMENT_OVERRIDES = {
    "SIMWORLD_TRAFFIC_VEHICLE_GREEN_S": "vehicle_green_s",
    "SIMWORLD_TRAFFIC_VEHICLE_YELLOW_S": "vehicle_yellow_s",
    "SIMWORLD_TRAFFIC_ALL_RED_S": "all_red_s",
    "SIMWORLD_TRAFFIC_PEDESTRIAN_WALK_S": "pedestrian_walk_s",
    "SIMWORLD_TRAFFIC_PEDESTRIAN_CLEARANCE_S": "pedestrian_clearance_s",
}


def apply_timing_environment_overrides(
    timing: TrafficPhaseTiming,
    environ: Optional[Dict[str, str]] = None,
) -> TrafficPhaseTiming:
    """Apply explicit fixed-cycle benchmark overrides.

    These values select the phase plan before a rollout starts.  They never
    inspect occupancy and never extend or hold an active phase.  This makes a
    deliberately short collision-probe cycle reproducible while keeping the
    production geometry-calibrated plan as the default.
    """
    environ = os.environ if environ is None else environ
    overrides = {}
    for environment_name, field_name in TIMING_ENVIRONMENT_OVERRIDES.items():
        raw = environ.get(environment_name)
        if raw is None or not str(raw).strip():
            continue
        try:
            overrides[field_name] = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{environment_name} must be numeric, got {raw!r}"
            ) from exc
    overridden = replace(timing, **overrides) if overrides else timing
    overridden.validate()
    return overridden


class TrafficPhaseStateMachine:
    """Deterministic reference controller for the upgraded UE Blueprint.

    The packaged Blueprint still owns runtime switching. This state machine is
    the executable phase contract that UE should match and lets us unit-test
    vehicle and pedestrian heads without depending on VLM output.
    """

    def __init__(
        self,
        timing: TrafficPhaseTiming,
        vehicle_movement_group_count: int = 1,
    ):
        if vehicle_movement_group_count < 1:
            raise ValueError("vehicle_movement_group_count must be at least 1")
        timing.validate()
        self.timing = timing
        self.vehicle_movement_group_count = vehicle_movement_group_count
        self.active_vehicle_group = 0
        self._steps = self._build_steps(timing)
        self._step_index = 0
        self.remaining_time_s = self._steps[0][1]
        self.pedestrian_occupied = False
        self.clearance_extended = False
        self.clearance_extension_elapsed_s = 0.0
        self.clearance_extension_quantum_s = 0.5

    @staticmethod
    def _build_steps(timing: TrafficPhaseTiming):
        return [
            (IntersectionPhase.VEHICLE_GREEN, timing.vehicle_green_s),
            (IntersectionPhase.VEHICLE_YELLOW, timing.vehicle_yellow_s),
            (IntersectionPhase.ALL_RED, timing.all_red_s),
            (IntersectionPhase.PEDESTRIAN_WALK, timing.pedestrian_walk_s),
            (
                IntersectionPhase.PEDESTRIAN_CLEARANCE,
                timing.pedestrian_clearance_s,
            ),
            (IntersectionPhase.ALL_RED, timing.all_red_s),
        ]

    def set_timing(
        self,
        timing: TrafficPhaseTiming,
        preserve_phase_progress: bool = True,
    ) -> Dict[str, Any]:
        """Apply a new cycle plan through an explicit policy interface.

        A future RL controller can call this method directly.  The update is
        independent of agent/VLM actions and normally preserves the fraction
        of the current phase that remains.
        """
        timing.validate()
        old_duration = self._steps[self._step_index][1]
        old_remaining = self.remaining_time_s
        self.timing = timing
        self._steps = self._build_steps(timing)
        new_duration = self._steps[self._step_index][1]
        if preserve_phase_progress and old_duration > 0:
            remaining_fraction = min(
                1.0,
                max(0.0, old_remaining / old_duration),
            )
            self.remaining_time_s = new_duration * remaining_fraction
        else:
            self.remaining_time_s = new_duration
        return self.snapshot()

    @property
    def phase(self) -> IntersectionPhase:
        return self._steps[self._step_index][0]

    def _next_step(self) -> None:
        previous_index = self._step_index
        self._step_index = (self._step_index + 1) % len(self._steps)
        if previous_index == len(self._steps) - 1:
            self.active_vehicle_group = (
                self.active_vehicle_group + 1
            ) % self.vehicle_movement_group_count
        self.remaining_time_s = self._steps[self._step_index][1]
        self.clearance_extended = False
        self.clearance_extension_elapsed_s = 0.0

    def set_pedestrian_occupancy(self, occupied: bool) -> Dict[str, Any]:
        """Update passive roadway occupancy without consulting agent output.

        If an admitted pedestrian is still present when the nominal change
        interval expires, the controller keeps flashing DON'T WALK and holds
        every vehicle group at red.  Once the detector reports clear, the
        controller immediately enters its all-red buffer.
        """
        self.pedestrian_occupied = bool(occupied)
        if (
            not self.pedestrian_occupied
            and self.clearance_extended
            and self.phase == IntersectionPhase.PEDESTRIAN_CLEARANCE
        ):
            self._next_step()
        return self.snapshot()

    def advance(self, delta_time_s: float) -> Dict[str, Any]:
        if delta_time_s < 0:
            raise ValueError("delta_time_s must be non-negative")
        remaining_delta = float(delta_time_s)
        zero_duration_guard = 0
        while remaining_delta >= self.remaining_time_s:
            elapsed_interval = self.remaining_time_s
            remaining_delta -= elapsed_interval
            if (
                self.phase == IntersectionPhase.PEDESTRIAN_CLEARANCE
                and self.clearance_extended
            ):
                self.clearance_extension_elapsed_s += elapsed_interval
            if (
                self.phase == IntersectionPhase.PEDESTRIAN_CLEARANCE
                and self.pedestrian_occupied
            ):
                self.clearance_extended = True
                self.remaining_time_s = self.clearance_extension_quantum_s
            else:
                self._next_step()
            if self.remaining_time_s == 0:
                zero_duration_guard += 1
                if zero_duration_guard > len(self._steps):
                    raise RuntimeError("traffic phase plan has no positive duration")
                continue
            zero_duration_guard = 0
        if (
            self.phase == IntersectionPhase.PEDESTRIAN_CLEARANCE
            and self.clearance_extended
        ):
            self.clearance_extension_elapsed_s += remaining_delta
        self.remaining_time_s -= remaining_delta
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        phase = self.phase
        vehicle_state = "RED"
        pedestrian_state = "DON'T WALK"
        released_vehicle_group = None
        if phase == IntersectionPhase.VEHICLE_GREEN:
            vehicle_state = "GREEN"
            released_vehicle_group = self.active_vehicle_group
        elif phase == IntersectionPhase.VEHICLE_YELLOW:
            vehicle_state = "YELLOW"
            released_vehicle_group = self.active_vehicle_group
        elif phase == IntersectionPhase.PEDESTRIAN_WALK:
            pedestrian_state = "WALK"
        elif phase == IntersectionPhase.PEDESTRIAN_CLEARANCE:
            pedestrian_state = "FLASHING_DONT_WALK"

        return {
            "phase": phase.value,
            "remaining_time_s": self.remaining_time_s,
            "active_vehicle_group": released_vehicle_group,
            "active_vehicle_group_state": vehicle_state,
            "other_vehicle_groups_state": "RED",
            "pedestrian_state": pedestrian_state,
            "pedestrian_occupied": self.pedestrian_occupied,
            "clearance_extended": self.clearance_extended,
            "clearance_extension_elapsed_s": (
                self.clearance_extension_elapsed_s
            ),
        }


def _signal_metadata(signal: Any, role: str) -> Dict[str, Any]:
    direction = getattr(signal, "direction", None)
    position = getattr(signal, "position", None)
    return {
        "signal_id": signal.id,
        "role": role,
        "type": getattr(signal, "type", role),
        "lane_id": getattr(signal, "lane_id", None),
        "crosswalk_id": getattr(signal, "crosswalk_id", None),
        "vehicle_group": getattr(signal, "vehicle_group", None),
        "direction": (
            [direction.x, direction.y] if direction is not None else None
        ),
        "position": [position.x, position.y] if position is not None else None,
    }


def derive_vehicle_group_by_signal_id(
    signals: Iterable[Any],
    parallel_threshold: float = 0.95,
) -> Dict[Any, int]:
    """Derive compatible approach groups from rendered signal geometry.

    RT maps do not consistently persist the Python ``vehicle_group``
    annotation through every agent-facing signal view.  Opposing, collinear
    approaches are compatible for the current permissive-through plan, while
    perpendicular approaches conflict.  Reconstructing the same grouping
    from direction vectors keeps the controller, rendered heads, evaluation,
    and VLM prompt aligned without relying on initialization order.
    """
    group_axes = []
    group_by_signal_id = {}
    for signal in signals or []:
        direction = getattr(signal, "direction", None)
        if direction is None:
            continue
        try:
            dx = float(direction.x)
            dy = float(direction.y)
        except (AttributeError, TypeError, ValueError):
            continue
        norm = math.hypot(dx, dy)
        if norm <= 1e-9:
            continue
        axis = (dx / norm, dy / norm)
        group_id = None
        for index, existing_axis in enumerate(group_axes):
            dot = axis[0] * existing_axis[0] + axis[1] * existing_axis[1]
            if abs(dot) >= parallel_threshold:
                group_id = index
                break
        if group_id is None:
            group_id = len(group_axes)
            group_axes.append(axis)
        group_by_signal_id[signal.id] = group_id
    return group_by_signal_id


def classify_observed_phase(signal_states: Iterable[Dict[str, Any]]) -> IntersectionPhase:
    """Classify the phase visible through the current packaged BP API."""
    states = list(signal_states)
    if not states:
        return IntersectionPhase.UNKNOWN

    any_vehicle_green = any(state.get("vehicle_green") for state in states)
    pedestrian_states = {
        str(state.get("pedestrian_state", "")).strip().upper().replace(" ", "_")
        for state in states
        if state.get("pedestrian_state") is not None
    }
    any_pedestrian_walk = any(state.get("pedestrian_walk") for state in states)
    any_pedestrian_walk = any_pedestrian_walk or "WALK" in pedestrian_states
    any_pedestrian_clearance = bool(
        pedestrian_states
        & {"CLEARANCE", "FLASHING_DONT_WALK", "FLASHING_DON'T_WALK"}
    )
    if any_vehicle_green and any_pedestrian_walk:
        return IntersectionPhase.CONFLICT
    if any_vehicle_green and any_pedestrian_clearance:
        return IntersectionPhase.CONFLICT
    if any_vehicle_green:
        return IntersectionPhase.VEHICLE_GREEN
    if any_pedestrian_walk:
        return IntersectionPhase.PEDESTRIAN_WALK
    if any_pedestrian_clearance:
        return IntersectionPhase.PEDESTRIAN_CLEARANCE

    # The legacy GetState API cannot distinguish vehicle yellow, all-red, and
    # pedestrian clearance.  Do not invent a more precise state than UE reports.
    return IntersectionPhase.UNKNOWN


def build_intersection_snapshot(
    intersection: Any,
    signal_states: List[Dict[str, Any]],
    timing: Optional[TrafficPhaseTiming] = None,
    timestamp_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Build one canonical, serializable snapshot for an entire intersection."""
    phase = classify_observed_phase(signal_states)
    vehicle_signal_ids = {
        light.id for light in getattr(intersection, "traffic_lights", [])
    }
    pedestrian_signal_ids = {
        light.id for light in getattr(intersection, "pedestrian_lights", [])
    }
    metadata = {}
    for signal in getattr(intersection, "traffic_lights", []):
        metadata[signal.id] = _signal_metadata(signal, "vehicle_and_pedestrian")
    for signal in getattr(intersection, "pedestrian_lights", []):
        metadata[signal.id] = _signal_metadata(signal, "pedestrian")

    # The intersection-owned plan is authoritative.  Some callers retain
    # signal objects created before group annotations were attached, so derive
    # a stable fallback directly from the movement-group table used by the
    # controller instead of leaking ``vehicle_group=None`` into the prompt.
    for group_id, group in enumerate(
        getattr(intersection, "vehicle_movement_groups", []) or []
    ):
        for signal in group:
            if signal.id in metadata:
                metadata[signal.id]["vehicle_group"] = group_id

    derived_groups = derive_vehicle_group_by_signal_id(
        getattr(intersection, "traffic_lights", [])
    )
    for signal_id, group_id in derived_groups.items():
        if (
            signal_id in metadata
            and metadata[signal_id].get("vehicle_group") is None
        ):
            metadata[signal_id]["vehicle_group"] = group_id

    enriched_states = []
    for state in signal_states:
        signal_metadata = metadata.get(state.get("signal_id"), {})
        item = dict(signal_metadata)
        item.update(state)
        if item.get("vehicle_group") is None:
            item["vehicle_group"] = signal_metadata.get("vehicle_group")
        enriched_states.append(item)

    violations = []
    if phase == IntersectionPhase.CONFLICT:
        violations.append("conflicting_vehicle_green_and_pedestrian_walk")
    active_vehicle_groups = {
        state.get("vehicle_group")
        for state in enriched_states
        if state.get("vehicle_green")
        and state.get("vehicle_group") is not None
    }
    if len(active_vehicle_groups) > 1:
        violations.append("conflicting_vehicle_movement_groups_green")

    remaining_times = [
        state.get("remaining_time_s")
        for state in enriched_states
        if state.get("remaining_time_s") is not None
    ]

    return {
        "schema_version": 1,
        "intersection_id": getattr(intersection, "id", None),
        "timestamp_s": time.time() if timestamp_s is None else timestamp_s,
        "observed_phase": phase.value,
        "phase_remaining_time_s": (
            min(remaining_times) if remaining_times else None
        ),
        "safe": not violations,
        "violations": violations,
        "vehicle_signal_ids": sorted(vehicle_signal_ids),
        "pedestrian_signal_ids": sorted(pedestrian_signal_ids),
        "active_vehicle_groups": sorted(active_vehicle_groups),
        "signal_states": enriched_states,
        "timing": timing.to_dict() if timing is not None else None,
        "source": "ue_blueprint:GetState",
        "capabilities": {
            "vehicle_green_exposed": True,
            "pedestrian_walk_exposed": True,
            "remaining_time_exposed": True,
            "vehicle_yellow_exposed": False,
            "all_red_exposed": False,
            "pedestrian_clearance_exposed": False,
        },
    }


def format_environment_traffic_context(snapshot: Optional[Dict[str, Any]]) -> str:
    """Describe synchronized signal facts without prescribing an agent action."""
    if not snapshot:
        return "TRAFFIC SYSTEM OBSERVATION: No synchronized intersection state is available."

    phase = snapshot.get("observed_phase", IntersectionPhase.UNKNOWN.value)
    remaining = snapshot.get("phase_remaining_time_s")
    lines = [
        "TRAFFIC SYSTEM OBSERVATION "
        "(captured from UE immediately before the rendered image):",
        f"- Observed phase: {phase}",
    ]
    if snapshot.get("intersection_id") is not None:
        lines.insert(1, f"- Intersection: {snapshot['intersection_id']}")
    if remaining is not None:
        lines.append(
            f"- Controller/SPaT time remaining in this phase: {remaining:.1f} s"
        )
    legal_walk_remaining = snapshot.get("walk_remaining_time_s")
    if legal_walk_remaining is not None:
        lines.append(
            "- Legal WALK time remaining before flashing DON'T WALK: "
            f"{float(legal_walk_remaining):.1f} s"
        )
    crossing_budget = snapshot.get("estimated_crossing_time_s")
    if crossing_budget is not None:
        lines.append(
            "- Estimated time needed to reach the far curb (travel + VLM/action "
            f"latency + safety margin): {crossing_budget:.1f} s"
        )
    crossing_permission = snapshot.get("route_crossing_permission")
    if snapshot.get("can_finish_before_signal_change") is not None:
        lines.append(
            "- Enough current WALK time to finish before pedestrian clearance: "
            + (
                "YES"
                if snapshot["can_finish_before_signal_change"]
                else "NO"
            )
        )
        if (
            not snapshot["can_finish_before_signal_change"]
            and crossing_permission == "CLEAR_ONLY"
        ):
            lines.append(
                "- You are already inside the roadway. The time sufficiency "
                "test is only an entry decision: do NOT wait in the traffic "
                "lane; continue promptly to the far curb."
            )
        elif (
            not snapshot["can_finish_before_signal_change"]
            and snapshot.get("route_subgoal_role") == "NEAR_CURB"
        ):
            lines.append(
                "- Continue along the sidewalk to the near curb. Enter only "
                "when enough WALK time remains to reach the far curb before "
                "the signal changes."
            )
        elif not snapshot["can_finish_before_signal_change"]:
            lines.append(
                "- Do not enter: the remaining WALK time is insufficient to "
                "reach the far curb without a red-light violation."
            )
    if snapshot.get("relevant"):
        lines.append(
            "- Evaluation rule: WALK must remain active until the far curb. "
            "Flashing/non-WALK while the pedestrian is still in the crossing "
            "is a red-light violation."
        )

    for state in snapshot.get("signal_states", []):
        vehicle_state = state.get("vehicle_state")
        if vehicle_state is None:
            vehicle_state = (
                "GREEN" if state.get("vehicle_green") else "NOT_GREEN"
            )
        pedestrian_state = state.get("pedestrian_state")
        if pedestrian_state is None:
            pedestrian_state = (
                "WALK" if state.get("pedestrian_walk") else "DONT_WALK"
            )
        lines.append(
            "- Signal {signal_id} ({role}, lane={lane_id}, crosswalk={crosswalk_id}, "
            "vehicle_group={vehicle_group}): vehicle={vehicle_state}, "
            "pedestrian={pedestrian_state}".format(
                signal_id=state.get("signal_id", state.get("name")),
                role=state.get("role", state.get("type", "unknown")),
                lane_id=state.get("lane_id"),
                crosswalk_id=state.get("crosswalk_id"),
                vehicle_group=state.get("vehicle_group"),
                vehicle_state=vehicle_state,
                pedestrian_state=pedestrian_state,
            )
        )
    if snapshot.get("violations"):
        lines.append(
            "- Consistency violations: " + ", ".join(snapshot["violations"])
        )
    return "\n".join(lines)
