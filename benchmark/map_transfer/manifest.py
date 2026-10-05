"""Validate portable map manifests before any editor mutation."""

from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Any

_IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}")
_RESERVED = {"agent", "goal"}


def _number(
    value: Any,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    *,
    exclusive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and (value <= minimum if exclusive else value < minimum):
        raise ValueError(f"{name} is below its allowed range")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} is above its allowed range")
    return value


def _vector(value: Any, name: str, length: int = 3) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must have {length} finite coordinates")
    for coordinate in value:
        _number(coordinate, name)


def _records(value: Any, name: str) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be a list of objects")
    identifiers = [item.get("id") for item in value]
    if any(
        not isinstance(item, str) or not _IDENTIFIER.fullmatch(item)
        for item in identifiers
    ):
        raise ValueError(f"{name} IDs must be short alphanumeric identifiers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must have unique IDs")
    return value


def _polygon(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{name} needs at least three vertices")
    for vertex in value:
        _vector(vertex, name, 2)
    if len({tuple(p) for p in value}) != len(value):
        raise ValueError(f"{name} has duplicate vertices")
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(value, value[1:] + value[:1]))
    if abs(area) < 1:
        raise ValueError(f"{name} must enclose a nonzero area")
    # Require a simple convex polygon. This also rules out crossing edges and
    # keeps the pilot's road/crosswalk annotations unambiguous.
    crosses = []
    for i, a in enumerate(value):
        b = value[(i + 1) % len(value)]
        c = value[(i + 2) % len(value)]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-7:
            crosses.append(cross > 0)
    if not crosses or len(set(crosses)) != 1:
        raise ValueError(f"{name} must be a simple convex polygon")
    for a, b in zip(value, value[1:] + value[:1]):
        sides = {
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0
            for c in value
            if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 1e-7
        }
        if len(sides) > 1:
            raise ValueError(f"{name} must be a simple convex polygon")


def validate_manifest(data: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(data, dict)
        or data.get("schema") != "rtsafe-native-map-v1"
        or data.get("units") != "cm"
    ):
        raise ValueError("Expected rtsafe-native-map-v1 in Unreal centimeters")
    if not isinstance(data.get("map_path"), str) or not data["map_path"].startswith(
        "/Game/"
    ):
        raise ValueError("Map must be an Unreal /Game/ package")
    if data.get("pilot") is not True:
        raise ValueError("Native map transfer results must currently be labeled pilot")
    agent = data.get("agent")
    if not isinstance(agent, dict):
        raise ValueError("Provide an agent configuration")
    for key, prefix in [("class_path", "/Script/"), ("mesh_path", "/Game/")]:
        if not isinstance(agent.get(key), str) or not agent[key].startswith(prefix):
            raise ValueError(f"agent.{key} must be an Unreal asset/class path")
    radius = _number(
        agent.get("capsule_radius_cm"), "capsule radius", 0, 100, exclusive=True
    )
    _number(agent.get("capsule_half_height_cm"), "capsule half-height", radius, 250)
    _number(agent.get("speed_cm_s"), "agent speed", 0, 1000, exclusive=True)
    _number(agent.get("turn_duration_s", 1.0), "turn duration", 0, 10, exclusive=True)
    _number(agent.get("fov_deg"), "field of view", 10, 150)
    _number(agent.get("camera_pitch_deg"), "camera pitch", -89, 89)
    interaction = agent.get("physics_interaction", {})
    if not isinstance(interaction, dict):
        raise ValueError("physics_interaction must be an object")
    for key, value in interaction.items():
        if key not in (
            "initial_push_force_factor",
            "push_force_factor",
            "touch_force_factor",
        ):
            raise ValueError(f"Unknown physics interaction parameter {key}")
        _number(value, key, 0, 1000000)
    for key in ("width", "height"):
        value = agent.get(key)
        _number(value, f"image {key}", 64, 4096)
        if not isinstance(value, int):
            raise ValueError(f"image {key} must be an integer")
    if not isinstance(data.get("disable_ambient_spawners", True), bool):
        raise ValueError("disable_ambient_spawners must be boolean")
    effects = data.get("hazard_effects", {})
    if not isinstance(effects, dict):
        raise ValueError("hazard_effects must be an object")
    _number(effects.get("trip_recovery_seconds", 6), "trip recovery duration", 0, 10)
    _number(
        effects.get("oil_speed_factor", 0.5), "oil speed factor", 0, 1, exclusive=True
    )
    tasks = _records(data.get("tasks"), "tasks")
    if not tasks:
        raise ValueError("Provide at least one task")
    for task in tasks:
        if "instruction" in task and (
            not isinstance(task["instruction"], str) or len(task["instruction"]) > 2000
        ):
            raise ValueError("Task instruction must be text of at most 2000 characters")
        _vector(task.get("start_cm"), "start_cm")
        _vector(task.get("goal_cm"), "goal_cm")
        _number(task.get("yaw_deg"), "yaw")
        _number(task.get("goal_radius_cm", 100), "goal radius", 0, 300, exclusive=True)
        _number(task.get("time_limit_s", 180), "time limit", 0, 3600, exclusive=True)
        for point in task.get("reference_route_cm", []):
            _vector(point, "reference route point")
        entities = _records(task.get("entities", []), "entities")
        for entity in entities:
            if entity["id"] in _RESERVED:
                raise ValueError(f'Reserved entity ID: {entity["id"]}')
            if entity.get("kind") not in {
                "pedestrian",
                "obstacle",
                "water",
                "oil",
                "trip",
                "vehicle",
                "robot",
                "movable",
                "falling",
            }:
                raise ValueError(f'Unknown entity kind {entity.get("kind")}')
            _vector(entity.get("position_cm"), "entity position")
            if entity["kind"] in ("water", "oil", "trip") and "size_cm" not in entity:
                raise ValueError("Hazards require explicit dimensions")
            if "size_cm" in entity:
                _vector(entity["size_cm"], "entity size")
                if min(entity["size_cm"]) <= 0:
                    raise ValueError("Entity dimensions must be positive")
            if "speed_cm_s" in entity:
                _number(entity["speed_cm_s"], "entity speed", 0, 2000, exclusive=True)
            if "mass_kg" in entity:
                _number(entity["mass_kg"], "object mass", 0, 1000, exclusive=True)
            if entity["kind"] == "falling":
                _number(entity.get("drop_height_cm"), "drop height", 200, 2000)
                _number(entity.get("release_delay_s"), "release delay", 0, 3600)
            if "activation_zone" in entity and entity["kind"] != "vehicle":
                raise ValueError("Traffic activation is only supported for vehicles")
            if "ground_clearance_cm" in entity:
                _number(
                    entity["ground_clearance_cm"], "vehicle ground clearance", 0, 100
                )
            if "loop" in entity and not isinstance(entity["loop"], bool):
                raise ValueError("Entity loop must be boolean")
            route = entity.get("path_cm", [])
            if not isinstance(route, list):
                raise ValueError("Entity route must be a list")
            for point in route:
                _vector(point, "entity route point")
            if route and len(route) < 2:
                raise ValueError("Moving entities need at least two route points")
        zones = _records(task.get("traffic_zones", []), "traffic zones")
        for entity in entities:
            if "activation_zone" in entity and entity["activation_zone"] not in {
                z["id"] for z in zones
            }:
                raise ValueError(
                    "Vehicle activation zone is not an authored traffic zone"
                )
        for zone in zones:
            for key in ("road_polygon_cm", "crosswalk_polygon_cm"):
                _polygon(zone.get(key), key)
            _vector(zone.get("signal_position_cm"), "signal position")
            for key in ("walk_seconds", "stop_seconds"):
                _number(zone.get(key), key, 0, 3600, exclusive=True)
            _number(zone.get("phase_offset_s", 0), "signal phase offset", 0)
    return data


def load_manifest(path: str | Path) -> dict[str, Any]:
    return validate_manifest(json.loads(Path(path).read_text()))
