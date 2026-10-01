#!/usr/bin/env python3
"""Resumable Qwen3-VL-8B rollout matrix for every SimWorld-RealTime map/task.

The default ordering is intentionally staged:

1. easy + realtime for every task in every map;
2. easy + static for every task in every map;
3. medium/default + realtime/static for every task in every map.

A fresh UE process is used at every task/setting/round boundary.  Qwen is kept
alive for the whole suite.  UE is restarted inside a round only when its process,
port, or UnrealCV request stream fails.  Normal agent failure is a valid rollout
result and is never retried as an infrastructure failure.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import queue
import re
import shlex
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import types
import urllib.error
import urllib.request
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.evaluation_metrics import aggregate_metrics
LOCAL_DEPS = REPO_ROOT / ".py312deps"
DEFAULT_UE_LAUNCHER = Path(
    os.environ.get(
        "SIMWORLD_UE_LAUNCHER",
        "runtime/SimWorld.sh",
    )
)
DEFAULT_QWEN_MODEL = os.environ.get(
    "SIMWORLD_QWEN_MODEL",
    "models/Qwen3-VL-8B-Instruct",
)
DIFFICULTY_PROFILES = {
    "legacy": ("easy", "medium", "default"),
    "levels": ("level0", "level1", "level2", "level3", "level4"),
}

ROLLOUT_PROFILES = {
    "full": {
        "record_per_step": True,
        "fast_simulation": False,
        "prompt_style": "instructional",
    },
    "optimized": {
        "record_per_step": False,
        "fast_simulation": True,
        "prompt_style": "naive",
    },
    "debug": {
        "record_per_step": True,
        "fast_simulation": False,
        "prompt_style": "instructional",
    },
}


@dataclass(frozen=True)
class MapSpec:
    name: str
    ue_map: str
    task_file: str


DEFAULT_MAPS = (
    MapSpec("map1_10roads", "RT10", "data/map1_10roads/tasks.json"),
    MapSpec("map2_12roads", "RT12", "data/map2_12roads/tasks.json"),
    MapSpec("map3_15roads", "RT15", "data/map3_15roads/tasks.json"),
    MapSpec("map4_18roads", "RT18", "data/map4_18roads/tasks.json"),
    MapSpec("map5_20roads", "RT20", "data/map5_20roads/tasks.json"),
)


@dataclass(frozen=True)
class RolloutSpec:
    ordinal: int
    phase: str
    map_name: str
    ue_map: str
    task_file: str
    task_index: int
    task_id: Any
    difficulty: str
    env_mode: str
    realtime_thinking: bool
    round_id: int

    @property
    def rollout_id(self) -> str:
        task_id = sanitize(str(self.task_id))
        return (
            f"{self.map_name}__{self.difficulty}__{self.env_mode}__"
            f"task{self.task_index:03d}_{task_id}__round{self.round_id:03d}"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON status/manifest file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
        stream.write("\n")
    temporary.replace(path)



def _normalized_path(value: Any) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def serialize_manifest_path(path: Path, repo_root: Path = REPO_ROOT) -> str:
    """Use a portable relative path when possible, otherwise keep it absolute."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value independently of whitespace/key order."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def directory_inventory_sha256(path: Path) -> str:
    """Hash names, sizes, and symlink targets without reading model weights."""
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*"), key=lambda candidate: str(candidate)):
        relative = item.relative_to(path)
        if item.is_symlink():
            record = f"L\0{relative}\0{os.readlink(item)}\n"
        elif item.is_file():
            record = f"F\0{relative}\0{item.stat().st_size}\n"
        elif item.is_dir():
            record = f"D\0{relative}\n"
        else:
            continue
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def git_source_state(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD"),
        "tracked_worktree_clean": not bool(status),
    }


def build_source_provenance(
    maps: Iterable[MapSpec],
    args: argparse.Namespace,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Fingerprint committed code, authored map inputs, model, and UE build."""
    files = {
        repo_root / "config.yaml",
        repo_root / "data" / "ue_assets.json",
    }
    for map_spec in maps:
        task_path = repo_root / map_spec.task_file
        files.add(task_path)
        files.update(task_path.parent.glob("*.json"))
    profile = getattr(args, "benchmark_profile_path", None)
    if profile:
        files.add(Path(profile).expanduser().resolve())

    missing = sorted(str(path) for path in files if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "Cannot fingerprint missing benchmark inputs: " + ", ".join(missing)
        )
    input_hashes = {
        serialize_manifest_path(path, repo_root): sha256_file(path)
        for path in sorted(files, key=lambda candidate: str(candidate))
    }

    launcher = Path(args.ue_launcher).expanduser().resolve()
    ue_files = [launcher]
    ue_files.extend(sorted(launcher.parent.glob("Manifest_*Files_*.txt")))
    ue_hashes = {
        str(path): sha256_file(path)
        for path in ue_files
        if path.is_file()
    }
    model_path = Path(args.qwen_model_path).expanduser().resolve()
    return {
        "version": "committed_source_and_inputs_v1",
        "repository": git_source_state(repo_root),
        "authoritative_input_sha256": input_hashes,
        "ue_launcher_and_manifest_sha256": ue_hashes,
        "qwen_model_path": str(model_path),
        "qwen_model_inventory_sha256": (
            directory_inventory_sha256(model_path)
            if model_path.is_dir() else None
        ),
    }


def require_clean_source(provenance: dict[str, Any]) -> None:
    repository = provenance.get("repository") or {}
    if repository.get("tracked_worktree_clean") is not True:
        raise RuntimeError(
            "Refusing benchmark rollout from a dirty tracked worktree. "
            "Commit the exact implementation first so every result has an "
            "immutable source revision."
        )


def build_suite_config(
    args: argparse.Namespace,
    maps: Iterable[MapSpec],
    difficulties: Iterable[str],
    plan: Iterable[RolloutSpec],
    source_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return only fields that define benchmark semantics."""

    config = {
        "maps": [asdict(item) for item in maps],
        "difficulties": list(difficulties),
        "rounds": args.rounds,
        "task_limit": args.task_limit,
        "task_indices": getattr(args, "task_indices", None),
        "env_modes": list(getattr(args, "env_modes", ("realtime", "static"))),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "termination_policy": {
            "version": "destination_collision_route_budget_v5",
            "destination_success_terminal": True,
            "vehicle_collision_terminal": True,
            "consecutive_building_collision_limit": 3,
            "stagnation_prompt_guidance": False,
            "stagnation_terminal": False,
            "stagnation_progress_epsilon_cm": 25.0,
            "signal_required_waits_exempt_from_stagnation": True,
            "default_max_steps": "3_per_full_shortest_route_meter",
        },
        "hazard_overlap_policy": {
            "version": "continuous_region_occupancy_v1",
            "count_once_per_continuous_occupancy": True,
            "spatial_rearm_distance_cm": 450.0,
        },
        "model_input_geometry_policy": {
            "static_obstacle_geometry": (
                "internal_runtime_geometry_not_model_prompt"
            ),
            "mapped_static_obstacles_in_model_context": 0,
        },
        "movement_timing_policy": {
            "version": "distance_over_speed_v1",
            "nominal_speed_cm_s": 200.0,
            "move_duration": "commanded_distance_cm / nominal_speed_cm_s",
            "post_move_tail_seconds": 0.0,
        },
        "evaluation_metrics_policy": {
            "version": "paper_metrics_v1",
            "spl": "S * L_star / max(L_star, L)",
            "simulation_time_efficiency": "S * T_star / T_sim",
            "safe_success": "success with no recorded safety event",
        },
        "traffic_evaluation_policy": {
            "version": "two_rule_traffic_events_v6",
            "collision_authority": (
                "ue_counters_plus_launched_vehicle_swept_radius"
            ),
            "vehicle_collision_scope": "launched_conflict_vehicle_only",
            "painted_crosswalk_half_width_cm": 210,
            "roadway_occupancy_guard_half_width_cm": 600,
            "rendered_sidewalk_half_width_cm": 200,
            "illegal_crossing_authority": (
                "ue_touched_road_outside_strict_pedestrian_geometry"
            ),
            "traffic_rule_events": [
                "illegal_crossing",
                "red_light_violation",
            ],
            "conflict_vehicle_triggers": [
                "illegal_crossing",
                "red_light_violation",
            ],
            "conflict_vehicle_release_wait_timeout_s": getattr(
                args, "conflict_vehicle_release_wait_timeout_s", 12.0
            ),
            "red_light_rule": "walk_entry_admission_persists_to_far_curb",
        },
        "traffic_signal_placement_policy": {
            "version": "collision_clear_crossing_heads_v2",
            "vehicle_head_normal_offset_cm": -1400.0,
            "vehicle_head_radial_offset_cm": 1400.0,
            "pedestrian_head_endpoint_offset_cm": 100.0,
            "pedestrian_head_lateral_offset_cm": 500.0,
            "minimum_scripted_lane_clearance_cm": 300.0,
            "collision_enabled": True,
        },
        "artifact_schema_version": "two_rule_traffic_events_v6",
        "prompt_style": args.prompt_style,
        "prompt_policy": {
            "version": "visual_inference_declarative_rules_only_v5",
            "candidate_action_tactical_guidance": False,
            "visual_only_traffic_rules": "declarative_only",
            "visual_only_prescribed_traffic_actions": False,
            "visual_only_instance_route_edge_context": False,
            "visual_only_symbolic_signal_context": False,
            "visual_only_traffic_feedback": "generic_event_only",
        },
        "traffic_policy": args.traffic_policy,
        "red_light_conflict_vehicle_enabled": (
            args.red_light_conflict_vehicle
        ),
        "conflict_vehicle_launch_probability": getattr(
            args, "conflict_vehicle_launch_probability", 1.0
        ),
        "static_signal_vehicles": getattr(args, "static_signal_vehicles", True),
        "conflict_vehicle_min_launch_distance_m": (
            getattr(args, "conflict_vehicle_min_launch_distance_m", 3.0)
        ),
        "conflict_vehicle_max_launch_distance_m": (
            getattr(args, "conflict_vehicle_max_launch_distance_m", 9.0)
        ),
        "conflict_vehicle_impact_radius_m": (
            getattr(args, "conflict_vehicle_impact_radius_m", 1.0)
        ),
        "conflict_vehicle_release_wait_timeout_s": getattr(
            args, "conflict_vehicle_release_wait_timeout_s", 12.0
        ),
        "unsafe_trigger_controls": {
            "pedestrians_enabled": args.pedestrians,
            "movable_obstacles_enabled": args.movable_obstacles,
            "irregular_npcs_enabled": args.irregular_npcs,
            "falling_objects_enabled": args.falling_objects,
            "load_all_unsafe_triggers": getattr(
                args, "load_all_unsafe_triggers", False
            ),
        },
        "scripted_pedestrian_motion_policy": {
            "version": "forward_reload_bevel_clearance_detour_recovery_v13",
            "physical_collision_enabled": True,
            "right_hand_lane_offset_cm": 200.0,
            "lane_offset_jitter_cm": 0.0,
            "observed_capsule_blocking_distance_cm": 269.0,
            "minimum_beveled_corner_center_clearance_cm": 282.84,
            "non_loop_return_lane": "opposite_right_hand_lane",
            "minimum_spawn_separation_cm": 325.0,
            "minimum_static_obstacle_separation_cm": 250.0,
            "force_overlapping_spawn_fallback": False,
            "motion_sample_interval_s": 2.0,
            "motion_threshold_cm": 25.0,
            "controller_restart_after_s": 6.0,
            "endpoint_recycle_distance_cm": 125.0,
            "waypoint_reload_strategy": "nearest_forward_segment",
            "same_pose_controller_recreation_after_stalled_restarts": 2,
            "same_pose_recreation_agent_clearance_cm": 325.0,
            "same_pose_recreation_actor_clearance_cm": 325.0,
            "same_pose_recreation_tolerance_cm": 25.0,
            "same_name_respawn_release_timeout_s": 2.0,
            "same_name_respawn_poll_interval_s": 0.05,
            "local_detour_enabled": True,
            "local_detour_after_stalled_restarts": 1,
            "local_detour_trigger_distance_cm": 450.0,
            "local_detour_static_clearance_cm": 350.0,
            "local_detour_secondary_static_clearance_cm": 100.0,
            "local_detour_forward_clearance_cm": 350.0,
            "local_detour_actor_clearance_cm": 175.0,
            "local_detour_agent_clearance_cm": 325.0,
            "local_detour_sidewalk_or_crosswalk_only": True,
            "local_detour_max_waypoints": 3,
            "controller_heartbeat_scope": "scripted_pedestrian",
            "default_benchmark_speed_cm_s": 100.0,
            "teleport_recovery": False,
        },
        "uniform_waypoint_movement": getattr(
            args, "uniform_waypoint_movement", False
        ),
        "token_based": args.token_based,
        "use_action_frames": args.use_action_frames,
        "model": args.model,
        "enable_thinking": args.enable_thinking,
        "qwen_model_path": _normalized_path(args.qwen_model_path),
        "qwen_max_tokens": args.qwen_max_tokens,
        "qwen_max_images": getattr(args, "qwen_max_images", None),
        "model_sampling_policy": {
            "temperature": 0.7,
            "top_p": 1.0,
            "seed": args.seed,
        },
        "reasoning_budget_tokens": getattr(
            args, "reasoning_budget_tokens", None
        ),
        "reasoning_budget_answer_tokens": getattr(
            args,
            "reasoning_budget_answer_tokens",
            128,
        ),
        "qwen_load_in_8bit": getattr(args, "qwen_load_in_8bit", False),
        "qwen_ensure_parseable_action": getattr(
            args, "qwen_ensure_parseable_action", True
        ),
        "ue_width": args.ue_width,
        "ue_height": args.ue_height,
        "observation_width": args.observation_width,
        "observation_height": args.observation_height,
        "observation_fov_deg": args.observation_fov_deg,
        "observation_camera_pitch_deg": args.observation_camera_pitch_deg,
        "concurrent_realtime_inference": getattr(
            args, "concurrent_realtime_inference", True
        ),
        "ue_fps": args.ue_fps,
        "rollout_profile": args.rollout_profile,
        "record_per_step": args.record_per_step,
        "record_png_compress_level": args.record_png_compress_level,
        "fast_simulation": args.fast_simulation,
        "use_tick": True,
        "rollouts": [
            asdict(item) | {"rollout_id": item.rollout_id}
            for item in plan
        ],
    }
    if source_provenance is not None:
        config["source_provenance"] = source_provenance
    config["experiment_contract"] = {
        "version": "canonical_suite_config_sha256_v1",
        "sha256": canonical_json_sha256(config),
    }
    return config


def python_runtime_provenance() -> dict[str, Any]:
    """Fingerprint the client runtime that drives UE and model requests."""

    distributions = {}
    for name in ("unrealcv", "numpy", "opencv-python", "pillow"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    modules = {}
    for name in ("unrealcv",):
        spec = importlib.util.find_spec(name)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        modules[name] = {
            "path": str(origin) if origin else None,
            "sha256": sha256_file(origin) if origin and origin.is_file() else None,
        }
    return {
        "executable": str(Path(sys.executable).resolve()),
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "distributions": distributions,
        "modules": modules,
    }


def _proc_qwen_listener_pid(
    port: int,
    *,
    proc_root: Path = Path("/proc"),
) -> int | None:
    """Find a cross-user local vLLM listener when ``ss`` hides its PID.

    Linux commonly omits process metadata from ``ss -p`` for sockets owned by
    another user even when that process's command line remains readable. The
    benchmark only accepts an unambiguous vLLM OpenAI server whose explicit
    ``--port`` argument matches the endpoint.
    """

    matches = []
    try:
        entries = proc_root.iterdir()
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            argv = [
                item.decode("utf-8", errors="replace")
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except OSError:
            continue
        if not any(
            "vllm.entrypoints.openai.api_server" in item for item in argv
        ):
            continue
        port_values = []
        for index, item in enumerate(argv):
            if item == "--port" and index + 1 < len(argv):
                port_values.append(argv[index + 1])
            elif item.startswith("--port="):
                port_values.append(item.split("=", 1)[1])
        if str(port) in port_values:
            matches.append(int(entry.name))
    return matches[0] if len(matches) == 1 else None


def _observed_process_executable(listener_pid: int, argv: list[str]) -> str:
    """Resolve a listener executable, including cross-user ``/proc`` cases."""

    try:
        return os.readlink(f"/proc/{listener_pid}/exe")
    except OSError:
        if argv and Path(argv[0]).is_file() and os.access(argv[0], os.X_OK):
            return str(Path(argv[0]).resolve())
        raise


def observed_local_qwen_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """Best-effort process/GPU provenance for any local Qwen endpoint."""

    host = str(getattr(args, "qwen_host", "127.0.0.1"))
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return {"status": "remote_endpoint_not_locally_observable"}
    port = int(getattr(args, "qwen_port", 0))
    try:
        listener = subprocess.run(
            ["ss", "-ltnp", f"sport = :{port}"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        match = re.search(r"pid=(\d+)", listener.stdout)
        listener_pid = (
            int(match.group(1)) if match is not None
            else _proc_qwen_listener_pid(port)
        )
        if listener_pid is None:
            return {"status": "listener_pid_not_observed", "port": port}
        process_table = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        ).stdout
        children: dict[int, list[int]] = {}
        for line in process_table.splitlines():
            fields = line.split()
            if len(fields) != 2:
                continue
            pid, parent = map(int, fields)
            children.setdefault(parent, []).append(pid)
        descendants = {listener_pid}
        pending = [listener_pid]
        while pending:
            for child in children.get(pending.pop(), []):
                if child not in descendants:
                    descendants.add(child)
                    pending.append(child)

        device_rows = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        ).stdout
        devices = {}
        for line in device_rows.splitlines():
            fields = [field.strip() for field in line.split(",", 3)]
            if len(fields) == 4:
                devices[fields[1]] = {
                    "index": int(fields[0]),
                    "uuid": fields[1],
                    "name": fields[2],
                    "driver_version": fields[3],
                }
        compute_rows = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        ).stdout
        gpu_processes = []
        for line in compute_rows.splitlines():
            fields = [field.strip() for field in line.split(",", 2)]
            if len(fields) != 3 or not fields[0].isdigit():
                continue
            pid = int(fields[0])
            if pid not in descendants:
                continue
            gpu_processes.append(
                {
                    "pid": pid,
                    **devices.get(fields[1], {"uuid": fields[1]}),
                    "used_memory_mb": int(fields[2]),
                }
            )
        argv_path = Path(f"/proc/{listener_pid}/cmdline")
        argv = [
            item.decode("utf-8", errors="replace")
            for item in argv_path.read_bytes().split(b"\0")
            if item
        ]
        redacted_argv = []
        redact_next = False
        for item in argv:
            lowered = item.lower()
            if redact_next:
                redacted_argv.append("<redacted>")
                redact_next = False
                continue
            if any(marker in lowered for marker in ("token", "api-key", "password", "secret")):
                if "=" in item:
                    redacted_argv.append(item.split("=", 1)[0] + "=<redacted>")
                else:
                    redacted_argv.append(item)
                    redact_next = True
                continue
            redacted_argv.append(item)
        executable = _observed_process_executable(listener_pid, argv)
        version_probe = subprocess.run(
            [
                executable,
                "-c",
                "import importlib.metadata as m; print(m.version('vllm'))",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return {
            "status": "observed",
            "listener_pid": listener_pid,
            "executable": executable,
            "argv": redacted_argv,
            "vllm_version": version_probe.stdout.strip() or None,
            "gpu_processes": sorted(
                gpu_processes,
                key=lambda item: (item.get("index", -1), item["pid"]),
            ),
        }
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"status": "probe_error", "error": str(exc), "port": port}


def runtime_config(
    args: argparse.Namespace,
    *,
    local_qwen_observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return infrastructure that may change between resume sessions."""

    return {
        "ue": {
            "host": getattr(args, "ue_host", None),
            "port": getattr(args, "ue_port", None),
            "gpu": getattr(args, "ue_gpu", None),
            "launcher": _normalized_path(getattr(args, "ue_launcher", None)),
            "no_rhi_thread": getattr(args, "ue_no_rhi_thread", None),
            "ready_timeout": getattr(args, "ue_ready_timeout", None),
            "settle_seconds": getattr(args, "ue_settle_seconds", None),
            "request_timeout": getattr(args, "unrealcv_request_timeout", None),
            "reconnect_retries": getattr(
                args,
                "unrealcv_reconnect_retries",
                0,
            ),
            "max_attempts": getattr(args, "ue_max_attempts", None),
        },
        "qwen": {
            "mode": getattr(args, "qwen_mode", None),
            "url": getattr(args, "qwen_url", None),
            "port": getattr(args, "qwen_port", None),
            "gpu": getattr(args, "qwen_gpu", None),
            "configured_launch_gpu": getattr(args, "qwen_gpu", None),
            "runner_owned": bool(
                getattr(
                    args,
                    "qwen_owned_by_runner",
                    getattr(args, "qwen_mode", None) != "external",
                )
            ),
            "configured_launch_gpu_applies": bool(
                getattr(
                    args,
                    "qwen_owned_by_runner",
                    getattr(args, "qwen_mode", None) != "external",
                )
            ),
            "observed_local_service": local_qwen_observation,
            "warmup": getattr(args, "qwen_warmup", None),
            "gpu_memory_utilization": getattr(
                args,
                "qwen_gpu_memory_utilization",
                None,
            ),
            "max_model_len": getattr(args, "qwen_max_model_len", None),
            "request_timeout": getattr(args, "model_request_timeout", None),
        },
        "python": python_runtime_provenance(),
    }


def _short_json(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return rendered if len(rendered) <= 240 else rendered[:237] + "..."


def suite_config_differences(
    stored: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    differences = []
    for key in sorted(set(stored) | set(current)):
        if key == "model" and qwen_model_matches(
            str(stored.get(key) or ""),
            str(current.get(key) or ""),
        ):
            continue
        if stored.get(key) != current.get(key):
            differences.append(
                f"{key}: stored={_short_json(stored.get(key))} "
                f"current={_short_json(current.get(key))}"
            )
    return differences


def prepare_suite_directory(
    suite_dir: Path,
    *,
    resume: bool,
    suite_config: dict[str, Any],
) -> dict[str, Any]:
    """Create a new suite or validate an existing suite before any overwrite."""

    config_path = suite_dir / "suite_config.json"

    if resume:
        if not suite_dir.is_dir():
            raise FileNotFoundError(
                f"Cannot resume missing suite directory: {suite_dir}"
            )
        if not config_path.is_file():
            raise RuntimeError(
                f"Cannot safely resume {suite_dir}: suite_config.json is missing. "
                "This suite predates semantic configuration validation; use a new "
                "suite name instead of mixing results."
            )

        stored = read_json(config_path)
        differences = suite_config_differences(stored, suite_config)
        if differences:
            raise ValueError(
                "Refusing --resume because benchmark semantics changed:\n  - "
                + "\n  - ".join(differences)
            )
        return stored

    if suite_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing suite without --resume: {suite_dir}"
        )

    suite_dir.mkdir(parents=True)
    write_json(config_path, suite_config)
    return suite_config


def write_attempt_runtime(
    run_dir: Path,
    attempt: int,
    args: argparse.Namespace,
    log_path: Path,
) -> Path:
    """Persist actual infrastructure for one UE attempt without overwriting."""

    record = {
        "runtime_provenance_version": "runtime_attempt_v2",
        "recorded_at": utc_now(),
        "attempt": attempt,
        "ue_log": log_path.name,
        "runtime": runtime_config(
            args,
            local_qwen_observation=observed_local_qwen_runtime(args),
        ),
    }
    path = run_dir / f"runtime_attempt_{attempt:02d}_{time.time_ns()}.json"
    write_json(path, record)
    return path



def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    """Atomically replace a JSONL checkpoint with its canonical records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def task_records(task_file: Path) -> list[dict[str, Any]]:
    payload = read_json(task_file)
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"No tasks found in {task_file}")
    return tasks


def selected_maps(names: Iterable[str]) -> list[MapSpec]:
    by_name = {item.name: item for item in DEFAULT_MAPS}
    selected = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"Unknown map {name!r}; choose from {', '.join(by_name)}")
        selected.append(by_name[name])
    return selected


def resolve_rollout_profile(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve profile defaults while allowing explicit low-level overrides."""
    defaults = ROLLOUT_PROFILES[args.rollout_profile]
    if args.record_per_step is None:
        args.record_per_step = defaults["record_per_step"]
    if args.fast_simulation is None:
        args.fast_simulation = defaults["fast_simulation"]
    if args.prompt_style is None:
        args.prompt_style = defaults["prompt_style"]
    if args.qwen_max_tokens is None:
        args.qwen_max_tokens = (
            64 if args.rollout_profile == "optimized" else 128
        )
    return args


def build_rollout_plan(
    maps: Iterable[MapSpec],
    difficulties: Iterable[str],
    rounds: int,
    repo_root: Path = REPO_ROOT,
    task_limit: int | None = None,
    task_indices: Iterable[int] | None = None,
    env_modes: Iterable[str] = ("realtime", "static"),
) -> list[RolloutSpec]:
    """Build the requested easy-first ordering without duplicate cells."""
    difficulties = tuple(difficulties)
    if not difficulties:
        raise ValueError("At least one difficulty is required")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    selected_indices = None if task_indices is None else tuple(task_indices)
    maps_with_tasks: list[
        tuple[MapSpec, list[tuple[int, dict[str, Any]]]]
    ] = []
    for map_spec in maps:
        tasks = list(enumerate(task_records(repo_root / map_spec.task_file)))
        if selected_indices is not None:
            invalid = [
                index for index in selected_indices
                if index < 0 or index >= len(tasks)
            ]
            if invalid:
                raise ValueError(
                    f"Task indices {invalid} are invalid for {map_spec.name}; "
                    f"valid range is 0..{len(tasks) - 1}"
                )
            by_index = dict(tasks)
            tasks = [(index, by_index[index]) for index in selected_indices]
        if task_limit is not None:
            tasks = tasks[:task_limit]
        maps_with_tasks.append((map_spec, tasks))

    selected_env_modes = tuple(env_modes)
    invalid_env_modes = sorted(set(selected_env_modes) - {"realtime", "static"})
    if invalid_env_modes or not selected_env_modes:
        raise ValueError(
            f"Invalid env modes {invalid_env_modes}; choose realtime and/or static"
        )

    lead = difficulties[0]
    axes: list[tuple[str, str, str]] = []
    if "realtime" in selected_env_modes:
        axes.append(("01_lead_realtime", lead, "realtime"))
    if "static" in selected_env_modes:
        axes.append(("02_lead_static", lead, "static"))
    for difficulty in difficulties[1:]:
        for env_mode in ("realtime", "static"):
            if env_mode in selected_env_modes:
                axes.append(("03_remaining", difficulty, env_mode))

    plan: list[RolloutSpec] = []
    for phase, difficulty, env_mode in axes:
        for map_spec, tasks in maps_with_tasks:
            for task_index, task in tasks:
                for round_id in range(1, rounds + 1):
                    plan.append(
                        RolloutSpec(
                            ordinal=len(plan) + 1,
                            phase=phase,
                            map_name=map_spec.name,
                            ue_map=map_spec.ue_map,
                            task_file=map_spec.task_file,
                            task_index=task_index,
                            task_id=task.get("task_id", task_index),
                            difficulty=difficulty,
                            env_mode=env_mode,
                            realtime_thinking=env_mode == "realtime",
                            round_id=round_id,
                        )
                    )
    return plan


def tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def tcp_listening(host: str, port: int) -> bool:
    """Check for a kernel TCP listener without opening a client connection.

    UnrealCV treats every successful TCP connect as a real protocol client.
    A conventional connect-based readiness probe can therefore displace the
    benchmark's response channel even when the probe immediately closes.
    Reading the kernel socket table is side-effect free.
    """
    del host  # The benchmark listeners are local; accept wildcard or loopback.
    expected_port = f"{int(port):04X}"
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            rows = table.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            if len(fields) < 4 or fields[3] != "0A":  # TCP_LISTEN
                continue
            if fields[1].rsplit(":", 1)[-1].upper() == expected_port:
                return True
    return False


def tcp_bind_available(host: str, port: int) -> bool:
    """Return whether a fresh wildcard listener can claim ``port`` now.

    A connect probe can report a closed port while an old accepted connection
    still prevents UnrealCV (which does not enable address reuse) from binding
    it. Probe the operation the renderer actually needs instead.
    """
    bind_host = "" if host in {"127.0.0.1", "localhost", "0.0.0.0"} else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((bind_host, port))
        return True
    except OSError:
        return False


def qwen_models_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/models"


def qwen_model_ids(base_url: str, timeout: float = 5.0) -> list[str]:
    try:
        with urllib.request.urlopen(qwen_models_url(base_url), timeout=timeout) as response:
            if response.status != 200:
                return []
            payload = json.loads(response.read().decode("utf-8"))
            return [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]
    except (OSError, ValueError, urllib.error.URLError):
        return []


def canonical_qwen_model_id(model_id: str) -> str:
    """Normalize a local alias and the official Qwen Instruct service ID."""
    basename = str(model_id).strip().rstrip("/").rsplit("/", 1)[-1].lower()
    if basename.endswith("-instruct"):
        basename = basename[: -len("-instruct")]
    return re.sub(r"[^a-z0-9]+", "", basename)


def qwen_model_matches(requested: str, served: str) -> bool:
    requested_id = canonical_qwen_model_id(requested)
    return bool(requested_id) and requested_id == canonical_qwen_model_id(served)


def resolve_qwen_model_id(requested: str, served_ids: Iterable[str]) -> str | None:
    return next(
        (served for served in served_ids if qwen_model_matches(requested, served)),
        None,
    )


def qwen_ready(base_url: str, expected_model: str | None = None, timeout: float = 5.0) -> bool:
    model_ids = qwen_model_ids(base_url, timeout=timeout)
    return bool(model_ids) and (
        expected_model is None
        or any(qwen_model_matches(expected_model, model_id) for model_id in model_ids)
    )


def qwen_supports_hard_thinking_budget(base_url: str, timeout: float = 5.0) -> bool:
    try:
        url = base_url.rstrip("/") + "/simworld/capabilities"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("hard_thinking_budget") is True
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def solid_png_data_url(width: int, height: int) -> str:
    """Create a dependency-free neutral RGB image for model warmup."""
    width = max(1, int(width))
    height = max(1, int(height))
    row = b"\x00" + (b"\x7f\x7f\x7f" * width)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(row * height, level=1))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def qwen_warmup_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build a one-image request matching the benchmark observation shape."""
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": solid_png_data_url(
                                args.observation_width,
                                args.observation_height,
                            )
                        },
                    },
                    {
                        "type": "text",
                        "text": "Warm up vision inference. Reply exactly: OK",
                    },
                ],
            }
        ],
        "max_tokens": 4,
        "temperature": 0,
    }


def parse_ue_renderer_log(text: str) -> dict[str, Any]:
    """Extract the selected Vulkan device and command-line fallback state."""
    names = re.findall(r"DeviceName:\s*([^\r\n]+)", text, flags=re.IGNORECASE)
    types = re.findall(r"Type=(VK_PHYSICAL_DEVICE_TYPE_[A-Z_]+)", text)
    return {
        "device_name": names[-1].strip() if names else None,
        "device_type": types[-1].strip() if types else None,
        "adapter_fallback": bool(
            re.search(
                r"graphics adapter .*falling back to first device",
                text,
                flags=re.IGNORECASE,
            )
        ),
    }


def validate_ue_renderer_log(text: str, requested_adapter: str) -> dict[str, Any]:
    """Reject silent Vulkan fallback and CPU/software rendering."""
    renderer = parse_ue_renderer_log(text)
    name = str(renderer.get("device_name") or "")
    device_type = str(renderer.get("device_type") or "")
    if renderer["adapter_fallback"]:
        raise UEInfrastructureError(
            f"UE graphics adapter {requested_adapter} is unavailable; Vulkan fell back "
            "to the first device"
        )
    if "llvmpipe" in name.lower() or device_type.endswith("_CPU"):
        raise UEInfrastructureError(
            f"UE graphics adapter {requested_adapter} selected software renderer "
            f"{name or device_type}"
        )
    return renderer


class ProcessGroup:
    """A locally-owned process group with recoverable logs and bounded shutdown."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.process: subprocess.Popen | None = None
        self.log_stream = None
        self.command: list[str] = []
        self.log_path: Path | None = None

    def start(
        self,
        command: list[str],
        cwd: Path,
        log_path: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        if self.alive:
            return
        if self.process is not None:
            # Reap/close resources from an earlier instance before replacing
            # the Popen handle during a service restart.
            self.stop(grace_seconds=1.0)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_stream = log_path.open("ab", buffering=0)
        self.command = command
        self.log_path = log_path
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=self.log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @staticmethod
    def _live_members(process_group: int) -> list[int]:
        """Return non-zombie Linux processes still belonging to a group.

        The packaged UE launcher is a shell wrapper.  That wrapper can exit
        before its SimWorld child, so Popen.poll() alone is not a sufficient
        shutdown check.  Zombies are excluded because they own no sockets or
        GPU resources and may briefly wait for PID 1 to reap them.
        """
        members: list[int] = []
        try:
            entries = Path("/proc").iterdir()
        except OSError:
            return members
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "stat").read_text(encoding="utf-8")
                fields = raw[raw.rfind(")") + 2 :].split()
                state, group = fields[0], int(fields[2])
            except (OSError, ValueError, IndexError):
                continue
            if group == process_group and state != "Z":
                members.append(int(entry.name))
        return members

    def stop(self, grace_seconds: float = 20.0) -> float:
        started = time.monotonic()
        process = self.process
        self.process = None
        if process is not None:
            process_group = process.pid
            try:
                os.killpg(process_group, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + grace_seconds
            while self._live_members(process_group) and time.monotonic() < deadline:
                process.poll()  # reap the launcher if it has already exited
                time.sleep(0.1)
            if self._live_members(process_group):
                try:
                    os.killpg(process_group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if self.log_stream is not None:
            self.log_stream.close()
            self.log_stream = None
        return time.monotonic() - started


class QwenServer:
    def __init__(self, args: argparse.Namespace, log_dir: Path) -> None:
        self.args = args
        self.log_dir = log_dir
        self.owned = False
        self.process = ProcessGroup("Qwen")
        self.startup_seconds = 0.0
        self.warmup_seconds = 0.0

    def _default_command(self, model_path: Path) -> list[str]:
        """Launch the low-latency image-only vLLM server used by this runner."""
        multimodal_limits = {"video": 0}
        qwen_max_images = getattr(self.args, "qwen_max_images", None)
        if qwen_max_images is not None:
            multimodal_limits["image"] = qwen_max_images
        return [
            self.args.qwen_python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model_path),
            "--served-model-name",
            self.args.model,
            "--host",
            self.args.qwen_host,
            "--port",
            str(self.args.qwen_port),
            "--gpu-memory-utilization",
            str(self.args.qwen_gpu_memory_utilization),
            "--max-model-len",
            str(self.args.qwen_max_model_len),
            "--max-num-seqs",
            "1",
            "--limit-mm-per-prompt",
            json.dumps(multimodal_limits, separators=(",", ":")),
            "--generation-config",
            "vllm",
        ]

    def _warmup(self) -> None:
        if not self.args.qwen_warmup:
            return
        started = time.monotonic()
        request = urllib.request.Request(
            self.args.qwen_url.rstrip("/") + "/chat/completions",
            data=json.dumps(qwen_warmup_payload(self.args)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.args.model_request_timeout,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if response.status != 200 or not payload.get("choices"):
                    raise RuntimeError(f"unexpected response: {payload}")
        except Exception as exc:
            raise RuntimeError(
                f"Qwen vision warmup failed at {request.full_url}: {exc}"
            ) from exc
        self.warmup_seconds += time.monotonic() - started

    def _require_reasoning_budget_capability(self) -> None:
        if getattr(self.args, "reasoning_budget_tokens", None) is None:
            return
        if not qwen_supports_hard_thinking_budget(self.args.qwen_url):
            raise RuntimeError(
                "Hard thinking budgets require the bundled controlled Qwen server"
            )

    def ensure_ready(self) -> None:
        model_ids = qwen_model_ids(self.args.qwen_url)
        resolved_model = resolve_qwen_model_id(self.args.model, model_ids)
        if resolved_model is not None:
            self.args.model = resolved_model
            self._require_reasoning_budget_capability()
            self._warmup()
            return
        if model_ids:
            raise RuntimeError(
                f"Qwen endpoint {self.args.qwen_url} serves {model_ids}, not "
                f"the requested model {self.args.model!r}; select a different port"
            )
        if self.args.qwen_mode == "external":
            started = time.monotonic()
            deadline = started + self.args.qwen_ready_timeout
            print(
                f"[all-maps] waiting up to {self.args.qwen_ready_timeout:g}s for "
                f"external Qwen at {qwen_models_url(self.args.qwen_url)}",
                flush=True,
            )
            while time.monotonic() < deadline:
                time.sleep(2)
                model_ids = qwen_model_ids(self.args.qwen_url)
                resolved_model = resolve_qwen_model_id(self.args.model, model_ids)
                if resolved_model is not None:
                    self.args.model = resolved_model
                    self.startup_seconds += time.monotonic() - started
                    self._require_reasoning_budget_capability()
                    self._warmup()
                    return
                if model_ids:
                    raise RuntimeError(
                        f"Qwen endpoint {self.args.qwen_url} serves {model_ids}, not "
                        f"the requested model {self.args.model!r}; select a different port"
                    )
            raise TimeoutError(
                f"External Qwen did not become ready at "
                f"{qwen_models_url(self.args.qwen_url)} within "
                f"{self.args.qwen_ready_timeout:g}s"
            )

        model_path = Path(self.args.qwen_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Qwen model path does not exist: {model_path}")
        if self.args.qwen_launch_command:
            command = shlex.split(self.args.qwen_launch_command)
        else:
            command = self._default_command(model_path)
        environment = os.environ.copy()
        if getattr(self.args, "qwen_pythonpath", None):
            environment["PYTHONPATH"] = str(self.args.qwen_pythonpath)
        if self.args.qwen_gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(self.args.qwen_gpu)
        started = time.monotonic()
        self.process.start(
            command,
            REPO_ROOT,
            self.log_dir / "qwen_server.log",
            environment,
        )
        self.owned = True
        deadline = started + self.args.qwen_ready_timeout
        while time.monotonic() < deadline:
            if qwen_ready(self.args.qwen_url, self.args.model):
                self.startup_seconds = time.monotonic() - started
                self._require_reasoning_budget_capability()
                self._warmup()
                return
            if not self.process.alive:
                raise RuntimeError(
                    f"Qwen exited during startup; inspect {self.process.log_path}"
                )
            time.sleep(2)
        raise TimeoutError(
            f"Qwen did not become ready in {self.args.qwen_ready_timeout}s; "
            f"inspect {self.process.log_path}"
        )

    def stop(self) -> None:
        if self.owned:
            self.process.stop(self.args.process_shutdown_timeout)


class UEServer:
    def __init__(self, args: argparse.Namespace, log_path: Path) -> None:
        self.args = args
        self.process = ProcessGroup("UE")
        self.log_path = log_path
        self.internal_log_path = self.log_path.with_name(
            self.log_path.stem + "_internal.log"
        )
        self.startup_seconds = 0.0
        self.renderer_info: dict[str, Any] = {}

    def _gpu_startup_error(self) -> str | None:
        """Return a fatal startup diagnostic visible before the port timeout."""
        try:
            log_text = self.internal_log_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            return None

        normalized = log_text.lower()
        if "cannot create a vulkan device" in normalized:
            return (
                f"UE could not create a Vulkan device on adapter "
                f"{self.args.ue_gpu}; check free graphics memory and competing "
                "renderer processes"
            )
        if "cannot start listening on port" in normalized:
            return (
                f"UE could not bind UnrealCV port {self.args.ue_port}; "
                "a previous renderer may still be releasing it"
            )
        if "falling back to first device" in normalized:
            return (
                f"UE rejected requested Vulkan adapter {self.args.ue_gpu} "
                "and fell back to another device"
            )

        software_markers = (
            "llvmpipe",
            "lavapipe",
            "swiftshader",
        )
        if any(marker in normalized for marker in software_markers):
            return (
                f"UE selected a software Vulkan renderer for adapter "
                f"{self.args.ue_gpu}"
            )

        return None

    def start(self, ue_map: str) -> None:
        launcher = Path(self.args.ue_launcher)
        if not launcher.is_file():
            raise FileNotFoundError(f"UE launcher does not exist: {launcher}")
        if tcp_listening(self.args.ue_host, self.args.ue_port):
            raise RuntimeError(
                f"UE port {self.args.ue_host}:{self.args.ue_port} is already occupied; "
                "refusing to terminate an unowned process"
            )
        command = [
            str(launcher),
            "-RenderoffScreen",
            ue_map,
            f"-graphicsadapter={self.args.ue_gpu}",
            # This UnrealCV build expects the port as the next token.
            "-cvport",
            str(self.args.ue_port),
            f"-ResX={self.args.ue_width}",
            f"-ResY={self.args.ue_height}",
            f"-FPSMAX={self.args.ue_fps}",
            # The packaged benchmark project already fixes r.RayTracing=0 and
            # r.Lumen.HardwareRayTracing=0.  Make that device requirement
            # explicit at process launch so Vulkan does not request unused
            # ray-tracing features and fail vkCreateDevice under otherwise
            # sufficient shared-GPU memory pressure.
            "-noraytracing",
            # The shared packaged runtime's Saved/Logs directory may belong to
            # another server user. Keep UE's internal log beside this attempt's
            # captured console log so failures retain their final diagnostic.
            f"-abslog={self.internal_log_path}",
        ]
        if os.environ.get(
            "SIMWORLD_UE_FORCE_WINDOWED", ""
        ).strip().lower() in {"1", "true", "yes", "on"}:
            # The packaged runtime can hang its render thread while switching
            # the off-screen viewport into WindowedFullscreen after UnrealCV
            # has already opened its listener.  Explicit windowed mode avoids
            # that redundant transition without changing capture resolution.
            command.append("-windowed")
        # Opt-in extra launch arguments, for example a longer render-fence
        # timeout for slow startups on shared GPUs:
        # -ini:Engine:[ConsoleVariables]:g.TimeoutForBlockOnRenderFence=600000
        command.extend(shlex.split(os.environ.get("SIMWORLD_UE_EXTRA_ARGS", "")))
        if self.args.ue_no_rhi_thread:
            command.append("-norhithread")
        environment = os.environ.copy()
        started = time.monotonic()
        self.process.start(command, launcher.parent, self.log_path, environment)
        deadline = started + self.args.ue_ready_timeout
        while time.monotonic() < deadline:
            gpu_error = self._gpu_startup_error()
            if gpu_error is not None:
                raise RuntimeError(
                    f"{gpu_error}; inspect {self.internal_log_path}"
                )

            if tcp_listening(self.args.ue_host, self.args.ue_port):
                # Re-read after readiness so a late-flushed device diagnostic
                # cannot be missed immediately before returning success.
                gpu_error = self._gpu_startup_error()
                if gpu_error is not None:
                    raise RuntimeError(
                        f"{gpu_error}; inspect {self.internal_log_path}"
                    )
                if self.args.ue_settle_seconds > 0:
                    time.sleep(self.args.ue_settle_seconds)
                try:
                    renderer_log = self.log_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                except OSError:
                    renderer_log = ""
                self.renderer_info = validate_ue_renderer_log(
                    renderer_log,
                    str(self.args.ue_gpu),
                )
                self.startup_seconds = time.monotonic() - started
                return
            if not self.process.alive:
                raise RuntimeError(f"UE exited during startup; inspect {self.log_path}")
            time.sleep(1)
        raise TimeoutError(
            f"UE did not open {self.args.ue_host}:{self.args.ue_port} in "
            f"{self.args.ue_ready_timeout}s; inspect {self.log_path}"
        )

    def healthy(self) -> bool:
        return self.process.alive and tcp_listening(
            self.args.ue_host, self.args.ue_port
        )

    def stop(self) -> float:
        shutdown_seconds = self.process.stop(
            self.args.process_shutdown_timeout
        )
        # A closed connect probe is insufficient: an accepted UnrealCV
        # connection can retain kernel TCP state that prevents the next UE
        # listener from binding. Wait until the same non-reuse bind operation
        # succeeds instead of spending several full renderer startups retrying.
        release_deadline = time.monotonic() + 75.0
        while (
            not tcp_bind_available(self.args.ue_host, self.args.ue_port)
            and time.monotonic() < release_deadline
        ):
            time.sleep(0.25)
        return shutdown_seconds


class UERequestTimeout(TimeoutError):
    pass


class UEInfrastructureError(RuntimeError):
    pass


class UnrealCVRequestMetrics:
    """Count blocking UnrealCV time and make single requests genuinely timeout."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self.health_probe = None
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "lock", threading.Lock()):
            self.request_count = 0
            self.failed_request_count = 0
            self.wall_seconds = 0.0

    def record(self, elapsed: float, failed: bool) -> None:
        with self.lock:
            self.request_count += 1
            self.failed_request_count += int(failed)
            self.wall_seconds += elapsed

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "request_count": self.request_count,
                "failed_request_count": self.failed_request_count,
                "wall_seconds": self.wall_seconds,
            }

    def install(self, unrealcv_package: Any) -> None:
        if getattr(unrealcv_package.Client.request, "_all_maps_instrumented", False):
            return
        tracker = self

        def request(client: Any, message: Any, timeout: float = 5):
            started = time.monotonic()
            failed = False
            try:
                if timeout < 0:
                    if isinstance(message, list):
                        client.request_batch_async(message)
                    else:
                        client.request_async(message)
                    return True

                messages = message if isinstance(message, list) else [message]
                for item in messages:
                    if not isinstance(item, bytes):
                        item = item.encode("utf-8")
                    raw = b"%d:%s" % (client.send_message_id, item)
                    if not client.send(raw):
                        raise ConnectionError("Failed to send: socket is closed")
                    client.send_message_id += 1
                client.recv_num_q.put(-len(messages))
                replies = []
                request_timeout = tracker.timeout_seconds if tracker.timeout_seconds > 0 else timeout
                for _ in messages:
                    deadline = time.monotonic() + request_timeout
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            try:
                                client.disconnect()
                            except Exception:
                                pass
                            raise UERequestTimeout(
                                f"UnrealCV request exceeded {request_timeout}s: {message!r}"
                            )
                        try:
                            reply = client.recv_data_q.get(timeout=min(1.0, remaining))
                            break
                        except queue.Empty:
                            if tracker.health_probe is not None and not tracker.health_probe():
                                try:
                                    client.disconnect()
                                except Exception:
                                    pass
                                raise ConnectionError(
                                    f"UE process exited during UnrealCV request: {message!r}"
                                )
                    if reply is None:
                        try:
                            client.disconnect()
                        except Exception:
                            pass
                        raise ConnectionError(
                            f"UE disconnected while handling UnrealCV request: {message!r}"
                        )
                    replies.append(reply)
                return replies if isinstance(message, list) else replies[0]
            except Exception:
                failed = True
                raise
            finally:
                tracker.record(time.monotonic() - started, failed)

        request._all_maps_instrumented = True
        request._simworld_request_metrics = tracker
        unrealcv_package.Client.request = request


def run_agent_until_terminal(manager) -> None:
    """Run until success, a terminal safety failure, or the step budget."""
    while manager.agent.step_num < manager.max_steps:
        if getattr(manager.agent, "success", False):
            break
        if getattr(manager.agent, "failed", False):
            break
        manager.agent.step(mode=manager.control_mode)
    if getattr(manager.agent, "success", False):
        manager._drain_signal_traffic_after_agent_completion()

def configure_unrealcv_request_timeout(
    request_timeout: float,
    reconnect_retries: int = 0,
) -> float:
    """Make the RPC deadline and retry policy authoritative per client."""
    value = float(request_timeout)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("UnrealCV request timeout must be positive and finite")
    retries = int(reconnect_retries)
    if retries < 0:
        raise ValueError("UnrealCV reconnect retries must be non-negative")
    os.environ["SIMWORLD_UNREALCV_REQUEST_TIMEOUT_SECONDS"] = f"{value:g}"
    os.environ["SIMWORLD_UNREALCV_RECONNECT_RETRIES"] = str(retries)
    return value


def prepare_runtime(
    request_timeout: float,
    reconnect_retries: int = 0,
    verbose_logging: bool = False,
):
    # ``.py312deps`` contains the pure-Python UnrealCV client, but it also
    # contains CPython-3.12 wheels (notably NumPy/OpenCV).  The model runtime
    # currently uses Python 3.11, so putting the whole directory permanently
    # at the front of ``sys.path`` makes those incompatible extension modules
    # shadow the working ones from the conda environment.  On non-3.12
    # runtimes, cache the native NumPy/OpenCV modules first and expose the
    # local directory only while importing UnrealCV.
    configure_unrealcv_request_timeout(request_timeout, reconnect_retries)
    native_local_deps = REPO_ROOT / f".py{sys.version_info.major}{sys.version_info.minor}deps"
    if native_local_deps.is_dir() and str(native_local_deps) not in sys.path:
        sys.path.insert(0, str(native_local_deps))

    unrealcv_package = None
    local_deps = str(LOCAL_DEPS)
    if LOCAL_DEPS.is_dir() and local_deps not in sys.path:
        if sys.version_info[:2] == (3, 12):
            sys.path.insert(0, local_deps)
        else:
            import cv2  # noqa: F401 - cache the ABI-compatible environment wheel
            import numpy  # noqa: F401 - cache the ABI-compatible environment wheel
            import PIL.Image  # noqa: F401 - cache the ABI-compatible Pillow wheel

            sys.path.insert(0, local_deps)
            try:
                import unrealcv as unrealcv_package
            finally:
                sys.path.remove(local_deps)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    simworld = types.ModuleType("simworld")
    simworld.__path__ = [str(REPO_ROOT / "SimWorld" / "simworld")]
    sys.modules.setdefault("simworld", simworld)

    if unrealcv_package is None:
        import unrealcv as unrealcv_package
    from base.rt_communicator import RTCommunicator
    from base.rt_unrealcv import RTUnrealCV
    from manager.world_manager import WorldManager

    metrics = UnrealCVRequestMetrics(request_timeout)
    metrics.install(unrealcv_package)

    class StrictWorldManager(WorldManager):
        """WorldManager variant that lets infrastructure errors reach the supervisor."""

        last_result_path: str | None = None
        model_inference_wall_seconds: float = 0.0

        def run(self):
            original_generate = self.agent.llm.generate_response_openai

            def measured_generate(*args, **kwargs):
                result = original_generate(*args, **kwargs)
                if isinstance(result, tuple) and len(result) > 1:
                    try:
                        self.model_inference_wall_seconds += float(result[1] or 0.0)
                    except (TypeError, ValueError):
                        pass
                return result

            self.agent.llm.generate_response_openai = measured_generate
            # Use the base implementation so transport/runtime failures are
            # recorded as explicit failed terminal results.  The old custom
            # loop skipped that fail-closed behavior and could publish a
            # misleading max_steps result for a crashed rollout.
            return super().run()

    return RTCommunicator, RTUnrealCV, StrictWorldManager, metrics


def result_from_run_dir(run_dir: Path) -> Path | None:
    results = sorted(run_dir.glob("task_*.json"), key=lambda item: item.stat().st_mtime)
    return results[-1] if results else None


def step_artifact_summary(run_dir: Path) -> dict[str, Any]:
    manifests = sorted(run_dir.glob("task_*_steps/step_*/step_*_manifest.json"))
    inference_seconds = 0.0
    missing: list[str] = []
    input_images = 0
    output_images = 0
    for manifest_path in manifests:
        data = read_json(manifest_path)
        metrics = data.get("metrics") or {}
        inference_seconds += float(metrics.get("response_time") or 0.0)
        step_dir = manifest_path.parent
        for filename in data.get("input_images") or []:
            input_images += 1
            if not (step_dir / filename).is_file():
                missing.append(str(step_dir / filename))
        output = data.get("output_image")
        if output:
            output_images += 1
            if not (step_dir / output).is_file():
                missing.append(str(step_dir / output))
    return {
        "step_manifest_count": len(manifests),
        "input_image_count": input_images,
        "output_image_count": output_images,
        "model_inference_wall_seconds": inference_seconds,
        "missing_artifacts": missing,
    }


def run_dir_for(suite_dir: Path, spec: RolloutSpec) -> Path:
    return (
        suite_dir
        / "runs"
        / spec.phase
        / spec.map_name
        / spec.difficulty
        / spec.env_mode
        / f"task_{spec.task_index:03d}_{sanitize(str(spec.task_id))}"
        / f"round_{spec.round_id:03d}"
    )


def terminal_status(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        status = read_json(path)
    except Exception:
        return None
    return status if status.get("status") in {"completed", "error"} else None


def error_record_is_retryable(record: dict[str, Any]) -> bool:
    """Honor explicit deterministic failures; retry legacy errors by default."""
    return (
        record.get("status") == "error"
        and record.get("retryable", True) is not False
    )


def rollout_reached_terminal_state(result: dict[str, Any]) -> bool:
    """Distinguish valid task failure from an episode cut short by UE I/O."""
    if result.get("success") is True or result.get("failed") is True:
        return True
    try:
        return int(result["final_step"]) >= int(result["max_steps"])
    except (KeyError, TypeError, ValueError):
        return False


def rollout_termination_reason(result: dict[str, Any]) -> str | None:
    """Return the canonical terminal outcome for a completed rollout."""
    explicit = result.get("termination_reason")
    if explicit:
        return str(explicit)
    if result.get("success") is True:
        return "success"
    if result.get("failed") is True:
        return str(result.get("failure_reason") or "failed")
    try:
        if int(result["final_step"]) >= int(result["max_steps"]):
            return "max_steps"
    except (KeyError, TypeError, ValueError):
        pass
    return None


def normalize_completed_status(status: dict[str, Any]) -> dict[str, Any]:
    """Repair legacy completed records that omitted their terminal outcome."""
    normalized = dict(status)
    if normalized.get("status") != "completed" or normalized.get(
        "termination_reason"
    ):
        return normalized
    result_name = normalized.get("result_path")
    if not result_name:
        return normalized
    try:
        result = read_json(REPO_ROOT / result_name)
    except (OSError, ValueError, TypeError):
        return normalized
    reason = rollout_termination_reason(result)
    if reason is not None:
        normalized["termination_reason"] = reason
    return normalized


def completed_status_is_valid(status: dict[str, Any]) -> bool:
    result_name = status.get("result_path")
    if not result_name:
        return False
    result_path = REPO_ROOT / result_name
    try:
        return result_path.is_file() and rollout_reached_terminal_state(read_json(result_path))
    except (OSError, ValueError, TypeError):
        return False


def archive_partial_attempt(run_dir: Path, label: str) -> Path | None:
    """Move incomplete result/artifact trees aside before another attempt."""
    outputs = (
        list(run_dir.glob("task_*.json"))
        + list(run_dir.glob("task_*_steps"))
        + [
            path
            for filename in (
                "rollout_agent_input_annotated.mp4",
                "rollout_alignment_check.json",
                "rollout_input_video_report.json",
                "rollout_traffic_e2e_audit.json",
            )
            if (path := run_dir / filename).exists()
        ]
    )
    if not outputs:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = run_dir / "failed_attempts" / f"{sanitize(label)}_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    for output in outputs:
        output.replace(destination / output.name)
    return destination


def next_ue_attempt_number(run_dir: Path) -> int:
    """Return a log number that will not overwrite an earlier UE attempt."""
    highest = 0
    for path in run_dir.glob("ue_attempt_*.log"):
        match = re.fullmatch(r"ue_attempt_(\d+)\.log", path.name)
        if match is not None:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def is_ue_failure(exc: BaseException, ue: UEServer | None) -> bool:
    if isinstance(
        exc,
        (UERequestTimeout, UEInfrastructureError, ConnectionError, BrokenPipeError, socket.error),
    ):
        return True
    text = str(exc).lower()
    # Do not classify a Qwen/OpenAI API connection failure as a UE crash.
    markers = (
        "unrealcv",
        "ue disconnected",
        "ue process exited",
        "ue exited",
        "ue did not open",
        "socket is closed",
    )
    return (ue is not None and not ue.healthy()) or any(marker in text for marker in markers)


def write_agent_config(path: Path, args: argparse.Namespace) -> None:
    reasoning_budget_tokens = getattr(
        args, "reasoning_budget_tokens", None
    )
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking}
    }
    if reasoning_budget_tokens is not None:
        extra_body.update(
            {
                "reasoning_budget_mode": "force_close",
                "reasoning_budget_tokens": reasoning_budget_tokens,
                "reasoning_budget_answer_tokens": getattr(
                    args, "reasoning_budget_answer_tokens", 128
                ),
                "reasoning_budget_force_close": True,
            }
        )
    write_json(
        path,
        {
            "experiment_contract_sha256": getattr(
                args,
                "experiment_contract_sha256",
                None,
            ),
            "llm": [
                {
                    "model": args.model,
                    "provider": "self-hosted",
                    "url": args.qwen_url,
                    "reasoning": args.enable_thinking,
                    "max_tokens": args.qwen_max_tokens,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "seed": args.seed,
                    "request_timeout": args.model_request_timeout,
                    # A transport/service failure is infrastructure, not a
                    # model parse error. Propagate it so the cell-level retry
                    # policy restarts the rollout from a clean environment.
                    "raise_on_api_error": True,
                    "extra_body": extra_body,
                }
            ]
        },
    )


def ensure_qwen_ready_and_sync_agent_config(
    qwen: QwenServer,
    agent_config: Path,
    args: argparse.Namespace,
) -> None:
    """Keep rollout requests aligned with the model ID served after a restart."""
    qwen.ensure_ready()
    # ensure_ready() may resolve a semantically equivalent endpoint alias.  The
    # rollout reads its model ID from this file, so refresh it before UE starts.
    write_agent_config(agent_config, args)


def execute_rollout(
    args: argparse.Namespace,
    suite_dir: Path,
    spec: RolloutSpec,
    runtime: tuple[Any, Any, Any, UnrealCVRequestMetrics],
) -> dict[str, Any]:
    RTCommunicator, RTUnrealCV, StrictWorldManager, request_metrics = runtime
    run_dir = run_dir_for(suite_dir, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "run_status.json"
    if args.resume:
        previous = terminal_status(status_path)
        if previous is not None:
            previous = normalize_completed_status(previous)
            write_json(status_path, previous)
        if previous is not None and (
            (
                previous.get("status") == "completed"
                and completed_status_is_valid(previous)
            )
            or (
                previous.get("status") == "error"
                and not error_record_is_retryable(previous)
            )
            or not args.retry_errors
        ):
            return {**previous, "resume_action": "skipped_terminal"}
        if previous is not None:
            archive_partial_attempt(run_dir, f"resume_{previous.get('status', 'unknown')}")
        elif status_path.is_file():
            # A killed supervisor leaves status=running and may also leave a
            # partial result/step tree.  Treat it as an interrupted attempt.
            archive_partial_attempt(run_dir, "resume_interrupted")

    condition = {**asdict(spec), "rollout_id": spec.rollout_id}
    contract_id = getattr(args, "experiment_contract_sha256", None)
    if contract_id is not None:
        condition["experiment_contract_sha256"] = contract_id
    write_json(run_dir / "condition.json", condition)
    write_json(
        status_path,
        {
            **condition,
            "status": "running",
            "started_at": utc_now(),
            "attempt": 0,
        },
    )
    agent_config = suite_dir / "agents_qwen3vl8b.json"
    last_error: BaseException | None = None

    first_ue_attempt = next_ue_attempt_number(run_dir)
    for attempt in range(1, args.ue_max_attempts + 1):
        ue_attempt = first_ue_attempt + attempt - 1
        ue: UEServer | None = None
        communicator = None
        condition_started = time.monotonic()
        attempt_started_at = utc_now()
        ue_shutdown_seconds = 0.0
        write_json(
            status_path,
            {
                **condition,
                "status": "starting_ue",
                "started_at": attempt_started_at,
                "attempt": ue_attempt,
            },
        )
        try:
            request_metrics.reset()
            log_path = run_dir / f"ue_attempt_{ue_attempt:02d}.log"
            write_attempt_runtime(run_dir, ue_attempt, args, log_path)
            ue = UEServer(args, log_path)
            ue.start(spec.ue_map)
            write_json(
                status_path,
                {
                    **condition,
                    "status": "running",
                    "started_at": attempt_started_at,
                    "attempt": ue_attempt,
                    "ue_startup_wall_seconds": ue.startup_seconds,
                },
            )
            request_metrics.health_probe = lambda: ue.process.alive
            rollout_started = time.monotonic()
            communicator = RTCommunicator(
                RTUnrealCV(
                    port=args.ue_port,
                    ip=args.ue_host,
                    # UnrealCV applies its constructor resolution immediately.
                    # Start at the policy-input size instead of performing a
                    # redundant legacy 1280x720 resize first.
                    resolution=(
                        args.observation_width,
                        args.observation_height,
                    ),
                )
            )
            manager = StrictWorldManager(
                communicator=communicator,
                agent_path=str(agent_config),
                task_file_path=spec.task_file,
                seed=args.seed + spec.round_id - 1,
                control_mode="llm",
                token_based=args.token_based,
                use_tick=True,
                realtime_thinking=spec.realtime_thinking,
                record_per_step=args.record_per_step,
                use_action_frames=args.use_action_frames,
                prompt_style=args.prompt_style,
                traffic_policy=args.traffic_policy,
                red_light_conflict_vehicle_enabled=(
                    args.red_light_conflict_vehicle
                ),
                red_light_conflict_vehicle_probability=(
                    args.conflict_vehicle_launch_probability
                ),
                static_signal_vehicles=args.static_signal_vehicles,
                conflict_vehicle_min_launch_distance_m=(
                    getattr(
                        args,
                        "conflict_vehicle_min_launch_distance_m",
                        3.0,
                    )
                ),
                conflict_vehicle_max_launch_distance_m=(
                    getattr(
                        args,
                        "conflict_vehicle_max_launch_distance_m",
                        9.0,
                    )
                ),
                conflict_vehicle_impact_radius_m=(
                    getattr(
                        args,
                        "conflict_vehicle_impact_radius_m",
                        1.0,
                    )
                ),
                conflict_vehicle_release_wait_timeout_s=(
                    getattr(
                        args,
                        "conflict_vehicle_release_wait_timeout_s",
                        12.0,
                    )
                ),
                pedestrians_enabled=args.pedestrians,
                movable_obstacles_enabled=args.movable_obstacles,
                irregular_npcs_enabled=args.irregular_npcs,
                falling_objects_enabled=args.falling_objects,
                load_all_unsafe_triggers=getattr(
                    args, "load_all_unsafe_triggers", False
                ),
                max_steps=args.max_steps,
                results_dir=str(run_dir),
            )
            manager._batch_suffix = spec.rollout_id
            manager.run_single_task(spec.task_index, difficulty=spec.difficulty)
            rollout_seconds = time.monotonic() - rollout_started
            result_path = Path(manager.last_result_path) if manager.last_result_path else result_from_run_dir(run_dir)
            if result_path is None or not result_path.is_file():
                raise RuntimeError("Rollout ended without an evaluation result JSON")
            result = read_json(result_path)
            if contract_id is not None:
                result["experiment_contract_sha256"] = contract_id
                write_json(result_path, result)
            if not rollout_reached_terminal_state(result):
                raise UEInfrastructureError(
                    "UE rollout ended before a terminal state "
                    f"(final_step={result.get('final_step')}, "
                    f"max_steps={result.get('max_steps')}, "
                    f"success={result.get('success')}, failed={result.get('failed')})"
                )
            artifacts = step_artifact_summary(run_dir) if args.record_per_step else {
                "step_manifest_count": 0,
                "input_image_count": 0,
                "output_image_count": 0,
                "model_inference_wall_seconds": manager.model_inference_wall_seconds,
                "missing_artifacts": [],
            }
            if args.record_per_step:
                if int(result.get("decision_count") or 0) > 0 and artifacts["step_manifest_count"] == 0:
                    raise RuntimeError("record_per_step produced no step manifests")
                if artifacts["missing_artifacts"]:
                    raise RuntimeError("record_per_step contains missing image artifacts")
            ue_metrics = request_metrics.snapshot()
            model_seconds = artifacts["model_inference_wall_seconds"]
            other_seconds = max(0.0, rollout_seconds - model_seconds - ue_metrics["wall_seconds"])
            # Close the client side first. If UE closes an active accepted
            # socket during process teardown, its listening port can remain
            # non-bindable for about a minute even though no listener exists.
            if communicator is not None:
                try:
                    communicator.unrealcv.disconnect()
                except Exception:
                    pass
                communicator = None
            ue_shutdown_seconds = ue.stop()
            condition_seconds = time.monotonic() - condition_started
            record = {
                **condition,
                "status": "completed",
                "attempt": ue_attempt,
                "finished_at": utc_now(),
                "success": result.get("success"),
                "failed": result.get("failed"),
                "termination_reason": rollout_termination_reason(result),
                "result_path": serialize_manifest_path(result_path),
                "ue_renderer": dict(ue.renderer_info),
                "artifact_summary": artifacts,
                "benchmark_metrics": result.get("benchmark_metrics", {}),
                "timing": {
                    "sim_time_seconds": result.get("sim_time"),
                    "ue_startup_wall_seconds": ue.startup_seconds,
                    "rollout_wall_seconds": rollout_seconds,
                    "model_inference_wall_seconds": model_seconds,
                    "ue_request_wall_seconds": ue_metrics["wall_seconds"],
                    "other_rollout_wall_seconds": other_seconds,
                    "ue_shutdown_wall_seconds": ue_shutdown_seconds,
                    "condition_wall_seconds": condition_seconds,
                    "ue_request_count": ue_metrics["request_count"],
                    "ue_failed_request_count": ue_metrics["failed_request_count"],
                },
            }
            write_json(status_path, record)
            return record
        except KeyboardInterrupt:
            write_json(
                status_path,
                {
                    **condition,
                    "status": "interrupted",
                    "attempt": ue_attempt,
                    "started_at": attempt_started_at,
                    "finished_at": utc_now(),
                    "interruption_reason": "keyboard_interrupt",
                },
            )
            raise
        except Exception as exc:
            last_error = exc
            infrastructure_failure = is_ue_failure(exc, ue)
            retry = infrastructure_failure and attempt < args.ue_max_attempts
            error_record = {
                **condition,
                "status": "retrying_ue" if retry else "error",
                "attempt": ue_attempt,
                "finished_at": utc_now(),
                "infrastructure_failure": infrastructure_failure,
                "retryable": bool(getattr(exc, "retryable", True)),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "timing": {
                    "condition_wall_seconds_before_shutdown": time.monotonic() - condition_started,
                    **request_metrics.snapshot(),
                },
            }
            write_json(status_path, error_record)
            if retry:
                archived = archive_partial_attempt(run_dir, f"ue_attempt_{ue_attempt:02d}")
                if archived is not None:
                    error_record["partial_artifacts_archived_to"] = (
                        serialize_manifest_path(archived)
                    )
                    write_json(status_path, error_record)
            if not retry:
                return error_record
        finally:
            if communicator is not None:
                try:
                    communicator.unrealcv.disconnect()
                except Exception:
                    pass
            request_metrics.health_probe = None
            if ue is not None and ue.process.process is not None:
                ue.stop()
        if args.retry_backoff_seconds:
            time.sleep(args.retry_backoff_seconds)

    raise AssertionError(f"unreachable after error: {last_error}")


def aggregate_summary(
    records: list[dict[str, Any]],
    plan_count: int,
    *,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    completed = [item for item in records if item.get("status") == "completed"]
    missing_contract_count = sum(
        not item.get("experiment_contract_sha256") for item in records
    )
    contract_ids = sorted({
        str(item["experiment_contract_sha256"])
        for item in records
        if item.get("experiment_contract_sha256")
    })
    if len(contract_ids) > 1:
        raise ValueError(
            "Refusing to aggregate records from different experiment contracts: "
            + ", ".join(contract_ids)
        )
    if contract_ids and missing_contract_count:
        raise ValueError(
            "Refusing to aggregate identified and unidentified experiment "
            f"contracts ({missing_contract_count} record(s) lack an ID)"
        )
    if (
        expected_contract_sha256 is not None
        and contract_ids
        and contract_ids != [expected_contract_sha256]
    ):
        raise ValueError(
            "Refusing to aggregate a record whose experiment contract differs "
            f"from the suite contract {expected_contract_sha256}: {contract_ids}"
        )
    paper_metrics = aggregate_metrics(completed)
    comparison_ineligibility_reasons = []
    if len(records) != plan_count:
        comparison_ineligibility_reasons.append(
            f"recorded {len(records)} of {plan_count} planned rollouts"
        )
    if len(completed) != plan_count:
        comparison_ineligibility_reasons.append(
            f"completed {len(completed)} of {plan_count} planned rollouts"
        )
    if missing_contract_count:
        comparison_ineligibility_reasons.append(
            f"{missing_contract_count} record(s) lack an experiment contract"
        )
    # Task completion alone is insufficient for a publishable comparison.  The
    # suite finalizer clears this gate only after every planned cell has a video
    # and passes the action/image and traffic/feedback audits.
    comparison_ineligibility_reasons.append(
        "artifact audits have not passed for every planned rollout"
    )
    return {
        "updated_at": utc_now(),
        "experiment_contract_sha256": (
            contract_ids[0] if contract_ids else expected_contract_sha256
        ),
        "planned_rollouts": plan_count,
        "recorded_rollouts": len(records),
        "completed_rollouts": len(completed),
        "comparison_eligible": not comparison_ineligibility_reasons,
        "comparison_ineligibility_reasons": comparison_ineligibility_reasons,
        "artifact_audits_passed": False,
        "artifact_audited_rollouts": 0,
        "error_rollouts": sum(item.get("status") == "error" for item in records),
        "timeout_error_rollouts": sum(
            item.get("status") == "error"
            and (
                "timeout" in str(item.get("error") or "").lower()
                or "timed out" in str(item.get("error") or "").lower()
            )
            for item in records
        ),
        "benchmark_metrics": paper_metrics,
        "successful_tasks": sum(item.get("success") is True for item in completed),
        "failed_tasks": sum(item.get("failed") is True for item in completed),
        "max_steps_tasks": sum(
            item.get("termination_reason") == "max_steps" for item in completed
        ),
        "non_successful_tasks": sum(
            item.get("success") is False for item in completed
        ),
        "sim_time_seconds": sum(float((item.get("timing") or {}).get("sim_time_seconds") or 0) for item in completed),
        "condition_wall_seconds": sum(float((item.get("timing") or {}).get("condition_wall_seconds") or 0) for item in completed),
        "model_inference_wall_seconds": sum(float((item.get("timing") or {}).get("model_inference_wall_seconds") or 0) for item in completed),
        "ue_request_wall_seconds": sum(float((item.get("timing") or {}).get("ue_request_wall_seconds") or 0) for item in completed),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--benchmark-profile-path", default=None)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--maps", nargs="+", default=[item.name for item in DEFAULT_MAPS])
    parser.add_argument("--difficulty-profile", choices=sorted(DIFFICULTY_PROFILES), default="legacy")
    parser.add_argument("--difficulties", nargs="+", default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument(
        "--env-modes",
        nargs="+",
        choices=["realtime", "static"],
        default=["realtime", "static"],
        help="Run only the selected environment timing modes.",
    )
    parser.add_argument(
        "--task-indices",
        nargs="+",
        type=int,
        default=None,
        help="Run only these zero-based task indices on each selected map",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--prompt-style", choices=["naive", "instructional", "future_state", "adaptive"], default="instructional")
    parser.add_argument(
        "--traffic-policy",
        choices=["visual_only", "safety_assisted"],
        default="visual_only",
        help=(
            "visual_only keeps traffic visible and active but removes "
            "symbolic traffic prompts and crossing overrides; "
            "safety_assisted retains the previous behavior"
        ),
    )
    parser.add_argument(
        "--red-light-conflict-vehicle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Launch a collision-enabled conflict vehicle after an eligible "
            "illegal crossing or red-light violation (default: enabled). Use "
            "--no-red-light-conflict-vehicle to keep violation evaluation "
            "without this scripted consequence."
        ),
    )
    parser.add_argument(
        "--conflict-vehicle-launch-probability",
        type=float,
        default=1.0,
        help=(
            "Probability of launching after each eligible traffic-rule "
            "violation (maintained benchmark default: 1.0). Values below "
            "1.0 are explicit ablations."
        ),
    )
    parser.add_argument(
        "--conflict-vehicle-min-launch-distance-m",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--conflict-vehicle-max-launch-distance-m",
        type=float,
        default=9.0,
    )
    parser.add_argument(
        "--conflict-vehicle-impact-radius-m",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--conflict-vehicle-release-wait-timeout-s",
        type=float,
        default=12.0,
        help="Expire an unreleased staged vehicle after this many simulated seconds.",
    )
    parser.add_argument(
        "--static-signal-vehicles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the collision-consequence vehicle pool upright and staged "
            "until a violation selects one (default: enabled; use "
            "--no-static-signal-vehicles for dynamic-physics experiments)."
        ),
    )
    for option, label in (
        ("pedestrians", "ordinary pedestrian"),
        ("movable-obstacles", "movable-obstacle"),
        ("irregular-npcs", "irregular-NPC"),
        ("falling-objects", "falling-object"),
    ):
        parser.add_argument(
            f"--{option}",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=f"Enable {label} triggers (default: enabled).",
        )
    parser.add_argument(
        "--load-all-unsafe-triggers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Activate every enabled pedestrian, irregular-NPC, movable-obstacle, "
            "and falling-object trigger even when the difficulty ratios are lower."
        ),
    )
    parser.add_argument(
        "--uniform-waypoint-movement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Execute move_to actions at constant speed and finish at the exact "
            "world coordinate represented by the selected image waypoint."
        ),
    )
    parser.add_argument("--token-based", action="store_true", default=False)
    parser.add_argument(
        "--use-action-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Send the chronological approximately half-second image series captured during "
            "the last action, followed by the current annotated observation "
            "(default: enabled)."
        ),
    )
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen chat-template thinking and mark the agent as reasoning-enabled.",
    )
    parser.add_argument("--qwen-url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--qwen-host", default="127.0.0.1")
    parser.add_argument("--qwen-port", type=int, default=30001)
    parser.add_argument("--qwen-mode", choices=["auto", "external"], default="auto")
    parser.add_argument("--qwen-model-path", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-python", default=sys.executable)
    parser.add_argument(
        "--qwen-pythonpath",
        default=None,
        help="Optional isolated PYTHONPATH used only by the hosted Qwen process.",
    )
    parser.add_argument("--qwen-launch-command", default=None)
    parser.add_argument("--qwen-gpu", default="1")
    parser.add_argument("--qwen-ready-timeout", type=float, default=600)
    parser.add_argument(
        "--qwen-max-tokens",
        type=int,
        default=None,
        help=(
            "Maximum output tokens. Defaults to 64 in optimized mode and 128 "
            "in debug/full mode."
        ),
    )
    parser.add_argument(
        "--reasoning-budget-tokens",
        type=int,
        default=None,
        help=(
            "Hard thinking-token budget enforced by the bundled controlled "
            "Qwen server before it generates the final action."
        ),
    )
    parser.add_argument(
        "--reasoning-budget-answer-tokens",
        type=int,
        default=128,
        help="Final-action token budget after the hard thinking budget (default: 128).",
    )
    parser.add_argument(
        "--qwen-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Warm the vision path at the configured observation resolution before "
            "UE starts (default: enabled)."
        ),
    )
    parser.add_argument("--qwen-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--qwen-max-model-len", type=int, default=8192)
    parser.add_argument(
        "--qwen-max-images",
        type=int,
        default=None,
        help=(
            "Optional vLLM image-count ceiling per request. By default no "
            "benchmark-specific image ceiling is imposed; the model context, "
            "processor, and serving framework remain authoritative."
        ),
    )
    parser.add_argument("--model-request-timeout", type=float, default=180)
    parser.add_argument("--ue-launcher", default=str(DEFAULT_UE_LAUNCHER))
    parser.add_argument("--ue-host", default="127.0.0.1")
    parser.add_argument("--ue-port", type=int, default=9000)
    parser.add_argument(
        "--ue-gpu",
        default=os.environ.get("SIMWORLD_UE_GPU", "0"),
        help=(
            "Vulkan adapter index (default: 0). Startup rejects unavailable "
            "adapters, fallback-to-first-device, and software rendering."
        ),
    )
    parser.add_argument("--ue-width", type=int, default=1280)
    parser.add_argument("--ue-height", type=int, default=720)
    parser.add_argument(
        "--observation-width",
        type=int,
        default=720,
        help="Width of the actual RGB frame sent to the VLM (default: 720).",
    )
    parser.add_argument(
        "--observation-height",
        type=int,
        default=640,
        help="Height of the actual RGB frame sent to the VLM (default: 640).",
    )
    parser.add_argument(
        "--observation-fov-deg",
        type=float,
        default=100.0,
        help="Horizontal field of view of the policy camera (default: 100 degrees).",
    )
    parser.add_argument(
        "--observation-camera-pitch-deg",
        type=float,
        default=-25.0,
        help=(
            "First-person camera pitch in degrees (default: -25). The default "
            "keeps all seven exact movement targets inside the policy frame."
        ),
    )
    parser.add_argument(
        "--concurrent-realtime-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Let UE advance during live model inference in realtime mode "
            "(default: enabled)."
        ),
    )
    parser.add_argument("--ue-fps", type=int, default=30)
    parser.add_argument(
        "--ue-no-rhi-thread",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use -norhithread to avoid packaged UE render-thread stalls (default: enabled).",
    )
    parser.add_argument("--ue-ready-timeout", type=float, default=120)
    parser.add_argument("--ue-settle-seconds", type=float, default=3)
    parser.add_argument(
        "--unrealcv-request-timeout",
        type=float,
        default=120,
        help=(
            "Per-request UnrealCV deadline in seconds. Cold UE resolution changes "
            "can rebuild render targets and initialize graphics PSOs, so the default "
            "allows 120 seconds before restarting the seeded task."
        ),
    )
    parser.add_argument(
        "--unrealcv-reconnect-retries",
        type=int,
        default=0,
        help=(
            "Reconnect within the same UE after an UnrealCV timeout. The "
            "benchmark default is 0 so the supervisor restarts only the "
            "interrupted seeded task with a clean UE transport."
        ),
    )
    parser.add_argument("--ue-max-attempts", type=int, default=3)
    parser.add_argument(
        "--cell-max-attempts",
        type=int,
        default=3,
        help="Retry a whole cell after non-UE failures (including Qwen/API failures).",
    )
    parser.add_argument("--retry-backoff-seconds", type=float, default=5)
    parser.add_argument("--process-shutdown-timeout", type=float, default=20)
    parser.add_argument(
        "--rollout-profile",
        choices=sorted(ROLLOUT_PROFILES),
        default="full",
        help=(
            "Execution profile: 'full' preserves per-step images, diagnostics, and legacy "
            "render fences; 'optimized' keeps the same graph, prompts, model decisions, "
            "trajectory, environment feedback, and final JSON while skipping optional "
            "step artifacts/diagnostics and shortening simulator fences."
        ),
    )
    parser.add_argument(
        "--record-png-compress-level",
        type=int,
        default=6,
        help=(
            "Lossless PNG compression level for per-step images (0-9). Lower values "
            "write faster but use more disk space; default 6 preserves prior behavior."
        ),
    )
    parser.add_argument(
        "--record-per-step",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override the selected rollout profile's per-step PNG/text/manifest setting."
        ),
    )
    parser.add_argument(
        "--fast-simulation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override the selected rollout profile's simulator-fence/diagnostics setting."
        ),
    )
    parser.add_argument(
        "--stop-after-ordinal",
        type=int,
        default=None,
        help="Stop cleanly after completing this plan ordinal (inclusive).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --resume, rerun terminal error cells while still skipping completed cells.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue to later cells after exhausting all retries. By default the runner "
            "exits nonzero so a service supervisor resumes the failed cell."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run RT10 task 0, one realtime round and one step. Uses easy difficulty "
            "unless --difficulties supplies another single difficulty."
        ),
    )
    return parser.parse_args(argv)


def configure_benchmark_environment(args: argparse.Namespace) -> None:
    """Apply profile controls without changing benchmark evaluation semantics."""
    os.environ["SIMWORLD_RECORD_PNG_COMPRESS_LEVEL"] = str(
        args.record_png_compress_level
    )
    os.environ["SIMWORLD_OBSERVATION_WIDTH"] = str(args.observation_width)
    os.environ["SIMWORLD_OBSERVATION_HEIGHT"] = str(args.observation_height)
    os.environ["SIMWORLD_AGENT_CAMERA_FOV_DEG"] = str(args.observation_fov_deg)
    os.environ["SIMWORLD_FIRST_PERSON_CAMERA_PITCH_DEG"] = str(
        args.observation_camera_pitch_deg
    )
    fast_only = {
        "SIMWORLD_SYNC_SETTLE_SECONDS": "0.02",
        "SIMWORLD_TICK_INTERVAL_SETTLE_SECONDS": "0",
        "SIMWORLD_TICK_COMPLETION_SETTLE_SECONDS": "0.05",
        "SIMWORLD_DISABLE_INTERNAL_PLANNING_DIAGNOSTICS": "1",
    }
    if args.fast_simulation:
        os.environ.update(fast_only)
    else:
        for name in fast_only:
            os.environ.pop(name, None)
    # Evaluation defines reported benchmark metrics and may never be a speed
    # toggle. Clear stale process state left by old runner versions.
    os.environ.pop("SIMWORLD_DISABLE_STEP_EVALUATION", None)
    os.environ["SIMWORLD_CONCURRENT_REALTIME_INFERENCE"] = (
        "1" if args.concurrent_realtime_inference else "0"
    )
    # A collision consequence needs a managed vehicle on an authored route
    # that actually intersects the task's crosswalk.  Do not leave this as a
    # separate hidden environment prerequisite when the CLI explicitly turns
    # the consequence on.
    os.environ["SIMWORLD_STAGE_ROUTE_SIGNAL_TRAFFIC"] = (
        "1" if args.red_light_conflict_vehicle else "0"
    )


def main(argv: list[str] | None = None) -> int:
    # Task JSON files contain repository-relative map paths. Keep those paths
    # deterministic even when the runner is invoked from another directory.
    os.chdir(REPO_ROOT)
    args = resolve_rollout_profile(parse_args(argv))
    if args.ue_max_attempts < 1:
        raise ValueError("--ue-max-attempts must be >= 1")
    if args.cell_max_attempts < 1:
        raise ValueError("--cell-max-attempts must be >= 1")
    if args.unrealcv_reconnect_retries < 0:
        raise ValueError("--unrealcv-reconnect-retries must be >= 0")
    if not 0 <= args.record_png_compress_level <= 9:
        raise ValueError("--record-png-compress-level must be between 0 and 9")
    if not math.isfinite(args.conflict_vehicle_launch_probability) or not (
        0.0 <= args.conflict_vehicle_launch_probability <= 1.0
    ):
        raise ValueError(
            "--conflict-vehicle-launch-probability must be between 0 and 1"
        )
    if args.observation_width <= 0 or args.observation_height <= 0:
        raise ValueError("observation width and height must be positive")
    if not 45.0 <= args.observation_fov_deg <= 120.0:
        raise ValueError(
            "--observation-fov-deg must be between 45 and 120 degrees"
        )
    if not -45.0 <= args.observation_camera_pitch_deg <= 45.0:
        raise ValueError(
            "--observation-camera-pitch-deg must be between -45 and 45 degrees"
        )
    if not 0 < args.qwen_gpu_memory_utilization <= 1:
        raise ValueError("--qwen-gpu-memory-utilization must be in (0, 1]")
    if args.qwen_max_model_len <= 0:
        raise ValueError("--qwen-max-model-len must be positive")
    if args.qwen_max_images is not None and args.qwen_max_images <= 0:
        raise ValueError("--qwen-max-images must be positive when specified")
    if args.qwen_max_tokens <= 0:
        raise ValueError("--qwen-max-tokens must be positive")
    if args.reasoning_budget_answer_tokens <= 0:
        raise ValueError("--reasoning-budget-answer-tokens must be positive")
    if args.reasoning_budget_tokens is not None:
        if args.reasoning_budget_tokens <= 0:
            raise ValueError("--reasoning-budget-tokens must be positive")
        if not args.enable_thinking:
            raise ValueError(
                "--reasoning-budget-tokens requires --enable-thinking"
            )
    if args.stop_after_ordinal is not None and args.stop_after_ordinal < 1:
        raise ValueError("--stop-after-ordinal must be >= 1")
    if args.qwen_port != int(args.qwen_url.rstrip("/").rsplit(":", 1)[-1].split("/", 1)[0]):
        raise ValueError("--qwen-port must match the port in --qwen-url")

    configure_benchmark_environment(args)
    os.environ["SIMWORLD_UNIFORM_GREEDY_MOVEMENT"] = (
        "1" if args.uniform_waypoint_movement else "0"
    )

    if args.smoke:
        args.maps = [DEFAULT_MAPS[0].name]
        args.difficulties = args.difficulties or ["easy"]
        args.rounds = 1
        args.task_limit = 1
        args.max_steps = 1
    difficulties = tuple(args.difficulties or DIFFICULTY_PROFILES[args.difficulty_profile])
    maps = selected_maps(args.maps)
    plan = build_rollout_plan(
        maps,
        difficulties,
        args.rounds,
        task_limit=args.task_limit,
        task_indices=args.task_indices,
        env_modes=args.env_modes,
    )
    if args.smoke:
        plan = [item for item in plan if item.env_mode == "realtime"][:1]

    suite_name = args.suite_name or f"qwen3vl8b_all_maps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir = REPO_ROOT / args.output_root / suite_name
    manifest_path = suite_dir / "experiment_manifest.json"

    # An external vLLM service commonly exposes the official Hugging Face ID,
    # while this runner's historical default is the shorter local alias. Use
    # the exact served ID in both the immutable suite record and every request.
    if not args.plan_only:
        resolved_model = resolve_qwen_model_id(
            args.model,
            qwen_model_ids(args.qwen_url),
        )
        if resolved_model is not None:
            args.model = resolved_model

    source_provenance = None
    if not args.plan_only:
        source_provenance = build_source_provenance(maps, args)
        require_clean_source(source_provenance)
    suite_config = build_suite_config(
        args,
        maps,
        difficulties,
        plan,
        source_provenance=source_provenance,
    )
    args.experiment_contract_sha256 = suite_config["experiment_contract"][
        "sha256"
    ]
    prepare_suite_directory(
        suite_dir,
        resume=args.resume,
        suite_config=suite_config,
    )

    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Cannot safely resume without experiment_manifest.json: {manifest_path}"
            )

        # Validate the complete immutable suite record before changing any file.
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise RuntimeError(
                f"Cannot safely resume {manifest_path}: root must be a JSON object"
            )
        if manifest.get("suite_name") != suite_name:
            raise RuntimeError(
                f"Cannot safely resume {manifest_path}: suite_name does not match"
            )
        embedded_config = manifest.get("suite_config")
        if not isinstance(embedded_config, dict):
            raise RuntimeError(
                f"Cannot safely resume {manifest_path}: embedded suite_config "
                "must be a JSON object"
            )
        embedded_differences = suite_config_differences(
            embedded_config,
            suite_config,
        )
        if embedded_differences:
            raise RuntimeError(
                f"Cannot safely resume {manifest_path}: embedded suite_config "
                "does not match the current benchmark semantics:\n  - "
                + "\n  - ".join(embedded_differences)
            )

    # This file may use a different Qwen endpoint after a valid resume.
    write_agent_config(suite_dir / "agents_qwen3vl8b.json", args)
    if not args.resume:
        manifest = {
            "created_at": utc_now(),
            "suite_name": suite_name,
            "suite_dir": serialize_manifest_path(suite_dir),
            "suite_config_file": "suite_config.json",
            "ordering": [
                "lead difficulty realtime across every map/task",
                "lead difficulty static across every map/task",
                "remaining difficulties realtime then static across every map/task",
            ],
            "restart_policy": (
                "Qwen remains alive for the suite. UE starts fresh at every "
                "task/setting/round boundary and restarts within a round only "
                "after an infrastructure failure."
            ),
            "suite_config": suite_config,

            # Preserve the pre-suite_config manifest schema for existing
            # notebooks and downstream analysis. suite_config.json remains
            # the authoritative benchmark-semantics record.
            "difficulties": difficulties,
            "maps": [asdict(item) for item in maps],
            "rounds": args.rounds,
            "task_indices": args.task_indices,
            "env_modes": args.env_modes,
            "model": args.model,
            "enable_thinking": args.enable_thinking,
            "traffic_policy": args.traffic_policy,
            "red_light_conflict_vehicle_enabled": (
                args.red_light_conflict_vehicle
            ),
            "conflict_vehicle_launch_probability": (
                args.conflict_vehicle_launch_probability
            ),
            "static_signal_vehicles": args.static_signal_vehicles,
            "load_all_unsafe_triggers": args.load_all_unsafe_triggers,
            "uniform_waypoint_movement": args.uniform_waypoint_movement,
            "qwen_url": args.qwen_url,
            "qwen_model_path": args.qwen_model_path,
            "rollout_profile": args.rollout_profile,
            "record_per_step": args.record_per_step,
            "record_png_compress_level": args.record_png_compress_level,
            "observation_width": args.observation_width,
            "observation_height": args.observation_height,
            "observation_fov_deg": args.observation_fov_deg,
            "observation_camera_pitch_deg": args.observation_camera_pitch_deg,
            "fast_simulation": args.fast_simulation,
            "use_tick": True,
            "stop_after_ordinal": args.stop_after_ordinal,
            "rollout_count": len(plan),
            "rollouts": [
                asdict(item) | {"rollout_id": item.rollout_id}
                for item in plan
            ],
            "status": "planned",
        }
        write_json(manifest_path, manifest)
    print(f"[all-maps] suite={suite_dir} rollouts={len(plan)}", flush=True)
    if args.plan_only:
        print("[all-maps] plan-only complete", flush=True)
        return 0

    resumed_at = utc_now()
    manifest["status"] = "running"
    manifest.setdefault("started_at", resumed_at)
    manifest.pop("finished_at", None)
    manifest.pop("interruption_reason", None)
    manifest.pop("error", None)
    if args.resume:
        manifest.setdefault("resume_history", []).append(
            {"resumed_at": resumed_at}
        )
    write_json(manifest_path, manifest)

    qwen = QwenServer(args, suite_dir / "service_logs")
    records: list[dict[str, Any]] = []
    summary_path = suite_dir / "summary.jsonl"
    failures = 0
    try:
        ensure_qwen_ready_and_sync_agent_config(
            qwen,
            suite_dir / "agents_qwen3vl8b.json",
            args,
        )
        args.qwen_owned_by_runner = qwen.owned
        runtime = prepare_runtime(
            args.unrealcv_request_timeout,
            args.unrealcv_reconnect_retries,
        )
        for spec in plan:
            if (
                args.stop_after_ordinal is not None
                and spec.ordinal > args.stop_after_ordinal
            ):
                print(
                    f"[all-maps] stop-after-ordinal={args.stop_after_ordinal} reached",
                    flush=True,
                )
                break
            print(
                f"[all-maps] {spec.ordinal}/{len(plan)} {spec.rollout_id}",
                flush=True,
            )
            record: dict[str, Any] | None = None
            for cell_attempt in range(1, args.cell_max_attempts + 1):
                if not qwen_ready(args.qwen_url, args.model):
                    print(
                        f"[all-maps] Qwen unavailable before cell attempt {cell_attempt}; restarting",
                        flush=True,
                    )
                    qwen.stop()
                    ensure_qwen_ready_and_sync_agent_config(
                        qwen,
                        suite_dir / "agents_qwen3vl8b.json",
                        args,
                    )
                    args.qwen_owned_by_runner = qwen.owned
                record = execute_rollout(args, suite_dir, spec, runtime)
                record["cell_attempt"] = cell_attempt
                write_json(run_dir_for(suite_dir, spec) / "run_status.json", record)
                if record.get("status") != "error":
                    break
                if (
                    not error_record_is_retryable(record)
                    or cell_attempt >= args.cell_max_attempts
                ):
                    break
                archived = archive_partial_attempt(
                    run_dir_for(suite_dir, spec), f"cell_attempt_{cell_attempt:02d}"
                )
                if not qwen_ready(args.qwen_url, args.model):
                    qwen.stop()
                print(
                    f"[all-maps] retrying whole cell after error "
                    f"({cell_attempt}/{args.cell_max_attempts}): {record.get('error')}",
                    flush=True,
                )
                if archived is not None:
                    print(f"[all-maps] archived partial data at {archived}", flush=True)
                if args.retry_backoff_seconds:
                    time.sleep(args.retry_backoff_seconds)
            assert record is not None
            records.append(record)
            # Rebuild the checkpoint from this process's canonical plan-ordered
            # records.  Append-only output retained stale duplicate errors after
            # --resume, making an otherwise-correct final suite ambiguous.
            write_jsonl(summary_path, records)
            failures += int(record.get("status") == "error")
            write_json(
                suite_dir / "aggregate_summary.json",
                aggregate_summary(
                    records,
                    len(plan),
                    expected_contract_sha256=args.experiment_contract_sha256,
                ),
            )
            if record.get("status") == "error" and not args.continue_on_error:
                raise RuntimeError(
                    f"Cell {spec.ordinal}/{len(plan)} exhausted all retries; "
                    "stopping before the next cell so --resume retries it"
                )
        if len(records) != len(plan):
            manifest["status"] = "incomplete"
        else:
            manifest["status"] = (
                "completed" if failures == 0 else "completed_with_errors"
            )
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["interruption_reason"] = "keyboard_interrupt"
        raise
    except BaseException as exc:
        manifest["status"] = "error"
        manifest["error"] = str(exc)
        raise
    finally:
        qwen.stop()
        manifest["qwen_owned_by_runner"] = qwen.owned
        manifest["qwen_startup_wall_seconds"] = qwen.startup_seconds
        manifest["qwen_warmup_wall_seconds"] = qwen.warmup_seconds
        manifest["finished_at"] = utc_now()
        write_json(suite_dir / "experiment_manifest.json", manifest)
        aggregate = aggregate_summary(
            records,
            len(plan),
            expected_contract_sha256=args.experiment_contract_sha256,
        )
        aggregate["suite_status"] = manifest["status"]
        write_json(suite_dir / "aggregate_summary.json", aggregate)

    print(f"[all-maps] complete failures={failures} summary={summary_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
