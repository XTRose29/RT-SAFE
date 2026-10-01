#!/usr/bin/env python3
"""One entry point for planning, running, finalizing, and inspecting benchmarks."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
RUNNER = REPO_ROOT / "evaluation" / "run_qwen3vl8b_all_maps.py"
FINALIZER = REPO_ROOT / "evaluation" / "finalize_qwen_rollout_videos.py"
DEFAULT_CONFIG = "all_tasks_easy_realtime_collision"
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

LIST_OPTIONS = {"maps", "difficulties", "env_modes", "task_indices"}
BOOLEAN_OPTIONAL = {
    "red_light_conflict_vehicle",
    "static_signal_vehicles",
    "pedestrians",
    "movable_obstacles",
    "irregular_npcs",
    "falling_objects",
    "load_all_unsafe_triggers",
    "uniform_waypoint_movement",
    "concurrent_realtime_inference",
    "qwen_load_in_8bit",
    "qwen_ensure_parseable_action",
    "ue_no_rhi_thread",
    "record_per_step",
    "fast_simulation",
    "retry_errors",
    "use_action_frames",
}
STORE_TRUE = {
    "token_based",
    "enable_thinking",
    "continue_on_error",
    "smoke",
}
VALUE_OPTIONS = {
    "difficulty_profile",
    "rounds",
    "task_limit",
    "seed",
    "max_steps",
    "prompt_style",
    "traffic_policy",
    "model",
    "qwen_url",
    "qwen_host",
    "qwen_port",
    "qwen_mode",
    "qwen_model_path",
    "qwen_python",
    "qwen_pythonpath",
    "qwen_launch_command",
    "qwen_gpu",
    "qwen_ready_timeout",
    "qwen_max_tokens",
    "qwen_max_images",
    "reasoning_budget_tokens",
    "reasoning_budget_answer_tokens",
    "model_request_timeout",
    "ue_launcher",
    "ue_host",
    "ue_port",
    "ue_gpu",
    "ue_width",
    "ue_height",
    "observation_width",
    "observation_height",
    "observation_fov_deg",
    "observation_camera_pitch_deg",
    "ue_fps",
    "ue_ready_timeout",
    "ue_settle_seconds",
    "unrealcv_request_timeout",
    "unrealcv_reconnect_retries",
    "ue_max_attempts",
    "cell_max_attempts",
    "retry_backoff_seconds",
    "process_shutdown_timeout",
    "rollout_profile",
    "record_png_compress_level",
    "stop_after_ordinal",
    "conflict_vehicle_launch_probability",
    "conflict_vehicle_min_launch_distance_m",
    "conflict_vehicle_max_launch_distance_m",
    "conflict_vehicle_impact_radius_m",
    "conflict_vehicle_release_wait_timeout_s",
}
ALLOWED_ROLLOUT_KEYS = LIST_OPTIONS | BOOLEAN_OPTIONAL | STORE_TRUE | VALUE_OPTIONS


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def resolve_config(value: str | Path) -> Path:
    path = Path(value)
    candidates = [path]
    if not path.suffix:
        candidates.extend((CONFIG_ROOT / f"{path}.json", CONFIG_ROOT / path))
    elif not path.is_absolute():
        candidates.append(CONFIG_ROOT / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Benchmark config not found: {value}")


def load_config(value: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_config(value)
    config = read_json(path)
    rollout = config.get("rollout")
    if not isinstance(rollout, dict):
        raise ValueError(f"{path}: 'rollout' must be an object")
    unknown = sorted(set(rollout) - ALLOWED_ROLLOUT_KEYS)
    if unknown:
        raise ValueError(f"{path}: unsupported rollout options: {', '.join(unknown)}")
    video = config.get("video", {})
    if not isinstance(video, dict):
        raise ValueError(f"{path}: 'video' must be an object")
    if video.get("enabled", True):
        profile = rollout.get("rollout_profile", "full")
        records_steps = rollout.get("record_per_step", profile == "full")
        if not records_steps:
            raise ValueError(
                f"{path}: video generation requires rollout.record_per_step=true "
                "(or rollout_profile='full')"
            )
        if not rollout.get("uniform_waypoint_movement", False):
            raise ValueError(
                f"{path}: aligned benchmark videos require "
                "rollout.uniform_waypoint_movement=true"
            )
    return path, config


def option_name(key: str, enabled: bool = True) -> str:
    name = key.replace("_", "-")
    return f"--{name}" if enabled else f"--no-{name}"


def rollout_arguments(config: dict[str, Any]) -> list[str]:
    rollout = config["rollout"]
    arguments: list[str] = []
    for key, value in rollout.items():
        if value is None:
            continue
        if key in LIST_OPTIONS:
            if not isinstance(value, list) or not value:
                raise ValueError(f"rollout.{key} must be a non-empty list")
            arguments.append(option_name(key))
            arguments.extend(str(item) for item in value)
        elif key in BOOLEAN_OPTIONAL:
            if not isinstance(value, bool):
                raise ValueError(f"rollout.{key} must be true or false")
            arguments.append(option_name(key, value))
        elif key in STORE_TRUE:
            if not isinstance(value, bool):
                raise ValueError(f"rollout.{key} must be true or false")
            if value:
                arguments.append(option_name(key))
        else:
            arguments.extend((option_name(key), str(value)))
    return arguments


def suite_path(output_root: str, suite_name: str) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / suite_name


def generated_suite_name(config_path: Path, config: dict[str, Any]) -> str:
    prefix = str(config.get("suite_prefix") or config_path.stem)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def run_command(args: argparse.Namespace) -> int:
    config_path, config = load_config(args.config)
    if args.resume and not args.suite_name:
        raise ValueError("--resume requires --suite-name so the existing suite is unambiguous")
    suite_name = args.suite_name or generated_suite_name(config_path, config)
    output_root = args.output_root or str(config.get("output_root", "results"))
    runner_args = [
        sys.executable,
        str(RUNNER),
        "--suite-name",
        suite_name,
        "--output-root",
        output_root,
        "--benchmark-profile-path",
        str(config_path),
        *rollout_arguments(config),
    ]
    if args.resume:
        runner_args.append("--resume")
    if args.plan_only:
        runner_args.append("--plan-only")
    runner_args.extend(args.runner_arg)

    print(f"[benchmark] config={config_path.relative_to(REPO_ROOT)}", flush=True)
    print(f"[benchmark] suite={suite_path(output_root, suite_name)}", flush=True)
    result = subprocess.run(runner_args, cwd=REPO_ROOT, check=False)
    if result.returncode != 0 or args.plan_only:
        return result.returncode

    video = config.get("video", {})
    if args.no_finalize or not video.get("enabled", True):
        return 0
    fps = args.fps if args.fps is not None else float(video.get("fps", 1.5))
    return finalize_suite(
        suite_path(output_root, suite_name),
        fps=fps,
        force=args.force_video,
        allow_incomplete=False,
    )


def finalize_suite(
    suite_dir: Path,
    *,
    fps: float,
    force: bool,
    allow_incomplete: bool,
) -> int:
    command = [
        sys.executable,
        str(FINALIZER),
        "--suite-dir",
        str(suite_dir),
        "--fps",
        str(fps),
    ]
    if force:
        command.append("--force")
    if allow_incomplete:
        command.append("--allow-incomplete")
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def finalize_command(args: argparse.Namespace) -> int:
    return finalize_suite(
        args.suite_dir.resolve(),
        fps=args.fps,
        force=args.force,
        allow_incomplete=args.allow_incomplete,
    )


def status_payload(suite_dir: Path) -> dict[str, Any]:
    manifest_path = suite_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing experiment manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    statuses = []
    for path in sorted(suite_dir.glob("runs/**/run_status.json")):
        try:
            status = read_json(path)
            if status.get("status") == "completed":
                result_name = status.get("result_path")
                result_path = Path(result_name) if result_name else None
                if result_path is not None and not result_path.is_absolute():
                    result_path = REPO_ROOT / result_path
                if result_path is None or not result_path.is_file():
                    status = dict(status)
                    status["status"] = "invalidated"
            statuses.append(status)
        except (OSError, ValueError, json.JSONDecodeError):
            statuses.append({"status": "invalid", "path": str(path)})
    counts = Counter(str(item.get("status", "missing")) for item in statuses)
    def is_active_artifact(path: Path) -> bool:
        try:
            relative_parts = path.relative_to(suite_dir).parts
        except ValueError:
            relative_parts = path.parts
        return "failed_attempts" not in relative_parts

    videos = [
        path
        for path in suite_dir.glob("runs/**/rollout_agent_input_annotated.mp4")
        if is_active_artifact(path)
    ]
    aligned = 0
    failed_alignment = 0
    for report in suite_dir.glob("runs/**/rollout_alignment_check.json"):
        if not is_active_artifact(report):
            continue
        if read_json(report).get("passed") is True:
            aligned += 1
        else:
            failed_alignment += 1
    return {
        "suite_dir": str(suite_dir),
        "planned": int(manifest.get("rollout_count") or len(manifest.get("rollouts") or [])),
        "recorded_statuses": len(statuses),
        "status_counts": dict(sorted(counts.items())),
        "videos": len(videos),
        "alignment_passed": aligned,
        "alignment_failed": failed_alignment,
        "finished_at": manifest.get("finished_at"),
    }


def status_command(args: argparse.Namespace) -> int:
    payload = status_payload(args.suite_dir.resolve())
    print(json.dumps(payload, indent=2))
    return 1 if payload["alignment_failed"] else 0


def tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def qwen_doctor_check(rollout: dict[str, Any]) -> tuple[str, bool, str]:
    """Mirror the runner's external-first behavior in auto mode."""
    qwen_mode = str(rollout.get("qwen_mode", "auto"))
    qwen_host = str(rollout.get("qwen_host", "127.0.0.1"))
    qwen_port = int(rollout.get("qwen_port", 30001))
    endpoint = f"{qwen_host}:{qwen_port}"
    if qwen_mode in {"auto", "external"} and tcp_open(qwen_host, qwen_port):
        return "external Qwen endpoint", True, endpoint
    if qwen_mode == "external":
        return "external Qwen endpoint", False, endpoint
    model = str(rollout.get("qwen_model_path") or DEFAULT_QWEN_MODEL)
    if model:
        return "Qwen model", Path(model).exists(), model
    return (
        "Qwen model",
        False,
        "start qwen_host:qwen_port or set SIMWORLD_QWEN_MODEL/rollout.qwen_model_path",
    )


def python_module_location(
    module: str,
    extra_roots: list[Path] | None = None,
) -> str | None:
    """Return where a runtime module is available, including local deps."""
    if importlib.util.find_spec(module) is not None:
        return "active environment"
    roots = extra_roots or [
        REPO_ROOT / f".py{sys.version_info.major}{sys.version_info.minor}deps",
        REPO_ROOT / ".py312deps",
    ]
    for root in roots:
        if root.is_dir() and importlib.machinery.PathFinder.find_spec(
            module, [str(root)]
        ) is not None:
            return str(root)
    return None


def doctor_command(args: argparse.Namespace) -> int:
    _, config = load_config(args.config)
    rollout = config["rollout"]
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0]))
    checks.append(("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not found"))
    for module in ("PIL", "numpy", "openai", "unrealcv"):
        location = python_module_location(module)
        checks.append(
            (
                f"Python module {module}",
                location is not None,
                location or "not found",
            )
        )
    for task_file in (
        "data/map1_10roads/tasks.json",
        "data/map2_12roads/tasks.json",
        "data/map3_15roads/tasks.json",
        "data/map4_18roads/tasks.json",
        "data/map5_20roads/tasks.json",
    ):
        path = REPO_ROOT / task_file
        checks.append((task_file, path.is_file(), str(path)))

    ue_value = str(rollout.get("ue_launcher") or DEFAULT_UE_LAUNCHER)
    if ue_value:
        ue_launcher = Path(ue_value)
        checks.append(("UE launcher", ue_launcher.is_file(), str(ue_launcher)))
    else:
        checks.append(
            (
                "UE launcher",
                False,
                "set SIMWORLD_UE_LAUNCHER or rollout.ue_launcher",
            )
        )
    checks.append(qwen_doctor_check(rollout))

    for label, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}: {detail}")
    return 0 if all(item[1] for item in checks) else 1


def list_configs_command(_: argparse.Namespace) -> int:
    for path in sorted(CONFIG_ROOT.glob("*.json")):
        config = read_json(path)
        print(f"{path.stem:28s} {config.get('description', '')}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a configured rollout suite")
    run_parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    run_parser.add_argument("--suite-name")
    run_parser.add_argument("--output-root")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--plan-only", action="store_true")
    run_parser.add_argument("--no-finalize", action="store_true")
    run_parser.add_argument("--force-video", action="store_true")
    run_parser.add_argument("--fps", type=float)
    run_parser.add_argument(
        "--runner-arg",
        action="append",
        default=[],
        help="Advanced: append one raw argument to the underlying rollout runner",
    )
    run_parser.set_defaults(handler=run_command)

    finalize_parser = subparsers.add_parser("finalize", help="Validate and render completed rollouts")
    finalize_parser.add_argument("suite_dir", type=Path)
    finalize_parser.add_argument("--fps", type=float, default=1.5)
    finalize_parser.add_argument("--force", action="store_true")
    finalize_parser.add_argument("--allow-incomplete", action="store_true")
    finalize_parser.set_defaults(handler=finalize_command)

    status_parser = subparsers.add_parser("status", help="Summarize a suite")
    status_parser.add_argument("suite_dir", type=Path)
    status_parser.set_defaults(handler=status_command)

    doctor_parser = subparsers.add_parser("doctor", help="Check local runtime prerequisites")
    doctor_parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    doctor_parser.set_defaults(handler=doctor_command)

    configs_parser = subparsers.add_parser("list-configs", help="List versioned benchmark presets")
    configs_parser.set_defaults(handler=list_configs_command)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"benchmark: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
