#!/usr/bin/env python3
"""Run resumable live collision trials across randomized launch ranges."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from analyze_collision_launch_probability import wilson_interval
from evaluation.run_qwen3vl8b_all_maps import UEServer


def _range(raw: str) -> tuple[float, float]:
    try:
        minimum, maximum = (float(item) for item in raw.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("range must be MIN_CM:MAX_CM") from exc
    if minimum < 0.0 or maximum < minimum:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= MIN_CM <= MAX_CM")
    return minimum, maximum


def _seeds(raw: str) -> list[int]:
    try:
        values = [int(item) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def _label(distance_range: tuple[float, float]) -> str:
    return "range_{}_{}".format(*(f"{value:g}".replace(".", "p") for value in distance_range))


def _cached_trial_matches(
    payload: object,
    *,
    task_file: str,
    task_number: int,
    mode: str,
    seed: int,
    distance_range: tuple[float, float],
    walking_speed_cm_s: float,
    simulation_step_seconds: float,
    camera_resolution: tuple[int, int],
) -> bool:
    if not isinstance(payload, dict):
        return False
    experiment = payload.get("experiment", {})
    recording = payload.get("full_task_recording", {})
    return bool(
        payload.get("passed")
        and payload.get("task_file") == task_file
        and payload.get("task_number") == task_number
        and payload.get("mode") == mode
        and payload.get("seed") == seed
        and experiment.get("pressure_fast_forward") is True
        and experiment.get("post_launch_agent_motion") == "continue"
        and experiment.get("launch_distance_range_cm") == list(distance_range)
        and experiment.get("walking_speed_cm_s") == walking_speed_cm_s
        and recording.get("simulation_step_seconds") == simulation_step_seconds
        and payload.get("camera", {}).get("resolution") == list(camera_resolution)
        and payload.get("camera", {}).get("fov_deg") == 70.0
    )


def _summary(output_dir: Path, trials: list[dict]) -> dict:
    ranges = []
    for label in sorted({trial["range_label"] for trial in trials}):
        selected = [trial for trial in trials if trial["range_label"] == label]
        valid = [trial for trial in selected if trial.get("passed")]
        collisions = sum(trial.get("outcome") == "collision" for trial in valid)
        ranges.append(
            {
                "range_label": label,
                "launch_distance_range_cm": selected[0]["launch_distance_range_cm"],
                "attempted": len(selected),
                "valid_launched_trials": len(valid),
                "collisions": collisions,
                "misses": len(valid) - collisions,
                "collision_probability_given_launch": (
                    collisions / len(valid) if valid else None
                ),
                "collision_probability_wilson_95": wilson_interval(
                    collisions, len(valid)
                ),
            }
        )
    return {
        "definition": "P(collision | production consequence disposition == launched)",
        "output_dir": str(output_dir.resolve()),
        "ranges": ranges,
        "trials": trials,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--task-number", type=int, required=True)
    parser.add_argument("--mode", choices=("red_light", "illegal_crossing"), default="red_light")
    parser.add_argument("--ue-launcher", type=Path)
    parser.add_argument("--ue-host", default="127.0.0.1")
    parser.add_argument("--ue-gpu", default="5")
    parser.add_argument("--ue-map", default="/Game/RealTimeBench/Maps/RT10")
    parser.add_argument("--ue-width", type=int, default=720)
    parser.add_argument("--ue-height", type=int, default=640)
    parser.add_argument("--ue-fps", type=int, default=30)
    parser.add_argument(
        "--ue-no-rhi-thread",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ue-ready-timeout", type=float, default=120.0)
    parser.add_argument("--ue-settle-seconds", type=float, default=1.0)
    parser.add_argument("--process-shutdown-timeout", type=float, default=20.0)
    parser.add_argument("--trial-max-attempts", type=int, default=3)
    parser.add_argument(
        "--rotate-ports",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a fresh UnrealCV port for each owned renderer attempt.",
    )
    parser.add_argument("--unrealcv-port", type=int, default=9001)
    parser.add_argument("--difficulty", default="level0")
    parser.add_argument("--range", dest="ranges", type=_range, action="append", required=True)
    parser.add_argument("--seeds", type=_seeds, default=[1, 2, 3, 4, 5])
    parser.add_argument("--walking-speed-cm-s", type=float, default=200.0)
    parser.add_argument("--simulation-step-seconds", type=float, default=1.0)
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.trial_max_attempts < 1:
        parser.error("--trial-max-attempts must be at least 1")
    # UEServer uses the benchmark runner's naming while the capture tool keeps
    # its historical UnrealCV spelling. Owning a fresh renderer per trial is
    # optional so the sweep can still target an already-running development UE.
    args.ue_port = args.unrealcv_port

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_script = Path(__file__).with_name("capture_triggered_collision_task.py")
    trials: list[dict] = []
    any_failure = False
    next_owned_port = args.unrealcv_port
    for distance_range in args.ranges:
        range_label = _label(distance_range)
        for seed in args.seeds:
            trial_dir = output_dir / range_label / f"seed_{seed:03d}"
            metrics_path = trial_dir / f"{args.mode}_metrics.json"
            payload = None
            if not args.no_resume and metrics_path.exists():
                try:
                    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = None
            returncode = 0
            infrastructure_error = None
            renderer_info = None
            attempts = []
            cached_pressure_trial = _cached_trial_matches(
                payload,
                task_file=args.task_file,
                task_number=args.task_number,
                mode=args.mode,
                seed=seed,
                distance_range=distance_range,
                walking_speed_cm_s=args.walking_speed_cm_s,
                simulation_step_seconds=args.simulation_step_seconds,
                camera_resolution=(args.camera_width, args.camera_height),
            )
            if not cached_pressure_trial:
                trial_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(capture_script),
                    "--mode",
                    args.mode,
                    "--task-file",
                    args.task_file,
                    "--task-number",
                    str(args.task_number),
                    "--seed",
                    str(seed),
                    "--unrealcv-port",
                    str(args.unrealcv_port),
                    "--difficulty",
                    args.difficulty,
                    "--camera-width",
                    str(args.camera_width),
                    "--camera-height",
                    str(args.camera_height),
                    "--camera-fov-deg",
                    "70",
                    "--fps",
                    "1",
                    "--capture-stride",
                    "10",
                    "--simulation-step-seconds",
                    str(args.simulation_step_seconds),
                    "--walking-speed-cm-s",
                    str(args.walking_speed_cm_s),
                    "--post-launch-agent-motion",
                    "continue",
                    "--accept-outcome",
                    "collision-or-miss",
                    "--pressure-fast-forward",
                    "--launch-distance-min-cm",
                    str(distance_range[0]),
                    "--launch-distance-max-cm",
                    str(distance_range[1]),
                    "--output-dir",
                    str(trial_dir),
                ]
                maximum_attempts = args.trial_max_attempts if args.ue_launcher else 1
                for attempt in range(1, maximum_attempts + 1):
                    if args.ue_launcher is not None and args.rotate_ports:
                        trial_port = next_owned_port
                        next_owned_port += 1
                    else:
                        trial_port = args.unrealcv_port
                    args.ue_port = trial_port
                    command[command.index("--unrealcv-port") + 1] = str(trial_port)
                    ue = None
                    attempt_error = None
                    try:
                        if args.ue_launcher is not None:
                            ue = UEServer(args, trial_dir / f"ue_attempt_{attempt}.log")
                            ue.start(args.ue_map)
                            renderer_info = ue.renderer_info
                        completed = subprocess.run(
                            command,
                            check=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                        )
                        returncode = completed.returncode
                        (trial_dir / f"capture_attempt_{attempt}.log").write_text(
                            completed.stdout, encoding="utf-8"
                        )
                        if returncode != 0:
                            attempt_error = f"capture exited {returncode}"
                    except Exception as exc:  # retain infrastructure failures in the sweep
                        returncode = 1
                        attempt_error = f"{type(exc).__name__}: {exc}"
                    finally:
                        if ue is not None:
                            ue.stop()
                    attempts.append(
                        {
                            "attempt": attempt,
                            "unrealcv_port": trial_port,
                            "returncode": returncode,
                            "error": attempt_error,
                        }
                    )
                    if returncode == 0:
                        break
                infrastructure_error = next(
                    (item["error"] for item in reversed(attempts) if item["error"]),
                    None,
                ) if returncode else None
                if metrics_path.exists():
                    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            event = (payload or {}).get("trigger_record") or {}
            passed = bool((payload or {}).get("passed")) and returncode == 0
            trial = {
                "range_label": range_label,
                "launch_distance_range_cm": list(distance_range),
                "seed": seed,
                "passed": passed,
                "returncode": returncode,
                "infrastructure_error": infrastructure_error,
                "renderer_info": renderer_info,
                "attempts": attempts,
                "outcome": (payload or {}).get("experiment", {}).get("observed_outcome"),
                "requested_launch_distance_cm": event.get("requested_launch_distance_cm"),
                "launch_distance_cm": event.get("launch_distance_cm"),
                "minimum_agent_distance_cm": event.get("minimum_agent_distance_cm"),
                "status": event.get("status"),
                "metrics": str(metrics_path),
            }
            trials.append(trial)
            any_failure |= not passed
            summary = _summary(output_dir, trials)
            (output_dir / "sweep_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(trial), flush=True)

    print(json.dumps(_summary(output_dir, trials), indent=2))
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
