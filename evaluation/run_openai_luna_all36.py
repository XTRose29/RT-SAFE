#!/usr/bin/env python3
"""Resumable Luna none/low/default campaign across the 36 SimWorld tasks."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.run_qwen3vl8b_all_maps import UEServer
from utils.route_steps import route_length_to_max_steps
from utils.task_routes import reconstruct_route_points, route_length_cm

MODEL = "gpt-5.6-luna"
RESULTS_ROOT = Path("results")
KEY_FILE = Path(".secrets/openai_api.txt")
UE_LAUNCHER = Path(os.environ.get("SIMWORLD_UE_LAUNCHER", str(REPO_ROOT / "runtime/SimWorld.sh")))
INPUT_PRICE = 0.20
OUTPUT_PRICE = 1.20
OBSERVATION_WIDTH = 720
OBSERVATION_HEIGHT = 640
UE_WIDTH = 1280
UE_HEIGHT = 720


@dataclass(frozen=True)
class MapSpec:
    name: str
    ue_map: str
    task_file: str


@dataclass(frozen=True)
class TaskSpec:
    map_name: str
    ue_map: str
    task_file: str
    task_index: int
    task_id: int
    route_length_m: float
    max_steps: int


@dataclass(frozen=True)
class Condition:
    ordinal: int
    difficulty: str
    env_mode: str
    prompt_family: str
    prompt_style: str
    effort: str

    @property
    def condition_id(self) -> str:
        label = "default_omitted_medium" if self.effort == "default" else self.effort
        return f"{self.difficulty}_{self.env_mode}_{self.prompt_family}_{label}"


MAPS = (
    MapSpec("map1_10roads", "RT10", "data/map1_10roads/tasks.json"),
    MapSpec("map2_12roads", "RT12", "data/map2_12roads/tasks.json"),
    MapSpec("map3_15roads", "RT15", "data/map3_15roads/tasks.json"),
    MapSpec("map4_18roads", "RT18", "data/map4_18roads/tasks.json"),
    MapSpec("map5_20roads", "RT20", "data/map5_20roads/tasks.json"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def all_conditions() -> list[Condition]:
    result: list[Condition] = []
    # Run the complete instructional family first, as requested.
    for difficulty in ("easy", "hard"):
        for env_mode in ("realtime", "static"):
            for effort in ("none", "low", "default"):
                result.append(
                    Condition(
                        len(result), difficulty, env_mode,
                        "instructional", "instructional", effort,
                    )
                )
    # One representative action-only setting: easy + realtime.
    for effort in ("none", "low", "default"):
        result.append(
            Condition(
                len(result), "easy", "realtime",
                "action_only", "openai_action_only", effort,
            )
        )
    return result


def all_tasks(selected: set[int] | None = None) -> list[TaskSpec]:
    result: list[TaskSpec] = []
    for map_spec in MAPS:
        tasks = read_json(REPO_ROOT / map_spec.task_file)["tasks"]
        for index, task in enumerate(tasks):
            task_id = int(task.get("task_id", index))
            if selected is not None and task_id not in selected:
                continue
            points = reconstruct_route_points(task)
            length_cm = route_length_cm(points)
            result.append(
                TaskSpec(
                    map_spec.name,
                    map_spec.ue_map,
                    map_spec.task_file,
                    index,
                    task_id,
                    length_cm / 100.0,
                    route_length_to_max_steps(length_cm),
                )
            )
    if selected is not None and {item.task_id for item in result} != selected:
        raise ValueError(f"Unknown task IDs: {sorted(selected - {x.task_id for x in result})}")
    return result


def planned_manifest() -> dict[str, Any]:
    tasks = all_tasks()
    conditions = all_conditions()
    cap_steps = sum(item.max_steps for item in tasks) * len(conditions)
    ceiling = cap_steps * (2000 * INPUT_PRICE + 32 * OUTPUT_PRICE) / 1_000_000
    return {
        "schema_version": 1,
        "created_at": now(),
        "model": MODEL,
        "api_mode": "responses",
        "reasoning_arms": {
            "none": "explicit reasoning.effort=none",
            "low": "explicit reasoning.effort=low",
            "default": "reasoning field omitted; GPT-5.6 documented default is medium, not adaptive",
        },
        "max_output_tokens": None,
        "max_steps_policy": "3 * floor(shortest route length in meters)",
        "seed": 0,
        "image_detail": "low",
        "text_verbosity": "low",
        "traffic_policy": "visual_only",
        "static_signal_vehicles": True,
        "record_per_step": True,
        "action_frame_mode": "image_history",
        "observation_resolution": [OBSERVATION_WIDTH, OBSERVATION_HEIGHT],
        "ue_resolution": [UE_WIDTH, UE_HEIGHT],
        "ue_fps": 30,
        "tasks": [asdict(item) for item in tasks],
        "conditions": [asdict(item) | {"condition_id": item.condition_id} for item in conditions],
        "rollout_count": len(tasks) * len(conditions),
        "sum_max_steps": cap_steps,
        "planning": {
            "assumed_input_tokens_per_step": 2000,
            "assumed_billed_output_tokens_per_step": 32,
            "input_price_usd_per_million": INPUT_PRICE,
            "output_price_usd_per_million": OUTPUT_PRICE,
            "all_caps_estimate_usd": round(ceiling, 4),
        },
    }


def ensure_suite(suite: Path) -> dict[str, Any]:
    path = suite / "suite_manifest.json"
    planned = planned_manifest()
    if path.exists():
        existing = read_json(path)
        for key in ("model", "reasoning_arms", "max_output_tokens", "max_steps_policy", "tasks", "conditions"):
            if existing.get(key) != planned.get(key):
                raise RuntimeError(f"Existing suite differs in immutable field {key}")
        return existing
    write_json(path, planned)
    return planned


def cell_path(suite: Path, task: TaskSpec, condition: Condition) -> Path:
    return suite / "cells" / condition.condition_id / task.map_name / f"task_{task.task_id:03d}"


@contextmanager
def claim_cell(path: Path, enabled: bool):
    """Hold a process-scoped nonblocking claim for one rollout cell."""
    if not enabled:
        yield True
        return
    path.mkdir(parents=True, exist_ok=True)
    lock_path = path / '.worker_claim.lock'
    with lock_path.open('a+', encoding='utf-8') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({
                'pid': os.getpid(),
                'claimed_at': now(),
            }) + '\n')
            stream.flush()
            yield True
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def png_size(path: Path) -> tuple[int, int] | None:
    try:
        header = path.read_bytes()[:24]
    except OSError:
        return None
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", header[16:24])


def result_files(attempt: Path) -> list[Path]:
    return sorted(attempt.glob("task_*.json"))


def validate_attempt(
    attempt: Path, task: TaskSpec, condition: Condition
) -> tuple[Path | None, list[str]]:
    files = result_files(attempt)
    if len(files) != 1:
        return None, [f"expected one result JSON; found {len(files)}"]
    path = files[0]
    try:
        result = read_json(path)
    except Exception as exc:
        return path, [f"unreadable result: {exc}"]
    issues: list[str] = []
    # WorldManager intentionally preserves the historical canonical result
    # label ``default`` for the human-facing ``hard`` CLI alias. Validate
    # against the canonical label while keeping condition IDs and commands
    # expressed as easy/hard for the experiment matrix.
    result_difficulty = "default" if condition.difficulty == "hard" else condition.difficulty
    expected = {
        "task_id": task.task_id,
        "model": MODEL,
        "difficulty": result_difficulty,
        "reasoning_effort": condition.effort,
        "prompt_style": condition.prompt_style,
        "realtime_thinking": condition.env_mode == "realtime",
        "max_steps": task.max_steps,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            issues.append(f"{key}={result.get(key)!r}; expected {value!r}")
    if result.get("max_llm_tokens") is not None:
        issues.append("output-token cap was not omitted")
    if result.get("rollout_error"):
        issues.append(f"rollout infrastructure error: {result['rollout_error']}")
    for key in ("total_prompt_tokens", "total_completion_tokens", "total_reasoning_tokens"):
        if not isinstance(result.get(key), (int, float)) or result[key] < 0:
            issues.append(f"invalid usage field {key}")
    decisions = int(result.get("decision_count") or 0)
    if decisions <= 0:
        issues.append("rollout contains no model decisions")
    if decisions > task.max_steps:
        issues.append("decision count exceeds cap")
    step_dirs = sorted(attempt.glob("task_*_steps"))
    manifests = sorted(step_dirs[0].glob("**/*_manifest.json")) if len(step_dirs) == 1 else []
    if decisions != len(manifests):
        issues.append(f"decision/manifest mismatch {decisions}/{len(manifests)}")
    previous_feedback: str | None = None
    previous_feedback_required = False
    for index, manifest_path in enumerate(manifests):
        try:
            manifest = read_json(manifest_path)
        except Exception as exc:
            issues.append(f"step {index} manifest unreadable: {exc}")
            continue
        prompt = manifest.get("prompt") or {}
        system = str(prompt.get("system") or "")
        user = str(prompt.get("user") or "")
        timing = (
            "environment continues to evolve during both your reasoning"
            if condition.env_mode == "realtime"
            else "The simulator is paused while you decide"
        )
        if timing not in system:
            issues.append(f"step {index} timing contract mismatch")
        if condition.prompt_family == "instructional":
            if "Reasoning: [ONE sentence, <=30 words]" not in system:
                issues.append(f"step {index} instructional contract missing")
            if "OUTPUT CONTRACT (this overrides" in system:
                issues.append(f"step {index} action-only override leaked")
        else:
            if "Return exactly these two lines and no other text" not in system:
                issues.append(f"step {index} action-only contract missing")
            if "Reasoning: [ONE sentence, <=30 words]" in system:
                issues.append(f"step {index} instructional contract leaked")
        if (
            previous_feedback_required
            and previous_feedback
            and previous_feedback not in user
        ):
            issues.append(f"step {index} previous feedback absent from prompt history")
        previous_feedback = str(manifest.get("feedback") or "")
        # Invalid/parse-error decisions are intentionally excluded from the
        # executable action history shown on the following step.
        previous_feedback_required = previous_feedback not in {
            "Invalid action (not executed)",
            "Parse error (no executable action)",
        }
        if not previous_feedback:
            issues.append(f"step {index} empty feedback")
        if not (manifest.get("model_output") or {}).get("parsed_action"):
            issues.append(f"step {index} missing parsed action")
        geometry = manifest.get("observation_geometry") or {}
        if geometry.get("all_waypoint_markers_visible") is not True:
            issues.append(f"step {index} waypoint marker projection failed")
        render = (manifest.get("timing") or {}).get("input_render") or {}
        if render.get("resolution_pixels") != [
            OBSERVATION_WIDTH,
            OBSERVATION_HEIGHT,
        ]:
            issues.append(f"step {index} observation resolution mismatch")
        images = manifest.get("input_images") or []
        retention = manifest.get("image_retention") or {"policy": "all", "retained": True}
        if retention.get("policy") == "first_last_ring_v1":
            first = int(retention.get("first_steps") or 0)
            last = int(retention.get("last_steps") or 0)
            should_retain = index < first or index >= max(0, decisions - last)
            if bool(retention.get("retained")) != should_retain:
                issues.append(f"step {index} image-retention state mismatch")
            if should_retain:
                if not 1 <= len(images) <= 2:
                    issues.append(f"step {index} expected one or two retained input images")
            else:
                if images:
                    issues.append(f"step {index} pruned manifest still references input images")
                if not 1 <= int(manifest.get("model_input_image_count") or 0) <= 2:
                    issues.append(f"step {index} missing pruned model-input image count")
                if list(manifest_path.parent.glob("*.png")):
                    issues.append(f"step {index} pruned image files remain")
        elif not 1 <= len(images) <= 2:
            issues.append(f"step {index} expected one or two input images")
        for name in images:
            if png_size(manifest_path.parent / name) != (
                OBSERVATION_WIDTH,
                OBSERVATION_HEIGHT,
            ):
                issues.append(f"step {index} missing/bad image {name}")
        usage = manifest.get("metrics") or {}
        for token_key in (
            "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"
        ):
            if token_key not in usage or usage[token_key] is None:
                issues.append(f"step {index} token field {token_key} absent")
        if len(issues) >= 50:
            issues.append("validation stopped after 50 issues")
            break
    return path, issues


def completed(path: Path, task: TaskSpec, condition: Condition) -> bool:
    try:
        status = read_json(path / "status.json")
        if status.get("state") != "completed":
            return False
        result_path = Path(status["result_path"])
        _, issues = validate_attempt(result_path.parent, task, condition)
        return not issues
    except Exception:
        return False


def make_ue_args(args: argparse.Namespace, gpu: int, port: int) -> argparse.Namespace:
    return argparse.Namespace(
        ue_launcher=str(args.ue_launcher), ue_gpu=gpu, ue_port=port,
        ue_host="127.0.0.1", ue_width=UE_WIDTH, ue_height=UE_HEIGHT, ue_fps=30,
        ue_no_rhi_thread=args.ue_no_rhi_thread,
        ue_ready_timeout=args.ue_ready_timeout,
        ue_settle_seconds=args.ue_settle_seconds,
        process_shutdown_timeout=args.process_shutdown_timeout,
    )


def start_ue(
    args: argparse.Namespace, worker_dir: Path, task: TaskSpec,
    gpu: int, port: int, generation: int,
) -> UEServer:
    log = worker_dir / "ue" / f"{task.map_name}_{generation:03d}.log"
    ue = UEServer(make_ue_args(args, gpu, port), log)
    ue.start(task.ue_map)
    write_json(log.with_suffix(".json"), {
        "time": now(), "map": task.map_name, "gpu": gpu, "port": port,
        "startup_seconds": ue.startup_seconds, "renderer": ue.renderer_info,
    })
    return ue


def latest_ue_generation(worker_dir: Path) -> int:
    """Return the highest persisted UE log generation for a worker."""
    generations: list[int] = []
    for path in (worker_dir / "ue").glob("*"):
        generation = path.stem.rsplit("_", 1)[-1]
        if generation.isdigit():
            generations.append(int(generation))
    return max(generations, default=0)


def runner_command(
    args: argparse.Namespace, task: TaskSpec, condition: Condition,
    attempt: Path, port: int,
) -> list[str]:
    command = [
        sys.executable, str(REPO_ROOT / "evaluation/run_openai_benchmark.py"),
        "--model", MODEL, "--key-file", str(args.key_file), "--api-mode", "responses",
        "--reasoning-effort", condition.effort, "--max-output-tokens", "none",
        "--image-detail", "low", "--text-verbosity", "low", "--service-tier", "default",
        "--task-file", task.task_file, "--task-index", str(task.task_index),
        "--difficulty", condition.difficulty, "--seed", "0", "--max-steps", "-1",
        "--ue-port", str(port), "--prompt-style", condition.prompt_style,
        "--traffic-policy", "visual_only", "--static-signal-vehicles",
        "--action-frame-mode", "image_history", "--results-dir", str(attempt),
        "--request-timeout", str(args.request_timeout),
    ]
    if condition.env_mode == "static":
        command.append("--static-thinking")
    return command


def run_monitored(
    command: list[str], log_path: Path, ue: UEServer, timeout: float,
    args: argparse.Namespace,
) -> tuple[int, str | None]:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    # Shared GPUs can leave UE responsive at the TCP level while its first
    # setres/asset-compilation command takes longer than the normal 30-second
    # guard. Keep later requests bounded, but allow cold startup.
    environment["SIMWORLD_UNREALCV_REQUEST_TIMEOUT_SECONDS"] = "120"
    environment["SIMWORLD_UNREALCV_RECONNECT_RETRIES"] = "0"
    environment["SIMWORLD_OBSERVATION_WIDTH"] = str(OBSERVATION_WIDTH)
    environment["SIMWORLD_OBSERVATION_HEIGHT"] = str(OBSERVATION_HEIGHT)
    environment["SIMWORLD_AGENT_CAMERA_FOV_DEG"] = "100"
    environment["SIMWORLD_FIRST_PERSON_CAMERA_PITCH_DEG"] = "-25"
    environment["SIMWORLD_RECORD_PNG_COMPRESS_LEVEL"] = str(
        args.record_png_compress_level
    )
    environment["SIMWORLD_RECORD_IMAGES_FIRST_STEPS"] = str(
        args.record_images_first_steps
    )
    environment["SIMWORLD_RECORD_IMAGES_LAST_STEPS"] = str(
        args.record_images_last_steps
    )
    environment["SIMWORLD_CONCURRENT_REALTIME_INFERENCE"] = "1"
    environment["SIMWORLD_RECORD_OUTPUT_IMAGES"] = "0"
    environment["SIMWORLD_RECORD_DEMO_IMAGES"] = "0"
    if args.fast_simulation:
        # UnrealCV requests are synchronous; these are proven Qwen benchmark
        # fences and do not change simulated time or static/realtime semantics.
        environment["SIMWORLD_SYNC_SETTLE_SECONDS"] = "0.02"
        environment["SIMWORLD_TICK_INTERVAL_SETTLE_SECONDS"] = "0"
        environment["SIMWORLD_TICK_COMPLETION_SETTLE_SECONDS"] = "0.05"
    with log_path.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        infrastructure_error = None
        while process.poll() is None:
            # Do not open TCP probes while the benchmark client owns an
            # UnrealCV connection; this server treats probes as real clients.
            if not ue.process.alive:
                infrastructure_error = "owned UE process exited"
                process.terminate()
                break
            if time.monotonic() - started > timeout:
                infrastructure_error = f"rollout exceeded {timeout:g}s timeout"
                process.terminate()
                break
            time.sleep(2)
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait(timeout=10)
    return code, infrastructure_error


def worker_main(args: argparse.Namespace) -> int:
    suite = args.suite_dir.resolve()
    ensure_suite(suite)
    tasks = all_tasks(set(args.task_ids) if args.task_ids else None)
    selected_condition_ids = set(args.condition_ids or ())
    eligible_conditions = [
        item for item in all_conditions()
        if not selected_condition_ids or item.condition_id in selected_condition_ids
    ]
    conditions = (
        eligible_conditions
        if args.dynamic_queue
        else [
            item for item in eligible_conditions
            if item.ordinal % args.worker_count == args.worker_id
        ]
    )
    gpu, port = args.gpus[args.worker_id], args.ports[args.worker_id]
    worker_dir = suite / "workers" / f"worker_{args.worker_id:02d}"
    events = worker_dir / "events.jsonl"
    ue: UEServer | None = None
    # Resume after the highest persisted generation so a restarted worker
    # cannot overwrite UE startup logs from an earlier process.
    generation = latest_ue_generation(worker_dir)
    needs_attention = 0
    done = 0
    work_items = [
        (task, [item for item in conditions if item.prompt_family == family])
        for family in ("instructional", "action_only")
        for task in tasks
    ]
    try:
        for task, phase_conditions in work_items:
            for condition in phase_conditions:
                cell = cell_path(suite, task, condition)
                with claim_cell(cell, args.dynamic_queue) as claimed:
                    if not claimed:
                        continue
                    if args.resume and completed(cell, task, condition):
                        done += 1
                        continue
                    cell_done = False
                    for attempt_number in range(1, args.max_attempts + 1):
                        attempt = cell / f"attempt_{attempt_number:03d}"
                        existing_artifacts = (
                            attempt.exists() and any(attempt.iterdir())
                        )
                        if existing_artifacts:
                            result_path, issues = validate_attempt(
                                attempt, task, condition
                            )
                            if result_path is not None and not issues:
                                write_json(cell / "status.json", {
                                    "state": "completed",
                                    "validated_at": now(),
                                    "attempt": attempt_number,
                                    "result_path": str(result_path),
                                })
                                done += 1
                                cell_done = True
                                break
                            # Never mix a resumed process with partial images,
                            # token logs, or API responses from an interrupted
                            # attempt. Preserve it for spend/provenance audit.
                            append_jsonl(events, {
                                "time": now(),
                                "event": "existing_attempt_skipped",
                                "worker": args.worker_id,
                                "task_id": task.task_id,
                                "condition": condition.condition_id,
                                "attempt": attempt_number,
                                "validation_issues": issues,
                            })
                            continue
                        if ue is None or not ue.healthy():
                            if ue is not None:
                                ue.stop()
                            generation += 1
                            ue = start_ue(
                                args, worker_dir, task, gpu, port, generation
                            )
                        attempt.mkdir(parents=True, exist_ok=True)
                        write_json(cell / "status.json", {
                            "state": "running", "time": now(),
                            "worker_id": args.worker_id,
                            "attempt": attempt_number,
                            "task": asdict(task),
                            "condition": asdict(condition) | {
                                "condition_id": condition.condition_id
                            },
                        })
                        write_json(worker_dir / "status.json", {
                            "state": "running", "time": now(),
                            "completed": done, "task_id": task.task_id,
                            "condition": condition.condition_id,
                            "attempt": attempt_number,
                            "ue_healthy": ue.healthy(),
                            "gpu": gpu, "port": port,
                            "dynamic_queue": args.dynamic_queue,
                        })
                        code, infrastructure_error = run_monitored(
                            runner_command(
                                args, task, condition, attempt, port
                            ),
                            attempt / "rollout.log",
                            ue,
                            args.cell_timeout,
                            args,
                        )
                        result_path, issues = validate_attempt(
                            attempt, task, condition
                        )
                        if infrastructure_error:
                            issues.insert(0, infrastructure_error)
                        if result_path is None:
                            issues.insert(0, f"exit {code} without a result")
                        record = {
                            "time": now(), "event": "attempt_finished",
                            "worker": args.worker_id,
                            "task_id": task.task_id,
                            "condition": condition.condition_id,
                            "attempt": attempt_number,
                            "return_code": code,
                            "result_path": (
                                str(result_path) if result_path else None
                            ),
                            "validation_issues": issues,
                        }
                        append_jsonl(events, record)
                        write_json(attempt / "validation.json", record)
                        if result_path is not None and not issues:
                            result = read_json(result_path)
                            write_json(cell / "status.json", {
                                "state": "completed", "time": now(),
                                "attempt": attempt_number,
                                "outcome": (
                                    "success" if result.get("success")
                                    else "task_failure"
                                ),
                                "return_code": code,
                                "result_path": str(result_path),
                            })
                            done += 1
                            cell_done = True
                            break
                        if not ue.healthy():
                            ue.stop()
                            ue = None
                        if attempt_number < args.max_attempts:
                            time.sleep(min(30, 5 * attempt_number))
                    if not cell_done:
                        needs_attention += 1
                        write_json(cell / "status.json", {
                            "state": "needs_attention", "time": now(),
                            "attempts": args.max_attempts,
                        })
                    # Isolate every measured cell behind a fresh UE process.
                    if ue is not None:
                        ue.stop()
                        ue = None
        write_json(worker_dir / "status.json", {
            "state": "finished", "time": now(), "completed": done,
            "needs_attention": needs_attention,
            "gpu": gpu, "port": port,
            "dynamic_queue": args.dynamic_queue,
        })
        return 0 if needs_attention == 0 else 2
    finally:
        if ue is not None:
            ue.stop()

def build_summary(
    suite: Path, *, validate_completed: bool = False
) -> dict[str, Any]:
    tasks = {item.task_id: item for item in all_tasks()}
    conditions = {item.condition_id: item for item in all_conditions()}
    groups = {condition_id: [] for condition_id in conditions}
    states: dict[str, int] = {}
    validation_failures: list[dict[str, Any]] = []
    for status_path in sorted((suite / "cells").glob("**/status.json")):
        try:
            status = read_json(status_path)
            state = str(status.get("state", "unknown"))
            states[state] = states.get(state, 0) + 1
            if state != "completed":
                continue
            condition_id = status_path.parents[2].name
            result_path = Path(status["result_path"])
            if validate_completed:
                task_id = int(status_path.parent.name.removeprefix("task_"))
                task = tasks[task_id]
                condition = conditions[condition_id]
                _, issues = validate_attempt(
                    result_path.parent, task, condition
                )
                if issues:
                    validation_failures.append({
                        "condition_id": condition_id,
                        "task_id": task_id,
                        "result_path": str(result_path),
                        "issues": issues,
                    })
                    continue
            result = read_json(result_path)
            groups[condition_id].append(result)
        except Exception:
            states["unreadable"] = states.get("unreadable", 0) + 1
    rows = []
    for condition in conditions.values():
        results = groups[condition.condition_id]
        prompt_tokens = sum(int(x.get("total_prompt_tokens") or 0) for x in results)
        output_tokens = sum(int(x.get("total_completion_tokens") or 0) for x in results)
        reasoning_tokens = sum(int(x.get("total_reasoning_tokens") or 0) for x in results)
        rows.append({
            "condition_id": condition.condition_id, "difficulty": condition.difficulty,
            "env_mode": condition.env_mode, "prompt_family": condition.prompt_family,
            "effort_request": condition.effort,
            "effective_effort": "medium_default" if condition.effort == "default" else condition.effort,
            "completed": len(results), "successes": sum(bool(x.get("success")) for x in results),
            "success_rate": sum(bool(x.get("success")) for x in results) / len(results) if results else None,
            "mean_spl": fmean(float(x.get("spl") or 0) for x in results) if results else None,
            "mean_decisions": fmean(float(x.get("decision_count") or 0) for x in results) if results else None,
            "collisions": sum(int(x.get("collision_count") or 0) for x in results),
            "red_light_violations": sum(int(x.get("red_light_violations_count") or 0) for x in results),
            "illegal_crossing_violations": sum(int(x.get("illegal_crossing_violations_count") or 0) for x in results),
            "parse_errors": sum(int(x.get("parse_error_count") or 0) for x in results),
            "input_tokens": prompt_tokens, "billed_output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "visible_output_tokens": output_tokens - reasoning_tokens,
            "estimated_api_cost_usd": round((prompt_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE) / 1_000_000, 6),
        })
    return {
        "generated_at": now(), "expected_rollouts": expected_rollouts_for_suite(suite), "status_counts": states,
        "completed": sum(len(value) for value in groups.values()),
        "fresh_validation_requested": validate_completed,
        "fresh_validation_failures": validation_failures,
        "estimated_api_cost_usd": round(sum(x["estimated_api_cost_usd"] for x in rows), 6),
        "conditions": rows,
    }


def expected_rollouts_for_suite(suite: Path) -> int:
    """Use a persisted subset size when a campaign selected only some cells."""
    default = len(all_tasks()) * len(all_conditions())
    try:
        selected = read_json(suite / "execution_profile.json").get(
            "selected_rollouts"
        )
        if isinstance(selected, int) and selected >= 0:
            return selected
    except Exception:
        pass
    return default


def write_summary(
    suite: Path, *, validate_completed: bool = False
) -> dict[str, Any]:
    summary = build_summary(
        suite, validate_completed=validate_completed
    )
    write_json(suite / "summary.json", summary)
    rows = summary["conditions"]
    with (suite / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def coordinator_main(args: argparse.Namespace) -> int:
    suite = args.suite_dir.resolve()
    manifest = ensure_suite(suite)
    selected_tasks = all_tasks(set(args.task_ids) if args.task_ids else None)
    requested_condition_ids = set(args.condition_ids or ())
    selected_conditions = [
        item for item in all_conditions()
        if not requested_condition_ids or item.condition_id in requested_condition_ids
    ]
    write_json(suite / "execution_profile.json", {
        "updated_at": now(),
        "workers": args.workers,
        "gpus": args.gpus[:args.workers],
        "ports": args.ports[:args.workers],
        "dynamic_queue": args.dynamic_queue,
        "ue_settle_seconds": args.ue_settle_seconds,
        "fast_simulation": args.fast_simulation,
        "concurrent_realtime_inference": True,
        "static_environment_paused_during_api": True,
        "record_png_compress_level": args.record_png_compress_level,
        "record_images_first_steps": args.record_images_first_steps,
        "record_images_last_steps": args.record_images_last_steps,
        "record_output_images": False,
        "record_demo_images": False,
        "selected_task_ids": [item.task_id for item in selected_tasks],
        "selected_condition_ids": [
            item.condition_id for item in selected_conditions
        ],
        "selected_rollouts": len(selected_tasks) * len(selected_conditions),
    })
    if len(args.gpus) < args.workers or len(args.ports) < args.workers:
        raise ValueError("one GPU and port are required per worker")
    processes: list[subprocess.Popen] = []
    streams = []
    print(f"[coordinator] suite={suite} matrix={manifest['rollout_count']} selected={args.task_ids or 'all'}", flush=True)
    try:
        for worker_id in range(args.workers):
            worker_dir = suite / "workers" / f"worker_{worker_id:02d}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            stream = (worker_dir / "worker.log").open("ab", buffering=0)
            streams.append(stream)
            command = [
                sys.executable, str(Path(__file__).resolve()), "--mode", "worker",
                "--suite-dir", str(suite), "--worker-id", str(worker_id),
                "--worker-count", str(args.workers), "--gpus", *map(str, args.gpus),
                "--ports", *map(str, args.ports), "--key-file", str(args.key_file),
                "--ue-launcher", str(args.ue_launcher), "--max-attempts", str(args.max_attempts),
                "--request-timeout", str(args.request_timeout), "--cell-timeout", str(args.cell_timeout),
                "--ue-ready-timeout", str(args.ue_ready_timeout),
                "--ue-settle-seconds", str(args.ue_settle_seconds),
                "--process-shutdown-timeout", str(args.process_shutdown_timeout), "--resume",
                "--record-png-compress-level", str(args.record_png_compress_level),
                "--record-images-first-steps", str(args.record_images_first_steps),
                "--record-images-last-steps", str(args.record_images_last_steps),
            ]
            if args.dynamic_queue:
                command.append("--dynamic-queue")
            if not args.fast_simulation:
                command.append("--no-fast-simulation")
            if args.ue_no_rhi_thread:
                command.append("--ue-no-rhi-thread")
            if args.task_ids:
                command.extend(["--task-ids", *map(str, args.task_ids)])
            if args.condition_ids:
                command.extend(["--condition-ids", *args.condition_ids])
            processes.append(subprocess.Popen(
                command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True,
            ))
        last_report = 0.0
        while any(process.poll() is None for process in processes):
            if time.monotonic() - last_report >= args.status_interval:
                summary = write_summary(suite)
                print(
                    f"[coordinator] {now()} completed={summary['completed']}/"
                    f"{manifest['rollout_count']} "
                    f"states={summary['status_counts']} cost=${summary['estimated_api_cost_usd']:.4f}",
                    flush=True,
                )
                last_report = time.monotonic()
            time.sleep(2)
        codes = [process.wait() for process in processes]
        summary = write_summary(suite)
        print(
            f"[coordinator] workers={codes} completed={summary['completed']}/"
            f"{manifest['rollout_count']} states={summary['status_counts']}",
            flush=True,
        )
        return 0 if all(code == 0 for code in codes) else 2
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        return 130
    finally:
        for stream in streams:
            stream.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "coordinator", "worker", "summary"), default="coordinator")
    parser.add_argument("--suite-dir", type=Path, default=RESULTS_ROOT / "openai_gpt5.6_luna_all36_matrix_20260902")
    parser.add_argument("--key-file", type=Path, default=KEY_FILE)
    parser.add_argument("--ue-launcher", type=Path, default=UE_LAUNCHER)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--gpus", nargs="+", type=int, default=[3, 0])
    parser.add_argument("--ports", nargs="+", type=int, default=[9012, 9013])
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument(
        "--condition-ids", nargs="+",
        help="Run only these exact condition IDs (resume-safe).",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--cell-timeout", type=float, default=14400.0)
    parser.add_argument("--ue-ready-timeout", type=float, default=180.0)
    parser.add_argument("--ue-settle-seconds", type=float, default=8.0)
    parser.add_argument("--process-shutdown-timeout", type=float, default=20.0)
    parser.add_argument("--status-interval", type=float, default=30.0)
    parser.add_argument("--dynamic-queue", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-simulation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-png-compress-level", type=int, default=0)
    parser.add_argument("--record-images-first-steps", type=int, default=3)
    parser.add_argument("--record-images-last-steps", type=int, default=3)
    parser.add_argument("--ue-no-rhi-thread", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.workers <= 0 or args.worker_count <= 0 or not 0 <= args.worker_id < args.worker_count:
        parser.error("invalid worker count/id")
    if not 0 <= args.record_png_compress_level <= 9:
        parser.error("--record-png-compress-level must be in [0, 9]")
    if args.record_images_first_steps < 0 or args.record_images_last_steps < 0:
        parser.error("image retention counts must be nonnegative")
    known_condition_ids = {item.condition_id for item in all_conditions()}
    unknown_condition_ids = set(args.condition_ids or ()) - known_condition_ids
    if unknown_condition_ids:
        parser.error(
            "unknown --condition-ids: " + ", ".join(sorted(unknown_condition_ids))
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "plan":
        print(json.dumps(ensure_suite(args.suite_dir.resolve()), indent=2))
        return 0
    if args.mode == "summary":
        print(json.dumps(
            write_summary(
                args.suite_dir.resolve(), validate_completed=True
            ),
            indent=2,
        ))
        return 0
    return worker_main(args) if args.mode == "worker" else coordinator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
