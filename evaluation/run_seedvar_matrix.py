#!/usr/bin/env python3
"""Plan or run the full OpenAI SimWorld-RealTime frontier matrix.

The default mode is plan-only and never starts Unreal Engine or calls an API.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import re
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
from llm.reasoning_capabilities import openrouter_reasoning_profile
from utils.route_steps import route_length_to_max_steps
from utils.task_routes import reconstruct_route_points, route_length_cm

# 2026-09-21: seed-variance runner. Identical to run_openai_frontier_matrix.py except that the
# protocol seed comes from SIMWORLD_SEED (default 0), so the same routes replay under a different
# environment seed. Untracked on purpose: run_pipeline.sh fingerprints `git diff`, and a new
# untracked file leaves that fingerprint unchanged.
SEED = int(os.environ.get("SIMWORLD_SEED", "0"))

MODEL = os.environ.get("SIMWORLD_CAMPAIGN_MODEL", "gpt-5.6-luna")
PROVIDER = os.environ.get("SIMWORLD_CAMPAIGN_PROVIDER", "openai")
API_MODE = os.environ.get("SIMWORLD_CAMPAIGN_API_MODE", "responses")
RESULTS_ROOT = Path("results")
KEY_FILE = Path(os.environ.get(
    "SIMWORLD_CAMPAIGN_KEY_FILE",
    ".secrets/openrouter_api.txt" if PROVIDER == "openrouter"
    else ".secrets/openai_api.txt",
))
UE_LAUNCHER = Path(os.environ.get("SIMWORLD_UE_LAUNCHER", str(REPO_ROOT / "runtime/SimWorld.sh")))
INPUT_PRICE = float(os.environ.get("SIMWORLD_CAMPAIGN_INPUT_PRICE", "0.20"))
OUTPUT_PRICE = float(os.environ.get("SIMWORLD_CAMPAIGN_OUTPUT_PRICE", "1.20"))

DIFFICULTIES = tuple(
    value.strip()
    for value in os.environ.get(
        "SIMWORLD_CAMPAIGN_DIFFICULTIES", "easy"
    ).split(",")
    if value.strip()
)
ENV_MODES = tuple(
    value.strip()
    for value in os.environ.get(
        "SIMWORLD_CAMPAIGN_ENV_MODES", "realtime"
    ).split(",")
    if value.strip()
)
EFFORTS = tuple(
    value.strip()
    for value in os.environ.get(
        "SIMWORLD_CAMPAIGN_EFFORTS", "none,low,high"
    ).split(",")
    if value.strip()
)
# Historical CLI name: this sends an ordered sequence of still images, not an MP4.
ACTION_FRAME_MODE = "video_history"
ASSUMED_INPUT_TOKENS_PER_STEP = 2000
ASSUMED_OUTPUT_TOKENS_PER_STEP = {
    "none": 32,
    "low": 160,
    "medium": 256,
    "high": 1024,
}

MATRIX_MODELS = (
    {"model": "gpt-6-astra", "efforts": ("low", "high"), "input": 10.00, "output": 50.00, "provider": "openai", "api_mode": "responses"},
    {"model": "gpt-5.6-sol", "efforts": ("none", "low", "high"), "input": 4.00, "output": 20.00, "provider": "openai", "api_mode": "responses"},
    {"model": "gpt-5.6-terra", "efforts": ("none", "low", "high"), "input": 2.00, "output": 12.00, "provider": "openai", "api_mode": "responses"},
    {"model": "gpt-5.6-luna", "efforts": ("none", "low", "high"), "input": 0.20, "output": 1.20, "provider": "openai", "api_mode": "responses"},
)


def matrix_specs() -> tuple[dict[str, Any], ...]:
    """Return the API frontier matrix or one explicitly selected CLI arm."""
    if PROVIDER == "codex-cli":
        return ({
            "model": MODEL,
            "efforts": EFFORTS,
            "input": 0.0,
            "output": 0.0,
            "provider": PROVIDER,
            "api_mode": API_MODE,
        },)
    return MATRIX_MODELS



def max_output_tokens(effort: str) -> int | None:
    """Use provider-required caps while preserving uncapped OpenAI runs."""
    if PROVIDER == "openrouter" and "claude-haiku-4.5" in MODEL.lower():
        return 1280 if effort == "low" else 128
    return None


def output_cap_label(effort: str) -> str:
    value = max_output_tokens(effort)
    return "none" if value is None else str(value)


def estimated_output_tokens_per_step(effort: str) -> int:
    if PROVIDER == "openrouter" and effort == "low":
        return 1056
    return ASSUMED_OUTPUT_TOKENS_PER_STEP.get(effort, 256)


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
        label = "default_omitted" if self.effort == "default" else self.effort
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

def retry_delay_seconds(log_path: Path, attempt_number: int) -> float:
    """Honor provider throttle hints before starting a fresh isolated attempt."""
    base_delay = float(min(30, 5 * attempt_number))
    try:
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
    except OSError:
        return base_delay
    lowered = log_tail.lower()
    throttle_markers = (
        "in_flight_budget_exhausted",
        "rate_limit_exceeded",
        "error code: 429",
        "status code: 429",
    )
    if not any(marker in lowered for marker in throttle_markers):
        return base_delay
    match = re.search(
        r"retry-after['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)",
        log_tail,
        re.IGNORECASE,
    )
    return max(base_delay, float(match.group(1)) if match else 120.0)


def wait_for_benchmark_launch_slot(
    suite: Path, minimum_interval_seconds: float
) -> float:
    """Serialize only benchmark launches, leaving active rollouts parallel."""
    lock_path = suite / ".benchmark_launch.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        try:
            previous_slot = float(stream.read().strip() or 0.0)
        except ValueError:
            previous_slot = 0.0
        now_epoch = time.time()
        launch_slot = max(
            now_epoch,
            previous_slot + float(minimum_interval_seconds),
        )
        # Reserve the slot before sleeping, so another worker reserves the
        # following slot instead of waiting on a slow filesystem flush.
        stream.seek(0)
        stream.truncate()
        stream.write(f"{launch_slot:.6f}\n")
        stream.flush()
    delay = max(0.0, launch_slot - time.time())
    if delay:
        time.sleep(delay)
    return delay


def all_conditions() -> list[Condition]:
    result: list[Condition] = []
    for difficulty in DIFFICULTIES:
        for env_mode in ENV_MODES:
            for effort in EFFORTS:
                result.append(Condition(
                    len(result), difficulty, env_mode,
                    "instructional", "instructional", effort,
                ))
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
    ceiling = sum(
        task.max_steps
        * sum(
            ASSUMED_INPUT_TOKENS_PER_STEP * INPUT_PRICE
            + estimated_output_tokens_per_step(condition.effort) * OUTPUT_PRICE
            for condition in conditions
        )
        for task in tasks
    ) / 1_000_000
    return {
        "schema_version": 1,
        "created_at": now(),
        "model": MODEL,
        "provider": PROVIDER,
        "api_mode": API_MODE,
        "inference_surface": (
            "chatgpt_codex_cli" if PROVIDER == "codex-cli"
            else "platform_api"
        ),
        "benchmark_protocol_comparable_across_surfaces": True,
        "strict_model_transport_comparable_across_surfaces": (
            PROVIDER != "codex-cli"
        ),
        "comparison_note": (
            "Codex CLI uses the same benchmark prompt, observations, world, and "
            "scoring, but codex exec adds a product-level agent instruction layer. "
            "Keep CLI and Responses API as separate access-surface arms."
            if PROVIDER == "codex-cli" else None
        ),
        "reasoning_arms": {
            effort: (
                "reasoning disabled" if effort == "none"
                else "reasoning.effort omitted; provider/model default"
                if effort == "default"
                else f"explicit reasoning.effort={effort}"
            )
            for effort in EFFORTS
        },
        "catalog_default_reasoning_effort": (
            (openrouter_reasoning_profile(MODEL) or {}).get("default")
            if PROVIDER == "openrouter" else None
        ),
        "max_output_tokens_by_effort": {
            effort: max_output_tokens(effort) for effort in EFFORTS
        },
        "max_steps_policy": "3 * floor(shortest route length in meters)",
        "seed": SEED,
        "image_detail": "low",
        "text_verbosity": "low",
        "traffic_policy": "visual_only",
        "static_signal_vehicles": True,
        "record_per_step": True,
        "action_frame_mode": ACTION_FRAME_MODE,
        "observation_resolution": [720, 640],
        "ue_resolution": [1280, 720],
        "ue_fps": 30,
        "tasks": [asdict(item) for item in tasks],
        "conditions": [asdict(item) | {"condition_id": item.condition_id} for item in conditions],
        "rollout_count": len(tasks) * len(conditions),
        "sum_max_steps": cap_steps,
        "planning": {
            "assumed_input_tokens_per_step": ASSUMED_INPUT_TOKENS_PER_STEP,
            "assumed_billed_output_tokens_per_step": {
                effort: estimated_output_tokens_per_step(effort)
                for effort in EFFORTS
            },
            "input_price_usd_per_million": INPUT_PRICE,
            "output_price_usd_per_million": OUTPUT_PRICE,
            "billing_mode": (
                "chatgpt_codex_credits" if PROVIDER == "codex-cli"
                else "api_usage_based"
            ),
            "all_caps_estimate_usd": (
                None if PROVIDER == "codex-cli" else round(ceiling, 4)
            ),
        },
    }


def ensure_suite(suite: Path) -> dict[str, Any]:
    path = suite / "suite_manifest.json"
    planned = planned_manifest()
    if path.exists():
        existing = read_json(path)
        for key in (
            "model", "provider", "api_mode", "reasoning_arms",
            "max_output_tokens_by_effort", "max_steps_policy", "tasks", "conditions",
        ):
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
    return sorted(
        path for path in attempt.glob("task_*.json")
        if not path.name.endswith("_alignment_validation.json")
    )


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
        "provider": PROVIDER,
        "api_mode": API_MODE,
        "difficulty": result_difficulty,
        "seed": SEED,
        "reasoning_effort": condition.effort,
        "prompt_style": condition.prompt_style,
        "traffic_policy": "visual_only",
        "realtime_thinking": condition.env_mode == "realtime",
        "use_action_frames": True,
        "image_detail": "low",
        "static_signal_vehicles": True,
        "max_steps": task.max_steps,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            issues.append(f"{key}={result.get(key)!r}; expected {value!r}")
    expected_cap = max_output_tokens(condition.effort)
    if result.get("max_llm_tokens") != expected_cap:
        issues.append(
            f"max_llm_tokens={result.get('max_llm_tokens')!r}; "
            f"expected {expected_cap!r}"
        )
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
        if "continuing to the far\n  curb is still legal" not in user:
            issues.append(f"step {index} corrected crossing-continuation rule missing")
        if "Remaining inside the crossing after WALK ends" in user:
            issues.append(f"step {index} obsolete crossing-continuation rule leaked")
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
        if render.get("resolution_pixels") != [720, 640]:
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
                if not images:
                    issues.append(f"step {index} expected retained input images")
            else:
                if images:
                    issues.append(f"step {index} pruned manifest still references input images")
                if int(manifest.get("model_input_image_count") or 0) < 1:
                    issues.append(f"step {index} invalid pruned model-input image count")
                if list(manifest_path.parent.glob("*.png")):
                    issues.append(f"step {index} pruned image files remain")
        elif not images:
            issues.append(f"step {index} expected input images")
        for name in images:
            if png_size(manifest_path.parent / name) != (720, 640):
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
        ue_host="127.0.0.1", ue_width=1280, ue_height=720, ue_fps=30,
        ue_no_rhi_thread=args.ue_no_rhi_thread,
        ue_ready_timeout=args.ue_ready_timeout,
        ue_settle_seconds=args.ue_settle_seconds,
        process_shutdown_timeout=args.process_shutdown_timeout,
    )


def ue_port_for_generation(base_port: int, generation: int, stride: int) -> int:
    port = base_port + generation * stride if stride > 0 else base_port
    if not 1 <= port <= 65535:
        raise ValueError(
            f"UE port {port} is outside 1..65535 (base={base_port}, "
            f"generation={generation}, stride={stride})"
        )
    return port


def start_ue(
    args: argparse.Namespace, worker_dir: Path, task: TaskSpec,
    gpu: int, port: int, generation: int,
) -> UEServer:
    log = worker_dir / "ue" / f"{task.map_name}_{generation:03d}.log"
    ue_args = make_ue_args(args, gpu, port)
    # A packaged UE process can briefly retain its UnrealCV listener after the
    # process group has been terminated.  Treat that narrow shutdown/startup
    # race as recoverable so a healthy worker does not permanently leave the
    # coordinator after completing one cell.
    port_retry_deadline = time.monotonic() + 180.0
    while True:
        ue = UEServer(ue_args, log)
        try:
            ue.start(task.ue_map)
            # UnrealCV can open its listener before the packaged renderer has
            # finished the final viewport transition.  If that transition
            # subsequently trips UE's render-thread watchdog, its crash handler
            # may keep both the process and listener alive.  Check the retained
            # engine log as well as the ordinary process/port health probe so a
            # doomed renderer never consumes a benchmark attempt.
            try:
                startup_log = ue.internal_log_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                startup_log = ""
            fatal_during_settle = any(marker in startup_log for marker in (
                "Fatal error:",
                "Unhandled Exception:",
                "GameThread timed out waiting for RenderThread",
            ))
            if fatal_during_settle or not ue.healthy():
                raise RuntimeError(
                    "UE failed during post-ready settling; "
                    f"inspect {ue.internal_log_path}"
                )
            break
        except RuntimeError as exc:
            message = str(exc).lower()
            retryable_port_race = (
                "port" in message
                and ("already occupied" in message or "could not bind" in message)
            )
            retryable_post_ready_failure = (
                "failed during post-ready settling" in message
            )
            # If start() rejected an already-listening port before spawning a
            # process, there is no owned UE to stop.  Calling UEServer.stop()
            # in that case unnecessarily waits its full port-release window.
            if ue.process.alive:
                ue.stop()
            if (
                not (retryable_port_race or retryable_post_ready_failure)
                or time.monotonic() >= port_retry_deadline
            ):
                raise
            time.sleep(2.0)
        except BaseException:
            # start_ue() has not yet returned, so worker_main does not own this
            # local UEServer object and cannot clean it in its outer finally.
            # Ensure startup timeouts and interrupts cannot orphan a renderer.
            if ue.process.alive:
                ue.stop()
            raise
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
        "--provider", PROVIDER, "--model", MODEL,
        "--api-mode", API_MODE,
        "--reasoning-effort", condition.effort,
        "--max-output-tokens", output_cap_label(condition.effort),
        "--estimated-output-tokens-per-step", str(estimated_output_tokens_per_step(condition.effort)),
        "--image-detail", "low", "--text-verbosity", "low", "--service-tier", "default",
        "--task-file", task.task_file, "--task-index", str(task.task_index),
        "--difficulty", condition.difficulty, "--seed", str(SEED), "--max-steps", "-1",
        "--ue-port", str(port), "--prompt-style", condition.prompt_style,
        "--traffic-policy", "visual_only", "--static-signal-vehicles",
        "--action-frame-mode", ACTION_FRAME_MODE, "--results-dir", str(attempt),
        "--request-timeout", str(args.request_timeout),
        "--record-png-compress-level", str(args.record_png_compress_level),
        "--record-images-first-steps", str(args.record_images_first_steps),
        "--record-images-last-steps", str(args.record_images_last_steps),
        (
            "--record-output-images"
            if args.record_output_images
            else "--no-record-output-images"
        ),
    ]
    if PROVIDER != "codex-cli":
        command.extend(["--key-file", str(args.key_file)])
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
    environment["SIMWORLD_UNREALCV_REQUEST_TIMEOUT_SECONDS"] = str(
        args.unrealcv_request_timeout
    )
    environment["SIMWORLD_UNREALCV_RECONNECT_RETRIES"] = "0"
    environment["SIMWORLD_OBSERVATION_WIDTH"] = "720"
    environment["SIMWORLD_OBSERVATION_HEIGHT"] = "640"
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
    environment["SIMWORLD_RECORD_OUTPUT_IMAGES"] = (
        "1" if args.record_output_images else "0"
    )
    environment["SIMWORLD_RECORD_DEMO_IMAGES"] = "1"
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
            start_new_session=True,
        )
        infrastructure_error = None
        try:
            while process.poll() is None:
                # Do not open TCP probes while the benchmark client owns an
                # UnrealCV connection; this server treats probes as real clients.
                if not ue.process.alive:
                    infrastructure_error = "owned UE process exited"
                    stop_process_group_by_pid(process)
                    break
                if time.monotonic() - started > timeout:
                    infrastructure_error = f"rollout exceeded {timeout:g}s timeout"
                    stop_process_group_by_pid(process)
                    break
                time.sleep(2)
        except BaseException:
            stop_process_group_by_pid(process)
            raise
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait(timeout=10)
    return code, infrastructure_error


def stop_process_group_by_pid(
    process: subprocess.Popen[Any], grace_seconds: float = 20.0
) -> int:
    """Stop a rollout and any active Codex CLI child it owns."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=10)


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
    gpu = args.gpus[args.worker_id]
    base_port = args.ports[args.worker_id]
    port = base_port
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
                        launch_delay = 0.0
                        if ue is None or not ue.healthy():
                            if ue is not None:
                                ue.stop()
                            launch_delay = wait_for_benchmark_launch_slot(
                                suite, args.ue_start_stagger_seconds
                            )
                            generation += 1
                            port = ue_port_for_generation(
                                base_port, generation, args.ue_port_stride
                            )
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
                            "benchmark_launch_delay_seconds": launch_delay,
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
                        # A failed attempt may leave partially spawned actors even
                        # when UE still answers health probes. Never reuse that world.
                        if ue is not None:
                            ue.stop()
                            ue = None
                        if attempt_number < args.max_attempts:
                            retry_delay = retry_delay_seconds(
                                attempt / "rollout.log", attempt_number
                            )
                            append_jsonl(events, {
                                "time": now(), "event": "retry_delay",
                                "worker": args.worker_id,
                                "task_id": task.task_id,
                                "condition": condition.condition_id,
                                "after_attempt": attempt_number,
                                "seconds": retry_delay,
                            })
                            time.sleep(retry_delay)
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
        reported_cost = sum(float(x.get("total_api_cost_usd") or 0) for x in results)
        estimated_api_cost = (
            round(
                (prompt_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE)
                / 1_000_000,
                6,
            )
            if PROVIDER != "codex-cli" else None
        )
        rows.append({
            "condition_id": condition.condition_id, "difficulty": condition.difficulty,
            "env_mode": condition.env_mode, "prompt_family": condition.prompt_family,
            "effort_request": condition.effort,
            "effective_effort": (
                (openrouter_reasoning_profile(MODEL) or {}).get(
                    "default", "provider_default_omitted"
                )
                if condition.effort == "default"
                else condition.effort
            ),
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
            "estimated_api_cost_usd": estimated_api_cost,
            "reported_api_cost_usd": round(reported_cost, 6),
        })
    estimated_api_costs = [
        row["estimated_api_cost_usd"] for row in rows
        if row["estimated_api_cost_usd"] is not None
    ]
    return {
        "generated_at": now(), "expected_rollouts": expected_rollouts_for_suite(suite), "status_counts": states,
        "completed": sum(len(value) for value in groups.values()),
        "fresh_validation_requested": validate_completed,
        "fresh_validation_failures": validation_failures,
        "billing_mode": (
            "chatgpt_codex_credits" if PROVIDER == "codex-cli"
            else "api_usage_based"
        ),
        "estimated_api_cost_usd": (
            round(sum(estimated_api_costs), 6)
            if estimated_api_costs else None
        ),
        "reported_api_cost_usd": round(sum(x["reported_api_cost_usd"] for x in rows), 6),
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
        "ue_start_stagger_seconds": args.ue_start_stagger_seconds,
        "fast_simulation": args.fast_simulation,
        "concurrent_realtime_inference": True,
        "static_environment_paused_during_api": True,
        "record_png_compress_level": args.record_png_compress_level,
        "record_images_first_steps": args.record_images_first_steps,
        "record_images_last_steps": args.record_images_last_steps,
        "record_output_images": args.record_output_images,
        "record_demo_images": True,
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
    print(
        f"[coordinator] suite={suite} matrix={manifest['rollout_count']} "
        f"selected={len(selected_tasks) * len(selected_conditions)}",
        flush=True,
    )
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
                "--unrealcv-request-timeout", str(args.unrealcv_request_timeout),
                "--ue-ready-timeout", str(args.ue_ready_timeout),
                "--ue-settle-seconds", str(args.ue_settle_seconds),
                "--ue-start-stagger-seconds", str(args.ue_start_stagger_seconds),
                "--process-shutdown-timeout", str(args.process_shutdown_timeout), "--resume",
                "--record-png-compress-level", str(args.record_png_compress_level),
                "--record-images-first-steps", str(args.record_images_first_steps),
                "--record-images-last-steps", str(args.record_images_last_steps),
            ]
            command.append(
                "--record-output-images"
                if args.record_output_images
                else "--no-record-output-images"
            )
            if args.dynamic_queue:
                command.append("--dynamic-queue")
            if not args.fast_simulation:
                command.append("--no-fast-simulation")
            if args.ue_no_rhi_thread:
                command.append("--ue-no-rhi-thread")
            if args.ue_port_stride > 0:
                command.extend(["--ue-port-stride", str(args.ue_port_stride)])
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
                    f"{summary['expected_rollouts']} "
                    f"states={summary['status_counts']} "
                    + (
                        "Codex-credit usage recorded in results"
                        if summary["estimated_api_cost_usd"] is None
                        else f"cost=${summary['estimated_api_cost_usd']:.4f}"
                    ),
                    flush=True,
                )
                last_report = time.monotonic()
            time.sleep(2)
        codes = [process.wait() for process in processes]
        summary = write_summary(suite)
        print(
            f"[coordinator] workers={codes} completed={summary['completed']}/"
            f"{summary['expected_rollouts']} states={summary['status_counts']}",
            flush=True,
        )
        matrix_complete = (
            summary["completed"] == summary["expected_rollouts"]
            and not summary["fresh_validation_failures"]
        )
        # In dynamic-queue mode another worker can finish every claimed cell
        # after one sibling suffers a GPU/startup failure. The validated matrix,
        # rather than a stale worker exit code, is the completion authority.
        if args.dynamic_queue and matrix_complete:
            return 0
        return 0 if all(code == 0 for code in codes) else 2
    except KeyboardInterrupt:
        # Workers run in independent sessions so an interrupt delivered to the
        # coordinator does not reach them automatically. Signal each owned
        # group once, then wait through its bounded UE cleanup before
        # escalating. Returning immediately here can orphan policy calls and
        # defeat an account-usage guard.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        graceful_deadline = time.monotonic() + 120.0
        while (
            any(process.poll() is None for process in processes)
            and time.monotonic() < graceful_deadline
        ):
            time.sleep(0.25)
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        terminate_deadline = time.monotonic() + 20.0
        while (
            any(process.poll() is None for process in processes)
            and time.monotonic() < terminate_deadline
        ):
            time.sleep(0.25)
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        return 130
    finally:
        for stream in streams:
            stream.close()




def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "_")


def matrix_environment(spec: dict[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "SIMWORLD_CAMPAIGN_MODEL": str(spec["model"]),
        "SIMWORLD_CAMPAIGN_PROVIDER": str(spec.get("provider", "openai")),
        "SIMWORLD_CAMPAIGN_API_MODE": str(spec.get("api_mode", "responses")),
        "SIMWORLD_CAMPAIGN_INPUT_PRICE": str(spec["input"]),
        "SIMWORLD_CAMPAIGN_OUTPUT_PRICE": str(spec["output"]),
        "SIMWORLD_CAMPAIGN_DIFFICULTIES": ",".join(DIFFICULTIES),
        "SIMWORLD_CAMPAIGN_ENV_MODES": ",".join(ENV_MODES),
        "SIMWORLD_CAMPAIGN_EFFORTS": ",".join(spec["efforts"]),
    })
    return environment


def matrix_child_command(
    args: argparse.Namespace,
    spec: dict[str, Any],
    mode: str,
) -> list[str]:
    suite = args.suite_dir.resolve() / model_slug(str(spec["model"]))
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--suite-dir",
        str(suite),
        "--ue-launcher",
        str(args.ue_launcher),
        "--workers",
        str(args.workers),
        "--worker-count",
        str(args.workers),
        "--gpus",
        *map(str, args.gpus),
        "--ports",
        *map(str, args.ports),
        "--max-attempts",
        str(args.max_attempts),
        "--request-timeout",
        str(args.request_timeout),
        "--unrealcv-request-timeout",
        str(args.unrealcv_request_timeout),
        "--cell-timeout",
        str(args.cell_timeout),
        "--ue-ready-timeout",
        str(args.ue_ready_timeout),
        "--ue-settle-seconds",
        str(args.ue_settle_seconds),
        "--ue-start-stagger-seconds",
        str(args.ue_start_stagger_seconds),
        "--process-shutdown-timeout",
        str(args.process_shutdown_timeout),
        "--status-interval",
        str(args.status_interval),
        "--record-png-compress-level",
        str(args.record_png_compress_level),
        "--record-images-first-steps",
        str(args.record_images_first_steps),
        "--record-images-last-steps",
        str(args.record_images_last_steps),
    ]
    if str(spec.get("provider", "openai")) != "codex-cli":
        command.extend(["--key-file", str(args.key_file)])
    if args.dynamic_queue:
        command.append("--dynamic-queue")
    if not args.fast_simulation:
        command.append("--no-fast-simulation")
    if args.record_output_images:
        command.append("--record-output-images")
    if args.ue_no_rhi_thread:
        command.append("--ue-no-rhi-thread")
    if args.ue_port_stride > 0:
        command.extend(["--ue-port-stride", str(args.ue_port_stride)])
    if args.task_ids:
        command.extend(["--task-ids", *map(str, args.task_ids)])
    if args.condition_ids:
        command.extend(["--condition-ids", *args.condition_ids])
    return command


def matrix_cost_scenario(
    tasks: list[TaskSpec],
    decisions_per_rollout: int | None,
) -> dict[str, Any]:
    total_cost = 0.0
    total_decisions = 0
    by_model: dict[str, float | None] = {}
    specs = matrix_specs()
    api_billed = all(
        str(spec.get("provider", "openai")) != "codex-cli"
        for spec in specs
    )
    for spec in specs:
        model_cost = 0.0
        for effort in spec["efforts"]:
            if decisions_per_rollout is None:
                decisions = (
                    sum(task.max_steps for task in tasks)
                    * len(DIFFICULTIES)
                    * len(ENV_MODES)
                )
            else:
                decisions = (
                    len(tasks)
                    * len(DIFFICULTIES)
                    * len(ENV_MODES)
                    * decisions_per_rollout
                )
            output_tokens = estimated_output_tokens_per_step(str(effort))
            model_cost += decisions * (
                ASSUMED_INPUT_TOKENS_PER_STEP * float(spec["input"])
                + output_tokens * float(spec["output"])
            ) / 1_000_000
            total_decisions += decisions
        by_model[str(spec["model"])] = (
            round(model_cost, 2) if api_billed else None
        )
        total_cost += model_cost
    return {
        "decisions_per_rollout": decisions_per_rollout,
        "total_decisions": total_decisions,
        "estimated_cost_usd": round(total_cost, 2) if api_billed else None,
        "by_model_usd": by_model,
    }


def matrix_plan(args: argparse.Namespace) -> dict[str, Any]:
    tasks = all_tasks(set(args.task_ids) if args.task_ids else None)
    specs = matrix_specs()
    effort_arms = sum(len(spec["efforts"]) for spec in specs)
    condition_count = effort_arms * len(DIFFICULTIES) * len(ENV_MODES)
    rollout_count = condition_count * len(tasks)
    commands = []
    for spec in matrix_specs():
        env_prefix = {
            key: value
            for key, value in matrix_environment(spec).items()
            if key.startswith("SIMWORLD_CAMPAIGN_")
        }
        command = matrix_child_command(args, spec, "coordinator")
        commands.append({
            "model": spec["model"],
            "suite_dir": str(
                args.suite_dir.resolve() / model_slug(str(spec["model"]))
            ),
            "environment": env_prefix,
            "command": command,
        })
    return {
        "schema_version": 1,
        "mode": "plan_only",
        "starts_rollouts": False,
        "models": [
            {
                "model": spec["model"],
                "reasoning_efforts": list(spec["efforts"]),
                "billing_mode": (
                    "chatgpt_codex_credits"
                    if str(spec.get("provider")) == "codex-cli"
                    else "api_usage_based"
                ),
                "standard_short_context_price_usd_per_million": (
                    None if str(spec.get("provider")) == "codex-cli"
                    else {"input": spec["input"], "output": spec["output"]}
                ),
            }
            for spec in specs
        ],
        "difficulties": list(DIFFICULTIES),
        "environment_modes": list(ENV_MODES),
        "tasks_per_condition": len(tasks),
        "condition_count": condition_count,
        "rollout_count": rollout_count,
        "sum_route_budget_decisions": sum(
            task.max_steps for task in tasks
        ) * condition_count,
        "shared_protocol": {
            "seed": SEED,
            "api_mode": API_MODE,
            "inference_surface": (
                "chatgpt_codex_cli" if PROVIDER == "codex-cli"
                else "platform_api"
            ),
            "service_tier": "default",
            "prompt_style": "instructional",
            "image_detail": "low",
            "text_verbosity": "low",
            "action_frame_mode": ACTION_FRAME_MODE,
            "action_frame_cadence_seconds": 0.5,
            "policy_image_count_cap": None,
            "max_output_tokens": None,
            "traffic_policy": "visual_only",
            "static_signal_vehicles": True,
            "max_steps": "3 * floor(shortest route length in meters)",
            "observation_resolution": [720, 640],
        },
        "cost_assumptions": {
            "input_tokens_per_decision": ASSUMED_INPUT_TOKENS_PER_STEP,
            "billed_output_tokens_per_decision": ASSUMED_OUTPUT_TOKENS_PER_STEP,
            "standard_short_context_pricing": PROVIDER != "codex-cli",
            "billing_mode": (
                "chatgpt_codex_credits" if PROVIDER == "codex-cli"
                else "api_usage_based"
            ),
            "prompt_cache_savings_assumed": False,
        },
        "cost_scenarios": {
            "30_decisions_per_rollout": matrix_cost_scenario(tasks, 30),
            "50_decisions_per_rollout": matrix_cost_scenario(tasks, 50),
            "full_route_budget": matrix_cost_scenario(tasks, None),
        },
        "time_estimate": {
            "recommended_workers": args.workers,
            "parallelism": (
                "one isolated UE instance and one in-flight Codex CLI turn per worker"
                if PROVIDER == "codex-cli"
                else "one isolated UE instance and one in-flight API call per worker"
            ),
            "empirical_reference": (
                "540 prior OpenAI rollouts completed in about 26.15 hours with "
                "7 workers (about 20.3 worker-minutes per rollout)"
            ),
            "projected_days_at_empirical_throughput": round(
                rollout_count * 20.3 / (args.workers * 60 * 24), 2
            ),
            "recommended_planning_range_days": [
                round(
                    rollout_count * 20.3 / (args.workers * 60 * 24) * 1.1,
                    2,
                ),
                round(
                    rollout_count * 20.3 / (args.workers * 60 * 24) * 1.55,
                    2,
                ),
            ],
            "full_route_budget_projection_days": round(
                rollout_count
                * 20.3
                * (sum(task.max_steps for task in tasks) / len(tasks))
                / 40
                / (args.workers * 60 * 24),
                2,
            ),
            "note": (
                "The planning range includes model-latency and retry margin; "
                "actual task termination usually occurs far below route caps."
            ),
        },
        "run_requires": (
            "--mode run plus --confirm-codex-cli-run"
            if PROVIDER == "codex-cli"
            else (
                "--mode run plus --confirm-estimated-cost-usd at least equal "
                "to the full-route-budget scenario"
            )
        ),
        "per_model_commands": commands,
    }


def matrix_run(args: argparse.Namespace) -> int:
    plan = matrix_plan(args)
    if PROVIDER == "codex-cli" and not args.confirm_codex_cli_run:
        print(
            "Refusing to start: pass --confirm-codex-cli-run after reviewing "
            "the plan and available ChatGPT Codex credits.",
            file=sys.stderr,
        )
        return 2
    required = float(
        plan["cost_scenarios"]["full_route_budget"]["estimated_cost_usd"] or 0
    )
    if PROVIDER != "codex-cli" and (
        args.confirm_estimated_cost_usd is None
        or args.confirm_estimated_cost_usd < required
    ):
        print(
            "Refusing to start: pass --confirm-estimated-cost-usd "
            f"{required:.2f} (or higher) after reviewing the plan.",
            file=sys.stderr,
        )
        return 2
    for spec in matrix_specs():
        command = matrix_child_command(args, spec, "coordinator")
        completed_process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=matrix_environment(spec),
            check=False,
        )
        if completed_process.returncode != 0:
            return completed_process.returncode
    return 0
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("plan", "run", "coordinator", "worker", "summary"),
        default="plan",
        help="Default plan mode is read-only; run requires explicit cost confirmation.",
    )
    parser.add_argument(
        "--confirm-codex-cli-run",
        action="store_true",
        help="Confirm use of signed-in ChatGPT Codex credits for --mode run.",
    )
    parser.add_argument(
        "--suite-dir", type=Path,
        default=RESULTS_ROOT / (
            "codex_cli_sol_protocol_matrix"
            if PROVIDER == "codex-cli"
            else "openai_frontier_video_history_matrix"
        ),
    )
    parser.add_argument("--key-file", type=Path, default=KEY_FILE)
    parser.add_argument("--ue-launcher", type=Path, default=UE_LAUNCHER)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=7)
    parser.add_argument("--gpus", nargs="+", type=int, default=[6, 3, 2, 5, 1, 7, 0])
    parser.add_argument(
        "--ports", nargs="+", type=int,
        default=[9012, 9013, 9014, 9015, 9016, 9017, 9018],
    )
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument(
        "--condition-ids", nargs="+",
        help="Run only these exact condition IDs (resume-safe).",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument(
        "--unrealcv-request-timeout",
        type=float,
        default=120.0,
        help=(
            "Wall-clock timeout for an individual UE/UnrealCV command. This is "
            "independent of the model-provider timeout so a stuck renderer can "
            "fail closed without shortening a long reasoning call."
        ),
    )
    parser.add_argument("--cell-timeout", type=float, default=14400.0)
    parser.add_argument("--ue-ready-timeout", type=float, default=180.0)
    parser.add_argument("--ue-settle-seconds", type=float, default=30.0)
    parser.add_argument("--ue-start-stagger-seconds", type=float, default=30.0)
    parser.add_argument("--process-shutdown-timeout", type=float, default=20.0)
    parser.add_argument(
        "--ue-port-stride", type=int, default=0,
        help=(
            "Use base_port + generation * stride for each fresh UE process; "
            "useful when a packaged runtime leaves an unusable listener behind."
        ),
    )
    parser.add_argument("--status-interval", type=float, default=30.0)
    parser.add_argument("--dynamic-queue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-simulation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-png-compress-level", type=int, default=0)
    parser.add_argument("--record-images-first-steps", type=int, default=3)
    parser.add_argument("--record-images-last-steps", type=int, default=3)
    parser.add_argument(
        "--record-output-images",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--ue-no-rhi-thread", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--confirm-estimated-cost-usd",
        type=float,
        default=None,
        help="Required only for --mode run; must cover the full-budget scenario.",
    )
    args = parser.parse_args(argv)
    if PROVIDER == "codex-cli":
        if API_MODE != "codex_cli":
            parser.error("codex-cli campaigns require SIMWORLD_CAMPAIGN_API_MODE=codex_cli")
        if "none" in EFFORTS:
            parser.error("codex-cli campaigns do not support reasoning effort 'none'")
    if args.workers <= 0 or args.worker_count <= 0 or not 0 <= args.worker_id < args.worker_count:
        parser.error("invalid worker count/id")
    if not 0 <= args.record_png_compress_level <= 9:
        parser.error("--record-png-compress-level must be in [0, 9]")
    if args.record_images_first_steps < 0 or args.record_images_last_steps < 0:
        parser.error("image retention counts must be nonnegative")
    if args.ue_start_stagger_seconds < 0:
        parser.error("--ue-start-stagger-seconds must be nonnegative")
    if args.unrealcv_request_timeout <= 0:
        parser.error("--unrealcv-request-timeout must be positive")
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
        print(json.dumps(matrix_plan(args), indent=2))
        return 0
    if args.mode == "run":
        return matrix_run(args)
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
