#!/usr/bin/env python3
"""Supervise the full OpenAI frontier matrix with resume-safe workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "evaluation/run_openai_frontier_matrix.py"
MODEL_SPECS = (
    {
        "model": "gpt-6-astra", "efforts": "low,high", "expected": 72,
        "input_price": "10", "output_price": "50", "port_base": 12012,
    },
    {
        "model": "gpt-5.6-sol", "efforts": "none,low,high", "expected": 108,
        "input_price": "4", "output_price": "20", "port_base": 22012,
    },
    {
        "model": "gpt-5.6-terra", "efforts": "none,low,high", "expected": 108,
        "input_price": "2", "output_price": "12", "port_base": 32012,
    },
    {
        "model": "gpt-5.6-luna", "efforts": "none,low,high", "expected": 108,
        "input_price": "0.2", "output_price": "1.2", "port_base": 42012,
    },
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": now(), **event}, sort_keys=True) + "\n")


def live_worker_ids(suite: Path) -> set[int]:
    result: set[int] = set()
    suite_text = str(suite.resolve())
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            decoded = [
                item.decode(errors="replace")
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except OSError:
            continue
        if (
            "--mode" not in decoded
            or "worker" not in decoded
            or "--suite-dir" not in decoded
            or suite_text not in decoded
            or "--worker-id" not in decoded
        ):
            continue
        try:
            result.add(int(decoded[decoded.index("--worker-id") + 1]))
        except (ValueError, IndexError):
            continue
    return result


def worker_in_maintenance(suite: Path, worker_id: int) -> bool:
    """Return true while a boundary helper is cleaning up this worker."""
    return (
        suite / "workers" / f"worker_{worker_id:02d}" / ".maintenance.lock"
    ).exists()


def suite_states(suite: Path, expected: int) -> dict[str, int]:
    states: dict[str, int] = {}
    for path in (suite / "cells").glob("**/status.json"):
        try:
            state = str(json.loads(path.read_text()).get("state", "unknown"))
        except (OSError, json.JSONDecodeError):
            continue
        states[state] = states.get(state, 0) + 1
    states["unstarted"] = max(0, expected - sum(states.values()))
    return states


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def startup_only_failure(cell: Path) -> bool:
    """True only when every failed attempt ended before any rollout step/result."""
    attempts = sorted(cell.glob("attempt_*"))
    if not attempts:
        return False
    for attempt in attempts:
        validation = load_json(attempt / "validation.json")
        if not validation or validation.get("result_path") is not None:
            return False
        if any(attempt.rglob("step_*_manifest.json")):
            return False
    return True


def requeue_startup_only_failures(suite: Path, events: Path) -> int:
    """Archive empty startup failures so the normal three-attempt policy can rerun them."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    requeued = 0
    for status_path in sorted((suite / "cells").glob("**/status.json")):
        if load_json(status_path).get("state") != "needs_attention":
            continue
        cell = status_path.parent
        if not startup_only_failure(cell):
            continue
        relative = cell.relative_to(suite / "cells")
        archive = suite / "recovery_archives" / relative / stamp
        archive.mkdir(parents=True, exist_ok=True)
        moved: list[str] = []
        for attempt in sorted(cell.glob("attempt_*")):
            destination = archive / attempt.name
            shutil.move(str(attempt), str(destination))
            moved.append(attempt.name)
        shutil.move(str(status_path), str(archive / "status.json"))
        append_event(events, {
            "event": "startup_only_cell_requeued",
            "cell": str(relative),
            "archive": str(archive),
            "attempts": moved,
        })
        requeued += 1
    return requeued


def worker_environment(spec: dict) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "SIMWORLD_CAMPAIGN_MODEL": str(spec["model"]),
        "SIMWORLD_CAMPAIGN_PROVIDER": "openai",
        "SIMWORLD_CAMPAIGN_API_MODE": "responses",
        "SIMWORLD_CAMPAIGN_INPUT_PRICE": str(spec["input_price"]),
        "SIMWORLD_CAMPAIGN_OUTPUT_PRICE": str(spec["output_price"]),
        "SIMWORLD_CAMPAIGN_EFFORTS": str(spec["efforts"]),
    })
    return environment


def worker_command(
    args: argparse.Namespace, spec: dict, suite: Path, worker_id: int,
) -> list[str]:
    ports = [int(spec["port_base"]) + index for index in range(args.workers)]
    command = [
        sys.executable, str(RUNNER),
        "--mode", "worker",
        "--suite-dir", str(suite),
        "--worker-id", str(worker_id),
        "--worker-count", str(args.workers),
        "--gpus", *map(str, args.gpus),
        "--ports", *map(str, ports),
        "--key-file", str(args.key_file),
        "--ue-launcher", str(args.ue_launcher),
        "--max-attempts", "3",
        "--request-timeout", "300",
        "--cell-timeout", str(args.cell_timeout),
        "--ue-ready-timeout", "180",
        "--ue-settle-seconds", str(args.ue_settle_seconds),
        "--ue-start-stagger-seconds", str(args.ue_start_stagger_seconds),
        "--process-shutdown-timeout", "20",
        "--resume",
        "--record-png-compress-level", "0",
        "--record-images-first-steps", "0",
        "--record-images-last-steps", "0",
        "--record-output-images",
        "--dynamic-queue",
        "--ue-no-rhi-thread",
        "--ue-port-stride", str(args.port_stride),
    ]
    if spec.get("task_ids"):
        command.extend(["--task-ids", *map(str, spec["task_ids"])])
    return command


def selected_model_specs(selection_file: Path | None) -> tuple[dict, ...]:
    """Return the default matrix or an explicitly selected correction subset."""
    if selection_file is None:
        return MODEL_SPECS
    selection = load_json(selection_file)
    selected_models = selection.get("models")
    if not isinstance(selected_models, dict) or not selected_models:
        raise ValueError("selection file must contain a non-empty 'models' object")
    by_name = {str(spec["model"]): spec for spec in MODEL_SPECS}
    unknown = sorted(set(selected_models) - set(by_name))
    if unknown:
        raise ValueError(f"unknown models in selection file: {unknown}")
    result = []
    for model, model_selection in selected_models.items():
        if not isinstance(model_selection, dict):
            raise ValueError(f"selection for {model} must be an object")
        task_ids = sorted({int(value) for value in model_selection.get("task_ids", [])})
        efforts = tuple(str(value) for value in model_selection.get("efforts", []))
        allowed_efforts = tuple(str(by_name[model]["efforts"]).split(","))
        if not task_ids:
            raise ValueError(f"selection for {model} has no task IDs")
        if not efforts or not set(efforts).issubset(allowed_efforts):
            raise ValueError(
                f"selection for {model} has invalid efforts {efforts}; "
                f"allowed={allowed_efforts}"
            )
        spec = dict(by_name[model])
        spec["task_ids"] = task_ids
        spec["efforts"] = ",".join(efforts)
        spec["expected"] = len(task_ids) * len(efforts)
        result.append(spec)
    return tuple(result)


def supervise_suite(
    args: argparse.Namespace, spec: dict, events: Path, stream,
) -> bool:
    model = str(spec["model"])
    suite = args.campaign_root / model
    suite.mkdir(parents=True, exist_ok=True)
    expected = int(spec["expected"])
    if spec.get("task_ids"):
        efforts = str(spec["efforts"]).split(",")
        write_json(suite / "execution_profile.json", {
            "updated_at": now(),
            "workers": args.workers,
            "gpus": args.gpus[:args.workers],
            "dynamic_queue": True,
            "fast_simulation": True,
            "concurrent_realtime_inference": True,
            "record_png_compress_level": 0,
            "record_images_first_steps": 0,
            "record_images_last_steps": 0,
            "record_output_images": True,
            "selected_task_ids": list(spec["task_ids"]),
            "selected_condition_ids": [
                f"easy_realtime_instructional_{effort}"
                for effort in efforts
            ],
            "selected_rollouts": expected,
            "geometry_correction": True,
        })
    append_event(events, {
        "event": "suite_supervision_started", "model": model,
        "suite": str(suite), "expected": expected,
    })
    while True:
        states = suite_states(suite, expected)
        completed = states.get("completed", 0)
        live = live_worker_ids(suite)
        if completed >= expected:
            if live:
                time.sleep(args.interval)
                continue
            append_event(events, {
                "event": "suite_complete", "model": model,
                "completed": completed, "expected": expected,
            })
            return True

        unresolved = states.get("needs_attention", 0)
        unstarted = states.get("unstarted", 0)
        running = states.get("running", 0)
        if not live and unresolved and not unstarted and not running:
            requeued = requeue_startup_only_failures(suite, events)
            if requeued:
                append_event(events, {
                    "event": "startup_only_requeue_batch",
                    "model": model,
                    "count": requeued,
                })
                continue
            append_event(events, {
                "event": "suite_needs_attention", "model": model,
                "completed": completed, "expected": expected,
                "needs_attention": unresolved,
            })
            time.sleep(max(args.interval, 60.0))
            continue

        # During queue drain, keep only enough workers for cells that can still
        # make progress.  Otherwise an exited idle worker is immediately
        # respawned, scans the suite, and exits again when unstarted == 0.
        # Counting running cells preserves crash recovery: if a worker dies
        # while its cell is still marked running, a replacement is still
        # launched to resume it.
        target_workers = min(args.workers, running + unstarted)
        if len(live) >= target_workers:
            time.sleep(args.interval)
            continue

        for worker_id in range(args.workers):
            if len(live) >= target_workers:
                break
            if worker_id in live:
                continue
            if worker_in_maintenance(suite, worker_id):
                continue
            process = subprocess.Popen(
                worker_command(args, spec, suite, worker_id),
                cwd=REPO_ROOT,
                env=worker_environment(spec),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            append_event(events, {
                "event": "worker_started", "model": model,
                "worker_id": worker_id, "pid": process.pid,
                "completed": completed, "expected": expected,
                "states": states,
            })
            live.add(worker_id)
            time.sleep(args.interval)
        time.sleep(args.interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gpus", nargs="+", type=int, default=[7, 6, 5, 4, 2, 3])
    parser.add_argument("--port-stride", type=int, default=100)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--ue-start-stagger-seconds", type=float, default=30.0)
    parser.add_argument("--ue-settle-seconds", type=float, default=30.0)
    parser.add_argument("--cell-timeout", type=float, default=14400.0)
    parser.add_argument("--key-file", type=Path, default=Path(".secrets/openai_api.txt"))
    parser.add_argument(
        "--selection-file",
        type=Path,
        help=(
            "Optional JSON model/task/effort subset. When omitted, supervise "
            "the original complete frontier matrix."
        ),
    )
    parser.add_argument(
        "--ue-launcher", type=Path,
        default=Path(os.environ.get("SIMWORLD_UE_LAUNCHER", str(REPO_ROOT / "runtime/SimWorld.sh"))),
    )
    args = parser.parse_args()
    args.campaign_root = args.campaign_root.resolve()
    if len(args.gpus) < args.workers:
        raise ValueError("one GPU is required per worker")
    specs = selected_model_specs(args.selection_file)
    if args.selection_file is not None:
        write_json(args.campaign_root / "campaign_selection.json", {
            "created_at": now(),
            "source": str(args.selection_file.resolve()),
            "specs": list(specs),
            "selected_rollouts": sum(int(spec["expected"]) for spec in specs),
        })

    events = args.campaign_root / "campaign_watchdog.jsonl"
    output = args.campaign_root / "campaign_watchdog_workers.log"
    append_event(events, {"event": "campaign_watchdog_started", "pid": os.getpid()})
    with output.open("ab", buffering=0) as stream:
        for spec in specs:
            supervise_suite(args, spec, events, stream)
    append_event(events, {"event": "campaign_complete"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
