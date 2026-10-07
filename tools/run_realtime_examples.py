#!/usr/bin/env python3
"""Run and compare a short static/realtime pair without model credentials.

Each condition gets a fresh Unreal process. Checks compare stable protocol
fields; rendered pixels and collision/movement trajectories are not expected
to match bit for bit across GPU load or physics scheduling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "examples/realtime/expected.json"


def compare(result: dict, expected: dict) -> dict:
    mismatches = {}
    for key, value in expected.items():
        actual = result.get(key)
        # bool and integer are different protocol values, even in Python.
        if actual != value or isinstance(actual, bool) != isinstance(value, bool):
            mismatches[key] = {"expected": value, "actual": actual}
    return mismatches


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run_case(args, mode, repetition, expected):
    folder = args.output / f"repeat-{repetition:02d}-{mode}"
    folder.mkdir()
    with socket.socket() as port:
        if port.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"Port {args.port} is occupied")
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join([str(ROOT / "SimWorld"), str(ROOT)]),
        SIMWORLD_TIME_ADVANCE_MODE="resume_pause",
        SIMWORLD_OBSERVATION_WIDTH="720", SIMWORLD_OBSERVATION_HEIGHT="640",
        SIMWORLD_SKIP_INITIAL_SETRES="1", SIMWORLD_SKIP_ASYNC_SKINNED_ASSET_COMPILATION="1",
    )
    launch = [str(args.launcher), "/Game/RealTimeBench/Maps/RT10", "-RenderOffScreen",
              "-windowed", f"-graphicsadapter={args.gpu}", "-cvport", str(args.port),
              "-ResX=720", "-ResY=640", "-FPSMAX=30", "-noraytracing",
              f"-abslog={folder / 'engine.log'}"]
    command = [sys.executable, str(ROOT / "evaluation/run_code_baselines.py"),
               "--task-index", "0", "--seed", "0", "--baselines", "greedy",
               "--env-modes", mode, "--time-modes", "sync", "--max-steps", "2",
               "--internal-latency-seconds", "1", "--max-rounds", "1",
               "--ue-port", str(args.port), "--output-root", str(folder),
               "--suite-name", "benchmark", "--no-record-per-step"]
    with (folder / "console.log").open("w") as console:
        engine = subprocess.Popen(launch, cwd=args.launcher.parent, env=environment,
                                  stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        runner = None
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if engine.poll() is not None:
                    raise RuntimeError(f"Unreal exited with {engine.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Unreal startup timed out")
                log = (folder / "console.log").read_text(errors="replace")
                if all(text in log for text in (
                    "LoadMap(/Game/RealTimeBench/Maps/RT10)",
                    "Engine is initialized. Leaving FEngineLoop::Init()",
                    f"Start listening on {args.port}",
                )):
                    break
                time.sleep(1)
            with (folder / "runner.log").open("w") as output:
                runner = subprocess.Popen(command, cwd=ROOT, env=environment,
                                          stdout=output, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                runner.wait(timeout=args.run_timeout)
            if runner.returncode:
                raise RuntimeError(f"Baseline exited with {runner.returncode}; see runner.log")
            records = [json.loads(line) for line in
                       (folder / "benchmark/summary.jsonl").read_text().splitlines()]
            if len(records) != 1 or records[0].get("status") != "completed":
                raise RuntimeError("Expected exactly one completed condition")
            result_path = Path(records[0]["result_path"])
            if not result_path.is_absolute():
                result_path = ROOT / result_path
            result = json.loads(result_path.read_text())
            mismatches = compare(result, expected)
            return {
                "mode": mode, "repetition": repetition, "passed": not mismatches,
                "observed": {key: result.get(key) for key in expected},
                "observations_not_compared": {key: result.get(key) for key in (
                    "oil_count", "total_collisions", "collision_count", "wall_time", "final_position")},
                "mismatches": mismatches,
                "result": str(result_path.relative_to(args.output)),
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            }
        finally:
            if runner is not None:
                stop(runner)
            stop(engine)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/realtime-examples"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=19091)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--run-timeout", type=float, default=600)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.launcher = args.launcher.resolve()
    if args.output.exists() or not args.launcher.is_file() or args.repeats < 1:
        parser.error("Use a new output directory, an existing launcher, and positive repeats")
    expected = json.loads(REFERENCE.read_text())
    args.output.mkdir(parents=True)
    report = {"scope": expected["scope"], "passed": False, "runs": []}
    for repetition in range(1, args.repeats + 1):
        for mode in ("static", "realtime"):
            try:
                run = run_case(args, mode, repetition, expected["conditions"][mode])
            except Exception as exc:
                run = {"mode": mode, "repetition": repetition, "passed": False, "error": str(exc)}
            report["runs"].append(run)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(run), flush=True)
    report["passed"] = all(run["passed"] for run in report["runs"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
