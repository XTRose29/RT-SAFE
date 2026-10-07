#!/usr/bin/env python3
"""Live, model-free smoke check of an isolated legacy RT-SAFE Linux runtime.

Launches and cleans up only its own Unreal process. A successful run verifies
rendering, required Blueprint classes, state APIs and static/realtime movement;
it is not a reproduction of the benchmark's model scores.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import logging
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from threading import Lock
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "SimWorld")]
# Importing OpenCV can modify the loader environment. Keep the engine's launch
# environment independent of Python's optional imaging-library paths.
ENGINE_ENV = os.environ.copy()


def checked_request(ue, command):
    response = ue.client.request(command)
    if response is None or (isinstance(response, str) and response.strip().lower().startswith("error")):
        raise RuntimeError(f"{command}: {response}")
    return response


def position(ue, name):
    values = [float(x) for x in checked_request(ue, f"vget /object/{name}/location").split()]
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise RuntimeError("Invalid actor position")
    return values


def probe(ue, level, output):
    import numpy as np
    from PIL import Image

    request = lambda command: checked_request(ue, command)
    result = {"level": level, "checks": {}}
    request("vset /action/game/pause")
    agent = "RTSAFE_ADDON_CHECK_AGENT"
    paths = [
        "/Game/TrafficSystem/UE_Manager.UE_Manager_C",
        "/Game/TrafficSystem/Pedestrian/Base_Pedestrian.Base_Pedestrian_C",
        *[f"/Game/RealTimeBench/Traffic/{name}.{name}_C" for name in (
            "RT_BP_Intersection", "RT_BP_street_light", "RT_BP_street_light_ped",
        )],
    ]
    assets = json.loads((ROOT / "data/ue_assets.json").read_text())
    paths += [a["asset_path"] for a in assets.values()
              if isinstance(a, dict) and a.get("asset_path", "").startswith("/Game/RealTimeBench/")]
    spawned = []
    for index, path in enumerate(dict.fromkeys(paths)):
        name = f"RTSAFE_ADDON_CHECK_{index}"
        request(f"vset /objects/spawn_bp_asset {path} {name}")
        # Spawn RPC success alone is insufficient: confirm an addressable actor.
        position(ue, name)
        if "street_light" in path:
            state = json.loads(request(f"vbp {name} GetState"))
            if not {"green light", "ped green", "ped time"}.issubset(state):
                raise RuntimeError(f"Invalid signal state: {state}")
        spawned.append(path)
        request(f"vset /object/{name}/destroy")
    result["checks"]["spawned_classes"] = spawned
    # Spawn the moving agent only after capability-test hazards are gone.
    # Otherwise their construction-time overlaps can apply a six-second
    # debuff to the agent at the default spawn origin, invalidating timing tests.
    request(f"vset /objects/spawn_bp_asset /Game/RealTimeBench/Agent/BP_RT_Agent.BP_RT_Agent_C {agent}")
    # Use the first authored route's sidewalk, facing the next route point.
    map_dirs = {"RT10": "map1_10roads", "RT12": "map2_12roads", "RT15": "map3_15roads",
                "RT18": "map4_18roads", "RT20": "map5_20roads"}
    task = json.loads((ROOT / "data" / map_dirs[level] / "tasks.json").read_text())["tasks"][0]
    x, y = task["start_point"]
    request(f"vset /object/{agent}/location {x} {y} 110")
    request(f"vset /object/{agent}/collision true")
    request(f"vbp {agent} SetMaxSpeed 200")
    request("vset /action/game/resume")
    time.sleep(1)
    request("vset /action/game/pause")
    start = position(ue, agent)
    target = task["start_edge"]["node2"]
    dx, dy = target[0] - x, target[1] - y
    norm = math.hypot(dx, dy)
    if norm < 1:
        target = task["start_edge"]["node1"]
        dx, dy = target[0] - x, target[1] - y
        norm = math.hypot(dx, dy)
    tx, ty = x + dx / norm * 600, y + dy / norm * 600
    request(f"vbp {agent} MoveTo {tx},{ty} 3")
    time.sleep(1)
    frozen = position(ue, agent)
    drift = math.dist(start, frozen)
    if drift > 1:
        raise RuntimeError(f"Static world moved {drift:.3f} cm while paused")
    ue.advance_simulation_time(1.5)
    stepped = position(ue, agent)
    tick_displacement = math.dist(frozen[:2], stepped[:2])
    if tick_displacement < 1:
        raise RuntimeError("Configured time-advance mode did not advance actor movement")
    time.sleep(0.5)
    if math.dist(stepped, position(ue, agent)) > 1:
        raise RuntimeError("Time advance did not return to a paused world")
    tx, ty = stepped[0] + dx / norm * 600, stepped[1] + dy / norm * 600
    request(f"vbp {agent} MoveTo {tx},{ty} 3")
    request("vset /action/game/resume")
    began = time.monotonic()
    time.sleep(1.5)
    request("vset /action/game/pause")
    elapsed = time.monotonic() - began
    moved = position(ue, agent)
    displacement = math.dist(stepped[:2], moved[:2])
    if displacement < 30:
        raise RuntimeError(f"Realtime movement did not advance: {displacement:.3f} cm; {start} -> {moved}")
    state = json.loads(request(f"vbp {agent} GetStates"))
    required = {"HumanCollision", "ObjectCollision", "BuildingCollision", "Intensity", "OverlapType", "TouchedRoad"}
    if not required.issubset(state):
        raise RuntimeError(f"Missing agent state fields: {state}")
    result["checks"].update(static_drift_cm=drift, realtime_displacement_cm=displacement,
                            step_displacement_cm=tick_displacement,
                            time_advance_mode=ue._time_advance_mode(),
                            realtime_wall_seconds=elapsed, agent_state=state,
                            initial_position=start, final_position=moved)
    camera_id = len(str(request("vget /cameras")).split())
    request("vset /cameras/spawn")
    request(f"vset /camera/{camera_id}/location {moved[0] - 400} {moved[1] - 400} {moved[2] + 500}")
    request(f"vset /camera/{camera_id}/rotation -35 45 0")
    frame = Image.open(BytesIO(request(f"vget /camera/{camera_id}/lit png"))).convert("RGB")
    pixels = np.asarray(frame)
    if min(frame.size) < 100 or float(pixels.std()) < 2:
        raise RuntimeError("Camera returned a blank or invalid image")
    frame.save(output / f"{level}.png")
    result["checks"]["render"] = {"size": list(frame.size), "pixel_std": float(pixels.std())}
    result["passed"] = True
    return result


def run_level(args, level):
    import unrealcv
    from simworld.communicator.unrealcv import UnrealCV

    with socket.socket() as check:
        if check.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"Port {args.port} is occupied; refusing to use an existing runtime")
    console = args.output / f"{level}.console.log"
    internal = args.output / f"{level}.engine.log"
    command = [str(args.launcher), f"/Game/RealTimeBench/Maps/{level}", "-RenderOffScreen",
               "-windowed", f"-graphicsadapter={args.gpu}", "-cvport", str(args.port),
               "-ResX=640", "-ResY=480", "-FPSMAX=30", "-noraytracing", f"-abslog={internal}"]
    ue = None
    with console.open("w") as log:
        process = subprocess.Popen(command, cwd=args.launcher.parent, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True, env=ENGINE_ENV)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Unreal exited with {process.returncode}; see {console.name}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Unreal startup timed out; see {console.name}")
                # The listener AND LoadMap can precede completion of engine
                # initialization by tens of seconds. Wait for the engine loop,
                # and avoid a throwaway TCP connection to the legacy server.
                startup_log = console.read_text(errors="replace")
                loaded = f"LoadMap(/Game/RealTimeBench/Maps/{level})" in startup_log
                initialized = "Engine is initialized. Leaving FEngineLoop::Init()" in startup_log
                if loaded and initialized and f"Start listening on {args.port}" in startup_log:
                    break
                time.sleep(1)
            ue = UnrealCV.__new__(UnrealCV)
            ue.client = unrealcv.Client(("127.0.0.1", args.port))
            ue.logger = logging.getLogger("runtime-addon-check")
            ue.lock = Lock()
            ue.request_timeout_seconds = args.request_timeout
            ue.request_reconnect_retries = 0
            ue._install_bounded_requests()
            if not ue.client.connect():
                raise ConnectionError("UnrealCV connection failed")
            time.sleep(2)
            result = probe(ue, level, args.output)
            text = console.read_text(errors="replace")
            if f"Bringing World /Game/RealTimeBench/Maps/{level}.{level} up for play" not in text or "Failed to load package" in text:
                raise RuntimeError("Expected map was not loaded cleanly")
            if any(s in text.lower() for s in ("llvmpipe", "lavapipe", "falling back to first device")):
                raise RuntimeError("Unreal selected an unexpected/software GPU")
            return result
        finally:
            if ue is not None:
                ue.client.disconnect()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=19091)
    parser.add_argument("--levels", nargs="+", choices=["RT10", "RT12", "RT15", "RT18", "RT20"],
                        default=["RT10", "RT12", "RT15", "RT18", "RT20"])
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument("--time-advance-mode", choices=["resume_pause", "legacy_tick"], default="resume_pause")
    args = parser.parse_args()
    args.launcher = args.launcher.resolve(strict=True)
    os.environ["SIMWORLD_TIME_ADVANCE_MODE"] = args.time_advance_mode
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {"scope": "live add-on smoke; no model calls or score reproduction",
               "requested_levels": args.levels, "levels": [], "passed": False}
    for level in args.levels:
        print(f"Checking {level}", flush=True)
        try:
            item = run_level(args, level)
        except Exception as exc:
            item = {"level": level, "passed": False, "error": str(exc)}
        results["levels"].append(item)
        results["passed"] = len(results["levels"]) == len(args.levels) and all(x["passed"] for x in results["levels"])
        (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(item), flush=True)
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
