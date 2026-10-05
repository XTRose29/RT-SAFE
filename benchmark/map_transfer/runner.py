"""Episode lifecycle, observations, and audit logs for the native-map pilot."""

from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from datetime import datetime, timezone

from .client import NativeClient
from .contract import Action
from .observation import annotate_waypoints


class NativeEpisode:
    def __init__(
        self,
        client: NativeClient,
        manifest: dict,
        task_id: str,
        output: Path,
        *,
        mode: str = "realtime",
        seed: int = 0,
        reload_source: bool = False,
    ):
        self.client = client
        self.manifest = manifest
        self.task_id = task_id
        self.output = Path(output)
        self.mode = mode
        self.seed = seed
        self.reload_source = reload_source
        self.trace = []
        self.started = False
        if mode not in ("static", "realtime"):
            raise ValueError("Unknown evaluation mode")
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError(
                "Choose an empty output directory; existing run evidence is never overwritten"
            )
        self.output.mkdir(parents=True, exist_ok=True)
        self.task = next((t for t in manifest["tasks"] if t["id"] == task_id), None)
        if self.task is None:
            raise ValueError(f"Unknown task: {task_id}")
        (self.output / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    def setup(self) -> dict:
        c = self.client
        c.load_runtime(Path(__file__).resolve().parents[2])
        if self.reload_source:
            c.execute(
                "import importlib, sys\n"
                "assert rt_transfer._state is None, 'Active episode must finish before reload'\n"
                "for name in ('contract','manifest','hazards','fixtures','dynamics'):\n"
                "    module=sys.modules.get('benchmark.map_transfer.'+name)\n"
                "    if module is not None: importlib.reload(module)\n"
                "rt_transfer=importlib.reload(rt_transfer)\nresult={'reloaded':True}"
            )
        c.call("cleanup_scene")
        time.sleep(0.2)
        prepared = c.call(
            "prepare", manifest=self.manifest, task_id=self.task_id, seed=self.seed
        )
        (self.output / "preparation.json").write_text(json.dumps(prepared, indent=2))
        c.call("begin_play")
        deadline = time.monotonic() + 60
        while not c.call("game_ready"):
            if time.monotonic() > deadline:
                raise TimeoutError("PIE did not become ready")
            time.sleep(0.2)
        time.sleep(0.5)  # allow gravity to settle the capsule onto native ground
        c.call("attach", mode=self.mode)
        c.game = None
        c.pool = None
        # First RGB captures populate runtime virtual-texture pages, including
        # road markings. This happens before episode timing/pedestrian motion.
        request = c.call("camera_request")
        for _ in range(4):
            c.capture(request)
            time.sleep(0.1)
        state = c.call("start_episode")
        self.started = True
        source_root = Path(__file__).parent
        source_hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_root.glob("*.py"))
        }
        provenance = {
            "schema": "rtsafe-native-pilot-run-v1",
            "pilot": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_id": self.task_id,
            "mode": self.mode,
            "seed": self.seed,
            "layout_randomization": "none; fixed authored fixtures, seed retained as run metadata",
            "map_path": self.manifest["map_path"],
            "manifest_sha256": hashlib.sha256(
                json.dumps(self.manifest, sort_keys=True).encode()
            ).hexdigest(),
            "editor_pid": c.pid,
            "initial_state": state,
            "adapter_source_sha256": source_hashes,
            "runtime": prepared.get("runtime", {}),
            "runtime_fingerprint": c.fingerprint,
            "protocol_differences": [
                "Native CharacterMovement capsules and collision-entry counting",
                "Continuous hazard overlap detection; trip recovery delay, next-move oil slowdown, and unperturbed water destination",
                "Kinematic vehicle fixtures use native collision sweeps; no Chaos drivetrain or ambient MassTraffic evaluation",
                "Robot dogs use swept body colliders with visual link animation; movable crates use native rigid-body physics and explicit character push settings",
                "Map-specific annotated traffic zones and explicit pedestrian signals",
                "No original-paper SPL comparison: shortest navigable paths are not certified",
            ],
        }
        (self.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
        return state

    def __enter__(self) -> "NativeEpisode":
        try:
            self.setup()
        except BaseException as error:
            self._write_failure(error, "setup")
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"Episode cleanup also failed: {cleanup_error}")
            raise
        return self

    def __exit__(self, kind, error, traceback) -> None:
        try:
            if error is not None:
                self._write_failure(error, "episode")
            if kind and self.started:
                self.finish("host_error")
        except Exception as finish_error:
            if error is None:
                raise
            error.add_note(f"Writing final state also failed: {finish_error}")
        finally:
            try:
                self.close()
            except Exception as cleanup_error:
                if error is None:
                    raise
                error.add_note(f"Episode cleanup also failed: {cleanup_error}")

    def _write_failure(self, error, stage):
        try:
            (self.output / "failure.json").write_text(
                json.dumps(
                    {
                        "stage": stage,
                        "type": type(error).__name__,
                        "message": str(error),
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
            )
        except Exception as recording_error:
            error.add_note(f"Writing failure evidence also failed: {recording_error}")

    def observe(self, index: int) -> tuple[bytes, dict]:
        before = self.client.call("status")
        request = self.client.call("camera_request")
        waypoints = self.client.call("waypoints")
        started = time.monotonic()
        data = self.client.capture(request)
        image = data["images"][0]
        urls = [
            v
            for v in image.values()
            if isinstance(v, str) and v.startswith("data:image/")
        ]
        if len(urls) != 1:
            raise RuntimeError("Expected one native RGB data URL")
        jpeg = base64.b64decode(urls[0].split(",", 1)[1])
        annotated = annotate_waypoints(jpeg, request, waypoints)
        (self.output / f"observation-{index:03d}-raw.jpg").write_bytes(jpeg)
        (self.output / f"observation-{index:03d}.png").write_bytes(annotated)
        after = self.client.call("status")
        metadata = {
            "camera": request,
            "waypoints_cm": waypoints,
            "state_before": before,
            "state_after": after,
            "capture_wall_seconds": time.monotonic() - started,
            "rgb_sha256": hashlib.sha256(jpeg).hexdigest(),
            "policy_image_sha256": hashlib.sha256(annotated).hexdigest(),
        }
        (self.output / f"observation-{index:03d}.json").write_text(
            json.dumps(metadata, indent=2)
        )
        return annotated, after

    def step(self, action: Action, *, policy_metadata: dict | None = None) -> dict:
        before = self.client.call("status")
        start = time.monotonic()
        self.client.call("act", action=action.as_dict())
        deadline = start + 30
        while True:
            state = self.client.call("status")
            if state["errors"]:
                raise RuntimeError(state["errors"][-1])
            if state["phase"] == "terminal" or state["command"] is None:
                break
            if time.monotonic() > deadline:
                self.client.call("terminate", reason="host_action_timeout")
                raise TimeoutError("Native action did not finish within 30 seconds")
            time.sleep(0.05)
        record = {
            "index": len(self.trace),
            "action": action.as_dict(),
            "before": before,
            "after": state,
            "action_wall_seconds": time.monotonic() - start,
            "policy": policy_metadata or {"kind": "scripted"},
        }
        self.trace.append(record)
        with (self.output / "decisions.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return state

    def wait_inference(self, seconds: float) -> dict:
        if not 0 <= seconds <= 60:
            raise ValueError("Smoke-test delay must be between 0 and 60 seconds")
        before = self.client.call("status")
        time.sleep(seconds)
        after = self.client.call("status")
        evidence = {"requested_delay_s": seconds, "before": before, "after": after}
        (
            self.output
            / f'inference-delay-{len(list(self.output.glob("inference-delay-*.json"))):03d}.json'
        ).write_text(json.dumps(evidence, indent=2))
        return evidence

    def finish(self, reason: str = "decision_limit") -> dict:
        state = self.client.call("status")
        if state["phase"] != "terminal":
            state = self.client.call("terminate", reason=reason)
        counts = state["counts"]
        passive = sum(
            e["type"] == "collision" and e["phase"] == "inference"
            for e in state["events"]
        )
        summary = {
            "schema": "rtsafe-native-pilot-summary-v1",
            "pilot": True,
            "task_id": self.task_id,
            "mode": self.mode,
            "success": state["goal_reached"],
            "safe_success": state["safe_success"],
            "safety_events": counts,
            "active_collisions": counts["collision"] - passive,
            "passive_collisions": passive,
            "decisions": len(self.trace),
            "simulation_time_s": state["simulation_time"],
            "wall_time_s": state["episode_wall_time_s"],
            "dynamic_activations": state["activations"],
            "traveled_m": state["traveled_cm"] / 100,
            "spl": None,
            "terminal_reason": state["terminal_reason"],
            "runtime_errors": state["errors"],
        }
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2))
        (self.output / "final-state.json").write_text(json.dumps(state, indent=2))
        trajectory = self.client.call("trajectory")
        (self.output / "trajectory.json").write_text(json.dumps(trajectory))
        return summary

    def close(self) -> None:
        self.client.call("end_play")
        deadline = time.monotonic() + 15
        while self.client.call("is_playing"):
            if time.monotonic() >= deadline:
                raise TimeoutError("PIE did not stop within 15 seconds")
            time.sleep(0.1)
        self.client.game = None
        self.client.pool = None
        self.started = False
