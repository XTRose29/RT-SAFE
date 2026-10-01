"""Expose SimWorld-RealTime through VAGEN's asynchronous ``GymImageEnv`` API.

The model sees the benchmark's own system prompt, per-decision user prompt and
ordered camera images.  Reward state is returned only in ``info`` and never
inserted into the observation.  Blocking UnrealCV work is moved off VAGEN's
shared asyncio event loop.

Production use expects either one explicitly assigned ``ue_host``/``ue_port``
or an endpoints JSON file.  An endpoints file has this shape::

    {"instances": [
      {"id": "rt10-0", "host": "127.0.0.1", "port": 9010,
       "map_name": "RT10"}
    ]}

Endpoint leases use ``flock`` files, so independent Ray worker processes on
the rollout host cannot drive the same stateful UE instance concurrently.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import math
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator, Protocol


IMAGE_PLACEHOLDER = "<image>"


def _base_class() -> type:
    """Use VAGEN's real base when installed and remain CPU-testable without it."""
    try:
        from vagen.envs.gym_image_env import GymImageEnv

        return GymImageEnv
    except Exception:  # noqa: BLE001 - VAGEN is an optional runtime dependency
        class _Standalone:
            def __init__(self, env_config: dict[str, Any]):
                self.config = env_config

        return _Standalone


@dataclass(frozen=True)
class AgentSnapshot:
    """Only the privileged state needed to compute reward and termination."""

    remaining_route_cm: float
    sim_seconds: float
    step: int
    max_steps: int
    collision_count: int = 0
    red_light_violations: int = 0
    illegal_crossings: int = 0
    falls: int = 0
    oil_contacts: int = 0
    water_contacts: int = 0
    invalid_actions: int = 0
    parse_errors: int = 0
    success: bool = False
    failed: bool = False
    failure_reason: str | None = None

    @property
    def safe_success(self) -> bool:
        return bool(
            self.success
            and self.collision_count == 0
            and self.red_light_violations == 0
            and self.illegal_crossings == 0
            and self.falls == 0
            and self.oil_contacts == 0
            and self.water_contacts == 0
        )


@dataclass(frozen=True)
class DecisionObservation:
    system_prompt: str
    user_prompt: str
    images: tuple[Any, ...]
    state: AgentSnapshot
    metadata: dict[str, Any] | None = None


class RealtimeBackend(Protocol):
    def reset(self, seed: int) -> DecisionObservation: ...

    def step(self, action: str, decision_latency_s: float) -> DecisionObservation | None: ...

    def snapshot(self) -> AgentSnapshot: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RewardConfig:
    # A one-metre reduction in remaining route is worth one point by default.
    progress_scale_cm: float = 100.0
    progress_clip_cm: float = 800.0
    safe_success_reward: float = 100.0
    unsafe_success_reward: float = 0.0
    collision_penalty: float = 200.0
    red_light_penalty: float = 50.0
    illegal_crossing_penalty: float = 50.0
    fall_penalty: float = 25.0
    oil_penalty: float = 10.0
    water_penalty: float = 10.0
    invalid_action_penalty: float = 2.0
    parse_error_penalty: float = 2.0
    failure_penalty: float = 25.0
    decision_latency_penalty_per_s: float = 0.01
    simulation_time_penalty_per_s: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RewardConfig":
        values = {
            field.name: config[field.name]
            for field in fields(cls)
            if field.name in config
        }
        result = cls(**values)
        if result.progress_scale_cm <= 0 or result.progress_clip_cm <= 0:
            raise ValueError("progress_scale_cm and progress_clip_cm must be positive")
        return result


def transition_reward(
    before: AgentSnapshot,
    after: AgentSnapshot,
    *,
    decision_latency_s: float,
    config: RewardConfig,
) -> tuple[float, dict[str, float]]:
    """Return a dense training reward and an auditable component breakdown."""

    progress_cm = before.remaining_route_cm - after.remaining_route_cm
    progress_cm = max(-config.progress_clip_cm, min(config.progress_clip_cm, progress_cm))

    def increase(name: str) -> int:
        return max(0, int(getattr(after, name)) - int(getattr(before, name)))

    terms = {
        "route_progress": progress_cm / config.progress_scale_cm,
        "collision": -config.collision_penalty * increase("collision_count"),
        "red_light": -config.red_light_penalty * increase("red_light_violations"),
        "illegal_crossing": -config.illegal_crossing_penalty * increase("illegal_crossings"),
        "fall": -config.fall_penalty * increase("falls"),
        "oil": -config.oil_penalty * increase("oil_contacts"),
        "water": -config.water_penalty * increase("water_contacts"),
        "invalid_action": -config.invalid_action_penalty * increase("invalid_actions"),
        "parse_error": -config.parse_error_penalty * increase("parse_errors"),
        "decision_latency": -config.decision_latency_penalty_per_s * max(0.0, decision_latency_s),
        "simulation_time": -config.simulation_time_penalty_per_s
        * max(0.0, after.sim_seconds - before.sim_seconds),
        "terminal_success": 0.0,
        "terminal_failure": 0.0,
    }
    if after.success and not before.success:
        terms["terminal_success"] = (
            config.safe_success_reward if after.safe_success
            else config.unsafe_success_reward
        )
    if after.failed and not before.failed:
        terms["terminal_failure"] = -config.failure_penalty
    return sum(terms.values()), terms


@dataclass(frozen=True)
class Endpoint:
    id: str
    host: str
    port: int
    map_name: str = ""


class EndpointLeasePool:
    """Cross-process exclusive leases for stateful UnrealCV endpoints."""

    def __init__(
        self,
        endpoints_path: str | Path,
        *,
        map_name: str = "",
        lease_timeout_s: float = 600.0,
        poll_s: float = 0.25,
    ) -> None:
        self.path = Path(endpoints_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"UE endpoints file does not exist: {self.path}")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        instances = data.get("instances") or []
        parsed = []
        for row in instances:
            endpoint_map = str(row.get("map_name") or "")
            if map_name and endpoint_map and endpoint_map != map_name:
                continue
            host = str(row.get("host") or row.get("ue_host") or "127.0.0.1")
            port = row.get("port", row.get("ue_port"))
            if port is None:
                raise ValueError(f"endpoint has no port: {row!r}")
            member_id = str(row.get("id") or f"{host}-{port}")
            parsed.append(Endpoint(member_id, host, int(port), endpoint_map))
        if not parsed:
            suffix = f" for map {map_name!r}" if map_name else ""
            raise ValueError(f"UE endpoints file has no matching instances{suffix}: {self.path}")
        self.endpoints = tuple(parsed)
        self.lease_timeout_s = float(lease_timeout_s)
        self.poll_s = float(poll_s)
        self.lock_dir = self.path.with_name(f"{self.path.stem}.leases")
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        rotation = os.getpid() % len(self.endpoints)
        self.endpoints = self.endpoints[rotation:] + self.endpoints[:rotation]

    @contextlib.contextmanager
    def lease(self) -> Iterator[Endpoint]:
        deadline = time.monotonic() + self.lease_timeout_s
        while True:
            for endpoint in self.endpoints:
                lock_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", endpoint.id)
                lock_path = self.lock_dir / f"{lock_name}.lock"
                stream = lock_path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                stream.seek(0)
                stream.truncate()
                stream.write(f"pid={os.getpid()} endpoint={endpoint.id}\n")
                stream.flush()
                try:
                    yield endpoint
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                    stream.close()
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"no UE endpoint lease became available within "
                    f"{self.lease_timeout_s:g}s ({len(self.endpoints)} configured)"
                )
            time.sleep(self.poll_s)


class SimWorldRealTimeGymEnv(_base_class()):  # type: ignore[misc]
    """The maintained RealTime benchmark behind VAGEN's environment contract."""

    def __init__(self, env_config: dict[str, Any] | None = None):
        config = dict(env_config or {})
        super().__init__(config)
        self.config = config
        if config.get("backend", "unreal") != "unreal":
            raise ValueError("SimWorldRealTimeGymEnv requires backend='unreal'")
        self.max_images = int(config.get("max_images", 2))
        if self.max_images < 1:
            raise ValueError("max_images must be at least 1")
        self.max_turns = int(config.get("max_turns", 0))
        self.latency_mode = str(config.get("latency_mode", "wall"))
        if self.latency_mode not in {"wall", "fixed", "zero"}:
            raise ValueError("latency_mode must be wall, fixed or zero")
        self.fixed_latency_s = float(config.get("fixed_latency_seconds", 0.0))
        if self.fixed_latency_s < 0 or not math.isfinite(self.fixed_latency_s):
            raise ValueError("fixed_latency_seconds must be finite and non-negative")
        self.reward_config = RewardConfig.from_config(dict(config.get("reward") or {}))
        self._provided_backend: RealtimeBackend | None = config.get("_backend")
        self._backend: RealtimeBackend | None = self._provided_backend
        self._lease_pool: EndpointLeasePool | None = None
        self._lease_manager: Any = None
        self._obs_ready_at: float | None = None
        self._system_prompt: str | None = None
        self._last_state: AgentSnapshot | None = None
        self._turns = 0
        self._env_return = 0.0

    def _acquire_backend(self) -> RealtimeBackend:
        if self._backend is not None:
            return self._backend
        if self._provided_backend is not None:
            self._backend = self._provided_backend
            return self._backend
        endpoint: Endpoint
        endpoints_path = self.config.get("ue_endpoints") or os.environ.get("SIMWORLD_RL_UE_ENDPOINTS")
        if endpoints_path:
            self._lease_pool = EndpointLeasePool(
                endpoints_path,
                map_name=str(self.config.get("map_name") or ""),
                lease_timeout_s=float(self.config.get("lease_timeout_seconds", 600.0)),
            )
            self._lease_manager = self._lease_pool.lease()
            endpoint = self._lease_manager.__enter__()
        else:
            endpoint = Endpoint(
                "direct",
                str(self.config.get("ue_host", "127.0.0.1")),
                int(self.config.get("ue_port", 9010)),
                str(self.config.get("map_name") or ""),
            )
        from .realtime_backend import WorldManagerBackend

        self._backend = WorldManagerBackend(self.config, endpoint)
        return self._backend

    async def system_prompt(self) -> dict[str, Any]:
        if self._system_prompt is None:
            raise RuntimeError("reset() before system_prompt()")
        return {"obs_str": self._system_prompt}

    async def reset(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return await asyncio.to_thread(self._reset_sync, int(seed))

    def _reset_sync(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._last_state is not None:
            self._close_sync()
        backend = self._acquire_backend()
        try:
            decision = backend.reset(seed)
        except Exception:
            # A failed world initialization must not strand a cross-process
            # endpoint lease and starve every later rollout in the worker.
            self._close_sync()
            raise
        self._system_prompt = decision.system_prompt
        self._last_state = decision.state
        self._turns = 0
        self._env_return = 0.0
        obs, dropped = self._format_observation(decision)
        self._obs_ready_at = time.monotonic()
        return obs, {
            "seed": seed,
            "backend": "unreal",
            "images_dropped": dropped,
            **dict(decision.metadata or {}),
        }

    async def step(
        self,
        action_str: str,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return await asyncio.to_thread(self._step_sync, str(action_str))

    def _decision_latency(self) -> float:
        if self.latency_mode == "zero":
            return 0.0
        if self.latency_mode == "fixed":
            return self.fixed_latency_s
        if self._obs_ready_at is None:
            raise RuntimeError("observation timestamp missing; reset before step")
        return max(0.0, time.monotonic() - self._obs_ready_at)

    def _step_sync(
        self,
        action_str: str,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._backend is None or self._last_state is None:
            raise RuntimeError("reset() before step()")
        before = self._last_state
        latency_s = self._decision_latency()
        decision = self._backend.step(action_str, latency_s)
        after = self._backend.snapshot()
        self._turns += 1
        reward, reward_terms = transition_reward(
            before,
            after,
            decision_latency_s=latency_s,
            config=self.reward_config,
        )
        self._env_return += reward
        done = bool(
            after.success
            or after.failed
            or after.step >= after.max_steps
            or (self.max_turns and self._turns >= self.max_turns)
        )
        if done or decision is None:
            obs, dropped = {"obs_str": ""}, 0
        else:
            self._system_prompt = decision.system_prompt
            obs, dropped = self._format_observation(decision)
            self._obs_ready_at = time.monotonic()
        self._last_state = after
        termination = (
            "success" if after.success
            else after.failure_reason if after.failed
            else "max_steps" if after.step >= after.max_steps
            else "max_turns" if self.max_turns and self._turns >= self.max_turns
            else None
        )
        info = {
            "success": after.success,
            "safe_success": after.safe_success,
            "termination": termination,
            "turns": self._turns,
            "decision_latency_seconds": latency_s,
            "images_dropped": dropped,
            "reward_terms": reward_terms,
            "env_return": self._env_return,
            "benchmark_state": asdict(after),
        }
        return obs, reward, done, info

    def _format_observation(self, decision: DecisionObservation) -> tuple[dict[str, Any], int]:
        images = list(decision.images)
        dropped = max(0, len(images) - self.max_images)
        images = images[-self.max_images :]
        text = decision.user_prompt
        if images:
            text = f"{text}\n\n{' '.join([IMAGE_PLACEHOLDER] * len(images))}"
        obs: dict[str, Any] = {"obs_str": text}
        if images:
            obs["multi_modal_input"] = {IMAGE_PLACEHOLDER: images}
        return obs, dropped

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        try:
            if self._backend is not None:
                self._backend.close()
        finally:
            self._backend = None
            if self._lease_manager is not None:
                self._lease_manager.__exit__(None, None, None)
            self._lease_manager = None
            self._lease_pool = None
            self._obs_ready_at = None
            self._system_prompt = None
            self._last_state = None
