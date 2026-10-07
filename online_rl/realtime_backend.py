"""Maintained WorldManager backend for :mod:`online_rl.vagen_env`.

This module is imported lazily so reward/adapter tests need neither UnrealCV
nor a running Unreal process.
"""

from __future__ import annotations

import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any

from .vagen_env import AgentSnapshot, DecisionObservation, Endpoint


class _PromptCaptured(RuntimeError):
    def __init__(self, system_prompt: str, user_prompt: str, images: list[Any]):
        super().__init__("prompt captured")
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.images = tuple(image for image in images if image is not None)


class _ExternalActionBridge:
    """Capture official prompts and parse externally generated VAGEN actions."""

    prompt_suffix = None

    def __init__(self) -> None:
        self.response: str | None = None
        self.latency_s = 0.0
        self.last_usage_metadata: dict[str, Any] = {}

    def queue(self, response: str, latency_s: float) -> None:
        self.response = response
        self.latency_s = max(0.0, float(latency_s))

    def generate_response_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        images=(),
        **_kwargs: Any,
    ):
        if self.response is None:
            raise _PromptCaptured(system_prompt, user_prompt, list(images))
        response = self.response
        self.response = None
        # The maintained benchmark deliberately loads ``simworld`` as a
        # namespace package so importing its action types does not execute the
        # optional desktop visualization stack. Keep this parser seam usable
        # in isolated Ray workers and focused smoke tests too.
        if "simworld" not in sys.modules:
            repo = Path(__file__).resolve().parents[1]
            simworld = types.ModuleType("simworld")
            simworld.__path__ = [str(repo / "SimWorld" / "simworld")]
            sys.modules["simworld"] = simworld
        from llm.rt_llm import RTLLM

        parser = object.__new__(RTLLM)
        action = RTLLM._parse_structured_response(parser, response)
        return action, self.latency_s, len(response), response, None, None, None


class WorldManagerBackend:
    """Drive one already-running UE endpoint one VAGEN action at a time."""

    def __init__(self, config: dict[str, Any], endpoint: Endpoint) -> None:
        self.config = dict(config)
        self.endpoint = endpoint
        self.manager: Any = None
        self.bridge = _ExternalActionBridge()
        self._scratch = tempfile.TemporaryDirectory(prefix="simworld-rt-vagen-")

    def reset(self, seed: int) -> DecisionObservation:
        self.close_world()
        from evaluation.run_qwen3vl8b_all_maps import prepare_runtime

        RTCommunicator, RTUnrealCV, StrictWorldManager, _metrics = prepare_runtime(
            float(self.config.get("request_timeout_seconds", 120.0)),
            int(self.config.get("reconnect_retries", 0)),
        )

        class StepWorldManager(StrictWorldManager):
            def run(self):
                # ``run_single_task`` still performs the maintained world and
                # agent setup, then stops at this seam for external actions.
                return None

        repo = Path(self.config.get("repo_root") or Path(__file__).resolve().parents[1])
        task_file = Path(self.config.get("task_file") or repo / "data/map1_10roads/tasks.json")
        if not task_file.is_absolute():
            task_file = repo / task_file
        agent_path = Path(self.config.get("agent_path") or repo / "data/agents.json")
        if not agent_path.is_absolute():
            agent_path = repo / agent_path
        task_indices = self.config.get("task_indices")
        if task_indices:
            task_index = int(task_indices[int(seed) % len(task_indices)])
        else:
            task_index = int(self.config.get("task_index", 0))

        communicator = RTCommunicator(
            RTUnrealCV(
                port=self.endpoint.port,
                ip=self.endpoint.host,
                resolution=(
                    int(self.config.get("observation_width", 720)),
                    int(self.config.get("observation_height", 640)),
                ),
            )
        )
        self.manager = StepWorldManager(
            communicator=communicator,
            agent_path=str(agent_path),
            task_file_path=str(task_file),
            seed=int(seed),
            control_mode="llm",
            token_based=False,
            use_tick=True,
            realtime_thinking=bool(self.config.get("realtime_thinking", True)),
            record_per_step=bool(self.config.get("record_per_step", False)),
            use_action_frames=bool(self.config.get("use_action_frames", False)),
            prompt_style=str(self.config.get("prompt_style", "naive")),
            traffic_policy=str(self.config.get("traffic_policy", "visual_only")),
            max_steps=int(self.config.get("max_steps", -1)),
            results_dir=str(Path(self._scratch.name) / f"seed-{seed}"),
            pedestrians_enabled=bool(self.config.get("pedestrians", True)),
            movable_obstacles_enabled=bool(self.config.get("movable_obstacles", True)),
            irregular_npcs_enabled=bool(self.config.get("irregular_npcs", True)),
            falling_objects_enabled=bool(self.config.get("falling_objects", True)),
            load_all_unsafe_triggers=bool(self.config.get("load_all_unsafe_triggers", False)),
        )
        self.manager.run_single_task(
            task_index,
            difficulty=str(self.config.get("difficulty", "easy")),
        )
        self.manager.agent.llm = self.bridge
        # VAGEN performs generation before calling step().  The elapsed time is
        # supplied by the bridge and advanced deterministically inside RTAgent.
        self.manager.agent.concurrent_realtime_inference = False
        # RTAgent normally performs these first-frame operations at the start
        # of step(). VAGEN needs that same first observation before it can
        # generate the first action, so perform the idempotent setup here.
        agent = self.manager.agent
        agent.start_time = time.time()
        agent.camera_resolution = getattr(agent, "camera_resolution", (720, 640))
        agent.communicator.unrealcv.set_camera_resolution(
            agent.camera_id, agent.camera_resolution
        )
        agent.communicator.unrealcv.set_camera_fov(agent.camera_id, agent.fov)
        agent.sync_ue()
        agent._remember_collision_free_state("initial_vagen_observation")
        return self._capture_decision()

    def _capture_decision(self) -> DecisionObservation:
        agent = self.manager.agent
        observation = agent.get_observation()
        decision_count = agent.decision_count
        safe_history_len = len(agent.safe_trajectory_history)
        try:
            agent.plan(observation, "llm")
        except _PromptCaptured as captured:
            return DecisionObservation(
                captured.system_prompt,
                captured.user_prompt,
                captured.images,
                self.snapshot(),
                {
                    "endpoint_id": self.endpoint.id,
                    "ue_host": self.endpoint.host,
                    "ue_port": self.endpoint.port,
                    "map_name": self.endpoint.map_name,
                    "task_index": int(self.manager.current_task_index),
                    "task_id": self.manager.current_task_id,
                    "difficulty": self.manager.difficulty,
                },
            )
        finally:
            # Prompt capture is observation, not a policy decision.
            agent.decision_count = decision_count
            del agent.safe_trajectory_history[safe_history_len:]
        raise RuntimeError("external action bridge did not capture a prompt")

    def step(self, action: str, decision_latency_s: float) -> DecisionObservation | None:
        if self.manager is None:
            raise RuntimeError("reset before step")
        self.bridge.queue(action, decision_latency_s)
        self.manager.agent.step(mode="llm")
        state = self.snapshot()
        if state.success or state.failed or state.step >= state.max_steps:
            return None
        return self._capture_decision()

    def snapshot(self) -> AgentSnapshot:
        if self.manager is None or self.manager.agent is None:
            raise RuntimeError("reset before snapshot")
        agent = self.manager.agent
        route = [agent.position, *(agent.shortest_path or [])]
        remaining = sum(a.distance(b) for a, b in zip(route, route[1:]))
        return AgentSnapshot(
            remaining_route_cm=float(remaining),
            sim_seconds=float(agent.sim_time_elapsed),
            step=int(agent.step_num),
            max_steps=int(self.manager.max_steps),
            collision_count=int(agent.collision_count),
            red_light_violations=int(agent.red_light_violations_count),
            illegal_crossings=int(agent.illegal_crossing_violations_count),
            falls=int(agent.fall_count),
            oil_contacts=int(agent.oil_count),
            water_contacts=int(agent.water_count),
            invalid_actions=int(agent.invalid_decision_count),
            parse_errors=int(agent.parse_error_count),
            success=bool(agent.success),
            failed=bool(agent.failed),
            failure_reason=agent.failure_reason,
        )

    def close_world(self) -> None:
        if self.manager is None:
            return
        try:
            self.manager.cleanup()
        finally:
            self.manager = None

    def close(self) -> None:
        self.close_world()

    def __del__(self) -> None:
        try:
            self.close_world()
            self._scratch.cleanup()
        except Exception:
            pass
