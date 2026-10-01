import asyncio
import json

import pytest

from online_rl.vagen_env import (
    AgentSnapshot,
    DecisionObservation,
    EndpointLeasePool,
    RewardConfig,
    SimWorldRealTimeGymEnv,
    transition_reward,
)
from online_rl.realtime_backend import _ExternalActionBridge, _PromptCaptured


class FakeBackend:
    def __init__(self):
        self.state = AgentSnapshot(remaining_route_cm=1000, sim_seconds=0, step=0, max_steps=3)
        self.latencies = []
        self.closed = False

    def reset(self, seed):
        self.state = AgentSnapshot(remaining_route_cm=1000, sim_seconds=0, step=0, max_steps=3)
        return DecisionObservation(
            "official system",
            f"seed {seed} prompt",
            ("old-image", "new-image", "extra-image"),
            self.state,
            {"task_id": "fake"},
        )

    def step(self, action, decision_latency_s):
        self.latencies.append(decision_latency_s)
        success = action == "Action: move_to\nParam: 1"
        self.state = AgentSnapshot(
            remaining_route_cm=800,
            sim_seconds=decision_latency_s + 2,
            step=1,
            max_steps=3,
            success=success,
        )
        if success:
            return None
        return DecisionObservation("official system", "next prompt", ("next",), self.state)

    def snapshot(self):
        return self.state

    def close(self):
        self.closed = True


def test_transition_reward_reports_every_component():
    before = AgentSnapshot(remaining_route_cm=1000, sim_seconds=3, step=0, max_steps=4)
    after = AgentSnapshot(
        remaining_route_cm=700,
        sim_seconds=8,
        step=1,
        max_steps=4,
        collision_count=1,
        red_light_violations=1,
        oil_contacts=1,
        failed=True,
        failure_reason="vehicle_collision",
    )
    reward, terms = transition_reward(
        before,
        after,
        decision_latency_s=2,
        config=RewardConfig(),
    )
    assert terms["route_progress"] == 3
    assert terms["collision"] == -200
    assert terms["red_light"] == -50
    assert terms["oil"] == -10
    assert terms["decision_latency"] == pytest.approx(-0.02)
    assert terms["terminal_failure"] == -25
    assert reward == pytest.approx(sum(terms.values()))


def test_vagen_contract_keeps_reward_state_out_of_observation():
    backend = FakeBackend()
    env = SimWorldRealTimeGymEnv(
        {
            "_backend": backend,
            "latency_mode": "fixed",
            "fixed_latency_seconds": 1.5,
            "max_images": 2,
        }
    )

    async def exercise():
        obs, reset_info = await env.reset(7)
        system = await env.system_prompt()
        next_obs, reward, done, info = await env.step("Action: move_to\nParam: 1")
        await env.close()
        return obs, reset_info, system, next_obs, reward, done, info

    obs, reset_info, system, next_obs, reward, done, info = asyncio.run(exercise())
    assert system == {"obs_str": "official system"}
    assert obs["obs_str"].count("<image>") == 2
    assert obs["multi_modal_input"]["<image>"] == ["new-image", "extra-image"]
    assert "remaining_route_cm" not in obs["obs_str"]
    assert reset_info["images_dropped"] == 1
    assert reset_info["task_id"] == "fake"
    assert next_obs == {"obs_str": ""}
    assert done is True
    assert info["success"] is True
    assert info["safe_success"] is True
    assert info["decision_latency_seconds"] == 1.5
    assert reward == pytest.approx(2 + 100 - 0.015)
    assert backend.latencies == [1.5]
    assert backend.closed is True


def test_max_turns_terminates_an_otherwise_live_episode():
    backend = FakeBackend()
    env = SimWorldRealTimeGymEnv(
        {"_backend": backend, "latency_mode": "zero", "max_turns": 1}
    )

    async def exercise():
        await env.reset(0)
        return await env.step("bad format")

    _obs, _reward, done, info = asyncio.run(exercise())
    assert done is True
    assert info["termination"] == "max_turns"


def test_injected_backend_can_be_reset_more_than_once():
    backend = FakeBackend()
    env = SimWorldRealTimeGymEnv({"_backend": backend, "latency_mode": "zero"})

    async def exercise():
        await env.reset(1)
        await env.reset(2)
        await env.close()

    asyncio.run(exercise())
    assert backend.closed is True


def test_backend_close_failure_still_releases_endpoint_lease():
    class FailingCloseBackend(FakeBackend):
        def close(self):
            raise RuntimeError("cleanup failed")

    class LeaseManager:
        released = False

        def __exit__(self, *_args):
            self.released = True

    backend = FailingCloseBackend()
    lease = LeaseManager()
    env = SimWorldRealTimeGymEnv({"_backend": backend, "latency_mode": "zero"})
    env._lease_manager = lease
    with pytest.raises(RuntimeError, match="cleanup failed"):
        asyncio.run(env.close())
    assert lease.released is True
    assert env._backend is None


def test_failed_reset_closes_backend_instead_of_stranding_it():
    class FailingBackend(FakeBackend):
        def reset(self, seed):
            raise RuntimeError("UE startup failed")

    backend = FailingBackend()
    env = SimWorldRealTimeGymEnv({"_backend": backend})
    with pytest.raises(RuntimeError, match="UE startup failed"):
        asyncio.run(env.reset(0))
    assert backend.closed is True


def test_endpoint_pool_filters_map_and_releases_lock(tmp_path):
    path = tmp_path / "endpoints.json"
    path.write_text(
        json.dumps(
            {
                "instances": [
                    {"id": "a", "host": "127.0.0.1", "port": 9010, "map_name": "RT10"},
                    {"id": "b", "host": "127.0.0.1", "port": 9020, "map_name": "RT12"},
                ]
            }
        )
    )
    pool = EndpointLeasePool(path, map_name="RT10", lease_timeout_s=0.01)
    with pool.lease() as endpoint:
        assert endpoint.id == "a"
        with pytest.raises(TimeoutError):
            with EndpointLeasePool(path, map_name="RT10", lease_timeout_s=0.0).lease():
                pass
    with pool.lease() as endpoint:
        assert endpoint.port == 9010


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        SimWorldRealTimeGymEnv({"backend": "album"})
    with pytest.raises(ValueError, match="latency_mode"):
        SimWorldRealTimeGymEnv({"latency_mode": "guess"})
    with pytest.raises(ValueError, match="progress_scale"):
        SimWorldRealTimeGymEnv({"reward": {"progress_scale_cm": 0}})


def test_external_action_bridge_uses_the_benchmark_parser():
    bridge = _ExternalActionBridge()
    with pytest.raises(_PromptCaptured):
        bridge.generate_response_openai("system", "user", images=["frame"])
    bridge.queue("Action: move_to\nParam: 6", 1.25)
    action, latency, chars, response, *_usage = bridge.generate_response_openai(
        "system", "user", images=["frame"]
    )
    assert action.action_type == "move_to"
    assert action.action_param == "6"
    assert latency == 1.25
    assert chars == len(response)
