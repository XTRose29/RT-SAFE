#!/usr/bin/env python3
"""Exercise live UE human, object, and repeated-building collision paths."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
simworld = types.ModuleType("simworld")
simworld.__path__ = [str(REPO_ROOT / "SimWorld" / "simworld")]
sys.modules.setdefault("simworld", simworld)

from base.rt_action_space import MOVE_TO
from base.rt_communicator import RTCommunicator
from base.rt_unrealcv import RTUnrealCV
from manager.world_manager import WorldManager
from simworld.utils.vector import Vector


class PreparedWorldManager(WorldManager):
    """Build a normal task world without entering the policy loop."""

    def run(self):
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", default="data/map1_10roads/tasks.json")
    parser.add_argument("--task-number", type=int, default=2)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--unrealcv-host", default="127.0.0.1")
    parser.add_argument("--unrealcv-port", type=int, default=9130)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def state_record(raw):
    return {
        "human": int(raw[0]),
        "object": int(raw[1]),
        "building": int(raw[2]),
        "intensity": float(raw[3]),
        "overlap_type": int(raw[4]),
        "touched_road": int(raw[5]),
    }


def state_delta(before, after):
    return {
        kind: max(0, int(after[kind]) - int(before[kind]))
        for kind in ("human", "object", "building")
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    unrealcv = RTUnrealCV(port=args.unrealcv_port, ip=args.unrealcv_host)
    communicator = RTCommunicator(unrealcv)
    manager = PreparedWorldManager(
        communicator,
        agent_path=args.agent_config,
        task_file_path=args.task_file,
        seed=args.seed,
        use_tick=True,
        realtime_thinking=True,
        use_action_frames=False,
        red_light_conflict_vehicle_enabled=False,
        static_signal_vehicles=True,
        pedestrian_signal_compliance_probability=1.0,
        pedestrians_enabled=True,
        movable_obstacles_enabled=True,
        irregular_npcs_enabled=True,
        falling_objects_enabled=True,
        load_all_unsafe_triggers=False,
    )
    result = {
        "task_file": args.task_file,
        "task_number": args.task_number,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "collision_authority": "unreal_engine_GetStates_counters",
        "probes": {},
        "passed": False,
    }

    def set_agent_pose(position: Vector, yaw: float):
        live = unrealcv.get_location(manager.agent.name)
        unrealcv.set_orientation((0.0, yaw, 0.0), manager.agent.name)
        unrealcv.set_location(
            (float(position.x), float(position.y), float(live[2])),
            manager.agent.name,
        )
        manager.agent.position = Vector(position.x, position.y)
        manager.agent.direction = yaw
        manager.agent.sync_ue()
        manager.agent.success = False
        manager.agent.failed = False
        manager.agent.failure_reason = None

    def baseline_agent_counters():
        raw = communicator.get_states(manager.agent.name)
        manager.agent.last_ue_collision_count["human"] = int(raw[0])
        manager.agent.last_ue_collision_count["object"] = int(raw[1])
        manager.agent.last_ue_collision_count["building"] = int(raw[2])
        return state_record(raw)

    def native_physical_move(start: Vector, target: Vector):
        """Exercise UE's Blueprint MoveTo even for an on-route contact ray."""
        distance = start.distance(target)
        move_time = max(0.1, round(distance / manager.agent.speed, 1))
        manager.agent.last_commanded_waypoint = Vector(target.x, target.y)
        manager.agent.last_move_command_duration_seconds = move_time
        manager.agent.last_move_execution_mode = "ue_blueprint_move_to_probe"
        manager.agent.last_physical_move_actor_positions = (
            manager.agent._sample_live_actor_positions_for_physical_move()
        )
        communicator.rt_agent_move_to(manager.agent.name, target, move_time)
        manager.agent._advance_simulation_time(move_time)

    def action_probe(
        start: Vector,
        target: Vector,
        expected: str,
        actor_name: str,
    ):
        delta = target - start
        yaw = math.degrees(math.atan2(delta.y, delta.x))
        set_agent_pose(start, yaw)
        manager.agent._advance_simulation_time(0.5)
        before = baseline_agent_counters()
        manager.agent.last_action_start_position = Vector(start.x, start.y)
        manager.agent.last_action_type = "MOVE"
        native_physical_move(start, target)
        manager.agent.sync_ue()
        pre_actor_position = manager.agent.last_physical_move_actor_positions.get(
            actor_name
        )
        feedback, details, _ = manager.agent.get_feedback()
        after = state_record(communicator.get_states(manager.agent.name))
        record = {
            "start_cm": {"x": start.x, "y": start.y},
            "target_cm": {"x": target.x, "y": target.y},
            "before": before,
            "after": after,
            "ue_delta": state_delta(before, after),
            "feedback": feedback,
            "collision_details": details,
            "pre_contact_actor_cm": (
                {
                    "x": pre_actor_position.x,
                    "y": pre_actor_position.y,
                }
                if pre_actor_position is not None
                else None
            ),
            "pre_contact_actor_count": len(
                manager.agent.last_physical_move_actor_positions
            ),
            "detected": int(details.get(expected, 0) or 0) > 0,
            "endpoint_cm": {
                "x": manager.agent.position.x,
                "y": manager.agent.position.y,
            },
        }
        print(
            "SAFETY_PROBE "
            f"kind={expected} detected={record['detected']} "
            f"delta={record['ue_delta']} start={start} target={target}",
            f"endpoint={record['endpoint_cm']} "
            f"pre_actor={record['pre_contact_actor_cm']} "
            f"pre_count={record['pre_contact_actor_count']} "
            f"physical_kind={details.get('ue_physical_block_contact_kind')}",
            flush=True,
        )
        return record

    def approaches(target: Vector, distance: float):
        for axis in (
            Vector(1.0, 0.0),
            Vector(-1.0, 0.0),
        ):
            # Traverse through the actor. Reaching the near-side actor center
            # is an ambiguous endpoint; stopping short of the far endpoint is
            # auditable evidence that native UE collision physically blocked
            # the command.
            yield target - axis * distance, target + axis * distance

    try:
        manager.run_single_task(args.task_number - 1, difficulty=args.difficulty)
        manager.agent.sync_ue()
        # Preserve the benchmark's 200 cm/s Python/Blueprint speed agreement;
        # native MoveTo duration is derived from this value.
        manager.agent.speed = 200.0
        # Keep the probe isolated from traffic-rule consequences; these checks
        # exercise only contact counters and building recovery.
        manager.agent.red_light_violation_callback = None
        manager.agent.illegal_crossing_callback = None

        available = {str(name) for name in unrealcv.get_objects()}

        human_attempts = []
        for pedestrian in list(getattr(manager, "pedestrians", []))[:4]:
            name = communicator.get_pedestrian_name(pedestrian.id)
            if name not in available:
                continue
            # Managed pedestrians use Base_Pedestrian's StopPedestrian entry
            # point, not the humanoid agent's StopAgent entry point.
            stop = getattr(unrealcv, "p_stop", None)
            if callable(stop):
                stop(name)
            unrealcv.rt_set_pedestrian_speed(name, 0.0)
            unrealcv.set_collision(name, True)
            location = unrealcv.get_location(name)
            target = Vector(float(location[0]), float(location[1]))
            for start, destination in approaches(target, 400.0):
                attempt = action_probe(start, destination, "human", name)
                attempt["actor"] = name
                human_attempts.append(attempt)
                if attempt["detected"]:
                    break
            if human_attempts and human_attempts[-1]["detected"]:
                break
        result["probes"]["human"] = {
            "passed": bool(human_attempts and human_attempts[-1]["detected"]),
            "attempts": human_attempts,
        }

        object_attempts = []
        movable_names = [
            name
            for name in getattr(manager, "selected_movable_obstacle_ids", [])
            if name in available
        ]
        for name in movable_names[:8]:
            unrealcv.set_physics(name, False)
            unrealcv.set_movable(name, False)
            unrealcv.set_collision(name, True)
            location = unrealcv.get_location(name)
            target = Vector(float(location[0]), float(location[1]))
            for start, destination in approaches(target, 400.0):
                attempt = action_probe(start, destination, "object", name)
                attempt["actor"] = name
                object_attempts.append(attempt)
                if attempt["detected"]:
                    break
            if object_attempts and object_attempts[-1]["detected"]:
                break
        result["probes"]["object"] = {
            "passed": bool(object_attempts and object_attempts[-1]["detected"]),
            "attempts": object_attempts,
        }

        building_attempts = []
        repeated = []
        buildings = [
            obstacle
            for obstacle in getattr(manager.agent, "static_obstacles", [])
            if obstacle.get("kind") == "building" and obstacle.get("bounds")
        ]
        # Building 20's east wall has a UE-confirmed collision ray in prior
        # Map1 rollouts. Put it first so this is deterministic and does not
        # depend on searching through large, sometimes hollow footprints.
        buildings.sort(key=lambda item: item.get("id") != "building_20")
        # Try short wall-normal paths through exported building bounds. This
        # avoids teleport overlap and uses the same physical MoveTo as an
        # off-route policy action.
        building_paths = []
        for building in buildings:
            bounds = building["bounds"]
            cx = (float(bounds["min_x"]) + float(bounds["max_x"])) / 2.0
            cy = (float(bounds["min_y"]) + float(bounds["max_y"])) / 2.0
            building_paths.extend([
                (
                    building,
                    Vector(float(bounds["min_x"]) - 500.0, cy),
                    Vector(float(bounds["min_x"]) + 250.0, cy),
                ),
                (
                    building,
                    Vector(float(bounds["max_x"]) + 500.0, cy),
                    Vector(float(bounds["max_x"]) - 250.0, cy),
                ),
                (
                    building,
                    Vector(cx, float(bounds["min_y"]) - 500.0),
                    Vector(cx, float(bounds["min_y"]) + 250.0),
                ),
                (
                    building,
                    Vector(cx, float(bounds["max_y"]) + 500.0),
                    Vector(cx, float(bounds["max_y"]) - 250.0),
                ),
            ])

        chosen = None
        action = SimpleNamespace(action_type=MOVE_TO, action_param="live_probe")
        for building, start, target in building_paths[:80]:
            set_agent_pose(start, math.degrees(math.atan2(
                target.y - start.y,
                target.x - start.x,
            )))
            manager.agent._advance_simulation_time(0.5)
            baseline_agent_counters()
            manager.agent._remember_collision_free_state("live_building_probe")
            manager.agent.last_action_start_position = Vector(start.x, start.y)
            manager.agent.last_action_type = "MOVE"
            native_physical_move(start, target)
            manager.agent.sync_ue()
            feedback, details, _ = manager.agent.get_feedback()
            attempt = {
                "building_id": building.get("id"),
                "building_type": building.get("type"),
                "start_cm": {"x": start.x, "y": start.y},
                "target_cm": {"x": target.x, "y": target.y},
                "collision_details": details,
                "feedback": feedback,
                "detected": int(details.get("building", 0) or 0) > 0,
            }
            building_attempts.append(attempt)
            print(
                "SAFETY_PROBE "
                f"kind=building detected={attempt['detected']} "
                f"building={attempt['building_id']} start={start} "
                f"target={target}",
                flush=True,
            )
            if not attempt["detected"]:
                manager.agent._reset_building_collision_streak()
                continue
            finalized, recovered = manager.agent._finalize_post_action_collisions(
                action,
                f"Move to {target}",
                feedback,
                details,
            )
            attempt["recovered"] = recovered
            attempt["final_feedback"] = finalized
            attempt["repeat_count"] = (
                manager.agent.consecutive_same_building_action_count
            )
            chosen = (building, start, target)
            repeated.append(attempt)
            break

        if chosen is not None:
            building, start, target = chosen
            for repeat_index in (2, 3):
                yaw = math.degrees(math.atan2(
                    target.y - start.y,
                    target.x - start.x,
                ))
                # A production repeat is a new decision after rollback,
                # observation, and inference. Re-establish the recovered safe
                # pose and give UE a short decision-boundary settle so the
                # prior MoveTo timeline cannot swallow the next command.
                set_agent_pose(start, yaw)
                manager.agent._advance_simulation_time(0.5)
                baseline_agent_counters()
                manager.agent.last_action_start_position = Vector(start.x, start.y)
                manager.agent.last_action_type = "MOVE"
                native_physical_move(start, target)
                manager.agent.sync_ue()
                feedback, details, _ = manager.agent.get_feedback()
                finalized, recovered = manager.agent._finalize_post_action_collisions(
                    action,
                    f"Move to {target}",
                    feedback,
                    details,
                )
                repeated.append({
                    "repeat_index": repeat_index,
                    "building_id": building.get("id"),
                    "collision_details": details,
                    "feedback": feedback,
                    "final_feedback": finalized,
                    "detected": int(details.get("building", 0) or 0) > 0,
                    "recovered": recovered,
                    "repeat_count": (
                        manager.agent.consecutive_same_building_action_count
                    ),
                    "failed": manager.agent.failed,
                    "failure_reason": manager.agent.failure_reason,
                })

        building_passed = bool(
            len(repeated) == 3
            and all(item.get("detected") for item in repeated)
            and all(item.get("recovered") for item in repeated)
            and repeated[-1].get("repeat_count") == 3
            and repeated[-1].get("failed") is True
            and repeated[-1].get("failure_reason")
            == "repeated_building_collision"
        )
        result["probes"]["building"] = {
            "passed": building_passed,
            "attempts": building_attempts,
            "repeated_collision_sequence": repeated,
            "terminal_failed": manager.agent.failed,
            "terminal_failure_reason": manager.agent.failure_reason,
            "recovery_event_count": len(
                manager.agent.building_collision_recovery_events
            ),
        }

        result["passed"] = all(
            probe.get("passed") for probe in result["probes"].values()
        )
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    finally:
        try:
            manager.cleanup()
        finally:
            unrealcv.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
