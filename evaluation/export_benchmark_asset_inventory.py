#!/usr/bin/env python3
"""Export the authored SimWorld-RealTime map/task/asset inventory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ue_assets = read_json(REPO_ROOT / "data/ue_assets.json")
    maps = []
    obstacle_totals: Counter[str] = Counter()
    building_totals: Counter[str] = Counter()
    total_tasks = 0
    total_roads = 0

    for task_path in sorted((REPO_ROOT / "data").glob("map*/tasks.json")):
        map_dir = task_path.parent
        tasks = read_json(task_path)["tasks"]
        roads = read_json(map_dir / "roads.json")["roads"]
        obstacles = read_json(map_dir / "obstacles.json")["nodes"]
        buildings = read_json(map_dir / "buildings.json")["buildings"]
        obstacle_types = Counter(item["instance_name"] for item in obstacles)
        building_types = Counter(item["type"] for item in buildings)
        movable = [
            item for item in obstacles if item["id"].startswith("GEN_RT_RT_")
        ]
        falling = [
            item for item in obstacles if item["id"].startswith("GEN_RT_FO_")
        ]
        static = [
            item for item in obstacles
            if not item["id"].startswith(("GEN_RT_RT_", "GEN_RT_FO_"))
        ]
        referenced_falling = {
            actor_id
            for task in tasks
            for actor_id in task.get("falling_objects", [])
        }
        maps.append(
            {
                "map": map_dir.name,
                "roads": len(roads),
                "tasks": len(tasks),
                "task_ids": [task["task_id"] for task in tasks],
                "task_hop_counts": dict(
                    sorted(Counter(task["total_hops"] for task in tasks).items())
                ),
                "task_crosswalk_hop_counts": dict(
                    sorted(
                        Counter(task["crosswalk_hops"] for task in tasks).items()
                    )
                ),
                "authored_obstacles": len(obstacles),
                "static_street_assets": len(static),
                "movable_hazard_assets": len(movable),
                "falling_hazard_assets_authored": len(falling),
                "falling_hazard_ids_referenced_by_tasks": len(
                    referenced_falling
                ),
                "buildings": len(buildings),
                "obstacle_type_counts": dict(sorted(obstacle_types.items())),
                "building_type_counts": dict(sorted(building_types.items())),
            }
        )
        total_tasks += len(tasks)
        total_roads += len(roads)
        obstacle_totals.update(obstacle_types)
        building_totals.update(building_types)

    all_asset_names = sorted(set(obstacle_totals) | set(building_totals))
    output = {
        "scope": "Five released procedural benchmark maps RT10/RT12/RT15/RT18/RT20",
        "totals": {
            "maps": len(maps),
            "roads": total_roads,
            "tasks": total_tasks,
            "authored_obstacle_instances": sum(obstacle_totals.values()),
            "building_instances": sum(building_totals.values()),
            "obstacle_asset_types": len(obstacle_totals),
            "building_asset_types": len(building_totals),
        },
        "maps": maps,
        "obstacle_asset_type_counts": dict(sorted(obstacle_totals.items())),
        "building_asset_type_counts": dict(sorted(building_totals.items())),
        "asset_library_entries": {
            name: ue_assets.get(name) for name in all_asset_names
        },
        "runtime_spawned_assets": {
            "agent": "/Game/RealTimeBench/Agent/BP_RT_Agent.BP_RT_Agent_C",
            "ordinary_pedestrian": "/Game/RealTimeBench/Agent/RT_Base_Pedestrian.RT_Base_Pedestrian_C",
            "irregular_robot_dog": "/Game/RealTimeBench/Agent/RT_Base_Dog.RT_Base_Dog_C",
            "road": ue_assets.get("BP_Road2_C"),
            "signal_controlled_pedestrian": "/Game/TrafficSystem/Pedestrian/Base_Pedestrian.Base_Pedestrian_C",
            "traffic_signal_vehicle_and_pedestrian": "/Game/city_props/BP/props/street_light/BP_street_light.BP_street_light_C",
            "traffic_signal_pedestrian": "/Game/city_props/BP/props/street_light/BP_street_light_ped.BP_street_light_ped_C",
            "legacy_intersection_controller": "/Game/RealTimeBench/Traffic/RT_BP_Intersection.RT_BP_Intersection_C",
            "native_intersection_controller": "/Script/SimWorld.RTTrafficIntersectionController",
            "signal_controlled_vehicles": {
                f"Vehicle{index}_C": (
                    f"/Game/TrafficSystem/Vehicle/Vehicle{index}."
                    f"Vehicle{index}_C"
                )
                for index in range(1, 6)
            },
        },
        "unsafe_trigger_semantics": [
            {
                "trigger": "ordinary or irregular actor contact",
                "assets": ["RT_Base_Pedestrian", "RT_Base_Dog"],
                "detected_by": "UE HumanCollision counter delta; also sampled before an action for passive realtime collisions",
                "result": "human collision feedback and collision counters",
                "penalty": "6 simulated seconds when reported impulse is nonzero; passive contact is counted without a separate intensity penalty",
            },
            {
                "trigger": "solid movable or falling object contact",
                "assets": [
                    "RT_Basketball_C", "RT_Bottle1_C", "RT_Bottle2_C",
                    "RT_Box_C", "RT_Soccer_C", "RT_Tennis_C",
                    "RT_Falling_Basketball_C", "RT_Falling_Box_C",
                    "RT_Falling_Box2_C", "RT_Falling_Soccer_C",
                ],
                "detected_by": "UE ObjectCollision counter delta and collision impulse",
                "result": "object collision feedback and collision counters",
                "penalty": "6 simulated seconds when reported impulse is nonzero",
            },
            {
                "trigger": "fall overlap",
                "assets": ["fall-enabled object Blueprint (OverlapType=1)"],
                "detected_by": "UE GetStates OverlapType=1",
                "result": "fall_count increment and fell feedback",
                "penalty": "6 simulated seconds recovery (PROB_FALL=1)",
            },
            {
                "trigger": "oil overlap",
                "assets": ["RT_Oil_C"],
                "detected_by": "UE GetStates OverlapType=2",
                "result": "oil_count increment; next move runs at 50% speed then restores speed",
                "penalty": "longer next movement exposure, no fixed additive penalty",
            },
            {
                "trigger": "water-puddle overlap",
                "assets": ["RT_WaterPuddle_C"],
                "detected_by": "UE GetStates OverlapType=3",
                "result": "water_count increment and slipping state; selected target remains unchanged",
                "penalty": "no fixed additive time penalty",
            },
            {
                "trigger": "building contact",
                "assets": ["all BP_Building_*_C types"],
                "detected_by": "UE BuildingCollision counter delta and collision impulse",
                "result": "rollback to last collision-free state; repeated identical action tracked",
                "penalty": "collision impulse penalty plus 3 simulated seconds recovery; terminate after 3 repeated identical-action contacts",
            },
            {
                "trigger": "road surface contact",
                "assets": ["BP_Road2_C / roadway collision surface"],
                "detected_by": "UE GetStates TouchedRoad flag",
                "result": "road occupancy is legal within any authored pedestrian crossing and is an illegal crossing otherwise",
                "penalty": "one event is recorded per continuous illegal crossing and the conflict-vehicle mechanism is invoked",
            },
            {
                "trigger": "red-light crossing violation",
                "assets": [
                    "BP_street_light_C",
                    "BP_street_light_ped_C",
                    "Vehicle1_C through Vehicle5_C",
                ],
                "detected_by": "authored-crosswalk roadway occupancy while the synchronized pedestrian state is not WALK, including a WALK-to-non-WALK transition before the far curb; counted once per crossing episode",
                "result": "red_light_violations_count and a conflict-vehicle launch record",
                "penalty": "launch an unused vehicle on a real lane with random 3--9m approach clearance; impact within the 1.0m swept-distance envelope is terminal",
            },
            {
                "trigger": "illegal crossing",
                "assets": [
                    "BP_Road2_C",
                    "Vehicle1_C through Vehicle5_C",
                ],
                "detected_by": "roadway entry outside the boundary of every authored pedestrian crossing",
                "result": "illegal_crossing_violations_count, entry pose/direction, and a conflict-vehicle launch record",
                "penalty": "launch an unused vehicle on a real lane with random 3--9m approach clearance; leaving the road or entering an authored crossing rearms the detector",
            },
            {
                "trigger": "vehicle impact",
                "assets": ["Vehicle1_C through Vehicle5_C"],
                "detected_by": "positive delta of UE GetCollisionNum.VehicleCollision after actions, waits, and realtime evolution",
                "result": "vehicle_collision_count, failed=true, failure_reason=vehicle_collision",
                "penalty": "terminal failure; no separate additive time penalty",
            },
        ],
        "requested_video_audit_scope": {
            "included": [
                "pedestrians and irregular NPCs",
                "static/movable/falling objects",
                "fall, oil, water, building, and road triggers",
            ],
            "documented_but_disabled_for_this_video_suite": [
                "symbolic traffic assistance",
                "red-light and illegal-crossing conflict-vehicle launches",
                "vehicle collision scoring",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
