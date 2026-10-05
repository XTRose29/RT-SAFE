"""Physical integration checks, with assertions on measured native events."""

from __future__ import annotations
import json
import math
import copy
from pathlib import Path
from .contract import Action, local_to_world
from .runner import NativeEpisode


def validate_suite(summaries: list[dict], delays: dict) -> None:
    cases = {row["case"]: row for row in summaries}
    for row in summaries:
        assert not row["runtime_errors"], row
    assert cases["clear"]["safe_success"], "Clear route should complete safely"
    assert cases["park-clear"]["safe_success"], "Park route should complete safely"
    assert (
        cases["obstacle"]["active_collisions"] == 1
    ), "Sustained box contact should be one collision"
    assert not cases["obstacle"]["success"], "Agent should be physically blocked by box"
    assert cases["hazards"]["success"] and not cases["hazards"]["safe_success"]
    assert all(
        cases["hazards"]["safety_events"][kind] == 1
        for kind in ("water", "oil", "trip")
    )
    assert cases["red"]["success"] and cases["red"]["safety_events"]["red_light"] == 1
    assert cases["green"][
        "safe_success"
    ], "Waiting for WALK should produce a safe crossing"
    assert cases["pedestrian-real"]["passive_collisions"] == 1
    assert cases["pedestrian-static"]["safety_events"]["collision"] == 0
    assert cases["pedestrian-park-real"]["passive_collisions"] == 1
    assert cases["pedestrian-park-static"]["safety_events"]["collision"] == 0
    for name in ("pedestrian-escape-side", "pedestrian-escape-back"):
        assert cases[name]["passive_collisions"] == 1
        assert cases[name]["active_collisions"] == 0
        assert cases[name]["escape_displacement_cm"] > 350
    assert cases["vehicle-real"]["passive_collisions"] == 1
    assert cases["vehicle-real"]["terminal_reason"] == "vehicle_collision"
    assert cases["vehicle-static"]["safety_events"]["collision"] == 0
    assert cases["robot-real"]["passive_collisions"] == 1
    assert cases["robot-static"]["safety_events"]["collision"] == 0
    assert cases["movable"]["active_collisions"] >= 1
    assert cases["movable"]["maximum_object_displacement_cm"] > 1
    assert cases["native-wall"]["active_collisions"] >= 1
    assert cases["falling-real"]["passive_collisions"] >= 1
    assert cases["falling-static"]["safety_events"]["collision"] == 0
    assert cases["falling-avoid"]["safe_success"]
    assert cases["conflict-red"]["terminal_reason"] == "vehicle_collision"
    assert cases["conflict-red"]["safety_events"]["red_light"] == 1
    assert len(cases["conflict-red"]["dynamic_activations"]) == 1
    assert cases["conflict-green"]["safe_success"]
    assert not cases["conflict-green"]["dynamic_activations"]
    assert cases["mixed-route"]["terminal_reason"] == "vehicle_collision"
    assert all(
        cases["mixed-route"]["safety_events"][kind] == 1
        for kind in ("water", "oil", "trip", "red_light", "collision")
    )
    assert cases["recontact"]["active_collisions"] == 2
    assert cases["off-crosswalk"]["safety_events"]["off_crosswalk"] == 1
    assert cases["off-crosswalk"]["safety_events"]["red_light"] == 0
    for name, evidence in delays.items():
        before = evidence["before"]
        after = evidence["after"]
        sim = after["simulation_time"] - before["simulation_time"]
        wall = after["wall_time"] - before["wall_time"]
        actor = (
            "walker"
            if name.startswith("pedestrian")
            else (
                "robot-dog"
                if name.startswith("robot")
                else "falling-box" if name.startswith("falling") else "car"
            )
        )
        distance = math.dist(before["entities"][actor], after["entities"][actor])
        if name.endswith("static"):
            assert (
                sim == 0 and distance == 0
            ), "Static inference must freeze simulation and pedestrian"
        elif name.startswith(("pedestrian", "robot")):
            assert (
                0.9 <= sim / wall <= 1.1
            ), f"Real-time clock ratio {sim/wall:.3f} is outside tolerance"
            assert (
                distance > 400
            ), "Pedestrian should travel toward the stationary agent"
        elif name.startswith("falling"):
            assert distance > 100, "Falling object should physically descend"


def run_suite(client, manifest: dict, output: Path) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Smoke output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("clear", "nyc-sidewalk-clear", "realtime", [], 8),
        ("park-clear", "nyc-park-clear", "realtime", [], 8),
        ("obstacle", "nyc-sidewalk-obstacle", "realtime", [], 4),
        ("hazards", "nyc-sidewalk-hazards", "realtime", [], 8),
        ("red", "nyc-crosswalk-signal", "realtime", [], 8),
        ("green", "nyc-crosswalk-signal", "static", [Action("wait", "3")] * 3, 8),
        ("pedestrian-real", "nyc-sidewalk-pedestrian", "realtime", [], 0),
        ("pedestrian-static", "nyc-sidewalk-pedestrian", "static", [], 0),
        ("pedestrian-park-real", "nyc-park-pedestrian", "realtime", [], 0),
        ("pedestrian-park-static", "nyc-park-pedestrian", "static", [], 0),
        (
            "pedestrian-escape-side",
            "nyc-park-pedestrian",
            "realtime",
            [Action("turn_around", "L90")],
            1,
        ),
        (
            "pedestrian-escape-back",
            "nyc-park-pedestrian",
            "realtime",
            [Action("turn_around", "L90")] * 2,
            1,
        ),
        ("vehicle-real", "nyc-road-vehicle-contact", "realtime", [], 0),
        ("vehicle-static", "nyc-road-vehicle-contact", "static", [], 0),
        ("robot-real", "nyc-sidewalk-robot", "realtime", [], 0),
        ("robot-static", "nyc-sidewalk-robot", "static", [], 0),
        ("falling-real", "nyc-sidewalk-falling", "realtime", [], 0),
        ("falling-static", "nyc-sidewalk-falling", "static", [], 0),
        ("falling-avoid", "nyc-sidewalk-falling", "static", [], 8),
        ("conflict-red", "nyc-crosswalk-conflict", "static", [], 8),
        (
            "conflict-green",
            "nyc-crosswalk-conflict",
            "static",
            [Action("wait", "3")] * 3,
            8,
        ),
        ("movable", "nyc-sidewalk-movable", "realtime", [], 3),
        (
            "native-wall",
            "nyc-sidewalk-clear",
            "static",
            [Action("turn_around", "R90")],
            3,
        ),
    ]
    summaries = []
    delays = {}
    for name, task, mode, prefix, steps in cases:
        with NativeEpisode(
            client, manifest, task, output / name, mode=mode, reload_source=True
        ) as episode:
            episode.observe(0)
            escape_displacement = None
            if name.startswith(("pedestrian", "vehicle", "robot")) or name in (
                "falling-real",
                "falling-static",
            ):
                delays[name] = episode.wait_inference(8)
                if name.startswith("pedestrian-escape"):
                    before = delays[name]["after"]["position_cm"]
                    for action in prefix + [Action("move_to", "5")] * steps:
                        state = episode.step(action)
                    escape_displacement = math.dist(
                        before[:2], state["position_cm"][:2]
                    )
            else:
                for action in prefix + [Action("move_to", "5")] * steps:
                    state = episode.step(action)
                    if state["phase"] == "terminal":
                        break
                    if (
                        name in ("obstacle", "native-wall")
                        and state["counts"]["collision"]
                    ):
                        break
            if name == "movable":
                episode.wait_inference(2)
            episode.observe(1)
            summary = {"case": name, **episode.finish()}
            if escape_displacement is not None:
                summary["escape_displacement_cm"] = escape_displacement
            if name == "movable":
                samples = json.loads((output / name / "trajectory.json").read_text())[
                    "samples"
                ]
                initial = samples[0]["entities"]["crate"]
                summary["maximum_object_displacement_cm"] = max(
                    math.dist(initial, sample["entities"]["crate"])
                    for sample in samples
                )
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
            (output / "summaries.json").write_text(json.dumps(summaries, indent=2))

    recontact = (
        [Action("move_to", "5")] * 3
        + [Action("turn_around", "L90")] * 2
        + [Action("move_to", "2")]
        + [Action("turn_around", "L90")] * 2
        + [Action("move_to", "5")]
    )
    shifted = copy.deepcopy(manifest)
    task = next(t for t in shifted["tasks"] if t["id"] == "nyc-crosswalk-signal")
    sidewalk = next(t for t in shifted["tasks"] if t["id"] == "nyc-sidewalk-clear")
    for key in ("start_cm", "goal_cm"):
        task[key] = list(local_to_world(task[key], sidewalk["yaw_deg"], 500, 0))
    task["reference_route_cm"] = [task["start_cm"], task["goal_cm"]]
    for name, scene, task_id, actions in [
        ("recontact", manifest, "nyc-sidewalk-obstacle", recontact),
        (
            "mixed-route",
            manifest,
            "nyc-city-route",
            [Action("move_to", "5")] * 9
            + [Action("turn_around", "L90")]
            + [Action("move_to", "5")] * 8,
        ),
        (
            "off-crosswalk",
            shifted,
            task["id"],
            [Action("wait", "3")] * 3 + [Action("move_to", "5")] * 3,
        ),
    ]:
        with NativeEpisode(
            client, scene, task_id, output / name, mode="static", reload_source=True
        ) as episode:
            episode.observe(0)
            for action in actions:
                state = episode.step(action)
                if state["phase"] == "terminal":
                    break
            episode.observe(1)
            summary = {"case": name, **episode.finish()}
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
            (output / "summaries.json").write_text(json.dumps(summaries, indent=2))
    validate_suite(summaries, delays)
    report = {"passed": True, "cases": len(summaries), "summaries": summaries}
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    return report
