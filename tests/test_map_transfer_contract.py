import math
import pytest
from benchmark.map_transfer.contract import (
    Action,
    WAYPOINTS,
    TURN_ANGLES,
    EventLedger,
    point_in_polygon,
)


def test_legacy_action_fan_uses_centimeters_and_visual_left():
    assert len(WAYPOINTS) + len(TURN_ANGLES) + 3 == 16
    assert Action("move_to", "5").target((100, 200, 30), 0) == (500, 200, 30)
    left = Action("move_to", "3").target((0, 0, 0), 0)
    assert left[0] == pytest.approx(200 / math.sqrt(2))
    assert left[1] == pytest.approx(-200 / math.sqrt(2))
    right = Action("move_to", "4").target((0, 0, 0), 90)
    assert right[0] < 0 and right[1] > 0
    assert TURN_ANGLES["L90"] == -90


def test_invalid_actions_fail_before_engine_command():
    for kind, param in [
        ("move_to", "8"),
        ("wait", "0"),
        ("teleport", "1"),
        ("turn_around", "90"),
    ]:
        with pytest.raises(ValueError):
            Action(kind, param)


def test_collision_requires_hit_and_deduplicates_until_separation():
    ledger = EventLedger()
    kwargs = dict(
        simulation_time=1,
        wall_time=1,
        phase="inference",
        evidence={"source": "unreal_blocking_hit"},
    )
    assert ledger.enter("collision", "pedestrian:1", **kwargs)
    assert not ledger.enter("collision", "pedestrian:1", **kwargs)
    ledger.leave("collision", "pedestrian:1")
    assert ledger.enter("collision", "pedestrian:1", **kwargs)
    assert ledger.counts()["collision"] == 2
    assert not ledger.safe_success(True)
    with pytest.raises(ValueError):
        ledger.enter(
            "collision",
            "pedestrian:2",
            **{**kwargs, "evidence": {"source": "distance_threshold"}}
        )


def test_arrival_with_hazard_is_not_safe_success():
    ledger = EventLedger()
    assert not ledger.safe_success(False)
    assert ledger.safe_success(True)
    ledger.enter(
        "water",
        "puddle",
        simulation_time=2,
        wall_time=2,
        phase="action",
        evidence={"source": "unreal_overlap"},
    )
    assert not ledger.safe_success(True)


def test_crosswalk_polygon_includes_boundary():
    crosswalk = [(0, 0), (100, 0), (100, 400), (0, 400)]
    assert point_in_polygon((50, 200), crosswalk)
    assert point_in_polygon((0, 200), crosswalk)
    assert not point_in_polygon((-1, 200), crosswalk)
    assert not point_in_polygon((50, 401), crosswalk)


def test_projection_matches_unreal_yaw_and_camera_image_axes():
    from benchmark.map_transfer.observation import project

    assert project((100, 0, 0), (0, 0, 0), (0, 0, 0), 720, 640, 100) == (360, 320)
    assert project((100, 100, 0), (0, 0, 0), (0, 0, 0), 720, 640, 100)[0] > 360
    assert project((100, -100, 0), (0, 0, 0), (0, 0, 0), 720, 640, 100)[0] < 360
    assert project((100, 0, 100), (0, 0, 0), (0, 0, 0), 720, 640, 100)[1] < 320
    assert project((-100, 0, 0), (0, 0, 0), (0, 0, 0), 720, 640, 100) is None


def test_manifest_requires_native_pilot_label_and_cm_units():
    from benchmark.map_transfer.manifest import load_manifest, validate_manifest
    import copy

    data = load_manifest("benchmark/map_transfer/maps/nyc_pilot.json")
    invalid = copy.deepcopy(data)
    invalid["units"] = "m"
    with pytest.raises(ValueError):
        validate_manifest(invalid)
    invalid = copy.deepcopy(data)
    invalid["pilot"] = False
    with pytest.raises(ValueError):
        validate_manifest(invalid)
    invalid = copy.deepcopy(data)
    invalid["tasks"][0]["start_cm"][0] = float("nan")
    with pytest.raises(ValueError):
        validate_manifest(invalid)


def test_manifest_rejects_invalid_physics_and_identity_before_engine():
    import copy
    from benchmark.map_transfer.manifest import load_manifest, validate_manifest

    original = load_manifest("benchmark/map_transfer/maps/nyc_pilot.json")
    mutations = [
        lambda d: d["agent"].update(speed_cm_s=float("inf")),
        lambda d: d["agent"].update(capsule_half_height_cm=10),
        lambda d: d["agent"].update(width=True),
        lambda d: d["tasks"][0].update(time_limit_s=float("nan")),
        lambda d: d["tasks"][1]["entities"][0].update(id="agent"),
        lambda d: d["tasks"][1]["entities"][0].update(size_cm=[1, 0, 1]),
        lambda d: d["tasks"][3]["entities"][0].pop("size_cm"),
        lambda d: d["tasks"][4]["traffic_zones"][0].update(walk_seconds=float("nan")),
        lambda d: d["tasks"][4]["traffic_zones"][0].update(
            road_polygon_cm=[[0, 0], [1, 1], [2, 2]]
        ),
        lambda d: d["tasks"][4]["traffic_zones"][0].update(
            road_polygon_cm=[[0, 0], [3, 3], [0, 3], [3, 0]]
        ),
    ]
    for mutate in mutations:
        invalid = copy.deepcopy(original)
        mutate(invalid)
        with pytest.raises(ValueError):
            validate_manifest(invalid)


def test_policy_actions_fail_closed_without_executing_text():
    from benchmark.map_transfer.policy import parse_action

    assert parse_action(
        '```json\n{"action_type":"wait","action_param":"2"}\n```'
    ) == Action("wait", "2")
    for text in [
        "move forward",
        "[]",
        '{"action_type":"teleport","action_param":"1"}',
        '{"action_type":"wait","action_param":"1","tool":"shell"}',
    ]:
        with pytest.raises(ValueError):
            parse_action(text)


def test_visual_policy_gets_goal_and_feedback_without_scene_oracle():
    from benchmark.map_transfer.policy import policy_prompt

    state = {
        "position_cm": [0, 0, 0],
        "yaw_deg": 0,
        "mode": "realtime",
        "simulation_time": 2,
        "counts": {"collision": 1},
        "entities": {"hidden_person": [4, 5, 6]},
        "signals": {"invisible_signal": "stop"},
    }
    prompt = policy_prompt(state, {"goal_cm": [0, -1000, 0]}, [])
    assert '"goal_bearing_deg": -90.0' in prompt
    assert "hidden_person" not in prompt and "invisible_signal" not in prompt


def test_hazard_consequences_restore_speed_and_use_simulation_time():
    from benchmark.map_transfer.hazards import HazardEffects

    effects = HazardEffects()
    effects.enter("oil", 10)
    assert effects.start_move() == 0.5
    assert effects.start_move() == 1
    effects.enter("trip", 20)
    assert effects.remaining(22) == 4
    assert effects.remaining(30) == 0
    water = effects.enter("water", 31)
    assert water["effect"] == "recorded_without_motion_perturbation"
    assert effects.start_move() == 1


def test_owned_project_keeps_config_local_and_source_content_untouched(tmp_path):
    from benchmark.map_transfer.project import create_project

    source = tmp_path / "source"
    source.mkdir()
    (source / "Config").mkdir()
    (source / "Content" / "City").mkdir(parents=True)
    (source / "SimWorld.uproject").write_text("{}")
    (source / "Config" / "DefaultEditorPerProjectUserSettings.ini").write_text(
        "; source config\n"
    )
    (source / "Content" / "City" / "mesh.uasset").write_bytes(b"source")
    destination = tmp_path / "owned"
    create_project(source / "SimWorld.uproject", destination)
    assert (destination / "Content" / "City").is_symlink()
    assert not (destination / "Content" / "RTSafeNYC").is_symlink()
    assert (source / "Content" / "City" / "mesh.uasset").read_bytes() == b"source"
    assert (
        source / "Config" / "DefaultEditorPerProjectUserSettings.ini"
    ).read_text() == "; source config\n"
    assert (
        "bAutoSaveEnable=False"
        in (
            destination / "Config" / "DefaultEditorPerProjectUserSettings.ini"
        ).read_text()
    )
    with pytest.raises(ValueError):
        create_project(source / "SimWorld.uproject", destination)


def test_source_package_digests_detect_changes_and_reject_missing_packages(tmp_path):
    from benchmark.map_transfer.project import source_digests

    (tmp_path / "City").mkdir()
    mesh = tmp_path / "City" / "Wall.uasset"
    mesh.write_bytes(b"original mesh")
    (tmp_path / "City" / "Map.umap").write_bytes(b"original map")
    packages = ["/Game/City/Wall.Wall", "/Game/City/Map"]
    before = source_digests(tmp_path, packages)
    assert before == source_digests(tmp_path, packages)
    mesh.write_bytes(b"changed mesh")
    assert before != source_digests(tmp_path, packages)
    with pytest.raises(ValueError):
        source_digests(tmp_path, ["/Game/Missing"])
    with pytest.raises(ValueError):
        source_digests(tmp_path, ["/Engine/BasicShapes/Cube"])


def test_cli_visual_policy_rejects_tool_calls_and_failed_or_incomplete_turns():
    import json
    from benchmark.map_transfer.cli_policy import inspect_cli_events

    complete = {
        "type": "turn.completed",
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }
    _, usage = inspect_cli_events(json.dumps(complete))
    assert usage["input_tokens"] == 12
    for event in [
        {"type": "item.completed", "item": {"type": "command_execution"}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call"}},
        {"type": "turn.failed", "error": "unavailable model"},
    ]:
        with pytest.raises(RuntimeError):
            inspect_cli_events(json.dumps(event) + "\n" + json.dumps(complete))
    with pytest.raises(RuntimeError):
        inspect_cli_events('{"type":"turn.started"}')


def test_native_dynamics_require_valid_timing_and_existing_traffic_zone():
    import copy
    from benchmark.map_transfer.manifest import load_manifest, validate_manifest

    original = load_manifest("benchmark/map_transfer/maps/nyc_pilot.json")
    invalid = copy.deepcopy(original)
    invalid["agent"]["turn_duration_s"] = 0
    with pytest.raises(ValueError, match="turn duration"):
        validate_manifest(invalid)
    invalid = copy.deepcopy(original)
    falling = next(t for t in invalid["tasks"] if t["id"] == "nyc-sidewalk-falling")
    falling["entities"][0]["drop_height_cm"] = float("nan")
    with pytest.raises(ValueError, match="drop height"):
        validate_manifest(invalid)
    invalid = copy.deepcopy(original)
    crossing = next(t for t in invalid["tasks"] if t["id"] == "nyc-crosswalk-conflict")
    crossing["entities"][0]["activation_zone"] = "unknown-zone"
    with pytest.raises(ValueError, match="activation zone"):
        validate_manifest(invalid)
