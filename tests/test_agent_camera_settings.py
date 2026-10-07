from types import SimpleNamespace

import numpy as np
import pytest

from base.rt_agent import RTAgent
from simworld.utils.vector import Vector
from utils.annotate_image import annotate_image, project_waypoints_to_image


def make_agent(**kwargs):
    communicator = kwargs.pop("communicator", SimpleNamespace())
    return RTAgent(
        position=Vector(0, 0),
        direction=Vector(1, 0),
        destination=Vector(100, 0),
        shortest_path=[Vector(100, 0)],
        required_time=100,
        communicator=communicator,
        llm=None,
        **kwargs,
    )


def test_route_and_packaged_ue_yaw_conventions_are_explicit_inverses():
    assert RTAgent._route_yaw_to_ue_yaw(0.0) == 0.0
    assert RTAgent._route_yaw_to_ue_yaw(90.0) == 90.0
    assert RTAgent._route_yaw_to_ue_yaw(-90.0) == -90.0
    assert RTAgent._ue_yaw_to_route_yaw(-90.0) == -90.0
    assert RTAgent._ue_yaw_to_route_yaw(90.0) == 90.0


def test_sync_ue_converts_rendered_yaw_to_route_direction():
    unrealcv = SimpleNamespace(
        get_camera_location=lambda camera_id: "-10700 -1700 227",
        get_camera_rotation=lambda camera_id: "0 -90 0",
        get_location=lambda name: [-10700.0, -1700.0, 92.0],
        get_orientation=lambda name: [0.0, 90.0, 0.0],
    )
    agent = make_agent(communicator=SimpleNamespace(unrealcv=unrealcv))
    agent.name = "RT_AGENT"
    agent.camera_id = 1
    agent.first_person_camera_enabled = False
    agent.position_history = []
    agent.step_num = 7
    agent.sync_settle_seconds = 0.0

    agent.sync_ue()

    assert agent.position == Vector(-10700.0, -1700.0)
    assert agent.yaw == 90.0
    assert agent.direction.x == pytest.approx(0.0, abs=1e-8)
    assert agent.direction.y == pytest.approx(1.0, abs=1e-8)


def test_first_person_sync_reuses_actor_pose_without_duplicate_ue_queries():
    calls = []

    def get_location(name):
        calls.append(("get_location", name))
        return [125.0, -75.0, 92.0]

    def get_orientation(name):
        calls.append(("get_orientation", name))
        return [0.0, 30.0, 0.0]

    unrealcv = SimpleNamespace(
        get_location=get_location,
        get_orientation=get_orientation,
        get_camera_location=lambda camera_id: pytest.fail(
            "first-person sync must reuse the commanded camera location"
        ),
        get_camera_rotation=lambda camera_id: pytest.fail(
            "first-person sync must reuse the commanded camera rotation"
        ),
        set_camera_location=lambda camera_id, location: calls.append(
            ("set_camera_location", camera_id, location)
        ),
        set_camera_rotation=lambda camera_id, rotation: calls.append(
            ("set_camera_rotation", camera_id, rotation)
        ),
    )
    agent = make_agent(communicator=SimpleNamespace(unrealcv=unrealcv))
    agent.name = "RT_AGENT"
    agent.camera_id = 2
    agent.first_person_camera_initialized = True
    agent.position_history = []
    agent.step_num = 4
    agent.sync_settle_seconds = 0.0

    agent.sync_ue()

    assert calls.count(("get_location", "RT_AGENT")) == 1
    assert calls.count(("get_orientation", "RT_AGENT")) == 1
    assert agent.position == Vector(125.0, -75.0)
    assert agent.yaw == pytest.approx(30.0)
    assert agent.camera_location == (125.0, -75.0, 227.0)
    assert agent.camera_rotation == (-25.0, -30.0, 0.0)


def test_stationary_pose_accepts_observation_pose_without_live_queries():
    unrealcv = SimpleNamespace(
        get_location=lambda name: pytest.fail("cached location should be reused"),
        get_orientation=lambda name: pytest.fail(
            "cached orientation should be reused"
        ),
    )
    agent = make_agent(communicator=SimpleNamespace(unrealcv=unrealcv))
    agent.name = "RT_AGENT"
    agent.position = Vector(10.0, 20.0)

    location, orientation = agent._accepted_stationary_pose(
        live_location=(10.0, 20.0, 92.0),
        live_orientation=(1.0, 45.0, 2.0),
    )

    assert location == (10.0, 20.0, 92.0)
    assert orientation == (1.0, 0.0, 2.0)


def test_camera_defaults_use_balanced_pedestrian_light_profile(monkeypatch):
    for name in (
        "SIMWORLD_AGENT_CAMERA_WIDTH",
        "SIMWORLD_AGENT_CAMERA_HEIGHT",
        "SIMWORLD_AGENT_CAMERA_FOV_DEG",
        "SIMWORLD_FIRST_PERSON_CAMERA_PITCH_DEG",
    ):
        monkeypatch.delenv(name, raising=False)
    agent = make_agent()
    assert agent.camera_resolution == (1280, 720)
    assert agent.fov == 100.0
    assert agent.first_person_camera_enabled is True
    assert agent.first_person_camera_mode == "first_person_free_follow"
    assert agent.first_person_eye_height_offset_cm == 135.0
    assert agent.first_person_camera_pitch_deg == -25.0


def test_default_policy_camera_keeps_all_seven_waypoints_distinct_and_visible():
    agent = make_agent()
    agent.position = Vector(0.0, 0.0)
    agent.direction = 0.0
    pixels = project_waypoints_to_image(
        agent._find_waypoints(),
        (0.0, 0.0, 245.0),
        (-25.0, 0.0, 0.0),
        agent.fov,
        720,
        640,
    )

    assert len(pixels) == 7
    assert all(pixel is not None for pixel in pixels)
    assert len(set(pixels)) == 7
    assert all(16 <= x < 704 and 16 <= y < 624 for x, y in pixels)
    # +45-degree rays are visual-left; -45-degree rays are visual-right.
    assert pixels[2][0] < pixels[1][0]
    assert pixels[3][0] > pixels[1][0]
    assert pixels[5][0] < pixels[4][0]
    assert pixels[6][0] > pixels[4][0]


def test_level_70_degree_camera_cannot_show_the_existing_45_degree_action_fan():
    agent = make_agent()
    agent.position = Vector(0.0, 0.0)
    agent.direction = 0.0
    pixels = project_waypoints_to_image(
        agent._find_waypoints(),
        (0.0, 0.0, 245.0),
        (0.0, 0.0, 0.0),
        70.0,
        720,
        640,
    )

    assert sum(pixel is not None for pixel in pixels) < 7


def test_first_person_camera_is_spawned_and_synced_to_agent_eye_pose():
    calls = []
    camera_lists = iter(("PawnSensor ThirdPersonCmear", "PawnSensor ThirdPersonCmear CameraActor"))
    unrealcv = SimpleNamespace(
        get_cameras=lambda: next(camera_lists),
        spawn_camera=lambda: "2",
        get_location=lambda name: [100.0, 200.0, 95.0],
        get_orientation=lambda name: [0.0, 45.0, 0.0],
        set_camera_location=lambda camera_id, location: calls.append(
            ("location", camera_id, location)
        ),
        set_camera_rotation=lambda camera_id, rotation: calls.append(
            ("rotation", camera_id, rotation)
        ),
        set_camera_resolution=lambda camera_id, resolution: calls.append(
            ("resolution", camera_id, resolution)
        ),
        set_camera_fov=lambda camera_id, fov: calls.append(
            ("fov", camera_id, fov)
        ),
    )
    agent = make_agent(communicator=SimpleNamespace(unrealcv=unrealcv))
    agent.name = "RT_AGENT"

    agent.initialize_first_person_camera()

    assert agent.camera_id == 2
    assert agent.first_person_camera_initialized is True
    assert ("resolution", 2, (1280, 720)) in calls
    assert ("fov", 2, 100.0) in calls
    assert ("location", 2, (100.0, 200.0, 230.0)) in calls
    assert ("rotation", 2, (-25.0, 45.0, 0.0)) in calls
    assert agent.camera_rotation == (-25.0, -45.0, 0.0)

    forward = 400.0 / (2.0 ** 0.5)
    annotated = annotate_image(
        np.zeros((640, 720, 3), dtype=np.uint8),
        [Vector(100.0 + forward, 200.0 + forward)],
        agent.camera_location,
        agent.camera_rotation,
        agent.fov,
    )
    pixels = np.asarray(annotated)
    red_marker = (
        (pixels[:, :, 0] > 200)
        & (pixels[:, :, 1] < 80)
        & (pixels[:, :, 2] < 80)
    )
    assert int(red_marker.sum()) > 0


def test_task_camera_uses_accepted_heading_when_collision_rotates_actor():
    calls = []
    unrealcv = SimpleNamespace(
        get_location=lambda name: [120.0, -30.0, 95.0],
        get_orientation=lambda name: [0.0, -8.0, 0.0],
        set_camera_location=lambda camera_id, location: calls.append(
            ("location", camera_id, location)
        ),
        set_camera_rotation=lambda camera_id, rotation: calls.append(
            ("rotation", camera_id, rotation)
        ),
    )
    agent = make_agent(
        communicator=SimpleNamespace(unrealcv=unrealcv),
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [100.0, 0.0],
                "type": "sidewalk",
            }
        ],
    )
    agent.name = "RT_AGENT"
    agent.camera_id = 2
    agent.first_person_camera_initialized = True
    agent.position = Vector(120.0, -30.0)

    agent._sync_first_person_camera()

    assert ("location", 2, (120.0, -30.0, 230.0)) in calls
    assert ("rotation", 2, (-25.0, 0.0, 0.0)) in calls
    assert agent.camera_rotation == (-25.0, -0.0, 0.0)


def test_task_camera_uses_accepted_position_when_collision_displaces_actor():
    calls = []
    unrealcv = SimpleNamespace(
        get_location=lambda name: [250.0, -175.0, 95.0],
        get_orientation=lambda name: [0.0, 0.0, 0.0],
        set_camera_location=lambda camera_id, location: calls.append(location),
        set_camera_rotation=lambda camera_id, rotation: None,
    )
    agent = make_agent(
        communicator=SimpleNamespace(unrealcv=unrealcv),
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [100.0, 0.0],
                "type": "sidewalk",
            }
        ],
    )
    agent.name = "RT_AGENT"
    agent.camera_id = 2
    agent.first_person_camera_initialized = True
    agent.position = Vector(120.0, -30.0)

    agent._sync_first_person_camera()

    assert calls == [(120.0, -30.0, 230.0)]


def test_task_camera_keeps_spawn_height_when_collision_lifts_actor():
    calls = []
    live_location = [120.0, -30.0, 95.0]
    unrealcv = SimpleNamespace(
        get_location=lambda name: list(live_location),
        get_orientation=lambda name: [0.0, 0.0, 0.0],
        set_camera_location=lambda camera_id, location: calls.append(location),
        set_camera_rotation=lambda camera_id, rotation: None,
    )
    agent = make_agent(
        communicator=SimpleNamespace(unrealcv=unrealcv),
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [100.0, 0.0],
                "type": "sidewalk",
            }
        ],
    )
    agent.name = "RT_AGENT"
    agent.camera_id = 2
    agent.first_person_camera_initialized = True
    agent.position = Vector(120.0, -30.0)

    agent._sync_first_person_camera()
    agent._lock_first_person_actor_base_height()
    live_location[2] = 220.0
    agent._sync_first_person_camera()

    assert calls == [
        (120.0, -30.0, 230.0),
        (120.0, -30.0, 230.0),
    ]
    assert agent.first_person_actor_base_height_cm == 95.0


def test_task_camera_accepts_small_spawn_settle_before_rejecting_lift():
    calls = []
    live_location = [120.0, -30.0, 110.0]
    unrealcv = SimpleNamespace(
        get_location=lambda name: list(live_location),
        get_orientation=lambda name: [0.0, 0.0, 0.0],
        set_camera_location=lambda camera_id, location: calls.append(location),
        set_camera_rotation=lambda camera_id, rotation: None,
    )
    agent = make_agent(
        communicator=SimpleNamespace(unrealcv=unrealcv),
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [100.0, 0.0],
                "type": "sidewalk",
            }
        ],
    )
    agent.name = "RT_AGENT"
    agent.camera_id = 2
    agent.first_person_camera_initialized = True
    agent.position = Vector(120.0, -30.0)
    agent._lock_first_person_actor_base_height()

    agent._sync_first_person_camera()
    live_location[2] = 95.0
    agent._sync_first_person_camera()
    live_location[2] = 220.0
    agent._sync_first_person_camera()

    assert calls == [
        (120.0, -30.0, 245.0),
        (120.0, -30.0, 230.0),
        (120.0, -30.0, 230.0),
    ]
    assert agent.first_person_actor_base_height_cm == 95.0


def test_task_idle_pose_anchors_spawn_height_after_collision_lift():
    live_location = [10.0, 20.0, 220.0]
    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: list(live_location),
            get_orientation=lambda name: [0.0, 0.0, 0.0],
        )
    )
    agent = make_agent(
        communicator=communicator,
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [100.0, 0.0],
                "type": "sidewalk",
            }
        ],
    )
    agent.name = "RT_AGENT"
    agent.first_person_actor_base_height_cm = 95.0

    location, _ = agent._accepted_stationary_pose()

    assert location == (0.0, 0.0, 95.0)


def test_camera_profile_can_be_selected_from_environment(monkeypatch):
    monkeypatch.setenv("SIMWORLD_AGENT_CAMERA_WIDTH", "800")
    monkeypatch.setenv("SIMWORLD_AGENT_CAMERA_HEIGHT", "640")
    monkeypatch.setenv("SIMWORLD_AGENT_CAMERA_FOV_DEG", "75")
    agent = make_agent()
    assert agent.camera_resolution == (800, 640)
    assert agent.fov == 75.0


def test_visual_only_crosswalk_context_is_hidden():
    crosswalk = SimpleNamespace(
        id=7,
        start=Vector(0.0, 0.0),
        end=Vector(100.0, 0.0),
    )
    signal = SimpleNamespace(id=8, crosswalk_id=7)
    agent = make_agent(
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [100.0, 0.0],
                "type": "crosswalk",
            }
        ],
        route_crosswalks=[crosswalk],
        crosswalk_signal_groups={crosswalk.id: [signal]},
    )
    agent.route_polyline = [Vector(0.0, 0.0), Vector(100.0, 0.0)]
    agent.original_shortest_path = [Vector(100.0, 0.0)]
    agent.shortest_path = [Vector(100.0, 0.0)]
    agent.current_destination = Vector(100.0, 0.0)

    context = agent._format_active_route_context()

    assert context == ""


def test_safety_assisted_crosswalk_context_retains_action_guidance():
    crosswalk = SimpleNamespace(
        id=7,
        start=Vector(0.0, 0.0),
        end=Vector(100.0, 0.0),
    )
    signal = SimpleNamespace(id=8, crosswalk_id=7)
    agent = make_agent(
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [100.0, 0.0],
                "type": "crosswalk",
            }
        ],
        route_crosswalks=[crosswalk],
        crosswalk_signal_groups={crosswalk.id: [signal]},
        traffic_policy="safety_assisted",
    )
    agent.route_polyline = [Vector(0.0, 0.0), Vector(100.0, 0.0)]
    agent.original_shortest_path = [Vector(100.0, 0.0)]
    agent.shortest_path = [Vector(100.0, 0.0)]
    agent.current_destination = Vector(100.0, 0.0)

    context = agent._format_active_route_context()

    assert "choose one sufficient 30/60/90-degree in-place turn" in context
    assert "within 30 degrees of view center" in context
    assert "wait at the curb" in context
    assert "move monotonically" in context


def test_visual_only_uncontrolled_crosswalk_context_is_hidden():
    crosswalk = SimpleNamespace(
        id=18,
        start=Vector(0.0, 0.0),
        end=Vector(100.0, 0.0),
    )
    agent = make_agent(
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [100.0, 0.0],
                "type": "crosswalk",
            }
        ],
        route_crosswalks=[crosswalk],
        crosswalk_signal_groups={crosswalk.id: []},
    )
    agent.route_polyline = [Vector(0.0, 0.0), Vector(100.0, 0.0)]
    agent.original_shortest_path = [Vector(100.0, 0.0)]
    agent.shortest_path = [Vector(100.0, 0.0)]
    agent.current_destination = Vector(100.0, 0.0)

    context = agent._format_active_route_context()

    assert context == ""


def test_sidewalk_context_does_not_claim_a_later_crosswalk_exists():
    agent = make_agent(
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [100.0, 0.0],
                "type": "sidewalk",
            }
        ],
        traffic_policy="safety_assisted",
    )
    agent.route_polyline = [Vector(0.0, 0.0), Vector(100.0, 0.0)]
    agent.original_shortest_path = [Vector(100.0, 0.0)]
    agent.shortest_path = [Vector(100.0, 0.0)]
    agent.current_destination = Vector(100.0, 0.0)

    context = agent._format_active_route_context()

    assert "proceeding to the next route edge" in context
    assert "later crosswalk" not in context


def test_action_demo_frames_are_recorded_without_changing_vlm_input(monkeypatch):
    monkeypatch.setenv("SIMWORLD_RECORD_ACTION_FRAMES", "1")
    monkeypatch.setenv("SIMWORLD_RECORD_ACTION_CAPTURE_INTERVAL_S", "1.25")
    agent = make_agent(record_per_step=True, use_action_frames=False)
    assert agent.record_action_frames is True
    assert agent.record_action_capture_interval_s == 1.25
    assert agent.use_action_frames is False


def test_turn_anchors_accepted_pose_before_rotation():
    calls = []
    live_orientation = [0.0, 15.0, 0.0]

    def set_orientation(orientation, name):
        live_orientation[:] = orientation
        calls.append(("set_orientation", orientation, name))

    locations = iter(
        (
            [123.5, -456.25, 90.0],
            [64.0, -20.0, 90.0],
        )
    )
    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: next(locations),
            get_orientation=lambda name: list(live_orientation),
            set_location=lambda location, name: calls.append(
                ("set_location", location, name)
            ),
            set_orientation=set_orientation,
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("anchor", name, waypoint, duration)
        ),
        rt_agent_turn_around=lambda name, angle, clockwise: calls.append(
            ("turn", name, angle, clockwise)
        ),
    )
    agent = RTAgent(
        position=Vector(0, 0),
        direction=Vector(1, 0),
        destination=Vector(100, 0),
        shortest_path=[Vector(100, 0)],
        required_time=100,
        communicator=communicator,
        llm=None,
    )
    agent.name = "RT_AGENT"
    agent.use_action_frames = False
    agent.record_action_frames = False
    agent._advance_simulation_time = lambda seconds: calls.append(
        ("advance", seconds)
    )

    agent.turn_around(90, True)

    assert calls[0][0:2] == ("anchor", "RT_AGENT")
    assert calls[0][2] == Vector(0.0, 0.0)
    assert calls[0][3] == 0.05
    assert calls[1] == (
        "set_location",
        (0.0, 0.0, 90.0),
        "RT_AGENT",
    )
    assert calls[2] == (
        "set_orientation",
        (0.0, 0.0, 0.0),
        "RT_AGENT",
    )
    assert calls[3] == ("turn", "RT_AGENT", 90, True)
    assert calls[4] == ("advance", 1)
    assert calls[5] == (
        "anchor",
        "RT_AGENT",
        Vector(0.0, 0.0),
        0.05,
    )
    assert calls[6] == (
        "set_location",
        (0.0, 0.0, 90.0),
        "RT_AGENT",
    )
    # R (clockwise) raises UE yaw: a visual-right turn.
    assert calls[7] == (
        "set_orientation",
        (0.0, 90.0, 0.0),
        "RT_AGENT",
    )


def test_turn_corrects_bounded_ue_montage_translation():
    calls = []
    locations = iter(
        (
            [9300.0, -700.0, 90.0],
            [9235.928, -700.0, 90.0],
        )
    )
    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: next(locations),
            get_orientation=lambda name: [0.0, 0.0, 0.0],
            set_location=lambda location, name: calls.append(
                ("set_location", location, name)
            ),
            set_orientation=lambda orientation, name: calls.append(
                ("set_orientation", orientation, name)
            ),
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("anchor", name, waypoint, duration)
        ),
        rt_agent_turn_around=lambda name, angle, clockwise: calls.append(
            ("turn", name, angle, clockwise)
        ),
    )
    agent = make_agent(communicator=communicator)
    agent.name = "RT_AGENT"
    agent.position = Vector(9300.0, -700.0)
    agent.use_action_frames = False
    agent.record_action_frames = False
    agent._advance_simulation_time = lambda seconds: calls.append(
        ("advance", seconds)
    )

    agent.turn_around(90, True)

    assert calls[-3] == (
        "anchor",
        "RT_AGENT",
        Vector(9300.0, -700.0),
        0.05,
    )
    assert calls[-2] == (
        "set_location",
        (9300.0, -700.0, 90.0),
        "RT_AGENT",
    )
    assert calls[-1] == (
        "set_orientation",
        (0.0, 90.0, 0.0),
        "RT_AGENT",
    )
    assert agent.position == Vector(9300.0, -700.0)


def test_turn_position_lock_reanchors_every_quarter_second():
    calls = []
    live = [0.0, 0.0, 90.0]
    live_orientation = [0.0, 0.0, 0.0]

    def set_location(location, name):
        live[:] = location
        calls.append(("set_location", location, name))

    def set_orientation(orientation, name):
        live_orientation[:] = orientation
        calls.append(("set_orientation", orientation, name))

    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: list(live),
            get_orientation=lambda name: list(live_orientation),
            set_location=set_location,
            set_orientation=set_orientation,
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("anchor", name, waypoint, duration)
        ),
        rt_agent_turn_around=lambda name, angle, clockwise: calls.append(
            ("turn", name, angle, clockwise)
        ),
    )
    agent = make_agent(communicator=communicator)
    agent.name = "RT_AGENT"
    agent.position = Vector(0.0, 0.0)
    agent.task_edges = [object()]
    agent.use_action_frames = False
    agent.record_action_frames = False

    def advance(seconds):
        # This montage speed would drift 140 cm in the old 0.5-second
        # interval, but only 70 cm before each quarter-second re-anchor.
        live[0] += 280.0 * seconds
        calls.append(("advance", seconds))

    agent._advance_simulation_time = advance

    agent.turn_around(60, False)

    assert [call for call in calls if call[0] == "advance"] == [
        ("advance", 0.25),
        ("advance", 0.25),
        ("advance", 0.25),
        ("advance", 0.25),
    ]
    assert agent.last_turn_execution["raw_blueprint_position_drift_cm"] == 70.0
    assert agent.last_turn_execution["verified_position_drift_cm"] == 0.0
    assert agent.position == Vector(0.0, 0.0)


def test_terminal_vehicle_collision_during_turn_preserves_task_failure():
    calls = []
    live = [0.0, 0.0, 90.0]
    live_orientation = [0.0, 0.0, 0.0]

    def set_location(location, name):
        live[:] = location
        calls.append(("set_location", location, name))

    def set_orientation(orientation, name):
        live_orientation[:] = orientation
        calls.append(("set_orientation", orientation, name))

    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: list(live),
            get_orientation=lambda name: list(live_orientation),
            set_location=set_location,
            set_orientation=set_orientation,
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("anchor", name, waypoint, duration)
        ),
        rt_agent_turn_around=lambda name, angle, clockwise: calls.append(
            ("turn", name, angle, clockwise)
        ),
    )
    agent = make_agent(communicator=communicator)
    agent.name = "RT_AGENT"
    agent.position = Vector(0.0, 0.0)
    agent.task_edges = [object()]
    agent.use_action_frames = False
    agent.record_action_frames = False
    agent.failed = False
    agent.failure_reason = None
    agent._pending_swept_vehicle_collisions = 1

    def advance(seconds):
        live[0] = 150.0
        calls.append(("advance", seconds))

    agent._advance_simulation_time = advance

    agent.turn_around(60, False)

    assert agent._pending_swept_vehicle_collisions == 1
    assert agent.last_turn_execution["raw_blueprint_position_drift_cm"] == 150.0
    assert agent.last_turn_execution["post_turn_position_correction_applied"] is True
    assert agent.position == Vector(0.0, 0.0)


def test_right_turn_increases_ue_yaw_toward_visual_right():
    calls = []
    live = [-10700.0, -1700.0, 92.0]
    live_orientation = [0.0, 90.0, 0.0]

    def set_location(location, name):
        live[:] = location
        calls.append(("set_location", location, name))

    def set_orientation(orientation, name):
        live_orientation[:] = orientation
        calls.append(("set_orientation", orientation, name))

    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: list(live),
            set_location=set_location,
            get_orientation=lambda name: list(live_orientation),
            set_orientation=set_orientation,
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("anchor", name, waypoint, duration)
        ),
    )
    agent = make_agent(communicator=communicator)
    agent.name = "RT_AGENT"
    agent.position = Vector(-10700.0, -1700.0)
    agent.direction = 90.0

    agent._anchor_completed_turn(
        (-10700.0, -1700.0, 92.0),
        (0.0, 90.0, 0.0),
        90,
        True,
    )

    # UE is left-handed: facing +Y, the visual right is -X (yaw 180).
    assert live_orientation[1] % 360.0 == pytest.approx(180.0)
    assert agent.position == Vector(-10700.0, -1700.0)
    assert abs(agent.yaw) == pytest.approx(180.0)
    assert agent.direction.x == pytest.approx(-1.0)
    assert agent.direction.y == pytest.approx(0.0, abs=1e-8)


def test_wait_anchors_accepted_pose_before_time_advance():
    calls = []
    live_orientation = [0.0, 90.0, 0.0]

    def set_orientation(orientation, name):
        live_orientation[:] = orientation
        calls.append(("set_orientation", orientation, name))

    def advance(seconds):
        live_orientation[1] = -25.0
        calls.append(("advance", seconds))

    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: [10.0, 20.0, 30.0],
            get_orientation=lambda name: list(live_orientation),
            set_location=lambda location, name: calls.append(
                ("set_location", location, name)
            ),
            set_orientation=set_orientation,
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("anchor", name, waypoint, duration)
        ),
    )
    agent = RTAgent(
        position=Vector(0, 0),
        direction=Vector(1, 0),
        destination=Vector(100, 0),
        shortest_path=[Vector(100, 0)],
        required_time=100,
        communicator=communicator,
        llm=None,
    )
    agent.name = "RT_AGENT"
    agent.use_action_frames = False
    agent.record_action_frames = False
    agent.direction = 90.0
    agent._advance_simulation_time = advance

    agent.wait(2)

    assert calls[0][0] == "anchor"
    assert calls[0][2] == Vector(0.0, 0.0)
    assert calls[1] == ("set_location", (0.0, 0.0, 30.0), "RT_AGENT")
    assert calls[2] == (
        "set_orientation",
        (0.0, 90.0, 0.0),
        "RT_AGENT",
    )
    advance_indices = [
        index for index, call in enumerate(calls) if call[0] == "advance"
    ]
    assert len(advance_indices) == 4
    for index in advance_indices:
        assert calls[index + 1] == (
            "set_location",
            (0.0, 0.0, 30.0),
            "RT_AGENT",
        )
        assert calls[index + 2] == (
            "set_orientation",
            (0.0, 90.0, 0.0),
            "RT_AGENT",
        )
    assert live_orientation == [0.0, 90.0, 0.0]
    assert agent.position == Vector(0.0, 0.0)
    assert agent.yaw == 90.0


def test_idle_pose_uses_accepted_yaw_not_stale_live_timeline_yaw():
    calls = []
    live = [10.0, 20.0, 30.0]
    live_orientation = [5.0, -25.0, 1.0]
    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: list(live),
            get_orientation=lambda name: list(live_orientation),
            set_location=lambda location, name: calls.append(
                ("set_location", location, name)
            ),
            set_orientation=lambda orientation, name: calls.append(
                ("set_orientation", orientation, name)
            ),
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("anchor", name, waypoint, duration)
        ),
    )
    agent = make_agent(communicator=communicator)
    agent.name = "RT_AGENT"
    agent.position = Vector(0.0, 0.0)
    agent.direction = 90.0

    location, orientation = agent._accepted_stationary_pose()
    assert location == (0.0, 0.0, 30.0)
    assert orientation == (5.0, 90.0, 1.0)

    agent._cancel_residual_motion()

    assert calls[0] == (
        "anchor",
        "RT_AGENT",
        Vector(0.0, 0.0),
        0.05,
    )
    assert calls[1] == (
        "set_location",
        (0.0, 0.0, 30.0),
        "RT_AGENT",
    )
    assert calls[2] == (
        "set_orientation",
        (5.0, 90.0, 1.0),
        "RT_AGENT",
    )
    assert agent.position == Vector(0.0, 0.0)
    assert agent.yaw == 90.0


def test_move_to_corrects_small_ue_timeline_residual_at_requested_waypoint():
    calls = []
    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: [131.8, 0.0, 90.0],
            set_location=lambda location, name: calls.append(
                ("set_location", location, name)
            ),
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("move", name, waypoint, duration)
        ),
    )
    agent = make_agent(communicator=communicator)
    agent.name = "RT_AGENT"
    agent.speed = 200
    agent.record_action_frames = False
    agent._advance_simulation_time = lambda seconds: calls.append(
        ("advance", seconds)
    )

    agent.move_to(Vector(100.0, 0.0))

    assert calls[0] == ("move", "RT_AGENT", Vector(100.0, 0.0), 0.5)
    assert calls[1] == ("advance", 0.5)
    assert calls[2] == ("move", "RT_AGENT", Vector(100.0, 0.0), 0.05)
    assert calls[3] == (
        "set_location",
        (100.0, 0.0, 90.0),
        "RT_AGENT",
    )
    assert agent.position == Vector(100.0, 0.0)


def test_off_route_waypoint_uses_physical_ue_movement_without_teleport():
    calls = []
    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            set_location=lambda location, name: calls.append(
                ("set_location", location, name)
            ),
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("move", name, waypoint, duration)
        ),
    )
    agent = RTAgent(
        position=Vector(0.0, 0.0),
        direction=Vector(1.0, 0.0),
        destination=Vector(1000.0, 0.0),
        shortest_path=[Vector(1000.0, 0.0)],
        required_time=100,
        communicator=communicator,
        llm=None,
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [1000.0, 0.0],
                "type": "sidewalk",
            }
        ],
    )
    agent.name = "RT_AGENT"
    agent.speed = 200
    agent.record_action_frames = False
    agent._advance_simulation_time = lambda seconds: calls.append(
        ("advance", seconds)
    )
    selected = Vector(0.0, 400.0)

    agent.move_to(selected)

    assert calls == [
        ("move", "RT_AGENT", selected, 2.0),
        ("advance", 2.0),
    ]
    assert agent.last_move_execution_mode == "ue_physical_off_route"
    assert agent.position == Vector(0.0, 0.0)

def test_ordered_edge_move_progresses_monotonically_on_centerline():
    calls = []
    live = [0.0, 0.0, 90.0]
    live_orientation = [0.0, 145.0, 0.0]

    def set_location(location, name):
        live[:] = location
        calls.append(("set_location", location, name))

    def set_orientation(orientation, name):
        live_orientation[:] = orientation
        calls.append(("set_orientation", orientation, name))

    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: list(live),
            set_location=set_location,
            get_orientation=lambda name: list(live_orientation),
            set_orientation=set_orientation,
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("move", name, waypoint, duration)
        ),
    )
    agent = RTAgent(
        position=Vector(0.0, 0.0),
        direction=Vector(0.0, 1.0),
        destination=Vector(0.0, 400.0),
        shortest_path=[Vector(0.0, 400.0)],
        required_time=100,
        communicator=communicator,
        llm=None,
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [0.0, 400.0],
                "type": "crosswalk",
            }
        ],
    )
    agent.name = "RT_AGENT"
    agent.speed = 200
    agent.record_action_frames = False
    agent._advance_simulation_time = lambda seconds: calls.append(
        ("advance", seconds)
    )

    agent.move_to(Vector(0.0, 400.0))

    positions = [
        call[1]
        for call in calls
        if call[0] == "set_location"
    ]
    assert positions
    assert all(position[0] == 0.0 for position in positions)
    assert [position[1] for position in positions] == sorted(
        position[1] for position in positions
    )
    assert positions[-1] == (0.0, 400.0, 90.0)
    assert agent.position == Vector(0.0, 400.0)
    assert ("set_orientation", (0.0, 90.0, 0.0), "RT_AGENT") in calls
    assert [
        call for call in calls if call[0] == "set_orientation"
    ][-1] == (
        "set_orientation",
        (0.0, 90.0, 0.0),
        "RT_AGENT",
    )
    assert agent.yaw == 90.0


def test_ordered_edge_move_reasserts_commanded_yaw_after_every_simulation_tick():
    calls = []
    live = [0.0, 0.0, 90.0]
    live_orientation = [0.0, 145.0, 0.0]

    def set_location(location, name):
        live[:] = location
        calls.append(("set_location", location, name))

    def set_orientation(orientation, name):
        live_orientation[:] = orientation
        calls.append(("set_orientation", orientation, name))

    def advance(seconds):
        # Reproduce the stale Blueprint MoveTo yaw write observed in the real
        # rollout immediately after each simulation tick.
        live_orientation[1] = -90.0
        calls.append(("advance", seconds))

    def move(name, waypoint, duration):
        # The zero-length cancellation at completion can restore the same
        # stale yaw as well; the final authoritative pose must overwrite it.
        live_orientation[1] = -90.0
        calls.append(("move", name, waypoint, duration))

    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: list(live),
            set_location=set_location,
            get_orientation=lambda name: list(live_orientation),
            set_orientation=set_orientation,
        ),
        rt_agent_move_to=move,
    )
    agent = RTAgent(
        position=Vector(0.0, 0.0),
        direction=Vector(0.0, 1.0),
        destination=Vector(0.0, 200.0),
        shortest_path=[Vector(0.0, 200.0)],
        required_time=100,
        communicator=communicator,
        llm=None,
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [0.0, 200.0],
                "type": "crosswalk",
            }
        ],
    )
    agent.name = "RT_AGENT"
    agent.speed = 200
    agent.record_action_frames = False
    agent._advance_simulation_time = advance

    agent.move_to(Vector(0.0, 200.0))

    advance_indices = [
        index for index, call in enumerate(calls) if call[0] == "advance"
    ]
    assert advance_indices
    for index in advance_indices:
        next_advance = next(
            (
                candidate
                for candidate in advance_indices
                if candidate > index
            ),
            len(calls),
        )
        assert (
            "set_orientation",
            (0.0, 90.0, 0.0),
            "RT_AGENT",
        ) in calls[index + 1:next_advance]
    assert live_orientation == [0.0, 90.0, 0.0]
    assert agent.yaw == 90.0


def test_ordered_edge_anchor_uses_commanded_yaw_not_stale_live_yaw():
    calls = []
    live_orientation = [5.0, -35.0, 1.0]

    def move(name, waypoint, duration):
        live_orientation[1] = -120.0
        calls.append(("move", name, waypoint, duration))

    def set_orientation(orientation, name):
        live_orientation[:] = orientation
        calls.append(("set_orientation", orientation, name))

    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: [0.0, 200.0, 90.0],
            set_location=lambda location, name: calls.append(
                ("set_location", location, name)
            ),
            get_orientation=lambda name: list(live_orientation),
            set_orientation=set_orientation,
        ),
        rt_agent_move_to=move,
    )
    agent = RTAgent(
        position=Vector(0.0, 200.0),
        direction=Vector(0.0, 1.0),
        destination=Vector(0.0, 400.0),
        shortest_path=[Vector(0.0, 400.0)],
        required_time=100,
        communicator=communicator,
        llm=None,
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [0.0, 400.0],
                "type": "crosswalk",
            }
        ],
    )
    agent.name = "RT_AGENT"

    agent._anchor_completed_move(Vector(0.0, 0.0), Vector(0.0, 200.0))

    assert calls[-1] == (
        "set_orientation",
        (5.0, 90.0, 1.0),
        "RT_AGENT",
    )
    assert live_orientation == [5.0, 90.0, 1.0]
    assert agent.yaw == 90.0


def test_move_to_fails_closed_instead_of_hiding_large_execution_miss():
    calls = []
    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: [300.0, 0.0, 90.0],
            set_location=lambda location, name: calls.append(
                ("set_location", location, name)
            ),
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("move", name, waypoint, duration)
        ),
    )
    agent = make_agent(communicator=communicator)
    agent.name = "RT_AGENT"
    agent.speed = 200
    agent.record_action_frames = False
    agent._advance_simulation_time = lambda seconds: None

    with pytest.raises(RuntimeError, match="fail-closed limit"):
        agent.move_to(Vector(100.0, 0.0))

    assert not [call for call in calls if call[0] == "set_location"]


def test_move_to_corrects_bounded_sideways_execution_drift():
    calls = []
    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: [100.0, 30.0, 90.0],
            set_location=lambda location, name: calls.append(
                ("set_location", location, name)
            ),
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("move", name, waypoint, duration)
        ),
    )
    agent = make_agent(communicator=communicator)
    agent.name = "RT_AGENT"
    agent.speed = 200
    agent.record_action_frames = False
    agent._advance_simulation_time = lambda seconds: None

    agent.move_to(Vector(100.0, 0.0))

    assert [call for call in calls if call[0] == "set_location"] == [
        ("set_location", (100.0, 0.0, 90.0), "RT_AGENT")
    ]


def test_move_to_fails_closed_on_large_sideways_execution_drift():
    calls = []
    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: [100.0, 60.0, 90.0],
            set_location=lambda location, name: calls.append(
                ("set_location", location, name)
            ),
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("move", name, waypoint, duration)
        ),
    )
    agent = make_agent(communicator=communicator)
    agent.name = "RT_AGENT"
    agent.speed = 200
    agent.record_action_frames = False
    agent._advance_simulation_time = lambda seconds: None

    with pytest.raises(RuntimeError, match="lateral=60.0cm"):
        agent.move_to(Vector(100.0, 0.0))

    assert not [call for call in calls if call[0] == "set_location"]


def test_long_ordered_crosswalk_move_corrects_legal_navigation_deflection():
    calls = []
    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: [92.6, 313.4, 90.0],
            set_location=lambda location, name: calls.append(
                ("set_location", location, name)
            ),
            get_orientation=lambda name: [0.0, 90.0, 0.0],
            set_orientation=lambda orientation, name: calls.append(
                ("set_orientation", orientation, name)
            ),
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("move", name, waypoint, duration)
        ),
    )
    agent = RTAgent(
        position=Vector(0.0, 0.0),
        direction=Vector(0.0, 1.0),
        destination=Vector(0.0, 400.0),
        shortest_path=[Vector(0.0, 400.0)],
        required_time=100,
        communicator=communicator,
        llm=None,
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [0.0, 400.0],
                "type": "crosswalk",
            }
        ],
    )
    agent.name = "RT_AGENT"
    agent.speed = 200
    agent.record_action_frames = False
    agent._advance_simulation_time = lambda seconds: None

    agent._anchor_completed_move(Vector(0.0, 0.0), Vector(0.0, 400.0))

    assert [call for call in calls if call[0] == "set_location"] == [
        ("set_location", (0.0, 400.0, 90.0), "RT_AGENT")
    ]


def test_long_ordered_crosswalk_move_rejects_excessive_lateral_deflection():
    calls = []
    communicator = SimpleNamespace(
        unrealcv=SimpleNamespace(
            get_location=lambda name: [180.0, 300.0, 90.0],
            set_location=lambda location, name: calls.append(
                ("set_location", location, name)
            ),
        ),
        rt_agent_move_to=lambda name, waypoint, duration: calls.append(
            ("move", name, waypoint, duration)
        ),
    )
    agent = RTAgent(
        position=Vector(0.0, 0.0),
        direction=Vector(0.0, 1.0),
        destination=Vector(0.0, 400.0),
        shortest_path=[Vector(0.0, 400.0)],
        required_time=100,
        communicator=communicator,
        llm=None,
        task_edges=[
            {
                "node1": [0.0, 0.0],
                "node2": [0.0, 400.0],
                "type": "crosswalk",
            }
        ],
    )
    agent.name = "RT_AGENT"
    agent.speed = 200
    agent.record_action_frames = False
    agent._advance_simulation_time = lambda seconds: None

    with pytest.raises(RuntimeError, match="lateral=180.0cm"):
        agent._anchor_completed_move(Vector(0.0, 0.0), Vector(0.0, 400.0))

    assert not [call for call in calls if call[0] == "set_location"]


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("SIMWORLD_AGENT_CAMERA_WIDTH", "wide"),
        ("SIMWORLD_AGENT_CAMERA_HEIGHT", "100"),
        ("SIMWORLD_AGENT_CAMERA_FOV_DEG", "nan"),
        ("SIMWORLD_AGENT_CAMERA_FOV_DEG", "20"),
    ),
)
def test_invalid_camera_profile_fails_closed(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        make_agent()
