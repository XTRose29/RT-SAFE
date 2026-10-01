import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SimWorld"))

from base.rt_agent import (
    HAZARD_OCCUPANCY_REARM_DISTANCE_CM,
    RTAgent,
)
from simworld.utils.vector import Vector


def make_route_agent(position=Vector(100.0, 0.0)):
    agent = RTAgent.__new__(RTAgent)
    start = Vector(0.0, 0.0)
    end = Vector(1000.0, 0.0)
    agent.position = Vector(position.x, position.y)
    agent._direction = Vector(1.0, 0.0)
    agent._yaw = 0.0
    agent.current_destination = end
    agent.route_polyline = [start, end]
    agent.original_shortest_path = [end]
    agent.shortest_path = [end]
    agent.task_edges = [
        {
            "node1": [start.x, start.y],
            "node2": [end.x, end.y],
            "type": "sidewalk",
        }
    ]
    agent.route_crosswalks = []
    agent.stagnation_count = 0
    agent.max_stagnation_count = 0
    agent.stagnation_events = []
    agent._stagnation_last_position = Vector(position.x, position.y)
    agent._stagnation_last_metric = None
    agent._stagnation_last_destination = None
    agent.last_action_type = "WAIT"
    agent.last_action_override = None
    agent.last_execution_traffic_light_snapshot = None
    agent.current_traffic_light_snapshot = {}
    agent.success = False
    agent.failed = False
    agent.failure_reason = None
    agent.stuck = False
    agent.stuck_reason = None
    agent.step_num = 0
    agent.decision_count = 0
    return agent


def test_continuous_hazard_overlap_counts_once_then_rearms_after_exit():
    agent = make_route_agent()
    details = {}

    assert agent._begin_hazard_overlap_episode(2, details) is True
    assert details["hazard_overlap_new_episode"] == 1

    repeated = {}
    assert agent._begin_hazard_overlap_episode(2, repeated) is False
    assert repeated["hazard_overlap_suppressed"] == 1
    assert agent.hazard_overlap_raw_counts["oil"] == 2
    assert agent.hazard_overlap_suppressed_counts["oil"] == 1

    agent.position = Vector(
        agent.position.x + HAZARD_OCCUPANCY_REARM_DISTANCE_CM + 1.0,
        agent.position.y,
    )
    assert agent._begin_hazard_overlap_episode(0, {}) is False
    assert agent._begin_hazard_overlap_episode(2, {}) is True


def test_off_route_context_does_not_claim_impossible_waypoint_is_on_edge():
    agent = make_route_agent(Vector(100.0, 1200.0))
    agent.traffic_policy = 'safety_assisted'
    waypoints = [
        Vector(200.0, 1200.0),
        Vector(300.0, 1200.0),
        Vector(500.0, 1200.0),
    ]

    context = agent._format_active_route_context(waypoints)

    assert "OFF-ROUTE RECOVERY" in context
    assert "none of the seven displayed waypoint endpoints returns" in context
    assert "Do not claim that a displayed marker lies on the active edge" in context
    assert "select only a point that remains on this active edge" not in context


def test_stagnation_does_not_add_recovery_guidance_to_route_context():
    agent = make_route_agent(Vector(100.0, 500.0))
    agent.traffic_policy = 'safety_assisted'
    agent.stagnation_count = 11

    context = agent._format_active_route_context(
        [Vector(200.0, 500.0), Vector(300.0, 500.0)]
    )

    assert "STAGNATION RECOVERY" not in context
    assert "no measurable route progress" not in context


def test_stagnation_is_telemetry_only_and_never_terminates():
    agent = make_route_agent()
    agent._update_stagnation_state()

    for decision in range(25):
        agent.decision_count = decision + 1
        update = agent._update_stagnation_state()
        assert update["terminal"] is False

    assert agent.stagnation_count == 25
    assert agent.max_stagnation_count == 25
    assert agent.failed is False
    assert agent.failure_reason is None
    assert agent.stuck is False
    assert agent.stagnation_events == []


def test_signal_required_wait_is_exempt_from_stagnation():
    agent = make_route_agent()
    agent._update_stagnation_state()
    agent.stagnation_count = 3
    agent.last_action_override = {"reason": "pedestrian_signal_not_walk"}

    update = agent._update_stagnation_state()

    assert update["signal_required_wait"] is True
    assert agent.stagnation_count == 0
    assert agent.failed is False


def test_authored_sidewalk_boundary_matches_rendered_four_meter_strip():
    agent = RTAgent.__new__(RTAgent)
    agent.traffic_sidewalks = [
        SimpleNamespace(start=Vector(0.0, 0.0), end=Vector(0.0, 1000.0))
    ]

    assert agent._is_within_authored_sidewalk(Vector(199.9, 500.0)) is True
    assert agent._is_within_authored_sidewalk(Vector(200.1, 500.0)) is False
