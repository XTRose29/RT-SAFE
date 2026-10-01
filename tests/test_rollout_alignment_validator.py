from evaluation.validate_rollout_waypoint_alignment import (
    action_was_executed,
    expected_move_duration_seconds,
    has_safety_explanation,
    retained_history_source_frames,
)


def test_move_duration_accounts_for_one_shot_oil_slowdown():
    assert expected_move_duration_seconds(200.0) == 1.0
    assert expected_move_duration_seconds(
        200.0,
        oil_slowdown_pending=True,
    ) == 2.0


def test_no_progress_evidence_explains_a_blocked_move():
    assert has_safety_explanation({
        "collision_details": {"off_route_move_no_progress": 1},
        "feedback": "The movement command made no progress.",
    })


def test_rejected_action_is_not_treated_as_executed():
    assert not action_was_executed({
        "feedback": "Invalid action (not executed)",
    })
    assert not action_was_executed({
        "feedback": "Parse error (no executable action)",
    })
    assert action_was_executed({
        "feedback": "Action completed successfully.",
    })


def test_sparse_retention_marks_missing_history_source_unverifiable():
    assert retained_history_source_frames([], ["history.png"]) is None
    assert retained_history_source_frames(
        ["older.png", "latest.png"],
        ["history.png"],
    ) == ["latest.png"]
