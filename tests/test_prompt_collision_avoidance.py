from pathlib import Path

from llm.prompt import (
    USER_PROMPT,
    append_passive_collision_feedback,
    append_traffic_rule_violation_feedback,
    get_image_description,
    get_system_prompt,
)


def test_visual_only_prompt_has_no_tactical_action_guidance():
    for tactical_guidance in (
        'actual corridor from your feet to each candidate waypoint',
        'an actor elsewhere in the image is not by itself a reason',
        'only when every useful route-aligned waypoint is blocked',
        'an actor is visibly\nconverging on its corridor',
        'proceed as soon\nas a clear route-aligned waypoint is available',
        'Collision avoidance has absolute priority',
        'could enter it before the move finishes',
    ):
        assert tactical_guidance not in USER_PROMPT


def test_prompt_defines_traffic_rules_without_prescribing_actions():
    traffic_rules = USER_PROMPT.split(
        'Traffic-rule definitions:\n', 1
    )[1].split(
        'Output your decision in the required format above.', 1
    )[0]

    assert 'Roadway entry is legal only within a marked crosswalk' in traffic_rules
    assert 'pedestrian signal shows WALK is legal' in traffic_rules
    assert 'vehicle traffic lights never authorize pedestrian entry' in traffic_rules
    assert 'walking-person icon is the pedestrian signal' in traffic_rules
    assert "Flashing DON'T WALK" in traffic_rules
    assert (
        "WALK changes to FLASHING DON'T WALK or steady DON'T WALK after entry"
        in traffic_rules
    )
    assert 'continuing to the far\n  curb is still legal' in traffic_rules
    assert (
        "Starting a crossing during FLASHING DON'T WALK or steady DON'T WALK"
        in traffic_rules
    )
    assert 'Roadway occupancy outside a marked crosswalk' in traffic_rules
    assert 'Remaining inside the crossing after WALK ends' not in traffic_rules
    for prescribed_action in ('wait', 'move', 'turn', 'proceed'):
        assert prescribed_action not in traffic_rules.lower()


def test_feedback_reports_only_violations_observed_during_the_action():
    feedback = append_traffic_rule_violation_feedback(
        'Action completed successfully.',
        [{'event_id': 'red-light-0001'}],
        [{'event_id': 'illegal-crossing-0001'}],
    )

    assert feedback == 'Traffic-rule violation recorded.'
    assert 'vehicle' not in feedback.lower()


def test_feedback_is_unchanged_without_a_new_traffic_violation():
    assert append_traffic_rule_violation_feedback(
        'Action completed successfully.', [], []
    ) == 'Action completed successfully.'


def test_feedback_does_not_reveal_traffic_violation_details():
    feedback = append_traffic_rule_violation_feedback(
        'Action completed successfully.',
        [{
            'event_id': 'red-light-0001',
            'violation_reason': 'entered_without_walk',
        }],
        [],
    )

    assert feedback == 'Traffic-rule violation recorded.'


def test_feedback_reports_collisions_that_happened_during_inference():
    assert append_passive_collision_feedback(
        'Action completed successfully.',
        {'human': 2, 'object': 1, 'building': 0, 'vehicle': 0},
    ) == 'While you were deciding, you collided with a pedestrian and an object.'


def test_passive_collision_feedback_ignores_vehicle_launch_metadata():
    assert append_passive_collision_feedback(
        'Action completed successfully.',
        {'human': 0, 'object': 0, 'building': 0, 'vehicle': 1},
    ) == 'Action completed successfully.'


def test_live_agent_step_applies_passive_collision_feedback():
    source = (
        Path(__file__).resolve().parents[1] / "base" / "rt_agent.py"
    ).read_text(encoding="utf-8")
    expected_call = (
        "feedback = append_passive_collision_feedback(\n"
        "            feedback,\n"
        "            passive_collision_details,\n"
        "        )"
    )
    assert expected_call in source


def test_system_prompt_describes_benchmark_clean_ray_waypoints():
    prompt = get_system_prompt(realtime_thinking=True)
    assert 'when they fall inside the first-person camera frame' in prompt
    assert 'Point 1: 100cm / 0 degrees' in prompt
    assert 'Points 2-4: 200cm (0 degrees / +/-45 degrees)' in prompt
    assert 'Points 5-7: 400cm (0 degrees / +/-45 degrees)' in prompt
    assert 'active ordered route centerline' not in prompt


def test_image_description_discloses_first_person_projection_clipping():
    description = get_image_description(False, is_first_step=False)
    assert 'all seven candidate waypoint projections marked in red' in description
    assert 'exact world target executed by move_to n' in description
    assert 'fixed 1-7 mapping' in description


def test_action_frame_description_discloses_half_second_history():
    description = get_image_description(True, is_first_step=False)
    assert 'approximately every 0.5s' in description
    assert 'last image is your current view' in description
