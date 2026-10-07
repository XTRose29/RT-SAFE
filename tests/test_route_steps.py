from utils.route_steps import route_length_to_max_steps


def test_default_step_budget_is_three_times_full_shortest_route_meters():
    assert route_length_to_max_steps(0.0) == 1
    assert route_length_to_max_steps(99.9) == 1
    assert route_length_to_max_steps(100.0) == 3
    assert route_length_to_max_steps(199.9) == 3
    assert route_length_to_max_steps(200.0) == 6
