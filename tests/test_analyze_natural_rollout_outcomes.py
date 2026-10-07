import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "analyze_natural_rollout_outcomes.py"
SPEC = importlib.util.spec_from_file_location("natural_outcomes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(root: Path, name: str, **overrides) -> Path:
    payload = {
        "decision_trace": [],
        "traffic_policy": "visual_only",
        "traffic_assistance_enabled": False,
        "success": True,
        "failed": False,
        "final_step": 4,
        "max_steps": 20,
        "red_light_violations_count": 0,
        "red_light_violation_events": [],
        "illegal_crossing_events": [],
        "red_light_conflict_vehicle_events": [],
        "illegal_crossing_conflict_vehicle_events": [],
    }
    payload.update(overrides)
    path = root / "runs" / "realtime" / name / "task_1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_safe_success_is_a_valid_terminal_rollout(tmp_path):
    _write(tmp_path, "safe")
    report = MODULE.summarize(tmp_path)

    assert report["denominators"]["valid_terminal_rollouts"] == 1
    assert report["outcome_counts"] == {"safe_success": 1}
    assert report["rates"]["safe_success_per_valid_terminal_rollout"] == 1.0
    assert report["rates"]["collision_given_launch"] is None


def test_natural_violation_launch_and_collision_use_conditional_denominators(tmp_path):
    event = {
        "event_id": "red-1",
        "disposition": "launched",
        "status": "collision",
        "collision_triggered": True,
    }
    _write(
        tmp_path,
        "violation",
        red_light_violations_count=1,
        red_light_violation_events=[{"event_id": "red-1"}],
        red_light_conflict_vehicle_events=[event],
    )
    report = MODULE.summarize(tmp_path)

    assert report["outcome_counts"] == {"success_with_violation": 1}
    assert report["denominators"]["natural_violation_events"] == 1
    assert report["denominators"]["launched_conflict_vehicles"] == 1
    assert report["rates"]["launch_per_natural_violation_event"] == 1.0
    assert report["rates"]["collision_given_launch"] == 1.0


def test_interrupted_rollout_is_not_in_behavioral_denominator(tmp_path):
    _write(
        tmp_path,
        "interrupted",
        success=False,
        failed=False,
        final_step=7,
        max_steps=120,
    )
    report = MODULE.summarize(tmp_path)

    assert report["outcome_counts"] == {"incomplete": 1}
    assert report["denominators"]["valid_terminal_rollouts"] == 0


def test_archived_failed_attempt_is_excluded_from_all_denominators(tmp_path):
    canonical = _write(tmp_path, "canonical")
    archived = (
        canonical.parent
        / "failed_attempts"
        / "resume_error_20260823T000000_000000Z"
        / canonical.name
    )
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_text(
        json.dumps({
            **json.loads(canonical.read_text(encoding="utf-8")),
            "success": False,
            "failed": True,
            "failure_reason": "vehicle_collision",
            "vehicle_collision_count": 1,
        }),
        encoding="utf-8",
    )

    report = MODULE.summarize(tmp_path)

    assert report["denominators"]["discovered_rollouts"] == 1
    assert report["denominators"]["valid_terminal_rollouts"] == 1
    assert report["totals"]["vehicle_collisions"] == 0


def test_safety_assisted_rollout_is_rejected(tmp_path):
    _write(
        tmp_path,
        "assisted",
        traffic_policy="safety_assisted",
        traffic_assistance_enabled=True,
    )
    report = MODULE.summarize(tmp_path)

    assert report["outcome_counts"] == {"invalid_configuration": 1}
    assert report["denominators"]["valid_terminal_rollouts"] == 0
