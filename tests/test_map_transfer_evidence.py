import hashlib
import json
import pytest

from benchmark.map_transfer.contract import EVENT_TYPES
from benchmark.map_transfer.evidence import verify_run


def test_saved_run_verification_rejects_count_and_image_tampering(tmp_path):
    counts = {kind: int(kind == "collision") for kind in EVENT_TYPES}
    state = {
        "events": [
            {
                "index": 0,
                "type": "collision",
                "phase": "inference",
                "simulation_time": 1.0,
                "evidence": {"source": "unreal_blocking_hit", "penetrating": False},
            }
        ],
        "counts": counts,
        "goal_reached": True,
        "simulation_time": 2.0,
        "errors": [],
        "terminal_reason": "goal_reached",
    }
    summary = {
        "pilot": True,
        "task_id": "check",
        "mode": "realtime",
        "safety_events": counts,
        "active_collisions": 0,
        "passive_collisions": 1,
        "success": True,
        "safe_success": False,
        "runtime_errors": [],
        "terminal_reason": "goal_reached",
        "decisions": 0,
    }

    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))

    write("summary.json", summary)
    write("final-state.json", state)
    write("provenance.json", {"pilot": True, "task_id": "check", "mode": "realtime"})
    write(
        "trajectory.json",
        {
            "samples": [
                {"simulation_time": 0, "agent_position_cm": [0, 0, 88]},
                {"simulation_time": 2, "agent_position_cm": [100, 0, 88]},
            ]
        },
    )
    image = b"fixture image bytes"
    (tmp_path / "observation-000-raw.jpg").write_bytes(image)
    (tmp_path / "observation-000.png").write_bytes(image)
    write(
        "observation-000.json",
        {
            "rgb_sha256": hashlib.sha256(image).hexdigest(),
            "policy_image_sha256": hashlib.sha256(image).hexdigest(),
        },
    )
    assert verify_run(tmp_path)["events"] == 1
    corrupted = {**summary, "safe_success": True}
    write("summary.json", corrupted)
    with pytest.raises(ValueError, match="Safe success"):
        verify_run(tmp_path)
    write("summary.json", {**summary, "passive_collisions": 0})
    with pytest.raises(ValueError, match="Passive count"):
        verify_run(tmp_path)
    write("summary.json", summary)
    (tmp_path / "observation-000.png").write_bytes(b"different image")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_run(tmp_path)
