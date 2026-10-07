from pathlib import Path
import subprocess
import json
from types import SimpleNamespace

import pytest

from evaluation import run_code_baselines as baselines
from tools import export_realtime_release as release
from tools.run_realtime_examples import compare


@pytest.mark.parametrize("status,expected", [
    ("completed", 0), ("skipped_existing", 0), ("error", 1),
    ("skipped_existing_error", 1), ("skipped_existing_completed", 1),
])
def test_baseline_exit_reflects_condition_error(monkeypatch, tmp_path, status, expected):
    monkeypatch.setattr("sys.argv", ["run_code_baselines.py", "--baselines", "greedy",
                                    "--env-modes", "static", "--output-root", str(tmp_path),
                                    "--suite-name", "test"])
    monkeypatch.setattr(baselines, "run_condition", lambda *args: {"status": status})
    assert baselines.main() == expected


def test_example_comparison_rejects_missing_or_incorrect_results():
    expected = {"decision_count": 2, "sim_time": 6.0, "success": False}
    assert compare(dict(expected), expected) == {}
    assert set(compare({"decision_count": 1, "sim_time": 4}, expected)) == set(expected)
    assert "success" in compare({**expected, "success": 0}, expected)
    assert compare({}, {"rollout_error": None})["rollout_error"]["missing"]


def test_saved_rollout_error_is_not_reported_as_completed(monkeypatch, tmp_path):
    monkeypatch.setenv("CODE_BASELINE_THINKING_SECONDS", "0")
    monkeypatch.setenv("SIMWORLD_UNIFORM_GREEDY_MOVEMENT", "0")
    monkeypatch.setattr("sys.argv", ["run_code_baselines.py", "--baselines", "greedy",
                                    "--env-modes", "static", "--output-root", str(tmp_path),
                                    "--suite-name", "failed"])
    monkeypatch.setattr("base.rt_unrealcv.RTUnrealCV", lambda **kwargs: object())
    monkeypatch.setattr("base.rt_communicator.RTCommunicator", lambda ue: object())

    def manager(**kwargs):
        output = baselines.REPO_ROOT / kwargs["results_dir"]
        def run(*args, **kwargs):
            (output / "task_0_test.json").write_text(json.dumps({
                "decision_count": 2, "rollout_error": "simulator connection lost"}))
        return SimpleNamespace(run_single_task=run, cleanup=lambda: None)

    monkeypatch.setattr("manager.world_manager.WorldManager", manager)
    assert baselines.main() == 1
    record = json.loads((tmp_path / "failed/summary.jsonl").read_text())
    assert record["status"] == "error"
    assert "simulator connection lost" in record["error"]


def test_source_export_uses_allowlist_and_ignores_local_files(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    contents = {
        "base/runtime.py": "# original benchmark\n",
        "benchmark/map_transfer/client.py": "# not released\n",
        "tests/test_map_transfer_contract.py": "# not released\n",
        "website/public/nyc.jpg": "not released",
        "website/app/page.tsx": "not released",
        "results/local.json": "not released",
        ".env": "not released",
        "release/realtime/README.md": "# portable source\n",
    }
    for name, content in contents.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], cwd=repo, check=True)
    (repo / "base/local-credentials.txt").write_text("untracked")
    (repo / "release/realtime/local-credentials.txt").write_text("untracked")
    monkeypatch.setattr(release, "ROOT", repo)
    output = tmp_path / "export"
    record = release.export(output)
    assert set(record["files"]) == {"base/runtime.py", "README.md", "release/realtime/README.md"}
    assert record["source_has_local_changes"]  # untracked files recorded, never exported
    assert (output / "SOURCE_MANIFEST.json").exists()
    with pytest.raises(ValueError, match="new directory"):
        release.export(output)
