import json

import pytest

from online_rl.launch_ue_fleet import build_command, parse_args
from online_rl.preflight import validate_gpu_partition


def test_fleet_command_matches_packaged_benchmark_flags(tmp_path):
    launcher = tmp_path / "SimWorld.sh"
    launcher.write_text("#!/bin/sh\n")
    command = build_command(
        launcher,
        map_name="RT10",
        gpu=3,
        port=9010,
        width=1280,
        height=720,
        fps=30,
        internal_log=tmp_path / "internal.log",
    )
    assert command[:3] == [str(launcher.resolve()), "-RenderoffScreen", "RT10"]
    assert "-graphicsadapter=3" in command
    assert command[command.index("-cvport") + 1] == "9010"
    assert "-noraytracing" in command
    assert "-norhithread" in command


def test_fleet_cli_accepts_parallel_gpu_port_lists(tmp_path):
    launcher = tmp_path / "SimWorld.sh"
    launcher.write_text("#!/bin/sh\n")
    args = parse_args(
        [
            "--launcher", str(launcher),
            "--gpus", "0", "2",
            "--ports", "9010", "9030",
            "--output", str(tmp_path / "endpoints.json"),
            "--logs", str(tmp_path / "logs"),
            "--dry-run",
        ]
    )
    assert args.gpus == [0, 2]
    assert args.ports == [9010, 9030]


def test_preflight_requires_disjoint_ue_and_training_gpus(tmp_path, monkeypatch):
    endpoints = tmp_path / "endpoints.json"
    endpoints.write_text(
        json.dumps(
            {
                "status": "ready",
                "instances": [
                    {"id": "a", "host": "127.0.0.1", "port": 9010, "gpu": 0},
                    {"id": "b", "host": "127.0.0.1", "port": 9020, "gpu": 1},
                ],
            }
        )
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert "disjoint" in validate_gpu_partition(str(endpoints))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,5")
    with pytest.raises(RuntimeError, match="overlap"):
        validate_gpu_partition(str(endpoints))
