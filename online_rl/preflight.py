#!/usr/bin/env python3
"""Fail-fast checks for the SimWorld-RealTime/VAGEN online-RL run kit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PATCHES = (
    "agent_loop_qwen3vl_rope",
    "agent_loop_image_safe_truncation",
    "agent_loop_generation_budget",
    "policy_loss_rollout_correction_field",
    "multiturn_image_safe_truncation",
    "silent_zero_scores",
)


@dataclass
class Check:
    label: str
    ok: bool
    detail: str


def check(label: str, operation: Callable[[], str]) -> Check:
    try:
        return Check(label, True, operation())
    except Exception as exc:  # noqa: BLE001 - every failure should be reported
        return Check(label, False, str(exc))


def require_path(value: str | None, label: str, relative: str | None = None) -> str:
    if not value:
        raise RuntimeError(f"{label} is unset")
    path = Path(value).expanduser().resolve()
    target = path / relative if relative else path
    if not target.exists():
        raise RuntimeError(f"missing {target}")
    return str(path)


def require_module(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise RuntimeError(f"cannot import {name!r} with {sys.executable}")
    return str(spec.origin or spec.submodule_search_locations)


def validate_benchmark_files() -> str:
    paths = (
        REPO / "data/map1_10roads/tasks.json",
        REPO / "data/agents.json",
        REPO / "evaluation/run_qwen3vl8b_all_maps.py",
    )
    missing = [str(path.relative_to(REPO)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing: {', '.join(missing)}")
    return ", ".join(str(path.relative_to(REPO)) for path in paths)


def validate_patches(nav_root: str | None) -> str:
    root = Path(require_path(nav_root, "SIMWORLD_NAV_ROOT"))
    patch_dir = root / "embodiedbench/training/vagen/patches"
    missing = [name for name in PATCHES if not (patch_dir / f"{name}.py").is_file()]
    if missing:
        raise RuntimeError(f"missing PR #3 patch modules: {', '.join(missing)}")
    return f"{root} ({len(PATCHES)}/{len(PATCHES)} patch modules)"


def validate_checkout_layout(vagen_root: str | None, nav_root: str | None) -> str:
    vagen = Path(require_path(vagen_root, "VAGEN_ROOT", "vagen/main_ppo.py"))
    nav = Path(require_path(nav_root, "SIMWORLD_NAV_ROOT"))
    expected = (nav / "vendor/vagen").resolve()
    if vagen != expected:
        raise RuntimeError(
            f"VAGEN_ROOT must be {expected}; PR #3 patches modify that vendored checkout, "
            f"not {vagen}"
        )
    if not (vagen / "verl/verl").is_dir():
        raise RuntimeError(f"missing vendored verl checkout: {vagen / 'verl'}")
    return f"VAGEN={vagen}; verl={vagen / 'verl'}"


def validate_gpu_partition(path_value: str | None) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly name the training GPUs")
    try:
        training = {int(value.strip()) for value in visible.split(",") if value.strip()}
    except ValueError as exc:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must use numeric physical GPU IDs") from exc
    if not path_value:
        raise RuntimeError("SIMWORLD_RL_UE_ENDPOINTS is unset")
    payload = json.loads(Path(path_value).expanduser().read_text(encoding="utf-8"))
    ue = {int(row["gpu"]) for row in payload.get("instances") or [] if "gpu" in row}
    if not ue:
        raise RuntimeError("endpoint manifest does not record UE GPU IDs")
    overlap = training & ue
    if overlap:
        raise RuntimeError(f"UE and training GPU sets overlap: {sorted(overlap)}")
    return f"UE={sorted(ue)}; training={sorted(training)}; disjoint"


def load_endpoints(path_value: str | None, *, connect: bool) -> str:
    if not path_value:
        raise RuntimeError("SIMWORLD_RL_UE_ENDPOINTS is unset")
    path = Path(path_value).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") not in {None, "ready"}:
        raise RuntimeError(f"fleet status is {payload.get('status')!r}, not 'ready'")
    instances = payload.get("instances") or []
    if not instances:
        raise RuntimeError("endpoint manifest contains no instances")
    seen: set[tuple[str, int]] = set()
    for row in instances:
        host = str(row.get("host") or "127.0.0.1")
        port = int(row["port"])
        address = (host, port)
        if address in seen:
            raise RuntimeError(f"duplicate endpoint {host}:{port}")
        seen.add(address)
        if row.get("map_name") not in {None, "", "RT10"}:
            raise RuntimeError(f"endpoint {host}:{port} has map {row.get('map_name')!r}")
        if connect:
            try:
                with socket.create_connection(address, timeout=1.0):
                    pass
            except OSError as exc:
                raise RuntimeError(f"cannot connect to {host}:{port}: {exc}") from exc
    return f"{len(instances)} RT10 endpoint(s) in {path}"


def validate_datasets() -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is not installed") from exc

    possible_seed_sets: list[set[int]] = []
    counts: list[int] = []
    for filename in ("train_vagen.yaml", "val_vagen.yaml"):
        payload = yaml.safe_load((REPO / "online_rl" / filename).read_text(encoding="utf-8"))
        env = payload["envs"][0]
        minimum, maximum, occurrence_limit = (int(value) for value in env["seed"])
        expected = int(env["n_envs"])
        # VAGEN interprets a three-value directive as inclusive
        # [minimum, maximum, maximum_occurrences_per_seed].
        capacity = (maximum - minimum + 1) * occurrence_limit
        if maximum < minimum or occurrence_limit < 1 or capacity < expected:
            raise RuntimeError(
                f"{filename}: seed directive cannot supply n_envs={expected} "
                f"(capacity={capacity})"
            )
        if occurrence_limit != 1:
            raise RuntimeError(f"{filename}: online RL requires unique seeds (limit must be 1)")
        possible_seed_sets.append(set(range(minimum, maximum + 1)))
        counts.append(expected)
    overlap = possible_seed_sets[0] & possible_seed_sets[1]
    if overlap:
        raise RuntimeError(f"training/validation seed overlap: {sorted(overlap)[:5]}")
    return f"train={counts[0]}, validation={counts[1]}, unique and disjoint seed ranges"


def validate_cuda() -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    count = torch.cuda.device_count()
    requested = int(os.environ.get("N_GPUS", count))
    if requested < 1:
        raise RuntimeError("N_GPUS must be positive")
    if count != requested:
        raise RuntimeError(f"N_GPUS={requested}, but PyTorch sees {count} device(s)")
    names = []
    minimum_free_mb = int(os.environ.get("MIN_TRAIN_GPU_FREE_MB", "18000"))
    for index in range(count):
        try:
            torch.zeros(8, device=f"cuda:{index}")
        except Exception as exc:  # noqa: BLE001 - surface CUDA initialization failures
            raise RuntimeError(f"cuda:{index} cannot allocate a probe tensor: {exc}") from exc
        free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
        free_mb = free_bytes // (1024 * 1024)
        if free_mb < minimum_free_mb:
            raise RuntimeError(
                f"cuda:{index} has {free_mb} MiB free; "
                f"MIN_TRAIN_GPU_FREE_MB={minimum_free_mb}"
            )
        names.append(f"{torch.cuda.get_device_name(index)} ({free_mb} MiB free)")
    return f"{count} visible device(s), N_GPUS={requested}: {', '.join(names)}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("setup", "runtime", "train"),
        default="train",
        help="setup checks files; runtime also checks live UE; train adds VAGEN/model/CUDA",
    )
    parser.add_argument(
        "--skip-connect",
        action="store_true",
        help="validate the endpoint manifest without opening its TCP ports",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = [
        check(
            "Python",
            lambda: (
                f"{sys.version.split()[0]} at {sys.executable}"
                if sys.version_info >= (3, 11)
                else (_ for _ in ()).throw(RuntimeError("Python 3.11+ is required"))
            ),
        ),
        check("benchmark files", validate_benchmark_files),
        check("VAGEN datasets", validate_datasets),
        check("adapter import", lambda: require_module("online_rl.vagen_env")),
    ]

    if args.phase in {"runtime", "train"}:
        checks.append(
            check(
                "UE fleet",
                lambda: load_endpoints(
                    os.environ.get("SIMWORLD_RL_UE_ENDPOINTS"),
                    connect=not args.skip_connect,
                ),
            )
        )

    if args.phase == "train":
        vagen_root = os.environ.get("VAGEN_ROOT")
        nav_root = os.environ.get("SIMWORLD_NAV_ROOT")
        checks.extend(
            [
                check("model", lambda: require_path(os.environ.get("MODEL_PATH"), "MODEL_PATH")),
                check("checkout layout", lambda: validate_checkout_layout(vagen_root, nav_root)),
                check("VAGEN import", lambda: require_module("vagen")),
                check("verl import", lambda: require_module("verl")),
                check("Ray import", lambda: require_module("ray")),
                check("vLLM import", lambda: require_module("vllm")),
                check("simworld-nav PR #3 patches", lambda: validate_patches(nav_root)),
                check(
                    "GPU partition",
                    lambda: validate_gpu_partition(os.environ.get("SIMWORLD_RL_UE_ENDPOINTS")),
                ),
                check("CUDA", validate_cuda),
            ]
        )

    for item in checks:
        marker = "PASS" if item.ok else "FAIL"
        print(f"[{marker}] {item.label}: {item.detail}")
    failures = sum(not item.ok for item in checks)
    if failures:
        print(f"Preflight failed: {failures} check(s).", file=sys.stderr)
        return 1
    print(f"Preflight passed for phase={args.phase}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
