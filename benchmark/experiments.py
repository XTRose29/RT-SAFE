#!/usr/bin/env python3
"""Plan and run the three controlled open-source Qwen3-VL studies."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_RUNNER = REPO_ROOT / "benchmark" / "run.py"
CONTROLLED_QWEN_SERVER = (
    REPO_ROOT / "evaluation" / "qwen3vl_thinking_openai_server.py"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT.parent / "SimWorld-RealTime-results"
TASKS_PER_CONDITION = 36
STUDIES = ("size", "reasoning", "timing")

SIZE_MODELS = (
    ("2b", "Qwen/Qwen3-VL-2B-Instruct"),
    ("4b", "Qwen/Qwen3-VL-4B-Instruct"),
    ("8b", "Qwen/Qwen3-VL-8B-Instruct"),
    ("32b", "Qwen/Qwen3-VL-32B-Instruct"),
    ("30b_a3b", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    ("235b_a22b", "Qwen/Qwen3-VL-235B-A22B-Instruct"),
)
INSTRUCT_8B = "Qwen/Qwen3-VL-8B-Instruct"
THINKING_8B = "Qwen/Qwen3-VL-8B-Thinking"
THINKING_BUDGETS = (64, 128, 256)
ANSWER_BUDGET = 128


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def condition(
    study: str,
    name: str,
    model: str,
    *,
    env_modes: Iterable[str] = ("realtime",),
    thinking_budget: int | None = None,
) -> dict[str, Any]:
    modes = list(env_modes)
    enable_thinking = thinking_budget is not None
    return {
        "study": study,
        "condition": name,
        "model": model,
        "checkpoint_variant": "Thinking" if enable_thinking else "Instruct",
        "enable_thinking": enable_thinking,
        "thinking_budget_tokens": thinking_budget,
        "answer_budget_tokens": ANSWER_BUDGET if enable_thinking else None,
        "env_modes": modes,
        "rollout_count": TASKS_PER_CONDITION * len(modes),
    }


def build_conditions(studies: Iterable[str] = STUDIES) -> list[dict[str, Any]]:
    requested = tuple(studies)
    unknown = sorted(set(requested) - set(STUDIES))
    if unknown:
        raise ValueError(f"Unknown studies: {', '.join(unknown)}")

    conditions: list[dict[str, Any]] = []
    if "size" in requested:
        conditions.extend(
            condition("size", f"qwen3vl_{label}_instruct", model)
            for label, model in SIZE_MODELS
        )
    if "reasoning" in requested:
        conditions.append(
            condition("reasoning", "qwen3vl_8b_instruct", INSTRUCT_8B)
        )
        conditions.extend(
            condition(
                "reasoning",
                f"qwen3vl_8b_thinking_tb{budget}",
                THINKING_8B,
                thinking_budget=budget,
            )
            for budget in THINKING_BUDGETS
        )
    if "timing" in requested:
        conditions.append(
            condition(
                "timing",
                "qwen3vl_8b_instruct_static_vs_realtime",
                INSTRUCT_8B,
                env_modes=("static", "realtime"),
            )
        )
    return conditions


def parse_model_paths(values: Iterable[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --model-path {value!r}; use CHECKPOINT=/absolute/path"
            )
        model, raw_path = value.split("=", 1)
        model = model.strip()
        path = Path(raw_path).expanduser()
        if not model or not raw_path:
            raise ValueError(
                f"Invalid --model-path {value!r}; use CHECKPOINT=/absolute/path"
            )
        paths[model] = path
    return paths


def required_models(conditions: Iterable[dict[str, Any]]) -> set[str]:
    return {str(item["model"]) for item in conditions}


def model_path_for(
    model: str,
    paths: dict[str, Path],
    *,
    require_existing: bool,
) -> str:
    path = paths.get(model)
    if path is None:
        if require_existing:
            raise ValueError(
                f"Missing --model-path {model}=/absolute/path/to/checkpoint"
            )
        return f"<MODEL_PATH:{model}>"
    if require_existing and not path.is_dir():
        raise ValueError(f"Checkpoint path is not a directory: {path}")
    return str(path.resolve()) if path.exists() else str(path)


def controlled_server_command(
    args: argparse.Namespace,
    model: str,
    model_path: str,
) -> str:
    command = [
        args.qwen_python,
        str(CONTROLLED_QWEN_SERVER),
        "--model-path",
        model_path,
        "--served-model-name",
        model,
        "--host",
        args.qwen_host,
        "--port",
        str(args.qwen_port),
        "--require-cuda",
    ]
    if args.qwen_load_in_8bit:
        command.append("--load-in-8bit")
    return shlex.join(command)


def runner_args_for(
    item: dict[str, Any],
    args: argparse.Namespace,
    model_path: str,
) -> list[str]:
    values = [
        "--model",
        str(item["model"]),
        "--qwen-model-path",
        model_path,
        "--qwen-url",
        f"http://{args.qwen_host}:{args.qwen_port}/v1",
        "--qwen-host",
        args.qwen_host,
        "--qwen-port",
        str(args.qwen_port),
        "--qwen-gpu",
        str(args.qwen_gpu),
        "--qwen-python",
        args.qwen_python,
        "--qwen-max-tokens",
        "128",
        "--env-modes",
        *item["env_modes"],
    ]
    if args.qwen_pythonpath:
        values.extend(("--qwen-pythonpath", args.qwen_pythonpath))
    budget = item["thinking_budget_tokens"]
    if budget is not None:
        total_completion_cap = int(budget) + ANSWER_BUDGET + 16
        max_tokens_index = values.index("--qwen-max-tokens") + 1
        values[max_tokens_index] = str(total_completion_cap)
        values.extend(
            (
                "--enable-thinking",
                "--reasoning-budget-tokens",
                str(budget),
                "--reasoning-budget-answer-tokens",
                str(ANSWER_BUDGET),
                "--qwen-launch-command",
                controlled_server_command(
                    args,
                    str(item["model"]),
                    model_path,
                ),
            )
        )
    return values


def benchmark_command(
    item: dict[str, Any],
    args: argparse.Namespace,
    model_path: str,
    *,
    smoke: bool,
) -> list[str]:
    suffix = "smoke" if smoke else "full"
    suite_name = (
        f"{args.program_name}__{item['study']}__"
        f"{slug(str(item['condition']))}__{suffix}"
    )
    command = [
        sys.executable,
        str(BENCHMARK_RUNNER),
        "run",
        "smoke" if smoke else "all_tasks_easy_realtime_collision",
        "--suite-name",
        suite_name,
        "--output-root",
        str(args.output_root),
    ]
    if args.resume and not smoke:
        command.append("--resume")
    for value in runner_args_for(item, args, model_path):
        command.append(f"--runner-arg={value}")
    return command


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def build_manifest(
    conditions: list[dict[str, Any]],
    args: argparse.Namespace,
    paths: dict[str, Path],
    *,
    require_existing: bool,
) -> dict[str, Any]:
    planned = []
    for item in conditions:
        model_path = model_path_for(
            str(item["model"]),
            paths,
            require_existing=require_existing,
        )
        record = dict(item)
        record["model_path"] = model_path
        record["smoke_command"] = shlex.join(
            benchmark_command(item, args, model_path, smoke=True)
        )
        record["full_command"] = shlex.join(
            benchmark_command(item, args, model_path, smoke=False)
        )
        planned.append(record)
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "program_name": args.program_name,
        "reference_config": "all_tasks_easy_realtime_collision",
        "selected_studies": list(args.studies),
        "condition_count": len(planned),
        "full_rollout_count": sum(item["rollout_count"] for item in planned),
        "smoke_rollout_count": 0 if args.no_smoke else len(planned),
        "controls": {
            "tasks_per_environment_condition": TASKS_PER_CONDITION,
            "difficulty": "easy",
            "seed": 0,
            "camera": "single 720x640 RGB, 100-degree horizontal FOV, -25-degree pitch",
            "action_markers": 7,
            "reasoning_answer_budget_tokens": ANSWER_BUDGET,
            "thinking_budget_backend": (
                "evaluation/qwen3vl_thinking_openai_server.py; hard force-close"
            ),
        },
        "conditions": planned,
    }


def run_program(
    conditions: list[dict[str, Any]],
    args: argparse.Namespace,
    paths: dict[str, Path],
    manifest_path: Path,
) -> int:
    manifest = build_manifest(
        conditions,
        args,
        paths,
        require_existing=True,
    )
    manifest["status"] = "running"
    manifest["started_at"] = utc_now()
    write_json(manifest_path, manifest)

    for index, item in enumerate(conditions, start=1):
        model_path = model_path_for(
            str(item["model"]),
            paths,
            require_existing=True,
        )
        print(
            f"[experiments {index}/{len(conditions)}] "
            f"{item['study']}: {item['condition']}",
            flush=True,
        )
        if not args.no_smoke and not args.resume:
            smoke_command = benchmark_command(
                item,
                args,
                model_path,
                smoke=True,
            )
            smoke_result = subprocess.run(
                smoke_command,
                cwd=REPO_ROOT,
                check=False,
            )
            if smoke_result.returncode:
                manifest["status"] = "stopped_on_smoke_error"
                manifest["failed_condition"] = item["condition"]
                manifest["returncode"] = smoke_result.returncode
                manifest["finished_at"] = utc_now()
                write_json(manifest_path, manifest)
                return smoke_result.returncode

        full_command = benchmark_command(
            item,
            args,
            model_path,
            smoke=False,
        )
        result = subprocess.run(full_command, cwd=REPO_ROOT, check=False)
        if result.returncode:
            manifest["status"] = "stopped_on_rollout_error"
            manifest["failed_condition"] = item["condition"]
            manifest["returncode"] = result.returncode
            manifest["finished_at"] = utc_now()
            write_json(manifest_path, manifest)
            return result.returncode

    manifest["status"] = "completed"
    manifest["finished_at"] = utc_now()
    write_json(manifest_path, manifest)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=STUDIES,
        default=list(STUDIES),
    )
    parser.add_argument(
        "--program-name",
        default=f"qwen3vl_open_source_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        metavar="CHECKPOINT=PATH",
        help="Repeat once for each selected checkpoint.",
    )
    parser.add_argument("--qwen-python", default=sys.executable)
    parser.add_argument("--qwen-pythonpath", default=None)
    parser.add_argument("--qwen-host", default="127.0.0.1")
    parser.add_argument("--qwen-port", type=int, default=30001)
    parser.add_argument("--qwen-gpu", default="1")
    parser.add_argument("--qwen-load-in-8bit", action="store_true")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.qwen_port <= 0:
        raise ValueError("--qwen-port must be positive")
    if args.resume and args.command != "run":
        raise ValueError("--resume is only valid with run")
    if args.resume and args.program_name.startswith("qwen3vl_open_source_"):
        raise ValueError("--resume requires the original explicit --program-name")

    args.output_root = args.output_root.expanduser().resolve()
    conditions = build_conditions(args.studies)
    paths = parse_model_paths(args.model_path)
    program_dir = args.output_root / args.program_name
    manifest_path = program_dir / "program_manifest.json"

    if args.command == "plan":
        manifest = build_manifest(
            conditions,
            args,
            paths,
            require_existing=False,
        )
        manifest["status"] = "planned"
        write_json(manifest_path, manifest)
        print(json.dumps({
            "manifest": str(manifest_path),
            "conditions": manifest["condition_count"],
            "full_rollouts": manifest["full_rollout_count"],
            "smoke_rollouts": manifest["smoke_rollout_count"],
            "required_models": sorted(required_models(conditions)),
        }, indent=2))
        return 0

    return run_program(conditions, args, paths, manifest_path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"experiments: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
