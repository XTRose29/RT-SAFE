#!/usr/bin/env python3
"""Validate and render every completed Qwen rollout in a suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.render_rollout_input_video import (  # noqa: E402
    VIDEO_REPORT_SCHEMA_VERSION,
)


PENDING_ARTIFACT_AUDIT_REASON = (
    "artifact audits have not passed for every planned rollout"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def update_aggregate_artifact_eligibility(
    suite_dir: Path,
    output: dict,
    *,
    canonical: bool,
) -> None:
    """Clear the comparison gate only after a complete canonical finalization."""

    if not canonical:
        return
    aggregate_path = suite_dir / "aggregate_summary.json"
    if not aggregate_path.is_file():
        return
    aggregate = read_json(aggregate_path)
    planned = int(aggregate.get("planned_rollouts") or 0)
    audited = int(output.get("completed_video_count") or 0)
    passed = (
        planned > 0
        and audited == planned
        and not output.get("failures")
        and not output.get("skipped_incomplete")
        and output.get("all_alignment_checks_passed") is True
        and output.get("all_traffic_e2e_audits_passed") is True
        and output.get("all_runtime_provenance_audits_passed") is True
    )
    reasons = [
        str(reason)
        for reason in aggregate.get("comparison_ineligibility_reasons") or []
        if str(reason) != PENDING_ARTIFACT_AUDIT_REASON
        and not str(reason).startswith("artifact audits passed for ")
        and not str(reason).startswith("artifact videos finalized for ")
        and not str(reason).startswith("artifact finalization reported ")
        and not str(reason).startswith("artifact finalization skipped ")
        and str(reason) not in {
            "action/image alignment audits did not all pass",
            "traffic/image/feedback audits did not all pass",
            "actual runtime provenance audits did not all pass",
        }
    ]
    if not passed:
        if audited != planned:
            reasons.append(
                f"artifact videos finalized for {audited} of {planned} planned rollouts"
            )
        if output.get("failures"):
            reasons.append(
                "artifact finalization reported "
                f"{len(output['failures'])} failure(s)"
            )
        if output.get("skipped_incomplete"):
            reasons.append(
                "artifact finalization skipped "
                f"{len(output['skipped_incomplete'])} incomplete rollout(s)"
            )
        if output.get("all_alignment_checks_passed") is not True:
            reasons.append("action/image alignment audits did not all pass")
        if output.get("all_traffic_e2e_audits_passed") is not True:
            reasons.append("traffic/image/feedback audits did not all pass")
        if output.get("all_runtime_provenance_audits_passed") is not True:
            reasons.append("actual runtime provenance audits did not all pass")
    aggregate.update(
        {
            "artifact_audits_completed_at": datetime.now(timezone.utc).isoformat(),
            "artifact_audits_passed": passed,
            "artifact_audited_rollouts": audited,
            "artifact_video_index": "video_index.json",
            "comparison_eligible": passed and not reasons,
            "comparison_ineligibility_reasons": reasons,
        }
    )
    write_json(aggregate_path, aggregate)


def runtime_provenance_audit(run_dir: Path, status: dict | None = None) -> dict:
    """Require machine-readable actual runtime provenance for every UE attempt."""

    paths = sorted(run_dir.glob("runtime_attempt_*.json"))
    failures = []
    for path in paths:
        try:
            record = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{path.name}: unreadable ({exc})")
            continue
        runtime = record.get("runtime") or {}
        qwen = runtime.get("qwen") or {}
        python = runtime.get("python") or {}
        unrealcv = (python.get("modules") or {}).get("unrealcv") or {}
        observed = qwen.get("observed_local_service") or {}
        gpu_processes = observed.get("gpu_processes") or []
        required = {
            "runtime_provenance_version": (
                record.get("runtime_provenance_version") == "runtime_attempt_v2"
            ),
            "ue_launcher": bool((runtime.get("ue") or {}).get("launcher")),
            "ue_gpu": (runtime.get("ue") or {}).get("gpu") is not None,
            "python_version": bool(python.get("version")),
            "unrealcv_path": bool(unrealcv.get("path")),
            "unrealcv_sha256": bool(unrealcv.get("sha256")),
            "qwen_observed": observed.get("status") == "observed",
            "qwen_argv": bool(observed.get("argv")),
            "qwen_vllm_version": bool(observed.get("vllm_version")),
            "qwen_gpu_process": bool(gpu_processes),
        }
        missing = [name for name, present in required.items() if not present]
        if missing:
            failures.append(f"{path.name}: missing {', '.join(missing)}")
    if not paths:
        failures.append("no runtime_attempt_*.json records")
    renderer = (status or {}).get("ue_renderer") or {}
    if (
        not renderer.get("device_name")
        or renderer.get("device_type") != "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"
        or renderer.get("adapter_fallback") is not False
    ):
        failures.append("completed status lacks validated discrete UE renderer")
    return {
        "passed": not failures,
        "attempt_count": len(paths),
        "failures": failures,
    }


def video_index_output_path(suite_dir: Path, task_ids: list[int] | None) -> Path:
    """Keep filtered finalization from replacing the canonical suite index."""
    if task_ids is None:
        return suite_dir / "video_index.json"
    suffix = "_".join(str(task_id) for task_id in sorted(set(task_ids))) or "none"
    return suite_dir / f"video_index_tasks_{suffix}.json"


def safety_event_category(label: str) -> str:
    normalized = label.lower()
    if "human collision" in normalized:
        return "human_collision"
    if "object collision" in normalized:
        return "object_collision"
    if "building collision" in normalized:
        return "building_collision"
    if "vehicle collision" in normalized:
        return "vehicle_collision"
    if "illegal crossing" in normalized:
        return "illegal_crossing"
    if "red light" in normalized or "red-light" in normalized:
        return "red_light_violation"
    if "touched road" in normalized:
        return "touched_road"
    if normalized == "fall":
        return "fall"
    if "oil" in normalized:
        return "oil"
    if "water" in normalized:
        return "water"
    return "other"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate videos even when a matching video report already exists.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Finalize completed cells without failing on planned cells still in progress.",
    )
    parser.add_argument(
        "--policy-label",
        default=(
            "Qwen3-VL-8B-Instruct, constant-speed exact-waypoint execution"
        ),
    )
    parser.add_argument("--header-label", default="QWEN3-VL-8B INPUT")
    parser.add_argument(
        "--task-ids",
        nargs="*",
        type=int,
        help="Optional scenario IDs to finalize; omitted means every completed run.",
    )
    args = parser.parse_args(argv)
    suite_dir = args.suite_dir.resolve()

    records = []
    failures = []
    skipped = []
    for condition_path in sorted(suite_dir.glob("runs/**/condition.json")):
        run_dir = condition_path.parent
        condition = read_json(condition_path)
        if args.task_ids is not None and int(condition["task_id"]) not in set(
            args.task_ids
        ):
            continue
        status_path = run_dir / "run_status.json"
        status = read_json(status_path) if status_path.is_file() else {}
        if status.get("status") != "completed":
            collection = skipped if args.allow_incomplete else failures
            collection.append(
                {
                    "run_dir": str(run_dir),
                    "reason": f"status={status.get('status')}",
                }
            )
            continue
        runtime_audit = runtime_provenance_audit(run_dir, status)
        if not runtime_audit["passed"]:
            failures.append(
                {
                    "run_dir": str(run_dir),
                    "reason": "runtime provenance audit failed",
                    "runtime_provenance_failures": runtime_audit["failures"],
                }
            )
        step_dirs = sorted(run_dir.glob("task_*_steps"))
        if len(step_dirs) != 1:
            failures.append(
                {
                    "run_dir": str(run_dir),
                    "reason": f"expected one step directory, found {len(step_dirs)}",
                }
            )
            continue
        steps_dir = step_dirs[0]
        alignment_path = run_dir / "rollout_alignment_check.json"
        video_path = run_dir / "rollout_agent_input_annotated.mp4"
        validation = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "evaluation/validate_rollout_waypoint_alignment.py"),
                "--steps-dir",
                str(steps_dir),
                "--output",
                str(alignment_path),
            ],
            check=False,
            cwd=REPO_ROOT,
        )
        alignment = read_json(alignment_path)
        if validation.returncode != 0 or not alignment.get("passed"):
            failures.append(
                {
                    "run_dir": str(run_dir),
                    "reason": "action-to-video alignment validation failed",
                    "alignment_report": str(alignment_path),
                }
            )
        result_path = Path(str(status.get("result_path") or ""))
        if not result_path.is_absolute():
            result_path = REPO_ROOT / result_path
        traffic_audit_path = run_dir / "rollout_traffic_e2e_audit.json"
        traffic_validation = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools/audit_traffic_e2e.py"),
                str(result_path),
                "--steps-dir",
                str(steps_dir),
                "--case",
                "any",
                "--output",
                str(traffic_audit_path),
            ],
            check=False,
            cwd=REPO_ROOT,
        )
        traffic_audit = read_json(traffic_audit_path)
        if traffic_validation.returncode != 0 or not traffic_audit.get("passed"):
            failures.append(
                {
                    "run_dir": str(run_dir),
                    "reason": "traffic/image/feedback/evaluation E2E audit failed",
                    "traffic_audit_report": str(traffic_audit_path),
                }
            )
        report_path = run_dir / "rollout_input_video_report.json"
        existing_report = read_json(report_path) if report_path.is_file() else {}
        needs_render = (
            args.force
            or not video_path.is_file()
            or not report_path.is_file()
            or existing_report.get("video_report_schema_version")
            != VIDEO_REPORT_SCHEMA_VERSION
            or float(existing_report.get("fps", -1)) != args.fps
        )
        if needs_render:
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "evaluation/render_rollout_input_video.py"),
                    "--steps-dir",
                    str(steps_dir),
                    "--output",
                    str(video_path),
                    "--task-id",
                    str(condition["task_id"]),
                    "--task-index",
                    str(condition["task_index"]),
                    "--difficulty",
                    str(condition["difficulty"]),
                    "--fps",
                    str(args.fps),
                    "--policy-label",
                    args.policy_label,
                    "--header-label",
                    args.header_label,
                    "--learned-agent-model-was-run",
                ],
                check=True,
                cwd=REPO_ROOT,
            )
        report = read_json(report_path)
        records.append(
            {
                "task_id": condition["task_id"],
                "task_index": condition["task_index"],
                "run_dir": str(run_dir),
                "video": str(video_path),
                "alignment_report": str(alignment_path),
                "traffic_e2e_audit_report": str(traffic_audit_path),
                "video_report": str(run_dir / "rollout_input_video_report.json"),
                "result": status.get("result_path"),
                "success": status.get("success"),
                "failed": status.get("failed"),
                "frame_count": report["frame_count"],
                "video_duration_seconds": report["video_duration_seconds"],
                "sim_time_duration_seconds": report[
                    "sim_time_duration_seconds"
                ],
                "video_to_sim_time_duration_error_seconds": report[
                    "video_to_sim_time_duration_error_seconds"
                ],
                "video_includes_every_policy_input_image": report[
                    "video_includes_every_policy_input_image"
                ],
                "safety_event_count": report["safety_event_count"],
                "mean_input_render_wall_seconds": report[
                    "mean_input_render_wall_seconds"
                ],
                "alignment_passed": alignment["passed"],
                "traffic_e2e_audit_passed": traffic_audit["passed"],
                "runtime_provenance_passed": runtime_audit["passed"],
                "runtime_attempt_count": runtime_audit["attempt_count"],
                "move_steps": alignment["move_steps"],
                "selected_dot_command_matches": alignment[
                    "selected_dot_command_matches"
                ],
                "reached_selected_dot_steps": alignment[
                    "reached_selected_dot_steps"
                ],
                "safety_explained_blocked_steps": alignment[
                    "safety_explained_blocked_steps"
                ],
                "turn_steps_with_following_observation": alignment[
                    "turn_steps_with_following_observation"
                ],
                "turn_label_heading_matches": alignment[
                    "turn_label_heading_matches"
                ],
                "turn_position_stationary_checks": alignment[
                    "turn_position_stationary_checks"
                ],
                "turn_position_stationary_matches": alignment[
                    "turn_position_stationary_matches"
                ],
                "safety_events": report["safety_events"],
            }
        )

    event_examples: dict[str, list[dict]] = {}
    for record in records:
        for event in record.pop("safety_events"):
            for label in event["events"]:
                category = safety_event_category(label)
                event_examples.setdefault(category, []).append(
                    {
                        "task_id": record["task_id"],
                        "task_index": record["task_index"],
                        "video": record["video"],
                        "step": event["step"],
                        "sim_time_start_seconds": event[
                            "sim_time_start_seconds"
                        ],
                        "sim_time_end_seconds": event["sim_time_end_seconds"],
                        "event": label,
                        "source_input_image": event["source_input_image"],
                    }
                )

    output = {
        "suite_dir": str(suite_dir),
        "completed_video_count": len(records),
        "failures": failures,
        "skipped_incomplete": skipped,
        "all_alignment_checks_passed": bool(records)
        and all(item["alignment_passed"] for item in records),
        "all_traffic_e2e_audits_passed": bool(records)
        and all(item["traffic_e2e_audit_passed"] for item in records),
        "all_runtime_provenance_audits_passed": bool(records)
        and all(item["runtime_provenance_passed"] for item in records),
        "rollouts": records,
        "unsafe_trigger_video_examples": event_examples,
    }
    output_path = video_index_output_path(suite_dir, args.task_ids)
    write_json(output_path, output)
    update_aggregate_artifact_eligibility(
        suite_dir,
        output,
        canonical=args.task_ids is None,
    )
    print(json.dumps(output, indent=2))
    return (
        1
        if failures
        or not output["all_alignment_checks_passed"]
        or not output["all_traffic_e2e_audits_passed"]
        or not output["all_runtime_provenance_audits_passed"]
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
