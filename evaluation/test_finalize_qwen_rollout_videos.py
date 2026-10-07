import json
from pathlib import Path

from evaluation.finalize_qwen_rollout_videos import (
    PENDING_ARTIFACT_AUDIT_REASON,
    runtime_provenance_audit,
    update_aggregate_artifact_eligibility,
    video_index_output_path,
)


def test_full_finalization_writes_canonical_suite_index():
    suite_dir = Path("/tmp/suite")

    assert video_index_output_path(suite_dir, None) == suite_dir / "video_index.json"


def test_filtered_finalization_cannot_replace_canonical_suite_index():
    suite_dir = Path("/tmp/suite")

    assert video_index_output_path(suite_dir, [11, 3, 11]) == (
        suite_dir / "video_index_tasks_3_11.json"
    )


def test_complete_canonical_finalization_clears_comparison_gate(tmp_path):
    aggregate = {
        "planned_rollouts": 2,
        "comparison_eligible": False,
        "comparison_ineligibility_reasons": [PENDING_ARTIFACT_AUDIT_REASON],
    }
    (tmp_path / "aggregate_summary.json").write_text(
        json.dumps(aggregate), encoding="utf-8"
    )
    output = {
        "completed_video_count": 2,
        "failures": [],
        "skipped_incomplete": [],
        "all_alignment_checks_passed": True,
        "all_traffic_e2e_audits_passed": True,
        "all_runtime_provenance_audits_passed": True,
    }

    update_aggregate_artifact_eligibility(tmp_path, output, canonical=True)

    updated = json.loads((tmp_path / "aggregate_summary.json").read_text())
    assert updated["comparison_eligible"] is True
    assert updated["comparison_ineligibility_reasons"] == []
    assert updated["artifact_audits_passed"] is True
    assert updated["artifact_audited_rollouts"] == 2


def test_partial_finalization_keeps_comparison_ineligible(tmp_path):
    aggregate = {
        "planned_rollouts": 2,
        "comparison_eligible": False,
        "comparison_ineligibility_reasons": [PENDING_ARTIFACT_AUDIT_REASON],
    }
    (tmp_path / "aggregate_summary.json").write_text(
        json.dumps(aggregate), encoding="utf-8"
    )
    output = {
        "completed_video_count": 1,
        "failures": [],
        "skipped_incomplete": [{"reason": "status=running"}],
        "all_alignment_checks_passed": True,
        "all_traffic_e2e_audits_passed": True,
        "all_runtime_provenance_audits_passed": True,
    }

    update_aggregate_artifact_eligibility(tmp_path, output, canonical=True)

    updated = json.loads((tmp_path / "aggregate_summary.json").read_text())
    assert updated["comparison_eligible"] is False
    assert updated["artifact_audits_passed"] is False
    assert updated["comparison_ineligibility_reasons"] == [
        "artifact videos finalized for 1 of 2 planned rollouts",
        "artifact finalization skipped 1 incomplete rollout(s)",
    ]


def test_failed_runtime_audit_has_specific_ineligibility_reason(tmp_path):
    aggregate = {
        "planned_rollouts": 2,
        "comparison_eligible": False,
        "comparison_ineligibility_reasons": [PENDING_ARTIFACT_AUDIT_REASON],
    }
    (tmp_path / "aggregate_summary.json").write_text(
        json.dumps(aggregate), encoding="utf-8"
    )
    output = {
        "completed_video_count": 2,
        "failures": [{"reason": "runtime provenance audit failed"}],
        "skipped_incomplete": [],
        "all_alignment_checks_passed": True,
        "all_traffic_e2e_audits_passed": True,
        "all_runtime_provenance_audits_passed": False,
    }

    update_aggregate_artifact_eligibility(tmp_path, output, canonical=True)

    updated = json.loads((tmp_path / "aggregate_summary.json").read_text())
    assert updated["comparison_eligible"] is False
    assert updated["comparison_ineligibility_reasons"] == [
        "artifact finalization reported 1 failure(s)",
        "actual runtime provenance audits did not all pass",
    ]


def test_filtered_finalization_does_not_change_canonical_aggregate(tmp_path):
    aggregate = {
        "planned_rollouts": 2,
        "comparison_eligible": False,
        "comparison_ineligibility_reasons": [PENDING_ARTIFACT_AUDIT_REASON],
    }
    aggregate_path = tmp_path / "aggregate_summary.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")

    update_aggregate_artifact_eligibility(
        tmp_path,
        {"completed_video_count": 1},
        canonical=False,
    )

    assert json.loads(aggregate_path.read_text()) == aggregate


def test_runtime_provenance_audit_requires_actual_local_service(tmp_path):
    attempt = {
        "runtime_provenance_version": "runtime_attempt_v2",
        "runtime": {
            "ue": {"launcher": "/opt/SimWorld.sh", "gpu": "6"},
            "qwen": {
                "observed_local_service": {
                    "status": "observed",
                    "argv": ["python", "-m", "vllm.entrypoints.openai.api_server"],
                    "vllm_version": "0.11.0",
                    "gpu_processes": [{"index": 4, "uuid": "GPU-test"}],
                }
            },
            "python": {
                "version": "3.12.13",
                "modules": {
                    "unrealcv": {
                        "path": "/deps/unrealcv/__init__.py",
                        "sha256": "abc",
                    }
                },
            },
        },
    }
    (tmp_path / "runtime_attempt_01_1.json").write_text(
        json.dumps(attempt), encoding="utf-8"
    )

    audit = runtime_provenance_audit(
        tmp_path,
        {
            "ue_renderer": {
                "device_name": "NVIDIA RTX A5000",
                "device_type": "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
                "adapter_fallback": False,
            }
        },
    )

    assert audit == {"passed": True, "attempt_count": 1, "failures": []}


def test_runtime_provenance_audit_rejects_legacy_requested_only_record(tmp_path):
    (tmp_path / "runtime_attempt_01_1.json").write_text(
        json.dumps({"runtime": {"ue": {"gpu": "6"}, "qwen": {"gpu": "4"}}}),
        encoding="utf-8",
    )

    audit = runtime_provenance_audit(tmp_path, {})

    assert audit["passed"] is False
    assert audit["attempt_count"] == 1
    assert "runtime_provenance_version" in audit["failures"][0]
