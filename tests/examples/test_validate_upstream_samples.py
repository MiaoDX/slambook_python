from pathlib import Path

from examples.reference.validate_upstream_samples import (
    ValidationCase,
    ValidationResult,
    build_report,
    build_validation_cases,
    parse_stdout_metrics,
    run_validation_case,
)


def test_parse_stdout_metrics_extracts_validation_summary_values():
    stdout = """
match count: 161
valid 3D-3D correspondence count: 150
registration RMSE m: 0.120947847
frame 0: keyframe=1 inserted_points=300 map_points=300
frame 1: tracked inliers=57 keyframe=1 inserted_points=300 map_points=600
descriptor shape: (10, 8)
candidates:
  index=4 score=0.125000000
  index=2 score=0.250000000
"""

    metrics = parse_stdout_metrics(stdout)

    assert metrics["match_count"] == 161
    assert metrics["valid_3d_3d_correspondence_count"] == 150
    assert metrics["registration_rmse_m"] == 0.120947847
    assert metrics["frame_0_inserted_points"] == 300
    assert metrics["frame_1_inliers"] == 57
    assert metrics["descriptor_shape"] == [10, 8]
    assert metrics["candidate_count"] == 2


def test_build_validation_cases_supports_chapter_group_selection(tmp_path):
    cases = build_validation_cases(
        upstream_root=tmp_path / "upstream",
        work_dir=tmp_path / "work",
        python="python",
        case_names={"ch12"},
    )

    assert [case.name for case in cases] == [
        "ch12_train_vocabulary",
        "ch12_build_descriptors",
        "ch12_retrieve_candidates",
    ]


def test_run_validation_case_reports_missing_inputs(tmp_path):
    case = ValidationCase(
        name="missing",
        command=["python", "-c", "raise SystemExit(2)"],
        required_paths=(tmp_path / "does-not-exist",),
    )

    result = run_validation_case(case, repo_root=Path.cwd(), env={})

    assert result.status == "missing_input"
    assert result.returncode is None
    assert result.missing_paths == [str(tmp_path / "does-not-exist")]


def test_build_report_summarizes_status_counts(tmp_path):
    completed = ValidationResult(
        name="case",
        status="passed",
        command=["python", "--version"],
        returncode=0,
        duration_seconds=0.01,
        metrics={},
        missing_paths=[],
        stdout_tail="",
        stderr_tail="",
    )
    report = build_report(
        results=[completed],
        upstream_root=tmp_path / "upstream",
        repo_root=tmp_path / "repo",
        work_dir=tmp_path / "work",
    )

    assert report["summary"]["case_count"] == 1
    assert report["summary"]["status_counts"]["passed"] == 1
    assert report["summary"]["passed"] is True
