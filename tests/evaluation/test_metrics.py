import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from slam.evaluation.metrics import bal_reprojection_report, pose_graph_report, trajectory_report
from slam.geometry.transforms import make_transform
from slam.io.trajectory import PoseStamped, write_tum_trajectory
from slam.optimization.bundle_adjustment import read_bal_problem
from slam.optimization.pose_graph import read_g2o_pose_graph


def test_trajectory_report_computes_translation_rmse_with_origin_alignment():
    estimated = [
        PoseStamped(0.0, make_transform(np.eye(3), np.array([10.0, 0.0, 0.0]))),
        PoseStamped(1.0, make_transform(np.eye(3), np.array([11.0, 0.0, 0.0]))),
    ]
    reference = [
        PoseStamped(0.0, make_transform(np.eye(3), np.array([0.0, 0.0, 0.0]))),
        PoseStamped(1.0, make_transform(np.eye(3), np.array([2.0, 0.0, 0.0]))),
    ]

    report = trajectory_report(estimated, reference)

    assert report.metrics["pose_count"] == 2
    np.testing.assert_allclose(report.metrics["rmse"], np.sqrt(0.5))
    assert report.metrics["align_origin"] is True


def test_bal_and_pose_graph_reports_include_problem_sizes():
    repo_root = Path(__file__).resolve().parents[2]
    bal = read_bal_problem(repo_root / "examples/ch10_bundle_adjustment/tiny_bal.txt")
    graph = read_g2o_pose_graph(repo_root / "examples/ch11_pose_graph/tiny_pose_graph.g2o")

    bal_report = bal_reprojection_report(bal)
    graph_report = pose_graph_report(graph)

    assert bal_report.metrics["observation_count"] > 0
    assert bal_report.metrics["reprojection_rmse"] >= 0.0
    assert graph_report.metrics["edge_count"] == 3
    assert graph_report.metrics["edge_rmse"] > 0.0


def test_benchmark_report_cli_writes_trajectory_json(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    estimated = tmp_path / "estimated.tum"
    reference = tmp_path / "reference.tum"
    output = tmp_path / "report.json"
    write_tum_trajectory(
        estimated,
        [
            PoseStamped(0.0, make_transform(np.eye(3), np.array([0.0, 0.0, 0.0]))),
            PoseStamped(1.0, make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))),
        ],
    )
    write_tum_trajectory(
        reference,
        [
            PoseStamped(0.0, make_transform(np.eye(3), np.array([0.0, 0.0, 0.0]))),
            PoseStamped(1.0, make_transform(np.eye(3), np.array([2.0, 0.0, 0.0]))),
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "examples/reference/benchmark_report.py",
            "trajectory",
            "--estimated-tum",
            str(estimated),
            "--reference-tum",
            str(reference),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["reports"][0]["name"] == "trajectory_translation"
