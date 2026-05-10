import os
import subprocess
import sys
from pathlib import Path


def test_ch10_scipy_bal_cli_runs_included_fixture(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    result = _run_example(
        repo_root,
        [
            "examples/ch10_bundle_adjustment/scipy_bal.py",
            "--bal",
            "examples/ch10_bundle_adjustment/tiny_bal.txt",
            "--fix-cameras",
            "--max-nfev",
            "100",
        ],
    )

    assert "initial reprojection RMSE: 3.183180" in result.stdout
    assert "final reprojection RMSE: 0.000000" in result.stdout
    assert "success: True" in result.stdout


def test_ch11_pose_graph_cli_runs_included_fixture(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    output_tum = tmp_path / "optimized.tum"
    output_kitti = tmp_path / "optimized.kitti"

    result = _run_example(
        repo_root,
        [
            "examples/ch11_pose_graph/optimize_pose_graph.py",
            "--g2o",
            "examples/ch11_pose_graph/tiny_pose_graph.g2o",
            "--fixed-vertex",
            "0",
            "--output-tum",
            str(output_tum),
            "--output-kitti",
            str(output_kitti),
        ],
    )

    assert "initial edge error: 0.140000000" in result.stdout
    assert "final edge error: 0.000000000" in result.stdout
    assert "success: True" in result.stdout
    assert len(output_tum.read_text(encoding="utf-8").splitlines()) == 3
    assert len(output_kitti.read_text(encoding="utf-8").splitlines()) == 3


def _run_example(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, *args],
        cwd=repo_root,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
