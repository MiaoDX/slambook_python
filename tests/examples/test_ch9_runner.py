import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def test_ch9_runner_exports_trajectory_on_tiny_image_sequence(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    cv2.imwrite(str(image_dir / "000000.png"), np.zeros((48, 64), dtype=np.uint8))
    cv2.imwrite(str(image_dir / "000001.png"), np.zeros((48, 64), dtype=np.uint8))
    output_tum = tmp_path / "trajectory.tum"
    output_kitti = tmp_path / "trajectory.kitti"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "examples/ch9_project/run_vo.py",
            "--images",
            str(image_dir),
            "--intrinsics",
            "500",
            "500",
            "32",
            "24",
            "--output-tum",
            str(output_tum),
            "--output-kitti",
            str(output_kitti),
        ],
        cwd=repo_root,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "frame 1: skipped, matches=0" in result.stdout
    assert "successful relative pose steps: 0" in result.stdout
    assert len(output_tum.read_text(encoding="utf-8").splitlines()) == 2
    assert len(output_kitti.read_text(encoding="utf-8").splitlines()) == 2
