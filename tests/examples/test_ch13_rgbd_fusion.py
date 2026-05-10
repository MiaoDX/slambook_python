import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def test_ch13_rgbd_fusion_writes_known_pose_sequence(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    color_dir = tmp_path / "color"
    depth_dir = tmp_path / "depth"
    color_dir.mkdir()
    depth_dir.mkdir()
    color = np.zeros((1, 1, 3), dtype=np.uint8)
    color[:, :] = [10, 20, 30]
    depth = np.array([[1000]], dtype=np.uint16)
    for index in range(1, 3):
        cv2.imwrite(str(color_dir / f"{index}.png"), color)
        cv2.imwrite(str(depth_dir / f"{index}.png"), depth)
    pose_file = tmp_path / "pose.txt"
    pose_file.write_text("0 0 0 0 0 0 1\n1 0 0 0 0 0 1\n", encoding="utf-8")
    output_path = tmp_path / "cloud.ply"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "examples/ch13_dense_mapping/rgbd_fusion.py",
            "--color-dir",
            str(color_dir),
            "--depth-dir",
            str(depth_dir),
            "--pose-file",
            str(pose_file),
            "--output",
            str(output_path),
            "--intrinsics",
            "100",
            "100",
            "0",
            "0",
            "--depth-scale",
            "1000",
        ],
        cwd=repo_root,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    text = output_path.read_text(encoding="utf-8")
    assert "frame count: 2" in result.stdout
    assert "point count: 2" in result.stdout
    assert "element vertex 2" in text
    assert "0.000000000 0.000000000 1.000000000" in text
    assert "1.000000000 0.000000000 1.000000000" in text
