import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def test_ch5_rgbd_pointcloud_writes_tiny_ply(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    color = np.zeros((2, 2, 3), dtype=np.uint8)
    color[:, :] = [10, 20, 30]
    depth = np.array([[1000, 0], [2000, 3000]], dtype=np.uint16)
    color_path = tmp_path / "color.png"
    depth_path = tmp_path / "depth.png"
    output_path = tmp_path / "cloud.ply"
    cv2.imwrite(str(color_path), color)
    cv2.imwrite(str(depth_path), depth)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "examples/ch5_camera_image/rgbd_pointcloud.py",
            "--color",
            str(color_path),
            "--depth",
            str(depth_path),
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

    assert "point count: 3" in result.stdout
    assert "element vertex 3" in output_path.read_text(encoding="utf-8")
