import numpy as np
import pytest

from slam.mapping.pointcloud import write_ply_ascii


def test_write_ply_ascii_writes_points_and_colors(tmp_path):
    path = tmp_path / "cloud.ply"
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    colors = np.array([[255, 0, 0], [0, 128, 255]])

    write_ply_ascii(path, points, colors)

    text = path.read_text(encoding="utf-8")
    assert "element vertex 2" in text
    assert "property uchar red" in text
    assert "1.000000000 2.000000000 3.000000000 255 0 0" in text
    assert "4.000000000 5.000000000 6.000000000 0 128 255" in text


def test_write_ply_ascii_rejects_color_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="colors"):
        write_ply_ascii(tmp_path / "bad.ply", np.zeros((2, 3)), np.zeros((1, 3)))
