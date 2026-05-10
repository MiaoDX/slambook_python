import numpy as np
import pytest

from slam.geometry.transforms import make_transform
from slam.mapping.pointcloud import fuse_point_clouds, voxel_downsample, write_ply_ascii


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


def test_fuse_point_clouds_applies_known_poses_and_colors():
    cloud0 = (np.array([[0.0, 0.0, 1.0]]), np.array([[255, 0, 0]], dtype=np.uint8))
    cloud1 = (np.array([[0.0, 0.0, 2.0]]), np.array([[0, 255, 0]], dtype=np.uint8))
    transform = make_transform(np.eye(3), np.array([1.0, 2.0, 3.0]))

    points, colors = fuse_point_clouds([cloud0, cloud1], transforms_wb=[np.eye(4), transform])

    np.testing.assert_allclose(points, [[0.0, 0.0, 1.0], [1.0, 2.0, 5.0]])
    np.testing.assert_array_equal(colors, [[255, 0, 0], [0, 255, 0]])


def test_voxel_downsample_uses_centroids_and_average_colors():
    points = np.array([[0.01, 0.0, 0.0], [0.09, 0.0, 0.0], [0.21, 0.0, 0.0]])
    colors = np.array([[0, 0, 0], [100, 0, 0], [0, 0, 200]], dtype=np.uint8)

    down_points, down_colors = voxel_downsample(points, colors, voxel_size=0.2)

    np.testing.assert_allclose(down_points, [[0.05, 0.0, 0.0], [0.21, 0.0, 0.0]])
    np.testing.assert_array_equal(down_colors, [[50, 0, 0], [0, 0, 200]])
