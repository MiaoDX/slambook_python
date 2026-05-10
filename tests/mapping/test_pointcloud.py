import numpy as np
import pytest

from slam.geometry.transforms import make_transform
from slam.mapping.pointcloud import (
    estimate_normals,
    fuse_point_clouds,
    occupancy_voxel_grid,
    voxel_downsample,
    write_occupancy_npz,
    write_ply_ascii,
)


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


def test_write_ply_ascii_can_include_normals(tmp_path):
    path = tmp_path / "cloud_normals.ply"
    points = np.array([[1.0, 2.0, 3.0]])
    normals = np.array([[0.0, 0.0, 1.0]])

    write_ply_ascii(path, points, normals=normals)

    text = path.read_text(encoding="utf-8")
    assert "property float nx" in text
    assert "1.000000000 2.000000000 3.000000000 0.000000000 0.000000000 1.000000000" in text


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


def test_estimate_normals_recovers_planar_normals():
    grid_x, grid_y = np.meshgrid(np.arange(3), np.arange(3))
    points = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])

    normals = estimate_normals(points, k=5)

    np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (len(points), 1)), atol=1e-12)


def test_estimate_normals_orients_toward_viewpoint():
    grid_x, grid_y = np.meshgrid(np.arange(3), np.arange(3))
    points = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])

    normals = estimate_normals(points, k=5, viewpoint=np.array([0.0, 0.0, -1.0]))

    assert np.all(normals[:, 2] < 0.0)


def test_estimate_normals_requires_three_points():
    with pytest.raises(ValueError, match="at least 3 points"):
        estimate_normals(np.zeros((2, 3)), k=3)


def test_occupancy_voxel_grid_groups_points_and_writes_npz(tmp_path):
    points = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.0, 0.0]])

    grid = occupancy_voxel_grid(points, voxel_size=1.0)
    path = tmp_path / "occupancy.npz"
    write_occupancy_npz(path, grid)
    loaded = np.load(path)

    np.testing.assert_array_equal(grid.indices, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(grid.counts, [2, 1])
    np.testing.assert_allclose(grid.centers, [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])
    np.testing.assert_array_equal(loaded["counts"], [2, 1])
