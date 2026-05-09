import numpy as np

from slam.camera.pinhole import CameraIntrinsics
from slam.geometry.transforms import make_transform
from slam.mapping.rgbd import rgbd_to_point_cloud


def test_rgbd_to_point_cloud_returns_expected_points_and_colors():
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0)
    depth = np.array([[1000, 0], [2000, 3000]], dtype=np.uint16)
    color = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    cloud = rgbd_to_point_cloud(color, depth, intrinsics, depth_scale=1000.0)

    assert cloud.points.shape == (3, 3)
    np.testing.assert_allclose(
        cloud.points,
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.02, 2.0],
            [0.03, 0.03, 3.0],
        ],
    )
    np.testing.assert_array_equal(cloud.colors, [[255, 0, 0], [0, 0, 255], [255, 255, 255]])


def test_rgbd_to_point_cloud_depth_trunc_filters_far_points():
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0)
    depth = np.array([[1000, 5000]], dtype=np.uint16)

    cloud = rgbd_to_point_cloud(None, depth, intrinsics, depth_scale=1000.0, depth_trunc=2.0)

    assert cloud.points.shape == (1, 3)
    np.testing.assert_allclose(cloud.points[0], [0.0, 0.0, 1.0])


def test_rgbd_point_cloud_transform_moves_points_consistently():
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0)
    depth = np.array([[1000]], dtype=np.uint16)
    cloud = rgbd_to_point_cloud(None, depth, intrinsics, depth_scale=1000.0)
    transform = make_transform(np.eye(3), np.array([1.0, 2.0, 3.0]))

    moved = cloud.transform(transform)

    np.testing.assert_allclose(moved.points, [[1.0, 2.0, 4.0]])
