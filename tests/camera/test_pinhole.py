import numpy as np

from slam.camera.pinhole import CameraIntrinsics


def test_pixel_camera_projection_round_trip():
    intrinsics = CameraIntrinsics(fx=500.0, fy=510.0, cx=320.0, cy=240.0)
    pixels = np.array([[320.0, 240.0], [420.0, 291.0], [120.0, 138.0]])
    depths = np.array([2.0, 4.0, 5.0])

    points = intrinsics.pixel_to_camera(pixels, depths)
    projected = intrinsics.camera_to_pixel(points)

    np.testing.assert_allclose(projected, pixels, atol=1e-12)


def test_from_matrix_preserves_intrinsic_values():
    matrix = np.array(
        [
            [520.9, 0.0, 325.1],
            [0.0, 521.0, 249.7],
            [0.0, 0.0, 1.0],
        ]
    )

    intrinsics = CameraIntrinsics.from_matrix(matrix)

    assert intrinsics.fx == 520.9
    assert intrinsics.fy == 521.0
    assert intrinsics.cx == 325.1
    assert intrinsics.cy == 249.7
    np.testing.assert_allclose(intrinsics.matrix, matrix)
