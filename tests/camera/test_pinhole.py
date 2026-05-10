import numpy as np

from slam.camera.pinhole import CameraIntrinsics, DistortionCoefficients


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


def test_default_distortion_is_zero_and_projection_is_opt_in():
    intrinsics = CameraIntrinsics(fx=100.0, fy=110.0, cx=10.0, cy=20.0)
    points = np.array([[0.5, -0.25, 2.0]])

    assert intrinsics.distortion.is_zero
    np.testing.assert_allclose(
        intrinsics.camera_to_pixel(points, apply_distortion=True),
        intrinsics.camera_to_pixel(points),
    )


def test_distortion_coefficients_apply_radial_and_tangential_terms():
    distortion = DistortionCoefficients(k1=0.5, p1=0.01, p2=-0.02)
    points = np.array([[0.1, 0.2]])

    distorted = distortion.distort_normalized(points)

    r2 = 0.1 * 0.1 + 0.2 * 0.2
    radial = 1.0 + 0.5 * r2
    expected_x = 0.1 * radial + 2.0 * 0.01 * 0.1 * 0.2 - 0.02 * (r2 + 2.0 * 0.1 * 0.1)
    expected_y = 0.2 * radial + 0.01 * (r2 + 2.0 * 0.2 * 0.2) - 2.0 * 0.02 * 0.1 * 0.2
    np.testing.assert_allclose(distorted, [[expected_x, expected_y]])
    np.testing.assert_allclose(distortion.as_opencv(), [0.5, 0.0, 0.01, -0.02, 0.0])


def test_camera_projection_can_apply_distortion():
    intrinsics = CameraIntrinsics(
        fx=100.0,
        fy=100.0,
        cx=10.0,
        cy=20.0,
        distortion=DistortionCoefficients(k1=0.5),
    )
    point = np.array([[0.5, 1.0, 5.0]])

    pixel = intrinsics.camera_to_pixel(point, apply_distortion=True)

    r2 = 0.1 * 0.1 + 0.2 * 0.2
    radial = 1.0 + 0.5 * r2
    np.testing.assert_allclose(pixel, [[100.0 * 0.1 * radial + 10.0, 100.0 * 0.2 * radial + 20.0]])
