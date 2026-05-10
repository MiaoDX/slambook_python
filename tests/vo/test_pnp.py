import cv2
import numpy as np

from slam.vo.pnp import project_points, solve_pnp, solve_pnp_ransac


def _rotation_error(rotation_a, rotation_b):
    delta = rotation_a @ rotation_b.T
    value = (np.trace(delta) - 1.0) / 2.0
    return np.arccos(np.clip(value, -1.0, 1.0))


def test_solve_pnp_recovers_pose_from_synthetic_correspondences():
    camera_matrix = np.array(
        [
            [700.0, 0.0, 320.0],
            [0.0, 710.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    points_3d = np.array(
        [
            [-1.0, -0.6, 4.0],
            [-0.2, -0.7, 4.5],
            [0.6, -0.3, 5.0],
            [1.0, 0.2, 5.5],
            [-0.8, 0.4, 6.0],
            [0.0, 0.7, 6.5],
            [0.9, 0.6, 7.0],
            [-0.4, 0.1, 7.5],
        ]
    )
    rotation, _ = cv2.Rodrigues(np.array([0.03, -0.08, 0.02]))
    translation = np.array([[0.3], [-0.1], [0.4]])
    points_2d = project_points(points_3d, rotation, translation, camera_matrix)

    result = solve_pnp(points_3d, points_2d, camera_matrix)

    assert _rotation_error(result.rotation, rotation) < 1e-5
    np.testing.assert_allclose(result.translation, translation, atol=1e-5)
    assert result.inlier_count == len(points_3d)


def test_solve_pnp_ransac_rejects_outlier_correspondences():
    camera_matrix = np.array(
        [
            [700.0, 0.0, 320.0],
            [0.0, 710.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    points_3d = np.array(
        [
            [-1.0, -0.6, 4.0],
            [-0.2, -0.7, 4.5],
            [0.6, -0.3, 5.0],
            [1.0, 0.2, 5.5],
            [-0.8, 0.4, 6.0],
            [0.0, 0.7, 6.5],
            [0.9, 0.6, 7.0],
            [-0.4, 0.1, 7.5],
            [0.4, -0.5, 8.0],
            [-0.6, 0.8, 8.5],
        ]
    )
    rotation, _ = cv2.Rodrigues(np.array([0.03, -0.08, 0.02]))
    translation = np.array([[0.3], [-0.1], [0.4]])
    points_2d = project_points(points_3d, rotation, translation, camera_matrix)
    points_2d[-2:] += np.array([[150.0, -80.0], [-120.0, 90.0]])

    result = solve_pnp_ransac(
        points_3d,
        points_2d,
        camera_matrix,
        iterations_count=200,
        reprojection_error=2.0,
    )

    assert result.inlier_count >= 8
    assert not result.mask[-1]
    assert not result.mask[-2]
    assert _rotation_error(result.rotation, rotation) < 1e-5
    np.testing.assert_allclose(result.translation, translation, atol=1e-5)
