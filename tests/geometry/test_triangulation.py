import cv2
import numpy as np

from slam.geometry.triangulation import triangulate_points
from slam.vo.pnp import project_points


def test_triangulate_points_returns_finite_camera0_points():
    camera_matrix = np.array(
        [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    points_3d = np.array(
        [
            [-0.5, -0.2, 4.0],
            [0.4, -0.1, 5.0],
            [0.1, 0.3, 6.0],
            [0.8, 0.4, 7.0],
            [-0.3, 0.5, 8.0],
        ]
    )
    rotation_10, _ = cv2.Rodrigues(np.array([0.01, -0.04, 0.02]))
    translation_10 = np.array([[0.5], [0.0], [0.1]])

    points0 = project_points(points_3d, np.eye(3), np.zeros((3, 1)), camera_matrix)
    points1 = project_points(points_3d, rotation_10, translation_10, camera_matrix)

    triangulated = triangulate_points(points0, points1, camera_matrix, rotation_10, translation_10)

    assert triangulated.shape == points_3d.shape
    assert np.isfinite(triangulated).all()
    np.testing.assert_allclose(triangulated, points_3d, atol=1e-6)
