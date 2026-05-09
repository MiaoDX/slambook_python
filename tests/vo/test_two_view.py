import cv2
import numpy as np

from slam.vo.pnp import project_points
from slam.vo.two_view import estimate_essential, estimate_fundamental, recover_relative_pose


def _rotation_error(rotation_a, rotation_b):
    delta = rotation_a @ rotation_b.T
    value = (np.trace(delta) - 1.0) / 2.0
    return np.arccos(np.clip(value, -1.0, 1.0))


def test_two_view_pose_recovers_plausible_synthetic_relative_pose():
    camera_matrix = np.array(
        [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rng = np.random.default_rng(7)
    points_3d = np.column_stack(
        [
            rng.uniform(-1.5, 1.5, 80),
            rng.uniform(-1.0, 1.0, 80),
            rng.uniform(4.0, 9.0, 80),
        ]
    )
    rotation_10, _ = cv2.Rodrigues(np.array([0.0, 0.06, 0.01]))
    translation_10 = np.array([[0.8], [0.0], [0.1]])

    points0 = project_points(points_3d, np.eye(3), np.zeros((3, 1)), camera_matrix)
    points1 = project_points(points_3d, rotation_10, translation_10, camera_matrix)

    fundamental = estimate_fundamental(points0, points1)
    essential = estimate_essential(points0, points1, camera_matrix)
    pose = recover_relative_pose(essential.matrix, points0, points1, camera_matrix, mask=essential.mask)

    assert fundamental.inlier_count >= 70
    assert essential.inlier_count >= 70
    assert pose.inlier_count >= 70
    assert _rotation_error(pose.rotation_10, rotation_10) < 0.02

    expected_t = translation_10.ravel() / np.linalg.norm(translation_10)
    actual_t = pose.translation_10.ravel() / np.linalg.norm(pose.translation_10)
    assert abs(float(np.dot(expected_t, actual_t))) > 0.99
