import numpy as np

from slam.camera.pinhole import CameraIntrinsics
from slam.mapping.monocular_dense import (
    dense_depth_from_known_pose,
    estimate_depth_by_epipolar_search,
    ncc_score,
)


def test_ncc_score_returns_one_for_identical_patches():
    image = np.arange(100, dtype=np.float64).reshape(10, 10)

    score = ncc_score(image, image, np.array([5.0, 5.0]), np.array([5.0, 5.0]), window_radius=1)

    np.testing.assert_allclose(score, 1.0)


def test_estimate_depth_by_epipolar_search_recovers_synthetic_shift_depth():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(40, 40))
    current = np.zeros_like(reference)
    point = np.array([[20.0, 20.0]])
    current[17:24, 22:29] = reference[17:24, 17:24]
    camera_matrix = CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0).matrix
    transform_cur_ref = np.eye(4)
    transform_cur_ref[0, 3] = 0.1

    depths, scores, valid = estimate_depth_by_epipolar_search(
        reference,
        current,
        point,
        camera_matrix,
        transform_cur_ref,
        min_depth=1.0,
        max_depth=3.0,
        depth_samples=9,
        window_radius=2,
        min_score=0.5,
    )

    assert valid[0]
    np.testing.assert_allclose(depths[0], 2.0)
    assert scores[0] > 0.9


def test_dense_depth_from_known_pose_returns_empty_when_no_texture():
    image = np.zeros((16, 16), dtype=np.float64)
    estimate = dense_depth_from_known_pose(
        image,
        image,
        CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0).matrix,
        np.eye(4),
        stride=4,
        gradient_threshold=1.0,
    )

    assert not np.any(estimate.valid)
