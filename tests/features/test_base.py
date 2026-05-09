import cv2
import numpy as np

from slam.features.base import matches_to_points


def test_matches_to_points_uses_query_and_train_indices():
    keypoints0 = [cv2.KeyPoint(1.0, 2.0, 1.0), cv2.KeyPoint(3.0, 4.0, 1.0)]
    keypoints1 = [cv2.KeyPoint(5.0, 6.0, 1.0), cv2.KeyPoint(7.0, 8.0, 1.0)]
    matches = [cv2.DMatch(_queryIdx=1, _trainIdx=0, _distance=12.5)]

    points0, points1, distances = matches_to_points(keypoints0, keypoints1, matches)

    np.testing.assert_allclose(points0, [[3.0, 4.0]])
    np.testing.assert_allclose(points1, [[5.0, 6.0]])
    np.testing.assert_allclose(distances, [12.5])
