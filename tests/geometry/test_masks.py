import numpy as np

from slam.geometry.masks import normalize_mask


def test_normalize_mask_treats_one_and_255_as_inliers():
    mask_01 = np.array([[0], [1], [1], [0]], dtype=np.uint8)
    mask_0255 = np.array([[0], [255], [255], [0]], dtype=np.uint8)

    np.testing.assert_array_equal(normalize_mask(mask_01), [False, True, True, False])
    np.testing.assert_array_equal(normalize_mask(mask_0255), [False, True, True, False])


def test_normalize_mask_builds_all_true_mask_when_none_has_expected_length():
    np.testing.assert_array_equal(normalize_mask(None, expected_length=3), [True, True, True])
