import numpy as np
import pytest

from slam.camera.stereo import disparity_to_depth


def test_disparity_to_depth_uses_positive_baseline_convention():
    disparity = np.array([[20.0, 40.0], [80.0, 0.0]])

    depth = disparity_to_depth(disparity, focal_length=500.0, baseline=0.2)

    np.testing.assert_allclose(depth[:2, :1].ravel(), [5.0, 1.25])
    assert depth[0, 1] == 2.5
    assert np.isnan(depth[1, 1])


def test_disparity_to_depth_respects_min_disparity():
    disparity = np.array([0.5, 1.0, 2.0])

    depth = disparity_to_depth(disparity, focal_length=100.0, baseline=0.1, min_disparity=1.0)

    assert np.isnan(depth[0])
    assert np.isnan(depth[1])
    assert depth[2] == 5.0


def test_disparity_to_depth_rejects_non_positive_camera_values():
    with pytest.raises(ValueError, match="focal_length"):
        disparity_to_depth(np.array([1.0]), focal_length=0.0, baseline=0.1)

    with pytest.raises(ValueError, match="baseline"):
        disparity_to_depth(np.array([1.0]), focal_length=100.0, baseline=0.0)
