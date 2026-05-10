import numpy as np
import pytest

from slam.geometry.registration import estimate_rigid_transform_3d
from slam.geometry.transforms import make_transform, rotation_matrix_from_rotvec, transform_points


def test_estimate_rigid_transform_3d_recovers_known_transform():
    source = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [-0.5, 0.25, 1.5],
        ]
    )
    rotation = rotation_matrix_from_rotvec(np.array([0.1, -0.2, 0.05]))
    translation = np.array([0.3, -0.1, 0.2])
    target = transform_points(make_transform(rotation, translation), source)

    result = estimate_rigid_transform_3d(source, target)

    np.testing.assert_allclose(result.rotation, rotation, atol=1e-12)
    np.testing.assert_allclose(result.translation, translation, atol=1e-12)
    np.testing.assert_allclose(result.transform, make_transform(rotation, translation), atol=1e-12)
    assert result.rmse < 1e-12


def test_estimate_rigid_transform_3d_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 3"):
        estimate_rigid_transform_3d(np.zeros((2, 3)), np.zeros((2, 3)))
