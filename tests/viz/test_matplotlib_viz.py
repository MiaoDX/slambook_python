import numpy as np
import pytest

from slam.geometry.transforms import make_transform
from slam.viz.matplotlib_viz import trajectory_xyz


def test_trajectory_xyz_extracts_translations():
    poses = [
        make_transform(np.eye(3), np.array([0.0, 0.0, 0.0])),
        make_transform(np.eye(3), np.array([1.0, 2.0, 3.0])),
    ]

    xyz = trajectory_xyz(poses)

    np.testing.assert_allclose(xyz, [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])


def test_trajectory_xyz_rejects_bad_pose_shape():
    with pytest.raises(ValueError, match="4x4"):
        trajectory_xyz([np.eye(3)])
