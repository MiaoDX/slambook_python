import builtins

import numpy as np
import pytest

from slam.geometry.transforms import make_transform
from slam.viz.matplotlib_viz import require_matplotlib, save_trajectory_plot, trajectory_xyz
from slam.viz.open3d_viz import OptionalVisualizationDependencyError


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


def test_require_matplotlib_has_install_guidance(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"matplotlib", "matplotlib.pyplot"}:
            raise ImportError("missing matplotlib for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OptionalVisualizationDependencyError, match=r"pip install -e \.\[core\]"):
        require_matplotlib()


def test_save_trajectory_plot_writes_image_when_matplotlib_is_available(tmp_path):
    pytest.importorskip("matplotlib")
    path = tmp_path / "trajectory.png"
    poses = [
        make_transform(np.eye(3), np.array([0.0, 0.0, 0.0])),
        make_transform(np.eye(3), np.array([1.0, 0.0, 0.0])),
    ]

    save_trajectory_plot(path, poses, label="test")

    assert path.stat().st_size > 0
