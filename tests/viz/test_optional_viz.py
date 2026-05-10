import builtins
import sys
import types

import numpy as np
import pytest

from slam.geometry.transforms import make_transform
from slam.viz.open3d_viz import OptionalVisualizationDependencyError, require_open3d
from slam.viz.rerun_viz import log_trajectory_rerun, require_rerun


def test_require_open3d_has_install_guidance(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "open3d":
            raise ImportError("missing open3d for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OptionalVisualizationDependencyError, match=r"pip install -e \.\[3d\]"):
        require_open3d()


def test_require_rerun_has_install_guidance(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rerun":
            raise ImportError("missing rerun for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OptionalVisualizationDependencyError, match=r"pip install -e \.\[modern\]"):
        require_rerun()


def test_log_trajectory_rerun_uses_line_strip(monkeypatch):
    calls = []

    class FakeLineStrips3D:
        def __init__(self, strips, **kwargs):
            self.strips = strips
            self.kwargs = kwargs

    fake_rerun = types.SimpleNamespace(
        LineStrips3D=FakeLineStrips3D,
        log=lambda entity_path, archetype: calls.append((entity_path, archetype)),
    )
    monkeypatch.setitem(sys.modules, "rerun", fake_rerun)
    poses = [
        make_transform(np.eye(3), np.array([0.0, 0.0, 0.0])),
        make_transform(np.eye(3), np.array([1.0, 2.0, 3.0])),
    ]

    log_trajectory_rerun("world/trajectory", poses, color=[0, 128, 255], radius=0.02)

    assert calls[0][0] == "world/trajectory"
    np.testing.assert_allclose(calls[0][1].strips[0], [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    assert calls[0][1].kwargs == {"colors": [0, 128, 255], "radii": 0.02}
