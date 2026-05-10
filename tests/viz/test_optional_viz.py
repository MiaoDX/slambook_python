import builtins

import pytest

from slam.viz.open3d_viz import OptionalVisualizationDependencyError, require_open3d
from slam.viz.rerun_viz import require_rerun


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
