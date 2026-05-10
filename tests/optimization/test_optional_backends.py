import builtins

import pytest

from slam.optimization.gtsam_backend import OptionalBackendDependencyError, require_gtsam


def test_require_gtsam_has_backend_install_guidance(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gtsam":
            raise ImportError("missing gtsam for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OptionalBackendDependencyError, match=r"pip install -e \.\[backend\]"):
        require_gtsam()
