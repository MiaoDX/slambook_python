import builtins
import sys
import types

import numpy as np
import pytest

from slam.geometry.transforms import make_transform
from slam.viz.open3d_viz import (
    OptionalVisualizationDependencyError,
    reconstruct_mesh_poisson,
    require_open3d,
    write_triangle_mesh_open3d,
)
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


def test_reconstruct_mesh_poisson_uses_open3d_backend(monkeypatch, tmp_path):
    fake_open3d = _install_fake_open3d(monkeypatch)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normals = np.tile([0.0, 0.0, 1.0], (3, 1))

    mesh, densities = reconstruct_mesh_poisson(points, normals=normals, depth=5)
    write_triangle_mesh_open3d(tmp_path / "mesh.ply", mesh)

    assert mesh.depth == 5
    assert mesh.point_count == 3
    np.testing.assert_allclose(densities, [0.1, 0.2, 0.3])
    assert fake_open3d.last_point_cloud.estimated_normals is False
    np.testing.assert_allclose(fake_open3d.last_point_cloud.normals, normals)
    assert fake_open3d.written_paths == [str(tmp_path / "mesh.ply")]


def test_reconstruct_mesh_poisson_estimates_normals_when_missing(monkeypatch):
    fake_open3d = _install_fake_open3d(monkeypatch)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    reconstruct_mesh_poisson(points, depth=4)

    assert fake_open3d.last_point_cloud.estimated_normals is True


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


def _install_fake_open3d(monkeypatch):
    fake = types.SimpleNamespace(written_paths=[])

    class FakePointCloud:
        def __init__(self):
            self.points = None
            self.colors = None
            self.normals = None
            self.estimated_normals = False

        def estimate_normals(self):
            self.estimated_normals = True

    class FakeTriangleMesh:
        def __init__(self, depth, point_count):
            self.depth = depth
            self.point_count = point_count

        @staticmethod
        def create_from_point_cloud_poisson(point_cloud, depth):
            fake.last_point_cloud = point_cloud
            return FakeTriangleMesh(depth, len(point_cloud.points)), np.array([0.1, 0.2, 0.3])

    fake.geometry = types.SimpleNamespace(PointCloud=FakePointCloud, TriangleMesh=FakeTriangleMesh)
    fake.utility = types.SimpleNamespace(Vector3dVector=lambda values: np.asarray(values, dtype=np.float64))
    fake.io = types.SimpleNamespace(write_triangle_mesh=lambda path, mesh: fake.written_paths.append(path) or True)
    monkeypatch.setitem(sys.modules, "open3d", fake)
    return fake
