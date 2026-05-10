import builtins
import sys
import types

import numpy as np
import pytest

from slam.optimization.gtsam_backend import OptionalBackendDependencyError, optimize_pose_graph_gtsam, require_gtsam
from slam.optimization.pose_graph import PoseGraph, PoseGraphEdge, PoseGraphVertex
from slam.optimization.pycolmap_backend import require_pycolmap


def test_require_gtsam_has_backend_install_guidance(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gtsam":
            raise ImportError("missing gtsam for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OptionalBackendDependencyError, match=r"pip install -e \.\[backend\]"):
        require_gtsam()


def test_require_pycolmap_has_modern_install_guidance(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pycolmap":
            raise ImportError("missing pycolmap for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OptionalBackendDependencyError, match=r"pip install -e \.\[modern\]"):
        require_pycolmap()


def test_optimize_pose_graph_gtsam_uses_optional_backend(monkeypatch):
    fake_gtsam = _install_fake_gtsam(monkeypatch)
    transform0 = np.eye(4, dtype=np.float64)
    transform1 = np.eye(4, dtype=np.float64)
    transform1[0, 3] = 2.0
    measurement = np.eye(4, dtype=np.float64)
    measurement[0, 3] = 1.0
    graph = PoseGraph(
        vertices={
            0: PoseGraphVertex(id=0, transform_wi=transform0),
            1: PoseGraphVertex(id=1, transform_wi=transform1),
        },
        edges=[PoseGraphEdge(from_id=0, to_id=1, measurement_ij=measurement, information=np.eye(6))],
    )

    result = optimize_pose_graph_gtsam(graph, fixed_vertex_id=0, max_iterations=5)

    assert result.success
    assert result.final_error < result.initial_error
    np.testing.assert_allclose(result.graph.vertices[1].transform_wi[:3, 3], [1.0, 0.0, 0.0])
    assert fake_gtsam.last_params.max_iterations == 5
    assert len(fake_gtsam.last_graph.factors) == 2


def _install_fake_gtsam(monkeypatch):
    fake = types.SimpleNamespace()

    class Rot3:
        def __init__(self, matrix):
            self._matrix = np.asarray(matrix, dtype=np.float64)

        def matrix(self):
            return self._matrix

    class Point3:
        def __init__(self, x, y=None, z=None):
            if y is None and z is None:
                self.vector = np.asarray(x, dtype=np.float64).reshape(3)
            else:
                self.vector = np.array([x, y, z], dtype=np.float64)

        def __array__(self, dtype=None):
            return self.vector.astype(dtype) if dtype is not None else self.vector

    class Pose3:
        def __init__(self, rotation=None, point=None):
            self.transform = np.eye(4, dtype=np.float64)
            if rotation is not None:
                self.transform[:3, :3] = rotation.matrix()
            if point is not None:
                self.transform[:3, 3] = np.asarray(point, dtype=np.float64).reshape(3)

        def matrix(self):
            return self.transform

    class Values:
        def __init__(self):
            self.poses = {}

        def insert(self, key, pose):
            self.poses[key] = pose

        def atPose3(self, key):
            return self.poses[key]

    class NonlinearFactorGraph:
        def __init__(self):
            self.factors = []

        def add(self, factor):
            self.factors.append(factor)

        def error(self, values):
            return 0.0

    class LevenbergMarquardtParams:
        def __init__(self):
            self.max_iterations = None

        def setMaxIterations(self, value):
            self.max_iterations = value

    class LevenbergMarquardtOptimizer:
        def __init__(self, factor_graph, initial, params=None):
            fake.last_graph = factor_graph
            fake.last_params = params
            self.initial = initial

        def optimize(self):
            optimized = Values()
            for key, pose in self.initial.poses.items():
                transform = pose.matrix().copy()
                if key == 1:
                    transform[:3, 3] = [1.0, 0.0, 0.0]
                optimized.insert(key, Pose3(Rot3(transform[:3, :3]), Point3(transform[:3, 3])))
            return optimized

    fake.Rot3 = Rot3
    fake.Point3 = Point3
    fake.Pose3 = Pose3
    fake.Values = Values
    fake.NonlinearFactorGraph = NonlinearFactorGraph
    fake.PriorFactorPose3 = lambda key, pose, noise: ("prior", key, pose, noise)
    fake.BetweenFactorPose3 = lambda key0, key1, pose, noise: ("between", key0, key1, pose, noise)
    fake.LevenbergMarquardtParams = LevenbergMarquardtParams
    fake.LevenbergMarquardtOptimizer = LevenbergMarquardtOptimizer
    fake.noiseModel = types.SimpleNamespace(
        Gaussian=types.SimpleNamespace(Information=lambda information: ("information", information)),
        Constrained=types.SimpleNamespace(All=lambda size: ("constrained", size)),
    )
    monkeypatch.setitem(sys.modules, "gtsam", fake)
    return fake
