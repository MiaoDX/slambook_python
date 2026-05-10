import builtins
import sys
import types

import numpy as np
import pytest

from slam.optimization.bundle_adjustment import BALObservation, BALProblem
from slam.optimization.gtsam_backend import (
    OptionalBackendDependencyError,
    optimize_bundle_adjustment_gtsam,
    optimize_pose_graph_gtsam,
    require_gtsam,
)
from slam.optimization.pose_graph import PoseGraph, PoseGraphEdge, PoseGraphVertex
from slam.optimization.pycolmap_backend import require_pycolmap, run_pycolmap_reconstruction


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


def test_run_pycolmap_reconstruction_uses_reference_pipeline(monkeypatch, tmp_path):
    fake_pycolmap = _install_fake_pycolmap(monkeypatch)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "pycolmap"

    result = run_pycolmap_reconstruction(
        image_dir,
        output_dir,
        image_names=["0001.png"],
        extraction_options={"sift": {"max_num_features": 512}},
        matching_method="sequential",
        matching_options={"sift": {"max_ratio": 0.8}},
        mapper_options={"min_num_matches": 12},
        device="cpu",
    )

    assert result.reconstruction_count == 1
    assert result.reconstruction_ids == (2,)
    assert result.database_path == output_dir / "database.db"
    assert result.reconstruction_paths == (output_dir,)
    assert result.summaries == ("fake reconstruction",)
    assert fake_pycolmap.calls == [
        (
            "extract_features",
            output_dir / "database.db",
            image_dir,
            {
                "image_names": ("0001.png",),
                "extraction_options": {"sift": {"max_num_features": 512}},
                "device": "cpu",
            },
        ),
        (
            "match_sequential",
            output_dir / "database.db",
            {
                "matching_options": {"sift": {"max_ratio": 0.8}},
                "device": "cpu",
            },
        ),
        (
            "incremental_mapping",
            output_dir / "database.db",
            image_dir,
            output_dir,
            {"options": {"min_num_matches": 12}},
        ),
    ]
    assert result.reconstructions[0].written_path == output_dir


def test_optimize_bundle_adjustment_gtsam_uses_optional_backend(monkeypatch):
    fake_gtsam = _install_fake_gtsam(monkeypatch)
    problem = BALProblem(
        camera_params=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0]]),
        points_3d=np.array([[0.5, 0.0, -5.0]]),
        observations=[BALObservation(camera_index=0, point_index=0, xy=np.array([0.0, 0.0]))],
    )

    result = optimize_bundle_adjustment_gtsam(
        problem,
        optimize_cameras=False,
        optimize_points=True,
        max_iterations=3,
    )

    assert result.success
    assert result.final_rmse < result.initial_rmse
    np.testing.assert_allclose(result.problem.points_3d, [[0.0, 0.0, -5.0]])
    assert fake_gtsam.last_params.max_iterations == 3
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

    class Point2:
        def __init__(self, x, y):
            self.vector = np.array([x, y], dtype=np.float64)

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
            self.points = {}

        def insert(self, key, pose):
            if isinstance(pose, Pose3):
                self.poses[key] = pose
            else:
                self.points[key] = pose

        def atPose3(self, key):
            return self.poses[key]

        def atPoint3(self, key):
            return self.points[key]

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
            for key, point in self.initial.points.items():
                vector = np.asarray(point, dtype=np.float64)
                if key >= 2_000_000:
                    vector = np.array([0.0, 0.0, -5.0], dtype=np.float64)
                optimized.insert(key, Point3(vector))
            return optimized

    fake.Rot3 = Rot3
    fake.Point2 = Point2
    fake.Point3 = Point3
    fake.Pose3 = Pose3
    fake.Values = Values
    fake.NonlinearFactorGraph = NonlinearFactorGraph
    fake.PriorFactorPose3 = lambda key, pose, noise: ("prior", key, pose, noise)
    fake.PriorFactorPoint3 = lambda key, point, noise: ("point_prior", key, point, noise)
    fake.BetweenFactorPose3 = lambda key0, key1, pose, noise: ("between", key0, key1, pose, noise)
    fake.Cal3Bundler = lambda focal, k1, k2, u0, v0: ("cal3bundler", focal, k1, k2, u0, v0)
    fake.GenericProjectionFactorCal3Bundler = (
        lambda measured, noise, pose_key, point_key, calibration: (
            "projection",
            measured,
            noise,
            pose_key,
            point_key,
            calibration,
        )
    )
    fake.LevenbergMarquardtParams = LevenbergMarquardtParams
    fake.LevenbergMarquardtOptimizer = LevenbergMarquardtOptimizer
    fake.noiseModel = types.SimpleNamespace(
        Gaussian=types.SimpleNamespace(Information=lambda information: ("information", information)),
        Isotropic=types.SimpleNamespace(Sigma=lambda size, sigma: ("isotropic", size, sigma)),
        Constrained=types.SimpleNamespace(All=lambda size: ("constrained", size)),
    )
    monkeypatch.setitem(sys.modules, "gtsam", fake)
    return fake


def _install_fake_pycolmap(monkeypatch):
    fake = types.ModuleType("pycolmap")
    fake.calls = []

    class Reconstruction:
        def __init__(self):
            self.written_path = None

        def write(self, path):
            self.written_path = path

        def summary(self):
            return "fake reconstruction"

    def extract_features(database_path, image_dir, **kwargs):
        fake.calls.append(("extract_features", database_path, image_dir, kwargs))

    def match_sequential(database_path, **kwargs):
        fake.calls.append(("match_sequential", database_path, kwargs))

    def incremental_mapping(database_path, image_dir, output_dir, **kwargs):
        fake.calls.append(("incremental_mapping", database_path, image_dir, output_dir, kwargs))
        return {2: Reconstruction()}

    fake.extract_features = extract_features
    fake.match_sequential = match_sequential
    fake.incremental_mapping = incremental_mapping
    monkeypatch.setitem(sys.modules, "pycolmap", fake)
    return fake
