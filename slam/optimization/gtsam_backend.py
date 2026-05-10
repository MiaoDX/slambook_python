"""Optional GTSAM backend entry points."""

from __future__ import annotations

import cv2
import numpy as np

from slam.optimization.bundle_adjustment import BALProblem, BundleAdjustmentResult, reprojection_rmse
from slam.optimization.pose_graph import (
    PoseGraph,
    PoseGraphOptimizationResult,
    PoseGraphVertex,
    total_edge_error,
)


class OptionalBackendDependencyError(ImportError):
    """Raised when an optional optimization backend is unavailable."""


def require_gtsam():
    """Import GTSAM or raise with project-specific install guidance."""

    try:
        import gtsam  # type: ignore
    except ImportError as exc:
        raise OptionalBackendDependencyError(
            "GTSAM is an optional backend. Install it with `pip install -e .[backend]` "
            "and verify that GTSAM wheels are available for your Python/platform."
        ) from exc
    return gtsam


def optimize_pose_graph_gtsam(
    graph: PoseGraph,
    *,
    fixed_vertex_id: int | None = None,
    max_iterations: int | None = None,
) -> PoseGraphOptimizationResult:
    """Optimize a `PoseGraph` with GTSAM when the optional backend is installed."""

    gtsam = require_gtsam()
    if fixed_vertex_id is None and graph.vertices:
        fixed_vertex_id = min(graph.vertices)

    factor_graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    for vertex_id in sorted(graph.vertices):
        initial.insert(_gtsam_key(vertex_id), _pose3_from_transform(gtsam, graph.vertices[vertex_id].transform_wi))

    if fixed_vertex_id is not None and fixed_vertex_id in graph.vertices:
        factor_graph.add(
            gtsam.PriorFactorPose3(
                _gtsam_key(fixed_vertex_id),
                _pose3_from_transform(gtsam, graph.vertices[fixed_vertex_id].transform_wi),
                _fixed_pose_noise(gtsam),
            )
        )

    for edge in graph.edges:
        factor_graph.add(
            gtsam.BetweenFactorPose3(
                _gtsam_key(edge.from_id),
                _gtsam_key(edge.to_id),
                _pose3_from_transform(gtsam, edge.measurement_ij),
                _edge_noise(gtsam, edge.information),
            )
        )

    optimizer = _make_lm_optimizer(gtsam, factor_graph, initial, max_iterations=max_iterations)
    optimized_values = optimizer.optimize()
    optimized_graph = PoseGraph(
        vertices={
            vertex_id: PoseGraphVertex(
                id=vertex_id,
                transform_wi=_transform_from_pose3(optimized_values.atPose3(_gtsam_key(vertex_id))),
            )
            for vertex_id in sorted(graph.vertices)
        },
        edges=graph.edges,
    )
    return PoseGraphOptimizationResult(
        graph=optimized_graph,
        initial_error=total_edge_error(graph),
        final_error=total_edge_error(optimized_graph),
        cost=_factor_graph_error(factor_graph, optimized_values),
        nfev=0,
        success=True,
        message="GTSAM Levenberg-Marquardt optimization completed.",
    )


def optimize_bundle_adjustment_gtsam(
    problem: BALProblem,
    *,
    optimize_cameras: bool = True,
    optimize_points: bool = True,
    fixed_camera_index: int | None = 0,
    measurement_sigma: float = 1.0,
    max_iterations: int | None = None,
) -> BundleAdjustmentResult:
    """Optimize BAL camera poses and/or points with GTSAM.

    BAL focal length and radial distortion parameters are kept fixed in this
    adapter. The SciPy backend remains the educational full 9-parameter BAL
    solver.
    """

    if not optimize_cameras and not optimize_points:
        rmse = reprojection_rmse(problem)
        return BundleAdjustmentResult(
            problem=problem,
            initial_rmse=rmse,
            final_rmse=rmse,
            cost=0.0,
            nfev=0,
            success=True,
            message="No parameters selected for optimization.",
        )

    gtsam = require_gtsam()
    factor_graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    for camera_index, camera_params in enumerate(problem.camera_params):
        pose = _pose3_from_bal_camera(gtsam, camera_params)
        initial.insert(_camera_key(camera_index), pose)
        if not optimize_cameras or camera_index == fixed_camera_index:
            factor_graph.add(gtsam.PriorFactorPose3(_camera_key(camera_index), pose, _fixed_pose_noise(gtsam)))

    for point_index, point in enumerate(problem.points_3d):
        point3 = _point3(gtsam, point)
        initial.insert(_point_key(point_index), point3)
        if not optimize_points:
            factor_graph.add(gtsam.PriorFactorPoint3(_point_key(point_index), point3, _fixed_point_noise(gtsam)))

    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, float(measurement_sigma))
    for observation in problem.observations:
        calibration, factor_cls = _projection_calibration_and_factor(
            gtsam,
            problem.camera_params[observation.camera_index],
        )
        factor_graph.add(
            factor_cls(
                _point2(gtsam, observation.xy),
                measurement_noise,
                _camera_key(observation.camera_index),
                _point_key(observation.point_index),
                calibration,
            )
        )

    optimizer = _make_lm_optimizer(gtsam, factor_graph, initial, max_iterations=max_iterations)
    optimized_values = optimizer.optimize()
    optimized_problem = _bal_problem_from_values(gtsam, problem, optimized_values, optimize_cameras, optimize_points)
    return BundleAdjustmentResult(
        problem=optimized_problem,
        initial_rmse=reprojection_rmse(problem),
        final_rmse=reprojection_rmse(optimized_problem),
        cost=_factor_graph_error(factor_graph, optimized_values),
        nfev=0,
        success=True,
        message="GTSAM bundle adjustment completed with fixed BAL intrinsics.",
    )


def _gtsam_key(vertex_id: int) -> int:
    return int(vertex_id)


def _camera_key(camera_index: int) -> int:
    return 1_000_000 + int(camera_index)


def _point_key(point_index: int) -> int:
    return 2_000_000 + int(point_index)


def _pose3_from_transform(gtsam, transform: np.ndarray):
    transform = np.asarray(transform, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    try:
        rot3 = gtsam.Rot3(rotation)
    except TypeError:
        rot3 = gtsam.Rot3(*rotation.reshape(-1).tolist())
    try:
        point3 = gtsam.Point3(*translation.tolist())
    except TypeError:
        point3 = gtsam.Point3(translation)
    return gtsam.Pose3(rot3, point3)


def _pose3_from_bal_camera(gtsam, camera_params: np.ndarray):
    camera_params = np.asarray(camera_params, dtype=np.float64).reshape(9)
    rotation_matrix, _ = cv2.Rodrigues(camera_params[:3])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = camera_params[3:6]
    return _pose3_from_transform(gtsam, transform)


def _transform_from_pose3(pose3) -> np.ndarray:
    if hasattr(pose3, "matrix"):
        return np.asarray(pose3.matrix(), dtype=np.float64)

    transform = np.eye(4, dtype=np.float64)
    rotation = pose3.rotation()
    if hasattr(rotation, "matrix"):
        transform[:3, :3] = np.asarray(rotation.matrix(), dtype=np.float64)
    else:
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(pose3.translation(), dtype=np.float64).reshape(3)
    return transform


def _point2(gtsam, point: np.ndarray):
    point = np.asarray(point, dtype=np.float64).reshape(2)
    try:
        return gtsam.Point2(*point.tolist())
    except (AttributeError, TypeError):
        return point


def _point3(gtsam, point: np.ndarray):
    point = np.asarray(point, dtype=np.float64).reshape(3)
    try:
        return gtsam.Point3(*point.tolist())
    except (AttributeError, TypeError):
        return gtsam.Point3(point)


def _point3_to_array(point3) -> np.ndarray:
    return np.asarray(point3, dtype=np.float64).reshape(3)


def _projection_calibration_and_factor(gtsam, camera_params: np.ndarray):
    camera_params = np.asarray(camera_params, dtype=np.float64).reshape(9)
    focal = float(camera_params[6])
    k1 = float(camera_params[7])
    k2 = float(camera_params[8])
    if hasattr(gtsam, "Cal3Bundler") and hasattr(gtsam, "GenericProjectionFactorCal3Bundler"):
        return (
            gtsam.Cal3Bundler(focal, k1, k2, 0.0, 0.0),
            gtsam.GenericProjectionFactorCal3Bundler,
        )
    return (
        gtsam.Cal3_S2(focal, focal, 0.0, 0.0, 0.0),
        gtsam.GenericProjectionFactorCal3_S2,
    )


def _bal_problem_from_values(
    gtsam,
    template: BALProblem,
    values,
    optimize_cameras: bool,
    optimize_points: bool,
) -> BALProblem:
    camera_params = template.camera_params.copy()
    if optimize_cameras:
        for camera_index in range(len(camera_params)):
            transform = _transform_from_pose3(values.atPose3(_camera_key(camera_index)))
            rvec, _ = cv2.Rodrigues(transform[:3, :3])
            camera_params[camera_index, :3] = rvec.reshape(3)
            camera_params[camera_index, 3:6] = transform[:3, 3]

    points_3d = template.points_3d.copy()
    if optimize_points:
        for point_index in range(len(points_3d)):
            points_3d[point_index] = _point3_to_array(values.atPoint3(_point_key(point_index)))

    return BALProblem(camera_params=camera_params, points_3d=points_3d, observations=template.observations)


def _edge_noise(gtsam, information: np.ndarray):
    try:
        return gtsam.noiseModel.Gaussian.Information(np.asarray(information, dtype=np.float64))
    except AttributeError:
        return gtsam.noiseModel.Diagonal.Sigmas(np.ones(6, dtype=np.float64))


def _fixed_pose_noise(gtsam):
    try:
        return gtsam.noiseModel.Constrained.All(6)
    except AttributeError:
        return gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-9, dtype=np.float64))


def _fixed_point_noise(gtsam):
    try:
        return gtsam.noiseModel.Constrained.All(3)
    except AttributeError:
        return gtsam.noiseModel.Diagonal.Sigmas(np.full(3, 1e-9, dtype=np.float64))


def _make_lm_optimizer(gtsam, factor_graph, initial, *, max_iterations: int | None):
    if max_iterations is None:
        return gtsam.LevenbergMarquardtOptimizer(factor_graph, initial)

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(int(max_iterations))
    return gtsam.LevenbergMarquardtOptimizer(factor_graph, initial, params)


def _factor_graph_error(factor_graph, values) -> float:
    if hasattr(factor_graph, "error"):
        return float(factor_graph.error(values))
    return float("nan")
