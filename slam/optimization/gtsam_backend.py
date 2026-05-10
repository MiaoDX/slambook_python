"""Optional GTSAM backend entry points."""

from __future__ import annotations

import numpy as np

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


def optimize_bundle_adjustment_gtsam(*args, **kwargs):
    """Placeholder for a future GTSAM bundle adjustment optimizer."""

    require_gtsam()
    raise NotImplementedError("GTSAM bundle adjustment adapter is not implemented yet.")


def _gtsam_key(vertex_id: int) -> int:
    return int(vertex_id)


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
