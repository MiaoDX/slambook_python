"""Small quantitative benchmark metrics for examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slam.io.trajectory import PoseStamped
from slam.optimization.bundle_adjustment import BALProblem, reprojection_rmse
from slam.optimization.pose_graph import PoseGraph, total_edge_error


@dataclass(frozen=True)
class MetricReport:
    """Named metric dictionary suitable for JSON output."""

    name: str
    metrics: dict[str, float | int | bool | str]

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "metrics": self.metrics}


def trajectory_translation_errors(
    estimated: list[PoseStamped],
    reference: list[PoseStamped],
    *,
    align_origin: bool = True,
) -> np.ndarray:
    """Return per-pose translation errors for timestamp-matched trajectories."""

    if len(estimated) != len(reference):
        raise ValueError("estimated and reference trajectories must have the same length")
    if not estimated:
        return np.empty(0, dtype=np.float64)
    estimated_xyz = np.asarray([pose.transform_wc[:3, 3] for pose in estimated], dtype=np.float64)
    reference_xyz = np.asarray([pose.transform_wc[:3, 3] for pose in reference], dtype=np.float64)
    if align_origin:
        estimated_xyz = estimated_xyz - estimated_xyz[0]
        reference_xyz = reference_xyz - reference_xyz[0]
    return np.linalg.norm(estimated_xyz - reference_xyz, axis=1)


def trajectory_report(
    estimated: list[PoseStamped],
    reference: list[PoseStamped],
    *,
    align_origin: bool = True,
) -> MetricReport:
    """Return trajectory translation RMSE/mean/max metrics."""

    errors = trajectory_translation_errors(estimated, reference, align_origin=align_origin)
    if len(errors) == 0:
        metrics = {"pose_count": 0, "rmse": 0.0, "mean": 0.0, "max": 0.0}
    else:
        metrics = {
            "pose_count": int(len(errors)),
            "rmse": float(np.sqrt(np.mean(errors * errors))),
            "mean": float(np.mean(errors)),
            "max": float(np.max(errors)),
        }
    metrics["align_origin"] = bool(align_origin)
    return MetricReport(name="trajectory_translation", metrics=metrics)


def bal_reprojection_report(problem: BALProblem, *, name: str = "bal_reprojection") -> MetricReport:
    """Return reprojection RMSE and problem size metrics for a BAL problem."""

    return MetricReport(
        name=name,
        metrics={
            "camera_count": int(problem.camera_params.shape[0]),
            "point_count": int(problem.points_3d.shape[0]),
            "observation_count": int(len(problem.observations)),
            "reprojection_rmse": float(reprojection_rmse(problem)),
        },
    )


def pose_graph_report(graph: PoseGraph, *, name: str = "pose_graph_edges") -> MetricReport:
    """Return unweighted pose-graph edge error metrics."""

    edge_count = len(graph.edges)
    total_error = float(total_edge_error(graph))
    return MetricReport(
        name=name,
        metrics={
            "vertex_count": int(len(graph.vertices)),
            "edge_count": int(edge_count),
            "total_edge_error": total_error,
            "edge_rmse": float(np.sqrt(total_error / edge_count)) if edge_count else 0.0,
        },
    )
