"""Optional Rerun logging adapters."""

from __future__ import annotations

from slam.viz.open3d_viz import OptionalVisualizationDependencyError


def require_rerun():
    """Import Rerun or raise with install guidance."""

    try:
        import rerun as rr  # type: ignore
    except ImportError as exc:
        raise OptionalVisualizationDependencyError(
            "Rerun is optional. Install it with `pip install -e .[modern]` "
            "and verify wheel support for your Python/platform."
        ) from exc
    return rr


def log_points_rerun(entity_path: str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Log a point cloud to Rerun."""

    import numpy as np

    rr = require_rerun()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    rr.log(entity_path, rr.Points3D(points, colors=colors))


def log_trajectory_rerun(
    entity_path: str,
    poses: list[np.ndarray],
    *,
    color: list[int] | None = None,
    radius: float | None = None,
) -> None:
    """Log a trajectory to Rerun as one 3D line strip."""

    from slam.viz.matplotlib_viz import trajectory_xyz

    rr = require_rerun()
    xyz = trajectory_xyz(poses)
    kwargs = {}
    if color is not None:
        kwargs["colors"] = color
    if radius is not None:
        kwargs["radii"] = radius
    rr.log(entity_path, rr.LineStrips3D([xyz], **kwargs))
