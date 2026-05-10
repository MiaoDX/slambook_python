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


def log_matches_rerun(
    entity_path: str,
    points0: np.ndarray,
    points1: np.ndarray,
    *,
    color: list[int] | None = None,
    radius: float | None = None,
) -> None:
    """Log matched 2D point pairs to Rerun."""

    import numpy as np

    rr = require_rerun()
    points0 = np.asarray(points0, dtype=np.float64)
    points1 = np.asarray(points1, dtype=np.float64)
    if points0.ndim != 2 or points0.shape[1] != 2:
        raise ValueError("points0 must be an Nx2 array")
    if points1.ndim != 2 or points1.shape[1] != 2:
        raise ValueError("points1 must be an Nx2 array")
    if len(points0) != len(points1):
        raise ValueError("points0 and points1 must have the same length")

    point_kwargs = {}
    line_kwargs = {}
    if color is not None:
        point_kwargs["colors"] = color
        line_kwargs["colors"] = color
    if radius is not None:
        point_kwargs["radii"] = radius
        line_kwargs["radii"] = radius

    rr.log(f"{entity_path}/points0", rr.Points2D(points0, **point_kwargs))
    rr.log(f"{entity_path}/points1", rr.Points2D(points1, **point_kwargs))
    rr.log(f"{entity_path}/lines", rr.LineStrips2D(np.stack([points0, points1], axis=1), **line_kwargs))


def log_tracks_rerun(
    entity_path: str,
    tracks: list[np.ndarray],
    *,
    color: list[int] | None = None,
    radius: float | None = None,
) -> None:
    """Log 2D feature tracks to Rerun as line strips."""

    import numpy as np

    rr = require_rerun()
    strips = []
    for track in tracks:
        track = np.asarray(track, dtype=np.float64)
        if track.ndim != 2 or track.shape[1] != 2:
            raise ValueError("each track must be an Nx2 array")
        strips.append(track)

    kwargs = {}
    if color is not None:
        kwargs["colors"] = color
    if radius is not None:
        kwargs["radii"] = radius
    rr.log(entity_path, rr.LineStrips2D(strips, **kwargs))


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
