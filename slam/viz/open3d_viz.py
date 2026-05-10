"""Optional Open3D visualization adapters."""

from __future__ import annotations


class OptionalVisualizationDependencyError(ImportError):
    """Raised when an optional visualization dependency is unavailable."""


def require_open3d():
    """Import Open3D or raise with install guidance."""

    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:
        raise OptionalVisualizationDependencyError(
            "Open3D is optional. Install it with `pip install -e .[3d]` "
            "and verify wheel support for your Python/platform."
        ) from exc
    return o3d


def pointcloud_to_open3d(points: np.ndarray, colors: np.ndarray | None = None):
    """Convert `Nx3` point/color arrays into an Open3D point cloud."""

    import numpy as np

    o3d = require_open3d()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) != len(points):
            raise ValueError("colors must be an Nx3 array with one color per point")
        if colors.max(initial=0.0) > 1.0:
            colors = colors / 255.0
        point_cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return point_cloud
