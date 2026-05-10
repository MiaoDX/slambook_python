"""Optional Open3D visualization adapters."""

from __future__ import annotations

from pathlib import Path


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


def reconstruct_mesh_poisson(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    *,
    depth: int = 8,
):
    """Reconstruct an Open3D triangle mesh from a point cloud with Poisson reconstruction."""

    import numpy as np

    if depth <= 0:
        raise ValueError("depth must be positive")
    o3d = require_open3d()
    point_cloud = pointcloud_to_open3d(points, colors)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
        points_array = np.asarray(points, dtype=np.float64)
        if normals.ndim != 2 or normals.shape[1] != 3 or len(normals) != len(points_array):
            raise ValueError("normals must be an Nx3 array with one normal per point")
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
    elif hasattr(point_cloud, "estimate_normals"):
        point_cloud.estimate_normals()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)
    return mesh, np.asarray(densities, dtype=np.float64)


def write_triangle_mesh_open3d(path: str | Path, mesh) -> None:
    """Write an Open3D triangle mesh to disk."""

    o3d = require_open3d()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(path), mesh)
    if ok is False:
        raise RuntimeError(f"Open3D failed to write mesh: {path}")
