"""Point cloud file helpers."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np

from slam.geometry.transforms import transform_points


@dataclass(frozen=True)
class OccupancyVoxelGrid:
    """Occupied voxel centers and observation counts."""

    voxel_size: float
    indices: np.ndarray
    centers: np.ndarray
    counts: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64)
        centers = np.asarray(self.centers, dtype=np.float64)
        counts = np.asarray(self.counts, dtype=np.int64)
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("indices must be an Nx3 array")
        if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) != len(indices):
            raise ValueError("centers must be an Nx3 array with one center per voxel")
        if counts.ndim != 1 or len(counts) != len(indices):
            raise ValueError("counts must have one value per voxel")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "counts", counts)


def transform_point_cloud(points: np.ndarray, transform_ab: np.ndarray) -> np.ndarray:
    """Transform an `Nx3` point cloud."""

    return transform_points(transform_ab, points)


def fuse_point_clouds(
    point_clouds: list[tuple[np.ndarray, np.ndarray | None]],
    *,
    transforms_wb: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fuse point clouds into one array, optionally transforming each to world."""

    if transforms_wb is not None and len(transforms_wb) != len(point_clouds):
        raise ValueError("transforms_wb must have one transform per point cloud")

    fused_points = []
    fused_colors = []
    has_any_colors = any(colors is not None for _, colors in point_clouds)
    for index, (points, colors) in enumerate(point_clouds):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be an Nx3 array")
        if transforms_wb is not None:
            points = transform_point_cloud(points, transforms_wb[index])
        fused_points.append(points)

        if has_any_colors:
            if colors is None:
                fused_colors.append(np.zeros((len(points), 3), dtype=np.uint8))
            else:
                colors = np.asarray(colors)
                if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) != len(points):
                    raise ValueError("colors must be an Nx3 array with one color per point")
                fused_colors.append(colors)

    points_out = np.vstack(fused_points) if fused_points else np.empty((0, 3), dtype=np.float64)
    colors_out = np.vstack(fused_colors) if has_any_colors else None
    return points_out, colors_out


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Downsample by replacing points in each voxel with their centroid."""

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) != len(points):
            raise ValueError("colors must be an Nx3 array with one color per point")

    if len(points) == 0:
        return points.copy(), None if colors is None else colors.copy()

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, voxel in enumerate(voxel_indices):
        buckets.setdefault(tuple(int(value) for value in voxel), []).append(index)

    down_points = []
    down_colors = []
    for key in sorted(buckets):
        indices = buckets[key]
        down_points.append(points[indices].mean(axis=0))
        if colors is not None:
            down_colors.append(colors[indices].mean(axis=0))

    points_out = np.asarray(down_points, dtype=np.float64)
    colors_out = None if colors is None else np.clip(np.asarray(down_colors), 0, 255).astype(np.uint8)
    return points_out, colors_out


def occupancy_voxel_grid(points: np.ndarray, *, voxel_size: float) -> OccupancyVoxelGrid:
    """Convert a point cloud into occupied voxel centers and counts."""

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if len(points) == 0:
        return OccupancyVoxelGrid(
            voxel_size=float(voxel_size),
            indices=np.empty((0, 3), dtype=np.int64),
            centers=np.empty((0, 3), dtype=np.float64),
            counts=np.empty(0, dtype=np.int64),
        )

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    unique, counts = np.unique(voxel_indices, axis=0, return_counts=True)
    order = np.lexsort((unique[:, 2], unique[:, 1], unique[:, 0]))
    unique = unique[order]
    counts = counts[order]
    centers = (unique.astype(np.float64) + 0.5) * voxel_size
    return OccupancyVoxelGrid(
        voxel_size=float(voxel_size),
        indices=unique,
        centers=centers,
        counts=counts,
    )


def write_occupancy_npz(path: str | Path, grid: OccupancyVoxelGrid) -> None:
    """Write occupied voxel grid arrays to a compressed `.npz` file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        voxel_size=np.asarray([grid.voxel_size], dtype=np.float64),
        indices=grid.indices,
        centers=grid.centers,
        counts=grid.counts,
    )


def estimate_normals(
    points: np.ndarray,
    *,
    k: int = 16,
    viewpoint: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate point normals with local PCA over `k` nearest neighbors."""

    if k < 3:
        raise ValueError("k must be at least 3")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if len(points) < 3:
        raise ValueError("normal estimation requires at least 3 points")
    if viewpoint is not None:
        viewpoint = np.asarray(viewpoint, dtype=np.float64).reshape(3)

    neighbor_count = min(k, len(points))
    normals = np.empty_like(points, dtype=np.float64)
    for index, point in enumerate(points):
        distances = np.sum((points - point) ** 2, axis=1)
        neighbor_indices = np.argsort(distances)[:neighbor_count]
        neighbors = points[neighbor_indices]
        centered = neighbors - neighbors.mean(axis=0)
        covariance = centered.T @ centered / len(neighbors)
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        if viewpoint is None:
            dominant_axis = int(np.argmax(np.abs(normal)))
            if normal[dominant_axis] < 0.0:
                normal = -normal
        elif float(normal @ (viewpoint - point)) < 0.0:
            normal = -normal
        normals[index] = normal / np.linalg.norm(normal)
    return normals


def write_ply_ascii(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> None:
    """Write an ASCII PLY point cloud.

    `points` must be `Nx3`. `colors`, when provided, must be `Nx3` RGB values
    in either `uint8` or numeric range compatible with clipping to `[0, 255]`.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")

    if colors is not None:
        colors = np.asarray(colors)
        if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) != len(points):
            raise ValueError("colors must be an Nx3 array with one color per point")
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
        if normals.ndim != 2 or normals.shape[1] != 3 or len(normals) != len(points):
            raise ValueError("normals must be an Nx3 array with one normal per point")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if normals is not None:
        lines.extend(
            [
                "property float nx",
                "property float ny",
                "property float nz",
            ]
        )
    if colors is not None:
        lines.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )
    lines.append("end_header")

    for index, point in enumerate(points):
        values = [f"{point[0]:.9f}", f"{point[1]:.9f}", f"{point[2]:.9f}"]
        if normals is not None:
            normal = normals[index]
            values.extend([f"{normal[0]:.9f}", f"{normal[1]:.9f}", f"{normal[2]:.9f}"])
        if colors is not None:
            color = colors[index]
            values.extend([str(int(color[0])), str(int(color[1])), str(int(color[2]))])
        lines.append(" ".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
