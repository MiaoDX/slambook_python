"""Point cloud file helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from slam.geometry.transforms import transform_points


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


def write_ply_ascii(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
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
        if colors is None:
            lines.append(f"{point[0]:.9f} {point[1]:.9f} {point[2]:.9f}")
        else:
            color = colors[index]
            lines.append(
                f"{point[0]:.9f} {point[1]:.9f} {point[2]:.9f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
