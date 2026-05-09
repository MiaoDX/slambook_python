"""RGB-D point cloud generation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slam.camera.pinhole import CameraIntrinsics
from slam.geometry.transforms import transform_points


@dataclass(frozen=True)
class RgbdPointCloud:
    """Point cloud arrays generated from one RGB-D frame."""

    points: np.ndarray
    colors: np.ndarray | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be an Nx3 array")
        object.__setattr__(self, "points", points)
        if self.colors is not None:
            colors = np.asarray(self.colors)
            if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) != len(points):
                raise ValueError("colors must be an Nx3 array with one color per point")
            object.__setattr__(self, "colors", colors)

    def transform(self, transform_ab: np.ndarray) -> "RgbdPointCloud":
        return RgbdPointCloud(points=transform_points(transform_ab, self.points), colors=self.colors)


def rgbd_to_point_cloud(
    color: np.ndarray | None,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    depth_scale: float = 1000.0,
    depth_trunc: float | None = None,
) -> RgbdPointCloud:
    """Back-project an RGB-D frame into camera-frame points.

    `depth_scale` converts raw depth values into metric depth. For example,
    `uint16` millimeter depth uses `depth_scale=1000`; TUM-style depth commonly
    uses `depth_scale=5000`.
    """

    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")

    if color is not None:
        color = np.asarray(color)
        if color.shape[:2] != depth.shape:
            raise ValueError("color and depth must have the same height and width")
        if color.ndim != 3 or color.shape[2] != 3:
            raise ValueError("color must be an HxWx3 array")

    metric_depth = depth.astype(np.float64) / depth_scale
    valid = np.isfinite(metric_depth) & (metric_depth > 0.0)
    if depth_trunc is not None:
        valid &= metric_depth <= depth_trunc

    ys, xs = np.nonzero(valid)
    pixels = np.column_stack([xs, ys]).astype(np.float64)
    points = intrinsics.pixel_to_camera(pixels, metric_depth[ys, xs])
    colors = color[ys, xs] if color is not None else None
    return RgbdPointCloud(points=points, colors=colors)
