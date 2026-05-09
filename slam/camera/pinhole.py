"""Pinhole camera projection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics with no distortion model."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_matrix(cls, camera_matrix: np.ndarray) -> "CameraIntrinsics":
        matrix = np.asarray(camera_matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("camera_matrix must have shape 3x3")
        return cls(fx=float(matrix[0, 0]), fy=float(matrix[1, 1]), cx=float(matrix[0, 2]), cy=float(matrix[1, 2]))

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def pixel_to_camera(self, pixels: np.ndarray, depths: np.ndarray | float = 1.0) -> np.ndarray:
        """Back-project `Nx2` pixel coordinates into camera coordinates."""

        pixels = _points2(pixels, name="pixels")
        depths = np.asarray(depths, dtype=np.float64)
        if depths.ndim == 0:
            depths = np.full(len(pixels), float(depths), dtype=np.float64)
        depths = depths.reshape(-1)
        if depths.shape[0] != len(pixels):
            raise ValueError("depths must be scalar or have one value per pixel")

        points = np.empty((len(pixels), 3), dtype=np.float64)
        points[:, 2] = depths
        points[:, 0] = (pixels[:, 0] - self.cx) / self.fx * depths
        points[:, 1] = (pixels[:, 1] - self.cy) / self.fy * depths
        return points

    def camera_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Project `Nx3` camera-frame points into pixel coordinates."""

        points = _points3(points, name="points")
        if np.any(points[:, 2] == 0):
            raise ValueError("cannot project points with zero depth")

        pixels = np.empty((len(points), 2), dtype=np.float64)
        pixels[:, 0] = self.fx * points[:, 0] / points[:, 2] + self.cx
        pixels[:, 1] = self.fy * points[:, 1] / points[:, 2] + self.cy
        return pixels

    def normalized_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Project normalized camera coordinates `Nx2` into pixels."""

        points = _points2(points, name="points")
        pixels = np.empty_like(points, dtype=np.float64)
        pixels[:, 0] = self.fx * points[:, 0] + self.cx
        pixels[:, 1] = self.fy * points[:, 1] + self.cy
        return pixels


def _points2(points: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must be an Nx2 array")
    return array


def _points3(points: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must be an Nx3 array")
    return array
