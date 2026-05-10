"""Pinhole camera projection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DistortionCoefficients:
    """Brown-Conrady distortion coefficients in OpenCV order."""

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def as_opencv(self) -> np.ndarray:
        """Return `[k1, k2, p1, p2, k3]` coefficients for OpenCV calls."""

        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    @property
    def is_zero(self) -> bool:
        return bool(np.allclose(self.as_opencv(), 0.0))

    def distort_normalized(self, points: np.ndarray) -> np.ndarray:
        """Apply radial/tangential distortion to normalized `Nx2` coordinates."""

        points = _points2(points, name="points")
        x = points[:, 0]
        y = points[:, 1]
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        distorted = np.empty_like(points, dtype=np.float64)
        distorted[:, 0] = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        distorted[:, 1] = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        return distorted


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics with optional distortion coefficients."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: DistortionCoefficients = field(default_factory=DistortionCoefficients)

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

    def camera_to_pixel(self, points: np.ndarray, *, apply_distortion: bool = False) -> np.ndarray:
        """Project `Nx3` camera-frame points into pixel coordinates."""

        points = _points3(points, name="points")
        if np.any(points[:, 2] == 0):
            raise ValueError("cannot project points with zero depth")

        normalized = points[:, :2] / points[:, 2:3]
        if apply_distortion:
            normalized = self.distortion.distort_normalized(normalized)
        return self.normalized_to_pixel(normalized)

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
