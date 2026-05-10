"""3D point registration helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slam.geometry.transforms import make_transform, transform_points


@dataclass(frozen=True)
class RigidRegistrationResult:
    """Rigid transform that aligns source points to target points."""

    rotation: np.ndarray
    translation: np.ndarray
    transform: np.ndarray
    rmse: float
    residuals: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64).reshape(3)
        transform = np.asarray(self.transform, dtype=np.float64)
        residuals = np.asarray(self.residuals, dtype=np.float64).reshape(-1, 3)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must have shape 3x3")
        if transform.shape != (4, 4):
            raise ValueError("transform must have shape 4x4")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "residuals", residuals)


def estimate_rigid_transform_3d(source_points: np.ndarray, target_points: np.ndarray) -> RigidRegistrationResult:
    """Estimate `T_target_source` from paired `Nx3` source and target points.

    The implementation is the SVD/Kabsch rigid alignment used by many ICP
    examples. It assumes one-to-one correspondences and does not estimate scale.
    """

    source_points = _points3(source_points, name="source_points")
    target_points = _points3(target_points, name="target_points")
    if len(source_points) != len(target_points):
        raise ValueError("source_points and target_points must have the same length")
    if len(source_points) < 3:
        raise ValueError("at least 3 point correspondences are required")

    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    transform = make_transform(rotation, translation)
    aligned = transform_points(transform, source_points)
    residuals = aligned - target_points
    rmse = _rmse(residuals)
    return RigidRegistrationResult(
        rotation=rotation,
        translation=translation,
        transform=transform,
        rmse=rmse,
        residuals=residuals,
    )


def _rmse(residuals: np.ndarray) -> float:
    squared = np.sum(residuals * residuals, axis=1)
    return float(np.sqrt(np.mean(squared))) if len(squared) else float("nan")


def _points3(points: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must be an Nx3 array")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    return points
