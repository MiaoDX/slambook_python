"""Perspective-n-Point helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PnPResult:
    """Camera pose estimated from 3D/2D correspondences."""

    rotation: np.ndarray
    translation: np.ndarray
    rvec: np.ndarray


def solve_pnp(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    dist_coeffs: np.ndarray | None = None,
    flags: int = cv2.SOLVEPNP_EPNP,
) -> PnPResult:
    """Estimate `T_cw` from `Nx3` object points and `Nx2` image points."""

    points_3d = np.asarray(points_3d, dtype=np.float64)
    points_2d = np.asarray(points_2d, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)

    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("points_3d must be an Nx3 array")
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError("points_2d must be an Nx2 array")
    if len(points_3d) != len(points_2d):
        raise ValueError("points_3d and points_2d must have the same length")
    if len(points_3d) < 4:
        raise ValueError("solve_pnp requires at least 4 correspondences")
    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape 3x3")

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, translation = cv2.solvePnP(
        points_3d,
        points_2d,
        camera_matrix,
        dist_coeffs,
        flags=flags,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP failed")

    rotation, _ = cv2.Rodrigues(rvec)
    return PnPResult(rotation=rotation, translation=translation.reshape(3, 1), rvec=rvec.reshape(3, 1))


def project_points(
    points_3d: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    """Project `Nx3` points into `Nx2` image coordinates."""

    points_3d = np.asarray(points_3d, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    rvec, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(points_3d, rvec, translation, camera_matrix, dist_coeffs)
    return projected.reshape(-1, 2)
