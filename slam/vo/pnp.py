"""Perspective-n-Point helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from slam.geometry.lie import se3_exp
from slam.geometry.transforms import make_transform


@dataclass(frozen=True)
class PnPResult:
    """Camera pose estimated from 3D/2D correspondences."""

    rotation: np.ndarray
    translation: np.ndarray
    rvec: np.ndarray
    mask: np.ndarray | None = None

    @property
    def inlier_count(self) -> int:
        if self.mask is None:
            return 0
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class MotionOnlyBAResult:
    """Pose-only reprojection refinement result."""

    pnp: PnPResult
    initial_rmse: float
    final_rmse: float
    cost: float
    nfev: int
    success: bool
    message: str


def solve_pnp(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    dist_coeffs: np.ndarray | None = None,
    flags: int = cv2.SOLVEPNP_EPNP,
) -> PnPResult:
    """Estimate `T_cw` from `Nx3` object points and `Nx2` image points."""

    points_3d, points_2d, camera_matrix, dist_coeffs = _validate_pnp_inputs(
        points_3d,
        points_2d,
        camera_matrix,
        dist_coeffs=dist_coeffs,
    )

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
    mask = np.ones(len(points_3d), dtype=bool)
    return PnPResult(rotation=rotation, translation=translation.reshape(3, 1), rvec=rvec.reshape(3, 1), mask=mask)


def solve_pnp_ransac(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    dist_coeffs: np.ndarray | None = None,
    flags: int = cv2.SOLVEPNP_EPNP,
    iterations_count: int = 100,
    reprojection_error: float = 8.0,
    confidence: float = 0.99,
) -> PnPResult:
    """Estimate `T_cw` with OpenCV PnP RANSAC and a boolean inlier mask."""

    points_3d, points_2d, camera_matrix, dist_coeffs = _validate_pnp_inputs(
        points_3d,
        points_2d,
        camera_matrix,
        dist_coeffs=dist_coeffs,
    )

    ok, rvec, translation, inliers = cv2.solvePnPRansac(
        points_3d,
        points_2d,
        camera_matrix,
        dist_coeffs,
        iterationsCount=iterations_count,
        reprojectionError=reprojection_error,
        confidence=confidence,
        flags=flags,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnPRansac failed")

    mask = np.zeros(len(points_3d), dtype=bool)
    if inliers is not None:
        mask[np.asarray(inliers, dtype=np.int64).reshape(-1)] = True
    rotation, _ = cv2.Rodrigues(rvec)
    return PnPResult(rotation=rotation, translation=translation.reshape(3, 1), rvec=rvec.reshape(3, 1), mask=mask)


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


def refine_pnp_pose_scipy(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    *,
    dist_coeffs: np.ndarray | None = None,
    loss: str = "linear",
    f_scale: float = 1.0,
    max_nfev: int | None = None,
) -> MotionOnlyBAResult:
    """Refine a PnP pose by minimizing reprojection residuals with SciPy."""

    points_3d, points_2d, camera_matrix, dist_coeffs = _validate_pnp_inputs(
        points_3d,
        points_2d,
        camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    initial_rotation = np.asarray(initial_rotation, dtype=np.float64)
    if initial_rotation.shape != (3, 3):
        raise ValueError("initial_rotation must have shape 3x3")
    initial_translation = _translation3(initial_translation, name="initial_translation")
    initial_transform = make_transform(initial_rotation, initial_translation)

    def residual_fn(delta_xi: np.ndarray) -> np.ndarray:
        transform_cw = se3_exp(delta_xi) @ initial_transform
        return _reprojection_residuals(
            points_3d,
            points_2d,
            camera_matrix,
            transform_cw[:3, :3],
            transform_cw[:3, 3],
            dist_coeffs=dist_coeffs,
        )

    initial_residuals = residual_fn(np.zeros(6, dtype=np.float64))
    result = least_squares(
        residual_fn,
        np.zeros(6, dtype=np.float64),
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
    )
    transform_cw = se3_exp(result.x) @ initial_transform
    final_residuals = residual_fn(result.x)
    rotation = transform_cw[:3, :3]
    translation = transform_cw[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rotation)
    return MotionOnlyBAResult(
        pnp=PnPResult(
            rotation=rotation,
            translation=translation,
            rvec=rvec.reshape(3, 1),
            mask=np.ones(len(points_3d), dtype=bool),
        ),
        initial_rmse=_reprojection_rmse(initial_residuals),
        final_rmse=_reprojection_rmse(final_residuals),
        cost=float(result.cost),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )


def _validate_pnp_inputs(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    dist_coeffs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    else:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
    return points_3d, points_2d, camera_matrix, dist_coeffs


def _reprojection_residuals(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    projected = project_points(points_3d, rotation, translation, camera_matrix, dist_coeffs=dist_coeffs)
    return (projected - points_2d).reshape(-1)


def _reprojection_rmse(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1, 2)
    squared = np.sum(residuals * residuals, axis=1)
    return float(np.sqrt(np.mean(squared))) if len(squared) else float("nan")


def _translation3(translation: np.ndarray, *, name: str) -> np.ndarray:
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if translation.shape != (3,):
        raise ValueError(f"{name} must have 3 values")
    return translation
