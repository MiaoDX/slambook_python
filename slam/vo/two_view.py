"""Two-view geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from slam.geometry.masks import normalize_mask


@dataclass(frozen=True)
class FundamentalResult:
    matrix: np.ndarray
    mask: np.ndarray

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class EssentialResult:
    matrix: np.ndarray
    mask: np.ndarray

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class PoseResult:
    rotation_10: np.ndarray
    translation_10: np.ndarray
    mask: np.ndarray

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class TwoViewResult:
    fundamental: FundamentalResult
    essential: EssentialResult
    pose: PoseResult


def _points2(points: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must be an Nx2 array")
    return np.ascontiguousarray(array)


def _camera_matrix(camera_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape 3x3")
    return matrix


def estimate_fundamental(
    points0: np.ndarray,
    points1: np.ndarray,
    *,
    method: int = cv2.FM_RANSAC,
    ransac_reproj_threshold: float = 1.0,
    confidence: float = 0.999,
) -> FundamentalResult:
    """Estimate a fundamental matrix from matched `Nx2` image points."""

    points0 = _points2(points0, name="points0")
    points1 = _points2(points1, name="points1")
    if len(points0) != len(points1):
        raise ValueError("points0 and points1 must have the same length")
    if len(points0) < 8:
        raise ValueError("estimate_fundamental requires at least 8 correspondences")

    matrix, mask = cv2.findFundamentalMat(
        points0,
        points1,
        method=method,
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=confidence,
    )
    if matrix is None:
        raise RuntimeError("cv2.findFundamentalMat failed")
    if matrix.shape != (3, 3):
        raise RuntimeError(f"expected one 3x3 fundamental matrix, got shape {matrix.shape}")
    return FundamentalResult(matrix=matrix, mask=normalize_mask(mask, expected_length=len(points0)))


def estimate_essential(
    points0: np.ndarray,
    points1: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    method: int = cv2.RANSAC,
    probability: float = 0.999,
    threshold: float = 1.0,
) -> EssentialResult:
    """Estimate an essential matrix from matched `Nx2` image points."""

    points0 = _points2(points0, name="points0")
    points1 = _points2(points1, name="points1")
    camera_matrix = _camera_matrix(camera_matrix)
    if len(points0) != len(points1):
        raise ValueError("points0 and points1 must have the same length")
    if len(points0) < 5:
        raise ValueError("estimate_essential requires at least 5 correspondences")

    matrix, mask = cv2.findEssentialMat(
        points0,
        points1,
        cameraMatrix=camera_matrix,
        method=method,
        prob=probability,
        threshold=threshold,
    )
    if matrix is None:
        raise RuntimeError("cv2.findEssentialMat failed")
    if matrix.shape != (3, 3):
        matrix = matrix[:3, :]
    return EssentialResult(matrix=matrix, mask=normalize_mask(mask, expected_length=len(points0)))


def recover_relative_pose(
    essential_matrix: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> PoseResult:
    """Recover `T_10` from an essential matrix and matched points."""

    essential_matrix = np.asarray(essential_matrix, dtype=np.float64)
    points0 = _points2(points0, name="points0")
    points1 = _points2(points1, name="points1")
    camera_matrix = _camera_matrix(camera_matrix)
    if len(points0) != len(points1):
        raise ValueError("points0 and points1 must have the same length")

    mask_cv = None
    if mask is not None:
        mask_cv = normalize_mask(mask, expected_length=len(points0)).astype(np.uint8).reshape(-1, 1)

    if mask_cv is None:
        _, rotation, translation, pose_mask = cv2.recoverPose(essential_matrix, points0, points1, camera_matrix)
    else:
        _, rotation, translation, pose_mask = cv2.recoverPose(
            essential_matrix,
            points0,
            points1,
            camera_matrix,
            mask=mask_cv,
        )

    return PoseResult(
        rotation_10=rotation,
        translation_10=translation.reshape(3, 1),
        mask=normalize_mask(pose_mask, expected_length=len(points0)),
    )


def estimate_two_view_pose(
    points0: np.ndarray,
    points1: np.ndarray,
    camera_matrix: np.ndarray,
) -> TwoViewResult:
    """Estimate fundamental, essential, and relative pose for a two-view pair."""

    fundamental = estimate_fundamental(points0, points1)
    essential = estimate_essential(points0, points1, camera_matrix)
    pose = recover_relative_pose(essential.matrix, points0, points1, camera_matrix, mask=essential.mask)
    return TwoViewResult(fundamental=fundamental, essential=essential, pose=pose)
