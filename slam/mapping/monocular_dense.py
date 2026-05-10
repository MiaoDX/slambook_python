"""Known-pose monocular dense depth estimation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from slam.vo.direct import bilinear_interpolate, project_reference_points_se3


@dataclass(frozen=True)
class DenseDepthEstimate:
    """Depth and NCC score maps estimated from a known relative pose."""

    depth: np.ndarray
    score: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth, dtype=np.float64)
        score = np.asarray(self.score, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        if depth.shape != score.shape or depth.shape != valid.shape:
            raise ValueError("depth, score, and valid maps must have the same shape")
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "valid", valid)


def ncc_score(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_point: np.ndarray,
    current_point: np.ndarray,
    *,
    window_radius: int = 2,
) -> float:
    """Compute normalized cross-correlation between two image patches."""

    if window_radius < 0:
        raise ValueError("window_radius must be non-negative")
    offsets = _patch_offsets(window_radius)
    reference_points = np.asarray(reference_point, dtype=np.float64).reshape(1, 2) + offsets
    current_points = np.asarray(current_point, dtype=np.float64).reshape(1, 2) + offsets
    reference_values, reference_valid = bilinear_interpolate(reference_image, reference_points)
    current_values, current_valid = bilinear_interpolate(current_image, current_points)
    if not np.all(reference_valid & current_valid):
        return float("nan")
    reference_values = reference_values - np.mean(reference_values)
    current_values = current_values - np.mean(current_values)
    denominator = np.linalg.norm(reference_values) * np.linalg.norm(current_values)
    if denominator == 0.0:
        return float("nan")
    return float(reference_values @ current_values / denominator)


def estimate_depth_by_epipolar_search(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_points: np.ndarray,
    camera_matrix: np.ndarray,
    transform_cur_ref: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    depth_samples: int = 64,
    window_radius: int = 2,
    min_score: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate sparse depths by sampling along the epipolar search interval."""

    reference_points = _points2(reference_points, name="reference_points")
    if min_depth <= 0 or max_depth <= min_depth:
        raise ValueError("depth range must satisfy 0 < min_depth < max_depth")
    if depth_samples <= 1:
        raise ValueError("depth_samples must be greater than 1")
    depths = np.linspace(min_depth, max_depth, depth_samples)
    estimated_depths = np.full(len(reference_points), np.nan, dtype=np.float64)
    scores = np.full(len(reference_points), np.nan, dtype=np.float64)
    valid = np.zeros(len(reference_points), dtype=bool)

    for point_index, point in enumerate(reference_points):
        repeated_points = np.repeat(point.reshape(1, 2), depth_samples, axis=0)
        projected, depth_valid = project_reference_points_se3(
            repeated_points,
            depths,
            camera_matrix,
            transform_cur_ref,
        )
        best_score = float("-inf")
        best_depth = float("nan")
        for depth_value, current_point, is_depth_valid in zip(depths, projected, depth_valid):
            if not is_depth_valid:
                continue
            score = ncc_score(
                reference_image,
                current_image,
                point,
                current_point,
                window_radius=window_radius,
            )
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_depth = float(depth_value)
        if best_score >= min_score:
            estimated_depths[point_index] = best_depth
            scores[point_index] = best_score
            valid[point_index] = True
    return estimated_depths, scores, valid


def dense_depth_from_known_pose(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    camera_matrix: np.ndarray,
    transform_cur_ref: np.ndarray,
    *,
    min_depth: float = 0.5,
    max_depth: float = 5.0,
    depth_samples: int = 64,
    stride: int = 4,
    gradient_threshold: float = 20.0,
    window_radius: int = 2,
    min_score: float = 0.8,
) -> DenseDepthEstimate:
    """Estimate a sparse/semi-dense depth map from two known-pose images."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    reference_image = np.asarray(reference_image, dtype=np.float64)
    current_image = np.asarray(current_image, dtype=np.float64)
    if reference_image.ndim != 2 or current_image.ndim != 2:
        raise ValueError("images must be single-channel 2D arrays")
    if reference_image.shape != current_image.shape:
        raise ValueError("reference_image and current_image must have the same shape")

    height, width = reference_image.shape
    margin = window_radius + 1
    gradient_x = cv2.Sobel(reference_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(reference_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.hypot(gradient_x, gradient_y)
    ys, xs = np.mgrid[margin : height - margin : stride, margin : width - margin : stride]
    candidates = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    strong = gradient[candidates[:, 1].astype(int), candidates[:, 0].astype(int)] >= gradient_threshold
    candidates = candidates[strong]

    depth_map = np.full(reference_image.shape, np.nan, dtype=np.float64)
    score_map = np.full(reference_image.shape, np.nan, dtype=np.float64)
    valid_map = np.zeros(reference_image.shape, dtype=bool)
    if len(candidates) == 0:
        return DenseDepthEstimate(depth=depth_map, score=score_map, valid=valid_map)

    depths, scores, valid = estimate_depth_by_epipolar_search(
        reference_image,
        current_image,
        candidates,
        camera_matrix,
        transform_cur_ref,
        min_depth=min_depth,
        max_depth=max_depth,
        depth_samples=depth_samples,
        window_radius=window_radius,
        min_score=min_score,
    )
    pixel_indices = candidates.astype(int)
    for (x, y), depth, score, is_valid in zip(pixel_indices, depths, scores, valid):
        if is_valid:
            depth_map[y, x] = depth
            score_map[y, x] = score
            valid_map[y, x] = True
    return DenseDepthEstimate(depth=depth_map, score=score_map, valid=valid_map)


def _patch_offsets(window_radius: int) -> np.ndarray:
    values = np.arange(-window_radius, window_radius + 1, dtype=np.float64)
    dx, dy = np.meshgrid(values, values)
    return np.column_stack([dx.ravel(), dy.ravel()])


def _points2(points: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must be an Nx2 array")
    return points
