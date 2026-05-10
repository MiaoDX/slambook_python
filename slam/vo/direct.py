"""Direct visual odometry image utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from slam.geometry.lie import se3_exp, se3_log


@dataclass(frozen=True)
class DirectAlignmentResult:
    """Result from sparse 2D direct image alignment."""

    translation: np.ndarray
    initial_translation: np.ndarray
    residual_rmse: float
    valid_count: int
    cost: float
    nfev: int
    success: bool
    message: str


@dataclass(frozen=True)
class DirectPoseAlignmentResult:
    """Result from sparse SE3 direct image alignment."""

    transform_cur_ref: np.ndarray
    initial_transform_cur_ref: np.ndarray
    residual_rmse: float
    valid_count: int
    cost: float
    nfev: int
    success: bool
    message: str

    def __post_init__(self) -> None:
        transform = np.asarray(self.transform_cur_ref, dtype=np.float64)
        initial = np.asarray(self.initial_transform_cur_ref, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("transform_cur_ref must have shape 4x4")
        if initial.shape != (4, 4):
            raise ValueError("initial_transform_cur_ref must have shape 4x4")
        object.__setattr__(self, "transform_cur_ref", transform)
        object.__setattr__(self, "initial_transform_cur_ref", initial)


def build_image_pyramid(image: np.ndarray, *, levels: int) -> list[np.ndarray]:
    """Build a Gaussian image pyramid with OpenCV `pyrDown`."""

    if levels <= 0:
        raise ValueError("levels must be positive")
    current = np.asarray(image)
    pyramid = [current]
    for _ in range(1, levels):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def bilinear_interpolate(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample an image at floating-point `Nx2` pixel coordinates.

    Returns `(values, valid_mask)`. Invalid samples are returned as `nan`.
    Coordinates are ordered as `[x, y]`.
    """

    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be a single-channel 2D array")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an Nx2 array")

    height, width = image.shape
    x = points[:, 0]
    y = points[:, 1]
    valid = (x >= 0.0) & (y >= 0.0) & (x <= width - 1.0) & (y <= height - 1.0)

    x0 = np.floor(np.clip(x, 0.0, width - 1.0)).astype(int)
    y0 = np.floor(np.clip(y, 0.0, height - 1.0)).astype(int)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    dx = np.clip(x - x0, 0.0, 1.0)
    dy = np.clip(y - y0, 0.0, 1.0)

    top = (1.0 - dx) * image[y0, x0] + dx * image[y0, x1]
    bottom = (1.0 - dx) * image[y1, x0] + dx * image[y1, x1]
    values = (1.0 - dy) * top + dy * bottom
    values = values.astype(np.float64)
    values[~valid] = np.nan
    return values, valid


def photometric_residuals(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_points: np.ndarray,
    current_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute `current(current_points) - reference(reference_points)` residuals."""

    reference_values, reference_valid = bilinear_interpolate(reference_image, reference_points)
    current_values, current_valid = bilinear_interpolate(current_image, current_points)
    valid = reference_valid & current_valid
    residuals = current_values - reference_values
    residuals[~valid] = np.nan
    return residuals, valid


def project_reference_points_se3(
    reference_points: np.ndarray,
    reference_depths: np.ndarray,
    camera_matrix: np.ndarray,
    transform_cur_ref: np.ndarray,
    *,
    min_depth: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Project reference pixels/depths into the current image under `T_cur_ref`."""

    reference_points = _points2(reference_points, name="reference_points")
    reference_depths = np.asarray(reference_depths, dtype=np.float64).reshape(-1)
    camera_matrix = _camera_matrix(camera_matrix)
    transform_cur_ref = _transform(transform_cur_ref, name="transform_cur_ref")
    if len(reference_points) != len(reference_depths):
        raise ValueError("reference_points and reference_depths must have the same length")

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    x_ref = (reference_points[:, 0] - cx) * reference_depths / fx
    y_ref = (reference_points[:, 1] - cy) * reference_depths / fy
    points_ref = np.column_stack([x_ref, y_ref, reference_depths])
    points_cur = (points_ref @ transform_cur_ref[:3, :3].T) + transform_cur_ref[:3, 3]
    depth_valid = points_cur[:, 2] > min_depth

    projected = np.empty((len(points_cur), 2), dtype=np.float64)
    safe_z = np.where(depth_valid, points_cur[:, 2], 1.0)
    projected[:, 0] = fx * points_cur[:, 0] / safe_z + cx
    projected[:, 1] = fy * points_cur[:, 1] / safe_z + cy
    return projected, depth_valid


def direct_pose_residuals(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_points: np.ndarray,
    reference_depths: np.ndarray,
    camera_matrix: np.ndarray,
    transform_cur_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sparse photometric residuals induced by an SE3 image warp."""

    current_points, depth_valid = project_reference_points_se3(
        reference_points,
        reference_depths,
        camera_matrix,
        transform_cur_ref,
    )
    residuals, image_valid = photometric_residuals(reference_image, current_image, reference_points, current_points)
    valid = depth_valid & image_valid
    residuals[~valid] = np.nan
    return residuals, valid


def refine_translation_2d(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_points: np.ndarray,
    *,
    initial_translation: np.ndarray | tuple[float, float] = (0.0, 0.0),
    loss: str = "linear",
    f_scale: float = 1.0,
    max_nfev: int | None = None,
) -> DirectAlignmentResult:
    """Refine a 2D translation by minimizing sparse photometric residuals.

    This is the smallest direct-method optimizer used by the examples. It is
    not a full SE3 direct VO solver, but it exercises the same residual shape,
    interpolation, robust loss, and convergence mechanics.
    """

    reference_points = np.asarray(reference_points, dtype=np.float64)
    if reference_points.ndim != 2 or reference_points.shape[1] != 2:
        raise ValueError("reference_points must be an Nx2 array")

    initial_translation = np.asarray(initial_translation, dtype=np.float64).reshape(2)

    def residual_fn(translation: np.ndarray) -> np.ndarray:
        current_points = reference_points + translation.reshape(1, 2)
        residuals, valid = photometric_residuals(reference_image, current_image, reference_points, current_points)
        if not np.any(valid):
            return np.full(len(reference_points), 1e6, dtype=np.float64)
        stable = residuals.copy()
        stable[~valid] = 0.0
        return stable

    result = least_squares(
        residual_fn,
        initial_translation,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
    )
    final_residuals = residual_fn(result.x)
    valid = np.isfinite(final_residuals)
    valid_residuals = final_residuals[valid]
    rmse = float(np.sqrt(np.mean(valid_residuals * valid_residuals))) if len(valid_residuals) else float("nan")
    return DirectAlignmentResult(
        translation=result.x,
        initial_translation=initial_translation,
        residual_rmse=rmse,
        valid_count=int(np.count_nonzero(valid)),
        cost=float(result.cost),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )


def refine_pose_se3(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_points: np.ndarray,
    reference_depths: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    initial_transform_cur_ref: np.ndarray | None = None,
    loss: str = "linear",
    f_scale: float = 1.0,
    max_nfev: int | None = None,
) -> DirectPoseAlignmentResult:
    """Refine `T_cur_ref` by minimizing sparse SE3 photometric residuals."""

    reference_points = _points2(reference_points, name="reference_points")
    initial_transform = (
        np.eye(4, dtype=np.float64)
        if initial_transform_cur_ref is None
        else _transform(initial_transform_cur_ref, name="initial_transform_cur_ref")
    )

    def residual_fn(xi: np.ndarray) -> np.ndarray:
        transform = se3_exp(xi)
        residuals, valid = direct_pose_residuals(
            reference_image,
            current_image,
            reference_points,
            reference_depths,
            camera_matrix,
            transform,
        )
        if not np.any(valid):
            return np.full(len(reference_points), 1e6, dtype=np.float64)
        stable = residuals.copy()
        stable[~valid] = 0.0
        return stable

    result = least_squares(
        residual_fn,
        se3_log(initial_transform),
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
    )
    transform = se3_exp(result.x)
    final_residuals, valid = direct_pose_residuals(
        reference_image,
        current_image,
        reference_points,
        reference_depths,
        camera_matrix,
        transform,
    )
    valid_residuals = final_residuals[valid]
    rmse = float(np.sqrt(np.mean(valid_residuals * valid_residuals))) if len(valid_residuals) else float("nan")
    return DirectPoseAlignmentResult(
        transform_cur_ref=transform,
        initial_transform_cur_ref=initial_transform,
        residual_rmse=rmse,
        valid_count=int(np.count_nonzero(valid)),
        cost=float(result.cost),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )


def _points2(points: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must be an Nx2 array")
    return points


def _camera_matrix(camera_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape 3x3")
    return matrix


def _transform(transform: np.ndarray, *, name: str) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape 4x4")
    return transform
