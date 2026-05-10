"""Direct visual odometry image utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


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
