"""Direct visual odometry image utilities."""

from __future__ import annotations

import cv2
import numpy as np


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
