"""Stereo camera helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoRectification:
    """OpenCV stereo rectification outputs and remap tables."""

    left_map_x: np.ndarray
    left_map_y: np.ndarray
    right_map_x: np.ndarray
    right_map_y: np.ndarray
    q_matrix: np.ndarray
    rotation_left: np.ndarray
    rotation_right: np.ndarray
    projection_left: np.ndarray
    projection_right: np.ndarray


def disparity_to_depth(
    disparity: np.ndarray,
    *,
    focal_length: float,
    baseline: float,
    min_disparity: float = 0.0,
    invalid_value: float = np.nan,
) -> np.ndarray:
    """Convert disparity to metric depth with `depth = fx * baseline / disparity`.

    `baseline` is the positive distance between camera centers in the same units
    desired for depth. Disparities less than or equal to `min_disparity` are
    marked with `invalid_value`.
    """

    if focal_length <= 0:
        raise ValueError("focal_length must be positive")
    if baseline <= 0:
        raise ValueError("baseline must be positive")

    disparity = np.asarray(disparity, dtype=np.float64)
    depth = np.full(disparity.shape, invalid_value, dtype=np.float64)
    valid = np.isfinite(disparity) & (disparity > min_disparity)
    depth[valid] = focal_length * baseline / disparity[valid]
    return depth


def stereo_rectify(
    camera_matrix_left: np.ndarray,
    camera_matrix_right: np.ndarray,
    image_size: tuple[int, int],
    rotation_right_left: np.ndarray,
    translation_right_left: np.ndarray,
    *,
    dist_coeffs_left: np.ndarray | None = None,
    dist_coeffs_right: np.ndarray | None = None,
    alpha: float = 0.0,
) -> StereoRectification:
    """Compute rectification transforms and OpenCV remap tables.

    `rotation_right_left` and `translation_right_left` define the transform from
    the left camera frame into the right camera frame.
    """

    camera_matrix_left = _camera_matrix(camera_matrix_left, name="camera_matrix_left")
    camera_matrix_right = _camera_matrix(camera_matrix_right, name="camera_matrix_right")
    rotation_right_left = np.asarray(rotation_right_left, dtype=np.float64)
    translation_right_left = np.asarray(translation_right_left, dtype=np.float64).reshape(3, 1)
    if rotation_right_left.shape != (3, 3):
        raise ValueError("rotation_right_left must have shape 3x3")

    if dist_coeffs_left is None:
        dist_coeffs_left = np.zeros(5, dtype=np.float64)
    if dist_coeffs_right is None:
        dist_coeffs_right = np.zeros(5, dtype=np.float64)

    rotation_left, rotation_right, projection_left, projection_right, q_matrix, _, _ = cv2.stereoRectify(
        camera_matrix_left,
        np.asarray(dist_coeffs_left, dtype=np.float64),
        camera_matrix_right,
        np.asarray(dist_coeffs_right, dtype=np.float64),
        image_size,
        rotation_right_left,
        translation_right_left,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )

    left_map_x, left_map_y = cv2.initUndistortRectifyMap(
        camera_matrix_left,
        np.asarray(dist_coeffs_left, dtype=np.float64),
        rotation_left,
        projection_left,
        image_size,
        cv2.CV_32FC1,
    )
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(
        camera_matrix_right,
        np.asarray(dist_coeffs_right, dtype=np.float64),
        rotation_right,
        projection_right,
        image_size,
        cv2.CV_32FC1,
    )

    return StereoRectification(
        left_map_x=left_map_x,
        left_map_y=left_map_y,
        right_map_x=right_map_x,
        right_map_y=right_map_y,
        q_matrix=q_matrix,
        rotation_left=rotation_left,
        rotation_right=rotation_right,
        projection_left=projection_left,
        projection_right=projection_right,
    )


def _camera_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape 3x3")
    return matrix
