"""Triangulation utilities."""

from __future__ import annotations

import cv2
import numpy as np


def _points2(points: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must be an Nx2 array")
    return array


def _camera_matrix(camera_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape 3x3")
    return matrix


def pixel_to_normalized(points: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """Convert `Nx2` pixel points into normalized camera coordinates."""

    points = _points2(points, name="points")
    camera_matrix = _camera_matrix(camera_matrix)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    normalized = np.empty_like(points, dtype=np.float64)
    normalized[:, 0] = (points[:, 0] - cx) / fx
    normalized[:, 1] = (points[:, 1] - cy) / fy
    return normalized


def triangulate_points(
    points0: np.ndarray,
    points1: np.ndarray,
    camera_matrix: np.ndarray,
    rotation_10: np.ndarray,
    translation_10: np.ndarray,
) -> np.ndarray:
    """Triangulate points into camera 0 coordinates.

    `rotation_10` and `translation_10` define `T_10`, which maps camera 0
    coordinates into camera 1 coordinates.
    """

    points0 = _points2(points0, name="points0")
    points1 = _points2(points1, name="points1")
    if len(points0) != len(points1):
        raise ValueError("points0 and points1 must have the same length")

    camera_matrix = _camera_matrix(camera_matrix)
    rotation_10 = np.asarray(rotation_10, dtype=np.float64)
    translation_10 = np.asarray(translation_10, dtype=np.float64).reshape(3, 1)
    if rotation_10.shape != (3, 3):
        raise ValueError("rotation_10 must have shape 3x3")

    normalized0 = pixel_to_normalized(points0, camera_matrix)
    normalized1 = pixel_to_normalized(points1, camera_matrix)

    projection0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    projection1 = np.hstack([rotation_10, translation_10])
    homogeneous = cv2.triangulatePoints(projection0, projection1, normalized0.T, normalized1.T)
    points4 = homogeneous.T
    return points4[:, :3] / points4[:, 3:4]
