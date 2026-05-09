"""Rigid-body transform helpers following the project coordinate conventions."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a `4x4` homogeneous transform from `R_ab` and `t_ab`."""

    rotation = _rotation(rotation)
    translation = _translation(translation)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    """Return the inverse of a rigid `4x4` transform."""

    transform = _transform(transform)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Compose transforms in left-to-right application order.

    `compose_transforms(T_cb, T_ba)` returns `T_ca`, which maps points from
    frame `a` to frame `c`.
    """

    if not transforms:
        return np.eye(4, dtype=np.float64)

    result = _transform(transforms[0])
    for transform in transforms[1:]:
        result = result @ _transform(transform)
    return result


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to `Nx3` points."""

    transform = _transform(transform)
    points = _points3(points)
    return (points @ transform[:3, :3].T) + transform[:3, 3]


def rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues/axis-angle rotation vector to a matrix."""

    return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_matrix()


def rotvec_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a Rodrigues/axis-angle vector."""

    return Rotation.from_matrix(_rotation(rotation)).as_rotvec()


def _rotation(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape 3x3")
    return rotation


def _translation(translation: np.ndarray) -> np.ndarray:
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if translation.shape != (3,):
        raise ValueError("translation must have 3 values")
    return translation


def _transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape 4x4")
    return transform


def _points3(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    return points
