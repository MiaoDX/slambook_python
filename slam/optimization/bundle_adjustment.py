"""Bundle adjustment data parsing and residual helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class BALObservation:
    """One BAL observation record."""

    camera_index: int
    point_index: int
    xy: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "xy", np.asarray(self.xy, dtype=np.float64).reshape(2))


@dataclass(frozen=True)
class BALProblem:
    """BAL problem with 9-parameter cameras and 3D points."""

    camera_params: np.ndarray
    points_3d: np.ndarray
    observations: list[BALObservation]

    def __post_init__(self) -> None:
        camera_params = np.asarray(self.camera_params, dtype=np.float64)
        points_3d = np.asarray(self.points_3d, dtype=np.float64)
        if camera_params.ndim != 2 or camera_params.shape[1] != 9:
            raise ValueError("camera_params must be an Nx9 array")
        if points_3d.ndim != 2 or points_3d.shape[1] != 3:
            raise ValueError("points_3d must be an Nx3 array")
        object.__setattr__(self, "camera_params", camera_params)
        object.__setattr__(self, "points_3d", points_3d)


def read_bal_problem(path: str | Path) -> BALProblem:
    """Read a BAL text problem.

    Camera parameter order follows the BAL convention:
    rotation vector, translation, focal length, k1, k2.
    """

    values = Path(path).read_text(encoding="utf-8").split()
    if len(values) < 3:
        raise ValueError("BAL file is missing header")

    cursor = 0
    num_cameras = int(values[cursor])
    cursor += 1
    num_points = int(values[cursor])
    cursor += 1
    num_observations = int(values[cursor])
    cursor += 1

    observations: list[BALObservation] = []
    for _ in range(num_observations):
        camera_index = int(values[cursor])
        point_index = int(values[cursor + 1])
        xy = np.array([float(values[cursor + 2]), float(values[cursor + 3])], dtype=np.float64)
        cursor += 4
        observations.append(BALObservation(camera_index=camera_index, point_index=point_index, xy=xy))

    camera_values = [float(value) for value in values[cursor : cursor + num_cameras * 9]]
    cursor += num_cameras * 9
    point_values = [float(value) for value in values[cursor : cursor + num_points * 3]]
    cursor += num_points * 3
    if cursor != len(values):
        raise ValueError("BAL file has trailing values")

    return BALProblem(
        camera_params=np.asarray(camera_values, dtype=np.float64).reshape(num_cameras, 9),
        points_3d=np.asarray(point_values, dtype=np.float64).reshape(num_points, 3),
        observations=observations,
    )


def project_bal_point(camera_params: np.ndarray, point_3d: np.ndarray) -> np.ndarray:
    """Project one point using BAL camera parameters."""

    camera_params = np.asarray(camera_params, dtype=np.float64).reshape(9)
    point_3d = np.asarray(point_3d, dtype=np.float64).reshape(3)
    rotation = camera_params[:3]
    translation = camera_params[3:6]
    focal = camera_params[6]
    k1 = camera_params[7]
    k2 = camera_params[8]

    rotation_matrix, _ = cv2.Rodrigues(rotation)
    point_camera = rotation_matrix @ point_3d + translation
    xp = -point_camera[0] / point_camera[2]
    yp = -point_camera[1] / point_camera[2]
    radius2 = xp * xp + yp * yp
    distortion = 1.0 + radius2 * (k1 + k2 * radius2)
    return np.array([focal * distortion * xp, focal * distortion * yp], dtype=np.float64)


def reprojection_residuals(problem: BALProblem) -> np.ndarray:
    """Return flattened `2M` reprojection residuals for a BAL problem."""

    residuals = []
    for observation in problem.observations:
        projected = project_bal_point(
            problem.camera_params[observation.camera_index],
            problem.points_3d[observation.point_index],
        )
        residuals.extend(projected - observation.xy)
    return np.asarray(residuals, dtype=np.float64)


def reprojection_rmse(problem: BALProblem) -> float:
    """Return reprojection RMSE in pixels."""

    residuals = reprojection_residuals(problem).reshape(-1, 2)
    if len(residuals) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
