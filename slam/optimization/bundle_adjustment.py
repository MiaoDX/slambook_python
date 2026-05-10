"""Bundle adjustment data parsing and residual helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


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


@dataclass(frozen=True)
class BundleAdjustmentResult:
    """Result from SciPy bundle adjustment."""

    problem: BALProblem
    initial_rmse: float
    final_rmse: float
    cost: float
    nfev: int
    success: bool
    message: str


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


def pack_bal_parameters(
    problem: BALProblem,
    *,
    optimize_cameras: bool = True,
    optimize_points: bool = True,
) -> np.ndarray:
    """Pack selected BAL camera and point parameters into one vector."""

    parts = []
    if optimize_cameras:
        parts.append(problem.camera_params.reshape(-1))
    if optimize_points:
        parts.append(problem.points_3d.reshape(-1))
    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(parts).astype(np.float64)


def unpack_bal_parameters(
    template: BALProblem,
    params: np.ndarray,
    *,
    optimize_cameras: bool = True,
    optimize_points: bool = True,
) -> BALProblem:
    """Unpack a parameter vector using `template` for fixed values and observations."""

    params = np.asarray(params, dtype=np.float64).reshape(-1)
    cursor = 0
    camera_params = template.camera_params.copy()
    points_3d = template.points_3d.copy()

    if optimize_cameras:
        count = template.camera_params.size
        camera_params = params[cursor : cursor + count].reshape(template.camera_params.shape)
        cursor += count
    if optimize_points:
        count = template.points_3d.size
        points_3d = params[cursor : cursor + count].reshape(template.points_3d.shape)
        cursor += count
    if cursor != len(params):
        raise ValueError("unused values while unpacking BAL parameters")

    return BALProblem(camera_params=camera_params, points_3d=points_3d, observations=template.observations)


def residuals_from_parameters(
    params: np.ndarray,
    template: BALProblem,
    *,
    optimize_cameras: bool = True,
    optimize_points: bool = True,
) -> np.ndarray:
    """Residual vector for a packed BAL parameter vector."""

    return reprojection_residuals(
        unpack_bal_parameters(
            template,
            params,
            optimize_cameras=optimize_cameras,
            optimize_points=optimize_points,
        )
    )


def bal_jacobian_sparsity(
    problem: BALProblem,
    *,
    optimize_cameras: bool = True,
    optimize_points: bool = True,
):
    """Return the residual/parameter sparsity pattern for SciPy least-squares."""

    num_residuals = len(problem.observations) * 2
    num_params = 0
    camera_offset = None
    point_offset = None
    if optimize_cameras:
        camera_offset = num_params
        num_params += problem.camera_params.size
    if optimize_points:
        point_offset = num_params
        num_params += problem.points_3d.size

    sparsity = lil_matrix((num_residuals, num_params), dtype=int)
    for obs_index, observation in enumerate(problem.observations):
        rows = slice(2 * obs_index, 2 * obs_index + 2)
        if camera_offset is not None:
            start = camera_offset + observation.camera_index * 9
            sparsity[rows, start : start + 9] = 1
        if point_offset is not None:
            start = point_offset + observation.point_index * 3
            sparsity[rows, start : start + 3] = 1
    return sparsity.tocsr()


def solve_bundle_adjustment(
    problem: BALProblem,
    *,
    optimize_cameras: bool = True,
    optimize_points: bool = True,
    loss: str = "linear",
    f_scale: float = 1.0,
    max_nfev: int | None = None,
) -> BundleAdjustmentResult:
    """Run SciPy least-squares on a BAL problem."""

    initial_params = pack_bal_parameters(
        problem,
        optimize_cameras=optimize_cameras,
        optimize_points=optimize_points,
    )
    if initial_params.size == 0:
        return BundleAdjustmentResult(
            problem=problem,
            initial_rmse=reprojection_rmse(problem),
            final_rmse=reprojection_rmse(problem),
            cost=0.0,
            nfev=0,
            success=True,
            message="No parameters selected for optimization.",
        )

    initial_rmse = reprojection_rmse(problem)
    result = least_squares(
        residuals_from_parameters,
        initial_params,
        jac_sparsity=bal_jacobian_sparsity(
            problem,
            optimize_cameras=optimize_cameras,
            optimize_points=optimize_points,
        ),
        args=(problem,),
        kwargs={"optimize_cameras": optimize_cameras, "optimize_points": optimize_points},
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
    )
    optimized = unpack_bal_parameters(
        problem,
        result.x,
        optimize_cameras=optimize_cameras,
        optimize_points=optimize_points,
    )
    return BundleAdjustmentResult(
        problem=optimized,
        initial_rmse=initial_rmse,
        final_rmse=reprojection_rmse(optimized),
        cost=float(result.cost),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )
