"""Optional PyCOLMAP pose-estimation adapters."""

from __future__ import annotations

import cv2
import numpy as np

from slam.optimization.pycolmap_backend import require_pycolmap
from slam.vo.pnp import PnPResult


def estimate_absolute_pose_pycolmap(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    image_size: tuple[int, int] | None = None,
    estimation_options: object | None = None,
    refinement_options: object | None = None,
    return_covariance: bool = False,
) -> PnPResult:
    """Estimate `T_cw` from 2D-3D correspondences with PyCOLMAP."""

    points_3d, points_2d, camera_matrix = _validate_pose_inputs(points_3d, points_2d, camera_matrix)
    pycolmap = require_pycolmap()
    camera = _camera_from_matrix(pycolmap, camera_matrix, image_size=image_size)
    estimate = getattr(pycolmap, "estimate_and_refine_absolute_pose", None)
    if estimate is None:
        estimate = getattr(pycolmap, "absolute_pose_estimation")
    answer = estimate(
        points_2d,
        points_3d,
        camera,
        **_drop_none(
            estimation_options=estimation_options,
            refinement_options=refinement_options,
            return_covariance=return_covariance,
        ),
    )
    if answer is None:
        raise RuntimeError("PyCOLMAP absolute pose estimation failed")

    transform_cw = _transform_from_rigid3d(answer["cam_from_world"])
    rvec, _ = cv2.Rodrigues(transform_cw[:3, :3])
    return PnPResult(
        rotation=transform_cw[:3, :3],
        translation=transform_cw[:3, 3].reshape(3, 1),
        rvec=rvec.reshape(3, 1),
        mask=_inlier_mask(answer, len(points_3d)),
    )


def _camera_from_matrix(pycolmap, camera_matrix: np.ndarray, *, image_size: tuple[int, int] | None):
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    if image_size is None:
        width = max(1, int(round(cx * 2.0)))
        height = max(1, int(round(cy * 2.0)))
    else:
        width, height = (int(image_size[0]), int(image_size[1]))

    camera_cls = pycolmap.Camera
    if hasattr(camera_cls, "create_from_model_name"):
        camera = camera_cls.create_from_model_name(1, "PINHOLE", 0.5 * (fx + fy), width, height)
        camera.params = np.array([fx, fy, cx, cy], dtype=np.float64)
        return camera
    return camera_cls(model="PINHOLE", width=width, height=height, params=[fx, fy, cx, cy])


def _transform_from_rigid3d(cam_from_world) -> np.ndarray:
    matrix = np.asarray(cam_from_world.matrix(), dtype=np.float64)
    if matrix.shape == (3, 4):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :] = matrix
        return transform
    if matrix.shape == (4, 4):
        return matrix
    raise ValueError("PyCOLMAP cam_from_world matrix must have shape 3x4 or 4x4")


def _inlier_mask(answer: dict[str, object], count: int) -> np.ndarray:
    if "inliers" in answer:
        mask = np.asarray(answer["inliers"], dtype=bool).reshape(-1)
    elif "inlier_mask" in answer:
        mask = np.asarray(answer["inlier_mask"], dtype=bool).reshape(-1)
    else:
        mask = np.ones(count, dtype=bool)
    if len(mask) != count:
        raise ValueError("PyCOLMAP inlier mask length does not match input correspondences")
    return mask


def _validate_pose_inputs(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_3d = np.asarray(points_3d, dtype=np.float64)
    points_2d = np.asarray(points_2d, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("points_3d must be an Nx3 array")
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError("points_2d must be an Nx2 array")
    if len(points_3d) != len(points_2d):
        raise ValueError("points_3d and points_2d must have the same length")
    if len(points_3d) < 4:
        raise ValueError("PyCOLMAP absolute pose estimation requires at least 4 correspondences")
    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape 3x3")
    return points_3d, points_2d, camera_matrix


def _drop_none(**kwargs: object) -> dict[str, object]:
    return {key: value for key, value in kwargs.items() if value is not None}
