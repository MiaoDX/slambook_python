"""Geometry helpers for transforms, masks, and triangulation."""

from slam.geometry.lie import perturb_transform, se3_exp, se3_log, so3_exp, so3_log
from slam.geometry.masks import normalize_mask
from slam.geometry.registration import RigidRegistrationResult, estimate_rigid_transform_3d
from slam.geometry.triangulation import pixel_to_normalized, triangulate_points
from slam.geometry.transforms import (
    compose_transforms,
    inverse_transform,
    make_transform,
    rotation_matrix_from_rotvec,
    rotvec_from_rotation_matrix,
    transform_points,
)

__all__ = [
    "compose_transforms",
    "inverse_transform",
    "make_transform",
    "normalize_mask",
    "perturb_transform",
    "pixel_to_normalized",
    "RigidRegistrationResult",
    "rotation_matrix_from_rotvec",
    "rotvec_from_rotation_matrix",
    "se3_exp",
    "se3_log",
    "so3_exp",
    "so3_log",
    "transform_points",
    "triangulate_points",
    "estimate_rigid_transform_3d",
]
