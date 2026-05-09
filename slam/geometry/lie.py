"""Minimal Lie group helpers for SO(3) and SE(3)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from slam.geometry.transforms import inverse_transform, make_transform


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the `3x3` skew-symmetric matrix for a 3-vector."""

    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Map a 3-vector from the tangent space to an SO(3) rotation matrix."""

    return Rotation.from_rotvec(np.asarray(phi, dtype=np.float64).reshape(3)).as_matrix()


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Map an SO(3) rotation matrix to a 3-vector tangent coordinate."""

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape 3x3")
    return Rotation.from_matrix(rotation).as_rotvec()


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Map an se(3) tangent vector to an SE(3) transform.

    The tangent vector order is `[rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]`,
    where `rho` is translation-like and `phi` is the SO(3) rotation vector.
    """

    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    rho = xi[:3]
    phi = xi[3:]
    rotation = so3_exp(phi)
    translation = _left_jacobian_so3(phi) @ rho
    return make_transform(rotation, translation)


def se3_log(transform: np.ndarray) -> np.ndarray:
    """Map an SE(3) transform to `[rho, phi]` tangent coordinates."""

    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape 4x4")

    phi = so3_log(transform[:3, :3])
    rho = _left_jacobian_so3_inverse(phi) @ transform[:3, 3]
    return np.concatenate([rho, phi])


def perturb_transform(transform: np.ndarray, perturbation: np.ndarray, *, side: str = "left") -> np.ndarray:
    """Apply an se(3) perturbation to a transform.

    `side="left"` returns `exp(perturbation) @ transform`; `side="right"`
    returns `transform @ exp(perturbation)`.
    """

    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape 4x4")

    delta = se3_exp(perturbation)
    if side == "left":
        return delta @ transform
    if side == "right":
        return transform @ delta
    raise ValueError("side must be 'left' or 'right'")


def _left_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    omega = skew(phi)
    omega2 = omega @ omega
    if theta < 1e-8:
        return np.eye(3) + 0.5 * omega + (1.0 / 6.0) * omega2

    theta2 = theta * theta
    return np.eye(3) + ((1.0 - np.cos(theta)) / theta2) * omega + ((theta - np.sin(theta)) / (theta2 * theta)) * omega2


def _left_jacobian_so3_inverse(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    omega = skew(phi)
    omega2 = omega @ omega
    if theta < 1e-8:
        return np.eye(3) - 0.5 * omega + (1.0 / 12.0) * omega2

    theta2 = theta * theta
    coefficient = (1.0 / theta2) - ((1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
    return np.eye(3) - 0.5 * omega + coefficient * omega2


def inverse_se3(transform: np.ndarray) -> np.ndarray:
    """Alias for `inverse_transform` for Lie-oriented examples."""

    return inverse_transform(transform)
