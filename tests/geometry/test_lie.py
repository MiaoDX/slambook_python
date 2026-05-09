import numpy as np

from slam.geometry.lie import perturb_transform, se3_exp, se3_log, skew, so3_exp, so3_log
from slam.geometry.transforms import make_transform


def test_skew_matrix_cross_product_equivalence():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-4.0, 5.0, -6.0])

    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))


def test_so3_exp_log_round_trip():
    phi = np.array([0.2, -0.1, 0.05])

    rotation = so3_exp(phi)
    recovered = so3_log(rotation)

    np.testing.assert_allclose(recovered, phi, atol=1e-12)


def test_se3_exp_log_round_trip_for_small_motion():
    xi = np.array([0.5, -0.2, 0.1, 0.01, -0.02, 0.03])

    transform = se3_exp(xi)
    recovered = se3_log(transform)

    np.testing.assert_allclose(recovered, xi, atol=1e-12)


def test_se3_log_exp_round_trip_for_transform():
    rotation = so3_exp(np.array([0.03, -0.04, 0.02]))
    transform = make_transform(rotation, np.array([1.0, -2.0, 0.4]))

    recovered = se3_exp(se3_log(transform))

    np.testing.assert_allclose(recovered, transform, atol=1e-12)


def test_perturb_transform_left_and_right_are_explicit():
    transform = make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))
    perturbation = np.array([0.0, 2.0, 0.0, 0.0, 0.0, np.pi / 2.0])

    left = perturb_transform(transform, perturbation, side="left")
    right = perturb_transform(transform, perturbation, side="right")

    assert not np.allclose(left, right)
    np.testing.assert_allclose(left[:3, :3], right[:3, :3], atol=1e-12)
