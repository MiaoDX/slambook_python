import numpy as np

from slam.geometry.transforms import (
    compose_transforms,
    inverse_transform,
    make_transform,
    rotation_matrix_from_rotvec,
    rotvec_from_rotation_matrix,
    transform_points,
)


def test_inverse_transform_composes_to_identity():
    rotation = rotation_matrix_from_rotvec(np.array([0.1, -0.2, 0.05]))
    transform = make_transform(rotation, np.array([1.0, -2.0, 0.5]))

    identity_a = compose_transforms(transform, inverse_transform(transform))
    identity_b = compose_transforms(inverse_transform(transform), transform)

    np.testing.assert_allclose(identity_a, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(identity_b, np.eye(4), atol=1e-12)


def test_compose_transforms_maps_source_to_destination_frame():
    t_ba = make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))
    t_cb = make_transform(np.eye(3), np.array([0.0, 2.0, 0.0]))
    t_ca = compose_transforms(t_cb, t_ba)

    point_a = np.array([[3.0, 4.0, 5.0]])
    point_c = transform_points(t_ca, point_a)

    np.testing.assert_allclose(point_c, [[4.0, 6.0, 5.0]])


def test_transform_points_matches_column_vector_convention():
    rotation = rotation_matrix_from_rotvec(np.array([0.0, 0.0, np.pi / 2.0]))
    transform = make_transform(rotation, np.array([1.0, 2.0, 3.0]))

    points_b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 2.0]])
    points_a = transform_points(transform, points_b)

    np.testing.assert_allclose(points_a, [[1.0, 3.0, 3.0], [0.0, 2.0, 5.0]], atol=1e-12)


def test_rotation_vector_round_trip():
    rotvec = np.array([0.2, -0.3, 0.1])

    rotation = rotation_matrix_from_rotvec(rotvec)
    recovered = rotvec_from_rotation_matrix(rotation)

    np.testing.assert_allclose(recovered, rotvec, atol=1e-12)
