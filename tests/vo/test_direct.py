import cv2
import numpy as np

from slam.vo.direct import bilinear_interpolate, build_image_pyramid, photometric_residuals, refine_translation_2d


def test_bilinear_interpolate_known_values():
    image = np.array(
        [
            [0.0, 10.0],
            [20.0, 30.0],
        ]
    )
    points = np.array([[0.5, 0.5], [1.0, 0.0], [-1.0, 0.0]])

    values, valid = bilinear_interpolate(image, points)

    np.testing.assert_allclose(values[:2], [15.0, 10.0])
    np.testing.assert_array_equal(valid, [True, True, False])
    assert np.isnan(values[2])


def test_build_image_pyramid_shapes_decrease_by_level():
    image = np.zeros((16, 20), dtype=np.uint8)

    pyramid = build_image_pyramid(image, levels=3)

    assert [level.shape for level in pyramid] == [(16, 20), (8, 10), (4, 5)]


def test_photometric_residuals_preserve_sign_and_validity():
    reference = np.array([[1.0, 2.0], [3.0, 4.0]])
    current = reference + 10.0
    reference_points = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    current_points = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])

    residuals, valid = photometric_residuals(reference, current, reference_points, current_points)

    np.testing.assert_allclose(residuals[:2], [10.0, 10.0])
    np.testing.assert_array_equal(valid, [True, True, False])
    assert np.isnan(residuals[2])


def test_refine_translation_2d_recovers_synthetic_shift():
    ys, xs = np.mgrid[0:80, 0:90]
    reference = (
        np.exp(-((xs - 40.0) ** 2 + (ys - 35.0) ** 2) / 400.0)
        + 0.2 * np.sin(xs / 7.0)
        + 0.1 * np.cos(ys / 5.0)
    ).astype(np.float64)
    true_translation = np.array([2.4, -1.7])
    warp = np.array([[1.0, 0.0, true_translation[0]], [0.0, 1.0, true_translation[1]]])
    current = cv2.warpAffine(reference, warp, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR)
    grid_x, grid_y = np.meshgrid(np.arange(15, 70, 5), np.arange(15, 65, 5))
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    result = refine_translation_2d(reference, current, points, initial_translation=(0.0, 0.0), max_nfev=80)

    assert result.success
    np.testing.assert_allclose(result.translation, true_translation, atol=0.08)
    assert result.residual_rmse < 0.01
