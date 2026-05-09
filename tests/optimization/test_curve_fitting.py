import numpy as np

from slam.optimization.curve_fitting import exponential_curve, fit_exponential_curve


def test_curve_fitting_recovers_synthetic_parameters():
    rng = np.random.default_rng(42)
    true_params = np.array([0.7, 1.4, 0.3])
    x = np.linspace(0.0, 1.0, 100)
    y = exponential_curve(x, true_params) + rng.normal(0.0, 0.01, size=x.shape)

    result = fit_exponential_curve(x, y, initial_params=(0.0, 0.0, 0.0))

    assert result.success
    np.testing.assert_allclose(result.params, true_params, atol=0.08)


def test_robust_loss_is_less_sensitive_to_outliers_than_linear_loss():
    rng = np.random.default_rng(7)
    true_params = np.array([0.8, 1.2, 0.4])
    x = np.linspace(0.0, 1.0, 120)
    y = exponential_curve(x, true_params) + rng.normal(0.0, 0.01, size=x.shape)
    y_with_outliers = y.copy()
    y_with_outliers[::15] += 5.0

    linear = fit_exponential_curve(x, y_with_outliers, initial_params=(0.0, 0.0, 0.0), loss="linear")
    robust = fit_exponential_curve(
        x,
        y_with_outliers,
        initial_params=(0.0, 0.0, 0.0),
        loss="soft_l1",
        f_scale=0.1,
    )

    linear_error = np.linalg.norm(linear.params - true_params)
    robust_error = np.linalg.norm(robust.params - true_params)
    assert robust_error < linear_error
