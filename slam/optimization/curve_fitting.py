"""SciPy curve fitting utilities for Chapter 6 optimization examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class CurveFitResult:
    """Result from fitting `exp(a*x^2 + b*x + c)`."""

    initial_params: np.ndarray
    params: np.ndarray
    cost: float
    optimality: float
    nfev: int
    success: bool
    message: str

    @property
    def final_params(self) -> np.ndarray:
        return self.params


def exponential_curve(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Evaluate `y = exp(a*x^2 + b*x + c)`."""

    x = np.asarray(x, dtype=np.float64)
    a, b, c = np.asarray(params, dtype=np.float64).reshape(3)
    return np.exp(a * x * x + b * x + c)


def exponential_residuals(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residual vector for `exp(a*x^2 + b*x + c) - y`."""

    return exponential_curve(x, params) - np.asarray(y, dtype=np.float64)


def exponential_jacobian(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytic residual Jacobian with columns `[d/da, d/db, d/dc]`."""

    del y
    x = np.asarray(x, dtype=np.float64)
    predicted = exponential_curve(x, params)
    return np.column_stack([x * x * predicted, x * predicted, predicted])


def fit_exponential_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    initial_params: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    loss: str = "linear",
    f_scale: float = 1.0,
    use_analytic_jacobian: bool = True,
    max_nfev: int | None = None,
) -> CurveFitResult:
    """Fit `exp(a*x^2 + b*x + c)` with `scipy.optimize.least_squares`."""

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    initial_params = np.asarray(initial_params, dtype=np.float64).reshape(3)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    jacobian = exponential_jacobian if use_analytic_jacobian else "2-point"
    result = least_squares(
        exponential_residuals,
        initial_params,
        jac=jacobian,
        args=(x, y),
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
    )
    return CurveFitResult(
        initial_params=initial_params,
        params=result.x,
        cost=float(result.cost),
        optimality=float(result.optimality),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )
