"""Fit the slambook Chapter 6 exponential curve with SciPy."""

from __future__ import annotations

import argparse

import numpy as np

from slam.optimization.curve_fitting import exponential_curve, fit_exponential_curve


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=float, default=1.0, help="Ground-truth a parameter.")
    parser.add_argument("--b", type=float, default=2.0, help="Ground-truth b parameter.")
    parser.add_argument("--c", type=float, default=1.0, help="Ground-truth c parameter.")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", default="linear", help="SciPy least_squares loss, e.g. linear or soft_l1.")
    parser.add_argument("--f-scale", type=float, default=1.0, help="Robust loss scale.")
    parser.add_argument(
        "--outlier-every",
        type=int,
        help="Inject a positive outlier every N samples to compare robust losses.",
    )
    parser.add_argument("--outlier-size", type=float, default=0.0)
    parser.add_argument("--initial", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("A0", "B0", "C0"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    true_params = np.array([args.a, args.b, args.c], dtype=np.float64)
    x = np.linspace(0.0, 1.0, args.samples)
    y = exponential_curve(x, true_params) + rng.normal(0.0, args.noise, size=x.shape)
    if args.outlier_every is not None and args.outlier_every > 0:
        y[:: args.outlier_every] += args.outlier_size

    result = fit_exponential_curve(
        x,
        y,
        initial_params=args.initial,
        loss=args.loss,
        f_scale=args.f_scale,
    )

    initial_error = float(np.linalg.norm(np.asarray(args.initial, dtype=np.float64) - true_params))
    final_error = float(np.linalg.norm(result.params - true_params))

    print(f"true params: {true_params}")
    print(f"initial params: {np.asarray(args.initial, dtype=np.float64)}")
    print(f"final params: {result.params}")
    print(f"initial parameter error: {initial_error:.6f}")
    print(f"final parameter error: {final_error:.6f}")
    print(f"final cost: {result.cost:.6f}")
    print(f"function evaluations: {result.nfev}")
    print(f"success: {result.success}")
    print(f"message: {result.message}")


if __name__ == "__main__":
    main()
