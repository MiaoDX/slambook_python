"""Run SciPy least-squares on a BAL bundle adjustment problem."""

from __future__ import annotations

import argparse
from pathlib import Path

from slam.optimization.bundle_adjustment import read_bal_problem, reprojection_rmse, solve_bundle_adjustment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bal", required=True, type=Path, help="Path to a BAL text problem.")
    parser.add_argument("--max-nfev", type=int, help="Maximum SciPy residual evaluations.")
    parser.add_argument("--loss", default="linear", help="SciPy least_squares loss, e.g. linear or soft_l1.")
    parser.add_argument("--f-scale", type=float, default=1.0)
    parser.add_argument("--fix-cameras", action="store_true", help="Optimize points only.")
    parser.add_argument("--fix-points", action="store_true", help="Optimize cameras only.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.fix_cameras and args.fix_points:
        raise SystemExit("Cannot use --fix-cameras and --fix-points together.")

    problem = read_bal_problem(args.bal)
    result = solve_bundle_adjustment(
        problem,
        optimize_cameras=not args.fix_cameras,
        optimize_points=not args.fix_points,
        loss=args.loss,
        f_scale=args.f_scale,
        max_nfev=args.max_nfev,
    )

    print(f"BAL file: {args.bal}")
    print(f"camera count: {len(problem.camera_params)}")
    print(f"point count: {len(problem.points_3d)}")
    print(f"observation count: {len(problem.observations)}")
    print(f"initial reprojection RMSE: {reprojection_rmse(problem):.6f}")
    print(f"final reprojection RMSE: {result.final_rmse:.6f}")
    print(f"final cost: {result.cost:.6f}")
    print(f"function evaluations: {result.nfev}")
    print(f"success: {result.success}")
    print(f"message: {result.message}")


if __name__ == "__main__":
    main()
