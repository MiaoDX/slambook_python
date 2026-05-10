"""Check the optional PyCOLMAP reference path for a BAL problem."""

from __future__ import annotations

import argparse
from pathlib import Path

from slam.optimization.bundle_adjustment import read_bal_problem
from slam.optimization.gtsam_backend import OptionalBackendDependencyError
from slam.optimization.pycolmap_backend import require_pycolmap


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bal", required=True, type=Path, help="Path to a BAL text problem.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        require_pycolmap()
    except OptionalBackendDependencyError as exc:
        raise SystemExit(str(exc)) from exc

    problem = read_bal_problem(args.bal)
    print(f"BAL file: {args.bal}")
    print(f"camera count: {len(problem.camera_params)}")
    print(f"point count: {len(problem.points_3d)}")
    print(f"observation count: {len(problem.observations)}")
    raise SystemExit("PyCOLMAP is installed; BAL reference adapter implementation is pending.")


if __name__ == "__main__":
    main()
