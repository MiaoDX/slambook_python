"""Demonstrate SO(3)/SE(3) exp-log operations."""

from __future__ import annotations

import argparse

import numpy as np

from slam.geometry.lie import perturb_transform, se3_exp, se3_log, so3_exp, so3_log


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xi",
        type=float,
        nargs=6,
        default=(0.5, -0.2, 0.1, 0.01, -0.02, 0.03),
        metavar=("RHO_X", "RHO_Y", "RHO_Z", "PHI_X", "PHI_Y", "PHI_Z"),
        help="se(3) tangent vector ordered as rho then phi.",
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        nargs=6,
        default=(0.01, 0.0, 0.0, 0.0, 0.0, 0.01),
        metavar=("DRHO_X", "DRHO_Y", "DRHO_Z", "DPHI_X", "DPHI_Y", "DPHI_Z"),
    )
    parser.add_argument("--side", choices=("left", "right"), default="left")
    return parser.parse_args()


def _format(matrix: np.ndarray) -> str:
    return np.array2string(np.asarray(matrix), precision=6, suppress_small=True)


def main() -> None:
    args = _parse_args()
    xi = np.asarray(args.xi, dtype=np.float64)
    perturbation = np.asarray(args.perturbation, dtype=np.float64)
    transform = se3_exp(xi)
    recovered_xi = se3_log(transform)
    rotation = so3_exp(xi[3:])
    recovered_phi = so3_log(rotation)
    perturbed = perturb_transform(transform, perturbation, side=args.side)

    print(f"xi: {_format(xi)}")
    print("SE3.exp(xi):")
    print(_format(transform))
    print(f"SE3.log(SE3.exp(xi)): {_format(recovered_xi)}")
    print("SO3.exp(phi):")
    print(_format(rotation))
    print(f"SO3.log(SO3.exp(phi)): {_format(recovered_phi)}")
    print(f"perturbed transform ({args.side}):")
    print(_format(perturbed))


if __name__ == "__main__":
    main()
