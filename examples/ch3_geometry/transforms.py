"""Demonstrate homogeneous transform construction and composition."""

from __future__ import annotations

import argparse

import numpy as np

from slam.geometry.transforms import (
    compose_transforms,
    inverse_transform,
    make_transform,
    rotation_matrix_from_rotvec,
    transform_points,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotvec", type=float, nargs=3, default=(0.1, -0.2, 0.05), metavar=("RX", "RY", "RZ"))
    parser.add_argument("--translation", type=float, nargs=3, default=(1.0, 2.0, 3.0), metavar=("TX", "TY", "TZ"))
    parser.add_argument(
        "--points",
        type=float,
        nargs="+",
        default=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0),
        help="Flat list of XYZ values. Length must be a multiple of 3.",
    )
    return parser.parse_args()


def _format(matrix: np.ndarray) -> str:
    return np.array2string(np.asarray(matrix), precision=6, suppress_small=True)


def main() -> None:
    args = _parse_args()
    flat_points = np.asarray(args.points, dtype=np.float64)
    if flat_points.size % 3 != 0:
        raise SystemExit("--points length must be a multiple of 3")

    points_b = flat_points.reshape(-1, 3)
    rotation_ab = rotation_matrix_from_rotvec(np.asarray(args.rotvec, dtype=np.float64))
    transform_ab = make_transform(rotation_ab, np.asarray(args.translation, dtype=np.float64))
    transform_ba = inverse_transform(transform_ab)
    identity = compose_transforms(transform_ab, transform_ba)
    points_a = transform_points(transform_ab, points_b)

    print("T_ab:")
    print(_format(transform_ab))
    print("T_ba:")
    print(_format(transform_ba))
    print("T_ab * T_ba:")
    print(_format(identity))
    print("points_b:")
    print(_format(points_b))
    print("points_a:")
    print(_format(points_a))


if __name__ == "__main__":
    main()
