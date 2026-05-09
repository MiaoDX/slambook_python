"""Estimate camera pose from first-image depth and second-image feature matches."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ch7_common import (
    add_camera_args,
    camera_matrix_from_args,
    format_matrix,
    object_points_from_depth,
    read_depth,
    read_grayscale,
)
from slam.camera.pinhole import CameraIntrinsics
from slam.features.opencv_features import SUPPORTED_FEATURES, match_images
from slam.vo.pnp import project_points, solve_pnp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0", required=True, type=Path, help="Path to the first image.")
    parser.add_argument("--image1", required=True, type=Path, help="Path to the second image.")
    parser.add_argument("--depth0", required=True, type=Path, help="Depth image aligned to image0.")
    parser.add_argument("--matcher", default="orb", choices=sorted(SUPPORTED_FEATURES))
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=5000.0,
        help="Raw depth divisor. Use 5000 for common TUM/slambook uint16 depth, 1 for metric float depth.",
    )
    parser.add_argument(
        "--min-correspondences",
        type=int,
        default=6,
        help="Minimum valid 3D-2D correspondences required before solvePnP.",
    )
    add_camera_args(parser)
    return parser.parse_args()


def _reprojection_rmse(points_3d: np.ndarray, points_2d: np.ndarray, result, camera_matrix: np.ndarray) -> float:
    projected = project_points(points_3d, result.rotation, result.translation, camera_matrix)
    residuals = projected - points_2d
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def main() -> None:
    args = _parse_args()
    image0 = read_grayscale(args.image0)
    image1 = read_grayscale(args.image1)
    depth0 = read_depth(args.depth0)
    camera_matrix = camera_matrix_from_args(args, image0.shape[:2])
    intrinsics = CameraIntrinsics.from_matrix(camera_matrix)

    match_set = match_images(image0, image1, feature=args.matcher, max_features=args.max_features)
    points_3d, points_2d, depth_mask = object_points_from_depth(
        match_set.points0,
        match_set.points1,
        depth0,
        intrinsics,
        depth_scale=args.depth_scale,
    )

    if len(points_3d) < args.min_correspondences:
        raise SystemExit(
            "Need at least "
            f"{args.min_correspondences} valid depth correspondences for PnP; found {len(points_3d)}."
        )

    result = solve_pnp(points_3d, points_2d, camera_matrix, flags=cv2.SOLVEPNP_EPNP)
    rmse = _reprojection_rmse(points_3d, points_2d, result, camera_matrix)

    print(f"matcher: {args.matcher}")
    print(f"image0: {args.image0}")
    print(f"image1: {args.image1}")
    print(f"depth0: {args.depth0}")
    print(f"match count: {len(match_set.matches)}")
    print(f"valid depth correspondence count: {int(np.count_nonzero(depth_mask))}")
    print("recovered R:")
    print(format_matrix(result.rotation))
    print("recovered t:")
    print(format_matrix(result.translation))
    print(f"reprojection RMSE px: {rmse:.6f}")
    print("camera matrix:")
    print(format_matrix(camera_matrix))


if __name__ == "__main__":
    main()
