"""Triangulate matched feature points from a calibrated image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ch7_common import add_camera_args, camera_matrix_from_args, format_matrix, read_grayscale
from slam.features.opencv_features import SUPPORTED_FEATURES, match_images
from slam.geometry.triangulation import triangulate_points
from slam.vo.two_view import estimate_essential, estimate_fundamental, recover_relative_pose


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0", required=True, type=Path, help="Path to the first image.")
    parser.add_argument("--image1", required=True, type=Path, help="Path to the second image.")
    parser.add_argument("--matcher", default="orb", choices=sorted(SUPPORTED_FEATURES))
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--essential-threshold", type=float, default=1.0)
    parser.add_argument("--fundamental-threshold", type=float, default=1.0)
    parser.add_argument("--output-points", type=Path, help="Optional .npy path for triangulated Nx3 points.")
    add_camera_args(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image0 = read_grayscale(args.image0)
    image1 = read_grayscale(args.image1)
    camera_matrix = camera_matrix_from_args(args, image0.shape[:2])

    match_set = match_images(image0, image1, feature=args.matcher, max_features=args.max_features)
    points0 = match_set.points0
    points1 = match_set.points1
    if len(match_set.matches) < 8:
        raise SystemExit(f"Need at least 8 matches for triangulation; found {len(match_set.matches)}.")

    fundamental = estimate_fundamental(points0, points1, ransac_reproj_threshold=args.fundamental_threshold)
    essential = estimate_essential(points0, points1, camera_matrix, threshold=args.essential_threshold)
    pose = recover_relative_pose(essential.matrix, points0, points1, camera_matrix, mask=essential.mask)

    inlier_points0 = points0[pose.mask]
    inlier_points1 = points1[pose.mask]
    points_3d = triangulate_points(
        inlier_points0,
        inlier_points1,
        camera_matrix,
        pose.rotation_10,
        pose.translation_10,
    )
    finite = np.isfinite(points_3d).all(axis=1)
    positive_depth = finite & (points_3d[:, 2] > 0.0)

    if args.output_points is not None:
        args.output_points.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_points, points_3d[finite])

    print(f"matcher: {args.matcher}")
    print(f"image0: {args.image0}")
    print(f"image1: {args.image1}")
    print(f"match count: {len(match_set.matches)}")
    print(f"Fundamental Matrix inlier count: {fundamental.inlier_count}")
    print(f"Essential Matrix inlier count: {essential.inlier_count}")
    print("recovered R:")
    print(format_matrix(pose.rotation_10))
    print("recovered t:")
    print(format_matrix(pose.translation_10))
    print(f"recoverPose inlier count: {pose.inlier_count}")
    print(f"triangulated point count: {int(np.count_nonzero(finite))}")
    print(f"positive-depth point count: {int(np.count_nonzero(positive_depth))}")
    if np.any(positive_depth):
        depths = points_3d[positive_depth, 2]
        print(f"positive-depth min/median/max: {depths.min():.6f} {np.median(depths):.6f} {depths.max():.6f}")
    if args.output_points is not None:
        print(f"saved points: {args.output_points}")


if __name__ == "__main__":
    main()
