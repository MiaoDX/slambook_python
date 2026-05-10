"""Estimate relative pose from depth-backed 3D-3D feature correspondences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ch7_common import (
    add_camera_args,
    camera_matrix_from_args,
    format_matrix,
    object_points_from_depth_pixels,
    read_depth,
    read_grayscale,
)
from slam.camera.pinhole import CameraIntrinsics
from slam.features.opencv_features import SUPPORTED_FEATURES, match_images
from slam.geometry.registration import estimate_rigid_transform_3d


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0", required=True, type=Path, help="Path to the first image.")
    parser.add_argument("--image1", required=True, type=Path, help="Path to the second image.")
    parser.add_argument("--depth0", required=True, type=Path, help="Depth image aligned to image0.")
    parser.add_argument("--depth1", required=True, type=Path, help="Depth image aligned to image1.")
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
        help="Minimum valid 3D-3D correspondences required before registration.",
    )
    parser.add_argument("--output-transform", type=Path, help="Optional .npy path for the estimated T_10.")
    add_camera_args(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image0 = read_grayscale(args.image0)
    image1 = read_grayscale(args.image1)
    depth0 = read_depth(args.depth0)
    depth1 = read_depth(args.depth1)
    camera_matrix = camera_matrix_from_args(args, image0.shape[:2])
    intrinsics = CameraIntrinsics.from_matrix(camera_matrix)

    match_set = match_images(image0, image1, feature=args.matcher, max_features=args.max_features)
    points_3d0_all, depth0_mask = object_points_from_depth_pixels(
        match_set.points0,
        depth0,
        intrinsics,
        depth_scale=args.depth_scale,
    )
    matched_pixels1 = match_set.points1[depth0_mask]
    points_3d1, depth1_mask = object_points_from_depth_pixels(
        matched_pixels1,
        depth1,
        intrinsics,
        depth_scale=args.depth_scale,
    )
    points_3d0 = points_3d0_all[depth1_mask]

    if len(points_3d0) < args.min_correspondences:
        raise SystemExit(
            "Need at least "
            f"{args.min_correspondences} valid 3D-3D correspondences; found {len(points_3d0)}."
        )

    result = estimate_rigid_transform_3d(points_3d0, points_3d1)
    if args.output_transform is not None:
        args.output_transform.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_transform, result.transform)

    print(f"matcher: {args.matcher}")
    print(f"image0: {args.image0}")
    print(f"image1: {args.image1}")
    print(f"depth0: {args.depth0}")
    print(f"depth1: {args.depth1}")
    print(f"match count: {len(match_set.matches)}")
    print(f"valid depth0 correspondence count: {int(np.count_nonzero(depth0_mask))}")
    print(f"valid 3D-3D correspondence count: {int(np.count_nonzero(depth1_mask))}")
    print("estimated T_10 R:")
    print(format_matrix(result.rotation))
    print("estimated T_10 t:")
    print(format_matrix(result.translation))
    print("estimated T_10:")
    print(format_matrix(result.transform))
    print(f"registration RMSE m: {result.rmse:.9f}")
    if args.output_transform is not None:
        print(f"output transform: {args.output_transform}")


if __name__ == "__main__":
    main()
