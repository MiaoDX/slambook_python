"""Estimate a two-view relative pose from matched OpenCV features."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from slam.features.opencv_features import SUPPORTED_FEATURES, match_images
from slam.geometry.triangulation import triangulate_points
from slam.vo.two_view import estimate_essential, estimate_fundamental, recover_relative_pose


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0", required=True, type=Path, help="Path to the first image.")
    parser.add_argument("--image1", required=True, type=Path, help="Path to the second image.")
    parser.add_argument(
        "--matcher",
        default="orb",
        choices=sorted(SUPPORTED_FEATURES),
        help="OpenCV feature detector and descriptor matcher to use.",
    )
    parser.add_argument("--max-features", type=int, default=1000, help="Maximum detector features.")
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        help="Camera intrinsics as fx fy cx cy.",
    )
    parser.add_argument(
        "--camera-matrix",
        type=float,
        nargs=9,
        metavar=("K00", "K01", "K02", "K10", "K11", "K12", "K20", "K21", "K22"),
        help="Camera matrix as 9 row-major values. Overrides --intrinsics.",
    )
    parser.add_argument("--essential-threshold", type=float, default=1.0, help="RANSAC threshold in pixels.")
    parser.add_argument(
        "--fundamental-threshold",
        type=float,
        default=1.0,
        help="RANSAC reprojection threshold in pixels.",
    )
    return parser.parse_args()


def _read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def _camera_matrix_from_args(args: argparse.Namespace, image_shape: tuple[int, int]) -> np.ndarray:
    if args.camera_matrix is not None:
        return np.asarray(args.camera_matrix, dtype=np.float64).reshape(3, 3)

    if args.intrinsics is not None:
        fx, fy, cx, cy = args.intrinsics
    else:
        height, width = image_shape
        fx = fy = float(max(width, height))
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _format_matrix(matrix: np.ndarray) -> str:
    return np.array2string(matrix, precision=6, suppress_small=True)


def main() -> None:
    args = _parse_args()
    image0 = _read_grayscale(args.image0)
    image1 = _read_grayscale(args.image1)
    camera_matrix = _camera_matrix_from_args(args, image0.shape[:2])

    match_set = match_images(image0, image1, feature=args.matcher, max_features=args.max_features)
    points0 = match_set.points0
    points1 = match_set.points1

    if len(match_set.matches) < 8:
        raise SystemExit(f"Need at least 8 matches for two-view geometry; found {len(match_set.matches)}.")

    fundamental = estimate_fundamental(
        points0,
        points1,
        ransac_reproj_threshold=args.fundamental_threshold,
    )
    essential = estimate_essential(
        points0,
        points1,
        camera_matrix,
        threshold=args.essential_threshold,
    )
    pose = recover_relative_pose(essential.matrix, points0, points1, camera_matrix, mask=essential.mask)

    pose_points0 = points0[pose.mask]
    pose_points1 = points1[pose.mask]
    triangulated = triangulate_points(
        pose_points0,
        pose_points1,
        camera_matrix,
        pose.rotation_10,
        pose.translation_10,
    )
    finite_points = np.isfinite(triangulated).all(axis=1)

    print(f"matcher: {args.matcher}")
    print(f"image0: {args.image0}")
    print(f"image1: {args.image1}")
    print(f"keypoints image0: {len(match_set.keypoints0)}")
    print(f"keypoints image1: {len(match_set.keypoints1)}")
    print(f"match count: {len(match_set.matches)}")
    print(f"Fundamental Matrix inlier count: {fundamental.inlier_count}")
    print(f"Essential Matrix inlier count: {essential.inlier_count}")
    print("recovered R:")
    print(_format_matrix(pose.rotation_10))
    print("recovered t:")
    print(_format_matrix(pose.translation_10))
    print(f"recoverPose inlier count: {pose.inlier_count}")
    print(f"triangulated point count: {int(np.count_nonzero(finite_points))}")
    print("camera matrix:")
    print(_format_matrix(camera_matrix))


if __name__ == "__main__":
    main()
