"""Evaluate sparse photometric residuals for matched image coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from slam.vo.direct import direct_pose_residuals, photometric_residuals, refine_pose_se3, refine_translation_2d


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Reference grayscale/color image.")
    parser.add_argument("--current", required=True, type=Path, help="Current grayscale/color image.")
    parser.add_argument(
        "--points",
        required=True,
        type=Path,
        help="Nx4 .npy array: reference_x reference_y current_x current_y.",
    )
    parser.add_argument(
        "--refine-translation",
        action="store_true",
        help="Refine a single 2D translation from the reference points instead of only evaluating supplied current points.",
    )
    parser.add_argument(
        "--refine-pose",
        action="store_true",
        help="Refine a sparse SE3 warp using --depths and --intrinsics.",
    )
    parser.add_argument("--depths", type=Path, help="N-element .npy depth array for reference points.")
    parser.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--initial-transform", type=Path, help="Optional 4x4 .npy initial T_cur_ref for --refine-pose.")
    return parser.parse_args()


def _read_gray_float(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image.astype(np.float64) / 255.0


def main() -> None:
    args = _parse_args()
    reference = _read_gray_float(args.reference)
    current = _read_gray_float(args.current)
    points = np.load(args.points).astype(np.float64)
    if points.ndim != 2 or points.shape[1] != 4:
        raise SystemExit("--points must be an Nx4 array")

    residuals, valid = photometric_residuals(reference, current, points[:, :2], points[:, 2:])
    valid_residuals = residuals[valid]

    print(f"reference: {args.reference}")
    print(f"current: {args.current}")
    print(f"points: {args.points}")
    print(f"point count: {len(points)}")
    print(f"valid residual count: {int(np.count_nonzero(valid))}")
    if len(valid_residuals):
        print(f"residual mean: {float(valid_residuals.mean()):.9f}")
        print(f"residual median: {float(np.median(valid_residuals)):.9f}")
        print(f"residual rmse: {float(np.sqrt(np.mean(valid_residuals * valid_residuals))):.9f}")

    if args.refine_translation:
        initial = np.nanmedian(points[:, 2:] - points[:, :2], axis=0)
        result = refine_translation_2d(reference, current, points[:, :2], initial_translation=initial)
        print(f"initial translation: {result.initial_translation}")
        print(f"refined translation: {result.translation}")
        print(f"refined residual rmse: {result.residual_rmse:.9f}")
        print(f"refinement evaluations: {result.nfev}")
        print(f"refinement success: {result.success}")

    if args.refine_pose:
        if args.depths is None or args.intrinsics is None:
            raise SystemExit("--refine-pose requires --depths and --intrinsics")
        depths = np.load(args.depths).astype(np.float64).reshape(-1)
        if len(depths) != len(points):
            raise SystemExit("--depths length must match --points row count")
        fx, fy, cx, cy = args.intrinsics
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        initial_transform = (
            np.load(args.initial_transform).astype(np.float64) if args.initial_transform is not None else np.eye(4)
        )
        initial_residuals, initial_valid = direct_pose_residuals(
            reference,
            current,
            points[:, :2],
            depths,
            camera_matrix,
            initial_transform,
        )
        initial_valid_residuals = initial_residuals[initial_valid]
        result = refine_pose_se3(
            reference,
            current,
            points[:, :2],
            depths,
            camera_matrix,
            initial_transform_cur_ref=initial_transform,
        )
        print(f"initial SE3 valid residual count: {int(np.count_nonzero(initial_valid))}")
        if len(initial_valid_residuals):
            initial_rmse = float(np.sqrt(np.mean(initial_valid_residuals * initial_valid_residuals)))
            print(f"initial SE3 residual rmse: {initial_rmse:.9f}")
        print("refined T_cur_ref:")
        print(result.transform_cur_ref)
        print(f"refined SE3 valid residual count: {result.valid_count}")
        print(f"refined SE3 residual rmse: {result.residual_rmse:.9f}")
        print(f"SE3 refinement evaluations: {result.nfev}")
        print(f"SE3 refinement success: {result.success}")


if __name__ == "__main__":
    main()
