"""Evaluate sparse photometric residuals for matched image coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from slam.vo.direct import photometric_residuals, refine_translation_2d


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


if __name__ == "__main__":
    main()
