"""Estimate a sparse/semi-dense depth map from two known-pose monocular images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from slam.camera.pinhole import CameraIntrinsics
from slam.mapping.monocular_dense import dense_depth_from_known_pose


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--transform-cur-ref", required=True, type=float, nargs=16)
    parser.add_argument("--intrinsics", required=True, type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--output-depth", required=True, type=Path)
    parser.add_argument("--output-score", type=Path)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--depth-samples", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--gradient-threshold", type=float, default=20.0)
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.8)
    return parser.parse_args()


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def main() -> None:
    args = _parse_args()
    fx, fy, cx, cy = args.intrinsics
    camera_matrix = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy).matrix
    transform_cur_ref = np.asarray(args.transform_cur_ref, dtype=np.float64).reshape(4, 4)
    estimate = dense_depth_from_known_pose(
        _read_gray(args.reference),
        _read_gray(args.current),
        camera_matrix,
        transform_cur_ref,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        depth_samples=args.depth_samples,
        stride=args.stride,
        gradient_threshold=args.gradient_threshold,
        window_radius=args.window_radius,
        min_score=args.min_score,
    )
    args.output_depth.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_depth, estimate.depth)
    if args.output_score is not None:
        args.output_score.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_score, estimate.score)

    print(f"reference: {args.reference}")
    print(f"current: {args.current}")
    print(f"output depth: {args.output_depth}")
    if args.output_score is not None:
        print(f"output score: {args.output_score}")
    print(f"valid depth count: {int(np.count_nonzero(estimate.valid))}")
    print(f"depth samples: {args.depth_samples}")
    print(f"stride: {args.stride}")


if __name__ == "__main__":
    main()
