"""Track sparse points between two images with OpenCV Lucas-Kanade flow."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0", required=True, type=Path)
    parser.add_argument("--image1", required=True, type=Path)
    parser.add_argument("--max-corners", type=int, default=500)
    parser.add_argument("--quality-level", type=float, default=0.01)
    parser.add_argument("--min-distance", type=float, default=8.0)
    parser.add_argument("--output", type=Path, help="Optional visualization output path.")
    return parser.parse_args()


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def main() -> None:
    args = _parse_args()
    image0 = _read_gray(args.image0)
    image1 = _read_gray(args.image1)
    corners = cv2.goodFeaturesToTrack(
        image0,
        maxCorners=args.max_corners,
        qualityLevel=args.quality_level,
        minDistance=args.min_distance,
    )
    if corners is None:
        raise SystemExit("No trackable corners found.")

    next_points, status, errors = cv2.calcOpticalFlowPyrLK(image0, image1, corners, None)
    status = status.reshape(-1) != 0
    previous_points = corners.reshape(-1, 2)
    tracked_points = next_points.reshape(-1, 2)
    tracked_errors = errors.reshape(-1)
    displacements = tracked_points[status] - previous_points[status]
    lengths = np.linalg.norm(displacements, axis=1)

    print(f"image0: {args.image0}")
    print(f"image1: {args.image1}")
    print(f"corner count: {len(previous_points)}")
    print(f"tracked count: {int(np.count_nonzero(status))}")
    if np.any(status):
        print(f"flow mean length: {float(lengths.mean()):.6f}")
        print(f"flow median length: {float(np.median(lengths)):.6f}")
        print(f"LK mean error: {float(tracked_errors[status].mean()):.6f}")

    if args.output is not None:
        vis = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        for p0, p1, ok in zip(previous_points, tracked_points, status):
            if not ok:
                continue
            start = tuple(np.rint(p0).astype(int))
            end = tuple(np.rint(p1).astype(int))
            cv2.line(vis, start, end, (0, 255, 0), 1)
            cv2.circle(vis, end, 2, (0, 0, 255), -1)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), vis):
            raise SystemExit(f"Could not write output: {args.output}")
        print(f"wrote visualization: {args.output}")


if __name__ == "__main__":
    main()
